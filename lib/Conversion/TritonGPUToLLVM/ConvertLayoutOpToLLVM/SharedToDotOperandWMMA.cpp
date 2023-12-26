/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef USE_ROCM

#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

static void printValues(Location loc, ConversionPatternRewriter &rewriter, std::string prefix, const std::vector<Value> &vs) {
  auto ctx = loc.getContext();
  std::vector<Value> values;
  for (const auto &v: vs) {
    auto vTy = v.getType();
    if (auto vecTy = dyn_cast<VectorType>(vTy)) {
      auto elemTy = vecTy.getElementType();
      for (int i = 0; i < vecTy.getNumElements(); ++i) {
        values.push_back(extract_element(elemTy, v, i32_val(i)));
      }
    } else if (vTy.isa<LLVM::LLVMPointerType>()) {
      values.push_back(ptrtoint(i32_ty, v));
    } else {
      values.push_back(v);
    }
  }
  auto prefixAttr = mlir::StringAttr::get(ctx, prefix);
  rewriter.create<triton::PrintOp>(loc, prefixAttr, values);
}

Type getShemPtrTy(Type elemTy) {
  if (elemTy.isBF16()) {
    auto ctx = elemTy.getContext();
    return ptr_ty(type::i16Ty(ctx), 3);
  }

  return ptr_ty(elemTy, 3);
}

// Get waveId inside block of waves.
Value getWaveIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value waveId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx) {
  if (nonKIdx == 1)
    waveId = udiv(waveId, i32_val(wpt[0]));
  return urem(urem(waveId, i32_val(wpt[nonKIdx])),
              i32_val(tensorSizeNonK / elemPerInstrNonK));
}

} // namespace

namespace SharedToDotOperandWMMA {

/**
 * @brief swizzling tensor element indexes according pattern encoded in
 * SharedEncodingAttr
 *
 * @param rewriter
 * @param loc
 * @param row row of target tensor element related to the start of smemObj
 * @param col col of target tensor element related to the start of smemObj
 * @param smemObj shared memory object, contains info about tensor in LDS
 * @param attr layout attribute, contains swizzling info
 * @return swizzled row, col indexes in tensor notation
 */
std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj, SharedEncodingAttr attr) {
  (void)smemObj; // unused in current pattern
  bool transposed = (attr.getOrder()[0] != 1);
  if (transposed) {
    // tensor is column-wise, so swapping col and row in computations
    std::swap(row, col);
  }
  auto vec = i32_val(attr.getVec());
  auto perPhase = i32_val(attr.getPerPhase());
  auto maxPhase = i32_val(attr.getMaxPhase());

  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h): Basic algorithm for row-major tensor is following:
  //
  // phase = (row // perPhase) % maxPhase
  // colOffSwizzled = ((col // vec) ^ phase) * vec
  // colOffOrdered = col % vec
  // colOff = colOffSwizzled + colOffOrdered
  auto phase = urem(udiv(row, perPhase), maxPhase);
  auto colOffSwizzled = mul(xor_(udiv(col, vec), phase), vec);
  auto colOffOrdered = urem(col, vec);
  auto colOff = add(colOffSwizzled, colOffOrdered);

  if (transposed)
    return {colOff, row};
  else
    return {row, colOff};
}

/**
 * @brief This function maps particular load of wmma dot operand to element
 * indexes(row, col)
 *
 * Whole tensor is broken into "blocks" of waves along "non-K" axis.
 * One block could be processed by multiple waves.
 * One wave works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of wave block in tensor
 * 2. Offset of wave inside one wave block
 * 3. Offset of tile in one wave
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axies independently
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one WMMA instruction
 * @param waveId id component of 2d wave grid along nono-K axis
 * @param laneId lane id in warp [0..63]
 * @param warpsPerGroup number of warps in one block
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repretition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension of dot operand
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMapping(ConversionPatternRewriter &rewriter, Location loc,
                         const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                         Value laneId, int warpsPerGroup, int numOfElems,
                         ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
                         int loadVecSize, unsigned iNonKDim) {
  auto numM = reps[0];
  auto numK = reps[1];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numM * numK *
                                                      loadsPerThread);

  Value _0 = i32_val(0);
  Value nonKDim = i32_val(iNonKDim);

  for (int block = 0; block < numM; ++block) {
    Value blockVOffset = i32_val(block * elemsPerInstr[0] * warpsPerGroup);
    Value blockHOffset = _0;
    Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));
    Value waveHOffset = _0;
    for (int tile = 0; tile < numK; ++tile) {
      Value tileVOffset = _0;
      Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

      Value laneVOffset = laneId;
      Value laneHOffset = _0;

      for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
        Value elemVOffset = _0;
        Value elemHOffset = i32_val(loadId * loadVecSize);

        Value sliceVOffset = add(
            add(add(add(blockVOffset, waveVOffset), tileVOffset), laneVOffset),
            elemVOffset);
        Value sliceHOffset = add(
            add(add(add(blockHOffset, waveHOffset), tileHOffset), laneHOffset),
            elemHOffset);

        //printValues(loc, rewriter, "sliceVOffset: ", {sliceVOffset});
        //printValues(loc, rewriter, "sliceHOffset: ", {sliceHOffset});
        Value row = add(sliceVOffset, smemOffsets[0]);
        Value col = add(sliceHOffset, smemOffsets[1]);

        mapping[numK * loadsPerThread * block + loadsPerThread * tile +
                loadId] = {row, col};
      }
    }
  }
  return mapping;
}

bool isSwizzled(SharedEncodingAttr layout) { return layout.getMaxPhase() != 1; }

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  auto [swizzledRow, swizzledCol] =
      swizzleIndexes(rewriter, loc, row, col, smemObj, srcLayout);
  auto &strides = smemObj.strides;
  Value rowOffset = mul(swizzledRow, strides[0]);
  Value colOffset = mul(swizzledCol, strides[1]);
  return add(rowOffset, colOffset);
}

llvm::SmallVector<Value>
computeOffsetsAType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout, unsigned nonKDim) {
  SmallVector<Value> strides{smemObj.strides[0], smemObj.strides[1]};
  SmallVector<Value> offsets{smemObj.offsets[0], smemObj.offsets[1]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 1) {
    if (isSwizzled(srcLayout)) {
      assert(false);
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    }
    else
      vectorSize = numOfElems;
  }

  auto mapping = computeTensorElemMapping(rewriter, loc, elemsPerInstr, waveId,
                                          laneId, warpsPerGroup, numOfElems,
                                          reps, offsets, vectorSize, nonKDim);
  llvm::SmallVector<Value> aOffsets(mapping.size());
  for (int i = 0; i < mapping.size(); ++i) {
    Value row = mapping[i][0];
    Value col = mapping[i][1];
    aOffsets[i] = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
  }
  return aOffsets;
}

llvm::SmallVector<Value>
computeOffsetsBType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout, unsigned nonKDim) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps{reps[1], reps[0]};
  SmallVector<Value> toffsets{smemObj.offsets[1], smemObj.offsets[0]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 0) {
    if (isSwizzled(srcLayout)) {
      assert(false);
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    }
    else
      vectorSize = numOfElems;
  }

  auto mapping = computeTensorElemMapping(rewriter, loc, tElemsPerInstr, waveId,
                                          laneId, warpsPerGroup, numOfElems,
                                          tReps, toffsets, vectorSize, nonKDim);
  llvm::SmallVector<Value> bOffsets(mapping.size());
  for (int i = 0; i < mapping.size(); ++i) {
    // swap row and col, because operand B layout is a transposed operand A
    // layout
    Value row = mapping[i][1];
    Value col = mapping[i][0];
    bOffsets[i] = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
  }
  return bOffsets;
}

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj) {
  Value base = smemObj.base;
  Type type = base.getType();
  for (int i = 0; i < smemObj.strides.size(); ++i) {
    Value offset = sub(i32_val(0), mul(smemObj.offsets[i], smemObj.strides[i]));
    base = gep(type, base, offset);
  }
  return base;
}

/**
 * @brief try find if value is an integer constant
 *
 * Trace def-use chain and return integer in case we can proof it is constant.
 * Current implementation can trace chains of insertValue->extractValue
 * operations.
 *
 * @param val Value for that we want to get constant
 * @return std::optional on found integer value or empty std::optional
 */
std::optional<int> findConstValue(Value val) {
  while (val && !val.getDefiningOp<LLVM::ConstantOp>()) {
    LLVM::ExtractValueOp extractValOp =
        val.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extractValOp)
      return std::optional<int>();
    auto extractPosArr = extractValOp.getPosition();
    if (extractPosArr.size() > 1)
      return std::optional<int>();
    int extractPos = extractPosArr[0];

    int insertPos = -1;
    LLVM::InsertValueOp insertValOp;
    Value container = extractValOp.getOperand();
    do {
      insertValOp = container.getDefiningOp<LLVM::InsertValueOp>();
      if (!insertValOp)
        return std::optional<int>();
      auto insertPosArr = insertValOp.getPosition();
      if (insertPosArr.size() > 1)
        return std::optional<int>();
      insertPos = insertPosArr[0];
      container = insertValOp.getContainer();
    } while (insertPos != extractPos);
    val = insertValOp.getValue();
  }
  if (!val)
    return std::optional<int>();
  auto cOp = val.getDefiningOp<LLVM::ConstantOp>();
  assert(cOp);
  auto valAttr = cOp.getValueAttr();
  auto intAttr = dyn_cast<mlir::IntegerAttr>(valAttr);
  assert(intAttr);
  return intAttr.getInt();
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  int kDimIdx = opIdx == 0 ? 1 : 0;
  int nonKDimIdx = opIdx == 0 ? 0 : 1;

  auto wmmaLayout = encoding.getParent().cast<WmmaEncodingAttr>();
  auto nonKDim = wmmaLayout.getKMNDimPerWMMAInstr()[nonKDimIdx];
  assert(nonKDim == 16);
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto elemTy = aTensorTy.getElementType();
  auto elemsPerInstr = encoding.getElemsPerMatrixCoreInstr();
  auto wmmaInstrNonK = elemsPerInstr[nonKDimIdx];
  auto wmmaInstrK = elemsPerInstr[kDimIdx];

  auto numReps = encoding.getMatrixCoreInstrRep(shape);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];

  unsigned iWaveSize = triton::gpu::getWarpSize(wmmaLayout);
  unsigned iNumLanes = iWaveSize / 2;
  assert(iWaveSize == 32);
  Value waveSize = i32_val(iWaveSize);
  Value numLanes = i32_val(iNumLanes);
  Value linearWaveId = udiv(thread, waveSize);
  Value lane = urem(thread, numLanes); // share elem between two threads

  unsigned numElemsPerThreadPerRep =
      wmmaLayout.getKMNDimPerWMMAInstr()[kDimIdx];

  Value spatialWaveId =
      getWaveIdInBlock(rewriter, loc, linearWaveId, warpsPerCTA, wmmaInstrNonK,
                       shape[nonKDimIdx], nonKDimIdx);

  unsigned int maxNumWarps = shape[nonKDimIdx] / wmmaInstrNonK;
  int warpsPerGroupNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  if (opIdx == 0) {
    offsets = computeOffsetsAType(
        rewriter, loc, elemsPerInstr, spatialWaveId, lane, warpsPerGroupNonK,
        numElemsPerThreadPerRep, numReps, smemObj, sharedLayout, nonKDim);
  } else {
    assert(opIdx == 1);
    offsets = computeOffsetsBType(
        rewriter, loc, elemsPerInstr, spatialWaveId, lane, warpsPerGroupNonK,
        numElemsPerThreadPerRep, numReps, smemObj, sharedLayout, nonKDim);
  }
  smemBase = computeBasePtr(rewriter, loc, smemObj);

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = getShemPtrTy(elemTy);

  int loadsPerThread = offsets.size() / (numRepNonK * numRepK);
  int elemsPerLoad = numElemsPerThreadPerRep / loadsPerThread;
  assert(numElemsPerThreadPerRep % loadsPerThread == 0);

  for (int nonK = 0; nonK < numRepNonK; ++nonK) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(resElemTy, numElemsPerThreadPerRep);
      Value valVec = undef(vecTy);
      for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
        auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
        Value loadOffset = offsets[nonK * loadsPerThread * numRepK +
                                   k * loadsPerThread + loadId];
        Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset),
                                    getShemPtrTy(loadVecTy));
        Value loadedValue = load(loadAddress);
        if (loadsPerThread > 1) {
          for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
            Value elemVal =
                extract_element(elemTy, loadedValue, i32_val(elemId));
            elemVal = bitcast(elemVal, resElemTy);
            valVec = insert_element(vecTy, valVec, elemVal,
                                    i32_val(loadId * elemsPerLoad + elemId));
          }
        } else {
          valVec = loadedValue;
        }
      }
      loadedValues.push_back(valVec);
    }
  }

  MLIRContext *ctx = wmmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      typeConverter->packLLElements(loc, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandWMMA

#endif // ifdef USE_ROCM
