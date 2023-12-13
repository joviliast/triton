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
#define USE_ROCM
#ifdef USE_ROCM

#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;
namespace {
Value loadA(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto wmmaLayout = encoding.getParent().cast<WmmaEncodingAttr>();
  auto nonKDim = 16;
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto aElemTy = aTensorTy.getElementType();
  auto aElemsPerInstr = encoding.getWMMAElemsPerInstr(); //
  auto wmmaInstrM = aElemsPerInstr[0];
  auto wmmaInstrK = aElemsPerInstr[1];

  auto numReps = encoding.getWMMARep(shape, aElemTy); //
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(wmmaLayout);
  assert(iWaveSize == 32);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM =
      getWaveM(rewriter, loc, wave, warpsPerCTA, wmmaInstrM, shape[0]);
  int numOfElems = wmmaInstrM * wmmaInstrK / iWaveSize;
  assert(numOfElems >= 1);
  unsigned int maxNumWarps = shape[0] / wmmaInstrM;
  int warpsPerGroupM = std::min(warpsPerCTA[0], maxNumWarps);
  aElemTy = typeConverter->convertType(aElemTy);

  SmallVector<Value> ha;
  if (fastPathAvailable(smemObj, sharedLayout, wmmaLayout)) { // local one
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offsets;
    if (isTransposed(order)) {
      SmallVector<int64_t> elemsPerInstr{wmmaInstrK, wmmaInstrM};
      SmallVector<int64_t> reps{numReps[1], numReps[0]};
      offsets = fastPathComputeOffsetsTy2(rewriter, loc, elemsPerInstr, waveM, // local one
                                          lane, warpsPerGroupM, numOfElems,
                                          reps, cSwizzleOffset);
    } else {
      offsets = fastPathComputeOffsetsTy1(rewriter, loc, aElemsPerInstr, waveM,  // local one
                                          lane, warpsPerGroupM, numOfElems,
                                          numReps, cSwizzleOffset);
    }
    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

    Type smemPtrTy = getShemPtrTy(aElemTy);
    Type resElemTy = typeConverter->convertType(aElemTy);

    int loadsPerThread = offsets.size() / (numRepM * numRepK); // 
    const int elemsPerLoad = numOfElems / loadsPerThread;
    assert(numOfElems % loadsPerThread == 0);

    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(aElemTy, elemsPerLoad);
          Value loadOffset =
              offsets[m * loadsPerThread * numRepK + k * loadsPerThread + loadId];
          Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset),
                                      getShemPtrTy(loadVecTy));
          Value vectorValue = load(loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(aElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(aElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (aElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (aElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        ha.push_back(valVec);
      }
    }
  } else { // normal path
    SmallVector<Value> offsets = computeOffsetsAType(
        rewriter, loc, aElemsPerInstr, waveM, lane, warpsPerGroupM, numOfElems,
        numReps, smemObj, sharedLayout, nonKDim);

    Value smemBase = computeBasePtr(rewriter, loc, smemObj);
    Type resElemTy = typeConverter->convertType(aElemTy);

    Type smemPtrTy = getShemPtrTy(aElemTy);

    int loadsPerThread = offsets.size() / (numReps[0] * numReps[1]);
    int elemsPerLoad = numOfElems / loadsPerThread;

    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(aElemTy, elemsPerLoad);
          Value loadOffset = offsets[m * loadsPerThread * numRepK +
                                     k * loadsPerThread + loadId];
          Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset),
                                      getShemPtrTy(loadVecTy));
          Value vectorValue = load(loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(aElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(aElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (aElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (aElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        ha.push_back(valVec);
      }
    }
  }

  MLIRContext *ctx = wmmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(ha.size(), ha[0].getType()));
  auto result = typeConverter->packLLElements(loc, ha, rewriter, structTy);
  return result;
}
} // anonymous namespace

namespace SharedToDotOperandWMMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  if (opIdx == 0) { // operand a
    return loadA(rewriter, loc, thread, encoding, typeConverter, tensor,
                smemObj);
  } else if (opIdx == 1) { // operand b
    return loadB(rewriter, loc, thread, encoding, typeConverter, tensor,
                smemObj);
  }
  assert(false && "unexpected operand idx");
  return Value();
}
} // namespace SharedToDotOperandWMMA

#endif