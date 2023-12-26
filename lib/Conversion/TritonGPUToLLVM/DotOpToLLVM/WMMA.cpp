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

#include "../DotOpToLLVM.h"
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::WmmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class MatrixCoreType : uint8_t {
  // D = AB + C;
  // typeof(D) == typeof(C)
  // typeof(A) == typeof(B)
  // typeof(D), typeof(A):
  FP32_FP16,
  FP32_BF16,
  FP16_FP16,
  BF16_BF16,
  INT32_IU8,
  INT32_IU4,
  NOT_APPLICABLE,
};

struct WMMAInstrDescr {
  MatrixCoreType coreType;
  unsigned size;
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpWMMAConversionHelper {
  WmmaEncodingAttr wmmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpWMMAConversionHelper(
      WmmaEncodingAttr wmmaLayout, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : wmmaLayout(wmmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(wmmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }


  Value generateWMMAOp(WMMAInstrDescr wmmaDescr, Value valA, Value valB,
                       Value valC) const {
    assert(16 == wmmaDescr.size);
    auto resType = valC.getType();
    Value falseFlag = int_val(1, false);
    switch (wmmaDescr.coreType) {
    case MatrixCoreType::FP32_FP16:
      return rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    case MatrixCoreType::FP32_BF16:
      return rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    case MatrixCoreType::FP16_FP16:
      return rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    case MatrixCoreType::BF16_BF16:
      return rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    case MatrixCoreType::INT32_IU8:
      return rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    case MatrixCoreType::INT32_IU4:
      return rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
          loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
    default:
      llvm::report_fatal_error("WMMA data type not supported");
    }
    return Value();
  }

  static MatrixCoreType getMatrixCoreTypeFromDot(DotOp op) {
    auto aOperandTy = op.getA().getType();
    auto aTensorTy = aOperandTy.cast<RankedTensorType>();
    auto aElemTy = aTensorTy.getElementType();
    auto bOperandTy = op.getB().getType();
    auto bTensorTy = bOperandTy.cast<RankedTensorType>();
    auto bElemTy = bTensorTy.getElementType();
    assert(aElemTy == bElemTy);
    auto cOperandTy = op.getC().getType();
    auto cTensorTy = cOperandTy.cast<RankedTensorType>();
    auto cElemTy = cTensorTy.getElementType();
    auto dOperandTy = op.getD().getType();
    auto dTensorTy = dOperandTy.cast<RankedTensorType>();
    auto dElemTy = dTensorTy.getElementType();
    assert(cElemTy == dElemTy);

    if (dElemTy.isF32() && aElemTy.isF16())
      return MatrixCoreType::FP32_FP16;
    if (dElemTy.isF32() && aElemTy.isBF16())
      return MatrixCoreType::FP32_BF16;
    if (dElemTy.isF16() && aElemTy.isF16())
      return MatrixCoreType::FP16_FP16;
    if (dElemTy.isBF16() && aElemTy.isBF16())
      return MatrixCoreType::BF16_BF16;
    if (dElemTy.isSignedInteger(32) && aElemTy.isUnsignedInteger(8))
      return MatrixCoreType::INT32_IU8;
    if (dElemTy.isSignedInteger(32) && aElemTy.isUnsignedInteger(4))
      return MatrixCoreType::INT32_IU4;

    return MatrixCoreType::NOT_APPLICABLE;
  }

  static WMMAInstrDescr getMatrixInstrDescr(DotOp op) {
    WMMAInstrDescr descr;
    auto tensorTy = op.getD().getType().cast<RankedTensorType>();
    auto encoding = tensorTy.getEncoding().cast<WmmaEncodingAttr>();
    descr.coreType = getMatrixCoreTypeFromDot(op);
    descr.size = 16;
    return descr;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int waveSize = 32;
    int subBlockSize = waveSize / numSubBlocks;
    Value laneId = urem(getThreadId(), i32_val(16));
    laneId = and_(laneId, i32_val(waveSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < waveSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              mlir::LLVM::shflSync(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = add(accScalar[i], other_acc);
          else
            accScalar[i] = fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = i32_val(0);
      else
        zero = f32_val(0.0);
      auto cond = icmp_ult(laneId, i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = select(cond, accScalar[i], zero);
    }

    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  int getNumSubmatrices(Type elementType, int nonKDim) const {
    return 1;
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
    auto mnkDim = WmmaEncodingAttr::getMNKDimPerWMMAInstr();
    auto wmmaInstrDescr = getMatrixInstrDescr(op);

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTy = aTensorTy.getElementType();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();

    auto repA = aEncoding.getMatrixCoreInstrRep(aTensorTy.getShape());
    auto repB = bEncoding.getMatrixCoreInstrRep(bTensorTy.getShape());

    assert(repA[1] == repB[0]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();
    auto numRepM = repA[0];
    auto numRepN = repB[1];
    auto numRepK = repA[1];

    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepM, numRepK, aTensorTy.getElementType());
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepN, numRepK, aTensorTy.getElementType());
    auto dstElemTy = dTensorTy.getElementType();
    auto fc =
        typeConverter->unpackLLElements(loc, loadedC, rewriter, dstElemTy);

    unsigned warpSize = triton::gpu::getWarpSize(wmmaLayout);
    unsigned laneSize = 2;
    // compute number of output elements that each thread holds for one WMMA
    // instruction.
    auto elemsPerVec = mnkDim[0] * mnkDim[1] * laneSize / warpSize;
    auto dElemsToStorePerThread = elemsPerVec / laneSize;
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mnkDim[2]);
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }
        acc = zeroAuxiliarBlocks(subBlocks, acc);
        for (size_t k = 0; k < numRepK; k++) {
          acc = generateWMMAOp(wmmaInstrDescr, ha[{m, k}], hb[{n, k}], acc);
        }
        acc = reduceSubBlocks(subBlocks, acc);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          fc[m * numRepN * dElemsToStorePerThread + n * dElemsToStorePerThread + v] =
              extract_element(dstElemTy, acc, i32_val(v * 2));
        }
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);
    return success();
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 Type type) const {
    auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
    ValueTable vals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        vals[{i, j}] = elems[n1 * i + j];
      }
    }
    return vals;
  }
};

} // namespace

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<WmmaEncodingAttr>() &&
         "Currently, we only support $c with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto wmmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<WmmaEncodingAttr>();

  DotOpWMMAConversionHelper helper(wmmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

#endif // ifdef USE_ROCM
