/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

#include "../PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

enum class WMMAInstrType : uint8_t {
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

static void printValues(Location loc, RewriterBase &rewriter,
                        std::string prefix, const SmallVector<Value> &vs) {
  auto ctx = loc.getContext();
  std::vector<Value> values;
  for (const auto &v : vs) {
    auto vTy = v.getType();
    if (auto vecTy = dyn_cast<VectorType>(vTy)) {
      auto elemTy = vecTy.getElementType();
      for (int i = 0; i < vecTy.getNumElements(); ++i) {
        values.push_back(extract_element(elemTy, v, i32_val(i)));
      }
    } else if (mlir::isa<LLVM::LLVMPointerType>(vTy)) {
      values.push_back(ptrtoint(i32_ty, v));
    } else {
      values.push_back(v);
    }
  }
  auto prefixAttr = mlir::StringAttr::get(ctx, prefix);
  rewriter.create<triton::PrintOp>(loc, prefixAttr, false, values);
}
using ValueTable = std::map<std::pair<unsigned, unsigned>, SmallVector<Value>>;

ValueTable
getValuesFromDotOperandLayoutStruct(ConversionPatternRewriter &rewriter,
                                    const LLVMTypeConverter *typeConverter,
                                    Value value, int n0, int numInstr, int n1,
                                    int kWidth, Type type, Location loc) {
  auto elems = unpackLLElements(loc, value, rewriter);
  if (numInstr == 2) {
    // printValues(loc, rewriter, "a00", elems);
  }
  ValueTable vals;
  for (int i = 0; i < n0; i++) {
    for (int j = 0; j < n1; j++) {
      for (int instrIdx = 0; instrIdx < numInstr; ++instrIdx) {
        Type elemTy = typeConverter->convertType(type);
        Type ty = vec_ty(elemTy, kWidth);
        Value rawElems = undef(ty);
        for (int k = 0; k < kWidth; ++k) {
          rawElems = insert_element(
              ty, rawElems,
              elems[i * n1 * numInstr * kWidth + j * numInstr * kWidth +
                    instrIdx * kWidth + k],
              i32_val(k));
        }

        Value convertedElems;
        if (type.isBF16() || type.isF16()) {
          convertedElems = rawElems;
        } else {
          convertedElems = bitcast(
              rawElems, vec_ty(i32_ty, kWidth * type.getIntOrFloatBitWidth() /
                                           i32_ty.getIntOrFloatBitWidth()));
        }
        vals[{i, j}].push_back(convertedElems);
      }
    }
  }
  return vals;
}

static WMMAInstrType getWMMAInstrTypeFromDot(DotOp op) {
  auto aOperandTy = op.getA().getType();
  auto aTensorTy = cast<RankedTensorType>(aOperandTy);
  auto aElemTy = aTensorTy.getElementType();
  auto bOperandTy = op.getB().getType();
  auto bTensorTy = cast<RankedTensorType>(bOperandTy);
  auto bElemTy = bTensorTy.getElementType();
  assert(aElemTy == bElemTy);
  auto cOperandTy = op.getC().getType();
  auto cTensorTy = cast<RankedTensorType>(cOperandTy);
  auto cElemTy = cTensorTy.getElementType();
  auto dOperandTy = op.getD().getType();
  auto dTensorTy = cast<RankedTensorType>(dOperandTy);
  auto dElemTy = dTensorTy.getElementType();
  assert(cElemTy == dElemTy);

  if (dElemTy.isF32() && aElemTy.isF16())
    return WMMAInstrType::FP32_FP16;
  if (dElemTy.isF32() && aElemTy.isBF16())
    return WMMAInstrType::FP32_BF16;
  if (dElemTy.isF16() && aElemTy.isF16())
    return WMMAInstrType::FP16_FP16;
  if (dElemTy.isBF16() && aElemTy.isBF16())
    return WMMAInstrType::BF16_BF16;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(8))
    return WMMAInstrType::INT32_IU8;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(4))
    return WMMAInstrType::INT32_IU4;

  return WMMAInstrType::NOT_APPLICABLE;
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     WMMAInstrType wmmaType, Value valA, Value valB, Value valC,
                     Type aElType, Type bElType, bool lowBits) {
  auto resType = valC.getType();
  Value bitsFlag = int_val(1, lowBits);
  switch (wmmaType) {
  case WMMAInstrType::FP32_FP16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP32_BF16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP16_FP16:
    return rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, bitsFlag});
  case WMMAInstrType::BF16_BF16:
    return rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, bitsFlag});
  case WMMAInstrType::INT32_IU8:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
        loc, TypeRange{resType},
        ValueRange{int_val(1, !aElType.isUnsignedInteger()), valA,
                   int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   bitsFlag});
  case WMMAInstrType::INT32_IU4:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
        loc, TypeRange{resType},
        ValueRange{int_val(1, !aElType.isUnsignedInteger()), valA,
                   int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   bitsFlag});
  default:
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return Value();
}

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();
  auto wmmaInstrType = getWMMAInstrTypeFromDot(op);

  auto loc = op.getLoc();
  Value a = op.getA();
  Value b = op.getB();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto elemTy = aTensorTy.getElementType();

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  int kWidth = aEncoding.getKWidth();

  auto repA =
      wmmaLayout.getWMMARepForOperands(aTensorTy.getShape(), elemTy, kWidth, 0);
  auto repB =
      wmmaLayout.getWMMARepForOperands(bTensorTy.getShape(), elemTy, kWidth, 1);

  assert(repA[1] == repB[0]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[0];
  auto numRepN = repB[1];
  auto numRepK = repA[1];

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedA, numRepM,
      wmmaLayout.getInstrPerStore()[0], numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedB, numRepN,
      wmmaLayout.getInstrPerStore()[1], numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = triton::gpu::getWarpSize(wmmaLayout);
  // TODO get rid of magic numbers
  unsigned vgprElemWidth = 32;
  unsigned paddedOutputElemSize =
      vgprElemWidth / dstElemTy.getIntOrFloatBitWidth();
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] * paddedOutputElemSize / warpSize;
  auto instPerStore = product(wmmaLayout.getInstrPerStore());
  auto dElemsToStorePerThreadPerInstr = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  int barrierMask = 0;
  for (int m = 0; m < numRepM; ++m) {
    if (m == 1) {
      // printValues(loc, rewriter, "a00", {ha[{m, 0}][0]});
      // printValues(loc, rewriter, "a01", {ha[{m, 0}][1]});
      // printValues(loc, rewriter, "a10", {ha[{m, 1}][0]});
      // printValues(loc, rewriter, "a11", {ha[{m, 1}][1]});
    }
    for (int n = 0; n < numRepN; ++n) {
      SmallVector<Value> acc(instPerStore, undef(vecTy));
      for (int i = 0; i < instPerStore; ++i) {
        for (unsigned v = 0; v < dElemsToStorePerThreadPerInstr; ++v) {
          acc[i] = insert_element(
              vecTy, acc[i],
              fc[m * numRepN * instPerStore * dElemsToStorePerThreadPerInstr +
                 n * instPerStore * dElemsToStorePerThreadPerInstr +
                 i * dElemsToStorePerThreadPerInstr + v],
              i32_val(v * instPerStore + i));
        }
      }
      for (size_t k = 0; k < numRepK; ++k) {
        for (int i = 0; i < instPerStore; ++i) {
          acc[i] =
              generateWMMAOp(rewriter, loc, wmmaInstrType, ha[{m, k}][i],
                             hb[{n, k}][0], acc[i], aTensorTy.getElementType(),
                             bTensorTy.getElementType(), i % 2 == 1);
        }
        // printValues(loc, rewriter, "HB", {hb[{n, k}][0]});
        /*if (k == 0) {
          rewriter.create<ROCDL::SchedBarrier>(loc, std::nullopt,
        barrierMask++); for (int z = 0; z < 256; ++z) barrier();
          rewriter.create<ROCDL::SchedBarrier>(loc, std::nullopt,
        barrierMask++);
        }*/
      }
      for (int i = 0; i < instPerStore; ++i) {
        for (int v = 0; v < dElemsToStorePerThreadPerInstr; ++v) {
          fc[m * numRepN * instPerStore * dElemsToStorePerThreadPerInstr +
             n * instPerStore * dElemsToStorePerThreadPerInstr +
             i * dElemsToStorePerThreadPerInstr + v] =
              extract_element(dstElemTy, acc[i], i32_val(v * instPerStore + i));
        }
      }
      // printValues(loc, rewriter, "FC", fc);
    }
  }
  // printValues(loc, rewriter, "acc", fc);
  // printValues(loc, rewriter, "acc", fc);
  // printValues(loc, rewriter, "acc", fc);
  //  replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

} // namespace

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDWmmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  return convertDot(op, adaptor, rewriter, typeConverter);
}
} // namespace mlir::triton::AMD
