#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::AMD {

namespace {
template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     RewriterBase &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

// Extend all values to 64-bit per printf call requirements.
Value printfPromoteValue(RewriterBase &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto loc = UnknownLoc::get(context);
  auto type = value.getType();

  if (isa<LLVM::LLVMPointerType>(type)) {
    // The llvm.ptrtoint op requires signless integer types.
    return ptrtoint(i64_ty, value);
  }

  assert(type.getIntOrFloatBitWidth() <= 64);

  if (auto floatType = dyn_cast<FloatType>(type)) {
    Value newValue = value;
    if (!floatType.isF64())
      newValue = fpext(f64_ty, newValue);
    return bitcast(newValue, i64_ty);
  }

  assert(type.isIntOrIndex());
  if (type.getIntOrFloatBitWidth() < 64) {
    if (type.isUnsignedInteger())
      return zext(ui64_ty, value);
    if (type.isSignedInteger())
      return sext(i64_ty, value);
    // Signless integers are printed using unsigned integer formats.
    return zext(i64_ty, value);
  }

  return value;
}
} // namespace

int TargetInfo::getSharedMemorySize() const { return 64 * 1024; }

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // On AMD hardware we don't have CTA clusters like NVIDIA. So this will always
  // be zero. Whoever calling into this should make sure the whole program does
  // not try to utilize CTA clusters.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto stringAttr = rewriter.getStringAttr("llvm.amdgcn.ballot");
  SmallVector<Value> operands = {cmp};
  Value asmResult =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, type, stringAttr, operands)
          ->getResult(0);
  return asmResult;
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::AMD::llStore(rewriter, loc, ptr, val, pred);
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  llvm::report_fatal_error("AMDGPU does not support stmatrix");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = rewriter.create<LLVM::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::AMD::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::AMD::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::AMD::llGetPid(loc, rewriter, moduleOp, axis);
}

void accumulate(RewriterBase &rewriter, Region &combineOp,
                SmallVector<Value> &acc, ValueRange cur, bool isFirst) {
  if (isFirst) {
    acc = SmallVector<Value>(cur.begin(), cur.end());
    return;
  }

  // Create a new copy of the reduce block, and inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();
  rewriter.cloneRegionBefore(combineOp, &parent.front());
  auto &newReduce = parent.front();
  auto returnOp = dyn_cast<triton::ReduceReturnOp>(newReduce.getTerminator());

  llvm::SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
                             combineArgs);

  auto results = returnOp.getResult();
  for (unsigned i = 0; i < acc.size(); ++i) {
    acc[i] = results[i];
  }

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  if (numLaneToReduce != 64) {
    return false;
  }
  for (auto currAcc : acc) {
    Type srcType = currAcc.getType(); // ??????
    Type llvmType = nullptr;
    if (srcType.getIntOrFloatBitWidth() < 32) {
      llvmType = rewriter.getI32Type();
    } else if (isa<FloatType>(srcType)) {
      llvmType = (srcType.getIntOrFloatBitWidth() == 32)
                     ? rewriter.getF32Type()
                     : rewriter.getF64Type();
    } else if (isa<IntegerType>(srcType)) {
      llvmType = (srcType.getIntOrFloatBitWidth() == 32)
                     ? rewriter.getI32Type()
                     : rewriter.getI64Type();
    }
    /*
         |       bank0 |       bank1 |       bank2 |       bank3 |
    row0 | 0  1  2  3  | 4  5  6  7  | 8  9  10 11 | 12 13 14 15 |
    row1 | 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
    row2 | 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
    row3 | 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |

    Reduction steps:
    1. Reduce values with row_shr
    1.1. bank0, bank1, bank2, bank3 (row_shr8)-> bank2, bank3
    1.2. bank2, bank3 (row_shr4)-> bank3
    1.3. bank3 (row_shr2)-> 2 vals per row
    1.4. bank3 (row_shr1)-> 1 vals per row
    2. Reduce rows in a single bank with row_bcast
    2.1. row0, row1, row2, row3 (row_bcast15)-> row1, row3
    2.2. row1, row3 (row_bcast31)-> row3
    */
    unsigned origLanesToReduce = numLaneToReduce;
    unsigned numRows = origLanesToReduce == 64 ? 4 : 2;
    Value updatedAcc = currAcc;
    for (; numLaneToReduce > 1; numLaneToReduce /= 2) {
      unsigned rowOffset = (numLaneToReduce / numRows) / 2;
      unsigned dppCtrl = numLaneToReduce > numRows
                             ? 0x110 + rowOffset // row_shr
                             : 0x142 + static_cast<unsigned>(
                                           origLanesToReduce == 64 &
                                           numLaneToReduce == 2); // row_bcast
      unsigned rowMask = 0xF;
      rowMask &= ~((static_cast<unsigned>(numLaneToReduce == numRows))
                   << 1); // collect result in the 1 row for the last iterations
      rowMask &= (static_cast<unsigned>(numLaneToReduce == numRows &&
                                        origLanesToReduce == 64))
                 << 3; // collect result also in the 3 row on the 2.1 step
      unsigned bankMask = 0xF;
      bankMask &=
          ~((static_cast<unsigned>(numLaneToReduce == origLanesToReduce)) << 2);
      bankMask &=
          ~((static_cast<unsigned>(numLaneToReduce >= origLanesToReduce / 2))
            << 3);
      unsigned boundCtrl = numLaneToReduce <= numRows ||
                           bankMask != 0x0; // row reduction within one bank

      auto dppMovOp = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, llvmType, updatedAcc, updatedAcc, dppCtrl, rowMask, bankMask,
          boundCtrl);
      updatedAcc = dppMovOp.getRes();
      accumulate(rewriter, op.getCombineOp(), acc, updatedAcc, false);
    }
  }
  return true;
}

bool TargetInfo::processReplicaUsingStMatrix(
    RewriterBase &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzleByteWidth) const {
  return false;
}

void TargetInfo::printfImpl(Value formatStrStart, int formatStrByteCount,
                            ValueRange args, RewriterBase &rewriter,
                            bool useStdErr) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn = getOrInsertFunction(
      moduleOp, loc, rewriter,
      useStdErr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin",
      LLVM::LLVMFunctionType::get(i64_ty,
                                  useStdErr ? ArrayRef<Type>() : i64_ty));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          i64_ty, {i64_ty, ptr_ty(ctx), /*length=*/i64_ty, /*isLast=*/i32_ty}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            i64_ty, {i64_ty, /*numArgs=*/i32_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                     i64_ty, i64_ty, i64_ty, /*isLast=*/i32_ty}));
  }

  // Emit the intrinsic function call to begin the printf.
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64_ty, 0);
  Value message =
      call(printBeginFn, useStdErr ? ValueRange() : zeroI64).getResult();

  // Emit the intrinsic function call to handle the printf format string.
  Value oneI32 = i32_val(1);
  Value zeroI32 = i32_val(0);
  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, i64_ty, formatStrByteCount);
  SmallVector<Value, 4> arguments = {message, formatStrStart, formatStrLen,
                                     args.empty() ? oneI32 : zeroI32};
  message = call(printStrFn, arguments).getResult();

  // Emit the intrinsic function call to handle arguments iteratively.
  // We can only handle at most 7 values each time.
  constexpr size_t kArgsPerGroup = 7;
  for (size_t group = 0; group < args.size(); group += kArgsPerGroup) {
    size_t bound = std::min(group + kArgsPerGroup, args.size());
    size_t numArgs = bound - group;

    SmallVector<Value, 2 + kArgsPerGroup + 1> arguments;
    arguments.push_back(message);
    arguments.push_back(i32_val(numArgs));
    for (size_t i = group; i < bound; ++i) {
      arguments.push_back(printfPromoteValue(rewriter, args[i]));
    }
    // Pad out to 7 arguments since the function always needs 7 args.
    for (size_t extra = numArgs; extra < kArgsPerGroup; ++extra) {
      arguments.push_back(zeroI64);
    }

    Value isLast = (bound == args.size()) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    message = call(printArgsFn, arguments).getResult();
  }
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__ockl_mul_hi_u32" : "__ockl_mul_hi_u64";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args) const {
  return printfImpl(formatStrStart, formatStrByteCount, args, rewriter,
                    /*useStdError=*/false);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  // Compose and print an assert message.
  llvm::SmallString<256> msgBuffer;
  llvm::Twine("device assertion failed: '" + message + "', in " + func +
              " at " + file + ":" + llvm::Twine(line) + "\n\0")
      .toStringRef(msgBuffer);
  Value msgValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgBuffer);
  printfImpl(msgValue, msgBuffer.size_in_bytes(), /*args=*/ValueRange(),
             rewriter, /*useStdError=*/true);

  // Set block barrrier before aborting kernel, give a chance for all
  // the threads in a block to check/print the assert failure.
  barrier();
  // Perform the trap to abort the kernel.
  rewriter.create<LLVM::Trap>(loc);
}

} // namespace mlir::triton::AMD
