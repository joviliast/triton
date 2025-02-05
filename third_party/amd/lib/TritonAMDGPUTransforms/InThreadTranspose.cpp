#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

void convertLayout(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  // Convert operands
  // For load/store with tensor pointers, we don't have to change the
  // operands' type, we do this by changing the outputs' type of
  // `make_tensor_ptr`
  SmallVector<Value, 4> newArgs;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType && !isa<triton::gpu::SwizzledSharedEncodingAttr>(
                          tensorType.getEncoding())) {
      Type newType = getNewType(tensorType, encoding);
      newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Convert output types
  SmallVector<Type, 4> newTypes;
  for (auto t : op->getResultTypes()) {
    bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
    newTypes.push_back(isAsync ? t : getNewType(t, encoding));
  }

  // Construct new op with the new encoding
  Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                    newArgs, newTypes, op->getAttrs());

  // Cast the results back to the original layout
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value newResult = newOp->getResult(i);
    if (newTypes[i] != op->getResultTypes()[i]) {
      newResult = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
    }
    op->getResult(i).replaceAllUsesWith(newResult);
  }
  op->erase();
}

ttg::LinearEncodingAttr
createInThreadTransposedEncoding(ArrayRef<int64_t> shape,
                                 ttg::BlockedEncodingAttr srcEncoding) {
  auto srcLL = srcEncoding.toLinearLayout(shape);
  SmallVector<unsigned> newInRegOrder(srcEncoding.getOrder());
  int rank = shape.size();
  std::swap(newInRegOrder[rank - 2], newInRegOrder[rank - 1]);

  // Make in-register transposed tile
  auto ctx = srcEncoding.getContext();
  auto regDimName = StringAttr::get(ctx, "register");
  auto inRegTransposeTile = tt::identityStandardND(
      regDimName, srcEncoding.getSizePerThread(), newInRegOrder);
  // make sure basis in same order as in srcLayout
  SmallVector<StringAttr> outDimNames(srcLL.getOutDimNames());
  inRegTransposeTile = inRegTransposeTile.transposeOuts(outDimNames);

  // Copy original bases, and replace register tile with transposed one
  tt::LinearLayout::BasesT bases = srcLL.getBases();
  auto &regBase = *bases.find(regDimName);
  int regsTransposed = inRegTransposeTile.getInDimSizeLog2(regDimName);
  for (int i = 0; i < regsTransposed; ++i)
    regBase.second[i] = inRegTransposeTile.getBasis(regDimName, i);

  tt::LinearLayout transposedLL(bases, SmallVector<StringAttr>(outDimNames));
  return ttg::LinearEncodingAttr::get(ctx, transposedLL);
}

void transposeInRegsitersBeforeLocalAlloc(ttg::LocalAllocOp alloc) {
  auto operand = alloc.getSrc();
  OpBuilder builder(operand.getDefiningOp());

  auto operandType = alloc.getSrc().getType();
  auto operandEncoding =
      cast<ttg::BlockedEncodingAttr>(operandType.getEncoding());
  auto transposedEncoding =
      createInThreadTransposedEncoding(operandType.getShape(), operandEncoding);
  auto newType = getNewType(operand.getType(), transposedEncoding);
  auto inThreadTransposed =
      builder.create<ttg::ConvertLayoutOp>(alloc->getLoc(), newType, operand);
  alloc.setOperand(0, inThreadTransposed);
}

void changeSharedEncoding(ttg::LocalAllocOp alloc) {
  auto originalType = cast<ttg::MemDescType>(alloc.getResult().getType());
  auto sharedEnc =
      cast<ttg::SwizzledSharedEncodingAttr>(originalType.getEncoding());
  auto ctx = sharedEnc.getContext();
  auto sharedVec = sharedEnc.getVec();
  auto perPhase = sharedEnc.getPerPhase();
  auto maxPhase = sharedEnc.getMaxPhase();
  auto order = sharedEnc.getOrder();
  auto ctaLayout = sharedEnc.getCTALayout();

  // TODO replace SwizzledSharedEncodingAttr with special swizzling pattern
  auto newSharedEnc = ttg::SwizzledSharedEncodingAttr::get(
      ctx, sharedVec, perPhase, maxPhase, order, ctaLayout);
  auto newType = ttg::MemDescType::get(
      originalType.getShape(), originalType.getElementType(), newSharedEnc,
      originalType.getMemorySpace());

  alloc.getResult().setType(newType);
  assert(false);
}

/// Structure describes operations involved in local_alloc->local_load pattern
struct loadStoreLoadChainComponents {
  tt::LoadOp globalLoad;
  ttg::LocalAllocOp localAlloc;
  ttg::LocalLoadOp localLoad;
};

llvm::FailureOr<loadStoreLoadChainComponents>
matchThreadRakePattern(Value operand) {
  loadStoreLoadChainComponents pattern;
  auto localLoadCandidate = operand.getDefiningOp();
  pattern.localLoad = dyn_cast<ttg::LocalLoadOp>(localLoadCandidate);
  if (!pattern.localLoad) {
    LDBG("Did not find local load operation");
    return failure();
  }
  auto localAllocCandidate = pattern.localLoad.getSrc().getDefiningOp();
  pattern.localAlloc = dyn_cast<ttg::LocalAllocOp>(localAllocCandidate);
  if (!pattern.localAlloc) {
    LDBG("Did not find local alloc operation");
    return failure();
  }
  auto loaded = pattern.localAlloc.getSrc();
  pattern.globalLoad = dyn_cast<tt::LoadOp>(loaded.getDefiningOp());
  if (!pattern.globalLoad) {
    LDBG("Did not find global load operation");
    return failure();
  }

  // TODO implement general heuristic,
  // analyzing local load/store vectorization and estimating bank conflicts
  auto opTensorTy = cast<RankedTensorType>(operand.getType());
  auto opEnc = opTensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  if (!opDotOpEnc)
    return failure();
  if (!isa<ttg::AMDMfmaEncodingAttr>(opDotOpEnc.getParent())) {
    LDBG("Operand's parent encoding is not MFMA");
    return failure();
  }
  auto loadedEnc = cast<RankedTensorType>(loaded.getType()).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadedEnc);
  if (!blockedEnc)
    return failure();
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  auto order = blockedEnc.getOrder();
  if (order[0] != kDimNum) {
    return failure();
  }

  return pattern;
}

ttg::BlockedEncodingAttr getThreadRakedBlockedEnc(Value operand,
                                                  ModuleOp &mod) {
  // get the K dim according to dotOp operand's index
  auto tensorTy = cast<RankedTensorType>(operand.getType());
  auto shape = tensorTy.getShape();
  auto opEnc = tensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // get the current blocked encoding
  auto cvtOperand = operand.getDefiningOp()->getOperand(0);
  auto cvtOperandEnc =
      cast<RankedTensorType>(cvtOperand.getType()).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(cvtOperandEnc);
  // compute the sizePerThread for the new encoding
  auto sizePerThread = blockedEnc.getSizePerThread();
  auto elemsPerIter = product(sizePerThread);
  auto elemsTotal = blockedEnc.getTotalElemsPerThread(shape, tensorTy);
  // we need to know how many iteration each thread will load
  LDBG("elemsPerIter = " << elemsPerIter << "; elemsTotal = " << elemsTotal);
  auto numMaxIters = elemsTotal / elemsPerIter;
  auto bitwidth = tensorTy.getElementType().getIntOrFloatBitWidth();
  // LDBG("bitwidth = " << bitwidth);
  // Current the widest is set to ds_write_b64
  auto newKOuterDim = std::min(numMaxIters, 64 / bitwidth);
  LDBG("Choose the minimum of numIters: " << numMaxIters << " and numDtype: "
                                          << 64 / bitwidth);
  SmallVector<unsigned> newSizePerThread(sizePerThread);
  newSizePerThread[kDimNum] = newKOuterDim;

  // return the new blocked encoding
  auto order = blockedEnc.getOrder();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  return ttg::BlockedEncodingAttr::get(mod.getContext(), shape,
                                       newSizePerThread, order, numWarps,
                                       threadsPerWarp, numCTAs);
}

} // namespace

class TritonAMDGPUInThreadTransposePass
    : public TritonAMDGPUInThreadTransposeBase<
          TritonAMDGPUInThreadTransposePass> {

public:
  TritonAMDGPUInThreadTransposePass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](tt::DotOp dotOp) {
      LDBG("DotOp under inspection: " << dotOp);
      auto mod = dotOp->getParentOfType<ModuleOp>();

      auto tryToConvertToThreadRaked = [&](Value operand) {
        LDBG("Consider " << operand);
        // Dot operand
        auto matchResult = matchThreadRakePattern(operand);
        if (llvm::succeeded(matchResult)) {
          auto pattern = matchResult.value();
          LDBG("operand is K-outer");
          auto newBlockedEnc = getThreadRakedBlockedEnc(operand, mod);
          LDBG("operand newBlockedEnc = " << newBlockedEnc);
          convertLayout(newBlockedEnc, (Operation *)pattern.globalLoad);
          transposeInRegsitersBeforeLocalAlloc(pattern.localAlloc);
          changeSharedEncoding(pattern.localAlloc);
        } else {
          LDBG("operand is K-inner and nothing to be done");
        }
      };
      // Check opA
      tryToConvertToThreadRaked(dotOp.getA());

      // Check opB
      tryToConvertToThreadRaked(dotOp.getB());
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUInThreadTransposePass() {
  return std::make_unique<TritonAMDGPUInThreadTransposePass>();
}
