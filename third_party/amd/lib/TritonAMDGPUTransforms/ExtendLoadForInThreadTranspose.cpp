#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-extend-load-for-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUEXTENDLOADFORINTHREADTRANSPOSE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static Type replaceEncoding(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

void refineGlobalLoadLayout(PatternRewriter &rewriter, Attribute encoding,
                            tt::LoadOp load) {
  auto loc = load->getLoc();
  rewriter.setInsertionPoint(load);
  // Convert operands
  SmallVector<Value, 4> newArgs;
  for (auto operand : load->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType) {
      Type newType = replaceEncoding(tensorType, encoding);
      newArgs.push_back(
          rewriter.create<ttg::ConvertLayoutOp>(loc, newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Construct new load with the new encoding
  auto attrs = load->getAttrs();
  auto newLoad = rewriter.create<tt::LoadOp>(loc, newArgs, attrs);

  // Cast the results back to the original layout
  auto loadType = load.getType();
  Value newResult = newLoad.getResult();
  rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(load, loadType, newResult);
}

class ExtendGlobalLoadPattern : public OpRewritePattern<tt::LoadOp> {
public:
  ExtendGlobalLoadPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tt::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value loaded = loadOp.getResult();
    for (auto &use : loaded.getUses()) {
      Operation *userOp = use.getOwner();

      // TODO: if use passes as arguments check bb internals recursively
      auto convertLayout = dyn_cast<ttg::ConvertLayoutOp>(userOp);
      if (!convertLayout)
        continue;
      auto tensorType =
          dyn_cast<RankedTensorType>(convertLayout->getResultTypes().front());
      if (!tensorType)
        continue;
      auto outEncoding = tensorType.getEncoding();
      refineGlobalLoadLayout(rewriter, outEncoding, loadOp);
    }

    return success();
  }
};

} // anonymous namespace

class TritonAMDGPUExtendLoadForInThreadTransposePass
    : public impl::TritonAMDGPUExtendLoadForInThreadTransposeBase<
          TritonAMDGPUExtendLoadForInThreadTransposePass> {

public:
  void runOnOperation() override {
    tt::FuncOp f = getOperation();

    auto ctx = f.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ExtendGlobalLoadPattern>(ctx, /*benefit=*/1);
    walkAndApplyPatterns(f, std::move(patterns));
  }
};

} // namespace mlir
