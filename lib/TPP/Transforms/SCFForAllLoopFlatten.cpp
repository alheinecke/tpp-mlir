//===- SCFForAllLoopFlatten.cpp - Flatten 2D scf.forall ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements flattening of 2D forall loops into 1D with index
// vectors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_SCFFORALLLOOPFLATTEN
#define GEN_PASS_DEF_SCFFORALLLOOPFLATTEN
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Flatten a 2D forall loop of the form:
///   scf.forall (%i, %j) in (%ub0, %ub1) {
///     // body
///   }
///
/// into:
///   %iv_i = arith.constant dense<[...]> : vector<NxI64>
///   %iv_j = arith.constant dense<[...]> : vector<NxI64>
///   scf.forall (%idx) in (%cN) {
///     %i_i64 = vector.extract %iv_i[%idx] : i64 from vector<NxI64>
///     %j_i64 = vector.extract %iv_j[%idx] : i64 from vector<NxI64>
///     %i = arith.index_cast %i_i64 : i64 to index
///     %j = arith.index_cast %j_i64 : i64 to index
///     // original body using %i and %j
///   }
///
/// where N is the total iteration count ub0 * ub1
static LogicalResult flattenForallLoop(ForallOp op, OpBuilder &builder) {
  // Only handle 2D forall loops
  if (op.getRank() != 2) {
    return failure();
  }

  Location loc = op.getLoc();
  builder.setInsertionPoint(op);

  // Get loop bounds - forall uses mixed bounds (can be values or attributes)
  SmallVector<OpFoldResult> lowerBounds = op.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = op.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = op.getMixedStep();

  // Helper to extract constant int from OpFoldResult
  auto getConstant = [](OpFoldResult ofr) -> std::optional<int64_t> {
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return intAttr.getInt();
      }
    }
    if (auto val = dyn_cast<Value>(ofr)) {
      if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
        return constOp.value();
      }
      if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>()) {
        return constOp.value();
      }
    }
    return std::nullopt;
  };

  // Extract constant values
  auto lb0 = getConstant(lowerBounds[0]);
  auto lb1 = getConstant(lowerBounds[1]);
  auto ub0 = getConstant(upperBounds[0]);
  auto ub1 = getConstant(upperBounds[1]);
  auto step0 = getConstant(steps[0]);
  auto step1 = getConstant(steps[1]);

  // We need constant bounds to generate the dense vectors
  if (!lb0 || !lb1 || !ub0 || !ub1 || !step0 || !step1) {
    return failure();
  }

  // Calculate iteration counts
  int64_t count0 = (*ub0 - *lb0) / *step0;
  int64_t count1 = (*ub1 - *lb1) / *step1;
  int64_t totalCount = count0 * count1;

  if (totalCount <= 0) {
    return failure();
  }

  // Build the flattened index vectors
  SmallVector<int64_t> iv0Values;
  SmallVector<int64_t> iv1Values;

  for (int64_t i = 0; i < count0; ++i) {
    for (int64_t j = 0; j < count1; ++j) {
      iv0Values.push_back(*lb0 + i * *step0);
      iv1Values.push_back(*lb1 + j * *step1);
    }
  }

  // Create dense constant vectors
  auto vectorType = VectorType::get(ArrayRef<int64_t>{totalCount}, builder.getI64Type());
  auto iv0Attr = DenseElementsAttr::get(vectorType, ArrayRef<int64_t>(iv0Values));
  auto iv1Attr = DenseElementsAttr::get(vectorType, ArrayRef<int64_t>(iv1Values));

  Value iv0Vector = builder.create<arith::ConstantOp>(loc, vectorType, iv0Attr);
  Value iv1Vector = builder.create<arith::ConstantOp>(loc, vectorType, iv1Attr);

  // Create the new 1D forall loop
  SmallVector<OpFoldResult> newLowerBound = {builder.getIndexAttr(*lb0)};
  SmallVector<OpFoldResult> newUpperBound = {builder.getIndexAttr(totalCount)};
  SmallVector<OpFoldResult> newStep = {builder.getIndexAttr(*step0)};

  auto newLoop = builder.create<ForallOp>(
      loc, newLowerBound, newUpperBound, newStep,
      op.getOutputs(), op.getMapping());

  // Build the body of the new loop
  builder.setInsertionPointToStart(newLoop.getBody());

  Value idx = newLoop.getInductionVars()[0];

  // Extract the original induction variable values using vector.extract with dynamic position
  Value i = builder.create<vector::ExtractOp>(loc, iv0Vector, idx);
  Value j = builder.create<vector::ExtractOp>(loc, iv1Vector, idx);

  // Convert from i64 to index
  Value iIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), i);
  Value jIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), j);

  // Clone the original loop body
  IRMapping mapper;
  mapper.map(op.getInductionVars()[0], iIndex);
  mapper.map(op.getInductionVars()[1], jIndex);

  // Map block arguments for outputs if any
  for (auto [oldArg, newArg] : llvm::zip(op.getRegionIterArgs(), newLoop.getRegionIterArgs())) {
    mapper.map(oldArg, newArg);
  }

  for (auto &bodyOp : op.getBody()->without_terminator()) {
    builder.clone(bodyOp, mapper);
  }

  // Handle the terminator (scf.forall.in_parallel)
  // Only clone terminator contents if there are outputs (shared_outs)
  if (!op.getOutputs().empty()) {
    Operation *oldTerminator = op.getBody()->getTerminator();
    Operation *newTerminator = newLoop.getBody()->getTerminator();
    
    if (auto oldInParallel = dyn_cast<scf::InParallelOp>(oldTerminator)) {
      if (auto newInParallel = dyn_cast<scf::InParallelOp>(newTerminator)) {
        // Clone the operations inside the in_parallel block
        builder.setInsertionPointToStart(newInParallel.getBody());
        for (auto &inParallelOp : oldInParallel.getBody()->without_terminator()) {
          builder.clone(inParallelOp, mapper);
        }
      }
    }
  }

  // Replace uses of the old forall with the new forall results
  op.replaceAllUsesWith(newLoop);

  // Erase the original forall loop
  op.erase();

  return success();
}

namespace {

// Helper to collect innermost forall loops
static void getInnermostForallLoops(Operation *rootOp,
                                     SmallVectorImpl<ForallOp> &result) {
  rootOp->walk([&](ForallOp forallOp) {
    // Check if this forall contains any nested forall ops
    bool hasNestedForall = false;
    forallOp->walk([&](ForallOp nestedOp) {
      if (nestedOp != forallOp) {
        hasNestedForall = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!hasNestedForall) {
      result.push_back(forallOp);
    }
  });
}

struct SCFForAllLoopFlatten
    : public tpp::impl::SCFForAllLoopFlattenBase<SCFForAllLoopFlatten> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    
    // Collect all innermost forall loops with 2 induction variables
    SmallVector<ForallOp, 4> innermostForalls;
    getInnermostForallLoops(parentOp, innermostForalls);

    OpBuilder builder(&getContext());

    // Process each innermost forall loop
    for (ForallOp forallOp : innermostForalls) {
      // Only process loops with exactly 2 induction variables
      if (forallOp.getRank() == 2) {
        if (failed(flattenForallLoop(forallOp, builder))) {
          // If flattening fails for any reason, just skip this loop
          // (e.g., non-constant bounds, etc.)
          continue;
        }
      }
    }
  }
};
} // namespace
