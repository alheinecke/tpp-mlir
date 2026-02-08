//===- SCFForAllLoopFlattenSFC.cpp - Flatten 2D scf.forall ---------------===//
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

/****************************************************************************************
  BSD 2-Clause License

  Copyright (c) 2018, Jakub Červený
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************************/

/* The generalized hilbert functions are from: https://github.com/jakubcerveny/gilbert */

#define TPP_MLIR_SIGN(A) (0 < (A) ? (1) : ( 0 == (A) ? (0) : (-1)))

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
#define GEN_PASS_DECL_SCFFORALLLOOPFLATTENSFC
#define GEN_PASS_DEF_SCFFORALLLOOPFLATTENSFC
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Free functions for calculting generlized hilbert index from multi-dimensional indices and bounds
static int64_t tpp_mlir_gilbert_d2xy_r(int64_t dst_idx, int64_t cur_idx,
                       int64_t *xres, int64_t *yres,
                       int64_t ax,int64_t ay,
                       int64_t bx,int64_t by );

static int64_t tpp_mlir_gilbert_d2xy(int64_t *x, int64_t *y, int64_t idx, int64_t w,int64_t h);

static int64_t tpp_mlir_gilbert_d2xy(int64_t *x, int64_t *y, int64_t idx,int64_t w,int64_t h) {
  *x = 0;
  *y = 0;

  if (w >= h) {
    return tpp_mlir_gilbert_d2xy_r(idx,0, x,y, w,0, 0,h);
  }
  return tpp_mlir_gilbert_d2xy_r(idx,0, x,y, 0,h, w,0);
}

static int64_t tpp_mlir_gilbert_d2xy_r(int64_t dst_idx, int64_t cur_idx,
                       int64_t *xres, int64_t *yres,
                       int64_t ax,int64_t ay,
                       int64_t bx,int64_t by ) {
  int64_t nxt_idx;
  int64_t w, h, x, y,
      dax, day,
      dbx, dby,
      di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = std::abs(ax + ay);
  h = std::abs(bx + by);

  x = *xres;
  y = *yres;

  /* unit major direction */
  dax = TPP_MLIR_SIGN(ax);
  day = TPP_MLIR_SIGN(ay);

  /* unit orthogonal direction */
  dbx = TPP_MLIR_SIGN(bx);
  dby = TPP_MLIR_SIGN(by);

  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax*di;
    *yres = y + day*di;
    return 0;
  }

  if (w == 1) {
    *xres = x + dbx*di;
    *yres = y + dby*di;
    return 0;
  }

  /* floor function */
  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;

  w2 = std::abs(ax2 + ay2);
  h2 = std::abs(bx2 + by2);

  if ((2*w) > (3*h)) {
    if ((w2 & 1) && (w > 2)) {
      /* prefer even steps */
      ax2 += dax;
      ay2 += day;
    }

    /* long case: split in two parts only */
    nxt_idx = cur_idx + std::abs((ax2 + ay2)*(bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return tpp_mlir_gilbert_d2xy_r(dst_idx, cur_idx,  xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;

    *xres = x + ax2;
    *yres = y + ay2;
    return tpp_mlir_gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax-ax2, ay-ay2, bx, by);
  }

  if ((h2 & 1) && (h > 2)) {
    /* prefer even steps */
    bx2 += dbx;
    by2 += dby;
  }

  /* standard case: one step up, one long horizontal, one step down */
  nxt_idx = cur_idx + std::abs((bx2 + by2)*(ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return tpp_mlir_gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, bx2,by2, ax2,ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + std::abs((ax + ay)*((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return tpp_mlir_gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, ax,ay, bx-bx2,by-by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return tpp_mlir_gilbert_d2xy_r(dst_idx, cur_idx,
                        xres,yres,
                        -bx2, -by2,
                        -(ax-ax2), -(ay-ay2));
}


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
#if 0
      int64_t iv0val = *lb0 + i * *step0;
      int64_t iv1val = *lb1 + j * *step1;
#else
      int64_t iv0val = 0;
      int64_t iv1val = 0;
      tpp_mlir_gilbert_d2xy(&iv0val, &iv1val, i*count1 + j, count0, count1);
#endif

      iv0Values.push_back(iv0val);
      iv1Values.push_back(iv1val);
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

struct SCFForAllLoopFlattenSFC
    : public tpp::impl::SCFForAllLoopFlattenSFCBase<SCFForAllLoopFlattenSFC> {
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
