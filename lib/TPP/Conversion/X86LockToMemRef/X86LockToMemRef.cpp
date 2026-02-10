//===- X86LockToMemRef.cpp - X86Lock to MemRef conversion ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Conversion/X86LockToMemRef/X86LockToMemRef.h"
#include "TPP/Dialect/X86Lock/X86LockDialect.h"
#include "TPP/Dialect/X86Lock/X86LockOps.h"
#include "TPP/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::x86lock;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTX86LOCKTOMEMREF
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

/// Lower SetLockOp to memref.generic_atomic_rmw with busy-waiting
struct SetLockOpLowering : public OpRewritePattern<SetLockOp> {
  using OpRewritePattern<SetLockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SetLockOp op,
                                  PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lockMemRef = op.getLock();
    
    // Implement: int expected = 0; while (!compare_exchange_weak(expected, 1)) { expected = 0; }
    // Using scf.while with atomic_rmw assign (swap)
    auto whileOp = rewriter.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    
    // Build the "before" region: try atomic assign 1, continue if old value was NOT 0
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *beforeBlock = rewriter.createBlock(&whileOp.getBefore());
      rewriter.setInsertionPointToStart(beforeBlock);
      
      Value c0_i8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
      Value c1_i8 = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
      
      // Atomic assign: swap lock to 1, get old value
      auto atomicOp = rewriter.create<memref::AtomicRMWOp>(
          loc, arith::AtomicRMWKind::assign, c1_i8, lockMemRef, ValueRange{});
      
      // Check if NOT acquired (old value != 0, meaning lock was already held)
      Value notAcquired = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, atomicOp.getResult(), c0_i8);
      
      // Continue looping if we didn't acquire the lock
      rewriter.create<scf::ConditionOp>(loc, notAcquired, ValueRange{});
    }
    
    // Build the "after" region: just yield to loop back
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *afterBlock = rewriter.createBlock(&whileOp.getAfter());
      rewriter.setInsertionPointToStart(afterBlock);
      rewriter.create<scf::YieldOp>(loc, ValueRange{});
    }
    
    rewriter.replaceOp(op, whileOp.getResults());
    return success();
  }
};

/// Lower UnsetLockOp to memref.generic_atomic_rmw
struct UnsetLockOpLowering : public OpRewritePattern<UnsetLockOp> {
  using OpRewritePattern<UnsetLockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnsetLockOp op,
                                  PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lockMemRef = op.getLock();
    
    // Implement: *lock_var = 0
    // Simple store to release the lock
    Value c0_i8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    rewriter.create<memref::StoreOp>(loc, c0_i8, lockMemRef);
    
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertX86LockToMemRefPass
    : public tpp::impl::ConvertX86LockToMemRefBase<ConvertX86LockToMemRefPass> {
  using ConvertX86LockToMemRefBase::ConvertX86LockToMemRefBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    
    RewritePatternSet patterns(&getContext());
    patterns.add<SetLockOpLowering, UnsetLockOpLowering>(&getContext());
    
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::x86lock::createConvertX86LockToMemRefPass() {
  return std::make_unique<ConvertX86LockToMemRefPass>();
}
