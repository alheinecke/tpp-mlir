//===- X86LockToMemRef.h - X86Lock to MemRef conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_CONVERSION_X86LOCKTOMEMREF_X86LOCKTOMEMREF_H
#define TPP_CONVERSION_X86LOCKTOMEMREF_X86LOCKTOMEMREF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace x86lock {

/// Create a pass to convert X86Lock operations to MemRef atomic operations.
std::unique_ptr<OperationPass<func::FuncOp>> createConvertX86LockToMemRefPass();

} // namespace x86lock
} // namespace mlir

#endif // TPP_CONVERSION_X86LOCKTOMEMREF_X86LOCKTOMEMREF_H
