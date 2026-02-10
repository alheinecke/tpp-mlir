//===- X86LockDialect.cpp - X86Lock dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/X86Lock/X86LockDialect.h"
#include "TPP/Dialect/X86Lock/X86LockOps.h"

using namespace mlir;
using namespace mlir::x86lock;

#include "TPP/Dialect/X86Lock/X86LockOpsDialect.cpp.inc"

void X86LockDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/X86Lock/X86LockOps.cpp.inc"
      >();
}
