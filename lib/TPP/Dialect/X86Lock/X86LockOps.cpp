//===- X86LockOps.cpp - X86Lock dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/X86Lock/X86LockOps.h"
#include "TPP/Dialect/X86Lock/X86LockDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::x86lock;

#define GET_OP_CLASSES
#include "TPP/Dialect/X86Lock/X86LockOps.cpp.inc"
