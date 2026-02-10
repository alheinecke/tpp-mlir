//===- X86LockOps.h - X86Lock dialect ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_X86LOCK_X86LOCKOPS_H
#define TPP_DIALECT_X86LOCK_X86LOCKOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/X86Lock/X86LockOps.h.inc"

#endif // TPP_DIALECT_X86LOCK_X86LOCKOPS_H
