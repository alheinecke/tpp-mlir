//===- X86LockOps.td - X86Lock dialect ops ---------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_X86LOCK_OPS
#define TPP_X86LOCK_OPS

include "TPP/Dialect/X86Lock/X86LockDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SetLockOp
//===----------------------------------------------------------------------===//

def X86Lock_SetLockOp : X86Lock_Op<"setLock", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Acquire a lock (busy-waiting).";
  let description = [{
    Acquire a lock using atomic operations. This operation implements
    busy-waiting until the lock is successfully acquired.
    
    The lock is represented as a scalar i8 memref where:
    - 0 indicates the lock is free
    - 1 indicates the lock is held
    
    This operation mirrors OpenMP's omp_set_lock functionality.
    
    Example:
    ```mlir
    %lock = memref.alloca() : memref<i8>
    x86lock.setLock %lock : memref<i8>
    ```
  }];

  let arguments = (ins MemRefRankOf<[I8], [0]>:$lock);
  let assemblyFormat = "$lock attr-dict `:` type($lock)";
}

//===----------------------------------------------------------------------===//
// UnsetLockOp
//===----------------------------------------------------------------------===//

def X86Lock_UnsetLockOp : X86Lock_Op<"unsetLock", [MemoryEffects<[MemWrite]>]> {
  let summary = "Release a lock.";
  let description = [{
    Release a previously acquired lock using atomic operations.
    
    The lock is set to 0 (free) using an atomic store operation.
    
    This operation mirrors OpenMP's omp_unset_lock functionality.
    
    Example:
    ```mlir
    %lock = memref.alloca() : memref<i8>
    x86lock.setLock %lock : memref<i8>
    // ... critical section ...
    x86lock.unsetLock %lock : memref<i8>
    ```
  }];

  let arguments = (ins MemRefRankOf<[I8], [0]>:$lock);
  let assemblyFormat = "$lock attr-dict `:` type($lock)";
}

#endif // TPP_X86LOCK_OPS
