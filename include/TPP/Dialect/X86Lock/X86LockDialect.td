//===- X86LockDialect.td - X86Lock dialect ---------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_X86LOCK_DIALECT
#define TPP_X86LOCK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// X86Lock dialect definition.
//===----------------------------------------------------------------------===//

def X86Lock_Dialect : Dialect {
    let name = "x86lock";
    let summary = "X86 lock operations dialect.";
    let description = [{
        This dialect provides atomic lock operations similar to OpenMP's
        omp_set_lock and omp_unset_lock functionality, using x86 atomic
        instructions.
    }];
    let cppNamespace = "::mlir::x86lock";
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class X86Lock_Op<string mnemonic, list<Trait> traits = []> :
        Op<X86Lock_Dialect, mnemonic, traits>;

#endif // TPP_X86LOCK_DIALECT
