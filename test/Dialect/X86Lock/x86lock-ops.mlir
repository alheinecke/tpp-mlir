// RUN: tpp-opt %s -convert-x86lock-to-memref -split-input-file | FileCheck %s
// RUN: tpp-opt %s -convert-x86lock-to-memref -convert-scf-to-cf -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -split-input-file | FileCheck %s --check-prefix=LLVM

func.func @test_lock() {
  %lock = memref.alloca() : memref<i8>
  
  // CHECK: %[[C1:.*]] = arith.constant 1 : i8
  // CHECK: %[[C0:.*]] = arith.constant 0 : i8
  // CHECK: scf.while
  // CHECK: %[[OLD:.*]] = memref.atomic_rmw assign %[[C1]], %{{.*}}[] : (i8, memref<i8>) -> i8
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[OLD]], %[[C0]] : i8
  // CHECK: scf.condition(%[[CMP]])
  // CHECK: scf.yield
  x86lock.setLock %lock : memref<i8>
  
  // Critical section would go here
  
  // CHECK: memref.store %[[C0]], %{{.*}}[] : memref<i8>
  
  // LLVM-LABEL: llvm.func @test_lock
  // LLVM: %[[C1:.*]] = llvm.mlir.constant(1 : i8) : i8
  // LLVM: %[[C0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // LLVM: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i8
  // LLVM: cf.br ^[[LOOP:.*]]
  // LLVM: ^[[LOOP]]:
  // LLVM: %[[PTR:.*]] = llvm.extractvalue %{{.*}}[1]
  // LLVM: %[[XCHG:.*]] = llvm.atomicrmw xchg %[[PTR]], %[[C1]] acq_rel
  // LLVM: %[[CMP:.*]] = llvm.icmp "ne" %[[XCHG]], %[[C0]]
  // LLVM: cf.cond_br %[[CMP]], ^[[LOOP]], ^[[EXIT:.*]]
  // LLVM: ^[[EXIT]]:
  // LLVM: %[[STORE_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
  // LLVM: llvm.store %[[C0]], %[[STORE_PTR]]
  x86lock.unsetLock %lock : memref<i8>
  
  return
}

// -----

func.func @test_multiple_locks() {
  %lock1 = memref.alloca() : memref<i8>
  %lock2 = memref.alloca() : memref<i8>
  
  // CHECK: %[[C1:.*]] = arith.constant 1 : i8
  // CHECK: %[[C0:.*]] = arith.constant 0 : i8
  // CHECK: scf.while
  // CHECK: memref.atomic_rmw assign %[[C1]]
  // CHECK: arith.cmpi ne
  x86lock.setLock %lock1 : memref<i8>
  // CHECK: scf.while
  // CHECK: memref.atomic_rmw assign %[[C1]]
  // CHECK: arith.cmpi ne
  x86lock.setLock %lock2 : memref<i8>
  
  // Critical section with both locks held
  
  // CHECK: memref.store %[[C0]]
  x86lock.unsetLock %lock2 : memref<i8>
  // CHECK: memref.store %[[C0]]
  
  // LLVM-LABEL: llvm.func @test_multiple_locks
  // LLVM: %[[C1:.*]] = llvm.mlir.constant(1 : i8) : i8
  // LLVM: %[[C0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // LLVM: llvm.alloca
  // LLVM: llvm.alloca
  // LLVM: cf.br ^[[LOOP1:.*]]
  // LLVM: ^[[LOOP1]]:
  // LLVM: llvm.atomicrmw xchg %{{.*}}, %[[C1]] acq_rel
  // LLVM: llvm.icmp "ne"
  // LLVM: cf.cond_br %{{.*}}, ^[[LOOP1]], ^[[NEXT:.*]]
  // LLVM: cf.br ^[[LOOP2:.*]]
  // LLVM: ^[[LOOP2]]:
  // LLVM: llvm.atomicrmw xchg %{{.*}}, %[[C1]] acq_rel
  // LLVM: llvm.icmp "ne"
  // LLVM: cf.cond_br
  // LLVM: llvm.store %[[C0]]
  // LLVM: llvm.store %[[C0]]
  x86lock.unsetLock %lock1 : memref<i8>
  
  return
}
