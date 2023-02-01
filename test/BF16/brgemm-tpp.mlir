// RUN: tpp-opt -pack-vnni -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -convert-vnni-to-tpp %s | FileCheck %s

// RUN: tpp-opt -pack-vnni %s | FileCheck %s -check-prefix=PACK

func.func @myfunc(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  // PACK: {{.+}} = tensor.pack %{{.+}} inner_dims_pos = [1] inner_tiles = [2] 
  // PACK-SAME:     into %{{.+}} : tensor<32x4x4xbf16> -> tensor<32x2x4x2xbf16>
  // CHECK: tpp.vnni_brgemm ins(%{{.*}} : memref<32x4x4xbf16>, %{{.*}} : memref<32x2x4x2xbf16>) out(%{{.*}} : memref<4x4xbf16>)
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}