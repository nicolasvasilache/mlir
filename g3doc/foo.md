```mlir {.mlir}
func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  
  %1 = "s.op" #attrs (%0): (tensor<?xf32>) -> tensor<?xvector<4xf32>>
  %_ = "s.op" {args_in: 1, args_out: 1} #attrs (%2, %3): (memref<?xf32>, memref<?xvector<4xf32>>) -> ()
  
  %M = "dim" %2, 0: index
  %N = "dim" %3, 0: index
  %eq = "eq" %M, %N: i1 // iteration space is consistent with data 
  %_ = "assert" i1: () -> ()
  for %i = 0 to %M
    %a = "load" %2, %i: memref<?xf32>
    %b = "load" %3, %i: memref<?xvector<4xf32>>
    %c = "some_compute"(%a, %b): (?) -> (?)
    %_ = "store" %c, %3, %i: memref<?xvector<4xf32>>
    
    
  %1 = "s.op" {indexing: (i, j) -> (j, i), (i, j) -> (j)} #attrs (%0): (tensor<8x?xf32>) -> tensor<?xvector<4xf32>>
  %_ = "s.op" {args_in: 1, args_out: 1, indexing: (i, j) -> (i, j), (i, j) -> (j)} #attrs (%2, %3): (memref<?x?xf32>, memref<?xvector<4xf32>>) -> ()
      
  %J = "dim" %2, 0: index
  %I = "dim" %2, 1: index
  %J2 = "dim" %3, 0: index
  %eq = "eq" %J, %J2: i1 // iteration space is consistent with data 
  %_ = "assert" i1: () -> ()
  for %i = 0 to %I       // loop order is fully defined by indexing maps
    for %j = 0 to %J     // arbitrary permutations are possible
      %a = "load" %2, %j, %i: memref<8x?xf32>
      %b = "load" %3, %j: memref<?xvector<4xf32>>
      %c = "some_compute"(%a, %b): (?) -> (?)
      %_ = "store" %c, %3, %i: memref<?xvector<4xf32>>
    
    
  %c = "s.op" #pointwise_2d (%a, %b) {
    ^bb0(%a: f32, %b: f32):
      %c = "addf" %c, %arg5 : f32
      "s.yield" %c : f32
    }: (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
        
  %_ = "s.op" #pointwise_2d (%A, %B, %C) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %r = "addf" %a, %b : f32
      "s.yield" %r : f32
    }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>

  return 
}

```

```mlir {.mlir}
  #matmul_accesses = [
    (m, n, k) -> (m, k),
    (m, n, k) -> (k, n),
    (m, n, k) -> (m, n)
  ]
  #matmul_trait = {
    n_views = [2, 1],
    n_loop_types = [2, 1, 0],
    indexing_maps = #matmul_accesses,
    library_call = "some_external_function_name_for_vector_outerproduct_matmul"
  }
  
  !vector_type_A = type vector<4xf32>
  !vector_type_B = type vector<4xf32>
  !vector_type_C = type vector<4x4xf32>
  
  !matrix_type_A = type memref<?x?x!vector_type_A>
  !matrix_type_B = type memref<?x?x!vector_type_B>
  !matrix_type_C = type memref<?x?x!vector_type_C>
  
  func @matmul_vec_impl(%A: !matrix_type_A, %B: !matrix_type_B, %C: !matrix_type_C) {
    %_ = linalg.generic #matmul_trait (%A, %B, %C) {
      ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
        %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
        linalg.yield %d: !vector_type_C
    } : !matrix_type_A, !matrix_type_B, !matrix_type_C
  
    return
  }
```
