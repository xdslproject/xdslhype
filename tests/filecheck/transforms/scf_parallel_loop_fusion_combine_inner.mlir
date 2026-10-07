// RUN: xdsl-opt -p scf-parallel-loop-fusion --split-input-file %s | filecheck %s

// After the outer loops are fused, the innermost loops are combined, the bound
// of the second being moved before the first.
func.func @combine_after_fusion(%n: index, %m: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %u = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %one, %m[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    %u2 = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u2, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %v = memref.load %m[%i2, %j2] : memref<?x?xf64>
      %w = arith.addf %v, %one : f64
      memref.store %w, %m[%i2, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @combine_after_fusion
// CHECK:         "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      %u = arith.subi %n, %c0
// CHECK-NEXT:      %u2 = arith.subi %n, %c0
// CHECK-NEXT:      "scf.parallel"(%c0, %u, %c1)
// CHECK-NEXT:      ^bb{{[0-9]+}}(%j: index):
// CHECK-NEXT:        memref.store %one, %m[%i, %j]
// CHECK-NEXT:        %v = memref.load %m[%i, %j]
// CHECK-NEXT:        %w = arith.addf %v, %one
// CHECK-NEXT:        memref.store %w, %m[%i, %j]
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      })
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    })
// CHECK-NEXT:    func.return

// -----

// Three adjacent innermost loops are combined into one, in order.
func.func @three_inner(%n: index, %a: memref<?x?xf64>, %b: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %v = memref.load %a[%i, %j] : memref<?x?xf64>
      memref.store %v, %b[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %v2 = memref.load %b[%i, %j2] : memref<?x?xf64>
      %w2 = arith.addf %v2, %v2 : f64
      memref.store %w2, %b[%i, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j3: index):
      %v3 = memref.load %b[%i, %j3] : memref<?x?xf64>
      %w3 = arith.mulf %v3, %v3 : f64
      memref.store %w3, %b[%i, %j3] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @three_inner
// CHECK:         "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:      ^bb{{[0-9]+}}(%j: index):
// CHECK-NEXT:        %v = memref.load %a[%i, %j]
// CHECK-NEXT:        memref.store %v, %b[%i, %j]
// CHECK-NEXT:        %v2 = memref.load %b[%i, %j]
// CHECK-NEXT:        %w2 = arith.addf %v2, %v2
// CHECK-NEXT:        memref.store %w2, %b[%i, %j]
// CHECK-NEXT:        %v3 = memref.load %b[%i, %j]
// CHECK-NEXT:        %w3 = arith.mulf %v3, %v3
// CHECK-NEXT:        memref.store %w3, %b[%i, %j]
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      })
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    })
// CHECK-NEXT:    func.return

// -----

// Innermost loops with different bounds are not combined, but the outer loops
// are still fused.
func.func @different_inner_bounds(%n: index, %n2: index, %m: memref<?x?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %one, %m[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %n2, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      memref.store %one, %o[%i2, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @different_inner_bounds
// CHECK:         ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      "scf.parallel"(%c0, %n, %c1)
// CHECK:           "scf.parallel"(%c0, %n2, %c1)
// CHECK-NEXT:      ^bb{{[0-9]+}}(%j2: index):
// CHECK-NEXT:        memref.store %one, %o[%i, %j2]

// -----

// The second loop reads an element the first writes in another iteration.
func.func @inner_neighbour_read(%n: index, %m: memref<?x?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %one, %m[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %next = arith.addi %j2, %c1 : index
      %v = memref.load %m[%i, %next] : memref<?x?xf64>
      memref.store %v, %o[%i, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @inner_neighbour_read
// CHECK:         ^bb{{[0-9]+}}(%j2: index):

// -----

// A side effect between the loops cannot be moved before the first.
func.func @effect_between(%n: index, %m: memref<?x?xf64>, %o: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %one, %m[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    memref.store %one, %o[%i] : memref<?xf64>
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      memref.store %one, %m[%i, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @effect_between
// CHECK:         ^bb{{[0-9]+}}(%j2: index):

// -----

// Only loops nested in another scf.parallel are combined.
func.func @top_level(%n: index, %m: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %one, %m[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    memref.store %one, %m[%i2] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @top_level
// CHECK:         ^bb{{[0-9]+}}(%i2: index):
