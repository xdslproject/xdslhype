// RUN: xdsl-opt -p "scf-parallel-loop-fusion{combine_inner=false}" --split-input-file %s | filecheck %s

// Nests of equal depth: the outer loops are fused, the innermost loops stay
// separate. Bounds defined inside the loops are compared structurally, and the
// second nest's are moved before the fused inner loop.
func.func @same_depth(%n: index, %m: memref<?x?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %u = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%k: index):
        memref.store %one, %m[%i, %j, %k] : memref<?x?x?xf64>
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    %u2 = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u2, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%k2: index):
        %v = memref.load %m[%i2, %j2, %k2] : memref<?x?x?xf64>
        %w = arith.addf %v, %one : f64
        memref.store %w, %m[%i2, %j2, %k2] : memref<?x?x?xf64>
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @same_depth
// CHECK:         "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      %u = arith.subi %n, %c0
// CHECK-NEXT:      %u2 = arith.subi %n, %c0
// CHECK-NEXT:      "scf.parallel"(%c0, %u, %c1)
// CHECK-NEXT:      ^bb{{[0-9]+}}(%j: index):
// CHECK-NEXT:        "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:        ^bb{{[0-9]+}}(%k: index):
// CHECK-NEXT:          memref.store %one, %m[%i, %j, %k]
// CHECK-NEXT:          scf.reduce
// CHECK-NEXT:        })
// CHECK-NEXT:        "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:        ^bb{{[0-9]+}}(%k2: index):
// CHECK-NEXT:          %v = memref.load %m[%i, %j, %k2]
// CHECK-NEXT:          %w = arith.addf %v, %one
// CHECK-NEXT:          memref.store %w, %m[%i, %j, %k2]
// CHECK-NEXT:          scf.reduce
// CHECK-NEXT:        })
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      })
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    })
// CHECK-NEXT:    func.return

// -----

// Depths differing by one: the shallower nest's innermost loop is fused with
// the outer loop of the deeper one.
func.func @depth_difference(%n: index, %m: memref<?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  %u = arith.addi %n, %c1 : index
  "scf.parallel"(%c0, %u, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %one, %m[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %u, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    %v = memref.load %m[%i2] : memref<?xf64>
    %ub = arith.muli %n, %c1 : index
    "scf.parallel"(%c0, %ub, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %v, %o[%i2, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @depth_difference
// CHECK:         "scf.parallel"(%c0, %u, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      memref.store %one, %m[%i]
// CHECK-NEXT:      %v = memref.load %m[%i]
// CHECK-NEXT:      %ub = arith.muli %n, %c1
// CHECK-NEXT:      "scf.parallel"(%c0, %ub, %c1)
// CHECK-NEXT:      ^bb{{[0-9]+}}(%j: index):
// CHECK-NEXT:        memref.store %v, %o[%i, %j]
// CHECK:             scf.reduce
// CHECK-NEXT:      })
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    })
// CHECK-NEXT:    func.return

// -----

// Three nests are fused into one.
func.func @three_way(%n: index, %a: memref<?x?xf64>, %b: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %u = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %v = memref.load %a[%i, %j] : memref<?x?xf64>
      memref.store %v, %b[%i, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    %u2 = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u2, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %v2 = memref.load %b[%i2, %j2] : memref<?x?xf64>
      %w2 = arith.addf %v2, %v2 : f64
      memref.store %w2, %b[%i2, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i3: index):
    %u3 = arith.subi %n, %c0 : index
    "scf.parallel"(%c0, %u3, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j3: index):
      %v3 = memref.load %b[%i3, %j3] : memref<?x?xf64>
      %w3 = arith.mulf %v3, %v3 : f64
      memref.store %w3, %b[%i3, %j3] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @three_way
// CHECK:         "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK-NEXT:      %u = arith.subi %n, %c0
// CHECK-NEXT:      "scf.parallel"(%c0, %u, %c1)
// CHECK:             memref.store %v, %b[%i, %j]
// CHECK:           %u2 = arith.subi %n, %c0
// CHECK-NEXT:      "scf.parallel"(%c0, %u2, %c1)
// CHECK:             %v2 = memref.load %b[%i, %j2]
// CHECK:             memref.store %w2, %b[%i, %j2]
// CHECK:           %u3 = arith.subi %n, %c0
// CHECK-NEXT:      "scf.parallel"(%c0, %u3, %c1)
// CHECK:             %v3 = memref.load %b[%i, %j3]
// CHECK:             memref.store %w3, %b[%i, %j3]
// CHECK-NOT:     "scf.parallel"
// CHECK:         func.return

// -----

// Loops with different bounds are not fused.
func.func @different_bounds(%n: index, %n2: index, %m: memref<?xf64>, %o: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      memref.store %one, %m[%j] : memref<?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n2, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      memref.store %one, %o[%j2] : memref<?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @different_bounds
// CHECK:         ^bb{{[0-9]+}}(%i2: index):

// -----

// Depths differing by two are not fused.
func.func @depth_two_apart(%n: index, %m: memref<?xf64>, %o: memref<?x?x?xf64>) {
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
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%k: index):
        memref.store %one, %o[%i2, %j, %k] : memref<?x?x?xf64>
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @depth_two_apart
// CHECK:         ^bb{{[0-9]+}}(%i2: index):

// -----

// The second nest reads an element the first writes in another iteration.
func.func @neighbour_read(%n: index, %m: memref<?x?xf64>, %o: memref<?x?xf64>) {
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
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %next = arith.addi %i2, %c1 : index
      %v = memref.load %m[%next, %j2] : memref<?x?xf64>
      memref.store %v, %o[%i2, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @neighbour_read
// CHECK:         ^bb{{[0-9]+}}(%i2: index):

// -----

// A conditional write does not prove that equal indices only meet in the same
// iteration.
func.func @conditional_write(%n: index, %c: i1, %m: memref<?x?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %x = arith.muli %i, %j : index
      scf.if %c {
        memref.store %one, %m[%x, %c0] : memref<?x?xf64>
      }
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      %x2 = arith.muli %i2, %j2 : index
      %v = memref.load %m[%x2, %c0] : memref<?x?xf64>
      memref.store %v, %o[%i2, %j2] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @conditional_write
// CHECK:         ^bb{{[0-9]+}}(%i2: index):

// -----

// Side effects outside the deepest fused level would be reordered.
func.func @outer_side_effect(%n: index, %m: memref<?xf64>, %o: memref<?x?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %one, %m[%i] : memref<?xf64>
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%k: index):
        memref.store %one, %o[%i, %j, %k] : memref<?x?x?xf64>
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j2: index):
      "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%k2: index):
        memref.store %one, %o[%i2, %j2, %k2] : memref<?x?x?xf64>
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @outer_side_effect
// CHECK:         ^bb{{[0-9]+}}(%i2: index):

// -----

func.func private @kernel(!llvm.ptr) -> ()

// Calls to functions not listed in cell_local_callees prevent fusion.
func.func @untrusted_call(%n: index, %m: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %base = "memref.extract_aligned_pointer_as_index"(%m) : (memref<?xf64>) -> index
    %off = arith.muli %i, %c8 : index
    %addr = arith.addi %base, %off : index
    %int = arith.index_cast %addr : index to i64
    %ptr = llvm.inttoptr %int : i64 to !llvm.ptr
    func.call @kernel(%ptr) : (!llvm.ptr) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %v = memref.load %m[%i2] : memref<?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @untrusted_call
// CHECK:         ^bb{{[0-9]+}}(%i2: index):
