// RUN: xdsl-opt -p "scf-parallel-loop-fusion{cell_local_callees=cell}" --split-input-file %s | filecheck %s

func.func private @cell(!llvm.ptr) -> ()

// @cell writes the 4 elements of the cell at its pointer; the next nest reads
// only that cell in the same iteration, so the nests are fused.
func.func @within_cell(%n: index, %m: memref<?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %base = "memref.extract_aligned_pointer_as_index"(%m) : (memref<?xf64>) -> index
    %cell = arith.muli %i, %c4 : index
    %off = arith.muli %cell, %c8 : index
    %addr = arith.addi %base, %off : index
    %int = arith.index_cast %addr : index to i64
    %ptr = llvm.inttoptr %int : i64 to !llvm.ptr
    func.call @cell(%ptr) : (!llvm.ptr) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %c4, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %cell2 = arith.muli %i2, %c4 : index
      %idx = arith.addi %cell2, %j : index
      %v = memref.load %m[%idx] : memref<?xf64>
      memref.store %v, %o[%i2, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @within_cell
// CHECK:         "scf.parallel"(%c0, %n, %c1)
// CHECK-NEXT:    ^bb{{[0-9]+}}(%i: index):
// CHECK:           func.call @cell(%ptr)
// CHECK-NEXT:      "scf.parallel"(%c0, %c4, %c1)
// CHECK:             %v = memref.load %m[%idx]
// CHECK:             scf.reduce
// CHECK-NEXT:      })
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    })
// CHECK-NEXT:    func.return

// -----

func.func private @cell(!llvm.ptr) -> ()

// The next nest reads 5 elements per cell of 4: one belongs to the next cell.
func.func @beyond_cell(%n: index, %m: memref<?xf64>, %o: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %base = "memref.extract_aligned_pointer_as_index"(%m) : (memref<?xf64>) -> index
    %cell = arith.muli %i, %c4 : index
    %off = arith.muli %cell, %c8 : index
    %addr = arith.addi %base, %off : index
    %int = arith.index_cast %addr : index to i64
    %ptr = llvm.inttoptr %int : i64 to !llvm.ptr
    func.call @cell(%ptr) : (!llvm.ptr) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i2: index):
    "scf.parallel"(%c0, %c5, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%j: index):
      %cell2 = arith.muli %i2, %c4 : index
      %idx = arith.addi %cell2, %j : index
      %v = memref.load %m[%idx] : memref<?xf64>
      memref.store %v, %o[%i2, %j] : memref<?x?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @beyond_cell
// CHECK:         ^bb{{[0-9]+}}(%i2: index):
