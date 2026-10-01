// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task depend=true single_region=true prove_pointer_accesses=true}" %s | filecheck %s

// Pointer accesses count as per-patch only if they provably stay within
// M[%i, ...]. In each function the first loop writes %M[%i, %j] and the second
// reads %M[%i, %j] through a pointer: aligned_pointer(M) + %i * 32 + %j * 8.

// All sizes are static, so the proof is static: %M gets per-patch tokens.
func.func @static_row(%M: memref<16x4xf64>, %T: memref<16x4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x4xf64>) -> index
    %row = arith.muli %i, %c32 : index
    %col = arith.muli %j, %c8 : index
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @static_row
// CHECK-NOT:         scf.if
// CHECK:             %[[M0:.*]] = "memref.extract_aligned_pointer_as_index"(%M)
// CHECK-NEXT:        scf.for %i = %c0 to %c16 step %c1 {
// CHECK-NEXT:          %[[MI:.*]] = arith.addi %[[M0]], %i : index
// CHECK-NEXT:          %[[MI64:.*]] = arith.index_cast %[[MI]] : index to i64
// CHECK-NEXT:          %[[MTOK:.*]] = llvm.inttoptr %[[MI64]] : i64 to !llvm.ptr
// CHECK-NEXT:          "omp.task"(%[[MTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NOT:         scf.if
// CHECK:             %[[M1:.*]] = "memref.extract_aligned_pointer_as_index"(%M)
// CHECK-NEXT:        %[[T1:.*]] = "memref.extract_aligned_pointer_as_index"(%T)
// CHECK-NEXT:        scf.for %[[I1:.*]] = %c0 to %c16 step %c1 {
// CHECK-NEXT:          %[[MI1:.*]] = arith.addi %[[M1]], %[[I1]] : index
// CHECK:               "omp.task"(%{{.*}}, %{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinout)>]
// CHECK:               llvm.load
// CHECK-NOT:         scf.if
// CHECK-LABEL:     func.func @runtime_row

// The access pattern of inlined physics code: dynamic sizes, and an inner
// offset computed in i32 from a runtime value %u (the number of unknowns):
//   aligned_pointer(M) + (%i * dim(M, 1) + %u * %j) * 8 + 8.
// %M gets per-patch tokens, completed by a runtime check that the offset
// stays within the row: the i32 arithmetic does not wrap
// (|%u|, |%m| <= 2**30), 0 <= %u * %j * 8 + 8, %u * (%m - 1) * 8 + 16 <=
// 8 * dim(M, 1), and %n <= dim(M, 0). If it fails, the loop's tasks are
// isolated from all others by a taskwait before and after creating them.
func.func @runtime_row(%M: memref<?x?xf64>, %T: memref<?x?xf64>, %n: index, %m: index, %u: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %n, %m, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %n, %m, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %j32 = arith.index_cast %j : index to i32
    %e = arith.muli %u, %j32 : i32
    %ei = arith.index_cast %e : i32 to index
    %d1 = memref.dim %M, %c1 : memref<?x?xf64>
    %r = arith.muli %i, %d1 : index
    %el = arith.addi %r, %ei : index
    %bytes = arith.muli %el, %c8 : index
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<?x?xf64>) -> index
    %addr = arith.addi %base, %bytes : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %q = llvm.getelementptr %p[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %x = llvm.load %q : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             %[[M0:.*]] = "memref.extract_aligned_pointer_as_index"(%M)
// CHECK-NEXT:        scf.for %i = %c0 to %n step %c1 {
// CHECK-NEXT:          %[[MI:.*]] = arith.addi %[[M0]], %i : index
// CHECK-NEXT:          %[[MI64:.*]] = arith.index_cast %[[MI]] : index to i64
// CHECK-NEXT:          %[[MTOK:.*]] = llvm.inttoptr %[[MI64]] : i64 to !llvm.ptr
// CHECK-NEXT:          "omp.task"(%[[MTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK:             %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[D1:.*]] = memref.dim %M, %[[ONE]] : memref<?x?xf64>
// CHECK-NEXT:        %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK-NEXT:        %[[ROW:.*]] = arith.muli %[[EIGHT]], %[[D1]] : index
// CHECK-NEXT:        %[[U:.*]] = arith.index_cast %u : i32 to index
// CHECK-NEXT:        %[[MINLEAF:.*]] = arith.constant -1073741824 : index
// CHECK-NEXT:        %[[UGE:.*]] = arith.cmpi sle, %[[MINLEAF]], %[[U]] : index
// CHECK-NEXT:        %[[MAXLEAF:.*]] = arith.constant 1073741824 : index
// CHECK-NEXT:        %[[ULE:.*]] = arith.cmpi sle, %[[U]], %[[MAXLEAF]] : index
// CHECK-NEXT:        %[[MGE:.*]] = arith.cmpi sle, %[[MINLEAF]], %m : index
// CHECK-NEXT:        %[[MLE:.*]] = arith.cmpi sle, %m, %[[MAXLEAF]] : index
// CHECK-NEXT:        %[[JMAX:.*]] = arith.subi %m, %[[ONE]] : index
// CHECK-NEXT:        %[[UJ:.*]] = arith.muli %[[U]], %[[JMAX]] : index
// CHECK-NEXT:        %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[ELO:.*]] = arith.minsi %[[ZERO]], %[[UJ]] : index
// CHECK-NEXT:        %[[EHI:.*]] = arith.maxsi %[[ZERO]], %[[UJ]] : index
// CHECK-NEXT:        %[[I32MIN:.*]] = arith.constant -2147483648 : index
// CHECK-NEXT:        %[[ELOOK:.*]] = arith.cmpi sle, %[[I32MIN]], %[[ELO]] : index
// CHECK-NEXT:        %[[I32MAX:.*]] = arith.constant 2147483647 : index
// CHECK-NEXT:        %[[EHIOK:.*]] = arith.cmpi sle, %[[EHI]], %[[I32MAX]] : index
// CHECK-NEXT:        %[[BLO0:.*]] = arith.muli %[[ELO]], %[[EIGHT]] : index
// CHECK-NEXT:        %[[BHI0:.*]] = arith.muli %[[EHI]], %[[EIGHT]] : index
// CHECK-NEXT:        %[[BLO:.*]] = arith.addi %[[BLO0]], %[[EIGHT]] : index
// CHECK-NEXT:        %[[BHI:.*]] = arith.addi %[[BHI0]], %[[EIGHT]] : index
// CHECK-NEXT:        %[[END:.*]] = arith.addi %[[BHI]], %[[EIGHT]] : index
// CHECK-NEXT:        %[[LOOK:.*]] = arith.cmpi sle, %[[ZERO]], %[[BLO]] : index
// CHECK-NEXT:        %[[HIOK:.*]] = arith.cmpi sle, %[[END]], %[[ROW]] : index
// CHECK-NEXT:        %[[D0:.*]] = memref.dim %M, %[[ZERO]] : memref<?x?xf64>
// CHECK-NEXT:        %[[NOK:.*]] = arith.cmpi sle, %n, %[[D0]] : index
// CHECK-NEXT:        %[[C1:.*]] = arith.andi %[[UGE]], %[[ULE]] : i1
// CHECK-NEXT:        %[[C2:.*]] = arith.andi %[[C1]], %[[MGE]] : i1
// CHECK-NEXT:        %[[C3:.*]] = arith.andi %[[C2]], %[[MLE]] : i1
// CHECK-NEXT:        %[[C4:.*]] = arith.andi %[[C3]], %[[ELOOK]] : i1
// CHECK-NEXT:        %[[C5:.*]] = arith.andi %[[C4]], %[[EHIOK]] : i1
// CHECK-NEXT:        %[[C6:.*]] = arith.andi %[[C5]], %[[LOOK]] : i1
// CHECK-NEXT:        %[[C7:.*]] = arith.andi %[[C6]], %[[HIOK]] : i1
// CHECK-NEXT:        %[[CHECK:.*]] = arith.andi %[[C7]], %[[NOK]] : i1
// CHECK-NEXT:        %[[M1:.*]] = "memref.extract_aligned_pointer_as_index"(%M)
// CHECK-NEXT:        %[[T1:.*]] = "memref.extract_aligned_pointer_as_index"(%T)
// CHECK-NEXT:        scf.if %[[CHECK]] {
// CHECK-NEXT:        } else {
// CHECK-NEXT:          "omp.taskwait"() : () -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.for %[[I1:.*]] = %c0 to %n step %c1 {
// CHECK-NEXT:          %[[MI1:.*]] = arith.addi %[[M1]], %[[I1]] : index
// CHECK-NEXT:          %[[MI164:.*]] = arith.index_cast %[[MI1]] : index to i64
// CHECK-NEXT:          %[[MTOK1:.*]] = llvm.inttoptr %[[MI164]] : i64 to !llvm.ptr
// CHECK:               "omp.task"(%[[MTOK1]], %{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinout)>]
// CHECK:                 %x = llvm.load %q : !llvm.ptr -> f64
// CHECK:             scf.if %[[CHECK]] {
// CHECK-NEXT:        } else {
// CHECK-NEXT:          "omp.taskwait"() : () -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        "omp.terminator"() : () -> ()

// Near misses, which must keep whole-buffer dependences on %M: the writing
// loop gets `inoutset` on aligned_pointer(M), after a separator task.

// The offset is in elements, not bytes: %i * 4 + %j.
func.func @elements_not_bytes(%M: memref<16x4xf64>, %T: memref<16x4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x4xf64>) -> index
    %row = arith.muli %i, %c4 : index
    %off = arith.addi %row, %j : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @elements_not_bytes
// CHECK-NOT:         scf.if
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK:             "omp.task"(%{{.*}}, %{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinout)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @short_row

// The row stride is 24 bytes, but a row of %M is 32 bytes.
func.func @short_row(%M: memref<16x4xf64>, %T: memref<16x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c3, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x4xf64>) -> index
    %row = arith.muli %i, %c24 : index
    %col = arith.muli %j, %c8 : index
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x3xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @past_row_end

// %j goes up to 4, so the last access, at byte 32, is in the next row.
func.func @past_row_end(%M: memref<16x4xf64>, %T: memref<16x5xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c5, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x4xf64>) -> index
    %row = arith.muli %i, %c32 : index
    %col = arith.muli %j, %c8 : index
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x5xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @wrong_induction_variable

// The row is selected by the inner induction variable %j.
func.func @wrong_induction_variable(%M: memref<16x16xf64>, %T: memref<16x16xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c16, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x16xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c16, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x16xf64>) -> index
    %row = arith.muli %j, %c128 : index
    %col = arith.muli %i, %c8 : index
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x16xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @strided_layout

// %M is a column of a larger buffer: its rows are 8 bytes apart but 16 bytes
// long in memory (so the ones of different patches interleave).
func.func @strided_layout(%M: memref<16x2xf64, strided<[1, 16]>>, %T: memref<16x2xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c2, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x2xf64, strided<[1, 16]>>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c2, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x2xf64, strided<[1, 16]>>) -> index
    %row = arith.muli %i, %c16 : index
    %col = arith.muli %j, %c8 : index
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x2xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @opaque_offset

// The inner offset is loaded from memory inside the loop, so it cannot be
// bounded.
func.func @opaque_offset(%M: memref<16x4xf64>, %T: memref<16x4xf64>, %O: memref<16x4xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %c16, %c4, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<16x4xf64>) -> index
    %row = arith.muli %i, %c32 : index
    %col = memref.load %O[%i, %j] : memref<16x4xindex>
    %off = arith.addi %row, %col : index
    %addr = arith.addi %base, %off : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<16x4xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
// CHECK-LABEL: func.func @alloc_size_row

// The row stride is the size an allocation was made with, as for the
// maxEigenvalues buffers of the kernels: per-patch tokens (the check only
// bounds the inner offset).
func.func @alloc_size_row(%n: index, %s: index, %T: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.000000e+00 : f64
  %M = memref.alloc(%n, %s) : memref<?x?xf64>
  "scf.parallel"(%c0, %c0, %n, %s, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %n, %s, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<?x?xf64>) -> index
    %row = arith.muli %i, %s : index
    %rowb = arith.muli %row, %c8 : index
    %col = arith.muli %j, %c8 : index
    %p0 = arith.addi %base, %rowb : index
    %addr = arith.addi %p0, %col : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  memref.dealloc %M : memref<?x?xf64>
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          scf.for
// CHECK:             scf.if %{{.*}} {
// CHECK-NEXT:        } else {
// CHECK-NEXT:          "omp.taskwait"() : () -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.for
// CHECK:               "omp.task"(%{{.*}}, %{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinout)>]
// CHECK-LABEL: func.func @parameter_row

// The row stride is a parameter, which may differ from dim(M, 1).
func.func @parameter_row(%M: memref<?x?xf64>, %T: memref<?x?xf64>, %n: index, %s: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %c0, %n, %s, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    memref.store %zero, %M[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %n, %s, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %base = "memref.extract_aligned_pointer_as_index"(%M) : (memref<?x?xf64>) -> index
    %row = arith.muli %i, %s : index
    %el = arith.addi %row, %j : index
    %bytes = arith.muli %el, %c8 : index
    %addr = arith.addi %base, %bytes : index
    %a64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %a64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %T[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK-NEXT:          "omp.terminator"
// CHECK:             "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NOT:         scf.if
