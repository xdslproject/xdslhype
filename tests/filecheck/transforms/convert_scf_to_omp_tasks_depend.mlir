// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task depend=true single_region=true}" %s | filecheck %s

// %A is only accessed as %A[%i, ...] by loops over [0, n) with step 1, so its
// tasks depend on individual patches. %B is also accessed through a pointer,
// so its tasks depend on the whole buffer. %C is never written by a task, so no
// dependences are needed for it. The dealloc must wait for the tasks.
func.func @flow(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %n: index, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %c0, %n, %m, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %C[%i, %j] : memref<?x?xf64>
    memref.store %v, %A[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  %tmp = memref.alloc(%n) : memref<?xf64>
  %c = memref.load %C[%c0, %c0] : memref<?x?xf64>
  "scf.parallel"(%c0, %c0, %n, %m, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %A[%i, %j] : memref<?x?xf64>
    %w = arith.addf %v, %c : f64
    memref.store %w, %B[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %base = "memref.extract_aligned_pointer_as_index"(%B) : (memref<?x?xf64>) -> index
    %row = arith.muli %i, %m : index
    %addr = arith.addi %base, %row : index
    %addr64 = arith.index_cast %addr : index to i64
    %p = llvm.inttoptr %addr64 : i64 to !llvm.ptr
    %x = llvm.load %p : !llvm.ptr -> f64
    memref.store %x, %tmp[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  memref.dealloc %tmp : memref<?xf64>
  func.return
}

// CHECK-LABEL: func.func @flow
// CHECK:           "omp.single"
// CHECK-NEXT:        %[[A0:.*]] = "memref.extract_aligned_pointer_as_index"(%A)
// CHECK-NEXT:        scf.for %i = %c0 to %n step %c1 {
// CHECK-NEXT:          %[[AI:.*]] = arith.addi %[[A0]], %i : index
// CHECK-NEXT:          %[[AI64:.*]] = arith.index_cast %[[AI]] : index to i64
// CHECK-NEXT:          %[[ATOK:.*]] = llvm.inttoptr %[[AI64]] : i64 to !llvm.ptr
// CHECK-NEXT:          "omp.task"(%[[ATOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:            scf.for %j = %c0 to %m step %c1 {
// CHECK-NEXT:              %v = memref.load %C[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:              memref.store %v, %A[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:            }
// CHECK-NEXT:            "omp.terminator"() : () -> ()
// CHECK-NEXT:          }) : (!llvm.ptr) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        %tmp = memref.alloc(%n) : memref<?xf64>
// CHECK-NEXT:        %c = memref.load %C[%c0, %c0] : memref<?x?xf64>
// CHECK-NEXT:        %[[A1:.*]] = "memref.extract_aligned_pointer_as_index"(%A)
// CHECK-NEXT:        %[[B1:.*]] = "memref.extract_aligned_pointer_as_index"(%B)
// CHECK-NEXT:        %[[B164:.*]] = arith.index_cast %[[B1]] : index to i64
// CHECK-NEXT:        %[[BTOK:.*]] = llvm.inttoptr %[[B164]] : i64 to !llvm.ptr
// CHECK-NEXT:        "omp.task"(%[[BTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (!llvm.ptr) -> ()
// CHECK-NEXT:        scf.for %[[I1:.*]] = %c0 to %n step %c1 {
// CHECK-NEXT:          %[[AI1:.*]] = arith.addi %[[A1]], %[[I1]] : index
// CHECK-NEXT:          %[[AI164:.*]] = arith.index_cast %[[AI1]] : index to i64
// CHECK-NEXT:          %[[ATOK1:.*]] = llvm.inttoptr %[[AI164]] : i64 to !llvm.ptr
// CHECK-NEXT:          "omp.task"(%[[ATOK1]], %[[BTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinoutset)>], operandSegmentSizes = array<i32: 0, 0, 2, 0, 0, 0, 0, 0, 0>}> ({
// CHECK:                 memref.store %{{.*}}, %B[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK:             "omp.terminator"() : () -> ()
// CHECK-NEXT:          }) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[B2:.*]] = "memref.extract_aligned_pointer_as_index"(%B)
// CHECK-NEXT:        %[[B264:.*]] = arith.index_cast %[[B2]] : index to i64
// CHECK-NEXT:        %[[BTOK2:.*]] = llvm.inttoptr %[[B264]] : i64 to !llvm.ptr
// CHECK-NEXT:        %[[T0:.*]] = "memref.extract_aligned_pointer_as_index"(%tmp)
// CHECK-NEXT:        scf.for %[[I2:.*]] = %c0 to %n step %c1 {
// CHECK-NEXT:          %[[TI:.*]] = arith.addi %[[T0]], %[[I2]] : index
// CHECK-NEXT:          %[[TI64:.*]] = arith.index_cast %[[TI]] : index to i64
// CHECK-NEXT:          %[[TTOK:.*]] = llvm.inttoptr %[[TI64]] : i64 to !llvm.ptr
// CHECK-NEXT:          "omp.task"(%[[BTOK2]], %[[TTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 2, 0, 0, 0, 0, 0, 0>}> ({
// CHECK:                 memref.store %x, %tmp[%[[I2]]] : memref<?xf64>
// CHECK-NEXT:            "omp.terminator"() : () -> ()
// CHECK-NEXT:          }) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        "omp.taskwait"() : () -> ()
// CHECK-NEXT:        memref.dealloc %tmp : memref<?xf64>
// CHECK-NEXT:        "omp.terminator"() : () -> ()

// The second loop stores through a pointer that is not derived from a memref,
// so it cannot be given dependences: all earlier tasks are waited for first,
// and its own tasks are waited for by a taskgroup.
func.func @opaque(%A: memref<?xf64>, %ptr: !llvm.ptr, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %A[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    llvm.store %zero, %ptr : f64, !llvm.ptr
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @opaque
// CHECK:           "omp.single"
// CHECK:               "omp.task"(%{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>]
// CHECK:           "omp.taskwait"() : () -> ()
// CHECK-NEXT:      "omp.taskgroup"
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %n step %c1 {
// CHECK-NEXT:          "omp.task"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:            llvm.store %zero, %ptr : f64, !llvm.ptr
