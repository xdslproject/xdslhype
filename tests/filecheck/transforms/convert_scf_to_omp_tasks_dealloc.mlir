// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task depend=true single_region=true chunk=4 defer_deallocs=true}" %s | filecheck %s

// %W is written as %W[%j, %i], so its tasks depend on the whole buffer. Its
// dealloc becomes a task with depend(inout) on the whole-buffer token, ordered
// after the writing (inoutset) and reading (in) tasks.
func.func @whole(%A: memref<?x?xf64>, %B: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %B[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  %W = memref.alloc(%n, %n) : memref<?x?xf64>
  "scf.parallel"(%c0, %c0, %n, %n, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %A[%i, %j] : memref<?x?xf64>
    memref.store %v, %W[%j, %i] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %c0, %n, %n, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %W[%j, %i] : memref<?x?xf64>
    memref.store %v, %A[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  memref.dealloc %W : memref<?x?xf64>
  func.return
}

// CHECK-LABEL: func.func @whole
// CHECK:         %W = memref.alloc(%n, %n) : memref<?x?xf64>
// CHECK:             memref.store %{{.*}}, %A[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK:           }) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[W:.*]] = "memref.extract_aligned_pointer_as_index"(%W) : (memref<?x?xf64>) -> index
// CHECK-NEXT:    %[[W64:.*]] = arith.index_cast %[[W]] : index to i64
// CHECK-NEXT:    %[[WTOK:.*]] = llvm.inttoptr %[[W64]] : i64 to !llvm.ptr
// CHECK-NEXT:    "omp.task"(%[[WTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      memref.dealloc %W : memref<?x?xf64>
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : (!llvm.ptr) -> ()
// CHECK-NEXT:    "omp.terminator"() : () -> ()
// CHECK-NOT:     "omp.taskwait"
// CHECK-LABEL: func.func @per_patch

// %P and %Q are only accessed as %P[%i, ...] / %Q[%i, ...], so their tasks
// depend on patches, in loops with different upper bounds. One empty task per
// chunk start below the larger bound is ordered after every task using its
// patches (inout), and joins an inoutset group on a token past the patch
// tokens of %Q; the dealloc task (inout on that token) runs after all of them.
// %Q is only read by tasks, but as it is freed, its readers get `in`
// dependences too.
func.func @per_patch(%A: memref<?x?xf64>, %B: memref<?xf64>, %n: index, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  %Q = memref.alloc(%m) : memref<?xf64>
  memref.store %zero, %Q[%c0] : memref<?xf64>
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %B[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  %P = memref.alloc(%n, %n) : memref<?x?xf64>
  "scf.parallel"(%c0, %c0, %n, %n, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %A[%i, %j] : memref<?x?xf64>
    memref.store %v, %P[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%c0, %m, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %v = memref.load %P[%i, %c0] : memref<?x?xf64>
    %q = memref.load %Q[%i] : memref<?xf64>
    %w = arith.addf %v, %q : f64
    memref.store %w, %A[%c0, %i] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  memref.dealloc %Q : memref<?xf64>
  memref.dealloc %P : memref<?x?xf64>
  func.return
}

// CHECK:         %P = memref.alloc(%n, %n) : memref<?x?xf64>
// CHECK:         scf.for %{{.*}} = %c0 to %m step %{{.*}} {
// CHECK:           "omp.task"(%{{.*}}, %{{.*}}, %{{.*}}) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependinoutset)>]
// CHECK-NEXT:        %{{.*}} = arith.addi
// CHECK-NEXT:        %{{.*}} = arith.minsi
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          memref.load %P
// CHECK-NEXT:          memref.load %Q
// CHECK:           }) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[UB:.*]] = arith.maxsi %m, %n : index
// CHECK-NEXT:    %[[Q:.*]] = "memref.extract_aligned_pointer_as_index"(%Q) : (memref<?xf64>) -> index
// CHECK-NEXT:    %[[P:.*]] = "memref.extract_aligned_pointer_as_index"(%P) : (memref<?x?xf64>) -> index
// CHECK-NEXT:    %[[X:.*]] = arith.addi %[[Q]], %[[UB]] : index
// CHECK-NEXT:    %[[X64:.*]] = arith.index_cast %[[X]] : index to i64
// CHECK-NEXT:    %[[XTOK:.*]] = llvm.inttoptr %[[X64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-NEXT:    scf.for %[[PS:.*]] = %[[ZERO]] to %[[UB]] step %[[FOUR]] {
// CHECK-NEXT:      %[[QP:.*]] = arith.addi %[[Q]], %[[PS]] : index
// CHECK-NEXT:      %[[QP64:.*]] = arith.index_cast %[[QP]] : index to i64
// CHECK-NEXT:      %[[QTOK:.*]] = llvm.inttoptr %[[QP64]] : i64 to !llvm.ptr
// CHECK-NEXT:      %[[PP:.*]] = arith.addi %[[P]], %[[PS]] : index
// CHECK-NEXT:      %[[PP64:.*]] = arith.index_cast %[[PP]] : index to i64
// CHECK-NEXT:      %[[PTOK:.*]] = llvm.inttoptr %[[PP64]] : i64 to !llvm.ptr
// CHECK-NEXT:      "omp.task"(%[[QTOK]], %[[PTOK]], %[[XTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>, #omp<clause_task_depend (taskdependinout)>, #omp<clause_task_depend (taskdependinoutset)>], operandSegmentSizes = array<i32: 0, 0, 3, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    "omp.task"(%[[XTOK]]) <{depend_kinds = [#omp<clause_task_depend (taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      memref.dealloc %Q : memref<?xf64>
// CHECK-NEXT:      memref.dealloc %P : memref<?x?xf64>
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : (!llvm.ptr) -> ()
// CHECK-NEXT:    "omp.terminator"() : () -> ()
// CHECK-NOT:     "omp.taskwait"
// CHECK-LABEL: func.func @untouched

// %U is not used by any task: its dealloc needs no wait.
func.func @untouched(%A: memref<?xf64>, %B: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %B[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  %U = memref.alloc(%n) : memref<?xf64>
  %u = memref.load %U[%c0] : memref<?xf64>
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %A[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  memref.dealloc %U : memref<?xf64>
  func.return
}

// CHECK:         %U = memref.alloc(%n) : memref<?xf64>
// CHECK:             memref.store %zero, %A
// CHECK:           }) : (!llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %U : memref<?xf64>
// CHECK-NEXT:    "omp.terminator"() : () -> ()
// CHECK-LABEL: func.func @full_waits

// A full wait is still needed before a store by the task-creating thread to a
// buffer used by tasks, before the dealloc of a buffer that tasks access
// through an alias (%V through %Vc), and before the dealloc of a buffer used by
// a loop whose upper bound is not available at the dealloc.
func.func @full_waits(%A: memref<?xf64>, %n: index, %flag: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %A[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  memref.store %zero, %A[%c0] : memref<?xf64>
  %V = memref.alloc(%n) : memref<?xf64>
  %Vc = "memref.cast"(%V) : (memref<?xf64>) -> memref<?xf64>
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    memref.store %zero, %Vc[%i] : memref<?xf64>
    scf.reduce
  }) : (index, index, index) -> ()
  memref.dealloc %V : memref<?xf64>
  %T = memref.alloc(%n) : memref<?xf64>
  scf.if %flag {
    %k = arith.addi %n, %c1 : index
    "scf.parallel"(%c0, %k, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%i: index):
      memref.store %zero, %T[%i] : memref<?xf64>
      scf.reduce
    }) : (index, index, index) -> ()
  }
  memref.dealloc %T : memref<?xf64>
  func.return
}

// CHECK:         "omp.single"
// CHECK:           }) : (!llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    "omp.taskwait"() : () -> ()
// CHECK-NEXT:    memref.store %zero, %A[%c0] : memref<?xf64>
// CHECK:           }) : (!llvm.ptr) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    "omp.taskwait"() : () -> ()
// CHECK-NEXT:    memref.dealloc %V : memref<?xf64>
// CHECK:         scf.if %flag {
// CHECK:               memref.store %zero, %T
// CHECK:             }) : (!llvm.ptr) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    "omp.taskwait"() : () -> ()
// CHECK-NEXT:    memref.dealloc %T : memref<?xf64>
// CHECK-NEXT:    "omp.terminator"() : () -> ()
