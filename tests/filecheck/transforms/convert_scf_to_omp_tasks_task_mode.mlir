// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task}" %s | filecheck %s
// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task chunk=4}" %s | filecheck %s --check-prefix=CHUNK
// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{mode=task single_region=true}" %s | filecheck %s --check-prefix=SINGLE

func.func @two_dims(%A: memref<?x?xf64>, %n: index, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %c0, %n, %m, %c1, %c1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%i: index, %j: index):
    %v = memref.load %A[%i, %j] : memref<?x?xf64>
    %w = arith.addf %v, %v : f64
    memref.store %w, %A[%i, %j] : memref<?x?xf64>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// One task per iteration of the outermost dimension, the rest runs
// sequentially inside the task.

// CHECK-LABEL: func.func @two_dims
// CHECK:         "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      "omp.single"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.taskgroup"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:          scf.for %i = %c0 to %n step %c1 {
// CHECK-NEXT:            "omp.task"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:              scf.for %j = %c0 to %m step %c1 {
// CHECK-NEXT:                %v = memref.load %A[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:                %w = arith.addf %v, %v : f64
// CHECK-NEXT:                memref.store %w, %A[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:              }
// CHECK-NEXT:              "omp.terminator"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    func.return

// Each task covers `chunk` iterations of the outermost dimension, clamped to
// the upper bound.

// CHUNK-LABEL: func.func @two_dims
// CHUNK:         %[[CHUNK:.*]] = arith.constant 4 : index
// CHUNK-NEXT:    %[[STEP:.*]] = arith.muli %c1, %[[CHUNK]] : index
// CHUNK-NEXT:    "omp.parallel"
// CHUNK-NEXT:      "omp.single"
// CHUNK-NEXT:        "omp.taskgroup"
// CHUNK-NEXT:          scf.for %[[START:.*]] = %c0 to %n step %[[STEP]] {
// CHUNK-NEXT:            "omp.task"
// CHUNK-NEXT:              %[[END:.*]] = arith.addi %[[START]], %[[STEP]] : index
// CHUNK-NEXT:              %[[CLAMPED:.*]] = arith.minsi %[[END]], %n : index
// CHUNK-NEXT:              scf.for %i = %[[START]] to %[[CLAMPED]] step %c1 {
// CHUNK-NEXT:                scf.for %j = %c0 to %m step %c1 {
// CHUNK-NEXT:                  %v = memref.load %A[%i, %j] : memref<?x?xf64>
// CHUNK-NEXT:                  %w = arith.addf %v, %v : f64
// CHUNK-NEXT:                  memref.store %w, %A[%i, %j] : memref<?x?xf64>
// CHUNK-NEXT:                }
// CHUNK-NEXT:              }
// CHUNK-NEXT:              "omp.terminator"() : () -> ()

func.func @three_dims(%A: memref<?x?x?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  "scf.parallel"(%c0, %c0, %c0, %n, %n, %n, %c1, %c2, %c1) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
  ^bb0(%i: index, %j: index, %k: index):
    "test.op"(%i, %j, %k) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index, index, index, index, index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @three_dims
// CHECK:             scf.for %i = %c0 to %n step %c1 {
// CHECK-NEXT:          "omp.task"
// CHECK-NEXT:            scf.for %j = %c0 to %n step %c2 {
// CHECK-NEXT:              scf.for %k = %c0 to %n step %c1 {
// CHECK-NEXT:                "test.op"(%i, %j, %k) : (index, index, index) -> ()
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            "omp.terminator"() : () -> ()

// Stack allocations in the body are scoped to a single iteration.
func.func @alloca(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    %a = memref.alloca() : memref<4xf64>
    "test.op"(%a) : (memref<4xf64>) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @alloca
// CHECK:             scf.for %i = %c0 to %n step %c1 {
// CHECK-NEXT:          "omp.task"
// CHECK-NEXT:            "memref.alloca_scope"() ({
// CHECK-NEXT:              %a = memref.alloca() : memref<4xf64>
// CHECK-NEXT:              "test.op"(%a) : (memref<4xf64>) -> ()
// CHECK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            "omp.terminator"() : () -> ()

// With single_region, the loops of a function share one parallel region and
// each becomes just a taskgroup.
func.func @two_loops(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "test.op"(%i) {first} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%j: index):
    "test.op"(%j) {second} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// SINGLE-LABEL: func.func @two_loops
// SINGLE:         "omp.parallel"
// SINGLE-NEXT:      "omp.single"
// SINGLE-NEXT:        "omp.taskgroup"
// SINGLE-NEXT:          scf.for %i = %c0 to %n step %c1 {
// SINGLE-NEXT:            "omp.task"
// SINGLE-NEXT:              "test.op"(%i) {first} : (index) -> ()
// SINGLE-NEXT:              "omp.terminator"() : () -> ()
// SINGLE-NEXT:            }) : () -> ()
// SINGLE-NEXT:          }
// SINGLE-NEXT:          "omp.terminator"() : () -> ()
// SINGLE-NEXT:        }) : () -> ()
// SINGLE-NEXT:        "omp.taskgroup"
// SINGLE-NEXT:          scf.for %j = %c0 to %n step %c1 {
// SINGLE-NEXT:            "omp.task"
// SINGLE-NEXT:              "test.op"(%j) {second} : (index) -> ()
// SINGLE-NEXT:              "omp.terminator"() : () -> ()
// SINGLE-NEXT:            }) : () -> ()
// SINGLE-NEXT:          }
// SINGLE-NEXT:          "omp.terminator"() : () -> ()
// SINGLE-NEXT:        }) : () -> ()
// SINGLE-NEXT:        "omp.terminator"() : () -> ()
// SINGLE-NEXT:      }) : () -> ()
// SINGLE-NEXT:      "omp.terminator"() : () -> ()
// SINGLE-NEXT:    }) : () -> ()
// SINGLE-NEXT:    func.return

// Reductions are not supported and are left untouched.
func.func @reduction(%n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f64
  %r = "scf.parallel"(%c0, %n, %c1, %zero) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
  ^bb0(%i: index):
    scf.reduce(%zero : f64) {
    ^bb0(%lhs: f64, %rhs: f64):
      %s = arith.addf %lhs, %rhs : f64
      scf.reduce.return %s : f64
    }
  }) : (index, index, index, f64) -> f64
  func.return %r : f64
}

// CHECK-LABEL: func.func @reduction
// CHECK-NOT:     omp.
// CHECK:         "scf.parallel"
