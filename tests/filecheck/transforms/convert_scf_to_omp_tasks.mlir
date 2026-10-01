// RUN: xdsl-opt -p convert-scf-to-omp-tasks %s | filecheck %s
// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{grainsize=4}" %s | filecheck %s --check-prefix=GRAIN
// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{num_tasks=8}" %s | filecheck %s --check-prefix=NTASKS

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

// CHECK-LABEL: func.func @two_dims
// CHECK:         %c1 = arith.constant 1 : index
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[NUB:.*]] = arith.subi %n, %[[ONE]] : index
// CHECK-NEXT:    %[[MUB:.*]] = arith.subi %m, %[[ONE]] : index
// CHECK-NEXT:    "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      "omp.single"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.taskloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%c0, %c0, %[[NUB]], %[[MUB]], %c1, %c1) <{loop_inclusive}> ({
// CHECK-NEXT:          ^bb0(%i: index, %j: index):
// CHECK-NEXT:            %v = memref.load %A[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:            %w = arith.addf %v, %v : f64
// CHECK-NEXT:            memref.store %w, %A[%i, %j] : memref<?x?xf64>
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    func.return

// GRAIN-LABEL: func.func @two_dims
// GRAIN:         %[[G:.*]] = arith.constant 4 : i64
// GRAIN-NEXT:    "omp.parallel"
// GRAIN-NEXT:      "omp.single"
// GRAIN-NEXT:        "omp.taskloop"(%[[G]]) <{operandSegmentSizes = array<i32: 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>}>

// NTASKS-LABEL: func.func @two_dims
// NTASKS:         %[[N:.*]] = arith.constant 8 : i64
// NTASKS-NEXT:    "omp.parallel"
// NTASKS-NEXT:      "omp.single"
// NTASKS-NEXT:        "omp.taskloop"(%[[N]]) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>}>

// Only the outermost scf.parallel becomes a taskloop.
func.func @nested(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%j: index):
      "test.op"(%i, %j) : (index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return
}

// CHECK-LABEL: func.func @nested
// CHECK:             "omp.taskloop"
// CHECK-NEXT:          "omp.loop_nest"(%c0, %{{.*}}, %c1) <{loop_inclusive}> ({
// CHECK-NEXT:          ^bb0(%i: index):
// CHECK-NEXT:            "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:            ^bb0(%j: index):
// CHECK-NEXT:              "test.op"(%i, %j) : (index, index) -> ()
// CHECK-NEXT:              scf.reduce
// CHECK-NEXT:            }) : (index, index, index) -> ()
// CHECK-NEXT:            omp.yield

// Loops inside other regions are converted in place.
func.func @inside_if(%cond: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.if %cond {
    "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%i: index):
      "test.op"(%i) : (index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @inside_if
// CHECK:         scf.if %cond {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:      %{{.*}} = arith.subi %n, %{{.*}} : index
// CHECK-NEXT:      "omp.parallel"
// CHECK-NEXT:        "omp.single"
// CHECK-NEXT:          "omp.taskloop"

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
// CHECK:           "omp.loop_nest"(%c0, %{{.*}}, %c1) <{loop_inclusive}> ({
// CHECK-NEXT:      ^bb0(%i: index):
// CHECK-NEXT:        "memref.alloca_scope"() ({
// CHECK-NEXT:          %a = memref.alloca() : memref<4xf64>
// CHECK-NEXT:          "test.op"(%a) : (memref<4xf64>) -> ()
// CHECK-NEXT:          "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        omp.yield

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
