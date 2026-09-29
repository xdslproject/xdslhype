// RUN: xdsl-opt -p "convert-scf-to-omp-tasks{single_region=true}" %s | filecheck %s

// One parallel region spans from the first to the last loop; code before and
// after it stays outside, code in between runs on the single thread.
func.func @span(%cond: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "test.op"() {before} : () -> ()
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "test.op"(%i) {first} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  %x = "test.op"() {between} : () -> index
  scf.if %cond {
    "scf.parallel"(%c0, %x, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%j: index):
      "test.op"(%j) {second} : (index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
  }
  "test.op"() {after} : () -> ()
  func.return
}

// CHECK-LABEL: func.func @span
// CHECK:         "test.op"() {before} : () -> ()
// CHECK-NEXT:    "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      "omp.single"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        %[[NUB:.*]] = arith.subi %n, %{{.*}} : index
// CHECK-NEXT:        "omp.taskloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%c0, %[[NUB]], %c1) <{loop_inclusive}> ({
// CHECK-NEXT:          ^bb0(%i: index):
// CHECK-NEXT:            "test.op"(%i) {first} : (index) -> ()
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        %x = "test.op"() {between} : () -> index
// CHECK-NEXT:        scf.if %cond {
// CHECK-NEXT:          %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:          %[[XUB:.*]] = arith.subi %x, %{{.*}} : index
// CHECK-NEXT:          "omp.taskloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:            "omp.loop_nest"(%c0, %[[XUB]], %c1) <{loop_inclusive}> ({
// CHECK-NEXT:            ^bb0(%j: index):
// CHECK-NEXT:              "test.op"(%j) {second} : (index) -> ()
// CHECK-NEXT:              omp.yield
// CHECK-NEXT:            }) : (index, index, index) -> ()
// CHECK-NEXT:          }) : () -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "test.op"() {after} : () -> ()
// CHECK-NEXT:    func.return

// The span is extended over later ops that use values defined in it, such as
// the dealloc of a buffer allocated between the loops.
func.func @extend(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "test.op"(%i) {first} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  %buf = memref.alloc(%n) : memref<?xf64>
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%j: index):
    "test.op"(%j, %buf) {second} : (index, memref<?xf64>) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  "test.op"() {unrelated} : () -> ()
  memref.dealloc %buf : memref<?xf64>
  "test.op"() {after} : () -> ()
  func.return
}

// CHECK-LABEL: func.func @extend
// CHECK:         "omp.parallel"
// CHECK-NEXT:      "omp.single"
// CHECK:             "test.op"(%i) {first} : (index) -> ()
// CHECK:           %buf = memref.alloc(%n) : memref<?xf64>
// CHECK:             "test.op"(%j, %buf) {second} : (index, memref<?xf64>) -> ()
// CHECK:           "test.op"() {unrelated} : () -> ()
// CHECK-NEXT:      memref.dealloc %buf : memref<?xf64>
// CHECK-NEXT:      "omp.terminator"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "omp.terminator"() : () -> ()
// CHECK-NEXT:  }) : () -> ()
// CHECK-NEXT:  "test.op"() {after} : () -> ()
// CHECK-NEXT:  func.return
// CHECK-NOT:   "omp.parallel"

// A value defined inside the span is used after it, so each loop gets its own
// parallel region instead.
func.func @escape(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%i: index):
    "test.op"(%i) {first} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  %x = "test.op"() {between} : () -> index
  "scf.parallel"(%c0, %n, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%j: index):
    "test.op"(%j) {second} : (index) -> ()
    scf.reduce
  }) : (index, index, index) -> ()
  func.return %x : index
}

// CHECK-LABEL: func.func @escape
// CHECK:         "omp.parallel"
// CHECK-NEXT:      "omp.single"
// CHECK-NEXT:        "omp.taskloop"
// CHECK:               "test.op"(%i) {first} : (index) -> ()
// CHECK:         %x = "test.op"() {between} : () -> index
// CHECK:         "omp.parallel"
// CHECK-NEXT:      "omp.single"
// CHECK-NEXT:        "omp.taskloop"
// CHECK:               "test.op"(%j) {second} : (index) -> ()
// CHECK:         func.return %x : index

// Functions without parallel loops are left alone.
func.func @no_loops() {
  "test.op"() : () -> ()
  func.return
}

// CHECK-LABEL: func.func @no_loops
// CHECK-NEXT:    "test.op"() : () -> ()
// CHECK-NEXT:    func.return
