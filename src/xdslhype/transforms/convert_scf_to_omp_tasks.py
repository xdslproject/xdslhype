from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, memref, omp, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException


def _is_nested_in_parallel_construct(op: Operation) -> bool:
    """True if op sits inside another scf.parallel or an OpenMP task/loop construct."""
    parent = op.parent_op()
    while parent is not None:
        if isinstance(
            parent,
            scf.ParallelOp | omp.TaskOp | omp.TaskloopOp | omp.LoopNestOp,
        ):
            return True
        parent = parent.parent_op()
    return False


def _contains_alloca(block: Block) -> bool:
    return any(
        isinstance(op, memref.AllocaOp | llvm.AllocaOp)
        for inner in block.ops
        for op in inner.walk()
    )


def _empty_parallel_region() -> tuple[omp.ParallelOp, omp.SingleOp]:
    """Build `omp.parallel { omp.single { } }`, returning both ops."""
    single = omp.SingleOp(operands=[[], [], [], []], regions=[Region(Block())])
    parallel = omp.ParallelOp(
        operands=[[], [], [], [], [], []],
        regions=[Region(Block([single, omp.TerminatorOp()]))],
    )
    return parallel, single


@dataclass
class ConvertParallelToTaskloop(RewritePattern):
    """
    Rewrites an outermost `scf.parallel` into

        omp.parallel {
          omp.single {
            omp.taskloop [grainsize(..) | num_tasks(..)] {
              omp.loop_nest (...) { <body>; omp.yield }
            }
            omp.terminator
          }
          omp.terminator
        }

    All dimensions of the parallel loop are collapsed into the taskloop. The
    implicit taskgroup of `omp.taskloop` guarantees all iterations are complete
    before execution continues past the construct.

    The loop nest is emitted with inclusive upper bounds (`ub - 1` and
    `loop_inclusive`). This is equivalent to the exclusive form, but the
    LLVM 22 translation of `omp.taskloop` passes the upper bound straight to
    `__kmpc_taskloop`, which treats it as inclusive, regardless of
    `loop_inclusive`; the exclusive form would run one extra iteration.
    """

    grainsize: int | None
    num_tasks: int | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, loop: scf.ParallelOp, rewriter: PatternRewriter, /):
        if loop.initVals:
            # Reductions are not supported
            return
        if _is_nested_in_parallel_construct(loop):
            return

        grainsize: list[SSAValue] = []
        num_tasks: list[SSAValue] = []
        with ImplicitBuilder(rewriter):
            one = arith.ConstantOp(builtin.IntegerAttr.from_index_int_value(1))
            inclusive_upper_bounds = [
                arith.SubiOp(ub, one).result for ub in loop.upperBound
            ]
            if self.grainsize is not None:
                grainsize = [
                    arith.ConstantOp.from_int_and_width(self.grainsize, 64).result
                ]
            if self.num_tasks is not None:
                num_tasks = [
                    arith.ConstantOp.from_int_and_width(self.num_tasks, 64).result
                ]

        # Reuse the scf.parallel body block (index-typed induction variables) as
        # the loop_nest body, swapping the scf.reduce terminator for omp.yield.
        body = loop.detach_region(loop.body)
        terminator = body.block.last_op
        assert isinstance(terminator, scf.ReduceOp)
        rewriter.erase(terminator)
        if _contains_alloca(body.block):
            # Match MLIR's convert-scf-to-openmp: bound stack allocations to a
            # single iteration.
            scope_block = Block()
            for op in list(body.block.ops):
                op.detach()
                scope_block.add_op(op)
            scope_block.add_op(memref.AllocaScopeReturnOp(operands=[[]]))
            body.block.add_op(
                memref.AllocaScopeOp(result_types=[[]], regions=[Region(scope_block)])
            )
        body.block.add_op(omp.YieldOp())

        loop_nest = omp.LoopNestOp(
            operands=[loop.lowerBound, inclusive_upper_bounds, loop.step],
            properties={"loop_inclusive": builtin.UnitAttr()},
            regions=[body],
        )
        taskloop = omp.TaskloopOp(
            operands=[[], [], [], grainsize, [], [], num_tasks, [], [], []],
            regions=[Region(Block([loop_nest]))],
        )
        parallel, single = _empty_parallel_region()
        single.region.block.add_ops([taskloop, omp.TerminatorOp()])

        rewriter.replace(loop, parallel)


@dataclass(frozen=True)
class ConvertScfToOmpTasks(ModulePass):
    """
    Lowers outermost `scf.parallel` loops to OpenMP tasks.

    Each loop becomes an `omp.taskloop` created by a single thread
    (`omp.parallel { omp.single { ... } }`), with every loop dimension collapsed
    into the taskloop's `omp.loop_nest`. Loops with reductions, and loops nested
    inside another parallel construct, are left untouched.

    Arguments (all optional, mutually exclusive):

    - grainsize: int: minimum number of loop iterations assigned to each task.
    - num_tasks: int: number of tasks to create for each loop.

    If neither is given the OpenMP runtime chooses how to divide iterations.
    """

    name = "convert-scf-to-omp-tasks"

    grainsize: int | None = None
    num_tasks: int | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.grainsize is not None and self.num_tasks is not None:
            raise PassFailedException("grainsize and num_tasks are mutually exclusive")
        for value, option in (
            (self.grainsize, "grainsize"),
            (self.num_tasks, "num_tasks"),
        ):
            if value is not None and value <= 0:
                raise PassFailedException(f"{option} must be positive")

        PatternRewriteWalker(
            ConvertParallelToTaskloop(self.grainsize, self.num_tasks),
            apply_recursively=False,
        ).rewrite_module(op)
