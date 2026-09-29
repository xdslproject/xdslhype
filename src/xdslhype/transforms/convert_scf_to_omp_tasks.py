from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, memref, omp, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import PassFailedException

from .omp_task_dependences import DependenceInfo


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


def _is_convertible(loop: scf.ParallelOp) -> bool:
    return not loop.initVals and not _is_nested_in_parallel_construct(loop)


def _is_in_single_region(op: Operation) -> bool:
    parent = op.parent_op()
    while parent is not None:
        if isinstance(parent, omp.SingleOp):
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


def _wrap_in_single_region(block: Block) -> bool:
    """
    Wrap the ops of `block` from the first to the last one containing a
    convertible `scf.parallel` in a single `omp.parallel { omp.single { } }`.

    Values cannot escape an OpenMP region, so the range is extended forward
    over any later op using a value defined in it (e.g. the `memref.dealloc` of
    a buffer allocated between loops). Returns False, leaving the block
    untouched, if there are no such loops or if the block terminator would need
    to be included.
    """
    ops = list(block.ops)
    containing = [
        i
        for i, op in enumerate(ops)
        if any(
            isinstance(inner, scf.ParallelOp) and _is_convertible(inner)
            for inner in op.walk()
        )
    ]
    if not containing:
        return False

    first, last = containing[0], containing[-1]
    position = {op: i for i, op in enumerate(ops)}
    checked = first
    while checked <= last:
        for result in ops[checked].results:
            for use in result.uses:
                user = block.find_ancestor_op_in_block(use.operation)
                assert user is not None
                last = max(last, position[user])
        checked += 1
    if ops[last].has_trait(IsTerminator):
        return False

    wrapped = ops[first : last + 1]
    parallel, single = _empty_parallel_region()
    block.insert_op_before(parallel, wrapped[0])
    for op in wrapped:
        op.detach()
    single.region.block.add_ops([*wrapped, omp.TerminatorOp()])
    return True


def _take_body(loop: scf.ParallelOp, rewriter: PatternRewriter) -> Block:
    """
    Detach the body of `loop`, with its index-typed induction variables as
    block arguments and without the `scf.reduce` terminator.

    Stack allocations are bounded to a single iteration with
    `memref.alloca_scope`, matching MLIR's convert-scf-to-openmp.
    """
    region = loop.detach_region(loop.body)
    body = region.detach_block(region.block)
    terminator = body.last_op
    assert isinstance(terminator, scf.ReduceOp)
    rewriter.erase(terminator)
    if _contains_alloca(body):
        scope_block = Block()
        for op in list(body.ops):
            op.detach()
            scope_block.add_op(op)
        scope_block.add_op(memref.AllocaScopeReturnOp(operands=[[]]))
        body.add_op(
            memref.AllocaScopeOp(result_types=[[]], regions=[Region(scope_block)])
        )
    return body


def _replace_with_construct(
    loop: scf.ParallelOp, construct: Sequence[Operation], rewriter: PatternRewriter
) -> None:
    """
    Replace `loop` with `construct`, wrapped in `omp.parallel { omp.single { } }`
    unless the loop already sits inside an `omp.single` region.
    """
    if _is_in_single_region(loop):
        rewriter.replace(loop, construct)
        return

    parallel, single = _empty_parallel_region()
    single.region.block.add_ops([*construct, omp.TerminatorOp()])
    rewriter.replace(loop, parallel)


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

    A loop that is already inside an `omp.single` region (see the pass's
    `single_region` option) becomes just the `omp.taskloop`.
    """

    grainsize: int | None
    num_tasks: int | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, loop: scf.ParallelOp, rewriter: PatternRewriter, /):
        if not _is_convertible(loop):
            # Reductions and nested parallel loops are not supported
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

        body = _take_body(loop, rewriter)
        body.add_op(omp.YieldOp())
        loop_nest = omp.LoopNestOp(
            operands=[loop.lowerBound, inclusive_upper_bounds, loop.step],
            properties={"loop_inclusive": builtin.UnitAttr()},
            regions=[Region(body)],
        )
        taskloop = omp.TaskloopOp(
            operands=[[], [], [], grainsize, [], [], num_tasks, [], [], []],
            regions=[Region(Block([loop_nest]))],
        )
        _replace_with_construct(loop, [taskloop], rewriter)


@dataclass
class ConvertParallelToTasks(RewritePattern):
    """
    Rewrites an outermost `scf.parallel` into explicit tasks, one per `chunk`
    iterations of the outermost dimension:

        omp.parallel {
          omp.single {
            omp.taskgroup {
              scf.for %p = %lb0 to %ub0 step %step0 * chunk {
                omp.task {
                  scf.for %i0 = %p to min(%p + %step0 * chunk, %ub0) step %step0 {
                    scf.for %i1 = %lb1 to %ub1 step %step1 {
                      ... <body>
                    }
                  }
                  omp.terminator
                }
              }
              omp.terminator
            }
            omp.terminator
          }
          omp.terminator
        }

    With `chunk == 1` the `%i0` loop is omitted and `%p` is used directly. The
    remaining dimensions run sequentially within each task, and the taskgroup
    waits for all tasks before execution continues past the construct.

    A loop that is already inside an `omp.single` region (see the pass's
    `single_region` option) becomes just the `omp.taskgroup`.

    With dependence information (the pass's `depend` option) the taskgroup is
    omitted for loops whose accesses could be analysed. Instead each task gets
    `depend` clauses on the memrefs written by tasks of any loop:

    - For memrefs only accessed as `M[%i0, ...]`, where `%i0` is the outermost
    induction variable of a loop over `[0, ub)` with step 1, a task depends on
    its own patch(es) only, through the token `aligned_pointer(M) + %p`:
    `in` if the loop only reads M, `inout` if it writes it.
    - For any other such memref, tasks depend on the whole buffer, through the
    token `aligned_pointer(M)`: `in` if the loop only reads it, `inoutset` if
    it writes it, so that the loop's own tasks do not serialise. An empty
    separator task with `depend(inout)` is created before the loop's tasks so
    that they also wait for the tasks of the previous loop writing M.

    Tokens only identify dependences; they lie within M's allocation so tokens
    of different buffers never collide.
    """

    chunk: int
    dependences: DependenceInfo | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, loop: scf.ParallelOp, rewriter: PatternRewriter, /):
        if not _is_convertible(loop):
            # Reductions and nested parallel loops are not supported
            return

        lb0, ub0, step0 = loop.lowerBound[0], loop.upperBound[0], loop.step[0]
        with ImplicitBuilder(rewriter):
            if self.chunk == 1:
                task_step = step0
            else:
                chunk = arith.ConstantOp(
                    builtin.IntegerAttr.from_index_int_value(self.chunk)
                )
                task_step = arith.MuliOp(step0, chunk).result

        body = _take_body(loop, rewriter)
        induction_vars = list(body.args)

        task_block = Block()
        task_start_block = Block(arg_types=[builtin.IndexType()])
        task_start = task_start_block.args[0]

        # (lower bound, upper bound, step, induction variable) of each
        # sequential loop inside the task.
        sequential: list[tuple[SSAValue, SSAValue, SSAValue, SSAValue]] = []
        if self.chunk == 1:
            induction_vars[0].replace_all_uses_with(task_start)
        else:
            with ImplicitBuilder(task_block):
                chunk_end = arith.AddiOp(task_start, task_step)
                task_end = arith.MinSIOp(chunk_end, ub0).result
            sequential.append((task_start, task_end, step0, induction_vars[0]))
        sequential.extend(
            zip(
                loop.lowerBound[1:],
                loop.upperBound[1:],
                loop.step[1:],
                induction_vars[1:],
                strict=True,
            )
        )

        innermost = task_block
        for lb, ub, step, induction_var in sequential:
            for_body = Block(arg_types=[builtin.IndexType()])
            innermost.add_op(scf.ForOp(lb, ub, step, [], for_body))
            if innermost is not task_block:
                innermost.add_op(scf.YieldOp())
            induction_var.replace_all_uses_with(for_body.args[0])
            innermost = for_body

        for op in list(body.ops):
            op.detach()
            innermost.add_op(op)
        if innermost is not task_block:
            innermost.add_op(scf.YieldOp())
        task_block.add_op(omp.TerminatorOp())

        accesses = (
            None if self.dependences is None else self.dependences.loops.get(loop)
        )
        if accesses is None or accesses.opaque:
            task = omp.TaskOp(
                operands=[[], [], [], [], [], [], [], [], []],
                regions=[Region(task_block)],
            )
            task_start_block.add_ops([task, scf.YieldOp()])
            taskgroup = omp.TaskgroupOp(
                operands=[[], [], []],
                regions=[
                    Region(
                        Block(
                            [
                                scf.ForOp(lb0, ub0, task_step, [], task_start_block),
                                omp.TerminatorOp(),
                            ]
                        )
                    )
                ],
            )
            _replace_with_construct(loop, [taskgroup], rewriter)
            return

        assert self.dependences is not None
        construct: list[Operation] = []
        depend_vars: list[SSAValue] = []
        depend_kinds: list[omp.DependKindAttr] = []
        for target in accesses.accessed():
            if target not in self.dependences.task_written:
                continue
            written = target in accesses.writes
            with ImplicitBuilder(rewriter):
                base = memref.ExtractAlignedPointerAsIndexOp.get(target)
            if target in self.dependences.per_patch:
                with ImplicitBuilder(task_start_block):
                    token = _token(arith.AddiOp(base, task_start).result)
                kind = omp.DependKind.TASKDEPENDINOUT
            else:
                with ImplicitBuilder(rewriter):
                    token = _token(base.aligned_pointer)
                kind = omp.DependKind.TASKDEPENDINOUTSET
                if written:
                    construct.append(_separator_task(token))
            if not written:
                kind = omp.DependKind.TASKDEPENDIN
            depend_vars.append(token)
            depend_kinds.append(omp.DependKindAttr(kind))

        task = omp.TaskOp(
            operands=[[], [], depend_vars, [], [], [], [], [], []],
            properties=(
                {"depend_kinds": builtin.ArrayAttr(depend_kinds)}
                if depend_kinds
                else {}
            ),
            regions=[Region(task_block)],
        )
        task_start_block.add_ops([task, scf.YieldOp()])
        construct.append(scf.ForOp(lb0, ub0, task_step, [], task_start_block))
        _replace_with_construct(loop, construct, rewriter)


def _token(address: SSAValue) -> SSAValue:
    """An `!llvm.ptr` dependence token for an index-typed address."""
    return llvm.IntToPtrOp(arith.IndexCastOp(address, builtin.i64)).results[0]


def _separator_task(token: SSAValue) -> omp.TaskOp:
    """An empty task ordering later `inoutset` tasks after earlier ones."""
    return omp.TaskOp(
        operands=[[], [], [token], [], [], [], [], [], []],
        properties={
            "depend_kinds": builtin.ArrayAttr(
                [omp.DependKindAttr(omp.DependKind.TASKDEPENDINOUT)]
            )
        },
        regions=[Region(Block([omp.TerminatorOp()]))],
    )


def _insert_taskwaits(
    block: Block, dependences: DependenceInfo, outstanding: bool
) -> bool:
    """
    Insert `omp.taskwait` in code run by the thread creating the tasks wherever
    it may touch memory used by tasks still in flight, and before the tasks of
    loops that could not be analysed.

    `outstanding` tells whether tasks may be in flight on entry to the block;
    the same is returned for its exit.
    """
    for op in list(block.ops):
        if isinstance(op, scf.ParallelOp) and op in dependences.loops:
            if dependences.loops[op].opaque:
                if outstanding:
                    block.insert_op_before(omp.TaskwaitOp(operands=[[]]), op)
                # Its tasks are waited for by a taskgroup.
                outstanding = False
            else:
                outstanding = True
        elif any(
            isinstance(inner, scf.ParallelOp) and inner in dependences.loops
            for inner in op.walk()
        ):
            # A loop may run its body more than once: tasks of an earlier
            # iteration may be in flight on entry.
            entry = outstanding or not isinstance(op, scf.IfOp)
            exits = [
                _insert_taskwaits(inner, dependences, entry)
                for region in op.regions
                for inner in region.blocks
            ]
            outstanding = outstanding or any(exits)
        elif outstanding and dependences.needs_wait(op):
            block.insert_op_before(omp.TaskwaitOp(operands=[[]]), op)
            outstanding = False
    return outstanding


@dataclass(frozen=True)
class ConvertScfToOmpTasks(ModulePass):
    """
    Lowers outermost `scf.parallel` loops to OpenMP tasks, created by a single
    thread (`omp.parallel { omp.single { ... } }`). Loops with reductions, and
    loops nested inside another parallel construct, are left untouched.

    Arguments (all optional):

    - mode: {"taskloop", "task"}: how loops become tasks.
      - "taskloop" (default): each loop becomes an `omp.taskloop`, with every
      loop dimension collapsed into its `omp.loop_nest`.
      - "task": each loop becomes an `omp.taskgroup` containing one explicit
      `omp.task` per `chunk` iterations of the outermost dimension; the other
      dimensions run sequentially inside each task.
    - grainsize: int: (taskloop mode) minimum number of loop iterations assigned
    to each task.
    - num_tasks: int: (taskloop mode) number of tasks to create for each loop.
    Mutually exclusive with grainsize. If neither is given the OpenMP runtime
    chooses how to divide iterations.
    - chunk: int: (task mode) iterations of the outermost dimension per task,
    default 1.
    - depend: bool: (task mode) instead of waiting for all tasks of a loop
    before continuing, give each task `depend` clauses on the memrefs it reads
    and writes, so tasks of different loops can overlap (see
    `omp_task_dependences` for the analysis and its assumptions). Loops whose
    accesses cannot be analysed keep their taskgroup and are preceded by a
    `omp.taskwait`, as is any other code touching memory in use by tasks. Only
    useful with single_region: otherwise every loop still ends its own parallel
    region, which waits for its tasks.
    - single_region: bool: instead of one parallel region per loop, create a
    single `omp.parallel { omp.single { ... } }` per function, spanning from the
    first to the last op that contains a converted loop, so the thread team is
    forked once. The code between loops then runs on the single thread, and the
    tasks of each loop still complete before execution continues past it. The
    span is extended over later ops that use values defined in it; if that would
    include the function's terminator, the function falls back to one region
    per loop.
    """

    name = "convert-scf-to-omp-tasks"

    mode: Literal["taskloop", "task"] = "taskloop"
    grainsize: int | None = None
    num_tasks: int | None = None
    chunk: int | None = None
    single_region: bool = False
    depend: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for value, option in (
            (self.grainsize, "grainsize"),
            (self.num_tasks, "num_tasks"),
            (self.chunk, "chunk"),
        ):
            if value is not None and value <= 0:
                raise PassFailedException(f"{option} must be positive")

        pattern: RewritePattern
        if self.mode == "taskloop":
            if self.grainsize is not None and self.num_tasks is not None:
                raise PassFailedException(
                    "grainsize and num_tasks are mutually exclusive"
                )
            if self.chunk is not None:
                raise PassFailedException("chunk is only supported in task mode")
            if self.depend:
                raise PassFailedException("depend is only supported in task mode")
            pattern = ConvertParallelToTaskloop(self.grainsize, self.num_tasks)
        elif self.mode == "task":
            if self.grainsize is not None or self.num_tasks is not None:
                raise PassFailedException(
                    "grainsize and num_tasks are only supported in taskloop mode"
                )
            pattern = ConvertParallelToTasks(1 if self.chunk is None else self.chunk)
        else:
            raise PassFailedException(
                f"unknown mode {self.mode!r}, expected 'taskloop' or 'task'"
            )

        if self.single_region:
            funcs = [f for f in op.walk() if isinstance(f, func.FuncOp)]
            for func_op in funcs:
                for block in func_op.body.blocks:
                    _wrap_in_single_region(block)

        if isinstance(pattern, ConvertParallelToTasks) and self.depend:
            funcs = [f for f in op.walk() if isinstance(f, func.FuncOp)]
            for func_op in funcs:
                dependences = DependenceInfo.build(
                    loop
                    for loop in func_op.walk()
                    if isinstance(loop, scf.ParallelOp) and _is_convertible(loop)
                )
                singles = [s for s in func_op.walk() if isinstance(s, omp.SingleOp)]
                for single in singles:
                    for block in single.region.blocks:
                        _insert_taskwaits(block, dependences, False)
                PatternRewriteWalker(
                    ConvertParallelToTasks(pattern.chunk, dependences),
                    apply_recursively=False,
                ).rewrite_region(func_op.body)

        PatternRewriteWalker(pattern, apply_recursively=False).rewrite_module(op)
