from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xdsl.builder import Builder, ImplicitBuilder
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
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import PassFailedException

from .omp_task_dependences import DependenceInfo, is_unaliased_allocation


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
    `depend` clauses on the memrefs written by tasks of any loop, and on the
    freed memrefs read by tasks (`DependenceInfo.tracked`):

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

    Pointer accesses count towards the first case only if they are proven to
    stay within `M[%i0, ...]`. When the proof needs values only known at
    runtime, the check completing it is computed before the loop's tasks are
    created (see `DependenceInfo.runtime_check`); if it fails, the loop's tasks
    are isolated from all others by an `omp.taskwait` before and after
    creating them.
    """

    chunk: int
    dependences: DependenceInfo | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, loop: scf.ParallelOp, rewriter: PatternRewriter, /):
        if not _is_convertible(loop):
            # Reductions and nested parallel loops are not supported
            return

        accesses = (
            None if self.dependences is None else self.dependences.loops.get(loop)
        )
        check: SSAValue | None = None
        if self.dependences is not None and accesses is not None:
            # Before the body moves: the check refers to the loop's bounds.
            check_ops, check = self.dependences.runtime_check(loop)
            rewriter.insert(check_ops)

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
            if target not in self.dependences.tracked:
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
        if check is not None:
            construct = [_wait_unless(check), *construct, _wait_unless(check)]
        _replace_with_construct(loop, construct, rewriter)


def _token(address: SSAValue) -> SSAValue:
    """An `!llvm.ptr` dependence token for an index-typed address."""
    return llvm.IntToPtrOp(arith.IndexCastOp(address, builtin.i64)).results[0]


def _wait_unless(check: SSAValue) -> scf.IfOp:
    """`scf.if %check {} else { omp.taskwait }`."""
    return scf.IfOp(
        check,
        [],
        [scf.YieldOp()],
        [omp.TaskwaitOp(operands=[[]]), scf.YieldOp()],
    )


def _task_with_depends(
    depends: Sequence[tuple[SSAValue, omp.DependKind]], body: Sequence[Operation]
) -> omp.TaskOp:
    """An `omp.task` running `body`, with the given (token, kind) dependences."""
    return omp.TaskOp(
        operands=[[], [], [token for token, _ in depends], [], [], [], [], [], []],
        properties={
            "depend_kinds": builtin.ArrayAttr(
                [omp.DependKindAttr(kind) for _, kind in depends]
            )
        },
        regions=[Region(Block([*body, omp.TerminatorOp()]))],
    )


def _separator_task(token: SSAValue) -> omp.TaskOp:
    """An empty task ordering later `inoutset` tasks after earlier ones."""
    return _task_with_depends([(token, omp.DependKind.TASKDEPENDINOUT)], [])


def _dominates(value: SSAValue, op: Operation) -> bool:
    """
    Conservatively, whether `value` is available at `op`: it is an argument of
    a block enclosing `op`, or defined before `op` (or an ancestor of `op`) in
    such a block.
    """
    owner = value.owner
    block = owner if isinstance(owner, Block) else owner.parent_block()
    ancestor: Operation | None = op
    while ancestor is not None:
        if ancestor.parent_block() is block:
            return isinstance(owner, Block) or owner.is_before_in_block(ancestor)
        ancestor = ancestor.parent_op()
    return False


def _defer_deallocs(
    deallocs: Sequence[memref.DeallocOp], dependences: DependenceInfo, chunk: int
) -> bool:
    """
    Order the consecutive `deallocs`, run by the thread creating the tasks,
    after the tasks in flight that use their buffers, instead of waiting for all
    tasks. Returns False, leaving the ops untouched, if a full wait is needed.

    This requires every buffer to be an unaliased allocation (see
    `is_unaliased_allocation`): the tasks using it are then exactly the tasks of
    `dependences.task_loops`, and they carry dependences on it, since freed
    buffers accessed by tasks are in `dependences.tracked`. Tasks of opaque loops
    are complete already (taskgroup). A buffer not used by any task in flight
    needs no wait at all; its dealloc stays where it is.

    The other deallocs are moved into one deferred task, created in their
    place, so the creating thread does not block:

        scf.for %p = 0 to max(%ub, ...) step chunk {       // if any per-patch
          omp.task depend(inout: tok(M1) + %p, ..., inoutset: %X) {}
        }
        omp.task depend(inout: %X, inout: tok(W1), ...) {
          memref.dealloc M1 ...; memref.dealloc W1 ...
        }

    - Whole-buffer memrefs W: all tasks using W depend on `tok(W)` with kind
    `in` or `inoutset`; `inout` orders the dealloc task after all of them.
    - Per-patch memrefs M: tasks depend on `tok(M) + %p` for the chunk starts
    `%p` of their loop, i.e. the multiples of chunk below the loop's upper
    bound (its lower bound is 0 and step 1). A depend clause has a static list,
    so one empty fan-in task per chunk start below the largest upper bound of
    all loops using these buffers takes `inout` on each patch token, which
    orders it after every task using that patch, and `inoutset` on a token %X
    shared by the fan-in tasks only, so they do not order each other. The
    dealloc task's `inout: %X` orders it after all fan-in tasks. If an upper
    bound is not available at the dealloc, a full wait is used instead.

    %X is `tok(M1)` + the fan-in upper bound: past every patch token of M1, and
    normally inside M1's allocation (a patch token is `tok(M1)` plus a patch
    index, not a byte offset), so no task uses it. A collision with another
    token could only add orderings between tasks that are created in program
    order, never remove one or cause a cycle, so it cannot make the scheme
    unsound.

    All tasks, including the deferred deallocs, complete at the implicit
    barrier of the `omp.single` region, so every buffer is still freed before
    the region ends.
    """
    whole: list[SSAValue] = []
    per_patch: list[SSAValue] = []
    upper_bounds: dict[SSAValue, None] = {}
    deferred: list[memref.DeallocOp] = []
    for dealloc in deallocs:
        target = dealloc.memref
        if not is_unaliased_allocation(target):
            return False
        loops = dependences.task_loops(target)
        if not loops:
            continue  # not used by any task in flight
        if target not in dependences.tracked:
            return False
        if target in dependences.per_patch:
            bounds = [loop.upperBound[0] for loop in loops]
            if not all(_dominates(bound, dealloc) for bound in bounds):
                return False
            upper_bounds.update(dict.fromkeys(bounds))
            per_patch.append(target)
        else:
            whole.append(target)
        deferred.append(dealloc)
    if not deferred:
        return True

    builder = Builder(InsertPoint.before(deferred[0]))
    depends: list[tuple[SSAValue, omp.DependKind]] = []
    if per_patch:
        body = Block(arg_types=[builtin.IndexType()])
        with ImplicitBuilder(builder):
            bounds = list(upper_bounds)
            upper = bounds[0]
            for bound in bounds[1:]:
                upper = arith.MaxSIOp(upper, bound).result
            bases = [
                memref.ExtractAlignedPointerAsIndexOp.get(target).aligned_pointer
                for target in per_patch
            ]
            fan_in = _token(arith.AddiOp(bases[0], upper).result)
            zero = arith.ConstantOp(builtin.IntegerAttr.from_index_int_value(0))
            step = arith.ConstantOp(builtin.IntegerAttr.from_index_int_value(chunk))
        with ImplicitBuilder(body):
            patch_depends = [
                (
                    _token(arith.AddiOp(base, body.args[0]).result),
                    omp.DependKind.TASKDEPENDINOUT,
                )
                for base in bases
            ]
        fan_in_depends = (fan_in, omp.DependKind.TASKDEPENDINOUTSET)
        body.add_ops(
            [_task_with_depends([*patch_depends, fan_in_depends], []), scf.YieldOp()]
        )
        builder.insert(scf.ForOp(zero, upper, step, [], body))
        depends.append((fan_in, omp.DependKind.TASKDEPENDINOUT))
    for target in whole:
        with ImplicitBuilder(builder):
            base = memref.ExtractAlignedPointerAsIndexOp.get(target).aligned_pointer
            depends.append((_token(base), omp.DependKind.TASKDEPENDINOUT))
    task = builder.insert(_task_with_depends(depends, []))
    terminator = task.region.block.last_op
    assert terminator is not None
    for dealloc in deferred:
        dealloc.detach()
        task.region.block.insert_op_before(dealloc, terminator)
    return True


def _insert_taskwaits(
    block: Block,
    dependences: DependenceInfo,
    chunk: int,
    defer_deallocs: bool,
    outstanding: bool,
) -> bool:
    """
    Insert `omp.taskwait` in code run by the thread creating the tasks wherever
    it may touch memory used by tasks still in flight, and before the tasks of
    loops that could not be analysed. With `defer_deallocs`, deallocs of
    buffers used by tasks are deferred into tasks ordered after those tasks
    instead, where possible (see `_defer_deallocs`).

    `outstanding` tells whether tasks may be in flight on entry to the block;
    the same is returned for its exit.
    """
    done: set[Operation] = set()
    for op in list(block.ops):
        if op in done:
            continue
        if outstanding and defer_deallocs and isinstance(op, memref.DeallocOp):
            run: list[memref.DeallocOp] = []
            next_op: Operation | None = op
            while isinstance(next_op, memref.DeallocOp):
                run.append(next_op)
                next_op = next_op.next_op
            done.update(run)
            if not _defer_deallocs(run, dependences, chunk):
                block.insert_op_before(omp.TaskwaitOp(operands=[[]]), op)
                outstanding = False
        elif isinstance(op, scf.ParallelOp) and op in dependences.loops:
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
                _insert_taskwaits(inner, dependences, chunk, defer_deallocs, entry)
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
    `omp.taskwait`, as is any other code touching memory in use by tasks.
    Only useful with single_region: otherwise every loop still ends its own
    parallel region, which waits for its tasks.
    - prove_pointer_accesses: bool: (with depend) let `llvm.load`/`llvm.store`
    through a pointer derived from a memref count as a per-patch access when it
    is proven to stay within `M[%i0, ...]`, statically or by a runtime check
    computed before the loop's tasks are created (see `_RowCheck`); if the
    check fails, the loop's tasks are isolated by `omp.taskwait`s. Default
    false: such accesses make the memref's dependences whole-buffer.
    - defer_deallocs: bool: (with depend) instead of an `omp.taskwait` before
    `memref.dealloc`s of unaliased buffers used by tasks in flight, move them
    into tasks ordered after the tasks using the buffers (see
    `_defer_deallocs`). Default false.
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
    prove_pointer_accesses: bool = False
    defer_deallocs: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for value, option in (
            (self.grainsize, "grainsize"),
            (self.num_tasks, "num_tasks"),
            (self.chunk, "chunk"),
        ):
            if value is not None and value <= 0:
                raise PassFailedException(f"{option} must be positive")

        if not self.depend:
            for enabled, option in (
                (self.prove_pointer_accesses, "prove_pointer_accesses"),
                (self.defer_deallocs, "defer_deallocs"),
            ):
                if enabled:
                    raise PassFailedException(f"{option} requires depend=true")

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
                    (
                        loop
                        for loop in func_op.walk()
                        if isinstance(loop, scf.ParallelOp) and _is_convertible(loop)
                    ),
                    (
                        dealloc.memref
                        for dealloc in func_op.walk()
                        if self.defer_deallocs
                        and isinstance(dealloc, memref.DeallocOp)
                        and _is_in_single_region(dealloc)
                        and not _is_nested_in_parallel_construct(dealloc)
                    ),
                    self.prove_pointer_accesses,
                )
                singles = [s for s in func_op.walk() if isinstance(s, omp.SingleOp)]
                for single in singles:
                    for block in single.region.blocks:
                        _insert_taskwaits(
                            block,
                            dependences,
                            pattern.chunk,
                            self.defer_deallocs,
                            False,
                        )
                PatternRewriteWalker(
                    ConvertParallelToTasks(pattern.chunk, dependences),
                    apply_recursively=False,
                ).rewrite_region(func_op.body)

        PatternRewriteWalker(pattern, apply_recursively=False).rewrite_module(op)
