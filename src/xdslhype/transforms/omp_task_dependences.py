"""
Memory-access analysis used to give OpenMP tasks created from `scf.parallel`
loops `depend` clauses, instead of waiting for every task of a loop to finish.

Each converted loop is summarised by the memrefs its body reads and writes.
Accesses are recognised as

- `memref.load` / `memref.store`, attributed to their memref operand;
- `llvm.load` / `llvm.store` through a pointer computed (by side-effect free
  ops) from `memref.extract_aligned_pointer_as_index` of a memref, attributed
  to that memref.

Memrefs and `llvm.alloca`s defined inside the loop body are private to an
iteration and need no dependences. Loads through pointers with no memref origin
(for instance parameters loaded from a struct passed to the function) are
assumed to read memory that no task writes. Any other side effect in the body
(a store through such a pointer, a call, ...) makes the loop opaque: it is
separated from all other tasks by a full wait instead.

Distinct memref values are assumed not to overlap in memory, unless they
describe the same buffer with the same layout.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, llvm, memref, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.traits import RecursiveMemoryEffect, is_side_effect_free


@dataclass
class LoopAccesses:
    """The memrefs read and written by one `scf.parallel` loop."""

    reads: dict[SSAValue, None] = field(default_factory=dict[SSAValue, None])
    """Memrefs read by the loop (ordered set)."""
    writes: dict[SSAValue, None] = field(default_factory=dict[SSAValue, None])
    """Memrefs written by the loop (ordered set)."""
    per_patch: dict[SSAValue, bool] = field(default_factory=dict[SSAValue, bool])
    """
    Whether every access to the memref is a memref.load/store whose first index
    is the loop's outermost induction variable, in a loop over [0, ub) with
    step 1.
    """
    opaque: bool = False
    """The loop has side effects that cannot be attributed to memrefs."""

    def accessed(self) -> list[SSAValue]:
        return list({**self.reads, **self.writes})


def _is_constant(value: SSAValue, expected: int) -> bool:
    owner = value.owner
    return (
        isinstance(owner, arith.ConstantOp)
        and isinstance(owner.value, builtin.IntegerAttr)
        and owner.value.value.data == expected
    )


def _is_container(op: Operation) -> bool:
    """Ops whose effects are exactly the effects of the ops they contain."""
    return bool(op.regions) and (
        op.has_trait(RecursiveMemoryEffect) or isinstance(op, memref.AllocaScopeOp)
    )


def _is_defined_inside(value: SSAValue, op: Operation) -> bool:
    owner = value.owner
    if isinstance(owner, Block):
        parent = owner.parent_op()
        return parent is not None and (parent is op or op.is_ancestor(parent))
    return owner is not op and op.is_ancestor(owner)


def pointer_memrefs(ptr: SSAValue) -> tuple[list[SSAValue], bool]:
    """
    Trace `ptr` back through side-effect free ops.

    Returns the memrefs whose aligned pointer the address is computed from, and
    whether the address is (also) computed from an `llvm.alloca`.
    """
    memrefs: dict[SSAValue, None] = {}
    from_alloca = False
    worklist = [ptr]
    visited: set[SSAValue] = set()
    while worklist:
        value = worklist.pop()
        if value in visited:
            continue
        visited.add(value)
        owner = value.owner
        if isinstance(owner, memref.ExtractAlignedPointerAsIndexOp):
            memrefs[owner.source] = None
        elif isinstance(owner, llvm.AllocaOp):
            from_alloca = True
        elif isinstance(owner, Operation) and is_side_effect_free(owner):
            worklist.extend(owner.operands)
    return list(memrefs), from_alloca


def analyse_loop(loop: scf.ParallelOp) -> LoopAccesses:
    """Summarise the memory accesses of the body of `loop`."""
    accesses = LoopAccesses()
    outer_iv = loop.body.block.args[0]
    unit_outer_loop = _is_constant(loop.lowerBound[0], 0) and _is_constant(
        loop.step[0], 1
    )

    def record(target: SSAValue, is_write: bool, per_patch: bool) -> None:
        if _is_defined_inside(target, loop):
            return  # private to an iteration
        (accesses.writes if is_write else accesses.reads)[target] = None
        accesses.per_patch[target] = accesses.per_patch.get(target, True) and (
            per_patch and unit_outer_loop
        )

    for op in loop.body.walk():
        if op is loop or _is_container(op) or is_side_effect_free(op):
            continue
        if isinstance(op, memref.LoadOp | memref.StoreOp):
            per_patch = bool(op.indices) and op.indices[0] is outer_iv
            record(op.memref, isinstance(op, memref.StoreOp), per_patch)
        elif isinstance(op, llvm.LoadOp | llvm.StoreOp):
            is_write = isinstance(op, llvm.StoreOp)
            targets, from_alloca = pointer_memrefs(op.ptr)
            for target in targets:
                record(target, is_write, False)
            if not targets and not from_alloca and is_write:
                accesses.opaque = True
        elif isinstance(op, memref.AllocaOp | llvm.AllocaOp):
            continue
        else:
            accesses.opaque = True
    return accesses


@dataclass
class DependenceInfo:
    """Dependence information for the converted loops of one function."""

    loops: dict[scf.ParallelOp, LoopAccesses]
    task_written: set[SSAValue]
    """Memrefs written by the tasks of a non-opaque loop."""
    per_patch: set[SSAValue]
    """
    Memrefs whose tasks can depend on individual patches (outermost loop
    iterations) rather than on the whole buffer.
    """

    @staticmethod
    def build(loops: Iterable[scf.ParallelOp]) -> "DependenceInfo":
        summaries = {loop: analyse_loop(loop) for loop in loops}
        task_written: set[SSAValue] = set()
        per_patch_votes: dict[SSAValue, bool] = {}
        for accesses in summaries.values():
            if accesses.opaque:
                continue
            task_written.update(accesses.writes)
            for target, per_patch in accesses.per_patch.items():
                per_patch_votes[target] = (
                    per_patch_votes.get(target, True) and per_patch
                )
        return DependenceInfo(
            summaries,
            task_written,
            {target for target, ok in per_patch_votes.items() if ok},
        )

    def needs_wait(self, op: Operation) -> bool:
        """
        Whether `op`, executed by the thread creating the tasks, may touch
        memory that a task in flight accesses.
        """
        for inner in op.walk():
            if _is_container(inner) or is_side_effect_free(inner):
                continue
            if isinstance(inner, memref.AllocOp | memref.AllocaOp | llvm.AllocaOp):
                continue  # fresh memory
            if isinstance(inner, memref.LoadOp):
                if inner.memref in self.task_written:
                    return True
                continue
            if isinstance(inner, llvm.LoadOp):
                targets, _ = pointer_memrefs(inner.ptr)
                if any(target in self.task_written for target in targets):
                    return True
                continue
            return True
        return False
