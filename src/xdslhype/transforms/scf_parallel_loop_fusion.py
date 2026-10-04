"""
Fusion of adjacent `scf.parallel` loop nests.

Two `scf.parallel` nests A and B, where B immediately follows A in the same
block, are fused level by level from the outside in, down to (but never
including) the innermost loop of the deeper nest. Their depths (the length of
the longest chain of directly nested `scf.parallel`s) may differ by at most one:

- depths 5 and 5: levels 1-4 are fused; the fused level-4 body holds A's
  level-4 body followed by B's, so the two innermost loops stay separate
  siblings;
- depths 4 and 5: levels 1-4 are fused; the fused level-4 body holds A's
  innermost body followed by B's level-4 body, including its innermost loop.

At every fused level both loops must have equivalent bounds and steps (see
`_Keys`), and no `init_vals`. Above the deepest fused level each body must hold
exactly one nested `scf.parallel`, all other ops being side-effect free (they
are moved freely). So within one iteration of the fused loops, everything A did
still happens before everything B did, and fusion is legal if no iteration of
B touches memory that a different iteration of A writes, or vice versa. This is
checked over every memory access of both nests (see `_collect_accesses` and
`_compatible`); if any access cannot be analysed the nests are left alone.

The check assumes that:

- distinct memrefs, memrefs loaded from different elements of a memref of
  memrefs, and memory reached through pointers with no memref origin, do not
  overlap. Loads through pointers with no memref origin read memory that is not
  written by either nest; stores through them prevent fusion;
- integer index arithmetic does not overflow (the frontend lowers C `int`s);
- `scf.parallel` loops are free of races: within one loop, different iterations
  never write the same location, nor read a location another iteration writes.

Calls prevent fusion, unless the callee is listed in `cell_local_callees`. Such a
callee is trusted to read only through pointer arguments with no memref origin,
and, through a pointer argument `p` computed from memref M, to read and write
only the cell `[p, p + s)`, where `s` is the distance between the pointers of
consecutive iterations (the coefficient of the innermost fused induction
variable in the address of `p`).
"""

from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, memref, scf
from xdsl.ir import Attribute, Block, BlockArgument, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import is_side_effect_free

_Poly = Counter[tuple[Hashable, ...]]
"""A sum of monomials (sorted tuples of atom keys) with integer coefficients."""

_FrozenPoly = tuple[tuple[tuple[Hashable, ...], int], ...]

_MAX_TERMS = 64
"""Bound on the number of terms of an expanded expression."""


class _Illegal(Exception):
    """The nests cannot be fused, or the fusion cannot be proven legal."""


def _freeze(poly: _Poly) -> _FrozenPoly:
    return tuple(sorted(((m, c) for m, c in poly.items() if c), key=repr))


def _add(lhs: _Poly, rhs: _Poly) -> _Poly:
    # Not `lhs + rhs`, which drops non-positive coefficients.
    result = Counter(lhs)
    result.update(rhs)
    return Counter({m: c for m, c in result.items() if c})


def _sub(lhs: _Poly, rhs: _Poly) -> _Poly:
    result = Counter(lhs)
    result.subtract(rhs)
    return Counter({m: c for m, c in result.items() if c})


def _scale(poly: _Poly, factor: _Poly) -> _Poly:
    result: _Poly = Counter()
    for m1, c1 in poly.items():
        for m2, c2 in factor.items():
            result[tuple(sorted(m1 + m2, key=repr))] += c1 * c2
    return Counter({m: c for m, c in result.items() if c})


def _constant(poly: _Poly) -> int | None:
    """The value of `poly` if it is a constant."""
    if any(m for m in poly):
        return None
    return poly[()]


def _is_pure(op: Operation) -> bool:
    # Some arith ops (e.g. arith.extsi) lack the Pure trait in xDSL.
    return is_side_effect_free(op) or (
        op.dialect_name() == "arith" and not op.regions
    )


def _is_integer(type: Attribute) -> bool:
    return isinstance(type, builtin.IndexType | builtin.IntegerType)


def _byte_size(type: Attribute) -> int | None:
    if isinstance(type, builtin.IntegerType | builtin.AnyFloat):
        if type.bitwidth % 8 == 0:
            return type.bitwidth // 8
    if isinstance(type, builtin.IndexType | llvm.LLVMPointerType):
        return 8
    return None


def _is_inside(value: SSAValue, roots: Iterable[Operation]) -> bool:
    owner = value.owner
    op = owner.parent_op() if isinstance(owner, Block) else owner
    return op is not None and any(r is op or r.is_ancestor(op) for r in roots)


def _induction_variables(loop: Operation) -> list[tuple[SSAValue, SSAValue, SSAValue]]:
    """The (lower bound, upper bound, step) of each induction variable."""
    if isinstance(loop, scf.ParallelOp):
        return list(zip(loop.lowerBound, loop.upperBound, loop.step))
    if isinstance(loop, scf.ForOp):
        return [(loop.lb, loop.ub, loop.step)]
    return []


class _Keys:
    """
    Keys identifying values of both nests, equal for values that are provably
    equal in corresponding iterations of A and B.

    Integer values are expanded into polynomials over atoms through
    `arith.addi/subi/muli`, integer constants, `arith.index_cast` and
    `arith.extsi`. The key of an atom is

    - for a value defined outside both nests, the value itself;
    - for an induction variable, its depth in the nest and the keys of its
      bounds and step, so corresponding loops of A and B with equivalent bounds
      have the same induction variables;
    - for a `memref.load` from a memref defined outside both nests and written
      by neither, the keys of the memref and indices;
    - for `memref.extract_aligned_pointer_as_index`, the key of the memref;
    - for other side-effect free ops, the op and the keys of its operands;
    - otherwise unique to the value.
    """

    def __init__(self, roots: tuple[Operation, ...], written: set[SSAValue]):
        self.roots = roots
        self.written = written
        self.atoms: dict[SSAValue, Hashable] = {}
        self.deps: dict[Hashable, frozenset[Hashable]] = {}
        """The induction variables each atom depends on."""
        self.bounds: dict[Hashable, tuple[_Poly, _Poly, _Poly]] = {}
        """The lower bound, upper bound and step of each induction variable."""

    def poly(self, value: SSAValue) -> _Poly:
        owner = value.owner
        if isinstance(owner, arith.ConstantOp) and isinstance(
            owner.value, builtin.IntegerAttr
        ):
            return Counter({(): owner.value.value.data})
        if isinstance(owner, arith.AddiOp):
            return _add(self.poly(owner.lhs), self.poly(owner.rhs))
        if isinstance(owner, arith.SubiOp):
            return _sub(self.poly(owner.lhs), self.poly(owner.rhs))
        if isinstance(owner, arith.MuliOp):
            product = _scale(self.poly(owner.lhs), self.poly(owner.rhs))
            if len(product) <= _MAX_TERMS:
                return product
        if isinstance(owner, arith.IndexCastOp | arith.ExtSIOp):
            return self.poly(owner.input)
        return Counter({(self.atom(value),): 1})

    def value(self, value: SSAValue) -> Hashable:
        if _is_integer(value.type):
            return ("poly", _freeze(self.poly(value)))
        return self.atom(value)

    def poly_deps(self, poly: _Poly) -> frozenset[Hashable]:
        return frozenset(d for m in poly for atom in m for d in self.deps[atom])

    def atom(self, value: SSAValue) -> Hashable:
        if value not in self.atoms:
            key, deps = self._atom(value)
            self.atoms[value] = key
            self.deps.setdefault(key, deps)
        return self.atoms[value]

    def _operands(self, values: Iterable[SSAValue]) -> tuple[tuple[Hashable, ...], frozenset[Hashable]]:
        keys = tuple(self.value(v) for v in values)
        deps = frozenset(
            d for v in values for d in self._value_deps(v)
        )
        return keys, deps

    def _value_deps(self, value: SSAValue) -> frozenset[Hashable]:
        if _is_integer(value.type):
            return self.poly_deps(self.poly(value))
        return self.deps[self.atom(value)]

    def _atom(self, value: SSAValue) -> tuple[Hashable, frozenset[Hashable]]:
        if not _is_inside(value, self.roots):
            return ("ext", id(value)), frozenset()
        owner = value.owner
        if isinstance(owner, Block):
            assert isinstance(value, BlockArgument)
            loop = owner.parent_op()
            ivs = _induction_variables(loop) if loop is not None else []
            if loop is None or value.index >= len(ivs):
                return ("opaque", id(value)), frozenset()
            lb, ub, step = ivs[value.index]
            keys, deps = self._operands((lb, ub, step))
            key = ("iv", self._depth(loop), value.index, keys)
            self.bounds[key] = (self.poly(lb), self.poly(ub), self.poly(step))
            return key, deps | {key}
        if (
            isinstance(owner, memref.LoadOp)
            and not _is_inside(owner.memref, self.roots)
            and owner.memref not in self.written
        ):
            keys, deps = self._operands((owner.memref, *owner.indices))
            return ("load", keys), deps
        if isinstance(owner, memref.ExtractAlignedPointerAsIndexOp):
            keys, deps = self._operands((owner.source,))
            return ("ptr", keys[0]), deps
        if _is_pure(owner) and not owner.regions:
            keys, deps = self._operands(owner.operands)
            attributes = (str(owner.properties), str(owner.attributes))
            index = cast(Operation, owner).results.index(value)
            return ("op", owner.name, attributes, index, keys), deps
        return ("opaque", id(value)), frozenset()

    def _depth(self, loop: Operation) -> int:
        depth = 0
        op: Operation | None = loop
        while op is not None and op not in self.roots:
            if isinstance(op, scf.ParallelOp | scf.ForOp):
                depth += 1
            op = op.parent_op()
        return depth


@dataclass(frozen=True)
class _Access:
    """A memory access of a nest."""

    base: Hashable
    """Key of the memref accessed."""
    write: bool
    unconditional: bool
    """Executed in every iteration of the loops enclosing it in the nest."""
    indices: tuple[Hashable, ...] | None
    """Keys of the indices, for `memref.load/store`."""
    offset: _Poly | None
    """Byte offset from the start of the memref, if known."""
    size: int | None
    """Number of bytes accessed, if known."""
    cell: bool = False
    """A pointer argument of a `cell_local_callees` call: accesses `[offset, offset + s)`."""


def _is_unconditional(op: Operation, root: Operation) -> bool:
    parent = op.parent_op()
    while parent is not None and parent is not root:
        if not isinstance(parent, scf.ParallelOp | scf.ForOp):
            return False
        parent = parent.parent_op()
    return True


def _is_private(memref_value: SSAValue, root: Operation) -> bool:
    """A memref allocated inside the nest, hence by one iteration."""
    return isinstance(
        memref_value.owner, memref.AllocOp | memref.AllocaOp
    ) and _is_inside(memref_value, (root,))


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
        elif isinstance(owner, Operation) and _is_pure(owner):
            worklist.extend(owner.operands)
    return list(memrefs), from_alloca


def _pointer_offset(keys: _Keys, ptr: SSAValue) -> tuple[Hashable, _Poly] | None:
    """
    The key of the memref `ptr` points into and the byte offset into it, or
    None if it has no memref origin (and is not computed from an `llvm.alloca`).
    """
    memrefs, from_alloca = pointer_memrefs(ptr)
    if from_alloca:
        raise _Illegal
    if not memrefs:
        return None
    if len(memrefs) != 1:
        raise _Illegal
    poly = _pointer_poly(keys, ptr)
    bases = [m for m in poly if any(_is_ptr(a) for a in m)]
    if len(bases) != 1 or len(bases[0]) != 1 or poly[bases[0]] != 1:
        raise _Illegal
    (ptr_atom,) = bases[0]
    del poly[bases[0]]
    return cast(tuple[str, Hashable], ptr_atom)[1], poly


def _is_ptr(atom: Hashable) -> bool:
    return isinstance(atom, tuple) and cast(tuple[Hashable, ...], atom)[:1] == ("ptr",)


def _pointer_poly(keys: _Keys, ptr: SSAValue) -> _Poly:
    owner = ptr.owner
    if isinstance(owner, llvm.IntToPtrOp):
        return keys.poly(owner.input)
    if isinstance(owner, llvm.GEPOp):
        indices = tuple(owner.rawConstantIndices.iter_values())
        size = _byte_size(owner.elem_type) if owner.elem_type else None
        if len(indices) != 1 or size is None:
            raise _Illegal
        (index,) = indices
        if index == llvm.GEP_USE_SSA_VAL:
            (ssa_index,) = owner.ssa_indices
            offset = _scale(keys.poly(ssa_index), Counter({(): size}))
        else:
            offset = Counter({(): index * size})
        return _add(_pointer_poly(keys, owner.ptr), offset)
    raise _Illegal


def _written_memrefs(roots: Iterable[Operation], cell_local: set[str]) -> set[SSAValue]:
    """Memrefs that may be written by the nests, before keys are available."""
    written: set[SSAValue] = set()
    for root in roots:
        for op in root.walk():
            if isinstance(op, memref.StoreOp):
                written.add(op.memref)
            elif isinstance(op, llvm.StoreOp):
                written.update(pointer_memrefs(op.ptr)[0])
            elif isinstance(op, func.CallOp) and op.callee.string_value() in cell_local:
                for arg in op.arguments:
                    if isinstance(arg.type, llvm.LLVMPointerType):
                        written.update(pointer_memrefs(arg)[0])
    return written


def _collect_accesses(
    keys: _Keys, root: Operation, cell_local: set[str]
) -> list[_Access]:
    accesses: list[_Access] = []
    for op in root.walk():
        if op is root:
            continue
        unconditional = _is_unconditional(op, root)
        if isinstance(op, memref.LoadOp | memref.StoreOp):
            if _is_private(op.memref, root):
                continue
            base = keys.atom(op.memref)
            if base[0] == "opaque":  # pyright: ignore[reportIndexIssue]
                raise _Illegal
            indices = tuple(keys.value(i) for i in op.indices)
            memref_type = cast(builtin.MemRefType[Attribute], op.memref.type)
            size = _byte_size(memref_type.element_type)
            offset = None
            if (
                len(op.indices) == 1
                and size is not None
                and isinstance(memref_type.layout, builtin.NoneAttr)
            ):
                offset = _scale(keys.poly(op.indices[0]), Counter({(): size}))
            accesses.append(
                _Access(
                    base,
                    isinstance(op, memref.StoreOp),
                    unconditional,
                    indices,
                    offset,
                    size,
                )
            )
        elif isinstance(op, llvm.LoadOp | llvm.StoreOp):
            target = _pointer_offset(keys, op.ptr)
            is_store = isinstance(op, llvm.StoreOp)
            if target is None:
                if is_store:
                    raise _Illegal
                continue
            type = op.value.type if isinstance(op, llvm.StoreOp) else op.dereferenced_value.type
            base, offset = target
            accesses.append(
                _Access(base, is_store, unconditional, None, offset, _byte_size(type))
            )
        elif isinstance(op, func.CallOp):
            if op.callee.string_value() not in cell_local:
                raise _Illegal
            for arg in op.arguments:
                if not isinstance(arg.type, llvm.LLVMPointerType):
                    continue
                target = _pointer_offset(keys, arg)
                if target is not None:
                    base, offset = target
                    accesses.append(
                        _Access(base, True, unconditional, None, offset, None, True)
                    )
        elif isinstance(op, memref.DeallocOp):
            if not _is_private(op.memref, root):
                raise _Illegal
        elif not (
            _is_pure(op)
            or isinstance(
                op,
                scf.ParallelOp
                | scf.ForOp
                | scf.IfOp
                | scf.ReduceOp
                | scf.ReduceReturnOp
                | scf.YieldOp
                | memref.AllocOp
                | memref.AllocaOp
                | llvm.AllocaOp,
            )
        ):
            raise _Illegal
    return accesses


class _Checker:
    def __init__(self, keys: _Keys, fused: list[Hashable]):
        self.keys = keys
        self.fused = fused
        """Induction variables of the fused loops, outermost first."""

    def compatible(self, a: _Access, b: _Access) -> bool:
        """
        No iteration of the fused loops accesses memory that `a` or `b` accesses
        in a different iteration, with one of them writing.
        """
        if a.cell or b.cell:
            return self._cells_compatible(a, b)
        if a.indices is not None and b.indices is not None:
            same = a.indices == b.indices
        elif a.offset is not None and b.offset is not None:
            same = _freeze(a.offset) == _freeze(b.offset) and a.size == b.size
        else:
            return False
        # Equal addresses can only meet in different iterations if the writing
        # access also writes the address in both, which would be a race.
        return same and (
            (a.write and a.unconditional) or (b.write and b.unconditional)
        )

    def _cells_compatible(self, a: _Access, b: _Access) -> bool:
        cell, other = (a, b) if a.cell else (b, a)
        assert cell.offset is not None
        stride = self._cell_size(cell)
        if stride is None or other.offset is None:
            return False
        if other.cell:
            return _freeze(cell.offset) == _freeze(other.offset)
        if other.size is None:
            return False
        # `other` must stay within the cell of its own iteration.
        span = self._span(_sub(other.offset, cell.offset))
        if span is None:
            return False
        low, high = span
        low_value = _constant(low)
        slack = _constant(_sub(_sub(stride, high), Counter({(): other.size})))
        return (
            low_value is not None
            and low_value >= 0
            and slack is not None
            and slack >= 0
        )

    def _extent(self, iv: Hashable) -> _Poly | None:
        lb, ub, step = self.keys.bounds[iv]
        if _constant(step) != 1:
            return None
        return _sub(ub, lb)

    def _cell_size(self, cell: _Access) -> _Poly | None:
        """
        The size of the cell of a `cell_local_callees` pointer, if cells of
        different iterations are proven not to overlap.
        """
        assert cell.offset is not None
        coefficients: dict[Hashable, _Poly] = {}
        for monomial, coefficient in cell.offset.items():
            ivs = [a for a in monomial if a in self.keys.bounds]
            rest = tuple(a for a in monomial if a not in self.keys.bounds)
            if self.keys.poly_deps(Counter({rest: 1})):
                return None
            if not ivs:
                continue
            if len(ivs) != 1 or ivs[0] not in self.fused:
                return None
            coefficients.setdefault(ivs[0], Counter())[rest] += coefficient
        # Iterations differing in a fused loop absent from the offset must access
        # different memrefs.
        base_deps = self.keys.deps.get(cell.base, frozenset())
        if any(iv not in coefficients and iv not in base_deps for iv in self.fused):
            return None
        ordered = [iv for iv in self.fused if iv in coefficients]
        if not ordered:
            return None
        for outer, inner in zip(ordered, ordered[1:]):
            extent = self._extent(inner)
            if extent is None:
                return None
            gap = _constant(_sub(coefficients[outer], _scale(coefficients[inner], extent)))
            if gap is None or gap < 0:
                return None
        return coefficients[ordered[-1]]

    def _span(self, poly: _Poly) -> tuple[_Poly, _Poly] | None:
        """
        Bounds of `poly` over its induction variables, which must be unfused
        loops' with constant coefficients.
        """
        low: _Poly = Counter()
        high: _Poly = Counter()
        for monomial, coefficient in poly.items():
            if not monomial:
                low[()] += coefficient
                high[()] += coefficient
                continue
            if len(monomial) != 1 or monomial[0] not in self.keys.bounds:
                return None
            (iv,) = monomial
            if iv in self.fused:
                return None
            lb, ub, _ = self.keys.bounds[iv]
            last = _sub(ub, Counter({(): 1}))
            first_term = _scale(lb, Counter({(): coefficient}))
            last_term = _scale(last, Counter({(): coefficient}))
            if coefficient < 0:
                first_term, last_term = last_term, first_term
            low = _add(low, first_term)
            high = _add(high, last_term)
        return low, high


def _nested_parallels(loop: scf.ParallelOp) -> list[scf.ParallelOp]:
    return [op for op in loop.body.block.ops if isinstance(op, scf.ParallelOp)]


def _depth(loop: scf.ParallelOp) -> int:
    return 1 + max((_depth(n) for n in _nested_parallels(loop)), default=0)


def _fused_levels(a: scf.ParallelOp, b: scf.ParallelOp) -> list[tuple[scf.ParallelOp, scf.ParallelOp]]:
    """The pairs of loops to fuse, outermost first, if A and B have the right shape."""
    depth_a, depth_b = _depth(a), _depth(b)
    if abs(depth_a - depth_b) > 1:
        raise _Illegal
    count = min(depth_a, depth_b, max(depth_a, depth_b) - 1)
    if count < 1:
        raise _Illegal
    levels: list[tuple[scf.ParallelOp, scf.ParallelOp]] = []
    for level in range(count):
        if a.initVals or b.initVals or len(a.lowerBound) != len(b.lowerBound):
            raise _Illegal
        levels.append((a, b))
        if level == count - 1:
            break
        nested: list[scf.ParallelOp] = []
        for loop in (a, b):
            inners = _nested_parallels(loop)
            if len(inners) != 1:
                raise _Illegal
            (inner,) = inners
            for op in loop.body.block.ops:
                if op is not inner and not isinstance(op, scf.ReduceOp) and (
                    op.regions or not _is_pure(op)
                ):
                    raise _Illegal
            nested.append(inner)
        a, b = nested
    return levels


def _check(
    a: scf.ParallelOp,
    b: scf.ParallelOp,
    levels: list[tuple[scf.ParallelOp, scf.ParallelOp]],
    cell_local: set[str],
) -> None:
    roots = (a, b)
    keys = _Keys(roots, _written_memrefs(roots, cell_local))
    fused: list[Hashable] = []
    for loop_a, loop_b in levels:
        for arg_a, arg_b in zip(loop_a.body.block.args, loop_b.body.block.args):
            if keys.atom(arg_a) != keys.atom(arg_b):
                raise _Illegal
            fused.append(keys.atom(arg_a))
    checker = _Checker(keys, fused)
    accesses_a = _collect_accesses(keys, a, cell_local)
    accesses_b = _collect_accesses(keys, b, cell_local)
    for access_a in accesses_a:
        for access_b in accesses_b:
            if access_a.base != access_b.base:
                continue
            if not (access_a.write or access_b.write):
                continue
            if not checker.compatible(access_a, access_b):
                raise _Illegal


def _fuse(levels: list[tuple[scf.ParallelOp, scf.ParallelOp]]) -> None:
    """Move B's ops into A at each level and erase B."""
    for index, (loop_a, loop_b) in enumerate(levels):
        for arg_a, arg_b in zip(loop_a.body.block.args, loop_b.body.block.args):
            arg_b.replace_all_uses_with(arg_a)
        if index + 1 < len(levels):
            target = levels[index + 1][0]
            skip = levels[index + 1][1]
        else:
            target = loop_a.body.block.last_op
            skip = None
        assert target is not None
        for op in list(loop_b.body.block.ops):
            if op is skip or isinstance(op, scf.ReduceOp):
                continue
            op.detach()
            Rewriter.insert_op(op, InsertPoint.before(target))
    Rewriter.erase_op(levels[0][1])


def _try_fuse(a: scf.ParallelOp, b: scf.ParallelOp, cell_local: set[str]) -> bool:
    try:
        levels = _fused_levels(a, b)
        _check(a, b, levels, cell_local)
    except _Illegal:
        return False
    _fuse(levels)
    return True


@dataclass(frozen=True)
class ScfParallelLoopFusion(ModulePass):
    """
    Fuses adjacent `scf.parallel` loop nests whose outer loops have equivalent
    bounds, keeping the innermost loops separate, when this provably preserves
    the program's behaviour. See the module documentation for details and
    assumptions.

    Arguments (all optional):

    - cell_local_callees: names of functions that, called inside a loop, are
    trusted to read only through pointer arguments with no memref origin, and
    to access, through a pointer argument computed from a memref, only the
    cell of the iteration making the call: up to the pointer passed by the
    next iteration of the innermost fused loop. Calls to other functions prevent
    fusion.
    """

    name = "scf-parallel-loop-fusion"

    cell_local_callees: tuple[str, ...] = ()

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        cell_local = set(self.cell_local_callees)
        changed = True
        while changed:
            changed = False
            for loop in op.walk():
                if not isinstance(loop, scf.ParallelOp):
                    continue
                following = loop.next_op
                if isinstance(following, scf.ParallelOp) and _try_fuse(
                    loop, following, cell_local
                ):
                    changed = True
                    break
