"""
Memory-access analysis used to give OpenMP tasks created from `scf.parallel`
loops `depend` clauses, instead of waiting for every task of a loop to finish.

Each converted loop is summarised by the memrefs its body reads and writes.
Accesses are recognised as

- `memref.load` / `memref.store`, attributed to their memref operand;
- `llvm.load` / `llvm.store` through a pointer computed (by side-effect free
  ops) from `memref.extract_aligned_pointer_as_index` of a memref, attributed
  to that memref.

A memref gets dependences on individual patches (outermost iterations) if
every access to it stays within the part of it that belongs to the iteration
making it, `M[%i0, ...]` (see `DependenceInfo.per_patch`). For
`memref.load`/`memref.store` this is the case when the first index is `%i0`.
For `llvm.load`/`llvm.store` the address must be recognised, and proven to lie
in that part (see `_RowCheck`), which usually needs values only known at
runtime: the proof is then completed by a check computed before the loop's
tasks are created, see `DependenceInfo.runtime_check`.

Memrefs and `llvm.alloca`s defined inside the loop body are private to an
iteration and need no dependences. Loads through pointers with no memref origin
(for instance parameters loaded from a struct passed to the function) are
assumed to read memory that no task writes. Any other side effect in the body
(a store through such a pointer, a call, ...) makes the loop opaque: it is
separated from all other tasks by a full wait instead.

Distinct memref values are assumed not to overlap in memory, unless they
describe the same buffer with the same layout.

`memref.dealloc` of a buffer used by tasks is ordered after those tasks by
dependences rather than by a full wait when the buffer provably has no
aliases (`is_unaliased_allocation`). Buffers only read by tasks then also need
dependences (`DependenceInfo.tracked`), so that the dealloc can be ordered after
the reading tasks.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import arith, builtin, llvm, memref, scf
from xdsl.ir import Attribute, Block, BlockArgument, Operation, SSAValue
from xdsl.traits import RecursiveMemoryEffect, is_side_effect_free


@dataclass(frozen=True)
class CheckedAccess:
    """A pointer access that stays within `M[%i0, ...]` if a runtime check passes."""

    op: llvm.LoadOp | llvm.StoreOp
    target: SSAValue
    """The memref M."""
    leaf_bits: int
    """The `_RowCheck.leaf_bits` the check can be built with."""


@dataclass
class LoopAccesses:
    """The memrefs read and written by one `scf.parallel` loop."""

    reads: dict[SSAValue, None] = field(default_factory=dict[SSAValue, None])
    """Memrefs read by the loop (ordered set)."""
    writes: dict[SSAValue, None] = field(default_factory=dict[SSAValue, None])
    """Memrefs written by the loop (ordered set)."""
    per_patch: dict[SSAValue, bool] = field(default_factory=dict[SSAValue, bool])
    """
    Whether every access to the memref stays within `M[%i0, ...]`, where `%i0`
    is the outermost induction variable of a loop over [0, ub) with step 1:
    memref.load/store whose first index is `%i0`, or pointer accesses
    recognised by `_RowCheck`.
    """
    checked: list[CheckedAccess] = field(default_factory=list[CheckedAccess])
    """
    The pointer accesses counted as per-patch above that only stay within
    `M[%i0, ...]` if a runtime check passes.
    """
    opaque: bool = False
    """The loop has side effects that cannot be attributed to memrefs."""

    def accessed(self) -> list[SSAValue]:
        return list({**self.reads, **self.writes})


def _constant_int(value: SSAValue) -> int | None:
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(
        owner.value, builtin.IntegerAttr
    ):
        return owner.value.value.data
    return None


def _is_constant(value: SSAValue, expected: int) -> bool:
    return _constant_int(value) == expected


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


class _NoMatch(Exception):
    """The address of an access does not have a recognised form."""


_MAX_BITS = 62
"""
Bound on the magnitude (as a power of two) of every value a runtime check
computes, so that its own 64-bit arithmetic is exact.
"""

_MAX_TERMS = 64
"""Bound on the number of terms of an expanded address."""

_Value = int | SSAValue | None
"""
An index value of a runtime check: a constant, an SSA value, or None for a value
only known at runtime when the check is not being built.
"""

_Cond = bool | SSAValue | None
"""A condition of a runtime check, as for `_Value`."""

_Poly = list[tuple[int, tuple[SSAValue, ...]]]
"""A sum of products of SSA values (atoms), each with a constant coefficient."""


@dataclass(frozen=True)
class _Interval:
    """Bounds of the exact integer value of an expression, over all iterations."""

    lo: _Value
    hi: _Value
    bits: int
    """Static bound: `|lo|, |hi| <= 2**bits`."""


def _constant_interval(interval: _Interval) -> int | None:
    if (
        isinstance(interval.lo, int)
        and isinstance(interval.hi, int)
        and interval.lo == interval.hi
    ):
        return interval.lo
    return None


def _wrap64(value: int) -> int:
    return (value + 2**63) % 2**64 - 2**63


def _int_width(type: Attribute) -> int | None:
    """The bitwidth of an integer or (assumed 64-bit) index type."""
    if isinstance(type, builtin.IndexType):
        return 64
    if isinstance(type, builtin.IntegerType):
        return type.bitwidth
    return None


def _byte_size(type: Attribute) -> int | None:
    """The size in memory of a scalar of the given type, if known."""
    if isinstance(type, builtin.IntegerType) and type.bitwidth % 8 == 0:
        return type.bitwidth // 8
    if isinstance(
        type,
        builtin.Float16Type
        | builtin.BFloat16Type
        | builtin.Float32Type
        | builtin.Float64Type,
    ):
        return type.bitwidth // 8
    if isinstance(type, builtin.IndexType | llvm.LLVMPointerType):
        return 8
    return None


def _element_size(memref: SSAValue) -> int | None:
    """The size in memory of an element of `memref`, if known."""
    if not isinstance(memref.type, builtin.MemRefType):
        return None
    return _byte_size(cast(builtin.MemRefType[Attribute], memref.type).element_type)


def _poly(value: SSAValue) -> _Poly:
    """
    Expand a 64-bit integer or index value into a sum of products of atoms,
    congruent to it modulo 2**64, through `arith.addi/subi/muli`, constants and
    casts between index and i64. Anything else is an atom.
    """
    if _int_width(value.type) != 64:
        raise _NoMatch
    owner = value.owner
    constant = _constant_int(value)
    if constant is not None:
        return [(constant, ())]
    if isinstance(owner, arith.AddiOp):
        return _poly(owner.lhs) + _poly(owner.rhs)
    if isinstance(owner, arith.SubiOp):
        return _poly(owner.lhs) + [(-c, atoms) for c, atoms in _poly(owner.rhs)]
    if isinstance(owner, arith.MuliOp):
        product = [
            (c1 * c2, atoms1 + atoms2)
            for c1, atoms1 in _poly(owner.lhs)
            for c2, atoms2 in _poly(owner.rhs)
        ]
        if len(product) > _MAX_TERMS:
            raise _NoMatch
        return product
    if isinstance(owner, arith.IndexCastOp) and _int_width(owner.input.type) == 64:
        return _poly(owner.input)
    return [(1, (value,))]


def _pointer_poly(ptr: SSAValue) -> _Poly:
    """
    The address held by `ptr` as a `_Poly` in bytes, through `llvm.inttoptr` of
    a 64-bit integer and `llvm.getelementptr` with one constant index.
    """
    owner = ptr.owner
    if isinstance(owner, llvm.IntToPtrOp):
        return _poly(owner.input)
    if isinstance(owner, llvm.GEPOp):
        indices = tuple(owner.rawConstantIndices.iter_values())
        size = _byte_size(owner.elem_type)
        if len(indices) != 1 or owner.ssa_indices or size is None:
            raise _NoMatch
        (index,) = indices
        if index == llvm.GEP_USE_SSA_VAL:
            raise _NoMatch
        return _pointer_poly(owner.ptr) + [(index * size, ())]
    raise _NoMatch


_Key = tuple[str, int, int]


def _dim_key(source: SSAValue, dim: int) -> int | _Key:
    """A value equal to `memref.dim(source, dim)`, for comparing expressions."""
    memref_type = source.type
    assert isinstance(memref_type, builtin.MemRefType)
    size = memref_type.get_shape()[dim]
    if size != builtin.DYNAMIC_INDEX:
        return size
    owner = source.owner
    if isinstance(owner, memref.AllocOp | memref.AllocaOp):
        dynamic = sum(
            1 for s in memref_type.get_shape()[:dim] if s == builtin.DYNAMIC_INDEX
        )
        return _key(owner.dynamic_sizes[dynamic])
    return ("dim", id(source), dim)


def _constant_dim(op: memref.DimOp) -> int | None:
    """The index of a `memref.dim` of a ranked memref, if constant and in range."""
    dim = _constant_int(op.index)
    memref_type = op.source.type
    if (
        dim is not None
        and isinstance(memref_type, builtin.MemRefType)
        and 0 <= dim < memref_type.get_num_dims()
    ):
        return dim
    return None


def _key(atom: SSAValue) -> int | _Key:
    """A key identifying the value of an atom, for comparing expressions."""
    owner = atom.owner
    constant = _constant_int(atom)
    if constant is not None:
        return constant
    if isinstance(owner, memref.DimOp):
        dim = _constant_dim(owner)
        if dim is not None:
            return _dim_key(owner.source, dim)
    if isinstance(owner, arith.IndexCastOp) and _int_width(owner.input.type) == 64:
        return _key(owner.input)
    return ("value", id(atom), 0)


def _canonical(
    poly: Iterable[tuple[int, Iterable[int | _Key]]],
) -> dict[tuple[_Key, ...], int]:
    """Collect terms with the same keys, modulo 2**64."""
    terms: Counter[tuple[_Key, ...]] = Counter()
    for coefficient, keys in poly:
        factors: list[_Key] = []
        for key in keys:
            if isinstance(key, int):
                coefficient *= key
            else:
                factors.append(key)
        terms[tuple(sorted(factors))] += coefficient
    return {k: c % 2**64 for k, c in terms.items() if c % 2**64}


class _RowCheck:
    """
    Proves that pointer accesses of a loop stay within `M[%i0, ...]`, where
    `%i0` is the loop's outermost induction variable, building (with `build`)
    the runtime check that completes the proof, or (without) folding it as far
    as possible statically.

    The address of an access, in bytes, must expand (`_pointer_poly`) to

        aligned_pointer(M) + %i0 * A + B

    where M has the identity layout and A and B do not depend on `%i0`. Such
    an address lies in `M[%i0, ...]`, of R = sizeof(element) * dim(M, 1) * ...
    * dim(M, rank - 1) bytes, if

    - A == R modulo 2**64, which must hold statically: A and R expand to the
    same terms once static sizes, and `memref.dim` of an allocation, are
    replaced by their values (otherwise the access is not recognised),
    - 0 <= B and B + sizeof(access) <= R for every iteration, and
    - the loop's upper bound is at most dim(M, 0), so that `%i0` is a row of M
    and the address lies in M's allocation.

    Addresses are computed modulo 2**64 (index, assumed 64-bit) and 2**W for
    narrower integers, while the conditions on B are on its exact value, which
    is bounded by interval arithmetic over the bounds of the induction
    variables (of the loop, and of `scf.for` loops nested in it, with step at
    least 1; an empty loop makes no accesses, so its inverted bounds are
    harmless). addi/subi/muli, constants and `arith.index_cast` are
    interpreted; any other value is a leaf, which must be defined outside the
    loop, or computed inside it only by those ops and `memref.dim`, which are
    then hoisted. Arithmetic is exact modulo 2**W, so a W-bit value is exact
    when the exact value is in range for W bits; this is checked where it is
    sign extended by `arith.index_cast`, and follows for B from the bounds
    above.

    Leaves wider than `leaf_bits` bits are checked to be at most `2**leaf_bits`
    in magnitude, so that every intermediate value of the check is statically
    at most `2**_MAX_BITS` (`max_bits`) and the check's own arithmetic, in
    64 bits, is exact.
    """

    def __init__(self, loop: scf.ParallelOp, leaf_bits: int, build: bool):
        self.loop = loop
        self.leaf_bits = leaf_bits
        self.build = build
        self.ops: list[Operation] = []
        """The ops computing the check, to insert before the loop."""
        self.conds: list[_Cond] = []
        """The conditions the check consists of."""
        self.max_bits = 0
        self._intervals: dict[SSAValue, _Interval] = {}
        self._hoisted: dict[SSAValue, SSAValue] = {}
        self._casts: dict[SSAValue, SSAValue] = {}
        self._constants: dict[int, SSAValue] = {}
        self._created: dict[tuple[str, SSAValue, SSAValue], SSAValue] = {}
        self._row_bytes: dict[SSAValue, _Value] = {}

    # Arithmetic on index values, folded when constant.

    def _ssa(self, value: int | SSAValue) -> SSAValue:
        if isinstance(value, SSAValue):
            return value
        if value not in self._constants:
            constant = arith.ConstantOp(builtin.IntegerAttr.from_index_int_value(value))
            self.ops.append(constant)
            self._constants[value] = constant.result
        return self._constants[value]

    def _binary(
        self,
        lhs: _Value,
        rhs: _Value,
        fold: Callable[[int, int], int],
        create: Callable[[SSAValue, SSAValue], Operation],
    ) -> _Value:
        if isinstance(lhs, int) and isinstance(rhs, int):
            return _wrap64(fold(lhs, rhs))
        if lhs is None or rhs is None:
            return None
        return self._create(create, self._ssa(lhs), self._ssa(rhs))

    def _create(
        self,
        create: Callable[[SSAValue, SSAValue], Operation],
        lhs: SSAValue,
        rhs: SSAValue,
    ) -> SSAValue:
        """`create(lhs, rhs)`, reusing an identical op built before."""
        op = create(lhs, rhs)
        key = (str(op.attributes | op.properties) + op.name, lhs, rhs)
        if key not in self._created:
            self.ops.append(op)
            self._created[key] = op.results[0]
        else:
            op.erase()
        return self._created[key]

    def _add(self, lhs: _Value, rhs: _Value) -> _Value:
        if lhs == 0:
            return rhs
        if rhs == 0:
            return lhs
        return self._binary(lhs, rhs, lambda a, b: a + b, arith.AddiOp)

    def _sub(self, lhs: _Value, rhs: _Value) -> _Value:
        if rhs == 0:
            return lhs
        return self._binary(lhs, rhs, lambda a, b: a - b, arith.SubiOp)

    def _mul(self, lhs: _Value, rhs: _Value) -> _Value:
        if lhs == 0 or rhs == 0:
            return 0
        if lhs == 1:
            return rhs
        if rhs == 1:
            return lhs
        return self._binary(lhs, rhs, lambda a, b: a * b, arith.MuliOp)

    def _min(self, lhs: _Value, rhs: _Value) -> _Value:
        if lhs is rhs:
            return lhs
        return self._binary(lhs, rhs, min, arith.MinSIOp)

    def _max(self, lhs: _Value, rhs: _Value) -> _Value:
        if lhs is rhs:
            return lhs
        return self._binary(lhs, rhs, max, arith.MaxSIOp)

    def _sle(self, lhs: _Value, rhs: _Value) -> _Cond:
        if isinstance(lhs, int) and isinstance(rhs, int):
            return lhs <= rhs
        if lhs is not None and lhs is rhs:
            return True
        if lhs is None or rhs is None:
            return None
        return self._create(
            lambda a, b: arith.CmpiOp(a, b, "sle"), self._ssa(lhs), self._ssa(rhs)
        )

    def require(self, cond: _Cond) -> None:
        if cond is not True:
            self.conds.append(cond)

    def folded(self) -> bool | None:
        """The conjunction of the conditions, as far as it folds statically."""
        if False in self.conds:
            return False
        if not self.conds:
            return True
        return None

    def result(self) -> SSAValue:
        """The conjunction of the conditions, when building."""
        assert self.build
        conds = [c for c in self.conds if isinstance(c, SSAValue)]
        if False in self.conds or not conds:
            constant = arith.ConstantOp(
                builtin.IntegerAttr(int(False not in self.conds), builtin.i1)
            )
            self.ops.append(constant)
            return constant.result
        cond = conds[0]
        for c in dict.fromkeys(conds[1:]):
            cond = self._create(arith.AndIOp, cond, c)
        return cond

    # Values available before the loop.

    def _hoist(self, value: SSAValue) -> SSAValue | None:
        """
        `value`, computed before the loop (None when not building). Raises
        _NoMatch if `value` is not loop-invariant or computed by other ops.
        """
        if not _is_defined_inside(value, self.loop):
            return value if self.build else None
        if value in self._hoisted:
            return self._hoisted[value]
        owner = value.owner
        if not isinstance(
            owner,
            arith.ConstantOp
            | arith.AddiOp
            | arith.SubiOp
            | arith.MuliOp
            | arith.IndexCastOp
            | memref.DimOp,
        ):
            raise _NoMatch
        if isinstance(owner, memref.DimOp) and _constant_dim(owner) is None:
            raise _NoMatch  # hoisting could introduce undefined behaviour
        operands = [self._hoist(operand) for operand in owner.operands]
        if not self.build:
            return None
        clone = owner.clone(
            value_mapper={
                old: new
                for old, new in zip(owner.operands, operands, strict=True)
                if new is not None
            }
        )
        self.ops.append(clone)
        self._hoisted[value] = clone.results[0]
        return clone.results[0]

    def _index(self, value: SSAValue) -> _Value:
        """An integer or index value before the loop, sign extended to index."""
        constant = _constant_int(value)
        if constant is not None:
            return constant
        hoisted = self._hoist(value)
        if hoisted is None or isinstance(hoisted.type, builtin.IndexType):
            return hoisted
        if hoisted not in self._casts:
            cast = arith.IndexCastOp(hoisted, builtin.IndexType())
            self.ops.append(cast)
            self._casts[hoisted] = cast.result
        return self._casts[hoisted]

    def _dim(self, source: SSAValue, dim: int) -> _Value:
        key = _dim_key(source, dim)
        if isinstance(key, int):
            return key
        if not self.build:
            return None
        return self._create(memref.DimOp, source, self._ssa(dim))

    # Bounds of values over all iterations.

    def _note(self, interval: _Interval) -> _Interval:
        self.max_bits = max(self.max_bits, interval.bits)
        return interval

    def _add_intervals(self, lhs: _Interval, rhs: _Interval) -> _Interval:
        return self._note(
            _Interval(
                self._add(lhs.lo, rhs.lo),
                self._add(lhs.hi, rhs.hi),
                max(lhs.bits, rhs.bits) + 1,
            )
        )

    def _sub_intervals(self, lhs: _Interval, rhs: _Interval) -> _Interval:
        return self._note(
            _Interval(
                self._sub(lhs.lo, rhs.hi),
                self._sub(lhs.hi, rhs.lo),
                max(lhs.bits, rhs.bits) + 1,
            )
        )

    def _mul_intervals(self, lhs: _Interval, rhs: _Interval) -> _Interval:
        bits = lhs.bits + rhs.bits
        if _constant_interval(lhs) is not None:
            lhs, rhs = rhs, lhs
        factor = _constant_interval(rhs)
        if factor is not None:
            # Multiplication by a constant: its sign orders the bounds.
            lo, hi = self._mul(lhs.lo, factor), self._mul(lhs.hi, factor)
            if factor < 0:
                lo, hi = hi, lo
            return self._note(_Interval(lo, hi, bits))
        products: list[_Value] = []
        for a in dict.fromkeys([lhs.lo, lhs.hi]):
            for b in dict.fromkeys([rhs.lo, rhs.hi]):
                products.append(self._mul(a, b))
        lo = hi = products[0]
        for product in products[1:]:
            lo, hi = self._min(lo, product), self._max(hi, product)
        return self._note(_Interval(lo, hi, bits))

    def _leaf(self, value: SSAValue) -> _Interval:
        width = _int_width(value.type)
        if width is None:
            raise _NoMatch
        index = self._index(value)
        bits = width - 1
        if bits > self.leaf_bits:
            bits = self.leaf_bits
            self.require(self._sle(-(2**bits), index))
            self.require(self._sle(index, 2**bits))
        return self._note(_Interval(index, index, bits))

    def _induction_interval(self, value: BlockArgument) -> _Interval:
        parent = value.block.parent_op()
        if parent is self.loop and value.index > 0:
            lb = self.loop.lowerBound[value.index]
            ub = self.loop.upperBound[value.index]
            step = self.loop.step[value.index]
        elif (
            isinstance(parent, scf.ForOp)
            and self.loop.is_ancestor(parent)
            and value is parent.body.block.args[0]
        ):
            lb, ub, step = parent.lb, parent.ub, parent.step
        else:
            raise _NoMatch
        lower, upper = self.interval(lb), self.interval(ub)
        self.require(self._sle(1, self.interval(step).lo))
        return self._note(
            _Interval(lower.lo, self._sub(upper.hi, 1), max(lower.bits, upper.bits + 1))
        )

    def interval(self, value: SSAValue) -> _Interval:
        """Bounds of the exact value of an integer or index `value`."""
        if value in self._intervals:
            return self._intervals[value]
        owner = value.owner
        constant = _constant_int(value)
        if constant is not None:
            result = _Interval(constant, constant, abs(constant).bit_length())
        elif value is self.loop.body.block.args[0]:
            raise _NoMatch
        elif isinstance(value, BlockArgument) and _is_defined_inside(value, self.loop):
            result = self._induction_interval(value)
        elif isinstance(owner, arith.AddiOp):
            result = self._add_intervals(
                self.interval(owner.lhs), self.interval(owner.rhs)
            )
        elif isinstance(owner, arith.SubiOp):
            result = self._sub_intervals(
                self.interval(owner.lhs), self.interval(owner.rhs)
            )
        elif isinstance(owner, arith.MuliOp):
            result = self._mul_intervals(
                self.interval(owner.lhs), self.interval(owner.rhs)
            )
        elif isinstance(owner, arith.IndexCastOp):
            result = self._cast_interval(owner)
        else:
            result = self._leaf(value)
        self._intervals[value] = self._note(result)
        return result

    def _cast_interval(self, cast: arith.IndexCastOp) -> _Interval:
        source = self.interval(cast.input)
        width = _int_width(cast.input.type)
        result_width = _int_width(cast.result.type)
        assert width is not None
        assert result_width is not None
        if width >= result_width:
            # Truncation: congruent modulo 2**result_width, which is all the
            # arithmetic on the result preserves until it is sign extended.
            return source
        # Sign extension: exact if the source is in range.
        if source.bits > width - 2:
            self.require(self._sle(-(2 ** (width - 1)), source.lo))
            self.require(self._sle(source.hi, 2 ** (width - 1) - 1))
        return _Interval(source.lo, source.hi, min(source.bits, width - 1))

    def _row_size(self, target: SSAValue) -> _Value:
        """The size in bytes of `target[%i0, ...]`."""
        if target not in self._row_bytes:
            memref_type = target.type
            assert isinstance(memref_type, builtin.MemRefType)
            size = _element_size(target)
            assert size is not None
            row: _Value = size
            for dim in range(1, memref_type.get_num_dims()):
                row = self._mul(row, self._dim(target, dim))
            self._row_bytes[target] = row
        return self._row_bytes[target]

    def access(self, op: llvm.LoadOp | llvm.StoreOp, target: SSAValue) -> None:
        """Add the conditions for `op` to access only `target[%i0, ...]`."""
        memref_type = target.type
        if (
            not isinstance(memref_type, builtin.MemRefType)
            or not isinstance(memref_type.layout, builtin.NoneAttr)
            or not memref_type.get_num_dims()
        ):
            raise _NoMatch
        element_size = _element_size(target)
        access_size = _byte_size(
            op.value.type
            if isinstance(op, llvm.StoreOp)
            else op.dereferenced_value.type
        )
        if element_size is None or access_size is None:
            raise _NoMatch

        # Split the address into aligned_pointer(M) + %i0 * stride + offset.
        outer = self.loop.body.block.args[0]
        bases = 0
        stride: _Poly = []
        offset: _Poly = []
        for coefficient, atoms in _pointer_poly(op.ptr):
            if any(
                isinstance(atom.owner, memref.ExtractAlignedPointerAsIndexOp)
                for atom in atoms
            ):
                owner = atoms[0].owner
                if not (
                    coefficient == 1
                    and len(atoms) == 1
                    and isinstance(owner, memref.ExtractAlignedPointerAsIndexOp)
                    and owner.source is target
                ):
                    raise _NoMatch
                bases += 1
                continue
            uses = sum(1 for atom in atoms if atom is outer)
            if uses == 0:
                offset.append((coefficient, atoms))
            elif uses == 1:
                stride.append(
                    (coefficient, tuple(atom for atom in atoms if atom is not outer))
                )
            else:
                raise _NoMatch
        if bases != 1:
            raise _NoMatch

        row = self._row_size(target)
        row_keys = [
            _dim_key(target, dim) for dim in range(1, memref_type.get_num_dims())
        ]
        if _canonical((c, [_key(a) for a in atoms]) for c, atoms in stride) != (
            _canonical([(element_size, row_keys)])
        ):
            raise _NoMatch

        bounds = _Interval(0, 0, 0)
        for coefficient, atoms in offset:
            term_bounds = _Interval(
                coefficient, coefficient, abs(coefficient).bit_length()
            )
            for atom in atoms:
                term_bounds = self._mul_intervals(term_bounds, self.interval(atom))
            bounds = self._add_intervals(bounds, term_bounds)
        self._note(_Interval(0, 0, max(bounds.bits, access_size.bit_length()) + 1))
        end = self._add(bounds.hi, access_size)
        self.require(self._sle(0, bounds.lo))
        self.require(self._sle(end, row))

        # %i0 must be a row of M.
        upper = self.loop.upperBound[0]
        if _canonical((c, [_key(a) for a in atoms]) for c, atoms in _poly(upper)) != (
            _canonical([(1, [_dim_key(target, 0)])])
        ):
            self.require(self._sle(self._index(upper), self._dim(target, 0)))


def _prove_in_row(
    loop: scf.ParallelOp, op: llvm.LoadOp | llvm.StoreOp, target: SSAValue
) -> CheckedAccess | bool:
    """
    Whether `op` accesses only `target[%i0, ...]` (see `_RowCheck`): True or
    False if known statically, or the access if it needs a runtime check, with
    the largest `leaf_bits` the check can be built with.
    """

    def max_bits(leaf_bits: int) -> int:
        check = _RowCheck(loop, leaf_bits, build=False)
        check.access(op, target)
        return check.max_bits

    # Whether the check folds does not depend on leaf_bits, as leaves are not
    # constants, and max_bits grows with it.
    check = _RowCheck(loop, 1, build=False)
    try:
        check.access(op, target)
    except _NoMatch:
        return False
    if check.max_bits > _MAX_BITS:
        return False
    folded = check.folded()
    if folded is not None:
        return folded
    lo, hi = 1, _MAX_BITS
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if max_bits(mid) <= _MAX_BITS:
            lo = mid
        else:
            hi = mid - 1
    return CheckedAccess(op, target, lo)


def _only_used_as_address(address: SSAValue) -> bool:
    """
    Whether every use of `address`, through side-effect free ops, is as the
    address of an `llvm.load` / `llvm.store` (so that it cannot escape).
    """
    worklist = [address]
    visited: set[SSAValue] = set()
    while worklist:
        value = worklist.pop()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            user = use.operation
            if isinstance(user, llvm.LoadOp) or (
                isinstance(user, llvm.StoreOp) and user.value is not value
            ):
                continue
            if not user.regions and is_side_effect_free(user):
                worklist.extend(user.results)
                continue
            return False
    return True


def is_unaliased_allocation(target: SSAValue) -> bool:
    """
    Whether `target` is the result of a `memref.alloc` whose memory can only be
    accessed through `target` itself, in ways the analysis attributes to it:
    `memref.load` / `memref.store` on it, `memref.dealloc` of it, and
    `llvm.load` / `llvm.store` through its
    `memref.extract_aligned_pointer_as_index`. Every task accessing the buffer
    is then known, and no other memref value or pointer aliases it.
    """
    if not isinstance(target.owner, memref.AllocOp):
        return False
    for use in target.uses:
        user = use.operation
        if isinstance(user, memref.LoadOp | memref.DeallocOp):
            continue
        if isinstance(user, memref.StoreOp) and user.value is not target:
            continue
        if isinstance(
            user, memref.ExtractAlignedPointerAsIndexOp
        ) and _only_used_as_address(user.aligned_pointer):
            continue
        return False
    return True


def analyse_loop(
    loop: scf.ParallelOp, prove_pointer_accesses: bool = False
) -> LoopAccesses:
    """
    Summarise the memory accesses of the body of `loop`. Pointer accesses count
    as per-patch only with `prove_pointer_accesses` (see `_RowCheck`).
    """
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
                in_row: CheckedAccess | bool = False
                if prove_pointer_accesses and unit_outer_loop and len(targets) == 1:
                    in_row = _prove_in_row(loop, op, target)
                if isinstance(in_row, CheckedAccess):
                    accesses.checked.append(in_row)
                record(target, is_write, in_row is not False)
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
    tracked: set[SSAValue]
    """
    Memrefs that tasks get `depend` clauses on: those in `task_written`, and
    unaliased allocations freed while tasks may be in flight, so that the free
    can be ordered after the tasks reading them.
    """
    per_patch: set[SSAValue]
    """
    Memrefs whose tasks can depend on individual patches (outermost loop
    iterations) rather than on the whole buffer.
    """

    @staticmethod
    def build(
        loops: Iterable[scf.ParallelOp],
        freed: Iterable[SSAValue] = (),
        prove_pointer_accesses: bool = False,
    ) -> "DependenceInfo":
        """
        Analyse `loops`. `freed` are the memrefs deallocated by the thread
        creating the tasks whose deallocs may be deferred into tasks.
        """
        summaries = {loop: analyse_loop(loop, prove_pointer_accesses) for loop in loops}
        task_written: set[SSAValue] = set()
        task_accessed: set[SSAValue] = set()
        per_patch_votes: dict[SSAValue, bool] = {}
        for accesses in summaries.values():
            if accesses.opaque:
                continue
            task_written.update(accesses.writes)
            task_accessed.update(accesses.accessed())
            for target, per_patch in accesses.per_patch.items():
                per_patch_votes[target] = (
                    per_patch_votes.get(target, True) and per_patch
                )
        tracked = task_written | {
            target
            for target in freed
            if target in task_accessed and is_unaliased_allocation(target)
        }
        return DependenceInfo(
            summaries,
            task_written,
            tracked,
            {target for target, ok in per_patch_votes.items() if ok},
        )

    def runtime_check(
        self, loop: scf.ParallelOp
    ) -> tuple[list[Operation], SSAValue | None]:
        """
        The ops computing, before `loop`, the runtime check that its pointer
        accesses to per-patch memrefs stay within their patch, and the
        resulting i1; None if no check is needed. If the check fails, the
        loop's tasks must not overlap with any other task.
        """
        accesses = self.loops[loop]
        if accesses.opaque:
            return [], None  # its tasks are isolated anyway
        checked = [
            access
            for access in accesses.checked
            if access.target in self.per_patch and access.target in self.task_written
        ]
        if not checked:
            return [], None
        check = _RowCheck(loop, min(access.leaf_bits for access in checked), build=True)
        for access in checked:
            check.access(access.op, access.target)
        assert check.max_bits <= _MAX_BITS
        cond = check.result()
        return check.ops, cond

    def task_loops(self, target: SSAValue) -> list[scf.ParallelOp]:
        """
        The loops whose tasks access `target` and are ordered by dependences
        rather than waited for by a taskgroup.
        """
        return [
            loop
            for loop, accesses in self.loops.items()
            if not accesses.opaque and target in accesses.accessed()
        ]

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
