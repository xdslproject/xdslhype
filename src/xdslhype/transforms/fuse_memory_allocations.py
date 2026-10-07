from xdsl.dialects import builtin, scf, memref, arith, llvm, func
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern
)
from xdsl.rewriter import InsertPoint
from xdsl.builder import Builder
from dataclasses import dataclass
from xdsl.ir import Operation, SSAValue, BlockArgument

def collect_dependencies(val: SSAValue, op_dependencies: list[Operation], visited_ops: set[Operation]):
    if isinstance(val, BlockArgument):
        return

    op = val.owner
    if isinstance(op, Operation) and op not in visited_ops and not isinstance(op, llvm.GEPOp):
        visited_ops.add(op)
        for operand in op.operands:
            collect_dependencies(operand, op_dependencies, visited_ops)
        op_dependencies.append(op)
    

def create_cloned_dependency_ops(val: SSAValue):
    dependencies = []
    visited_ops = set()
    collect_dependencies(val, dependencies, visited_ops)

    value_mapper: dict[SSAValue, SSAValue] = {}
    cloned_ops: list[Operation] = []
    for nested_op in dependencies:
        cloned_ops.append(nested_op.clone(value_mapper=value_mapper))
        for old_res, new_res in zip(nested_op.results, cloned_ops[-1].results):
            value_mapper[old_res] = new_res

    return cloned_ops

@dataclass(frozen=True)
class FuseMemoryAllocations(ModulePass):
    name = "fuse-memory-allocations"

    """!
    FusememoryAllocations assumes that memrefs have been flattened. This is because
    trying to fuse allocations while they are still nested would fuse over each individual
    patch. If flattening was then performed each data block would not be stored contiguously.
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # Collect inner allocations
        example_allocation = None
        allocation_ops: list[Operation] = []
        allocation_ops.append(arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType())))

        offsets = {}
        inner_sizes = {}
        
        for nested_op in op.walk():
            if isinstance(nested_op, memref.AllocOp):
                if example_allocation is None:
                    example_allocation = nested_op

                inner_size = nested_op.operands[1]
                outer_size = nested_op.operands[0]
                inner_dependencies = create_cloned_dependency_ops(inner_size)
                outer_dependencies = create_cloned_dependency_ops(outer_size)
                assert(isinstance(inner_size.type, builtin.IndexType))
                assert(isinstance(outer_size.type, builtin.IndexType))
                size_op = arith.MuliOp(inner_dependencies[-1].result, outer_dependencies[-1].result)

                offsets[nested_op.memref] = allocation_ops[-1].result
                inner_sizes[nested_op.memref] = inner_dependencies[-1].result

                size_update_op = arith.AddiOp(size_op.result, allocation_ops[-1].result)
                allocation_ops.extend(inner_dependencies)
                allocation_ops.extend(outer_dependencies)
                allocation_ops.append(size_op)
                allocation_ops.append(size_update_op)

        if example_allocation is None:
            return
        
        allocation_ops.append(memref.AllocOp(allocation_ops[-1].result, [], builtin.MemRefType(builtin.Float64Type(), [builtin.IntAttr(builtin.DYNAMIC_INDEX)])))

        assert(isinstance(allocation_ops[-1], memref.AllocOp))
        fused_memref = allocation_ops[-1].memref
        
        insert_point = None
        for nested_op in op.walk():
            if isinstance(nested_op, builtin.UnrealizedConversionCastOp):
                insert_point = nested_op
        assert(insert_point is not None)
        builder = Builder(InsertPoint.after(insert_point))
        for nested_op in allocation_ops:
            builder.insert(nested_op)

        for nested_op in op.walk():
            if isinstance(nested_op, memref.LoadOp) and nested_op.memref in offsets.keys():
                offset_multiplication = arith.MuliOp(nested_op.indices[0], inner_sizes[nested_op.memref])
                index = arith.AddiOp(offset_multiplication.result, offsets[nested_op.memref])
                final_index = arith.AddiOp(index.result, nested_op.indices[1])
                new_load = memref.LoadOp.get(fused_memref, [final_index.result])
                builder = Builder(InsertPoint.before(nested_op))
                builder.insert(offset_multiplication)
                builder.insert(index)
                builder.insert(final_index)
                nested_op.results[0].replace_all_uses_with(new_load.results[0])
                builder.insert(new_load)
                nested_op.detach()
                nested_op.erase()
            if isinstance(nested_op, memref.StoreOp) and nested_op.memref in offsets.keys():
                offset_multiplication = arith.MuliOp(nested_op.indices[0], inner_sizes[nested_op.memref])
                index = arith.AddiOp(offset_multiplication.result, offsets[nested_op.memref])
                final_index = arith.AddiOp(index.result, nested_op.indices[1])
                new_store = memref.StoreOp.get(nested_op.value, fused_memref, [final_index.result])
                #new_store = memref.LoadOp.get(fused_memref, [final_index.result])
                builder = Builder(InsertPoint.before(nested_op))
                builder.insert(offset_multiplication)
                builder.insert(index)
                builder.insert(final_index)
                #nested_op.results[0].replace_all_uses_with(new_store.results[0])
                builder.insert(new_store)
                nested_op.detach()
                nested_op.erase()
            if isinstance(nested_op, memref.ExtractAlignedPointerAsIndexOp) and nested_op.source in offsets.keys():
                new_op = memref.ExtractAlignedPointerAsIndexOp.get(fused_memref)
                builder = Builder(InsertPoint.before(nested_op))
                builder.insert(new_op)
                c8 = arith.ConstantOp(builtin.IntegerAttr(8, builtin.IndexType()))
                offset = arith.MuliOp(offsets[nested_op.source], c8.result)
                offset_pointer = arith.AddiOp(new_op.aligned_pointer, offset.result)
                builder.insert(c8)
                builder.insert(offset)
                builder.insert(offset_pointer)
                nested_op.aligned_pointer.replace_all_uses_with(offset_pointer.result)
                nested_op.detach()
                nested_op.erase()

        # Delete all unnecessary allocs and deallocs
        for nested_op in op.walk():
            if isinstance(nested_op, memref.DeallocOp):
                nested_op.detach()
                nested_op.erase()
        
        num_allocs = 0
        for nested_op in op.walk():
            if isinstance(nested_op, memref.AllocOp):
                num_allocs = num_allocs + 1
                if num_allocs > 1:
                    nested_op.detach()
                    nested_op.erase()

        for nested_op in op.walk():
            if isinstance(nested_op, func.ReturnOp):
                insert_point = nested_op
        builder = Builder(InsertPoint.before(insert_point))
        builder.insert(memref.DeallocOp.get(fused_memref))