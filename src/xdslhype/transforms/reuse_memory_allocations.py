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
from queue import Queue

def determine_max_live_memrefs(op: builtin.ModuleOp):
    live_memrefs = set()
    allocated_memrefs = set()
    max_live_memrefs = 0

    for nested_op in op.walk():
        if isinstance(nested_op, memref.AllocOp):
            if not isinstance(nested_op.parent.parent.parent, scf.ForOp):
                for result in nested_op.results:
                    allocated_memrefs.add(result)
                    outer_memref_type = result.type
            else:
                for result in nested_op.results:
                    inner_memref_type = result.type

    for nested_op in op.walk(reverse=True):
        # Here we are interested in which outer memrefs are being allocated
        if isinstance(nested_op, memref.AllocOp) and not isinstance(nested_op.parent.parent.parent, scf.ForOp):
            for result in nested_op.results:
                live_memrefs.discard(result)

        # Here we are interested in which outer memrefs are being accessed
        if isinstance(nested_op, scf.ParallelOp):
            is_innermost_loop = True
            for loop_op in nested_op.body.blocks[0].ops:
                if isinstance(loop_op, scf.ParallelOp):
                    is_innermost_loop = False

            if is_innermost_loop:
                for loop_op in nested_op.body.blocks[0].ops:
                    if isinstance(loop_op, memref.LoadOp) or isinstance(loop_op, memref.LoadOp):
                        for operand in loop_op.operands:
                            if operand.type == outer_memref_type:
                                live_memrefs.add(operand)

        current_live_memrefs = len(live_memrefs)
        if current_live_memrefs > max_live_memrefs:
            max_live_memrefs = current_live_memrefs

    return outer_memref_type, inner_memref_type, max_live_memrefs, allocated_memrefs

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
class ResuseMemoryAllocations(ModulePass):
    name = "reuse-memory-allocations"
    max_allocation_size: int

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        outer_memref_type, inner_memref_type, max_live_memrefs, allocated_memrefs = determine_max_live_memrefs(op)

        # Create new memory allocations
        for nested_op in op.walk():
            if isinstance(nested_op, memref.AllocOp):
                example_outer_allocation = nested_op
                break

        for nested_op in op.walk():
            if isinstance(nested_op, scf.ForOp):
                example_inner_allocation_loop = nested_op

        size_constant_integer = arith.ConstantOp(builtin.IntegerAttr(self.max_allocation_size, 64))
        size_constant_index = arith.IndexCastOp(size_constant_integer.result, builtin.IndexType())
        allocation: list[Operation] = [size_constant_integer, size_constant_index]

        scratchpad_memrefs = []

        
        for i in range(max_live_memrefs):
            
            lower_bound_dependencies = create_cloned_dependency_ops(example_inner_allocation_loop.lb)
            lower_bound_operation = lower_bound_dependencies[-1]
            upper_bound_dependencies = create_cloned_dependency_ops(example_inner_allocation_loop.ub)
            upper_bound_operation = upper_bound_dependencies[-1]
            step_dependencies = create_cloned_dependency_ops(example_inner_allocation_loop.step)
            step_operation = step_dependencies[-1]

            outer_allocation_dependencies = []
            outer_allocation_operands = []
            for operand in example_outer_allocation.operands:
                operand_dependencies = create_cloned_dependency_ops(operand)
                outer_allocation_dependencies.extend(operand_dependencies)
                outer_allocation_operands.append(operand_dependencies[-1])
            outer_allocation = memref.AllocOp(outer_allocation_operands, [], outer_memref_type)
            scratchpad_memrefs.append(outer_allocation.memref)

            inner_allocation_loop_body = Block(arg_types=[builtin.IndexType()])
            inner_allocation = memref.AllocOp([size_constant_index.result], [], inner_memref_type)
            inner_allocation_loop_body.add_op(inner_allocation)
            inner_allocation_loop_body.add_op(memref.StoreOp.get(inner_allocation.memref, outer_allocation.memref, [inner_allocation_loop_body.args[0]]))
            inner_allocation_loop_body.add_op(scf.YieldOp())

            inner_allocation_loop = scf.ForOp(lower_bound_operation.results[0], upper_bound_operation.results[0], step_operation.results[0], [], inner_allocation_loop_body)

            allocation.extend(
                lower_bound_dependencies + 
                upper_bound_dependencies +
                step_dependencies +
                outer_allocation_dependencies +
                [
                    outer_allocation,
                    inner_allocation_loop,
                ]
            )

        for nested_op in op.walk():
            if isinstance(nested_op, builtin.UnrealizedConversionCastOp):
                insert_point = nested_op
        
        builder = Builder(InsertPoint.after(insert_point))
        for nested_op in allocation:
            builder.insert(nested_op)


        inuse_memrefs = dict()

        # Replace existing memref references with new ones
        for nested_op in op.walk(reverse=True):
            if isinstance(nested_op, scf.ParallelOp):
                is_innermost_loop = True
                for loop_op in nested_op.body.blocks[0].ops:
                    if isinstance(loop_op, scf.ParallelOp):
                        is_innermost_loop = False
    
                if is_innermost_loop:
                    for innermost_op in nested_op.walk():
                        if isinstance(innermost_op, memref.LoadOp):
                            if innermost_op.memref.type == outer_memref_type and innermost_op.memref in allocated_memrefs:
                                for i, scratchpad_memref in enumerate(scratchpad_memrefs):
                                    if scratchpad_memref not in inuse_memrefs.keys() and scratchpad_memref not in inuse_memrefs.values():
                                        inuse_memrefs.update({scratchpad_memref: innermost_op.memref})
                                        innermost_op.memref.replace_all_uses_with(scratchpad_memref)
                                        break

                                #scratchpad_memref = available_scratchpad_memrefs.get()
                        elif isinstance(innermost_op, memref.StoreOp):
                            if innermost_op.memref.type == outer_memref_type and innermost_op.memref in allocated_memrefs:
                                #for scratchpad_memref in scratchpad_memrefs:
                                    #if scratchpad_memref not in inuse_memrefs:
                                innermost_op.memref.replace_all_uses_with(scratchpad_memrefs[0])
                                        #inuse_memrefs.add(scratchpad_memref)
            elif isinstance(nested_op, memref.AllocOp) and nested_op.memref.type == outer_memref_type:
                key_to_delete = None
                for key, value in inuse_memrefs.items():
                    if key == nested_op.memref or value == nested_op.memref:
                        key_to_delete = key
                if key_to_delete is not None:
                    del inuse_memrefs[key_to_delete]
        


        # Delete all unnecessary allocs and deallocs
        num_fors = 0
        num_allocs = 0
        for nested_op in op.walk():
            if isinstance(nested_op, scf.ForOp):
                num_fors = num_fors + 1
                if num_fors > max_live_memrefs:
                    nested_op.detach()
                    nested_op.erase()
            if isinstance(nested_op, memref.AllocOp) and nested_op.memref.type == outer_memref_type:
                num_allocs = num_allocs + 1
                if num_allocs > max_live_memrefs:
                    nested_op.detach()
                    nested_op.erase()
            if isinstance(nested_op, memref.DeallocOp):
                nested_op.detach()
                nested_op.erase()

        # Create deallocs for scratchpad memrefs
        for nested_op in op.walk():
            if isinstance(nested_op, func.ReturnOp):
                insert_point = nested_op
        builder = Builder(InsertPoint.before(insert_point))
        for scratchpad_memref in scratchpad_memrefs:
            builder.insert(memref.DeallocOp.get(scratchpad_memref))