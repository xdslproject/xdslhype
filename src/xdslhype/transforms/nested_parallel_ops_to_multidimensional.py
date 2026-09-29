from xdsl.context import Context
from xdsl.dialects import builtin, scf
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
class ConvertNestedParallelOpsToMultidimensional(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter):
        # Check if loop is perfectly nested
        # For example, loops that reduce over a value won't be
        inner_loop = op
        loop_contains_nested_parallel = True
        while loop_contains_nested_parallel:
            loop_contains_non_parallel_op = False
            loop_contains_nested_parallel = False
            for block in inner_loop.body.blocks:
                for nested_op in block.ops:
                    if not isinstance(nested_op, scf.ParallelOp) and not isinstance(nested_op, scf.ReduceOp):
                        loop_contains_non_parallel_op = True
                    if isinstance(nested_op, scf.ParallelOp):
                        inner_loop = nested_op
                        loop_contains_nested_parallel = True
            if loop_contains_non_parallel_op and loop_contains_nested_parallel:
                # Loop is not perfectly nested
                return


        lower_bounds = []
        upper_bounds = []
        steps = []

        loops_args = []
        inner_loop = op
        loop_contains_nested_parallel = True
        while loop_contains_nested_parallel:
            loop_contains_nested_parallel = False
            loops_args.append(inner_loop.body.blocks[0].args)
            lower_bounds.append(inner_loop.operands[0])
            upper_bounds.append(inner_loop.operands[1])
            steps.append(inner_loop.operands[2])

            for block in inner_loop.body.blocks:
                for nested_op in block.ops:
                    if isinstance(nested_op, scf.ParallelOp):
                        inner_loop = nested_op
                        loop_contains_nested_parallel = True

        new_arg_types = []
        for args in loops_args:
            for arg in args:
                new_arg_types.append(arg.type)
        new_block = Block(arg_types=new_arg_types)

        i = 0
        for arg_list in loops_args:
            for old_arg in arg_list:
                old_arg.replace_all_uses_with(new_block.args[i])
                i = i + 1

        for inner_op in inner_loop.body.blocks[0].ops:
            inner_op.detach()
            new_block.add_op(inner_op)
        
        new_parallel_op = scf.ParallelOp(lower_bounds, upper_bounds, steps, [new_block])
        rewriter.replace(op, new_parallel_op)

class NestedParallelOpsToMultidimensional(ModulePass):

    name = "nested-parallel-ops-to-multidimensional"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertNestedParallelOpsToMultidimensional(),
            apply_recursively=False
        ).rewrite_module(op)
        