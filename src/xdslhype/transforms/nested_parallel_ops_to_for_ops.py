from xdsl.context import Context
from xdsl.dialects.scf import ParallelOp, ForOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern
)
from dataclasses import dataclass
from xdsl.dialects.builtin import ModuleOp

class TransformParallelOp(RewritePattern):
    def __init__(self, number_of_parallel_ops: int):
        self.number_of_parallel_ops = number_of_parallel_ops

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, ParallelOp):
            return
        
        self.transform_nested_parallel_ops(op, rewriter, 0)

    def transform_nested_parallel_ops(self, op: ParallelOp, rewriter: PatternRewriter, nested_level = 1):
        if nested_level == self.number_of_parallel_ops:
            self.transform_parallel_op_to_for(op, rewriter)
        else:
            for block in op.body.blocks:
                for nested_op in block.ops:
                    if isinstance(nested_op, ParallelOp):
                        self.transform_nested_parallel_ops(nested_op, rewriter, nested_level=nested_level + 1)
                        break
    
    def transform_parallel_op_to_for(self, op: ParallelOp, rewriter: PatternRewriter):
        for block in op.body.blocks:
            for nested_op in block.ops:
                if isinstance(nested_op, ParallelOp):
                    self.transform_parallel_op_to_for(nested_op, rewriter)
        loop = ForOp(op.lowerBound, op.upperBound, op.step, [], op.body.clone())
        rewriter.replace(op, loop)

@dataclass(frozen=True)
class NestedParallelOpsToForOps(ModulePass):
    name = "nested-parallel-ops-to-for-ops"
    number_of_parallel_ops: int

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(TransformParallelOp(self.number_of_parallel_ops)).rewrite_module(op)