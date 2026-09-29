from xdsl.passes import ModulePass
from xdsl.context import Context
from xdsl.dialects import builtin, llvm

class StripVisibilityProperty(ModulePass):
    """
    Moves operations without side effects out of loops, provided they do not depend on
    values defined in the loops.
    """

    name = "strip-visibility-property"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for nested_op in op.walk():
            if isinstance(nested_op, llvm.GlobalOp):
                del nested_op.properties["visibility_"]