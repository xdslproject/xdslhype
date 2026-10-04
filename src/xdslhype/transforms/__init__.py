from collections.abc import Callable

from xdsl.passes import ModulePass
from .nested_parallel_ops_to_for_ops import NestedParallelOpsToForOps
from .imperfect_nested_loops_to_perfect import ImperfectNestedLoopsToPerfect
from .nested_parallel_ops_to_multidimensional import NestedParallelOpsToMultidimensional
from .reuse_memory_allocations import ResuseMemoryAllocations
from .strip_visibility_property import StripVisibilityProperty
from .convert_scf_to_omp_tasks import ConvertScfToOmpTasks
from .scf_parallel_loop_fusion import ScfParallelLoopFusion

def get_transform_parallel_ops_to_for_pass():
    return NestedParallelOpsToForOps

def get_imperfect_nested_loops_to_perfect_pass():
    return ImperfectNestedLoopsToPerfect

def get_nested_parallel_ops_to_multidimensional_pass():
    return NestedParallelOpsToMultidimensional

def get_reuse_memory_allocations_pass():
    return ResuseMemoryAllocations

def get_strip_visibility_property_pass():
    return StripVisibilityProperty

def get_convert_scf_to_omp_tasks_pass():
    return ConvertScfToOmpTasks

def get_scf_parallel_loop_fusion_pass():
    return ScfParallelLoopFusion

def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    # Add your passes here to be discovered by `xdsl-opt`
    return {
        "nested-parallel-ops-to-for-ops": get_transform_parallel_ops_to_for_pass,
        "imperfect-nested-loops-to-perfect": get_imperfect_nested_loops_to_perfect_pass,
        "nested-parallel-ops-to-multidimensional": get_nested_parallel_ops_to_multidimensional_pass,
        "strip-visibility-property": get_strip_visibility_property_pass,
        "reuse-memory-allocations": get_reuse_memory_allocations_pass,
        "convert-scf-to-omp-tasks": get_convert_scf_to_omp_tasks_pass,
        "scf-parallel-loop-fusion": get_scf_parallel_loop_fusion_pass,
    }
