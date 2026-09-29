from xdsl.universe import Universe

from xdslhype.dialects import get_all_dialects
from xdslhype.transforms import get_all_passes

UNIVERSE = Universe(
    all_dialects=get_all_dialects(),
    all_passes=get_all_passes(),
)
