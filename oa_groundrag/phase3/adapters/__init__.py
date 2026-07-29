"""OA-LandslideDesc source adapters。"""

from .disasterm3 import DisasterM3Adapter
from .mmrs import MMRS1MAdapter
from .oa import OABenchmarkAdapter
from .rsgpt import RSGPTAdapter

__all__ = [
    "DisasterM3Adapter",
    "MMRS1MAdapter",
    "OABenchmarkAdapter",
    "RSGPTAdapter",
]
