"""RS-GeneralDesc External source adapters。"""

from .disasterm3 import DisasterM3Adapter
from .mmrs import MMRS1MAdapter
from .rsgpt import RSGPTAdapter

__all__ = [
    "DisasterM3Adapter",
    "MMRS1MAdapter",
    "RSGPTAdapter",
]
