"""Phase 2：光学锚定任意辅助模态滑坡分割。"""

from .contracts import (
    CandidateRegion,
    ModelRegistry,
    OAAuxSegBatch,
    OAAuxSegConfig,
    OAAuxSegOutput,
    PackedAuxiliary,
)
from .model import OAAuxSegModel

__all__ = [
    "CandidateRegion",
    "ModelRegistry",
    "OAAuxSegBatch",
    "OAAuxSegConfig",
    "OAAuxSegModel",
    "OAAuxSegOutput",
    "PackedAuxiliary",
]
