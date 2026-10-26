"""OA-GroundRAG 研究代码。"""

from .segmentation.contracts import OAAuxSegBatch, OAAuxSegOutput
from .segmentation.model import OAAuxSegModel

__all__ = ["OAAuxSegBatch", "OAAuxSegModel", "OAAuxSegOutput"]
