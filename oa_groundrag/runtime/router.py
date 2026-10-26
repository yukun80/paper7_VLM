"""纯确定性 capability router；本模块不得导入模型或 torch。"""

from __future__ import annotations

from .contracts import (
    ExecutionPlan,
    RegionSource,
    ResponseKind,
    UnifiedRequest,
    UnifiedTask,
)


class CapabilityRouter:
    def route(self, request: UnifiedRequest) -> ExecutionPlan:
        request.validate_task_contract()
        task = request.task
        if task is UnifiedTask.VLM_ONLY:
            return ExecutionPlan(False, True, False, False, RegionSource.NONE, ResponseKind.TEXT)
        if task is UnifiedTask.SEGMENT_ONLY:
            return ExecutionPlan(True, False, False, False, RegionSource.NONE, ResponseKind.SPATIAL)
        if task is UnifiedTask.REGION_UNDERSTANDING:
            return ExecutionPlan(False, True, True, False, RegionSource.USER_MASK, ResponseKind.REGION_OBSERVATION)
        if task is UnifiedTask.SEGMENT_AND_UNDERSTAND:
            return ExecutionPlan(True, True, True, False, RegionSource.OA_AUXSEG_GLOBAL, ResponseKind.REGION_OBSERVATION)
        if task is UnifiedTask.KNOWLEDGE_QA:
            return ExecutionPlan(False, True, False, True, RegionSource.NONE, ResponseKind.KNOWLEDGE_ANSWER)
        if task is UnifiedTask.REGION_INTERPRETATION:
            return ExecutionPlan(
                request.region_source is RegionSource.OA_AUXSEG_CANDIDATE,
                True,
                True,
                True,
                request.region_source,
                ResponseKind.REGION_INTERPRETATION,
            )
        raise AssertionError(f"未覆盖 UnifiedTask：{task}")
