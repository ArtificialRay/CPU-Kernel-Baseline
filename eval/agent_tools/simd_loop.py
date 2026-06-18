"""SIMDLoopAgentTools — placeholder for the simd-loop dataset.

Not yet implemented. Raises NotImplementedError on all abstract methods so
that the registry rejects "simd-loop" dataset requests early rather than
silently producing wrong behaviour.
"""

from __future__ import annotations

from bench.data.solution import Solution

from .base import AgentTools


class SIMDLoopAgentTools(AgentTools):
    """Placeholder — simd-loop agent tool support not yet implemented."""

    dataset = "simd-loop"

    @classmethod
    def can_handle(cls, dataset: str) -> bool:
        return dataset == "simd-loop"

    def make_solution(self, code: str) -> Solution:
        raise NotImplementedError(
            "SIMDLoopAgentTools is not yet implemented. "
            "Use NCNNAgentTools for ncnn-dataset definitions."
        )

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        raise NotImplementedError("SIMDLoopAgentTools is not yet implemented.")


__all__ = ["SIMDLoopAgentTools"]
