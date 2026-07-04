"""resolve_tools — dataset-to-AgentTools dispatch.

Simple dict lookup; no priority ordering needed since datasets are disjoint.
"""

from __future__ import annotations

from typing import Type

from .base import AgentTools
from .llama_cpp import LlamaCppAgentTools
from .ncnn import NCNNAgentTools
from .simd_loop import SIMDLoopAgentTools

_TOOLS: dict[str, Type[AgentTools]] = {
    "ncnn": NCNNAgentTools,
    "simd-loop": SIMDLoopAgentTools,
    "llama.cpp": LlamaCppAgentTools,
}


def resolve_tools(dataset: str) -> Type[AgentTools]:
    """Return the AgentTools class for the given dataset string.

    Raises KeyError with a clear message if the dataset is unknown.
    """
    cls = _TOOLS.get(dataset)
    if cls is None:
        raise KeyError(
            f"No AgentTools registered for dataset {dataset!r}. "
            f"Available: {sorted(_TOOLS)}"
        )
    return cls


__all__ = ["resolve_tools"]
