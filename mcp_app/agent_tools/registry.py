"""resolve_tools — dataset-to-KernelSession dispatch.

Same flat-dict-lookup pattern as eval/agent_tools/registry.py (not imported —
adapted fresh). A 4th dataset added later needs only one new entry here.
"""

from __future__ import annotations

from typing import Type

from .base import KernelSession
from .llama_cpp import LlamaCppKernelSession
from .ncnn import NCNNKernelSession
from .simd_loop import SIMDLoopKernelSession

_TOOLS: dict[str, Type[KernelSession]] = {
    "ncnn": NCNNKernelSession,
    "simd-loop": SIMDLoopKernelSession,
    "llama.cpp": LlamaCppKernelSession,
}


def resolve_tools(dataset: str) -> Type[KernelSession]:
    """Return the KernelSession class for the given dataset string.

    Raises KeyError with a clear message if the dataset is unknown.
    """
    cls = _TOOLS.get(dataset)
    if cls is None:
        raise KeyError(
            f"No KernelSession registered for dataset {dataset!r}. "
            f"Available: {sorted(_TOOLS)}"
        )
    return cls


__all__ = ["resolve_tools"]
