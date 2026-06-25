"""eval.agent_tools — per-dataset tool surfaces backed by the bench/ harness.

Usage:
    from eval.agent_tools import AgentTools, resolve_tools

    ToolsCls = resolve_tools(dataset)        # "ncnn" → NCNNAgentTools
    tools = ToolsCls(pipeline, definition, trace_set, author="claude-opus-4-8")

    result = tools.compile(code)             # SSH → remote_runner compile
    result = tools.evaluate()               # SSH → remote_runner evaluate
    result = tools.disassemble()            # SSH → remote_runner disassemble
    result = tools.submit(code, explanation) # compile + evaluate + persist

    # Or via LLM tool-call dispatch:
    result = tools.dispatch_tool_call("compile", {"code": "..."})

    tools.cleanup()                          # close trajectory writer, rm build dirs
"""

from .base import AgentTools
from .ncnn import NCNNAgentTools
from .registry import resolve_tools
from .simd_loop import SIMDLoopAgentTools
from .trajectory import TrajectoryWriter

__all__ = [
    "AgentTools",
    "NCNNAgentTools",
    "SIMDLoopAgentTools",
    "TrajectoryWriter",
    "resolve_tools",
]
