"""mcp_app.agent_tools — per-dataset tool surfaces backed by bench/ in-process.

Self-contained: zero imports from eval/agent_tools/ (see mcp_app/README.md).
Adapted from eval/agent_tools/, with the SSH abstraction dropped since
execution is always local to the machine this code runs on.

Usage:
    from mcp_app.agent_tools import resolve_tools

    ToolsCls = resolve_tools(dataset)        # "ncnn" -> NCNNKernelSession
    tools = ToolsCls(definition, trace_set, author, bench_cfg, run_dir, isa)

    result = tools.compile(code)
    result = tools.evaluate()
    result = tools.disassemble()
    result = tools.submit(explanation)

    # Or via tool-call dispatch (what mcp_app/server.py uses):
    result = tools.dispatch_tool_call("compile", {"code": "..."})

    tools.cleanup()
"""

from .base import KernelSession
from .llama_cpp import LlamaCppKernelSession
from .ncnn import NCNNKernelSession
from .registry import resolve_tools
from .simd_loop import SIMDLoopKernelSession

__all__ = [
    "KernelSession",
    "NCNNKernelSession",
    "SIMDLoopKernelSession",
    "LlamaCppKernelSession",
    "resolve_tools",
]
