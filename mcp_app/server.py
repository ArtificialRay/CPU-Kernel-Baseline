"""mcp_app.server — the MCP server process: one per (instance, dataset) session.

Registers compile/evaluate/disassemble/submit as MCP tools (no `read_code` —
retired, see mcp_app/agent_tools/base.py) and the session's trajectory files
as MCP Resources (mcp_app/resources.py), backed by mcp_app/agent_tools's
in-process KernelSession (this process runs directly on the target instance).
`compile` takes `definition` as a per-call argument — one server process can
compile/evaluate/submit many definitions across the same dataset without
restarting; see agent_tools/base.py's KernelSession.

Built on the low-level `mcp.server.lowlevel.Server` rather than FastMCP:
tool_schemas() already produces ready-made JSON Schema (no need to re-derive
it from a typed function signature), and resource listing must be dynamic
since vN.cpp/vN.s files appear mid-session as the agent compiles more
versions — a data-driven list_tools/call_tool/list_resources/read_resource
handler set maps onto both needs directly.

Usage:
    python -m mcp_app.server --dataset ncnn --author test --isa sve2 \\
        --run-dir <path>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from . import resources as resources_mod
from .agent_tools import isa as isa_mod
from .agent_tools.base import KernelSession
from .session import SessionConfig, build_tools


def build_server(tools: KernelSession) -> Server:
    server: Server = Server("armbench-kernel-session")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=s["name"],
                description=s["description"],
                inputSchema=s["parameters"],
            )
            for s in tools.tool_schemas()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> dict[str, Any]:
        return tools.dispatch_tool_call(name, arguments)

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return resources_mod.list_run_dir_resources(tools._run_dir)

    @server.read_resource()
    async def _read_resource(uri: Any) -> str:
        return resources_mod.read_run_dir_resource(tools._run_dir, str(uri))

    return server


async def _run_stdio(server: Server) -> None:
    import mcp.server.stdio as stdio

    async with stdio.stdio_server() as (read_stream, write_stream):
        # stdout is the JSON-RPC channel for stdio transport — never print there.
        # stderr is the MCP-conventional channel for server diagnostics/logging.
        print("[mcp_app.server] MCP server ready (stdio transport).", file=sys.stderr, flush=True)
        await server.run(read_stream, write_stream, server.create_initialization_options())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    p.add_argument("--author", required=True)
    p.add_argument("--baseline-author", default=None,
                    help="Override only — auto-derived from --dataset by default "
                         "(see agent_tools/baseline_readiness.py::DEFAULT_BASELINE_AUTHOR).")
    p.add_argument("--isa", required=True, choices=sorted(isa_mod.SUPPORTED_ISAS),
                    help="Explicit, never auto-detected — drives compile flags deterministically.")
    p.add_argument("--bench-trace-root", default="bench-trace",
                    help="Relative to cwd by default (server is launched from the repo root).")
    p.add_argument("--run-dir", required=True,
                    help="Session root, e.g. <remote_root>/agent-runs-mcp/<author> — each "
                         "definition compile()'d gets its own <run-dir>/<definition>/ subdir.")
    p.add_argument("--instance-label", default=None,
                    help="Cosmetic only (e.g. 'c8g.large') — never used for compile-flag decisions.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = SessionConfig(
        dataset=args.dataset,
        author=args.author,
        baseline_author=args.baseline_author,
        isa=args.isa,
        bench_trace_root=Path(args.bench_trace_root),
        run_dir=Path(args.run_dir),
        instance_label=args.instance_label,
    )
    print(f"[mcp_app.server] Initializing session (dataset={args.dataset!r}, "
          f"isa={args.isa!r}, author={args.author!r})...", file=sys.stderr, flush=True)
    tools = build_tools(cfg)
    print("[mcp_app.server] Session initialized.", file=sys.stderr, flush=True)
    server = build_server(tools)
    try:
        asyncio.run(_run_stdio(server))
    finally:
        tools.cleanup()


if __name__ == "__main__":
    main()
