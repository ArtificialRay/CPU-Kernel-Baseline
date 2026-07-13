"""mcp_app.server — the MCP server process: one per (instance, definition) session.

Registers compile/evaluate/disassemble/submit as MCP tools (no `read_code` —
retired, see mcp_app/agent_tools/base.py) and the session's trajectory files
as MCP Resources (mcp_app/resources.py), backed by mcp_app/agent_tools's
in-process KernelSession (this process runs directly on the target instance).

Built on the low-level `mcp.server.lowlevel.Server` rather than FastMCP:
tool_schemas() already produces ready-made JSON Schema (no need to re-derive
it from a typed function signature), and resource listing must be dynamic
since vN.cpp/vN.s files appear mid-session as the agent compiles more
versions — a data-driven list_tools/call_tool/list_resources/read_resource
handler set maps onto both needs directly.

Usage:
    python -m mcp_app.server --dataset ncnn --definition <name> --author test \\
        --baseline-author baseline-ncnn-arm --isa sve2 --run-dir <path> \\
        --transport stdio
"""

from __future__ import annotations

import argparse
import asyncio
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
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _run_sse(server: Server, bind_host: str, port: int) -> None:
    """Fallback transport — see mcp_app/README.md: only build/exercise this if
    stdio-over-ssh turns out not to work with nanobot's real MCP config format.
    """
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=transport.handle_post_message),
    ])
    config = uvicorn.Config(app, host=bind_host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    p.add_argument("--definition", required=True, dest="definition_name")
    p.add_argument("--author", required=True)
    p.add_argument("--baseline-author", required=True,
                    help="No internal dataset->author mapping — caller must know this.")
    p.add_argument("--isa", required=True, choices=sorted(isa_mod.SUPPORTED_ISAS),
                    help="Explicit, never auto-detected — drives compile flags deterministically.")
    p.add_argument("--bench-trace-root", default="bench-trace",
                    help="Relative to cwd by default (server is launched from the repo root).")
    p.add_argument("--run-dir", required=True,
                    help="Where trajectory files land, e.g. <remote_root>/agent-runs-mcp/<definition>.")
    p.add_argument("--instance-label", default=None,
                    help="Cosmetic only (e.g. 'c8g.large') — never used for compile-flag decisions.")
    p.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = SessionConfig(
        definition_name=args.definition_name,
        dataset=args.dataset,
        author=args.author,
        baseline_author=args.baseline_author,
        isa=args.isa,
        bench_trace_root=Path(args.bench_trace_root),
        run_dir=Path(args.run_dir),
        instance_label=args.instance_label,
    )
    tools = build_tools(cfg)
    server = build_server(tools)
    try:
        if args.transport == "stdio":
            asyncio.run(_run_stdio(server))
        else:
            asyncio.run(_run_sse(server, args.bind_host, args.port))
    finally:
        tools.cleanup()


if __name__ == "__main__":
    main()
