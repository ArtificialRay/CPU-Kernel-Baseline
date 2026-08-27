"""mcp_app.server — the MCP server process: one per (instance, dataset) session.

Registers compile/evaluate/disassemble/submit as MCP tools and the session's
trajectory files as MCP Resources (mcp_app/resources.py), backed by
mcp_app/agent_tools's in-process KernelSession (this process runs directly on
the target instance). `compile` takes `definition` as a per-call argument —
one server process can compile/evaluate/submit many definitions across the
same dataset without restarting; see agent_tools/base.py's KernelSession.

Usage:
    python -m mcp_app.server --dataset ncnn --author test --isa sve2 \\
        --run-dir <path> --transport stdio

Two transports:
- stdio: harness and server share a host (spawn-command integration —
  local dev, or already SSHed into the target instance). Most broadly
  supported by MCP client configs since it needs no network stack.
- streamable-http: harness reaches a server on a different host (the
  normal case — server runs on a provisioned Graviton instance, harness
  runs wherever the user's agent lives). skills/launch/launch_session.py
  reaches this over an SSH local-port-forward rather than exposing the
  port publicly — the compile/evaluate tool surface is effectively remote
  code execution.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp.server.lowlevel import Server

from . import resources as resources_mod
from .agent_tools import isa as isa_mod
from .agent_tools.base import KernelSessionLike
from .agent_tools.dispatcher import DispatcherKernelSession
from .session import SessionConfig, build_tools

if TYPE_CHECKING:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

# compile()/evaluate() run synchronously and can take minutes, set a tool call ping to notify agent that the tool is still running
TOOL_CALL_PING_INTERVAL_S = 120

# Debug-log truncation: 
ARG_LOG_TRUNCATE_CHARS = 300
RESULT_LOG_TRUNCATE_CHARS = 2000


def _truncate_repr(obj: Any, limit: int) -> str:
    text = repr(obj)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit} more chars truncated)"


def _elog(msg: str) -> None:
    """Write a diagnostic to stderr, swallowing I/O errors.

    If the server's stderr consumer goes away (closed terminal, a dead tmux
    session whose pipe we inherited, a rotated log fd), a bare
    ``print(..., file=sys.stderr)`` raises ``BrokenPipeError``. In ``_call_tool``
    that write is the *first* thing every tool call does, so an unguarded write
    turns a dead stderr into "every compile/evaluate fails with [Errno 32]
    Broken pipe while resource reads keep working" — a baffling partial outage.
    Diagnostics must never be able to fail a tool call, so swallow the error.
    """
    with contextlib.suppress(BrokenPipeError, OSError):
        print(msg, file=sys.stderr, flush=True)


def build_server(tools: KernelSessionLike) -> Server:
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
        # Run the (synchronous, potentially long-running) dispatch off the event
        # loop so pings can still go out while it's in flight — dispatch_tool_call
        # itself stays untouched, it doesn't need to know about sessions/pings.
        _elog(f"[mcp_app.server] tool call: {name}({_truncate_repr(arguments, ARG_LOG_TRUNCATE_CHARS)})")
        started = time.monotonic()
        session = server.request_context.session
        task = asyncio.ensure_future(
            asyncio.to_thread(tools.dispatch_tool_call, name, arguments)
        )
        while True:
            done, _ = await asyncio.wait({task}, timeout=TOOL_CALL_PING_INTERVAL_S)
            if task in done:
                break
            _elog(f"[mcp_app.server] tool '{name}' still running "
                  f"({time.monotonic() - started:.0f}s elapsed)...")
            with contextlib.suppress(Exception):
                await session.send_ping()
        result = task.result()
        _elog(f"[mcp_app.server] tool '{name}' done in {time.monotonic() - started:.1f}s: "
              f"{_truncate_repr(result, RESULT_LOG_TRUNCATE_CHARS)}")
        # Resource-visibility bookkeeping only (see resources_mod / KernelSession
        # .note_session_definition) — never consulted by dispatch_tool_call's own
        # definition-match guard, so this can't affect tool-call correctness.
        if name == "compile" and result.get("status") == "OK":
            tools.note_session_definition(session, arguments["definition"])
        return result

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        session = server.request_context.session
        return resources_mod.list_run_dir_resources(
            tools._run_dir, visible_definitions=tools.session_definitions(session),
        )

    @server.read_resource()
    async def _read_resource(uri: Any) -> str:
        session = server.request_context.session
        return resources_mod.read_run_dir_resource(
            tools._run_dir, str(uri), visible_definitions=tools.session_definitions(session),
        )

    return server


async def _run_stdio(server: Server) -> None:
    import mcp.server.stdio as stdio

    async with stdio.stdio_server() as (read_stream, write_stream):
        # stdout is the JSON-RPC channel for stdio transport — never print there.
        # stderr is the MCP-conventional channel for server diagnostics/logging.
        print("[mcp_app.server] MCP server ready (stdio transport).", file=sys.stderr, flush=True)
        await server.run(read_stream, write_stream, server.create_initialization_options())


class _StreamableHTTPASGIApp:
    """Thin ASGI wrapper so Starlette's Route treats this as an already-ASGI
    endpoint (scope, receive, send) rather than wrapping it as a
    request->response function — a plain `async def` here would get the
    wrong calling convention (see mcp.server.fastmcp.server's equivalent)."""

    def __init__(self, session_manager: "StreamableHTTPSessionManager") -> None:
        self._session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:
        await self._session_manager.handle_request(scope, receive, send)


async def _run_streamable_http(server: Server, bind_host: str, port: int) -> None:
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Route

    # session_idle_timeout reaps sessions abandoned by a dropped tunnel/client
    # without waiting on process teardown to free the KernelSession/compile cache.
    session_manager = StreamableHTTPSessionManager(app=server, session_idle_timeout=1800)
    asgi_app = _StreamableHTTPASGIApp(session_manager)

    app = Starlette(
        routes=[Route("/mcp", endpoint=asgi_app)],
        lifespan=lambda _app: session_manager.run(),
    )
    config = uvicorn.Config(app, host=bind_host, port=port, log_level="warning")
    print("[mcp_app.server] MCP server ready (streamable-http transport).", file=sys.stderr, flush=True)
    await uvicorn.Server(config).serve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", required=True,
                    choices=["ncnn", "simd-loop", "llama.cpp"],
                    help="Repeatable — pass more than once (e.g. --dataset ncnn "
                         "--dataset llama.cpp) to serve several datasets from one "
                         "process, dispatching each tool call by which dataset its "
                         "`definition` belongs to (see agent_tools/dispatcher.py). "
                         "A single --dataset behaves exactly as before.")
    p.add_argument("--author", required=True)
    p.add_argument("--baseline-author", default=None,
                    help="Override only — auto-derived from --dataset by default "
                         "(see agent_tools/baseline_readiness.py::DEFAULT_BASELINE_AUTHOR). "
                         "Not allowed together with multiple --dataset values, since a "
                         "single override can't correctly apply to more than one dataset.")
    p.add_argument("--isa", required=True, choices=sorted(isa_mod.SUPPORTED_ISAS),
                    help="Explicit, never auto-detected — drives compile flags deterministically.")
    p.add_argument("--bench-trace-root", default="bench-trace",
                    help="Relative to cwd by default (server is launched from the repo root).")
    p.add_argument("--run-dir", required=True,
                    help="Session root, e.g. <remote_root>/agent-runs-mcp/<author> — each "
                         "definition compile()'d gets its own <run-dir>/<definition>/ subdir.")
    p.add_argument("--instance-label", default=None,
                    help="Cosmetic only (e.g. 'c8g.large') — never used for compile-flag decisions.")
    p.add_argument("--transport", choices=["stdio", "streamable-http"])
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    args.dataset = list(dict.fromkeys(args.dataset))  # dedupe, preserve order
    if len(args.dataset) > 1 and args.baseline_author is not None:
        p.error(
            "--baseline-author can't be used with multiple --dataset values — "
            "each dataset auto-derives its own (see DEFAULT_BASELINE_AUTHOR)."
        )
    return args


def _build_session_config(args: argparse.Namespace, dataset: str, *, baseline_author) -> SessionConfig:
    return SessionConfig(
        dataset=dataset,
        author=args.author,
        baseline_author=baseline_author,
        isa=args.isa,
        bench_trace_root=Path(args.bench_trace_root),
        run_dir=Path(args.run_dir),
        instance_label=args.instance_label,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets: list[str] = args.dataset

    print(f"[mcp_app.server] Initializing session (dataset(s)={datasets!r}, "
          f"isa={args.isa!r}, author={args.author!r})...", file=sys.stderr, flush=True)

    tools: KernelSessionLike
    if len(datasets) == 1:
        cfg = _build_session_config(args, datasets[0], baseline_author=args.baseline_author)
        tools = build_tools(cfg)
    else:
        sessions = {
            dataset: build_tools(_build_session_config(args, dataset, baseline_author=None))
            for dataset in datasets
        }
        tools = DispatcherKernelSession(sessions, run_dir=Path(args.run_dir))

    print("[mcp_app.server] Session initialized.", file=sys.stderr, flush=True)
    server = build_server(tools)
    try:
        if args.transport == "stdio":
            asyncio.run(_run_stdio(server))
        else:
            asyncio.run(_run_streamable_http(server, args.bind_host, args.port))
    finally:
        tools.cleanup()


if __name__ == "__main__":
    main()
