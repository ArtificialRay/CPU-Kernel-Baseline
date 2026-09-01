"""eval/mcp_client.py — MCP client bridge for eval/evaluator.py's litellm loop.

This loop's tools are driven over a real MCP client of mcp_app/server.py —
the same server nanobot/Claude Code drive.

Two public entry points:
- `attach(endpoint, ...)` — opens an MCP ClientSession over an
  already-running mcp_app.server's SSH-tunneled streamable-http endpoint
  and returns an `MCPKernelClient`. Provisioning the instance and starting
  that server (skills/launch/launch_session.py's `prepare_session()`) is
  the caller's job (test_scripts/bench_fleet.py's shared driver, which owns
  that lifecycle identically for every harness) — `attach()` only does the
  genuinely own-harness-specific part: opening the session itself. One
  MCPKernelClient lives for a whole batch of definitions (potentially many
  — mcp_app's KernelSession is explicitly designed to serve many
  definitions off one long-lived connection, so there's no per-definition
  session teardown/rebuild here).
- `MCPKernelClient.tools_for(definition_name)` — a thin per-definition
  facade exposing `dispatch_tool_call`/`cleanup`, the interface 
  `eval/evaluator.py::run_agentic_eval`'s turn loop calls.

Tool surface presented to the model IS derived directly from
`session.list_tools()` — compile/evaluate/disassemble's schemas (including
`definition`/`version`) are forwarded to litellm exactly as mcp_app wrote
them, no stripping or re-injection. The model tracks and passes
`definition`/`version` itself, same as nanobot/Claude Code already do
against this same server — `build_user_prompt()` tells it which
`definition` this session is for, and it reads `version` back from each
`compile()` result. `dispatch_tool_call` is therefore a near-total
passthrough: whatever the model calls, forward as-is. This also means any
NEW tool mcp_app/server.py adds shows up here automatically, with zero
eval/ code changes — the only exception:

- `read_code` is reimplemented here over MCP Resources (`list_resources`/
  `read_resource`), the same primitive mcp_app/resources.py serves to
  nanobot/Claude Code.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from contracts import MCP_CLIENT_DEFAULTS
from skills.launch.launch_session import RemoteTarget, sync_results

# Generous per-call timeouts — evaluate() runs a full compile+timed-repeat
# pass and can legitimately take minutes; mcp_app/server.py pings the MCP
# session every 120s while a call is in flight (server.py's
# TOOL_CALL_PING_INTERVAL_S) specifically so a slow call doesn't look dead,
# so waiting generously here is the correct match, not a band-aid. Values
# from config/kernel_contracts.yaml's mcp_client section.
_CONNECT_TIMEOUT_S = MCP_CLIENT_DEFAULTS["connect_timeout_s"]
_TOOL_CALL_TIMEOUT_S = MCP_CLIENT_DEFAULTS["tool_call_timeout_s"]
_RESOURCE_CALL_TIMEOUT_S = MCP_CLIENT_DEFAULTS["resource_call_timeout_s"]
_COMPILE_TIMEOUT_S = MCP_CLIENT_DEFAULTS["compile_timeout_s"]
_DISASSEMBLE_TIMEOUT_S = MCP_CLIENT_DEFAULTS["disassemble_timeout_s"]

# Per-tool-name timeout override for dispatch_tool_call's generic forwarding
# below — pure tuning, not dispatch logic: a tool with no entry here just
# gets _TOOL_CALL_TIMEOUT_S, so a new mcp_app tool needs no entry to work.
_TOOL_TIMEOUTS_S: dict[str, float] = {
    "compile": _COMPILE_TIMEOUT_S,
    "disassemble": _DISASSEMBLE_TIMEOUT_S,
}


# Pseudo-tool: read_code never appears in session.list_tools() —
# reconstructed here as a litellm-facing schema wrapping
# list_resources()/read_resource().
_READ_CODE_SCHEMA: dict = {"type": "function", "function": {
    "name": "read_code",
    "description": (
        "Read a source file or disassembly saved during this session. "
        "Compiled versions are saved as v1.cpp, v2.cpp, ... (N from compile() result). "
        "Disassembled versions are saved as v1.s, v2.s, ... (written by disassemble()). "
        "On error, returns the list of available files so you can pick the right one."
    ),
    "parameters": {
        "type": "object",
        "properties": {"filename": {"type": "string", "description": "File to read, e.g. 'v2.cpp' or 'v1.s'."}},
        "required": ["filename"],
    },
}}


def _tool_schemas_from_raw(raw_tools: list) -> list[dict]:
    """mcp.types.Tool list (from session.list_tools()) -> litellm
    {"type": "function", ...} shape."""
    schemas = [
        {"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.inputSchema,
        }}
        for t in raw_tools
    ]
    schemas.append(_READ_CODE_SCHEMA)
    return schemas


class _SessionThread:
    """Runs one MCP ClientSession's async context managers alive on a
    dedicated background thread + event loop, exposing a blocking `call()`
    for evaluator.py's fully-synchronous turn loop. The MCP SDK is
    async-only; evaluator.py's history compression / retry-loop / litellm
    calls are all sync and out of scope to convert (see the plan) — this is
    the bridge, not a rewrite.
    """

    def __init__(self, endpoint: str, *, connect_timeout: float = _CONNECT_TIMEOUT_S) -> None:
        self._loop = asyncio.new_event_loop()
        self._session: Optional[ClientSession] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._error: Optional[BaseException] = None
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(endpoint, ready), daemon=True, name="mcp-client-session",
        )
        self._thread.start()
        if not ready.wait(timeout=connect_timeout):
            raise TimeoutError(f"MCP session to {endpoint!r} did not become ready within {connect_timeout}s")
        if self._error is not None:
            raise self._error

    def _run(self, endpoint: str, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main(endpoint, ready))
        finally:
            self._loop.close()

    async def _main(self, endpoint: str, ready: threading.Event) -> None:
        try:
            async with streamablehttp_client(endpoint) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._stop_event = asyncio.Event()
                    ready.set()
                    await self._stop_event.wait()
        except BaseException as e:  # noqa: BLE001 — surfaced to the constructor / next call()
            self._error = e
            ready.set()

    def call(self, coro_factory: Callable[[ClientSession], Coroutine[Any, Any, Any]], *, timeout: float) -> Any:
        if self._error is not None:
            raise RuntimeError(f"MCP session already failed: {self._error}") from self._error
        assert self._session is not None, "call() before session ready"
        future = asyncio.run_coroutine_threadsafe(coro_factory(self._session), self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._stop_event is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=15)


def _call_tool_result_to_dict(result: Any) -> dict:
    """mcp.types.CallToolResult -> plain dict, preferring structuredContent
    (mirrors mcp_app/scripts/test_mcp_client.py's `_tool_result_dict`, kept
    as a separate copy here since importing a `mcp_app.scripts` test helper
    from eval/'s runtime path would be an odd direction of dependency for
    non-test code)."""
    if result.structuredContent is not None:
        return result.structuredContent
    text = "".join(getattr(c, "text", "") for c in result.content)
    if not text:
        return {}
    import json
    return json.loads(text)


class MCPKernelClient:
    """One MCP session shared across every definition in a run. Construct
    via `attach()`, not directly."""

    def __init__(self, session_thread: _SessionThread, *, target: RemoteTarget,
                 author: str, remote_root: str) -> None:
        self._session = session_thread
        self._target = target
        self._author = author
        self._remote_root = remote_root
        self._schemas: Optional[list[dict]] = None  # cached — the server's tool set is fixed for its lifetime

    # ── raw MCP calls ───────────────────────────────────────────────────

    def _call_tool(self, name: str, args: dict, *, timeout: float = _TOOL_CALL_TIMEOUT_S) -> dict:
        result = self._session.call(lambda s: s.call_tool(name, args), timeout=timeout)
        return _call_tool_result_to_dict(result)

    def _list_resources(self):
        return self._session.call(lambda s: s.list_resources(), timeout=_RESOURCE_CALL_TIMEOUT_S)

    def _read_resource(self, uri):
        return self._session.call(lambda s: s.read_resource(uri), timeout=_RESOURCE_CALL_TIMEOUT_S)

    # ── public surface ──────────────────────────────────────────────────

    def tool_schemas(self) -> list[dict]:
        if self._schemas is None:
            result = self._session.call(lambda s: s.list_tools(), timeout=_RESOURCE_CALL_TIMEOUT_S)
            self._schemas = _tool_schemas_from_raw(result.tools)
        return self._schemas

    def tools_for(self, definition_name: str) -> "MCPToolsForDefinition":
        return MCPToolsForDefinition(self, definition_name)

    def sync_bench_trace_back(self, *, local_results_dir: str, definition: Optional[str] = None) -> dict:
        """Pull this run's trajectory + any new bench-trace solutions/traces
        back from the remote instance. See skills/launch/launch_session.py's
        `sync_results(sync_bench_trace=...)` docstring for why this step
        exists at all: mcp_app.server runs ON the remote instance, so
        KernelSession.compile()/submit() persist into the *remote*
        bench-trace, not the caller's local one."""
        return sync_results(
            self._target, self._author, definition=definition,
            remote_root=self._remote_root, local_results_dir=local_results_dir,
            sync_bench_trace=True,
        )

    def close(self) -> None:
        """Closes the MCP ClientSession only. The SSH tunnel/remote
        mcp_app.server process is owned by whoever called `attach()` (its
        docstring), not by this client — that caller's own `stop_tunnel()`
        call handles it."""
        self._session.close()


class MCPToolsForDefinition:
    """Per-definition facade matching AgentTools' public interface
    (dispatch_tool_call/cleanup) exactly, so
    eval/evaluator.py::run_agentic_eval's turn loop needs no changes beyond
    its construction line. Backed by a shared MCPKernelClient — cleanup()
    is a no-op here on purpose; the shared session's real teardown
    (MCPKernelClient.close()) happens once, after every definition in the
    run is done, not per-definition (mcp_app's KernelSession is explicitly
    designed to serve many definitions off one connection — rebuilding the
    tunnel + remote server process per definition would be wasteful and
    defeats that design)."""

    def __init__(self, client: MCPKernelClient, definition_name: str) -> None:
        self._client = client
        self._definition_name = definition_name

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Forward whatever the model called, as-is — it supplies its own
        `definition`/`version` (see module docstring on why this isn't
        stripped/re-injected). `read_code` is the one exception: not a real
        MCP tool, so it can't be forwarded at all."""
        try:
            if name == "read_code":
                return self._read_code(args.get("filename", ""))
            return self._client._call_tool(
                name, args, timeout=_TOOL_TIMEOUTS_S.get(name, _TOOL_CALL_TIMEOUT_S),
            )
        except Exception as e:  # noqa: BLE001 — surfaced to the agent loop as a normal tool error, not a crash
            return {"error": str(e)}

    def _read_code(self, filename: str) -> dict:
        if not filename:
            return {"error": "filename is required"}
        resource_name = f"{self._definition_name}/{filename}"
        listing = self._client._list_resources()
        match = next((r for r in listing.resources if r.name == resource_name), None)
        if match is None:
            available = sorted(
                r.name.split("/", 1)[1] for r in listing.resources
                if r.name.startswith(f"{self._definition_name}/")
                and (r.name.endswith(".cpp") or r.name.endswith(".s"))
            )
            return {"error": f"{filename!r} not found", "available": available}
        read_result = self._client._read_resource(match.uri)
        content = read_result.contents[0].text
        return {"filename": filename, "content": content}

    def cleanup(self) -> None:
        pass  # see class docstring — real teardown is MCPKernelClient.close()


def attach(
    endpoint: str,
    *,
    author: str,
    remote_root: str,
    target: RemoteTarget,
) -> MCPKernelClient:
    """Open an MCP ClientSession against an already-running mcp_app.server
    at `endpoint`
    """
    session_thread = _SessionThread(endpoint)
    return MCPKernelClient(session_thread, target=target, author=author, remote_root=remote_root)


__all__ = ["MCPKernelClient", "MCPToolsForDefinition", "attach"]
