"""eval/mcp_client.py — MCP client bridge for eval/evaluator.py's litellm loop.

This loop's tools are driven over a real MCP client of mcp_app/server.py —
the same server nanobot/Claude Code drive.

Two public entry points:
- `attach(endpoint, ...)` — opens an MCP ClientSession over an
  already-running mcp_app.server's SSH-tunneled streamable-http endpoint
  and returns an `MCPKernelClient`. Provisioning and starting the server
  (skills/launch/launch_session.py's `prepare_session()`) is the caller's
  job. One MCPKernelClient serves a whole batch of definitions; mcp_app's
  KernelSession is built for many definitions per connection.
- `MCPKernelClient.tools_for(definition_name)` — a thin per-definition
  facade exposing `dispatch_tool_call`/`cleanup` for
  `eval/evaluator.py::run_agentic_eval`'s turn loop.

The tool schemas come straight from `session.list_tools()`, unmodified; the
model passes `definition`/`version` itself, as nanobot/Claude Code do, so
`dispatch_tool_call` forwards calls as-is and new server tools appear with
no eval/ changes. The one exception is `read_code`, reimplemented over MCP
Resources (`list_resources`/`read_resource`, served by mcp_app/resources.py).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from contracts import MCP_CLIENT_DEFAULTS
from skills.launch.launch_session import RemoteTarget

# Generous: evaluate() can legitimately take minutes, and mcp_app/server.py
# pings the session every TOOL_CALL_PING_INTERVAL_S so a slow call doesn't
# look dead. Values from config/kernel_contracts.yaml's mcp_client section.
_CONNECT_TIMEOUT_S = MCP_CLIENT_DEFAULTS["connect_timeout_s"]
_TOOL_CALL_TIMEOUT_S = MCP_CLIENT_DEFAULTS["tool_call_timeout_s"]
_RESOURCE_CALL_TIMEOUT_S = MCP_CLIENT_DEFAULTS["resource_call_timeout_s"]
_COMPILE_TIMEOUT_S = MCP_CLIENT_DEFAULTS["compile_timeout_s"]
_DISASSEMBLE_TIMEOUT_S = MCP_CLIENT_DEFAULTS["disassemble_timeout_s"]

# Per-tool timeout overrides; any other tool gets _TOOL_CALL_TIMEOUT_S.
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
    """Keeps one MCP ClientSession alive on a background thread + event loop
    and exposes a blocking `call()`: the MCP SDK is async-only, evaluator.py's
    turn loop is synchronous."""

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
    (a copy of mcp_app/scripts/test_mcp_client.py's `_tool_result_dict`, so
    runtime code doesn't import a test helper)."""
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

    def _call_tool(self, name: str, args: dict, *, timeout: float = _TOOL_CALL_TIMEOUT_S) -> dict:
        result = self._session.call(lambda s: s.call_tool(name, args), timeout=timeout)
        return _call_tool_result_to_dict(result)

    def _list_resources(self):
        return self._session.call(lambda s: s.list_resources(), timeout=_RESOURCE_CALL_TIMEOUT_S)

    def _read_resource(self, uri):
        return self._session.call(lambda s: s.read_resource(uri), timeout=_RESOURCE_CALL_TIMEOUT_S)

    def tool_schemas(self) -> list[dict]:
        if self._schemas is None:
            result = self._session.call(lambda s: s.list_tools(), timeout=_RESOURCE_CALL_TIMEOUT_S)
            self._schemas = _tool_schemas_from_raw(result.tools)
        return self._schemas

    def tools_for(self, definition_name: str) -> "MCPToolsForDefinition":
        return MCPToolsForDefinition(self, definition_name)

    def close(self) -> None:
        """Closes the MCP ClientSession only; the SSH tunnel and remote server
        belong to attach()'s caller (`stop_tunnel()`)."""
        self._session.close()


class MCPToolsForDefinition:
    """Per-definition facade (dispatch_tool_call/cleanup) for
    eval/evaluator.py::run_agentic_eval. cleanup() is a no-op on purpose: the
    shared session is torn down once per run via MCPKernelClient.close()."""

    def __init__(self, client: MCPKernelClient, definition_name: str) -> None:
        self._client = client
        self._definition_name = definition_name

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Forward the model's call as-is, except `read_code`, which is not a
        real MCP tool."""
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
    at `endpoint`."""
    session_thread = _SessionThread(endpoint)
    return MCPKernelClient(session_thread, target=target, author=author, remote_root=remote_root)


__all__ = ["MCPKernelClient", "MCPToolsForDefinition", "attach"]
