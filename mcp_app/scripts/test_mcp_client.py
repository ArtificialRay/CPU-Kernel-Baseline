"""Plain MCP client for end-to-end verification without nanobot.

Drives list_tools() (assert no read_code) -> call_tool("compile", {"definition":
..., "code": ...}) -> call_tool("evaluate", {}) -> call_tool("disassemble", {}) ->
list_resources()/read_resource() -> call_tool("submit", ...) for one
definition. Exposes both a callable API (used by mcp_app/smoke_test_driver.py)
and a standalone CLI for manual runs.

    # stdio mode: spawn the exact command a real MCP client (nanobot, or
    # skills/nanobot/nanobot-kernel-session/scripts/launch_session.py's
    # prepare_session(..., transport="stdio")'s output) would use.
    python -m mcp_app.scripts.test_mcp_client --transport stdio --definition <name> \\
        --command ssh --spawn-args ubuntu@host "cd ~/arm-bench && python3 -m mcp_app.server ..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult


def _tool_result_dict(result: CallToolResult) -> dict[str, Any]:
    """CallToolResult -> plain dict, preferring structuredContent."""
    if result.structuredContent is not None:
        return result.structuredContent
    text = "".join(getattr(c, "text", "") for c in result.content)
    return json.loads(text) if text else {}


async def run_tool_sequence(
    session: ClientSession,
    definition: str,
    *,
    starter_code: Optional[str] = None,
    submit_explanation: str = "smoke test",
    verbose: bool = True,
) -> dict[str, Any]:
    """Drive one full compile/evaluate/disassemble/submit sequence for
    `definition` over an already-initialized ClientSession. Raises
    AssertionError on any check failure. Returns submit()'s result dict.
    """
    await session.initialize()

    tools_result = await session.list_tools()
    tool_names = {t.name for t in tools_result.tools}
    assert "read_code" not in tool_names, f"read_code should be retired, got {tool_names}"
    assert tool_names == {"compile", "evaluate", "disassemble", "submit"}, tool_names
    if verbose:
        print(f"  tools: {sorted(tool_names)}")

    code = starter_code
    ref_name = f"{definition}/reference-scalar-kernel.cpp"
    if code is None:
        resources = await session.list_resources()
        ref = next((r for r in resources.resources if r.name == ref_name), None)
        assert ref is not None, f"no {ref_name} resource found — pass --code-file"
        read_result = await session.read_resource(ref.uri)
        code = read_result.contents[0].text  # type: ignore[union-attr]

    r = await session.call_tool("compile", {"definition": definition, "code": code})
    compile_result = _tool_result_dict(r)
    assert compile_result.get("status") == "OK", compile_result
    if verbose:
        print(f"  compile: {compile_result}")

    r = await session.call_tool("evaluate", {})
    eval_result = _tool_result_dict(r)
    assert eval_result.get("status") == "PASSED", eval_result
    if verbose:
        print(f"  evaluate: {eval_result.get('performance')}")

    r = await session.call_tool("disassemble", {})
    disasm_result = _tool_result_dict(r)
    assert "asm" in disasm_result, disasm_result
    if verbose:
        print(f"  disassemble: {disasm_result.get('asm', '').count(chr(10))} lines")

    v1_name = f"{definition}/v1.cpp"
    resources = await session.list_resources()
    names = {res.name for res in resources.resources}
    assert v1_name in names, sorted(names)
    v1 = next(res for res in resources.resources if res.name == v1_name)
    read_result = await session.read_resource(v1.uri)
    v1_text = read_result.contents[0].text  # type: ignore[union-attr]
    assert v1_text == code, f"{v1_name} resource content doesn't match the code that was compiled"

    r = await session.call_tool("submit", {"explanation": submit_explanation})
    submit_result = _tool_result_dict(r)
    assert submit_result.get("status") == "PASSED", submit_result
    if verbose:
        print(f"  submit: {submit_result}")

    return submit_result


async def run_stdio_sequence(
    command: str,
    args: list[str],
    definition: str,
    *,
    starter_code: Optional[str] = None,
    submit_explanation: str = "smoke test",
    verbose: bool = True,
) -> dict[str, Any]:
    params = StdioServerParameters(command=command, args=args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            return await run_tool_sequence(
                session, definition, starter_code=starter_code,
                submit_explanation=submit_explanation, verbose=verbose,
            )


async def run_sse_sequence(
    endpoint: str,
    definition: str,
    *,
    starter_code: Optional[str] = None,
    submit_explanation: str = "smoke test",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fallback path — see mcp_app/README.md: only needed if stdio-over-ssh
    turns out not to work with nanobot's real MCP config format."""
    from mcp.client.sse import sse_client

    async with sse_client(endpoint) as (read, write):
        async with ClientSession(read, write) as session:
            return await run_tool_sequence(
                session, definition, starter_code=starter_code,
                submit_explanation=submit_explanation, verbose=verbose,
            )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    p.add_argument("--command", default="ssh", help="stdio mode: spawn command.")
    p.add_argument("--spawn-args", nargs="*", default=[], help="stdio mode: spawn command args.")
    p.add_argument("--endpoint", help="sse mode: e.g. http://127.0.0.1:8765/sse")
    p.add_argument("--definition", required=True,
                    help="Definition name to compile/evaluate/disassemble/submit.")
    p.add_argument("--code-file", help="Path to a kernel.cpp to compile; defaults to "
                                        "the session's reference-scalar-kernel.cpp resource.")
    args = p.parse_args(argv)

    starter_code = Path(args.code_file).read_text() if args.code_file else None

    if args.transport == "stdio":
        result = asyncio.run(run_stdio_sequence(
            args.command, args.spawn_args, args.definition, starter_code=starter_code,
        ))
    else:
        if not args.endpoint:
            p.error("--endpoint is required for --transport sse")
        result = asyncio.run(run_sse_sequence(args.endpoint, args.definition, starter_code=starter_code))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
