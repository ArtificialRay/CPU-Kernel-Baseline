"""MCP Resources over a session's run_dir — the retired read_code tool's replacement.

Reading previously-written vN.cpp/vN.s/trajectory.jsonl/reference-scalar-kernel.cpp
happens via the protocol's native Resources mechanism instead of a bespoke
`read_code` tool — works identically regardless of transport or harness
co-location, unlike returning a bare filesystem path (which compile()'s
`source_file` and disassemble()'s `asm_file` responses still do too, as a
convenience for co-located callers, but Resources are the protocol-correct
read path).
"""

from __future__ import annotations

from pathlib import Path

import mcp.types as types
from contracts import REFERENCE_SCALAR_FILENAME

_MIME_TYPES = {
    ".cpp": "text/x-c++src",
    ".s": "text/x-asm",
    ".jsonl": "application/x-ndjson",
}

# The unsolved starting point for a definition — written for every definition
# at server startup (see session.py::_write_reference_scalar_kernels), never
# authored by an agent. Visible to every MCP session regardless of
# `visible_definitions`
_REFERENCE_PATTERN = REFERENCE_SCALAR_FILENAME

# Glob patterns for files exposed as resources, in listing order.
_PATTERNS = ["trajectory.jsonl", _REFERENCE_PATTERN, "v*.cpp", "v*.s"]


def list_run_dir_resources(
    run_dir: Path, visible_definitions: frozenset[str] = frozenset(),
) -> list[types.Resource]:
    """Rescan run_dir on every call — new vN.cpp/vN.s appear mid-session.

    run_dir is a session root with one subdirectory per definition
    (run_dir/<definition_name>/{trajectory.jsonl, vN.cpp, ...}), shared by
    every MCP session connected to this server process (one process serves
    the whole dataset, not one process per definition — see server.py).

    `_REFERENCE_PATTERN` is exposed for every definition, always — it's the
    shared, unsolved problem statement, not another session's solution.
    Every other pattern (trajectory.jsonl, vN.cpp, vN.s — an agent's own
    submitted work) is scoped to `visible_definitions`: the set of
    definitions *this* MCP session has itself compile()'d (see
    KernelSession.session_definitions / server.py's
    note_session_definition). This keeps one job's already-optimized
    kernels from leaking into another job's session purely because they
    share this process and run_dir. A session that hasn't compile()'d
    anything yet (fresh connection, or after a reset) sees none of these —
    see SKILL.md's "Recovering from an MCP session reset" for what that
    means in practice.

    Dedupe/name resources by their path relative to run_dir (not the bare
    filename) since e.g. `v1.cpp` exists once per definition and bare-name
    dedup would silently drop all but one definition's copy.
    """
    if not run_dir.exists():
        return []
    seen: set[str] = set()
    resources: list[types.Resource] = []
    for pattern in _PATTERNS:
        if pattern == _REFERENCE_PATTERN:
            definition_dirs = sorted(d for d in run_dir.glob("*") if d.is_dir())
        else:
            definition_dirs = [run_dir / d for d in sorted(visible_definitions)]
        for definition_dir in definition_dirs:
            for path in sorted(definition_dir.glob(pattern)):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(run_dir))
                if rel in seen:
                    continue
                seen.add(rel)
                resources.append(
                    types.Resource(
                        uri=f"file://{path.resolve()}",
                        name=rel,
                        mimeType=_MIME_TYPES.get(path.suffix, "text/plain"),
                    )
                )
    return resources


def read_run_dir_resource(
    run_dir: Path, uri: str, visible_definitions: frozenset[str] = frozenset(),
) -> str:
    """Resolve a file:// URI back to a path, enforce containment in run_dir
    AND the same session scope as list_run_dir_resources — a session can't
    bypass the listing scope by guessing/reusing another definition's URI
    (e.g. from its own earlier turns, before this scoping existed, or by
    pattern-guessing a filename) — then read it.
    """
    prefix = "file://"
    if not uri.startswith(prefix):
        raise ValueError(f"Unsupported resource URI scheme: {uri!r}")
    target = Path(uri[len(prefix):]).resolve()
    run_dir_resolved = run_dir.resolve()
    try:
        rel = target.relative_to(run_dir_resolved)
    except ValueError:
        raise ValueError(f"Resource {uri!r} is outside the session run directory") from None
    if not target.exists():
        raise FileNotFoundError(f"Resource not found: {uri!r}")

    definition = rel.parts[0] if rel.parts else ""
    if target.name != _REFERENCE_PATTERN and definition not in visible_definitions:
        raise ValueError(
            f"Resource {uri!r} belongs to definition {definition!r}, which this "
            "session has not compiled — not visible to this session."
        )
    return target.read_text(encoding="utf-8")


__all__ = ["list_run_dir_resources", "read_run_dir_resource"]
