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

_MIME_TYPES = {
    ".cpp": "text/x-c++src",
    ".s": "text/x-asm",
    ".jsonl": "application/x-ndjson",
}

# Glob patterns for files exposed as resources, in listing order.
_PATTERNS = ["trajectory.jsonl", "reference-scalar-kernel.cpp", "v*.cpp", "v*.s"]


def list_run_dir_resources(run_dir: Path) -> list[types.Resource]:
    """Rescan run_dir on every call — new vN.cpp/vN.s appear mid-session.

    run_dir is a session root with one subdirectory per definition
    (run_dir/<definition_name>/{trajectory.jsonl, vN.cpp, ...}) — glob one
    level deeper than the pattern itself, and dedupe/name resources by their
    path relative to run_dir (not the bare filename) since e.g. `v1.cpp`
    exists once per definition and bare-name dedup would silently drop all
    but one definition's copy.
    """
    if not run_dir.exists():
        return []
    seen: set[str] = set()
    resources: list[types.Resource] = []
    for pattern in _PATTERNS:
        for path in sorted(run_dir.glob(f"*/{pattern}")):
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


def read_run_dir_resource(run_dir: Path, uri: str) -> str:
    """Resolve a file:// URI back to a path, enforce containment in run_dir, read it."""
    prefix = "file://"
    if not uri.startswith(prefix):
        raise ValueError(f"Unsupported resource URI scheme: {uri!r}")
    target = Path(uri[len(prefix):]).resolve()
    run_dir_resolved = run_dir.resolve()
    try:
        target.relative_to(run_dir_resolved)
    except ValueError:
        raise ValueError(f"Resource {uri!r} is outside the session run directory") from None
    if not target.exists():
        raise FileNotFoundError(f"Resource not found: {uri!r}")
    return target.read_text(encoding="utf-8")


__all__ = ["list_run_dir_resources", "read_run_dir_resource"]
