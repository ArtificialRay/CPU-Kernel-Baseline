"""TrajectoryWriter — per-session append-only audit trail for one agent loop.

Copied from eval/agent_tools/trajectory.py (pure file I/O, no SSH or
eval.agent_tools-specific coupling — the copy needs no adaptation).

Layout under `agent-runs-mcp/<def_name>/` (mcp_app's harness-neutral
equivalent of eval/agent_tools's `agent-runs/<def_name>/`):
    trajectory.jsonl   — one JSON line per turn, written immediately after each tool call
    v1.cpp             — full source for compile version 1
    v2.cpp             — full source for compile version 2 (etc.)
    v1.s               — full asm for version 1 (written when disassemble is called)
    v3.s               — full asm for version 3 (etc.; gap is fine if not disassembled)

The version counter is internal to TrajectoryWriter and is bumped on each
compile call. It is used only for file naming — it is NOT a solution identity
(that lives in solution.hash()).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class TrajectoryWriter:
    """Append-only writer for one agent optimization session."""

    def __init__(self, run_dir: Path) -> None:
        self._dir = run_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fh = (run_dir / "trajectory.jsonl").open("a", encoding="utf-8")
        self._version = 0  # bumped on each compile; used for v{n}.cpp / v{n}.s naming

    # ── version / file management ─────────────────────────────────────────────

    def next_version(self) -> int:
        """Bump and return the current compile version number."""
        self._version += 1
        return self._version

    @property
    def current_version(self) -> int:
        return self._version

    def write_source(self, code: str, version: int) -> str:
        """Write source to v{version}.cpp; return the filename."""
        fname = f"v{version}.cpp"
        (self._dir / fname).write_text(code, encoding="utf-8")
        return fname

    def write_asm(self, asm: str, version: int) -> str:
        """Write asm to v{version}.s; return the filename."""
        fname = f"v{version}.s"
        (self._dir / fname).write_text(asm, encoding="utf-8")
        return fname

    # ── JSONL line ────────────────────────────────────────────────────────────

    def write_turn(
        self,
        *,
        turn: int,
        tool: str,
        reasoning: str = "",
        source_file: Optional[str] = None,
        asm_file: Optional[str] = None,
        metrics: Optional[dict] = None,
        solution_ref: Optional[str] = None,
    ) -> None:
        """Append one line to trajectory.jsonl immediately (flush after write)."""
        record: dict[str, Any] = {"turn": turn, "tool": tool}
        if reasoning:
            record["reasoning"] = reasoning
        if source_file is not None:
            record["source_file"] = source_file
        if asm_file is not None:
            record["asm_file"] = asm_file
        if metrics is not None:
            record["metrics"] = metrics
        if solution_ref is not None:
            record["solution_ref"] = solution_ref
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["TrajectoryWriter"]
