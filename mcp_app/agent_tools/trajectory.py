"""TrajectoryWriter — per-session append-only audit trail for one agent loop.

Layout under `agent-runs-mcp/<def_name>/`:
    trajectory.jsonl   — one JSON line per turn, written immediately after each tool call
    v1.cpp             — full source for compile version 1
    v2.cpp             — full source for compile version 2 (etc.)
    v1.s               — full asm for version 1 (written when disassemble is called)
    v3.s               — full asm for version 3 (etc.; gap is fine if not disassembled)

The version/turn counters are internal to TrajectoryWriter, seeded from
whatever's already in trajectory.jsonl on construction (see
`_scan_resume_state`) so a new process picking up an existing run_dir continues numbering 
instead of colliding with files an earlier session already wrote. Convenient for kernel 
checkpoint restart
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
        traj_path = run_dir / "trajectory.jsonl"
        self._version, self._last_turn = self._scan_resume_state(traj_path)
        self._fh = traj_path.open("a", encoding="utf-8")

    @staticmethod
    def read_records(traj_path: Path) -> list[dict]:
        """Parse every line of an existing trajectory.jsonl into dicts (empty
        list if the file doesn't exist yet). Shared by `_scan_resume_state`
        and `KernelSession.check_progress` (base.py) so both read the same
        file the same way instead of duplicating the parse loop."""
        if not traj_path.exists():
            return []
        return [
            json.loads(line)
            for line in traj_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @classmethod
    def _scan_resume_state(cls, traj_path: Path) -> tuple[int, int]:
        """(max compile version, max turn) already recorded — 0/0 if there's
        no prior trajectory.jsonl for this definition."""
        max_version = 0
        max_turn = 0
        for rec in cls.read_records(traj_path):
            max_turn = max(max_turn, rec.get("turn") or 0)
            if rec.get("tool") == "compile":
                max_version = max(max_version, rec.get("metrics", {}).get("version") or 0)
        return max_version, max_turn

    # ── version / turn / file management ────────────────────────────────────

    def next_version(self) -> int:
        """Bump and return the current compile version number."""
        self._version += 1
        return self._version

    @property
    def current_version(self) -> int:
        return self._version

    @property
    def last_turn(self) -> int:
        """Highest turn number already recorded (0 if this definition has no
        prior trajectory) — the turn counter a resuming KernelSession should
        continue incrementing from, not restart at 0."""
        return self._last_turn

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
