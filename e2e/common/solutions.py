"""Reading back what an agent session (or a hand-written set) left in bench-trace/solutions/."""
from __future__ import annotations

import json
from pathlib import Path

from e2e.errors import StepFailed

REPO = Path(__file__).resolve().parents[2]
SOLUTIONS = REPO / "bench-trace" / "solutions"

# What the MCP server keeps next to <definition>.json while a session runs
# (mcp_app/agent_tools/base.py); a leftover one is the last compile, not the submitted best.
INTERMEDIATE_SUFFIXES = ("_current", "_best")


def load_author(dataset: str, author: str, root: Path = SOLUTIONS, op_type: str = "gemm") -> dict:
    """definition name -> solution JSON for every `<definition>.json` of one author."""
    sol_dir = Path(root) / dataset / author / op_type
    if not sol_dir.is_dir():
        raise StepFailed(f"no such solutions dir: {sol_dir}")
    out = {}
    for f in sorted(sol_dir.glob("*.json")):
        if f.stem.endswith(INTERMEDIATE_SUFFIXES):
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def source_text(solution: dict, path: str = "kernel.cpp") -> str:
    """The content of one source file of a solution JSON (the agent's kernel by default)."""
    return next(s["content"] for s in solution["sources"] if s["path"] == path)
