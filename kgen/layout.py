"""Where a Definition, its workloads and its Solutions live in a bench-trace warehouse."""
from __future__ import annotations

from pathlib import Path


def definition_dir(definition) -> str:
    """Directory under definitions/ and workloads/. simd-loop definitions are the exception: each is its own
    op_type (the loop id) and they share one directory."""
    return "simd-loop" if definition.simd_loop_meta is not None else definition.op_type


def definition_path(root: Path, definition) -> Path:
    return root / "definitions" / definition_dir(definition) / f"{definition.name}.json"


def workload_path(root: Path, definition) -> Path:
    return root / "workloads" / definition_dir(definition) / f"{definition.name}.jsonl"


def solution_path(root: Path, definition, solution) -> Path:
    """solutions/<dataset>/<author>/<op_type>/<definition>.json; simd-loop nests one more level
    (<author>/<loop>/<author>_<loop>.json)."""
    base = root / "solutions" / solution.dataset.value / solution.author / definition_dir(definition)
    if solution.dataset.value == "simd-loop":
        return base / definition.name / f"{solution.name}.json"
    return base / f"{definition.name}.json"
