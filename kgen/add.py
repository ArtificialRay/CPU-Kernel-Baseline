"""`kgen add`: lint a file the agent wrote, then place it in bench-trace.

Nothing is written when there is a problem or when the target already exists (unless replace=True).
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.data import Definition

from .ids import uuid_def_axes
from .io import write_json, write_jsonl
from .layout import definition_path, solution_path, workload_path
from .lint import lint_definition, lint_solution, lint_workloads


class AddError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except OSError as e:
        raise AddError([f"{path}: cannot read ({e.strerror})"])
    except json.JSONDecodeError as e:
        raise AddError([f"{path}: not valid JSON ({e})"])


def find_definition(root: Path, name: str) -> Definition | None:
    """The definition `name` in the warehouse (its file is definitions/<dir>/<name>.json)."""
    for path in (root / "definitions").glob(f"*/{name}.json"):
        return Definition.model_validate_json(path.read_text())
    return None


def require_free(target: Path, replace: bool) -> None:
    if target.exists() and not replace:
        raise AddError([f"{target} already exists; pass --replace to overwrite it"])


def add_definition(file: Path, root: Path, replace: bool = False) -> Path:
    raw = load_json(file)
    definition, problems = lint_definition(raw)
    if problems:
        raise AddError([f"{file}: {p}" for p in problems])
    target = definition_path(root, definition)
    require_free(target, replace)
    write_json(target, raw)
    return target


def add_workload(file: Path, definition_name: str, root: Path, replace: bool = False) -> Path:
    definition = find_definition(root, definition_name)
    if definition is None:
        raise AddError([f"definition {definition_name!r} is not in {root}; add the definition first"])
    try:
        lines = Path(file).read_text().splitlines()
    except OSError as e:
        raise AddError([f"{file}: cannot read ({e.strerror})"])
    rows, problems = lint_workloads(lines, definition)
    if problems:
        raise AddError([f"{file}: {p}" for p in problems])
    target = workload_path(root, definition)
    require_free(target, replace)
    write_jsonl(target, [{**row, "uuid": uuid_def_axes(definition.name, row["axes"])} for row in rows])
    return target


def add_solution(file: Path, root: Path, replace: bool = False) -> Path:
    raw = load_json(file)
    name = raw.get("definition") if isinstance(raw, dict) else None
    definition = find_definition(root, name) if isinstance(name, str) else None
    solution, problems = lint_solution(raw, definition)
    if problems:
        raise AddError([f"{file}: {p}" for p in problems])
    target = solution_path(root, definition, solution)
    require_free(target, replace)
    write_json(target, raw)
    return target
