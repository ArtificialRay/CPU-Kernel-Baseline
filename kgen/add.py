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
from .select import filter_axes


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


def axes_key(row: dict) -> str:
    return json.dumps(row["axes"], sort_keys=True)


def existing_uuids(path: Path) -> dict:
    """axes -> uuid of the workloads already in `path`: a workload keeps its uuid for as long as it lives."""
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {axes_key(r): r["uuid"] for r in rows if r.get("uuid")}


def add_definition(file: Path, root: Path, replace: bool = False) -> Path:
    raw = load_json(file)
    definition, problems = lint_definition(raw)
    if problems:
        raise AddError([f"{file}: {p}" for p in problems])
    target = definition_path(root, definition)
    require_free(target, replace)
    write_json(target, raw)
    return target


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fill_inputs(rows: list[dict], existing: list[dict], file: Path) -> None:
    """Give rows that have no `inputs` the ones the definition's workloads already use: a row with the same axes
    keeps that row's inputs and tags (real tensors included); a new row takes the first existing row's inputs with
    each `tensor` turned into `random` (so `bytes` layouts and scalars carry over)."""
    by_axes = {axes_key(r): r for r in existing}
    for n, row in enumerate(rows, 1):
        if "inputs" in row:
            continue
        same = by_axes.get(axes_key(row))
        if same is not None:
            row["inputs"] = same["inputs"]
            row.setdefault("tags", dict(same.get("tags", {})))
        elif existing:
            row["inputs"] = {name: {"type": "random"} if spec.get("type") == "tensor" else spec
                             for name, spec in existing[0]["inputs"].items()}
        else:
            raise AddError([f"{file}: row {n} has no inputs, and the definition has no workloads to take them from"])
        row.setdefault("tags", {})


def select_rows(rows: list[dict], batch_key: str, max_count: int, max_dups: int, file: Path) -> list[dict]:
    """The rows `filter_axes` keeps (all rows must have the same axis names)."""
    names = {tuple(sorted(r.get("axes", {}))) for r in rows}
    if len(names) != 1:
        raise AddError([f"{file}: rows must all have the same axes to be filtered, found {sorted(names)}"])
    batch_key = batch_key or next(iter(rows[0]["axes"]))
    if batch_key not in rows[0]["axes"]:
        raise AddError([f"{file}: --batch-key {batch_key!r} is not one of the axes {sorted(rows[0]['axes'])}"])
    by_id = {id(r["axes"]): r for r in rows}
    return [by_id[id(axes)] for axes in filter_axes([r["axes"] for r in rows], batch_key, max_count, max_dups)]


def add_workload(file: Path, definition_name: str, root: Path, replace: bool = False,
                 max_count: int = 0, batch_key: str = "", max_dups: int = 2) -> Path:
    """With `max_count`, the file is a candidate pool: `filter_axes` keeps `max_count` of its rows. A row may
    be just `{"axes": {...}}` (see fill_inputs)."""
    definition = find_definition(root, definition_name)
    if definition is None:
        raise AddError([f"definition {definition_name!r} is not in {root}; add the definition first"])
    target = workload_path(root, definition)
    require_free(target, replace)
    try:
        lines = Path(file).read_text().splitlines()
    except OSError as e:
        raise AddError([f"{file}: cannot read ({e.strerror})"])
    try:
        parsed = [json.loads(l) for l in lines if l.strip()]
    except json.JSONDecodeError as e:
        raise AddError([f"{file}: not valid JSON ({e})"])
    if not all(isinstance(r, dict) for r in parsed):
        raise AddError([f"{file}: every line must be a JSON object"])
    existing = read_rows(target)
    if max_count and parsed:
        parsed = select_rows(parsed, batch_key, max_count, max_dups, file)
    fill_inputs(parsed, existing, file)
    rows, problems = lint_workloads([json.dumps(r) for r in parsed], definition, root=root)
    if problems:
        raise AddError([f"{file}: {p}" for p in problems])
    kept = existing_uuids(target) if replace else {}
    write_jsonl(target, [{**row, "uuid": row.get("uuid") or kept.get(axes_key(row)) or uuid_def_axes(definition.name, row["axes"])}
                         for row in rows])
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
