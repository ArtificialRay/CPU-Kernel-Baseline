"""Static checks on the files an agent writes, before they are placed in bench-trace.

Each function returns a list of problems as plain sentences naming the field, what was expected and what was
found, so the agent can fix the file and try again. An empty list means the file is acceptable. The schema
itself (types, enums, reference syntax, entry_point) is bench.data's; only the rules on top are here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from bench.data import Definition, Solution
from bench.data.workload import Workload

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def schema_problems(err: ValidationError, where: str = "") -> list[str]:
    prefix = f"{where}: " if where else ""
    return [f"{prefix}{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in err.errors()]


def lint_definition(raw: Any) -> tuple[Optional[Definition], list[str]]:
    try:
        d = Definition.model_validate(raw)
    except ValidationError as err:
        return None, schema_problems(err)
    problems = []
    if not NAME.match(d.name):
        problems.append(f"name: {d.name!r} must match {NAME.pattern} (it becomes the file name)")
    if not NAME.match(d.op_type):
        problems.append(f"op_type: {d.op_type!r} must match {NAME.pattern}")
    for axis, spec in d.axes.items():
        parent = getattr(spec, "parent", None)
        if parent and parent not in d.axes:
            problems.append(f"axes.{axis}.parent: {parent!r} is not declared in axes")
    if set(d.inputs) & set(d.outputs):
        problems.append(f"inputs and outputs share names: {sorted(set(d.inputs) & set(d.outputs))}")
    return d, problems


def lint_workloads(lines: list[str], definition: Definition) -> tuple[list[dict], list[str]]:
    """Parse JSONL text lines into workload dicts (without uuid). The file must not carry uuids: kgen assigns them."""
    var_axes = {a for a, s in definition.axes.items() if s.type == "var"}
    problems, rows, seen = [], [], {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {n}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where}: not valid JSON ({e})")
            continue
        if "uuid" in raw:
            problems.append(f"{where}: remove 'uuid'; kgen assigns it from the definition name and the axes")
        try:
            w = Workload.model_validate({**raw, "uuid": "x"})
        except ValidationError as err:
            problems += schema_problems(err, where)
            continue
        missing, extra = var_axes - set(w.axes), set(w.axes) - var_axes
        if missing:
            problems.append(f"{where}: axes missing the var axes {sorted(missing)} of {definition.name}")
        for axis in sorted(extra):
            why = "is const in the definition, leave it out" if axis in definition.axes else "is not an axis of the definition"
            problems.append(f"{where}: axes.{axis} {why}")
        if set(w.inputs) != set(definition.inputs):
            problems.append(f"{where}: inputs must be exactly {sorted(definition.inputs)}, got {sorted(w.inputs)}")
        key = json.dumps(w.axes, sort_keys=True)
        if key in seen:
            problems.append(f"{where}: same axes as line {seen[key]} ({key}); the uuid would collide")
        seen.setdefault(key, n)
        rows.append(raw)
    if not rows and not problems:
        problems.append("no workloads in the file")
    return rows, problems


def lint_solution(raw: Any, definition: Optional[Definition]) -> tuple[Optional[Solution], list[str]]:
    try:
        s = Solution.model_validate(raw)
    except ValidationError as err:
        return None, schema_problems(err)
    problems = []
    if not NAME.match(s.author):
        problems.append(f"author: {s.author!r} must match {NAME.pattern}")
    if s.name != f"{s.author}_{s.definition}":
        problems.append(f"name: expected {s.author}_{s.definition!s}, got {s.name!r}")
    if definition is None:
        problems.append(f"definition: {s.definition!r} is not in the warehouse; add the definition first")
    return s, problems
