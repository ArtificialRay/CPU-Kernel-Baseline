"""Static checks on the files an agent writes, before they are placed in bench-trace.

Each function returns a list of problems as plain sentences naming the field, what was expected and what was
found, so the agent can fix the file and try again. An empty list means the file is acceptable. The schema
itself (types, enums, reference syntax, entry_point) is bench.data's; only the rules on top are here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import ValidationError

from bench.data import Definition, Solution
from bench.data.workload import Workload

from .axes import input_shape, resolve_axes

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


HALF_TYPES = ("bfloat16", "float16")


def tensor_input_problems(where: str, name: str, path: str, definition: Definition, axes: dict, root: Path) -> list[str]:
    """A `tensor` input must name a file under `root` whose shape is the one the definition declares for
    `name` under the workload's axes, and whose dtype can be loaded as the declared one."""
    target = (root / path).resolve()
    if root.resolve() not in target.parents:
        return [f"{where}: inputs.{name}.path {path!r} must stay inside {root}"]
    if not target.exists():
        return [f"{where}: inputs.{name}.path {path!r} does not exist under {root}"]
    try:
        array = np.load(target, mmap_mode="r")
    except (ValueError, OSError) as e:
        return [f"{where}: inputs.{name}.path {path!r} is not a readable .npy file ({e})"]
    problems = []
    want = input_shape(definition, name, resolve_axes(definition, axes, strict=False))
    if want is not None and tuple(array.shape) != want:
        problems.append(f"{where}: inputs.{name} file has shape {tuple(array.shape)}, the definition declares {want} for axes {axes}")
    declared = definition.inputs[name].dtype.value
    got = array.dtype
    # .npy has no bfloat16: half-precision data is stored as its 16-bit pattern (uint16 / void) and bench reads it
    # back bit for bit; float32 is also fine, every half-precision value is exact in it.
    loadable = (got.name == declared
                or (declared in HALF_TYPES and ((got.itemsize == 2 and got.kind in "uV") or got.name == "float32"))
                or (declared not in HALF_TYPES and got.kind != "V" and np.can_cast(got, np.dtype(declared), "same_kind")))
    if not loadable:
        problems.append(f"{where}: inputs.{name} file has dtype {got}, which cannot be loaded as the declared {declared}")
    return problems


def lint_workloads(lines: list[str], definition: Definition, require_uuid: bool = False,
                   root: Optional[Path] = None) -> tuple[list[dict], list[str]]:
    """Parse JSONL text lines into workload dicts. A uuid is optional (`add` assigns a missing one) unless
    `require_uuid` (a file already in the warehouse); when present it must be unique within the file.
    With `root`, the files that `tensor` inputs point to are checked too."""
    var_axes = {a for a, s in definition.axes.items() if s.type == "var"}
    used_in_inputs = {a for t in definition.inputs.values() for a in (t.shape or []) if a in var_axes}
    problems, rows, seen, uuids = [], [], {}, {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {n}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where}: not valid JSON ({e})")
            continue
        if not isinstance(raw, dict):
            problems.append(f"{where}: expected a JSON object, got {type(raw).__name__}")
            continue
        if require_uuid and not raw.get("uuid"):
            problems.append(f"{where}: no uuid")
        elif raw.get("uuid") in uuids:
            problems.append(f"{where}: uuid {raw['uuid']} is also on line {uuids[raw['uuid']]}")
        elif raw.get("uuid"):
            uuids[raw["uuid"]] = n
        try:
            w = Workload.model_validate({"uuid": "x", **raw})
        except ValidationError as err:
            problems += schema_problems(err, where)
            continue
        missing, extra = used_in_inputs - set(w.axes), set(w.axes) - var_axes
        if missing:
            problems.append(f"{where}: axes missing {sorted(missing)}, var axes that appear in the input shapes of {definition.name}")
        for axis in sorted(extra):
            why = "is const in the definition, leave it out" if axis in definition.axes else "is not an axis of the definition"
            problems.append(f"{where}: axes.{axis} {why}")
        if set(w.inputs) != set(definition.inputs):
            problems.append(f"{where}: inputs must be exactly {sorted(definition.inputs)}, got {sorted(w.inputs)}")
        if root is not None:
            for name, spec in w.inputs.items():
                if spec.type == "tensor" and name in definition.inputs:
                    problems += tensor_input_problems(where, name, spec.path, definition, w.axes, Path(root))
        key = json.dumps(w.axes, sort_keys=True)
        if key in seen:
            problems.append(f"{where}: same axes as line {seen[key]} ({key})")
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
