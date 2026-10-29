"""`kgen show <definition>`: what is already in bench-trace for a definition, to write the next file against."""
from __future__ import annotations

import json
from pathlib import Path

from .add import find_definition
from .check import find_definition_file, solution_files
from .layout import definition_dir


def show(root: Path, name: str, full: bool = False) -> str:
    """Axes, tensors, tags, constraints, workload points and the solutions of `name`; `full` adds the reference."""
    path = find_definition_file(root, name)
    definition = find_definition(root, name)
    if path is None or definition is None:
        return f"definition {name!r} is not under {root / 'definitions'}"
    lines = [f"{definition.name}  ({path})", f"op_type: {definition.op_type}"]
    if definition.description:
        lines.append(f"description: {definition.description}")
    lines.append("axes: " + ", ".join(f"{a}={s.value}" if s.type == "const" else f"{a} (var)" for a, s in definition.axes.items()))
    for group, tensors in (("inputs", definition.inputs), ("outputs", definition.outputs)):
        for tensor, spec in tensors.items():
            shape = "scalar" if spec.shape is None else "[" + ", ".join(spec.shape) + "]"
            lines.append(f"{group[:-1]} {tensor}: {spec.dtype.value} {shape}")
    if definition.constraints:
        lines.append("constraints: " + "; ".join(definition.constraints))
    lines.append("tags: " + (", ".join(definition.tags) or "-"))

    wl_path = root / "workloads" / definition_dir(definition) / f"{name}.jsonl"
    rows = [json.loads(l) for l in wl_path.read_text().splitlines() if l.strip()] if wl_path.exists() else []
    lines.append(f"workloads: {len(rows)} ({wl_path})")
    for axis in definition.var_axes:
        values = sorted({r["axes"][axis] for r in rows if axis in r["axes"]})
        if values:
            lines.append(f"  {axis}: {values}")
    kinds = sorted({v.get("type", "?") for r in rows for v in r.get("inputs", {}).values()})
    lines.append(f"  input kinds: {', '.join(kinds) or '-'}")

    solutions = solution_files(root, definition)
    lines.append(f"solutions: {len(solutions)}")
    for file in solutions:
        solution = json.loads(file.read_text())
        lines.append(f"  {solution['dataset']}/{solution['author']}: entry {solution['spec']['entry_point']}, "
                     f"sources {', '.join(s['path'] for s in solution['sources'])}")
    if full:
        lines += ["reference:", definition.reference]
    return "\n".join(lines)
