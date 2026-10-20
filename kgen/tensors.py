"""Putting real tensors into workloads: save an array as .npy and point a workload input at it.

This is the one place that knows how a `tensor` input is stored, whatever produced the data. A workload keeps
its uuid when its input changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np

from bench.data import Definition

from .errors import KgenError
from .io import write_jsonl
from .layout import definition_dir, workload_path
from .lint import lint_workloads

NUMPY_DTYPES = {"bfloat16": ml_dtypes.bfloat16, "float16": np.float16, "float32": np.float32, "float64": np.float64,
                "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8}


def tensor_file(definition: Definition, input_name: str, workload_axes: dict, label: str) -> str:
    """Path, relative to the warehouse, of the file holding `input_name` for a workload."""
    axes = "_".join(f"{a}{v}" for a, v in sorted(workload_axes.items()))
    return f"tensors/{definition_dir(definition)}/{definition.name}/{input_name}_{axes}_{label}.npy"


def attach_inputs(root: Path, definition: Definition, input_name: str, items: list, source: str,
                  dry_run: bool = False) -> int:
    """items: (index of the workload in its file, array, label) triples. Each array is cast to the dtype the
    definition declares for `input_name`, saved under tensors/, and the workload's input becomes a `tensor`
    pointing at it, tagged `inputs: real-activations`. Returns how many workloads were changed."""
    root = Path(root)
    path = workload_path(root, definition)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    dtype = NUMPY_DTYPES.get(definition.inputs[input_name].dtype.value)
    if dtype is None:
        raise KgenError(f"input {input_name!r} of {definition.name} has dtype {definition.inputs[input_name].dtype.value}, "
                        f"which capture cannot write")
    for index, array, label in items:
        row = rows[index]
        rel = tensor_file(definition, input_name, row["axes"], label)
        if not dry_run:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / rel, np.ascontiguousarray(np.asarray(array).astype(dtype)))
        row["inputs"][input_name] = {"type": "tensor", "path": rel, "source": source}
        row.setdefault("tags", {})["inputs"] = "real-activations"
    if not dry_run:
        _, problems = lint_workloads([json.dumps(r) for r in rows], definition, require_uuid=True, root=root)
        if problems:
            raise KgenError(f"{path}: the rewritten workloads do not pass lint:\n  " + "\n  ".join(problems))
        write_jsonl(path, rows)
    return len(items)
