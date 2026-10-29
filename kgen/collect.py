"""Collect the shapes a real model produces, as candidate workloads.

The twin of kgen.capture: the same loader file (`load() -> (model, batches)` and `adapters()`, see kgen/capture.py)
and the same hook pass, but only the sizes of each call are kept, not the data. For every selected definition it writes
the distinct values of the definition's var axes that the matching layers saw, one `{"axes": {...}}` row each, to
DIR/<definition>.jsonl. Those files are pools: `kgen add workload DIR/<d>.jsonl --definition <d> --max-count N
--batch-key AXIS --replace` picks N diverse rows from one (the rows may be given to another definition with the
same var axes: how many tokens a prompt has does not depend on the model).

    python -m kgen collect pytorch --loader load_shapes.py --definitions gemm_q4_K_n2048_k2048 --out-dir pools/

What the model is fed decides the distribution (prompt lengths for a language model, input resolutions for a
vision model), so that is the loader's business.
"""
from __future__ import annotations

import json
from pathlib import Path

from .capture import load_loader, make_targets_for, report_unmatched, run_hooked, select_targets
from .io import write_jsonl


def collect_pytorch(root: Path, loader: Path, out_dir: Path, definitions=None, tag=None, input_name: str | None = None,
                    modules_per_shape: int = 1) -> dict:
    """Write the candidate pool of every selected definition that a layer of the model matches. Returns
    {definition name: number of distinct rows}."""
    targets = select_targets(root, definitions, tag, input_name)
    bound: dict = {}

    def record(name, adapter, found, tensor, axes):
        for target in found:
            target.shapes.append(dict(zip(axes, (int(n) for n in tensor.shape))))

    model, batches, adapters = load_loader(loader)
    run_hooked(model, batches, adapters, make_targets_for(targets, bound), modules_per_shape, None, record)
    report_unmatched(targets, bound)

    pools: dict = {}        # definition -> {axes key: axes}; layers of different shapes both feed a definition
    for target in bound.values():
        d = target.definition
        var = {a for a, spec in d.axes.items() if spec.type == "var"}
        rows = pools.setdefault(d.name, {})
        for dims in target.shapes:
            axes = {a: n for a, n in dims.items() if a in var}
            rows[json.dumps(axes, sort_keys=True)] = axes
    written = {}
    for name, rows in pools.items():
        ordered = sorted(rows.values(), key=lambda axes: tuple(axes.values()))
        write_jsonl(Path(out_dir) / f"{name}.jsonl", [{"axes": axes} for axes in ordered])
        first = ", ".join(f"{a} {min(r[a] for r in ordered)}..{max(r[a] for r in ordered)}" for a in ordered[0])
        print(f"{name}: {len(ordered)} distinct ({first}) -> {Path(out_dir) / f'{name}.jsonl'}")
        written[name] = len(ordered)
    return written
