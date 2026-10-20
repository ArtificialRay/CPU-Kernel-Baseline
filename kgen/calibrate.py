#!/usr/bin/env python3
"""Record what the *baseline* kernel scores on each workload, so the SQNR gate is relative.

`correctness:sqnr` definitions were gated on a fixed 20 dB floor. For a weight-quantized
gemm that floor is far below what the reference implementation itself achieves -- ggml's
own q4_K arithmetic sits near 40-45 dB -- so a candidate could discard 20+ dB of signal
the baseline keeps and still pass. It did: see e2e/README.md.

This runs a reference solution (for llama.cpp `reference-scalar`, a scalar port of ggml's own
vec_dot, so the number is "what the real kernel's quantization costs") over every workload
of a definition and writes `baseline_sqnr_db` into that workload's tags. The evaluator then
gates at `baseline_sqnr_db - EvalConfig.sqnr_margin_db`.

Re-run this whenever the workload inputs change -- the floor is only valid for the exact
inputs it was measured on.

    calibrate_sqnr_floors(author="reference-scalar", baseline_author="baseline-llamacpp-arm",
                          tag="e2e:qwen3.5-4b")             # or: python -m kgen calibrate --author ... --baseline-author ... --tag ...

Nothing here is specific to one framework: which author is the reference and which is the
baseline comes from the caller (e2e passes its adapter's reference_author / baseline_author).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from bench.config import EvalConfig
from bench.data import TraceSet
from bench.runner import run_solution_on_workloads
from .errors import KgenError
from .io import write_jsonl
from .layout import workload_path

REPO = Path(__file__).resolve().parents[1]
SQNR_TAG = "correctness:sqnr"


def calibrate_sqnr_floors(root: Path = REPO / "bench-trace", *, author: str, baseline_author: str,
                          definitions=None, tag=None, dry_run: bool = False) -> tuple[int, int]:
    """Write `baseline_sqnr_db` / `baseline_sqnr_author` into the tags of every workload of
    the selected definitions. Returns (workloads calibrated, problems).

    definitions     definition names to calibrate
    tag             also calibrate every definition carrying this tag (e.g. e2e:qwen3.5-4b)
                    (with neither: every definition tagged correctness:sqnr)
    author          whose solution to measure (the reference implementation)
    baseline_author the solution the evaluator scores speedup against (EvalConfig.baseline_author)
    dry_run         measure and print, but don't rewrite the workload files
    """
    root = Path(root)
    ts = TraceSet.from_path(root)
    names = list(definitions or [])
    if tag:
        names += [n for n, d in ts.definitions.items() if tag in d.tags and n not in names]
    if not definitions and not tag:      # every definition gated on SQNR
        names = [n for n, d in ts.definitions.items() if SQNR_TAG in d.tags]
    if not names:
        raise KgenError("nothing selected (pass definitions or tag, or tag a definition correctness:sqnr)")

    cfg = EvalConfig(baseline_author=baseline_author)
    updated = failed = 0
    for name in sorted(names):
        d = ts.definitions.get(name)
        if d is None:
            print(f"{name}: no such definition"); failed += 1; continue
        sol = next((s for s in ts.solutions.get(name, []) if s.author == author), None)
        if sol is None:
            print(f"{name}: no '{author}' solution"); failed += 1; continue
        wls = list(ts.workloads.get(name, []))
        if not wls:
            print(f"{name}: no workloads"); failed += 1; continue

        traces = run_solution_on_workloads(d, sol, wls, cfg=cfg, trace_set=ts)
        by_uuid = {}
        for t in traces:
            ev = t.evaluation
            sq = (ev.correctness.extra or {}).get("sqnr_db") if ev and ev.correctness else None
            if sq is not None:
                by_uuid[t.workload.uuid] = float(sq)
            elif ev is not None:
                print(f"  {name}: {ev.status} {(ev.log or '')[:140]}")

        path = workload_path(root, d)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        vals = []
        for row in rows:       # the file's own rows are edited, so nothing else about a workload is touched
            sq = by_uuid.get(row["uuid"])
            if sq is None:
                print(f"  {name} {row['axes']}: {author} produced no SQNR"); failed += 1
                continue
            row.setdefault("tags", {}).update(baseline_sqnr_db=f"{sq:.2f}", baseline_sqnr_author=author)
            vals.append((row["axes"], sq))
            updated += 1
        if vals and not dry_run:
            write_jsonl(path, rows)
        print(f"{name}: " + "  ".join(f"{axes}:{s:.1f}dB" for axes, s in vals))

    print(f"\n{updated} workloads calibrated, {failed} problems")
    return updated, failed
