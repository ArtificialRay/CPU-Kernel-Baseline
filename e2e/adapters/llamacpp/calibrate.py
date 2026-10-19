#!/usr/bin/env python3
"""Record what the *baseline* kernel scores on each workload, so the SQNR gate is relative.

`correctness:sqnr` definitions were gated on a fixed 20 dB floor. For a weight-quantized
gemm that floor is far below what the reference implementation itself achieves -- ggml's
own q4_K arithmetic sits near 40-45 dB -- so a candidate could discard 20+ dB of signal
the baseline keeps and still pass. It did: see e2e/README.md.

This runs a reference solution (default `reference-scalar`, a scalar port of ggml's own
vec_dot, so the number is "what the real kernel's quantization costs") over every workload
of a definition and writes `baseline_sqnr_db` into that workload's tags. The evaluator then
gates at `baseline_sqnr_db - EvalConfig.sqnr_margin_db`.

Re-run this whenever the workload inputs change -- the floor is only valid for the exact
inputs it was measured on.

    calibrate_sqnr_floors(definitions=["gemm_ggml_q4_K_n9216_k2560"])
    calibrate_sqnr_floors(tag="e2e:qwen3.5-4b")             # or: python -m e2e prepare --steps calibrate
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bench.config import EvalConfig
from bench.data import TraceSet
from bench.runner import run_solution_on_workloads
from e2e.errors import StepFailed

REPO = Path(__file__).resolve().parents[3]


def calibrate_sqnr_floors(root: Path = REPO / "bench-trace", *, definitions=None, tag=None,
                          author: str = "reference-scalar", baseline_author: str = "baseline-llamacpp-arm",
                          dry_run: bool = False) -> tuple[int, int]:
    """Write `baseline_sqnr_db` / `baseline_sqnr_author` into the tags of every workload of
    the selected definitions. Returns (workloads calibrated, problems).

    definitions     definition names to calibrate
    tag             also calibrate every definition carrying this tag (e.g. e2e:qwen3.5-4b)
    author          whose solution to measure
    dry_run         measure and print, but don't rewrite the workload files
    """
    root = Path(root)
    ts = TraceSet.from_path(root)
    names = list(definitions or [])
    if tag:
        names += [n for n, d in ts.definitions.items() if tag in d.tags and n not in names]
    if not names:
        raise StepFailed("nothing selected (pass definitions or tag)")

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

        vals = []
        for w in wls:
            sq = by_uuid.get(w.uuid)
            if sq is None:
                print(f"  {name} M={w.axes.get('M')}: {author} produced no SQNR"); failed += 1
                continue
            w.tags["baseline_sqnr_db"] = f"{sq:.2f}"
            w.tags["baseline_sqnr_author"] = author
            vals.append((w.axes.get("M"), sq))
            updated += 1
        if vals and not dry_run:
            path = root / "workloads" / d.op_type / f"{name}.jsonl"
            path.write_text("".join(w.model_dump_json(exclude_none=True) + "\n" for w in wls))
        print(f"{name}: " + "  ".join(f"M={m}:{s:.1f}dB" for m, s in vals))

    print(f"\n{updated} workloads calibrated, {failed} problems")
    return updated, failed
