#!/usr/bin/env python3
"""Record what the *baseline* kernel scores on each workload, so the SQNR gate is relative.

`correctness:sqnr` definitions were gated on a fixed 20 dB floor. For a weight-quantized
gemm that floor is far below what the reference implementation itself achieves -- ggml's
own q4_K arithmetic sits near 40-45 dB -- so a candidate could discard 20+ dB of signal
the baseline keeps and still pass. It did: see docs/e2e_qwen35.md.

This runs a reference solution (default `reference-scalar`, a scalar port of ggml's own
vec_dot, so the number is "what the real kernel's quantization costs") over every workload
of a definition and writes `baseline_sqnr_db` into that workload's tags. The evaluator then
gates at `baseline_sqnr_db - EvalConfig.sqnr_margin_db`.

Re-run this whenever the workload inputs change -- the floor is only valid for the exact
inputs it was measured on.

    python scripts/e2e/calibrate_sqnr_floor.py --definitions gemm_ggml_q4_K_n9216_k2560 ...
    python scripts/e2e/calibrate_sqnr_floor.py --tag e2e:qwen3.5-4b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.config import EvalConfig
from bench.data import TraceSet
from bench.runner import run_solution_on_workloads

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "bench-trace")
    ap.add_argument("--definitions", nargs="*", default=None)
    ap.add_argument("--tag", default=None, help="calibrate every definition carrying this tag")
    ap.add_argument("--author", default="reference-scalar", help="solution author to measure")
    ap.add_argument("--baseline-author", default="baseline-llamacpp-arm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ts = TraceSet.from_path(args.root)
    names = list(args.definitions or [])
    if args.tag:
        names += [n for n, d in ts.definitions.items() if args.tag in d.tags and n not in names]
    if not names:
        print("nothing selected (pass --definitions or --tag)")
        return 1

    cfg = EvalConfig(baseline_author=args.baseline_author)
    updated = failed = 0
    for name in sorted(names):
        d = ts.definitions.get(name)
        if d is None:
            print(f"{name}: no such definition"); failed += 1; continue
        sol = next((s for s in ts.solutions.get(name, []) if s.author == args.author), None)
        if sol is None:
            print(f"{name}: no '{args.author}' solution"); failed += 1; continue
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
                print(f"  {name} M={w.axes.get('M')}: {args.author} produced no SQNR"); failed += 1
                continue
            w.tags["baseline_sqnr_db"] = f"{sq:.2f}"
            w.tags["baseline_sqnr_author"] = args.author
            vals.append((w.axes.get("M"), sq))
            updated += 1
        if vals and not args.dry_run:
            path = args.root / "workloads" / d.op_type / f"{name}.jsonl"
            path.write_text("".join(w.model_dump_json(exclude_none=True) + "\n" for w in wls))
        print(f"{name}: " + "  ".join(f"M={m}:{s:.1f}dB" for m, s in vals))

    print(f"\n{updated} workloads calibrated, {failed} problems")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
