#!/usr/bin/env python3
"""Render a measure_e2e.py results JSON as the table that goes in the paper.

Speed and perplexity come from the same binaries in one run, so they belong in one table:
a speedup is only meaningful next to what it cost in model quality.

    python scripts/e2e/results_table.py e2e_results2.json --baseline stock
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def medians(samples):
    out = {}
    for s in samples:
        out.setdefault((s["build"], s["threads"], s["kind"]), []).append(s["tok_per_s"])
    return {k: statistics.median(v) for k, v in out.items()}, {k: len(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--baseline", default="stock")
    args = ap.parse_args()
    d = json.loads(args.results.read_text())
    med, n = medians(d.get("samples", []))
    builds = list(dict.fromkeys(s["build"] for s in d.get("samples", [])))
    threads = sorted({s["threads"] for s in d.get("samples", [])})
    ppl = d.get("perplexity", {}) or {}

    print(f"host: {d.get('host','?')}")
    print(f"model: {Path(d.get('model','?')).name}   pp={d.get('pp')} tg={d.get('tg')} reps={d.get('reps')}\n")

    print("absolute throughput (tok/s, median of reps)")
    hdr = "build".ljust(18) + "".join(f"  t={t:<3} pp      tg" for t in threads)
    print(hdr); print("-" * len(hdr))
    for b in builds:
        row = b.ljust(18)
        for t in threads:
            pp = med.get((b, t, "pp")); tg = med.get((b, t, "tg"))
            row += f"  {pp:7.1f} {tg:7.2f}" if pp and tg else "        -       -"
        print(row)

    if args.baseline in builds:
        print(f"\nspeedup vs {args.baseline}")
        print(hdr); print("-" * len(hdr))
        for b in builds:
            if b == args.baseline:
                continue
            row = b.ljust(18)
            for t in threads:
                num_pp, den_pp = med.get((b, t, "pp")), med.get((args.baseline, t, "pp"))
                num_tg, den_tg = med.get((b, t, "tg")), med.get((args.baseline, t, "tg"))
                row += (f"  {num_pp/den_pp:6.2f}x {num_tg/den_tg:6.2f}x"
                        if num_pp and den_pp and num_tg and den_tg else "        -       -")
            print(row)

    def ppl_of(name):
        """measure_e2e stores llama-perplexity's raw tail, so pull the number out of it."""
        v = ppl.get(name)
        if isinstance(v, dict):
            v = v.get("ppl", v.get("value"))
        if v is None:
            return None
        m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", str(v))
        if m:
            return float(m.group(1))
        try:
            return float(str(v).split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    if ppl:
        print("\nperplexity (wikitext-2)")
        base_v = ppl_of(args.baseline)
        for b in builds:
            v = ppl_of(b)
            if v is None:
                continue
            delta = f"  ({(v / base_v - 1) * 100:+.1f}% vs {args.baseline})" if base_v else ""
            print(f"  {b.ljust(18)} {v:8.3f}{delta}")

    short = [k for k, c in n.items() if c < (d.get("reps") or 0)]
    if short:
        print(f"\nwarning: {len(short)} (build, threads, kind) cells have fewer than {d.get('reps')} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
