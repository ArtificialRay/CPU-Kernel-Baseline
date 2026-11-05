#!/usr/bin/env python3
"""Render a measure.py results JSON as one results table.

Speed and perplexity come from the same binaries in one run, so they belong in one table:
a speedup is only meaningful next to what it cost in model quality.

    render_report("e2e_results.json", baseline="stock")      # or: python -m e2e evaluate --steps report
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


def medians(samples):
    out = {}
    for s in samples:
        out.setdefault((s["build"], s["threads"], s["kind"]), []).append(s["tok_per_s"])
    return {k: statistics.median(v) for k, v in out.items()}, {k: len(v) for k, v in out.items()}


def render_report(results, baseline: str = "stock") -> str:
    """The table of a measure() results file (a path, or the dict measure() returned):
    throughput, speedup vs `baseline`, and perplexity, in one place because a speedup only
    means something next to what it cost in model quality."""
    lines: list[str] = []
    say = lines.append
    d = results if isinstance(results, dict) else json.loads(Path(results).read_text())
    med, n = medians(d.get("samples", []))
    builds = list(dict.fromkeys(s["build"] for s in d.get("samples", [])))
    threads = sorted({s["threads"] for s in d.get("samples", [])})
    ppl = d.get("perplexity", {}) or {}

    say(f"host: {d.get('host','?')}")
    say(f"model: {Path(d.get('model','?')).name}   pp={d.get('pp')} tg={d.get('tg')} reps={d.get('reps')}\n")

    say("absolute throughput (tok/s, median of reps)")
    hdr = "build".ljust(18) + "".join(f"  t={t:<3} pp      tg" for t in threads)
    say(hdr); say("-" * len(hdr))
    for b in builds:
        row = b.ljust(18)
        for t in threads:
            pp = med.get((b, t, "pp")); tg = med.get((b, t, "tg"))
            row += f"  {pp:7.1f} {tg:7.2f}" if pp and tg else "        -       -"
        say(row)

    if baseline in builds:
        say(f"\nspeedup vs {baseline}")
        say(hdr); say("-" * len(hdr))
        for b in builds:
            if b == baseline:
                continue
            row = b.ljust(18)
            for t in threads:
                num_pp, den_pp = med.get((b, t, "pp")), med.get((baseline, t, "pp"))
                num_tg, den_tg = med.get((b, t, "tg")), med.get((baseline, t, "tg"))
                row += (f"  {num_pp/den_pp:6.2f}x {num_tg/den_tg:6.2f}x"
                        if num_pp and den_pp and num_tg and den_tg else "        -       -")
            say(row)

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
        say("\nperplexity (wikitext-2)")
        base_v = ppl_of(baseline)
        for b in builds:
            v = ppl_of(b)
            if v is None:
                continue
            delta = f"  ({(v / base_v - 1) * 100:+.1f}% vs {baseline})" if base_v else ""
            say(f"  {b.ljust(18)} {v:8.3f}{delta}")

    short = [k for k, c in n.items() if c < (d.get("reps") or 0)]
    if short:
        say(f"\nwarning: {len(short)} (build, threads, kind) cells have fewer than {d.get('reps')} samples")
    return "\n".join(lines)
