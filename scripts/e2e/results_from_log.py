#!/usr/bin/env python3
"""Rebuild a measure_e2e.py results JSON from its streamed stdout log (the
"[  123s] rep 0 stock    t= 1 pp    13.73 tok/s" lines), optionally merging a
perplexity log ("=== <build> ..." headers followed by llama-perplexity's
"Final estimate: PPL = ..." line). Same schema as measure_e2e.py writes, so
wandb_log_e2e.py accepts it. Used when the JSON itself was lost.

  python3 scripts/e2e/results_from_log.py measure_stdout.log --out e2e_results.json \
      [--perplexity-log perplexity.log] [--manifest manifest.json] [--model ...] [--host ...]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

SAMPLE = re.compile(r"\[\s*(\d+)s\] rep (\d+) (\S+)\s+t=\s*(\d+) (pp|tg)\s+([\d.]+) tok/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stdout_log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perplexity-log", type=Path)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--host", default="")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--note", default="reconstructed from the streamed stdout log")
    args = ap.parse_args()

    samples = []
    for l in args.stdout_log.read_text().splitlines():
        m = SAMPLE.match(l)
        if m:
            samples.append({"build": m.group(3), "threads": int(m.group(4)), "rep": int(m.group(2)), "kind": m.group(5),
                            "n": args.pp if m.group(5) == "pp" else args.tg, "tok_per_s": float(m.group(6)),
                            "t_elapsed_s": int(m.group(1))})
    builds = list(dict.fromkeys(s["build"] for s in samples))
    threads = sorted({s["threads"] for s in samples})
    summary = {}
    for b in builds:
        for t in threads:
            for k in ("pp", "tg"):
                xs = [s["tok_per_s"] for s in samples if s["build"] == b and s["threads"] == t and s["kind"] == k]
                if xs:
                    summary[f"{b}/t{t}/{k}"] = {"median": statistics.median(xs), "max": max(xs), "min": min(xs), "n": len(xs)}
    for key, v in summary.items():
        b, t, k = key.split("/")
        base = summary.get(f"stock/{t}/{k}")
        if base and b != "stock":
            v["speedup_vs_stock_median"] = v["median"] / base["median"]

    ppl = {}
    if args.perplexity_log and args.perplexity_log.exists():
        cur = None
        for l in args.perplexity_log.read_text().splitlines():
            m = re.match(r"=== (\S+)", l)
            if m and m.group(1) not in ("PPL",):
                cur = m.group(1)
            elif cur and ("PPL" in l or "estimate" in l.lower() or "error" in l.lower()):
                ppl[cur] = l.strip()

    reps = max((s["rep"] for s in samples), default=-1) + 1
    manifest = args.manifest
    build_info = {b: {"dir": "", "manifest": manifest if b.startswith("agent") else None,
                      "env": {"ARMBENCH_OVERRIDE_THREADS": "1"} if b == "agent1t" else {}} for b in builds}
    res = {"model": args.model, "pp": args.pp, "tg": args.tg, "reps": reps, "builds": build_info, "host": args.host,
           "summary": summary, "perplexity": ppl, "samples": samples, "note": args.note}
    args.out.write_text(json.dumps(res, indent=1))
    print(f"{len(samples)} samples, {len(summary)} configs, reps={reps}, perplexity for {list(ppl)} -> {args.out}")


if __name__ == "__main__":
    main()
