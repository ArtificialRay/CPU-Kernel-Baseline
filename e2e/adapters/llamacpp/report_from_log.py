#!/usr/bin/env python3
"""Rebuild a measure.py results JSON from its streamed stdout log (the
"[  123s] rep 0 stock    t= 1 pp    13.73 tok/s" lines), optionally merging a
perplexity log ("=== <build> ..." headers followed by llama-perplexity's
"Final estimate: PPL = ..." line). Same schema as measure.py writes, so
report.py accepts it. Used when the JSON itself was lost.

  results_from_log("measure_stdout.log", "e2e_results.json", perplexity_log="perplexity.log",
                   manifest="manifest.json", model="...", host="...")
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

SAMPLE = re.compile(r"\[\s*(\d+)s\] rep (\d+) (\S+)\s+t=\s*(\d+) (pp|tg)\s+([\d.]+) tok/s")


def results_from_log(stdout_log: Path, out: Path, *, perplexity_log: Path | None = None,
                     manifest: str | None = None, model: str = "", host: str = "", pp: int = 512,
                     tg: int = 128, note: str = "reconstructed from the streamed stdout log") -> dict:
    """Rebuild a measure() results file from its streamed stdout log (and optionally the
    perplexity log), for when the JSON itself was lost. Writes `out`, returns the results."""
    stdout_log, out = Path(stdout_log), Path(out)
    perplexity_log = Path(perplexity_log) if perplexity_log else None
    samples = []
    for l in stdout_log.read_text().splitlines():
        m = SAMPLE.match(l)
        if m:
            samples.append({"build": m.group(3), "threads": int(m.group(4)), "rep": int(m.group(2)), "kind": m.group(5),
                            "n": pp if m.group(5) == "pp" else tg, "tok_per_s": float(m.group(6)),
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
    if perplexity_log and perplexity_log.exists():
        cur = None
        for l in perplexity_log.read_text().splitlines():
            m = re.match(r"=== (\S+)", l)
            if m and m.group(1) not in ("PPL",):
                cur = m.group(1)
            elif cur and ("PPL" in l or "estimate" in l.lower() or "error" in l.lower()):
                ppl[cur] = l.strip()

    reps = max((s["rep"] for s in samples), default=-1) + 1
    build_info = {b: {"dir": "", "manifest": manifest if b.startswith("agent") else None,
                      "env": {"ARMBENCH_OVERRIDE_THREADS": "1"} if b == "agent1t" else {}} for b in builds}
    res = {"model": model, "pp": pp, "tg": tg, "reps": reps, "builds": build_info, "host": host,
           "summary": summary, "perplexity": ppl, "samples": samples, "note": note}
    out.write_text(json.dumps(res, indent=1))
    print(f"{len(samples)} samples, {len(summary)} configs, reps={reps}, perplexity for {list(ppl)} -> {out}")
    return res
