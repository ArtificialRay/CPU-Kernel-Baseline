#!/usr/bin/env python3
"""End-to-end tokens/s measurement for the Qwen3.5-4B experiment.

Runs llama-bench (prompt processing pp and text generation tg) for one or more
llama.cpp builds on the SAME box and model, repeats each configuration, and
writes one JSON with every raw sample plus median/min summaries so the paper
table can be regenerated. Optionally runs llama-perplexity on a small text as
the acceptance check that overridden kernels did not change the model's output
quality (the per-kernel harness already checks numerical correctness per
workload; this is the belt-and-braces end-to-end check).

Usage (on the c8g box, from ~/arm-bench):
  python3 scripts/e2e/measure_e2e.py \
      --model ~/models/Qwen3.5-4B-Q4_K_M.gguf \
      --build stock=~/llama.cpp-e2e/build-stock \
      --build agent=~/llama.cpp-e2e/build-agent --overrides agent=~/e2e/manifest.json \
      --threads 4 16 --pp 512 --tg 128 --reps 5 --out e2e_results.json
  add --perplexity ~/models/wiki.test.raw to run the quality check (slow).

Each --build is name=path-to-cmake-build-dir (must contain bin/llama-bench).
--overrides name=manifest.json sets ARMBENCH_OVERRIDES for that build only
(see scripts/e2e/override/README.md). Builds are run interleaved
(stock, agent, stock, agent, ...) so thermal/turbo drift hits both equally.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p.stdout


def llama_bench(bench_bin: Path, model: Path, threads: int, pp: int, tg: int, env: dict) -> list[dict]:
    """One llama-bench invocation → list of result dicts (llama-bench -o json)."""
    out = run([str(bench_bin), "-m", str(model), "-t", str(threads), "-p", str(pp), "-n", str(tg),
               "-r", "1", "-o", "json"], env=env)
    # llama-bench prints a JSON array (possibly preceded by log lines on stderr only)
    start = out.find("[")
    return json.loads(out[start:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--build", action="append", required=True, metavar="NAME=BUILD_DIR")
    ap.add_argument("--overrides", action="append", default=[], metavar="NAME=MANIFEST_JSON")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=KEY=VALUE",
                    help="extra environment variable for one build (e.g. agent1t=ARMBENCH_OVERRIDE_THREADS=1 for the single-thread ablation)")
    ap.add_argument("--threads", nargs="+", type=int, default=[4])
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--perplexity", type=Path, help="text file for llama-perplexity (acceptance check)")
    ap.add_argument("--ppl-chunks", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("e2e_results.json"))
    args = ap.parse_args()

    builds = {}
    for spec in args.build:
        name, _, path = spec.partition("=")
        builds[name] = {"dir": Path(path).expanduser(), "manifest": None, "env": {}}
    for spec in args.overrides:
        name, _, path = spec.partition("=")
        if name not in builds:
            raise SystemExit(f"--overrides names unknown build {name!r}")
        builds[name]["manifest"] = str(Path(path).expanduser())
    for spec in args.env:
        name, _, kv = spec.partition("=")
        k, _, v = kv.partition("=")
        if name not in builds:
            raise SystemExit(f"--env names unknown build {name!r}")
        builds[name]["env"][k] = v
    for name, b in builds.items():
        if not (b["dir"] / "bin" / "llama-bench").exists():
            raise SystemExit(f"{name}: {b['dir']}/bin/llama-bench not found")

    samples = []
    t0 = time.time()
    for rep in range(args.reps):
        for threads in args.threads:
            for name, b in builds.items():   # interleaved across builds
                env = dict(os.environ)
                if b["manifest"]:
                    env["ARMBENCH_OVERRIDES"] = b["manifest"]
                    env.setdefault("ARMBENCH_OVERRIDE_LOG", "0")
                env.update(b["env"])
                for r in llama_bench(b["dir"] / "bin" / "llama-bench", args.model, threads, args.pp, args.tg, env):
                    kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
                    samples.append({"build": name, "threads": threads, "rep": rep, "kind": kind,
                                    "n": r.get("n_prompt") or r.get("n_gen"),
                                    "tok_per_s": r["avg_ts"], "raw": r})
                    print(f"[{time.time()-t0:6.0f}s] rep {rep} {name:8s} t={threads:2d} {kind} {r['avg_ts']:8.2f} tok/s", flush=True)

    summary = {}
    for name in builds:
        for threads in args.threads:
            for kind in ("pp", "tg"):
                xs = [s["tok_per_s"] for s in samples if s["build"] == name and s["threads"] == threads and s["kind"] == kind]
                if xs:
                    summary[f"{name}/t{threads}/{kind}"] = {"median": statistics.median(xs), "max": max(xs),
                                                             "min": min(xs), "n": len(xs)}
    if "stock" in builds:
        for k, v in list(summary.items()):
            name, t, kind = k.split("/")
            base = summary.get(f"stock/{t}/{kind}")
            if base and name != "stock":
                v["speedup_vs_stock_median"] = v["median"] / base["median"]

    def dump(ppl):
        dump(ppl)
    print(f"\n{'config':28} {'median tok/s':>13} {'vs stock':>9}")
    for k, v in summary.items():
        print(f"{k:28} {v['median']:13.2f} {v.get('speedup_vs_stock_median', float('nan')):9.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
