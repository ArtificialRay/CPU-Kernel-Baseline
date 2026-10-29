#!/usr/bin/env python3
"""End-to-end tokens/s measurement for the Qwen3.5-4B experiment.

Runs llama-bench (prompt processing pp and text generation tg) for one or more
llama.cpp builds on the SAME box and model, repeats each configuration, and
writes one JSON with every raw sample plus median/min summaries so the paper
table can be regenerated. Optionally runs llama-perplexity on a small text as
the acceptance check that overridden kernels did not change the model's output
quality (the per-kernel harness already checks numerical correctness per
workload; this is the belt-and-braces end-to-end check).

Run it on the c8g box (the repo checkout there), e.g.
  python3 -m e2e evaluate --model qwen3.5-4b --steps measure --threads 4 16 --pp 512 --tg 128 \
      --reps 5 --fa off [--perplexity ~/wiki.test.raw]
which calls measure() with the stock / norepack / agent builds under ~/llama.cpp-e2e and the
run directory's manifest.json as the agent's override.

fa is required: llama-bench's default (auto) picks flash attention on, which is
~1.6x slower than off on Graviton and silently changes every prefill ratio.
Before any timing, every build with --overrides is run once with
ARMBENCH_OVERRIDE_LOG=1 and must hit every mul_mat shape in its manifest; all
override runs set ARMBENCH_OVERRIDE_STRICT=1 so a dlopen failure aborts instead of
quietly measuring the no-override build.

Each build is name -> path-to-cmake-build-dir (must contain bin/llama-bench).
An override name -> manifest.json sets ARMBENCH_OVERRIDES for that build only
(see e2e/adapters/llamacpp/override/README.md). Builds are run interleaved
(stock, agent, stock, agent, ...) so thermal/turbo drift hits both equally.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from e2e.errors import StepFailed


def run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise StepFailed(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p.stdout


def llama_bench(bench_bin: Path, model: Path, threads: int, pp: int, tg: int, env: dict,
                fa: str, ub: int | None) -> list[dict]:
    """One llama-bench invocation → list of result dicts (llama-bench -o json)."""
    cmd = [str(bench_bin), "-m", str(model), "-t", str(threads), "-p", str(pp), "-n", str(tg),
           "-fa", fa, "-r", "1", "-o", "json"]
    if ub:
        cmd += ["-ub", str(ub)]
    out = run(cmd, env=env)
    # llama-bench prints a JSON array (possibly preceded by log lines on stderr only)
    start = out.find("[")
    return json.loads(out[start:])


HIT_RX = re.compile(r"hit mul_mat type=(\S+) K=(\d+) N=(\d+)")


def verify_hook(name: str, b: dict, model: Path, threads: int, fa: str, ub: int | None) -> dict:
    """Short pp+tg run with logging on; every mul_mat entry of the manifest must be hit."""
    want = {(k["type"], int(k["K"]), int(k["N"]))
            for k in json.loads(Path(b["manifest"]).read_text())["kernels"] if k.get("op") == "mul_mat"}
    env = {**os.environ, **b["env"], "ARMBENCH_OVERRIDES": b["manifest"], "ARMBENCH_OVERRIDE_LOG": "1",
           "ARMBENCH_OVERRIDE_STRICT": "1"}
    cmd = [str(b["dir"] / "bin" / "llama-bench"), "-m", str(model), "-t", str(threads),
           "-p", "64", "-n", "4", "-fa", fa, "-r", "1"]
    if ub:
        cmd += ["-ub", str(ub)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    hit = {(t, int(k), int(n)) for t, k, n in HIT_RX.findall(p.stderr)}
    missing = sorted(want - hit)
    if p.returncode != 0 or missing:
        sys.stderr.write(p.stderr[-4000:])
        raise StepFailed(f"{name}: override hook check failed (rc={p.returncode}, {len(hit & want)}/{len(want)} "
                         f"manifest shapes hit; missing {missing})")
    print(f"hook {name}: {len(want)}/{len(want)} manifest shapes intercepted", flush=True)
    return {"expected": len(want), "hit": len(hit & want)}


def measure(
    model: Path, builds: dict, *, overrides: dict | None = None, env: dict | None = None,
    threads=(4,), pp: int = 512, tg: int = 128, reps: int = 5, fa: str, ub: int | None = None,
    perplexity: Path | None = None, ppl_chunks: int = 64, out: Path = Path("e2e_results.json"),
) -> dict:
    """llama-bench (pp / tg) and optionally llama-perplexity for every build, interleaved, on THIS
    machine (it has to be the Graviton box); writes the raw samples and summary to `out` and returns them.

    builds      name -> cmake build dir (must contain bin/llama-bench)
    overrides   name -> manifest.json for the builds that run the ARMBENCH_OVERRIDES hook
    env         name -> {KEY: VALUE} extra environment for one build (e.g. the single-thread ablation
                {"agent1t": {"ARMBENCH_OVERRIDE_THREADS": "1"}})
    fa          "on" / "off", passed to llama-bench and llama-perplexity; never left to auto
    ub          micro-batch size (-ub); llama.cpp's default (512) if None
    perplexity  text file for llama-perplexity (the quality check); ppl_chunks: 64 resolves
                sub-1% differences, 8 only catches catastrophic regressions
    """
    model, out, threads = Path(model), Path(out), list(threads)
    builds = {name: {"dir": Path(path).expanduser(), "manifest": None, "env": {}} for name, path in builds.items()}
    for name, path in (overrides or {}).items():
        if name not in builds:
            raise StepFailed(f"overrides names unknown build {name!r}")
        builds[name]["manifest"] = str(Path(path).expanduser())
    for name, extra in (env or {}).items():
        if name not in builds:
            raise StepFailed(f"env names unknown build {name!r}")
        builds[name]["env"].update(extra)
    for name, b in builds.items():
        if not (b["dir"] / "bin" / "llama-bench").exists():
            raise StepFailed(f"{name}: {b['dir']}/bin/llama-bench not found")

    for b in builds.values():
        if b["manifest"]:
            b["env"].setdefault("ARMBENCH_OVERRIDE_STRICT", "1")
    hook = {name: verify_hook(name, b, model, max(threads), fa, ub)
            for name, b in builds.items() if b["manifest"]}

    samples = []
    t0 = time.time()
    for rep in range(reps):
        for nthreads in threads:
            for name, b in builds.items():   # interleaved across builds
                env = dict(os.environ)
                if b["manifest"]:
                    env["ARMBENCH_OVERRIDES"] = b["manifest"]
                    env.setdefault("ARMBENCH_OVERRIDE_LOG", "0")
                env.update(b["env"])
                for r in llama_bench(b["dir"] / "bin" / "llama-bench", model, nthreads, pp, tg, env,
                                     fa, ub):
                    kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
                    samples.append({"build": name, "threads": nthreads, "rep": rep, "kind": kind,
                                    "n": r.get("n_prompt") or r.get("n_gen"),
                                    "tok_per_s": r["avg_ts"], "raw": r})
                    print(f"[{time.time()-t0:6.0f}s] rep {rep} {name:8s} t={nthreads:2d} {kind} {r['avg_ts']:8.2f} tok/s", flush=True)

    summary = {}
    for name in builds:
        for nthreads in threads:
            for kind in ("pp", "tg"):
                xs = [s["tok_per_s"] for s in samples if s["build"] == name and s["threads"] == nthreads and s["kind"] == kind]
                if xs:
                    summary[f"{name}/t{nthreads}/{kind}"] = {"median": statistics.median(xs), "max": max(xs),
                                                             "min": min(xs), "n": len(xs)}
    if "stock" in builds:
        for k, v in list(summary.items()):
            name, t, kind = k.split("/")
            base = summary.get(f"stock/{t}/{kind}")
            if base and name != "stock":
                v["speedup_vs_stock_median"] = v["median"] / base["median"]

    def dump(ppl):
        result = {"model": str(model), "pp": pp, "tg": tg, "reps": reps,
                  "fa": fa, "ub": ub, "ppl_chunks": ppl_chunks if perplexity else None,
                  "hook": hook,
                  "builds": {n: {"dir": str(b["dir"]), "manifest": b["manifest"], "env": b["env"]} for n, b in builds.items()},
                  "host": os.uname().nodename, "summary": summary, "perplexity": ppl, "samples": samples}
        out.write_text(json.dumps(result, indent=1))
        return result

    result = dump({})   # bench results are safe on disk before the (slow, optional) quality check
    print(f"wrote {out} (bench only)", flush=True)

    ppl = {}
    if perplexity:
        for name, b in builds.items():
            env = dict(os.environ)
            if b["manifest"]:
                env["ARMBENCH_OVERRIDES"] = b["manifest"]
            env.update(b["env"])
            try:
                cmd = [str(b["dir"] / "bin" / "llama-perplexity"), "-m", str(model), "-f", str(perplexity),
                       "--chunks", str(ppl_chunks), "-t", str(max(threads)), "-fa", fa]
                if ub:
                    cmd += ["-ub", str(ub)]
                p = subprocess.run(cmd,
                                   capture_output=True, text=True, env=env, timeout=7200)
                output = p.stdout + p.stderr
                line = [l for l in output.splitlines() if "estimate" in l.lower() or "PPL" in l or "Final" in l]
                ppl[name] = (line[-1] if line else output[-300:]).strip() if p.returncode == 0 else f"FAILED rc={p.returncode}: {output[-300:].strip()}"
            except Exception as e:  # never lose the bench results to the quality check
                ppl[name] = f"FAILED: {e}"
            print(f"perplexity {name}: {ppl[name]}", flush=True)
        result = dump(ppl)

    print(f"\n{'config':28} {'median tok/s':>13} {'vs stock':>9}")
    for k, v in summary.items():
        print(f"{k:28} {v['median']:13.2f} {v.get('speedup_vs_stock_median', float('nan')):9.3f}")
    print(f"wrote {out}", flush=True)
    return result
