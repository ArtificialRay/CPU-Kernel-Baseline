#!/usr/bin/env python3
"""Log one claude-code fleet job to Weights & Biases (comprehensive).

Driven by `claude -p`, the agent loop runs inside the CLI (a black box), so we
can't auto-trace individual LLM calls (that needs the SDK + Weave). Instead we
parse the two artifacts the CLI DOES emit — the trajectory (`trajectory.jsonl`)
and the stream-json session log — into one rich W&B run per definition:

  per-evaluate (the metric curve = the spaghetti line):
    time/cycle speedup, best-so-far, ipc, cache-misses, max abs/rel error, status
  run summary:
    best_speedup, best_version + iteration it was found, starting (v1/scalar)
    speedup, weak-baseline signal (baseline-vs-scalar), iters-to-1x,
    iters-to-plateau, error taxonomy, cost, tokens, retries, wall-time,
    cost-per-speedup, cost-per-iteration, baseline hash
  tables/artifacts:
    winning kernel (browsable) + techniques used per version, and a versioned
    'kernel' artifact bundling every vN.cpp (+ vN.s if disassembled) + trajectory

One W&B run per definition; group all defs of a sweep under one `group` and tag
by model/dataset/isa/author so every teammate's runs merge into one shared
project. No-ops safely if wandb isn't installed or files are missing.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# SIMD / optimization idioms we detect in the agent's kernels (what did it do?).
TECHNIQUES = {
    "bf16_bfdot": r"\bsvbfdot",
    "bf16_bfmmla": r"bfmmla",
    "int8_dot": r"\bsvdot_|vdot|vsdot",
    "int8_mmla": r"\bsvmmla|smmla",
    "fma": r"\bsvmla|\bsvmad|vfma|svfma",
    "prefetch": r"__builtin_prefetch|\bprfm\b",
    "predication": r"svwhilelt|svptrue|svcnt[bwdh]",
    "neon": r"float32x4|\bvld1|\bvst1q?|vmlaq",
    "unroll": r"#pragma\s+(GCC\s+)?unroll",
    "cache_blocking": r"\b(block|tile|BLOCK|TILE)\w*\b",
}


def _find(d, key):
    if isinstance(d, dict):
        if d.get(key) is not None:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _locate_trajectory(results_dir: str, name: str):
    hits = glob.glob(f"{results_dir}/**/{name}/trajectory.jsonl", recursive=True)
    return Path(hits[0]) if hits else None


def parse_trajectory(path: Path):
    """Parse a trajectory into perf-eval rows + full taxonomy signal.

    `evaluate` runs in two modes and a given version is usually hit by both:
      - correctness: {status, max_absolute_error, max_relative_error} (no speedup)
      - perf:        {status, time_speedup_geomean, cycle_speedup_geomean, ...}
    The metric curve is driven by the perf rows, but the taxonomy (how many
    evals passed / were numerically wrong / crashed / timed out) and the
    numerical-error signal come from the correctness rows too — so we collect
    EVERY evaluate row's status here, not just the ones carrying a speedup.

    Returns (perf rows, per-version best speedup, ordered compile statuses,
             ordered evaluate statuses, per-version worst (abs, rel) error).
    """
    rows = []
    ver_best = {}          # "vN" -> best speedup seen for it
    compile_status = []    # ordered compile statuses (for error taxonomy)
    eval_status = []       # ordered statuses of EVERY evaluate row (all modes)
    ver_err = {}           # "vN" -> (worst max_abs_error, worst max_rel_error)
    cur_ver = None
    best = 0.0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("metrics") or {}
        if d.get("tool") == "compile":
            if m.get("version") is not None:
                cur_ver = f"v{m['version']}"
            else:
                mm = re.search(r"(v\d+)\.cpp", d.get("source_file") or "")
                if mm:
                    cur_ver = mm.group(1)
            compile_status.append(m.get("status") or "UNKNOWN")
        if d.get("tool") != "evaluate":
            continue
        eval_status.append(m.get("status") or "UNKNOWN")
        # capture numerical error from correctness-mode rows, keyed by version
        abs_e, rel_e = _find(d, "max_absolute_error"), _find(d, "max_relative_error")
        if cur_ver and (abs_e is not None or rel_e is not None):
            pa, pr = ver_err.get(cur_ver, (None, None))
            ver_err[cur_ver] = (max(x for x in (pa, abs_e) if x is not None),
                                max(x for x in (pr, rel_e) if x is not None))
        sp = _find(d, "time_speedup_geomean")
        if sp is None:
            continue
        best = max(best, sp)
        if cur_ver:
            ver_best[cur_ver] = max(ver_best.get(cur_ver, 0.0), sp)
        ve = ver_err.get(cur_ver, (None, None))
        rows.append({
            "version": cur_ver,
            "time_speedup": sp,
            "best_so_far": best,
            "cycle_speedup": _find(d, "cycle_speedup_geomean"),
            "ipc": _find(d, "ipc_mean"),
            "cache_misses": _find(d, "cache_misses_mean"),
            "max_abs_error": ve[0],   # from this version's correctness check
            "max_rel_error": ve[1],
            "status": m.get("status"),
        })
    return rows, ver_best, compile_status, eval_status, ver_err


def parse_session_log(path: Path):
    cost = turns = dur_ms = None
    retries = compile_errors = 0
    tok_in = tok_out = tok_cache_r = tok_cache_c = 0
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("subtype") == "api_retry":
            retries += 1
        if d.get("type") == "result":
            cost = d.get("total_cost_usd", cost)
            turns = d.get("num_turns", turns)
            dur_ms = d.get("duration_ms", dur_ms)
        u = _find(d, "usage")
        if isinstance(u, dict):
            tok_in += u.get("input_tokens", 0) or 0
            tok_out += u.get("output_tokens", 0) or 0
            tok_cache_r += u.get("cache_read_input_tokens", 0) or 0
            tok_cache_c += u.get("cache_creation_input_tokens", 0) or 0
        for c in (((d.get("message") or {}).get("content")) or []):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                t = c.get("content")
                if isinstance(t, str) and "kernel.cpp" in t and "error" in t.lower():
                    compile_errors += 1
    return {
        "cost_usd": cost, "num_turns": turns,
        "wall_time_s": round(dur_ms / 1000.0, 1) if dur_ms else None,
        "api_retries": retries, "session_compile_errors": compile_errors,
        "tokens_input": tok_in, "tokens_output": tok_out,
        "tokens_cache_read": tok_cache_r, "tokens_cache_created": tok_cache_c,
    }


def detect_techniques(text: str):
    return sorted(k for k, pat in TECHNIQUES.items() if re.search(pat, text))


def baseline_hash(dataset: str, name: str):
    """Content hash of the local baseline kernel this run was (nominally) scored
    against — so runs are comparable across baseline changes. Best-effort."""
    auth = {"llama.cpp": "baseline-llamacpp-arm", "ncnn": "baseline-ncnn-arm",
            "simd-loop": "reference"}.get(dataset)
    if not auth:
        return None
    hits = glob.glob(f"bench-trace/solutions/{dataset}/{auth}/*/{name}.json")
    if not hits:
        return None
    try:
        srcs = json.load(open(hits[0])).get("sources", [])
        k = next((s["content"] for s in srcs if s["path"] == "kernel.cpp"), "")
        return hashlib.sha256(k.encode()).hexdigest()[:12] if k else None
    except Exception:
        return None


def find_best_kernel(traj: Path, ver_best: dict):
    d = traj.parent
    if ver_best:
        ver, sp = max(ver_best.items(), key=lambda kv: kv[1])
        if (d / f"{ver}.cpp").exists():
            return d / f"{ver}.cpp", sp, ver
    cpps = sorted(d.glob("v*.cpp"), key=lambda p: int(re.search(r"v(\d+)", p.name).group(1)))
    return (cpps[-1], None, cpps[-1].stem) if cpps else (None, None, None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--isa", required=True)
    p.add_argument("--model", default="unknown")
    p.add_argument("--author", default="unknown")
    p.add_argument("--log-file", default="")
    p.add_argument("--results-dir", default="")
    p.add_argument("--trajectory", default="")
    p.add_argument("--project", default="arm-bench-kernels")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default=None)
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        print("[wandb_log_run] wandb not installed — skipping", file=sys.stderr)
        return 0

    traj = Path(args.trajectory) if args.trajectory else _locate_trajectory(args.results_dir, args.name)
    rows, ver_best, compile_status, eval_status, ver_err = (
        parse_trajectory(traj) if traj and traj.exists() else ([], {}, [], [], {}))
    sess = parse_session_log(Path(args.log_file)) if args.log_file and Path(args.log_file).exists() else {}
    if not rows and not sess:
        print(f"[wandb_log_run] no data for {args.name} — skipping", file=sys.stderr)
        return 0

    op_type = args.name.split("_")[0]
    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, reinit=True,
        tags=[args.model, args.dataset, args.isa, args.author, op_type],
        config={
            "definition": args.name, "dataset": args.dataset, "op_type": op_type,
            "isa": args.isa, "model": args.model, "author": args.author,
            "instance_type": os.environ.get("WANDB_INSTANCE_TYPE", "unknown"),
            "baseline_kernel_sha": baseline_hash(args.dataset, args.name),
        },
    )

    # ── per-evaluate curve ────────────────────────────────────────────────────
    for i, r in enumerate(rows, 1):
        prev = rows[i - 2]["best_so_far"] if i > 1 else 0.0
        wandb.log({"iteration": i, "marginal_gain": round(r["best_so_far"] - prev, 5),
                   **{k: v for k, v in r.items() if v is not None and k not in ("status", "version")}},
                  step=i)

    # ── derived summary ───────────────────────────────────────────────────────
    speeds = [r["time_speedup"] for r in rows]
    best = max(speeds) if speeds else None
    starting = speeds[0] if speeds else None                    # v1 = reference-scalar candidate
    base_vs_scalar = round(1.0 / starting, 3) if starting else None  # baseline speed vs naive scalar
    best_idx = (speeds.index(best) + 1) if best is not None else None
    iters_1x = next((i for i, r in enumerate(rows, 1) if r["best_so_far"] >= 1.0), None)
    iters_plateau = next((i for i, r in enumerate(rows, 1)
                          if best and r["best_so_far"] >= 0.98 * best), None)
    # taxonomy over EVERY evaluate row (both correctness + perf modes), so
    # numerical-wrong / crashed / timed-out attempts are actually counted.
    tax = Counter(s for s in eval_status if s)
    ctax = Counter(compile_status)
    worst_abs = max((e[0] for e in ver_err.values() if e[0] is not None), default=None)
    worst_rel = max((e[1] for e in ver_err.values() if e[1] is not None), default=None)
    cost = sess.get("cost_usd")

    run.summary.update({
        "best_speedup": best,
        "best_version_iteration": best_idx,
        "n_perf_evals": len(rows),                    # speedup-bearing evals (curve length)
        "n_evaluations": len(eval_status),            # total evaluate calls (all modes)
        "final_status": (eval_status[-1] if eval_status else None),
        "starting_speedup": starting,                 # scalar-ref vs baseline
        "baseline_vs_scalar": base_vs_scalar,         # >1 = baseline faster than naive; ~1 = WEAK baseline
        "weak_baseline": (base_vs_scalar is not None and base_vs_scalar < 2.0),
        "iters_to_parity": iters_1x,
        "iters_to_plateau": iters_plateau,
        "n_passed": tax.get("PASSED", 0),
        "n_incorrect": tax.get("INCORRECT_NUMERICAL", 0),
        "n_runtime_error": tax.get("RUNTIME_ERROR", 0),
        "n_timeout": tax.get("TIMEOUT", 0),
        "n_compile_error": ctax.get("COMPILE_ERROR", 0),
        "worst_max_abs_error": worst_abs,
        "worst_max_rel_error": worst_rel,
        "cost_per_speedup": round(cost / best, 4) if (cost and best) else None,
        "cost_per_eval": round(cost / len(rows), 4) if (cost and rows) else None,
        **{k: v for k, v in sess.items() if v is not None},
    })

    # ── winning kernel + techniques-per-version + artifact ────────────────────
    if traj and traj.exists():
        best_cpp, best_sp, ver = find_best_kernel(traj, ver_best)
        cpps = sorted(traj.parent.glob("v*.cpp"),
                      key=lambda p: int(re.search(r"v(\d+)", p.name).group(1)))
        if best_cpp and best_cpp.exists():
            techs = detect_techniques(best_cpp.read_text())
            run.summary["best_kernel_version"] = ver
            run.summary["best_kernel_techniques"] = ", ".join(techs)
            # techniques introduced across versions (the optimization story)
            tech_tbl = wandb.Table(columns=["version", "speedup", "techniques", "source"])
            for cpp in cpps:
                v = cpp.stem
                tech_tbl.add_data(v, ver_best.get(v), ", ".join(detect_techniques(cpp.read_text())),
                                  cpp.read_text())
            run.log({"kernels": tech_tbl})
            art = wandb.Artifact(f"{args.name}-kernels", type="kernel",
                                 metadata={"best_version": ver, "best_speedup": best_sp,
                                           "techniques": techs})
            for cpp in cpps:
                art.add_file(str(cpp))
            for s in sorted(traj.parent.glob("v*.s")):   # disassembly, if the agent produced it
                art.add_file(str(s))
            art.add_file(str(traj), name="trajectory.jsonl")
            run.log_artifact(art, aliases=["best", ver] if ver else ["best"])

    run.finish()
    print(f"[wandb_log_run] logged {args.name}: best={best} evals={len(rows)} "
          f"weak_baseline={base_vs_scalar is not None and base_vs_scalar < 2.0} "
          f"cost={cost} tokens_out={sess.get('tokens_output')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
