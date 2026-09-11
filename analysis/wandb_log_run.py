#!/usr/bin/env python3
"""Log one fleet job to Weights & Biases (comprehensive).

Imported directly by test_scripts/bench_fleet.py (log_run_to_wandb()) —
not invoked as a subprocess. See analysis/README.md for what every field
below means and where to find it on the W&B run page.

Parses the trajectory (`trajectory.jsonl`, written by mcp_app's
TrajectoryWriter — same format regardless of which harness drove the
session) into one rich W&B run per definition:

  per-evaluate (the metric curve = the spaghetti line):
    time/cycle speedup, best-so-far, ipc, cache-misses, max abs/rel error, status
  run summary:
    best_speedup, best_version + iteration it was found, starting (v1/scalar)
    speedup, weak-baseline signal (baseline-vs-scalar), iters-to-1x,
    iters-to-plateau, error taxonomy, baseline hash
  tables/artifacts:
    winning kernel (browsable) + techniques used per version, and a versioned
    'kernel' artifact bundling every vN.cpp (+ vN.s if disassembled) + trajectory

One W&B run per definition; group all defs of a sweep under one `group` and tag
by model/dataset/isa/author so every teammate's runs merge into one shared
project. No-ops safely if wandb isn't installed, or if there's nothing to log.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

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
            "turn": d.get("turn"),
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


def log_run_to_wandb(
    *, name: str, dataset: str, isa: str, model: str, author: str,
    trajectory_path: Optional[Path],
    project: str = "arm-bench-kernels",
    entity: Optional[str] = None,
    group: Optional[str] = None,
) -> None:
    """Log one definition's run to W&B. Never raises: the caller
    (bench_fleet.py) wraps this in a try/except so a wandb hiccup never
    aborts the batch."""
    try:
        import wandb
    except ImportError:
        print("[wandb_log_run] wandb not installed — skipping", file=sys.stderr)
        return

    traj = trajectory_path
    rows, ver_best, compile_status, eval_status, ver_err = (
        parse_trajectory(traj) if traj and traj.exists() else ([], {}, [], [], {}))
    if not rows:
        print(f"[wandb_log_run] no data for {name} — skipping", file=sys.stderr)
        return

    op_type = name.split("_")[0]
    # Stable across repeated logging of the same (group, author, dataset,
    # isa, definition) — e.g. re-synced across --until-complete rounds — so
    # wandb.init(resume="allow") reattaches to and overwrites that one run
    # instead of creating a fresh one every time.
    run_id = hashlib.sha1(f"{group or ''}:{author}:{dataset}:{isa}:{name}".encode()).hexdigest()[:16]
    run = wandb.init(
        project=project, entity=entity, group=group,
        id=run_id, resume="allow", reinit=True, allow_val_change=True,
        name=name,
        tags=[model, dataset, isa, author, op_type],
        config={
            "definition": name, "dataset": dataset, "op_type": op_type,
            "isa": isa, "model": model, "author": author,
            "instance_type": os.environ.get("WANDB_INSTANCE_TYPE", "unknown"),
            "baseline_kernel_sha": baseline_hash(dataset, name),
        },
    )

    # ── per-evaluate curve ────────────────────────────────────────────────────
    # step is the record's total tool-call index for this definition,
    # compile/disassemble/submit included, which is different to `i`: number of evaluate
    # tool call with speedup provided
    for i, r in enumerate(rows, 1):
        prev = rows[i - 2]["best_so_far"] if i > 1 else 0.0
        wandb.log({"iteration": i, "marginal_gain": round(r["best_so_far"] - prev, 5),
                   **{k: v for k, v in r.items() if v is not None and k not in ("status", "version", "turn")}},
                  step=r["turn"])

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
            art = wandb.Artifact(f"{name}-kernels", type="kernel",
                                 metadata={"best_version": ver, "best_speedup": best_sp,
                                           "techniques": techs})
            for cpp in cpps:
                art.add_file(str(cpp))
            for s in sorted(traj.parent.glob("v*.s")):   # disassembly, if the agent produced it
                art.add_file(str(s))
            art.add_file(str(traj), name="trajectory.jsonl")
            run.log_artifact(art, aliases=["best", ver] if ver else ["best"])

    run.finish()
    print(f"[wandb_log_run] logged {name}: best={best} evals={len(rows)} "
          f"weak_baseline={base_vs_scalar is not None and base_vs_scalar < 2.0}")


def _cli_main() -> int:
    """Thin manual-backfill entry point: `python analysis/wandb_log_run.py
    --name ... --results-dir ...` re-logs one already-finished job. Not used
    by bench_fleet.py (which calls log_run_to_wandb() directly) — this is
    for manually re-running or debugging the W&B logging for one definition
    after the fact."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--isa", required=True)
    p.add_argument("--model", default="unknown")
    p.add_argument("--author", default="unknown")
    p.add_argument("--results-dir", default="")
    p.add_argument("--trajectory", default="")
    p.add_argument("--project", default="arm-bench-kernels")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default=None)
    args = p.parse_args()

    traj = Path(args.trajectory) if args.trajectory else _locate_trajectory(args.results_dir, args.name)
    log_run_to_wandb(
        name=args.name, dataset=args.dataset, isa=args.isa, model=args.model, author=args.author,
        trajectory_path=traj,
        project=args.project, entity=args.entity, group=args.group,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
