#!/usr/bin/env python3
"""Log one fleet job to Weights & Biases (comprehensive).

Imported directly by test_scripts/bench_fleet.py (log_run_to_wandb()) —
not invoked as a subprocess. See analysis/README.md for what every field
below means and where to find it on the W&B run page.

Parses two artifacts into one rich W&B run per definition:
  - the trajectory (`trajectory.jsonl`, written by mcp_app's TrajectoryWriter
    — same format regardless of which harness drove the session; its rows
    are server-stamped with `ts` + `elapsed_s`, which is where the per-turn
    latency curve comes from for every harness alike)
  - a SessionMetrics object (test_scripts/harness_adapters.py) — harness-
    specific session telemetry (cost, tokens, retries), already parsed
    by the calling HarnessAdapter before this module ever sees it. This
    module has no harness-format knowledge of its own. Cost a harness
    doesn't report is estimated here from its token counts (litellm pricing
    table) and flagged via `cost_source`.

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
project. No-ops safely if wandb isn't installed, or if there's nothing to log.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
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
             ordered evaluate statuses, per-version worst (abs, rel) error,
             timeline of (tool, ts, elapsed_s) for EVERY row — ts/elapsed_s are
             None in trajectories written before the server stamped them).
    """
    rows = []
    ver_best = {}          # "vN" -> best speedup seen for it
    compile_status = []    # ordered compile statuses (for error taxonomy)
    eval_status = []       # ordered statuses of EVERY evaluate row (all modes)
    ver_err = {}           # "vN" -> (worst max_abs_error, worst max_rel_error)
    timeline = []          # (tool, ts datetime|None, elapsed_s|None) per row
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
        timeline.append((d.get("tool") or "", _parse_ts(d.get("ts")), d.get("elapsed_s")))
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
    return rows, ver_best, compile_status, eval_status, ver_err, timeline


def _parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


@dataclass
class _TrajTurn:
    """Duck-types harness_adapters.TurnRow (same attributes) without importing
    that module's heavy dependency chain here."""
    total_s: float
    llm_s: float
    tool_s: float
    tool: str = ""


def turn_rows_from_trajectory(timeline) -> list:
    """Per-turn latency from the server-stamped trajectory — one clock and one
    definition of "a turn" for every harness. A turn = one tool call:
    total_s runs from the previous tool's row to this one (model thinking +
    transport + this tool), tool_s is this tool's own wall time (elapsed_s),
    llm_s the remainder. Empty for pre-timestamp trajectories, in which case
    log_run_to_wandb() falls back to the harness's own SessionMetrics.turn_rows."""
    out, prev = [], None
    for tool, ts, elapsed in timeline:
        if ts is None:
            continue
        if prev is not None:
            total = (ts - prev).total_seconds()
            if 0 <= total < 7200:
                tool_s = float(elapsed or 0.0)
                out.append(_TrajTurn(total_s=round(total, 1), tool_s=round(tool_s, 1),
                                     llm_s=round(max(total - tool_s, 0.0), 1), tool=tool))
        prev = ts
    return out


def estimate_cost(model: str, tok_in: int, tok_out: int) -> Optional[float]:
    """Best-effort $ from litellm's pricing table, for harnesses that don't
    report cost (nanobot; own when litellm couldn't price a call). Cache
    discounts are ignored, so it's an upper-bound-ish estimate — logged with
    cost_source='litellm-estimate'. None if litellm is missing or the model
    is unpriced (openrouter/ prefixes are stripped; bare model names tried)."""
    if not model or not (tok_in or tok_out):
        return None
    try:
        import litellm
    except ImportError:
        return None
    m = re.sub(r"^openrouter/", "", model)
    for cand in (m, m.split("/")[-1]):
        try:
            pi, po = litellm.cost_per_token(model=cand, prompt_tokens=tok_in or 0,
                                            completion_tokens=tok_out or 0)
        except Exception:  # noqa: BLE001 — unknown model → try the next spelling
            continue
        if pi or po:
            return round(pi + po, 4)
    return None


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
    session,  # test_scripts.harness_adapters.SessionMetrics
    project: str = "arm-bench-kernels",
    entity: Optional[str] = None,
    group: Optional[str] = None,
) -> None:
    """Log one definition's run to W&B. `session` carries whatever
    session-level telemetry the calling HarnessAdapter could extract from
    its own log format — fields it couldn't extract are left at their
    SessionMetrics defaults (None/0/empty) and simply don't appear in the
    summary. Never raises: the caller (bench_fleet.py) wraps this in a
    try/except so a wandb hiccup never aborts the batch."""
    try:
        import wandb
    except ImportError:
        print("[wandb_log_run] wandb not installed — skipping", file=sys.stderr)
        return

    traj = trajectory_path
    rows, ver_best, compile_status, eval_status, ver_err, timeline = (
        parse_trajectory(traj) if traj and traj.exists() else ([], {}, [], [], {}, []))
    # per-turn latency: prefer the server-stamped trajectory (same clock, same
    # turn definition for every harness); older trajectories fall back to what
    # the harness's own log could tell us.
    turn_rows = turn_rows_from_trajectory(timeline)
    turn_timing_source = "trajectory" if turn_rows else ("harness-log" if session.turn_rows else None)
    turn_rows = turn_rows or session.turn_rows
    if not rows and not turn_rows and session.cost_usd is None:
        print(f"[wandb_log_run] no data for {name} — skipping", file=sys.stderr)
        return
    harness = getattr(session, "harness", None) or "unknown"

    op_type = name.split("_")[0]
    run = wandb.init(
        project=project, entity=entity, group=group,
        name=name, reinit=True,
        tags=[model, dataset, isa, author, op_type, harness],
        config={
            "definition": name, "dataset": dataset, "op_type": op_type,
            "isa": isa, "model": model, "author": author, "harness": harness,
            "instance_type": os.environ.get("WANDB_INSTANCE_TYPE", "unknown"),
            "baseline_kernel_sha": baseline_hash(dataset, name),
        },
    )

    # ── per-evaluate curve ────────────────────────────────────────────────────
    for i, r in enumerate(rows, 1):
        prev = rows[i - 2]["best_so_far"] if i > 1 else 0.0
        wandb.log({"iteration": i, "marginal_gain": round(r["best_so_far"] - prev, 5),
                   **{k: v for k, v in r.items() if v is not None and k not in ("status", "version")}},
                  step=i)

    # ── per-turn latency curve (own x-axis so it doesn't fight the eval steps) ─
    if turn_rows:
        run.define_metric("turn/idx")
        run.define_metric("turn/*", step_metric="turn/idx")
        for i, t in enumerate(turn_rows, 1):
            wandb.log({"turn/idx": i, "turn/total_s": t.total_s,
                       "turn/llm_s": t.llm_s, "turn/tool_s": t.tool_s})
        totals = [t.total_s for t in turn_rows]
        run.summary.update({
            "sec_per_turn_mean": round(sum(totals) / len(totals), 1),
            "sec_per_turn_median": round(statistics.median(totals), 1),
            "sec_per_turn_max": round(max(totals), 1),
            "llm_time_s": round(sum(t.llm_s for t in turn_rows), 1),
            "tool_time_s": round(sum(t.tool_s for t in turn_rows), 1),
        })

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
    cost, cost_source = session.cost_usd, getattr(session, "cost_source", None)
    if cost is not None and cost_source is None:
        cost_source = "harness"
    elif cost is None:
        cost = estimate_cost(model, session.tokens_input, session.tokens_output)
        cost_source = "litellm-estimate" if cost is not None else None

    session_fields = {
        "harness": harness, "harness_status": getattr(session, "harness_status", None),
        "cost_usd": cost, "cost_source": cost_source,
        "n_tool_calls": len(timeline) or None,     # every server tool call, all harnesses
        "turn_timing_source": turn_timing_source,
        "num_turns": session.num_turns,
        "wall_time_s": session.wall_time_s, "api_retries": session.api_retries,
        "session_compile_errors": session.session_compile_errors,
        "tokens_input": session.tokens_input, "tokens_output": session.tokens_output,
        "tokens_cache_read": session.tokens_cache_read,
        "tokens_cache_created": session.tokens_cache_created,
    }
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
        **{k: v for k, v in session_fields.items() if v is not None},
    })

    # ── winning kernel + techniques-per-version + artifact ────────────────────
    # wandb.Table lazily imports pandas (+ its tz deps); if that env is
    # incomplete the curve + summary above must still survive, so this block
    # only ever degrades, never aborts the run.
    try:
        _log_kernels(run, name, traj, ver_best)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb_log_run] kernels table/artifact skipped for {name}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)

    run.finish()
    print(f"[wandb_log_run] logged {name} ({harness}): best={best} evals={len(rows)} "
          f"weak_baseline={base_vs_scalar is not None and base_vs_scalar < 2.0} "
          f"cost={cost} ({cost_source}) tokens_out={session.tokens_output} "
          f"turn_timing={turn_timing_source}")


def _log_kernels(run, name: str, traj: Optional[Path], ver_best: dict) -> None:
    import wandb
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


def _cli_main() -> int:
    """Thin manual-backfill entry point: `python analysis/wandb_log_run.py
    --name ... --log-file ... --results-dir ...` re-logs one already-finished
    job of any harness (--harness, or sniffed from the log's format). Not
    used by bench_fleet.py (which calls log_run_to_wandb() directly with an
    already-parsed SessionMetrics from whichever HarnessAdapter ran the job)
    — this is for manually re-running or debugging the W&B logging for one
    definition after the fact."""
    import argparse

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from test_scripts.harness_adapters import SessionMetrics, parse_session_log_auto

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--isa", required=True)
    p.add_argument("--model", default="unknown")
    p.add_argument("--author", default="unknown")
    p.add_argument("--harness", default="", choices=["", "claude-code", "nanobot", "own"],
                   help="Parser for --log-file. Empty = sniff the log's format.")
    p.add_argument("--log-file", default="", help="The job log bench_fleet tee'd for this definition.")
    p.add_argument("--results-dir", default="")
    p.add_argument("--trajectory", default="")
    p.add_argument("--project", default="arm-bench-kernels")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default=None)
    args = p.parse_args()

    traj = Path(args.trajectory) if args.trajectory else _locate_trajectory(args.results_dir, args.name)
    session = (parse_session_log_auto(Path(args.log_file), args.harness)
               if args.log_file and Path(args.log_file).exists()
               else SessionMetrics())
    log_run_to_wandb(
        name=args.name, dataset=args.dataset, isa=args.isa, model=args.model, author=args.author,
        trajectory_path=traj, session=session,
        project=args.project, entity=args.entity, group=args.group,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
