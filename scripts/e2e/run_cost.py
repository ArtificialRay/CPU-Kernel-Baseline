#!/usr/bin/env python3
"""What a live Claude Code sweep has actually spent, per kernel, from the harness logs.

A sweep's cost is only visible after the fact unless you read it out of the harness
log stream, which carries a `usage` block on every assistant message. That matters
when the budget is a hard constraint: these runs were stopped by a spend ceiling,
and the decision to stop needs a number, not a guess.

  python3 scripts/e2e/run_cost.py --logs ~/arm-bench-e2e/harness_trajs/claude-code/<author> \
      --runs ~/arm-bench-e2e/agent-runs-<label> [--definitions a b] [--budget 36] [--json]

Prints per definition: spend, model calls, the turn it is on, and its best PASSED
speedup so far, then the total against --budget. Messages are de-duplicated by id,
since a streaming log repeats a message as it grows.

Rates default to Claude Fable 5.1 ($/Mtok in/cache-write/cache-read/out); pass
--rates for another model. They are list prices, so treat the total as an estimate
good to a few percent, not a bill.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FABLE_51 = (15.0, 18.75, 1.5, 75.0)   # input, cache write, cache read, output ($/Mtok)


def log_cost(path: Path, rates) -> tuple:
    r_in, r_cw, r_cr, r_out = rates
    cost, seen = 0.0, set()
    if not path.exists():
        return 0.0, 0
    for line in path.open(errors="replace"):
        if '"type":"assistant"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        m = o.get("message") or {}
        mid = m.get("id")
        if mid in seen:
            continue
        seen.add(mid)
        u = m.get("usage") or {}
        cost += (u.get("input_tokens", 0) * r_in
                 + u.get("cache_creation_input_tokens", 0) * r_cw
                 + u.get("cache_read_input_tokens", 0) * r_cr
                 + u.get("output_tokens", 0) * r_out) / 1e6
    return cost, len(seen)


def run_progress(traj: Path) -> tuple:
    turns, best = 0, None
    if not traj.exists():
        return 0, None
    for line in traj.open(errors="replace"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        turns = max(turns, o.get("turn", 0))
        m = o.get("metrics") or {}
        if o.get("tool") == "evaluate" and m.get("status") == "PASSED":
            s = m.get("time_speedup_geomean")
            if s and (best is None or s > best):
                best = s
    return turns, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True, help="harness_trajs/<harness>/<author> dir")
    ap.add_argument("--runs", type=Path, required=True, help="agent-run root for this sweep")
    ap.add_argument("--definitions", nargs="*", default=None, help="default: every run dir present")
    ap.add_argument("--log-pattern", default="llama.cpp_sve2_{definition}.log")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--rates", nargs=4, type=float, default=list(FABLE_51),
                    metavar=("IN", "CACHE_WRITE", "CACHE_READ", "OUT"), help="$/Mtok")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    defs = args.definitions or sorted(
        d.name for d in args.runs.iterdir() if (d / "trajectory.jsonl").exists()) if args.runs.is_dir() else []
    rows, total = [], 0.0
    for name in defs:
        cost, calls = log_cost(args.logs / args.log_pattern.format(definition=name), args.rates)
        turns, best = run_progress(args.runs / name / "trajectory.jsonl")
        total += cost
        rows.append({"definition": name, "cost": cost, "calls": calls, "turn": turns, "best_speedup": best})

    if args.json:
        print(json.dumps({"rows": rows, "total": total, "budget": args.budget}, indent=1))
        return
    for r in rows:
        b = f"{r['best_speedup']:.3f}x" if r["best_speedup"] else "   --"
        st = f"turn {r['turn']:>3}" if r["turn"] else ("running" if r["calls"] else "starting")
        print(f"  {r['definition']:<32} ${r['cost']:7.2f}  {r['calls']:>4} calls  {st:>9}  best {b}")
    line = f"  {'TOTAL':<32} ${total:7.2f}"
    if args.budget:
        line += f"   ({total / args.budget * 100:.0f}% of the ${args.budget:.0f} guide)"
    print(line)


if __name__ == "__main__":
    main()
