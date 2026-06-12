"""scripts/diff_traces.py — Regression diff between two trace warehouses.

Used as the bit-exact gate for the candidate-binding migration
(PLAN_candidate_binding_into_sources.md): snapshot traces before a change, re-run
after, and diff. Reports any (definition, solution, workload) whose evaluation
changed.

Traces are append-only (bench/data/trace_set.py), so each <op>/<def>.jsonl may
hold several runs of the same (solution, workload). We keep the LATEST record per
key in each side and compare those.

Compared per key = (definition, solution, workload.uuid):
    status, correctness.max_absolute_error, correctness.max_relative_error
(No output hash — see PLAN §4.2: a pure-rewiring migration moves max_abs too, so
status+max_abs+max_rel is a sufficient oracle.)

Floats are compared exactly: a bit-exact rewiring must not move them at all.

Usage (from cpu-kernel-baseline/):
    python -m scripts.diff_traces <golden_traces_dir> <current_traces_dir>
    python -m scripts.diff_traces bench-trace/traces.golden bench-trace/traces --author reference-scalar
Exit code 0 = identical, 1 = any difference (or missing/added keys).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# key = (definition, solution, workload_uuid); value = (status, max_abs, max_rel)
Key = Tuple[str, str, str]
Val = Tuple[str, Optional[float], Optional[float]]


def _load_latest(traces_dir: Path, author: Optional[str]) -> Dict[Key, Val]:
    """Map each (def, solution, workload) → its LATEST evaluation summary."""
    latest: Dict[Key, Val] = {}
    for p in sorted(traces_dir.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sol = rec.get("solution")
            if sol is None:
                continue  # workload-only trace
            if author is not None and not sol.startswith(author + "_"):
                continue
            key: Key = (rec["definition"], sol, rec["workload"]["uuid"])
            ev = rec.get("evaluation") or {}
            corr = ev.get("correctness") or {}
            val: Val = (
                ev.get("status", "<none>"),
                corr.get("max_absolute_error"),
                corr.get("max_relative_error"),
            )
            latest[key] = val  # later line wins → latest run
    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff two trace warehouses for regressions.")
    ap.add_argument("golden", type=Path, help="baseline (pre-change) traces dir.")
    ap.add_argument("current", type=Path, help="post-change traces dir.")
    ap.add_argument("--author", default=None,
                    help="only compare solutions of this author (e.g. reference-scalar).")
    args = ap.parse_args()

    for d in (args.golden, args.current):
        if not d.exists():
            print(f"ERROR: traces dir not found: {d}", file=sys.stderr)
            return 2

    g = _load_latest(args.golden, args.author)
    c = _load_latest(args.current, args.author)

    gk, ck = set(g), set(c)
    removed = sorted(gk - ck)   # in golden, gone in current
    added = sorted(ck - gk)     # new in current
    changed = []
    for k in sorted(gk & ck):
        if g[k] != c[k]:
            changed.append((k, g[k], c[k]))

    def _fmt(k: Key) -> str:
        d, s, w = k
        return f"{s} / {w[:8]}"

    if removed:
        print(f"MISSING in current ({len(removed)}):")
        for k in removed:
            print(f"  - {_fmt(k)}  was {g[k]}")
    if added:
        print(f"ADDED in current ({len(added)}):")
        for k in added:
            print(f"  + {_fmt(k)}  now {c[k]}")
    if changed:
        print(f"CHANGED ({len(changed)}):")
        for k, gv, cv in changed:
            print(f"  ~ {_fmt(k)}")
            print(f"      golden : status={gv[0]} max_abs={gv[1]} max_rel={gv[2]}")
            print(f"      current: status={cv[0]} max_abs={cv[1]} max_rel={cv[2]}")

    n_diff = len(removed) + len(added) + len(changed)
    print(f"\ncompared {len(gk & ck)} common keys; "
          f"{len(changed)} changed, {len(removed)} missing, {len(added)} added.")
    if n_diff == 0:
        print("IDENTICAL — no regression.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
