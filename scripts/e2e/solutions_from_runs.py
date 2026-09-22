#!/usr/bin/env python3
"""Write bench-trace solution JSONs straight from agent run directories.

The run dir is the authoritative record of what an agent produced: trajectory.jsonl
names the submitted version and v<N>.cpp holds its source. Assembling the solution
folder by hand from some other checkout is how ten of eleven gate-v2 "solutions"
ended up being the previous round's kernels -- see scripts/e2e/verify_kernel_set.py.

  python3 scripts/e2e/solutions_from_runs.py --runs <root>... --author <name> \
      --description "..." --template-author claude-code-claude-fable-5-1-sve2

Non-kernel sources (gemm.h, gemm.cpp) and `spec` come from --template-author's
solution for the same definition, since they are harness files the agent never edits.
Definitions present in the template author but not in --runs are copied through
unchanged (that is how a set mixes reused kernels with re-optimized ones) and are
listed as "reused" -- pass --only-runs to leave them out instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.e2e.build_manifest import pick_version  # noqa: E402


def run_stats(traj: Path) -> dict:
    recs = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
    return {
        "evaluations": sum(1 for r in recs if r.get("tool") == "evaluate"),
        "server_calls": len(recs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="agent-run roots (later roots win)")
    ap.add_argument("--author", required=True, help="solution author to write under")
    ap.add_argument("--template-author", required=True,
                    help="existing author whose solutions supply spec + harness sources")
    ap.add_argument("--dataset", default="llama.cpp")
    ap.add_argument("--op-type", default="gemm")
    ap.add_argument("--description", default=None, help="per-solution description; {definition} is substituted")
    ap.add_argument("--run-label", default=None, help="provenance.run")
    ap.add_argument("--only-runs", action="store_true",
                    help="write only definitions found in --runs (default: also copy the template author's others)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tdir = REPO / "bench-trace/solutions" / args.dataset / args.template_author / args.op_type
    odir = REPO / "bench-trace/solutions" / args.dataset / args.author / args.op_type
    if not tdir.is_dir():
        raise SystemExit(f"no template author dir: {tdir}")

    found = {}
    for root in args.runs:
        for d in sorted(Path(root).expanduser().iterdir()):
            if (d / "trajectory.jsonl").exists():
                found[d.name] = d

    if not args.dry_run:
        odir.mkdir(parents=True, exist_ok=True)
    wrote, reused, missing = [], [], []
    for tf in sorted(tdir.glob("*.json")):
        sol = json.load(open(tf))
        name = sol["definition"]
        d = found.get(name)
        if d is None:
            if args.only_runs:
                continue
            reused.append(name)
        else:
            src, speedup = pick_version(d / "trajectory.jsonl", False)
            if not src:
                missing.append(name)
                continue
            for s in sol["sources"]:
                if s["path"] == "kernel.cpp":
                    s["content"] = (d / src).read_text()
            sol["provenance"] = {"run": args.run_label or d.parent.name,
                                 "submitted_version": src,
                                 "time_speedup": speedup,
                                 **run_stats(d / "trajectory.jsonl")}
            if args.description:
                sol["description"] = args.description.format(definition=name)
            wrote.append(f"{name} <- {src} ({speedup:.3f}x)")
        sol["author"] = args.author
        sol["name"] = f"{args.author}_{name}"
        if not args.dry_run:
            (odir / f"{name}.json").write_text(json.dumps(sol, indent=1))

    for line in wrote:
        print("  from run dir:", line)
    for name in reused:
        print("  reused from template author:", name)
    for name in missing:
        print("  SKIPPED (no submitted version):", name)
    print(f"{len(wrote)} from runs, {len(reused)} reused, {len(missing)} skipped -> "
          f"{odir}{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
