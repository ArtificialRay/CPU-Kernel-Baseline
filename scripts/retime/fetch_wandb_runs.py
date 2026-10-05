#!/usr/bin/env python3
"""Download the stored kernels of finished agent runs from W&B, one run dir each.

Selects runs the way the results tables do: a cell is
<harness>__<model>__<dataset>__<isa>, a group with a fifth `__` part is a
separate variant cell (`__sandboxed` reruns count as the main cell), and within
a cell the newest run per definition wins. Then it downloads each selected
run's `kernel` artifact (trajectory.jsonl + v1.cpp ... vN.cpp) into
<out>/<cell>/<definition>/ and writes <out>/manifest.json, the input of
scripts/retime/extract_run_solutions.py.

    python3 scripts/retime/fetch_wandb_runs.py --dataset simd-loop --isa sve \\
        --definitions-file bench-trace/expected_sets_sve.json --out runs-sve
    python3 scripts/retime/fetch_wandb_runs.py --dataset simd-loop --isa sve --list   # cells only

Needs `wandb` and a logged-in W&B account.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

# Separate experiments that carry the main sweep's config; never mix them in.
ABLATION_PROJECTS = {"arm-bench-kernels-extdocs", "arm-bench-kernels-share-workspace",
                     "arm-bench-kernels-TEST"}


def cell_of(run) -> str:
    c = run.config or {}
    group = run.group or ""
    harness = c.get("harness")
    if not harness or str(harness).lower() == "unknown":
        harness = group.split("__")[0] if "__" in group else next(
            (p for p in ("codex", "nanobot", "own", "claude-code", "tcloop")
             if group.startswith(p + "-")), "unknown")
    model = str(c.get("model") or "unknown").split("/")[-1]
    cell = f"{harness}__{model}__{c.get('dataset')}__{c.get('isa')}"
    parts = group.split("__")
    if parts[4:] == ["sandboxed"]:
        parts = parts[:4]
    if len(parts) >= 5:
        cell += "__" + "__".join(parts[4:])
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entity", default="ArmBench")
    ap.add_argument("--project", action="append", help="repeatable; default: every project")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--isa", required=True)
    ap.add_argument("--definitions-file", help="expected_sets*.json; keep only its definitions for --dataset")
    ap.add_argument("--cells", nargs="*", help="keep only these cells (default: every main cell, no variant suffix)")
    ap.add_argument("--include-variants", action="store_true", help="also keep variant cells (fifth group part)")
    ap.add_argument("--out", default="wandb-runs")
    ap.add_argument("--list", action="store_true", help="print the cells and counts, download nothing")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=120)
    projects = args.project or [p.name for p in api.projects(args.entity)]
    wanted = None
    if args.definitions_file:
        sets = json.load(open(args.definitions_file))
        wanted = set(sets[args.dataset] if isinstance(sets, dict) else sets)

    latest = {}
    for proj in projects:
        if proj in ABLATION_PROJECTS:
            continue
        filters = {"config.dataset": args.dataset, "config.isa": args.isa}
        for run in api.runs(f"{args.entity}/{proj}", filters=filters, per_page=500):
            defn = run.name.split("/")[-1]
            if wanted is not None and defn not in wanted:
                continue
            cell = cell_of(run)
            if cell.count("__") > 3 and not args.include_variants:
                continue
            if args.cells and cell not in args.cells:
                continue
            key = (cell, defn)
            if key not in latest or str(run.created_at) > str(latest[key].created_at):
                latest[key] = run

    by_cell = collections.Counter(c for c, _ in latest)
    for cell, n in sorted(by_cell.items()):
        print(f"{cell:60} {n} definitions")
    if args.list:
        return 0

    out = Path(args.out)
    manifest = []
    for (cell, defn), run in sorted(latest.items()):
        d = out / cell / defn
        d.mkdir(parents=True, exist_ok=True)
        arts = [a for a in run.logged_artifacts() if a.type == "kernel"]
        if not arts:
            print(f"  no kernel artifact: {cell} {defn} ({run.id})", file=sys.stderr)
            continue
        arts[-1].download(root=str(d))
        s = run.summary or {}
        manifest.append(dict(
            cell=cell, definition=defn, run_dir=str(d.relative_to(out)),
            run=f"{run.project}/{run.id}", group=run.group, created=str(run.created_at),
            step40=s.get("performance_step_40"), best_speedup=s.get("best_speedup")))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} run dirs under {out}/, manifest.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
