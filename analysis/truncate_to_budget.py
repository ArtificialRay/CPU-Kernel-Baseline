#!/usr/bin/env python3
"""Cut over-budget trajectories at the N-th evaluate and re-log them to W&B.

For every <results_dir>/<kernel>/trajectory.jsonl with more than N evaluate
rows: keep rows up to and including the N-th evaluate (plus the auto-submit
row that immediately follows it, if any), save the original as
trajectory_full.jsonl, move vK.cpp/vK.s for versions beyond the cut into
beyond_budget/, then delete the kernel's existing W&B run in the group and
log the truncated one. --dry-run only reports.

Usage (on the controller):
  python analysis/truncate_to_budget.py --max 40 --dataset ncnn --dry-run
"""
import argparse, json, re, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "test_scripts")); sys.path.insert(0, str(REPO / "analysis"))

def cut_rows(rows, n):
    seen = 0
    for i, r in enumerate(rows):
        if r.get("tool") == "evaluate":
            seen += 1
            if seen == n:
                j = i + 1
                if j < len(rows) and rows[j].get("tool") == "submit":
                    j += 1
                return rows[:j]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--author", default="claude-code-claude-sonnet-4-6-sve")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--isa", default="sve")
    ap.add_argument("--dataset", action="append", default=None, help="restrict to dataset(s)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    proto = json.loads((REPO / "analysis" / "m2_protocol_sets.json").read_text())
    ds_of = {}
    for ds, names in proto.items():
        for n in names: ds_of[n] = ds
    for p in (REPO / "bench-trace" / "definitions" / "simd-loop").glob("*.json"):
        ds_of.setdefault(json.loads(p.read_text())["name"], "simd-loop")
    results = REPO / f"agent-runs-{a.author}"
    todo = []
    for d in sorted(results.iterdir()):
        t = d / "trajectory.jsonl"
        if not t.exists(): continue
        rows = [json.loads(l) for l in t.open() if l.strip()]
        n_ev = sum(r.get("tool") == "evaluate" for r in rows)
        ds = ds_of.get(d.name)
        if n_ev > a.max and ds and (not a.dataset or ds in a.dataset):
            cut = cut_rows(rows, a.max)
            best_full = max((r["metrics"].get("time_speedup_geomean") or 0) for r in rows if r.get("tool") == "evaluate")
            best_cut = max((r["metrics"].get("time_speedup_geomean") or 0) for r in cut if r.get("tool") == "evaluate")
            todo.append((d, ds, rows, cut, n_ev, best_full, best_cut))
            print(f"{d.name:44} {ds:9} evals={n_ev:3} -> {a.max}  best {best_full:.3f} -> {best_cut:.3f}  rows {len(rows)}->{len(cut)}")
    if a.dry_run or not todo:
        print("dry run — nothing changed" if a.dry_run else "nothing over budget"); return
    import wandb, wandb_log_run
    from harness_adapters import ClaudeCodeAdapter
    api = wandb.Api()
    for d, ds, rows, cut, n_ev, _, _ in todo:
        full = d / "trajectory_full.jsonl"
        if not full.exists(): shutil.copy(d / "trajectory.jsonl", full)
        (d / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cut))
        last_v = max((int(m.group(1)) for r in cut for m in [re.search(r"v(\d+)\.cpp", r.get("source_file") or "")] if m), default=0)
        bb = d / "beyond_budget"; bb.mkdir(exist_ok=True)
        for f in list(d.glob("v*.cpp")) + list(d.glob("v*.s")):
            m = re.match(r"v(\d+)\.", f.name)
            if m and int(m.group(1)) > last_v: shutil.move(str(f), bb / f.name)
        group = f"claude-code__{a.model}__{ds}__{a.isa}"
        for r in api.runs("ArmBench/arm-bench-kernels", filters={"group": group}):
            if r.name.split("/")[-1] == d.name:
                print("  deleting W&B run", r.id, "for", d.name); r.delete()
        log = REPO / "harness_trajs" / "claude-code" / a.author / f"{ds}_{a.isa}_{d.name}.log"
        session = ClaudeCodeAdapter(model=a.model, max_budget_usd=None).parse_session_metrics(log) if log.exists() else None
        wandb_log_run.log_run_to_wandb(name=d.name, dataset=ds, isa=a.isa, model=a.model, author=a.author,
            trajectory_path=d / "trajectory.jsonl", session=session, project="arm-bench-kernels", entity="ArmBench", group=group)
        print(f"  re-logged {d.name} (cut at {a.max} evaluates, kept v1..v{last_v})")

if __name__ == "__main__":
    main()
