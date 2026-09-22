#!/usr/bin/env python3
"""Copy a running kernel's artifacts off the eval box, verify they arrived, and only
then let the lane be stopped.

An agent's run directory and its submitted solution JSON live on the EVAL BOX until
the kernel finishes normally. Stopping the lane and tearing the box down in the
obvious order destroys them -- that is how one kernel's whole search was lost. The
order has to be rescue, verify, then stop, and the verify has to be able to say no.

  python3 scripts/e2e/rescue_run.py --box <ip> --definition <name> --author <author> \
      --runs-dest ~/agent-runs-<label> [--wait-for-evaluate] [--min-versions 1]

Exits 0 only when the run dir is on this machine with at least one submitted version
recorded and --min-versions sources present; anything less exits non-zero with the
lane untouched. Stopping the lane is deliberately NOT done here -- this tool's job
is to make stopping safe, and a spend decision belongs to whoever is paying.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SSH = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def remote(box: str, cmd: str) -> str:
    p = subprocess.run(SSH + [f"ubuntu@{box}", cmd], capture_output=True, text=True)
    return p.stdout.strip()


def count_evaluates(box: str, traj: str) -> int:
    out = remote(box, f"grep -c '\"tool\": *\"evaluate\"' {traj} 2>/dev/null || echo 0")
    try:
        return int(out.splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--definition", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--runs-dest", type=Path, required=True, help="local agent-run root to rescue into")
    ap.add_argument("--solutions-dest", type=Path, default=None,
                    help="local bench-trace/solutions/<dataset>/<author>/<op> dir (optional)")
    ap.add_argument("--remote-runs", default="arm-bench/agent-runs-mcp",
                    help="agent-run root on the box (default: where the MCP harness writes)")
    ap.add_argument("--dataset", default="llama.cpp")
    ap.add_argument("--op", default="gemm")
    ap.add_argument("--wait-for-evaluate", action="store_true",
                    help="wait for one more evaluate to land first, so the rescue includes it")
    ap.add_argument("--wait-timeout", type=int, default=1200)
    ap.add_argument("--min-versions", type=int, default=1)
    args = ap.parse_args()

    rtraj = f"{args.remote_runs}/{args.author}/{args.definition}/trajectory.jsonl"
    if args.wait_for_evaluate:
        n0 = count_evaluates(args.box, rtraj)
        print(f"[rescue] {n0} evaluates so far; waiting for the next (timeout {args.wait_timeout}s)")
        deadline = time.time() + args.wait_timeout
        while time.time() < deadline:
            if count_evaluates(args.box, rtraj) > n0:
                print("[rescue] next evaluate landed")
                break
            time.sleep(20)
        else:
            print("[rescue] timed out waiting; rescuing what is there")

    dest = args.runs_dest / args.definition
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[rescue] copying {args.box}:{args.remote_runs}/{args.author}/{args.definition}/ -> {dest}")
    rc = subprocess.run(["rsync", "-az", "-e", " ".join(SSH),
                         f"ubuntu@{args.box}:{args.remote_runs}/{args.author}/{args.definition}/",
                         f"{dest}/"]).returncode
    if rc != 0:
        print("[rescue] RSYNC FAILED -- nothing has been stopped; the box still has the only copy")
        return 1

    if args.solutions_dest:
        args.solutions_dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-az", "-e", " ".join(SSH),
                        f"ubuntu@{args.box}:arm-bench/bench-trace/solutions/{args.dataset}/"
                        f"{args.author}/{args.op}/{args.definition}.json",
                        f"{args.solutions_dest}/"])

    traj = dest / "trajectory.jsonl"
    if not traj.exists():
        print("[rescue] NO TRAJECTORY in the rescued copy -- do not stop the lane")
        return 1
    submitted, best = None, None
    for line in traj.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        m = o.get("metrics") or {}
        if "submit" in (o.get("tool") or ""):
            submitted = o.get("source_file") or m.get("source_file") or submitted
            best = m.get("time_speedup_geomean") or m.get("time_speedup") or best
    versions = len(list(dest.glob("v*.cpp")))
    print(f"[rescue] {versions} versions, submitted {submitted}, speedup "
          f"{f'{best:.3f}x' if best else 'NONE'}")
    if submitted is None or versions < args.min_versions:
        print("[rescue] ARTIFACTS INCOMPLETE -- do not stop the lane")
        return 1
    print("[rescue] OK -- artifacts are safe on this machine; the lane can now be stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
