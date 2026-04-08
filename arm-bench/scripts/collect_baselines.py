"""
scripts/collect_baselines.py — Collect ARM baseline timings for ncnn starter kernels.

Compiles each starter .cpp file (from starter/problems.json), builds the
ARM-optimised baseline binary, runs it N times, and records timing + perf
counters to baselines/arm_baselines.json.

Run once after provisioning. Safe to re-run (overwrites existing entries).

Usage (from arm-bench/):
    python -m scripts.collect_baselines                          # local
    python -m scripts.collect_baselines --n 20                   # more iterations
    python -m scripts.collect_baselines --provision --isa sve    # remote instance
"""

import argparse
import json
import re
import sys

from eval.config import REPO_ROOT
from eval.tools import SIMDTools

STARTER_DIR = REPO_ROOT / "starter"
STARTER_PROBLEMS = STARTER_DIR / "problems.json"
BASELINES_DIR = REPO_ROOT / "baselines"

DEFAULT_N = 10
DEFAULT_ISA = "sve"


def load_starter_problems() -> list[dict]:
    """Load starter/problems.json."""
    if not STARTER_PROBLEMS.exists():
        print(f"ERROR: {STARTER_PROBLEMS} not found.")
        sys.exit(1)
    return json.loads(STARTER_PROBLEMS.read_text())


def collect_one(tools: SIMDTools, problem: dict, n: int) -> dict | None:
    """
    Compile a starter problem and benchmark the ARM baseline binary.

    Returns dict with baseline timing and perf counters, or None on failure.
    """
    starter = problem["starter"]

    # Compile (produces both candidate and baseline binaries)
    cr = tools.compile(starter)
    if not cr.success:
        print(f"    COMPILE FAIL: {cr.errors[:200]}")
        return None

    if not tools.remote_baseline_binary:
        print(f"    NO BASELINE BINARY")
        return None

    # Correctness check on baseline
    baseline_cmd = f"{tools.remote_baseline_binary}"
    rc, stdout, _ = tools._run(baseline_cmd, timeout=120)
    if rc != 0 or "FAIL" in stdout:
        print(f"    BASELINE CORRECTNESS FAIL: {stdout[:200]}")
        return None

    # Timing: run baseline N times
    timing_cmd = (
        f"t0=$(date +%s%N); "
        f"for i in $(seq 1 {n}); do {tools.remote_baseline_binary} > /dev/null 2>&1; done; "
        f"t1=$(date +%s%N); "
        f'echo "TIME_NS=$((t1-t0))"'
    )
    _, stdout, _ = tools._run(timing_cmd, timeout=600)
    baseline_ms = None
    m = re.search(r"TIME_NS=(\d+)", stdout)
    if m:
        baseline_ms = round(int(m.group(1)) / 1e6 / n, 3)  # per-iteration

    # Perf counters on baseline (single run)
    perf_probe = (
        "PERF=$(ls /usr/lib/linux-aws-*-tools-*/perf 2>/dev/null | head -1); "
        "PERF=${PERF:-perf}; "
    )
    perf_cmd = (
        f"{perf_probe}"
        f"sudo $PERF stat "
        f"-e cycles,instructions,r04,r03 "
        f"{tools.remote_baseline_binary} "
        f"2>&1"
    )
    _, perf_output, _ = tools._run(perf_cmd, timeout=300)
    perf_result = SIMDTools._parse_perf_output(perf_output)

    return {
        "starter": starter,
        "baseline_ms": baseline_ms,
        "n_iters": n,
        "perf": {
            "cycles": perf_result.cycles,
            "instructions": perf_result.instructions,
            "ipc": perf_result.ipc,
            "l1d_miss_pct": perf_result.l1d_miss_pct,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect ARM baseline timings for all ncnn starter kernels"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Iterations per binary for timing (default: {DEFAULT_N})")
    parser.add_argument("--isa", default=DEFAULT_ISA, choices=["neon", "sve", "sve2"],
                        help="ISA tier (default: sve)")
    parser.add_argument("--provision", action="store_true",
                        help="Provision/reuse a remote instance instead of running locally")
    args = parser.parse_args()

    problems = load_starter_problems()

    # Get handle (None = local, otherwise SSH)
    handle = None
    if args.provision:
        from eval.provision import get_or_provision
        handle = get_or_provision(args.isa)
        print(f"Using remote instance: {handle.host}")

    print(f"\nCollecting ARM baselines for {len(problems)} kernels (n={args.n}, isa={args.isa})")
    print(f"{'='*60}\n")

    tools = SIMDTools(handle=handle, problem_id="baseline_collection", isa=args.isa)
    print("Uploading ncnn tree...")
    tools.upload_ncnn_tree()
    print(f"Synced to {tools.remote_project_root}\n")

    results = {}
    for prob in problems:
        pid = prob["id"]
        print(f"  [{pid}] {prob['starter']}")

        entry = collect_one(tools, prob, n=args.n)
        if entry is None:
            print(f"    SKIPPED\n")
            continue

        results[pid] = entry
        cp = entry["perf"]
        print(f"    baseline={entry['baseline_ms']}ms/iter  "
              f"cycles={cp['cycles']}  ipc={cp['ipc']}  l1d_miss%={cp['l1d_miss_pct']}")
        print()

    # Save results
    BASELINES_DIR.mkdir(exist_ok=True)
    out_path = BASELINES_DIR / "arm_baselines.json"

    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))

    print(f"{'='*60}")
    print(f"Wrote {len(results)} entries to {out_path}")

    # Summary table
    print(f"\n{'Problem':<20} {'Baseline ms':<15} {'Cycles':<15} {'IPC':<10}")
    print("-" * 60)
    for pid, entry in results.items():
        b = entry.get("baseline_ms", "FAIL")
        cy = entry["perf"].get("cycles", "N/A")
        ipc = entry["perf"].get("ipc", "N/A")
        print(f"{pid:<20} {str(b):<15} {str(cy):<15} {str(ipc):<10}")


if __name__ == "__main__":
    main()
