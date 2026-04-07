"""
scripts/collect_baselines.py — Collect scalar and autovec baseline timings.

Builds scalar and autovec-sve2 targets on the instance, runs each loop,
and records timing to baselines/{tier}.json. These timings anchor the
speedup calculations in EvalResult.

Run once after provisioning. Safe to re-run (overwrites existing baselines).

Usage:
    python scripts/collect_baselines.py --isa sve2
    python scripts/collect_baselines.py --isa sme2
    python scripts/collect_baselines.py --isa sve2 --loop 001   # single loop
    python scripts/collect_baselines.py --isa sve2 --n 2000     # iteration count
"""

import argparse
import json
import re
import time
from pathlib import Path

from eval.config import REPO_ROOT, load_problems, ISA_TIER
from eval.provision import get_or_provision, get_running_instance, InstanceHandle

BASELINES_DIR = REPO_ROOT / "baselines"

# Iteration count used for baseline measurements
DEFAULT_N = 1000


def build_target(handle: InstanceHandle, target: str, extra_flags: str = ""):
    """Run make <target> on the instance."""
    flags_arg = f"EXTRA_FLAGS='{extra_flags}'" if extra_flags else ""
    cmd = f"cd ~/simd-loops && make {target} {flags_arg} 2>&1"
    rc, output, _ = handle.run(cmd, timeout=180)
    if rc != 0:
        raise RuntimeError(f"make {target} failed:\n{output[:500]}")
    return output


def run_loop(handle: InstanceHandle, binary: str, loop_num: str, n: int) -> float | None:
    """
    Run a loop and return timing in milliseconds.
    loop_num is decimal, e.g. "001" → int 1 → hex not needed, main.c uses decimal.
    """
    loop_decimal = int(loop_num)
    time_cmd = (
        f"t0=$(date +%s%N); "
        f"{binary} -k {loop_decimal} -n {n}; "
        f"rc=$?; "
        f"t1=$(date +%s%N); "
        f'echo "TIME_NS=$((t1-t0))"; '
        f"exit $rc"
    )
    rc, stdout, _ = handle.run(time_cmd, timeout=300)

    if "Checksum correct." not in stdout:
        return None  # wrong result

    m = re.search(r"TIME_NS=(\d+)", stdout)
    if m:
        return round(int(m.group(1)) / 1e6, 3)
    return None


def collect_baselines(
    handle: InstanceHandle,
    isa: str,
    problem_ids: list[str],
    n: int = DEFAULT_N,
) -> dict:
    """
    Build scalar, autovec, and hand-written ISA targets; run all problems.

    Returns:
        { "loop_001": { "scalar_ms": 9.4, "autovec_ms": 7.0, "ref_ms": 2.1 }, ... }

    ref_ms is the hand-written ISA reference (sve/sve2/sme2 build), which is
    the primary target for agents to beat.
    """
    tier = ISA_TIER.get(isa, "c7g")
    remote_root = "~/simd-loops"

    print(f"\n[baselines] Building c-scalar target (HAVE_NATIVE)...")
    build_target(handle, "c-scalar")
    scalar_binary = f"{remote_root}/build/c-scalar/bin/simd_loops"

    autovec_target = "autovec-sve2" if isa in ("sve2", "sme2") else "autovec-sve"
    print(f"[baselines] Building {autovec_target} target...")
    try:
        build_target(handle, autovec_target)
        autovec_binary = f"{remote_root}/build/{autovec_target}/bin/simd_loops"
    except RuntimeError as e:
        print(f"  WARNING: {autovec_target} build failed: {e}")
        autovec_binary = None

    # Hand-written ISA reference — the expert SVE/SVE2/SME2 implementation
    ref_target = isa  # "sve", "sve2", or "sme2"
    print(f"[baselines] Building {ref_target} target (hand-written reference)...")
    try:
        build_target(handle, ref_target)
        ref_binary = f"{remote_root}/build/{ref_target}/bin/simd_loops"
    except RuntimeError as e:
        print(f"  WARNING: {ref_target} build failed: {e}")
        ref_binary = None

    results = {}

    for pid in problem_ids:
        num = pid.split("_")[1]  # "loop_001" → "001"
        entry = {}

        print(f"  {pid}: ", end="", flush=True)

        # Scalar timing
        scalar_ms = run_loop(handle, scalar_binary, num, n)
        if scalar_ms is not None:
            entry["scalar_ms"] = scalar_ms
            print(f"scalar={scalar_ms:.1f}ms", end="", flush=True)
        else:
            print(f"scalar=FAIL", end="", flush=True)

        # Autovec timing
        if autovec_binary:
            autovec_ms = run_loop(handle, autovec_binary, num, n)
            if autovec_ms is not None:
                entry["autovec_ms"] = autovec_ms
                print(f", autovec={autovec_ms:.1f}ms", end="", flush=True)
            else:
                print(f", autovec=FAIL", end="", flush=True)

        # Hand-written ISA reference timing
        if ref_binary:
            ref_ms = run_loop(handle, ref_binary, num, n)
            if ref_ms is not None:
                entry["ref_ms"] = ref_ms
                print(f", ref={ref_ms:.1f}ms", end="", flush=True)
            else:
                print(f", ref=FAIL", end="", flush=True)

        print()  # newline
        if entry:
            results[pid] = entry

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect scalar/autovec baseline timings on Arm EC2 instance"
    )
    parser.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"],
                        help="ISA tier to collect baselines for")
    parser.add_argument("--loop", help="Collect for a single loop only, e.g. 001")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Iteration count (default: {DEFAULT_N})")
    parser.add_argument("--provision", action="store_true",
                        help="Provision a new instance if none is running")
    args = parser.parse_args()

    if args.isa == "sme2":
        print("ERROR: SME2 is not yet supported — no AWS EC2 instance implements SME2. "
              "SME2 problems are reserved for future hardware.")
        return

    # Get instance
    handle = get_running_instance(args.isa)
    if handle is None:
        if args.provision:
            print(f"Provisioning instance for ISA={args.isa}...")
            handle = get_or_provision(args.isa)
        else:
            print(f"No running instance for ISA={args.isa}.")
            print(f"Run: python eval/provision.py --isa {args.isa}")
            print(f"Or pass --provision to auto-provision.")
            return

    # Select problems
    problems = load_problems()
    tier = ISA_TIER.get(args.isa, "c7g")

    if args.loop:
        pid = f"loop_{args.loop.zfill(3)}"
        if pid not in problems:
            print(f"Problem {pid} not found.")
            return
        problem_ids = [pid]
    else:
        problem_ids = [
            pid for pid, p in problems.items()
            if p.get("isa_target") == args.isa
        ]
        print(f"Collecting baselines for {len(problem_ids)} problems (ISA: {args.isa})")

    # Collect
    BASELINES_DIR.mkdir(exist_ok=True)
    results = collect_baselines(handle, args.isa, problem_ids, n=args.n)

    # Merge with existing baselines
    out_path = BASELINES_DIR / f"{tier}.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))

    print(f"\n[baselines] Wrote {len(results)} entries to {out_path}")

    # Print summary
    n_scalar = sum(1 for v in results.values() if "scalar_ms" in v)
    n_autovec = sum(1 for v in results.values() if "autovec_ms" in v)
    n_ref    = sum(1 for v in results.values() if "ref_ms" in v)
    print(f"  scalar timings:  {n_scalar}/{len(results)}")
    print(f"  autovec timings: {n_autovec}/{len(results)}")
    print(f"  ref timings:     {n_ref}/{len(results)}  ← hand-written {args.isa}")


if __name__ == "__main__":
    main()
