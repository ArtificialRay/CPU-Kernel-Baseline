"""
scripts/collect_baselines.py — Collect scalar and autovec baseline timings.

Builds scalar and autovec-sve2 targets on the instance, runs each loop,
and records timing to baselines/{tier}.json. These timings anchor the
speedup calculations in EvalResult.

Timing format: ms per iteration at the largest PERF_SIZE for each problem.
For loops with no PERF_SIZES (fixed-dimension kernels), falls back to
ms per iteration at the default compiled-in size with n=100 iterations.

This matches the scoring in eval/tools.py submit(), which also uses the
largest PERF_SIZE result as the authoritative runtime_ms.

Run once after provisioning (or after changing PERF_SIZES). Safe to re-run.

Usage:
    python scripts/collect_baselines.py --isa sve2
    python scripts/collect_baselines.py --isa sme2
    python scripts/collect_baselines.py --isa sve2 --loop 001   # single loop
    python scripts/collect_baselines.py --isa sve2 --n 100      # iteration count
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import REPO_ROOT, load_problems, ISA_TIER, load_problem_sizes
from eval.provision import get_or_provision, get_running_instance, InstanceHandle

BASELINES_DIR = REPO_ROOT / "baselines"

# Iteration count for timing measurements (per-size; total wall time ≈ n × ms_per_iter)
DEFAULT_N = 10


def build_target(handle: InstanceHandle, target: str, extra_flags: str = ""):
    """Run make <target> on the instance."""
    flags_arg = f"EXTRA_FLAGS='{extra_flags}'" if extra_flags else ""
    cmd = f"cd ~/arm-bench && make {target} {flags_arg} 2>&1"
    rc, output, _ = handle.run(cmd, timeout=180)
    if rc != 0:
        raise RuntimeError(f"make {target} failed:\n{output[:500]}")
    return output


def compile_loop_at_size(
    handle: InstanceHandle,
    loop_num: str,
    target: str,
    base_extra_flags: str,
    size: int,
) -> str | None:
    """
    Recompile a single loop at a specific SIZE without rebuilding all other loops.

    Mirrors the incremental strategy in SIMDTools._compile_at_size: first warm-up
    builds all OTHER loops at the baseline flags, then recompiles just this loop's
    .o at -DSIZE={size} and relinks.  This avoids _Static_assert failures in other
    loops that have their own SIZE constraints.

    Returns the binary path on success, None on failure.
    """
    remote_root = "~/arm-bench"
    obj_dir = f"{remote_root}/build/{target}/_obj"
    loop_o = f"{obj_dir}/loops/loop_{loop_num}.o"
    lnk_o = f"{obj_dir}/_lnk/loop_{loop_num}.o"

    # Step 1: recreate any missing _lnk symlinks using an absolute path so ~ expands
    # correctly inside the SSH for-loop.  ln -sf is idempotent — safe to always run.
    lnk_dir = f"{obj_dir}/_lnk"
    restore_lnk = (
        f"cd {remote_root} && "
        f"for f in build/{target}/_obj/loops/*.o build/{target}/_obj/common/*.o; do "
        f"  [ -f \"$f\" ] && ln -sf \"$PWD/$f\" \"build/{target}/_obj/_lnk/$(basename $f)\" 2>/dev/null || true; "
        f"done"
    )
    handle.run(restore_lnk, timeout=30)

    warmup_cmd = f"cd {remote_root} && make {target} EXTRA_FLAGS='{base_extra_flags}' 2>&1"
    rc, _, _ = handle.run(warmup_cmd, timeout=180)
    if rc != 0:
        return None

    # Step 2: recompile just this loop at the requested SIZE, then relink
    sized_flags = f"{base_extra_flags} -DSIZE={size}"
    relink_cmd = (
        f"cd {remote_root} && rm -f {loop_o} {lnk_o} && "
        f"make {target} EXTRA_FLAGS='{sized_flags}' 2>&1"
    )
    rc, _, _ = handle.run(relink_cmd, timeout=180)
    if rc != 0:
        return None
    return f"{remote_root}/build/{target}/bin/simd_loops"


def run_loop(handle: InstanceHandle, binary: str, loop_num: str, n: int) -> float | None:
    """
    Run a loop and return ms per iteration (total_ms / n).
    Returns None if the checksum is wrong or the run fails.
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
    rc, stdout, _ = handle.run(time_cmd, timeout=600)

    # rc=0: correct checksum; rc=1: wrong checksum (expected at non-default SIZE)
    # both mean the loop ran; rc=2 = ABORT (alloc fail / no impl)
    if rc == 2 or "ABORT" in stdout:
        return None

    m = re.search(r"TIME_NS=(\d+)", stdout)
    if m:
        total_ms = int(m.group(1)) / 1e6
        return round(total_ms / n, 4)  # ms per iteration
    return None


def collect_baselines(
    handle: InstanceHandle,
    isa: str,
    problem_ids: list[str],
    n: int = DEFAULT_N,
) -> dict:
    """
    Build scalar, autovec, and hand-written ISA targets; run all problems.

    Timing is measured at the largest PERF_SIZE for each problem (cache-busting,
    DRAM-bound), matching the scoring in eval/tools.py submit().  For loops with
    no PERF_SIZES (fixed-dimension kernels), the default compiled-in size is used.

    All values are stored as ms per iteration so they are directly comparable
    to the runtime_ms values reported by submit().

    Returns:
        {
          "loop_001": {
            "scalar_ms": 8.3,   # ms/iter at largest PERF_SIZE (or default size)
            "autovec_ms": 4.1,
            "ref_ms": 2.0,
            "perf_size": 16000000  # size used for timing (0 = default)
          }, ...
        }
    """
    tier = ISA_TIER.get(isa, "c7g")
    remote_root = "~/arm-bench"

    autovec_target = "autovec-sve2" if isa in ("sve2", "sme2") else "autovec-sve"
    ref_target = isa  # "sve", "sve2", or "sme2"

    # The Makefile already injects -DHAVE_NATIVE, -DHAVE_AUTOVEC, etc. per target.
    # EXTRA_FLAGS only needs to carry -DSIZE=N for the sized recompilations.
    # Use empty base flags so the warm-up build matches a plain `make <target>`.
    target_base_flags = {
        "c-scalar": "",
        autovec_target: "",
        ref_target: "",
    }

    # Warm-up: build all targets at default size once (populates all other loops' .o files)
    print(f"\n[baselines] Warm-up builds...")
    for tgt, flags in target_base_flags.items():
        print(f"  make {tgt}...", end="", flush=True)
        try:
            build_target(handle, tgt, flags)
            print(" OK")
        except RuntimeError as e:
            print(f" FAIL: {e}")

    results = {}

    for pid in problem_ids:
        num = pid.split("_")[1]  # "loop_001" → "001"
        entry: dict = {}

        # Determine the size to use for this problem
        _, perf_sizes = load_problem_sizes(pid)
        perf_size = max(perf_sizes) if perf_sizes else None
        size_label = f"size={perf_size}" if perf_size else "default-size"

        print(f"  {pid} ({size_label}): ", end="", flush=True)

        for tgt_key, base_flags in target_base_flags.items():
            label = {
                "c-scalar": "scalar",
                autovec_target: "autovec",
                ref_target: "ref",
            }[tgt_key]
            ms_key = f"{label}_ms"

            if perf_size is not None:
                # Recompile just this loop at the PERF_SIZE and run
                binary = compile_loop_at_size(handle, num, tgt_key, base_flags, perf_size)
                if binary is None:
                    print(f"{label}=FAIL(compile) ", end="", flush=True)
                    continue
                ms = run_loop(handle, binary, num, n)
            else:
                # No PERF_SIZES — use the default-size binary already built
                binary = f"{remote_root}/build/{tgt_key}/bin/simd_loops"
                ms = run_loop(handle, binary, num, n)

            if ms is not None:
                entry[ms_key] = ms
                print(f"{label}={ms:.2f}ms/iter ", end="", flush=True)
            else:
                print(f"{label}=FAIL ", end="", flush=True)

        if perf_size is not None:
            entry["perf_size"] = perf_size

        print()  # newline
        if entry:
            results[pid] = entry

        # Restore the default-size binaries so the next loop's warm-up is a no-op
        if perf_size is not None:
            for tgt_key, base_flags in target_base_flags.items():
                compile_loop_at_size(handle, num, tgt_key, base_flags,
                                     _read_default_size(num) or perf_size)

    return results


def _read_default_size(loop_num: str) -> int | None:
    """Parse #define SIZE N from the local loop source file."""
    src = REPO_ROOT / "loops" / f"loop_{loop_num}.c"
    if not src.exists():
        return None
    m = re.search(r"#define SIZE\s+(\d+)", src.read_text())
    return int(m.group(1)) if m else None


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
