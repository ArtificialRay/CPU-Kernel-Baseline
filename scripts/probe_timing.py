"""
scripts/probe_timing.py — Measure actual ms/iter for every PERF_SIZE on the live instance.

For each loop with non-empty PERF_SIZES, compiles c-scalar at each size and times
n=10 iterations.  Reports ms/iter, working-set bytes, and whether the data is
DRAM-bound (ms/iter > threshold for L3 bandwidth).

Usage:
    python scripts/probe_timing.py [--isa sve] [--n 10] [--loop 001]
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Estimated working-set bytes per iteration for each loop (used to infer cache level).
# Computed as: sum(n_arrays * size * bytes_per_element).
# These are approximate — some loops have fixed overhead or non-linear access.
BYTES_PER_ITER = {
    # fp32/uint32 (2 arrays × 4B)
    "loop_001": lambda n: n * 8,   # 2 × fp32
    "loop_002": lambda n: n * 8,   # 2 × uint32
    "loop_010": lambda n: n * 8,
    "loop_027": lambda n: n * 8,
    "loop_035": lambda n: n * 12,  # 3 arrays: a, b, out
    "loop_126": lambda n: n * 8,
    "loop_127": lambda n: n * 8,
    "loop_128": lambda n: n * 8,
    # fp64/uint64 (2 arrays × 8B)
    "loop_003": lambda n: n * 16,
    "loop_004": lambda n: n * 16,
    "loop_008": lambda n: n * 16,
    "loop_028": lambda n: n * 16,
    "loop_029": lambda n: n * 16,
    "loop_032": lambda n: n * 16,
    "loop_033": lambda n: n * 16,
    "loop_111": lambda n: n * 16,
    # uint8 (2 arrays × 1B)
    "loop_024": lambda n: n * 2,
    "loop_031": lambda n: n * 2,
    "loop_101": lambda n: n * 2,
    # uint8 single buffer
    "loop_022": lambda n: n * 1,
    # uint16
    "loop_026": lambda n: n * 3,   # uint16 in + uint8 out
    # RGBA + luma
    "loop_108": lambda n: n * 5,   # uint32 rgba + uint8 luma
    # uint32 histogram records
    "loop_102": lambda n: n * 4,
    # complex uint32 / fp32 pairs (3 arrays × 8B)
    "loop_037": lambda n: n * 24,
    "loop_109": lambda n: n * 24,
    "loop_110": lambda n: n * 24,
    "loop_112": lambda n: n * 24,
    "loop_113": lambda n: n * 16,  # 2 arrays of pairs
    # object_t (16B) + uint32 indexes (4B)
    "loop_019": lambda n: n * 20,
    # fp64 sparse: double a, b + uint32 indexes
    "loop_023": lambda n: n * 20,
    # particle motion: 6+ fp64 arrays
    "loop_012": lambda n: n * 48,
    # uint128 × uint128 → uint256: 3 arrays × (16+16+32)B
    "loop_107": lambda n: n * 64,
    # SAG: 3 uint32 arrays
    "loop_106": lambda n: n * 12,
    # pointer chasing: 1 array of 24B nodes
    "loop_009": lambda n: n * 24,
    # fp32 cascade (2 arrays)
    "loop_105": lambda n: n * 8,
    # sorts (various)
    "loop_120": lambda n: n * 4,
    "loop_121": lambda n: n * 8,   # data + temp
    "loop_122": lambda n: n * 4,
    "loop_123": lambda n: n * 8,
    "loop_124": lambda n: n * 16,  # data + temp + hist + prfx
    # string ops — bounded, not a simple n*bytes formula
    "loop_005": lambda n: n * 1,
    "loop_006": lambda n: n * 1,
    "loop_034": lambda n: n * 1,
}

# Approximate Graviton3 (c7g.large) single-core DRAM bandwidth (GB/s)
DRAM_BW_GBS = 20.0
# Approximate Graviton3 L3 bandwidth (GB/s) — above this = L3-or-faster
L3_BW_GBS = 80.0


def _ms_floor(size_bytes: int, bw_gbs: float) -> float:
    """Minimum ms/iter if bandwidth = bw_gbs GB/s (lower bound = data is in that tier)."""
    return (size_bytes / (bw_gbs * 1e9)) * 1000.0


def run_on_instance(handle, cmd: str, timeout: int = 300):
    rc, stdout, stderr = handle.run(cmd, timeout=timeout)
    return rc, stdout, stderr


def compile_at_size(handle, loop_num: str, target: str, size: int, remote_root: str = "~/arm-bench") -> bool:
    """Incremental compile: warm-up then recompile just this loop at SIZE=size."""
    obj_dir = f"{remote_root}/build/{target}/_obj"
    loop_o = f"{obj_dir}/loops/loop_{loop_num}.o"
    lnk_o = f"{obj_dir}/_lnk/loop_{loop_num}.o"

    # Recreate any missing _lnk symlinks using $PWD so paths resolve correctly over SSH.
    restore_lnk = (
        f"cd {remote_root} && "
        f"for f in build/{target}/_obj/loops/*.o build/{target}/_obj/common/*.o; do "
        f"  [ -f \"$f\" ] && ln -sf \"$PWD/$f\" \"build/{target}/_obj/_lnk/$(basename $f)\" 2>/dev/null || true; "
        f"done"
    )
    handle.run(restore_lnk, timeout=30)

    # Warm-up (builds all OTHER loops at default flags — no-op if up to date)
    rc, _, _ = handle.run(f"cd {remote_root} && make {target} 2>&1", timeout=180)
    if rc != 0:
        return False

    # Recompile just this loop at the requested SIZE
    cmd = (
        f"cd {remote_root} && rm -f {loop_o} {lnk_o} && "
        f"make {target} EXTRA_FLAGS='-DSIZE={size}' 2>&1"
    )
    rc, _, _ = handle.run(cmd, timeout=180)
    return rc == 0


def time_loop(handle, loop_num: str, target: str, n: int, remote_root: str = "~/arm-bench") -> float | None:
    """Run n iterations of loop_num and return ms/iter, or None on failure."""
    binary = f"{remote_root}/build/{target}/bin/simd_loops"
    loop_decimal = int(loop_num)
    cmd = (
        f"t0=$(date +%s%N); "
        f"{binary} -k {loop_decimal} -n {n}; rc=$?; "
        f"t1=$(date +%s%N); "
        f'echo "TIME_NS=$((t1-t0))"; exit $rc'
    )
    rc, stdout, _ = handle.run(cmd, timeout=600)
    # rc=0: correct checksum; rc=1: wrong checksum (expected at non-default SIZE).
    # Both mean the loop actually ran. rc=2 = ABORT (alloc fail / no impl).
    if rc == 2 or "ABORT" in stdout:
        return None
    if rc not in (0, 1):
        return None
    m = re.search(r"TIME_NS=(\d+)", stdout)
    if not m:
        return None
    total_ms = int(m.group(1)) / 1e6
    return round(total_ms / n, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2"])
    parser.add_argument("--n", type=int, default=10, help="Iterations per timing run")
    parser.add_argument("--loop", help="Only probe this loop, e.g. 001")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from eval.provision import get_running_instance
    from eval.config import load_problems, ISA_TIER

    handle = get_running_instance(args.isa)
    if handle is None or not _is_reachable(handle):
        print(f"No reachable instance for isa={args.isa}. Run: python eval/provision.py --isa {args.isa}")
        sys.exit(1)

    target = "c-scalar"
    problems = load_problems(with_code=False)

    print(f"\n{'='*90}")
    print(f"  Timing probe  isa={args.isa}  instance={handle.host}  target={target}  n={args.n}")
    print(f"  Graviton3 ref: DRAM≈{DRAM_BW_GBS}GB/s, L3≈{L3_BW_GBS}GB/s")
    print(f"  Goal: ms/iter≈10ms (0.1s for n={args.n}), working_set>>L3(32MB)")
    print(f"{'='*90}\n")

    # Warm-up build
    print(f"Warm-up build of {target}...")
    rc, out, _ = handle.run(f"cd ~/arm-bench && make {target} 2>&1", timeout=300)
    if rc != 0:
        print(f"  Build FAILED:\n{out[:300]}")
        sys.exit(1)
    print(f"  Build OK\n")

    results = {}  # loop_id -> {size: ms_per_iter}

    problem_ids = sorted(problems.keys())
    if args.loop:
        pid = f"loop_{args.loop.zfill(3)}"
        problem_ids = [pid] if pid in problems else []

    for pid in problem_ids:
        from eval.config import load_problem_sizes
        _, perf_sizes = load_problem_sizes(pid)
        if not perf_sizes:
            continue

        num = pid.split("_")[1]
        bytes_fn = BYTES_PER_ITER.get(pid)

        print(f"{pid}  perf_sizes={perf_sizes}")

        sizes_result = {}
        for size in perf_sizes:
            wb = bytes_fn(size) if bytes_fn else 0
            wb_mb = wb / 1e6

            ok = compile_at_size(handle, num, target, size)
            if not ok:
                print(f"  size={size:>12,}  compile FAILED")
                sizes_result[size] = None
                continue

            ms_iter = time_loop(handle, num, target, args.n)
            if ms_iter is None:
                print(f"  size={size:>12,}  run FAILED")
                sizes_result[size] = None
                continue

            total_s = ms_iter * args.n / 1000
            dram_floor = _ms_floor(wb, DRAM_BW_GBS) if wb else 0
            l3_floor   = _ms_floor(wb, L3_BW_GBS) if wb else 0

            # Classify cache level
            if wb == 0:
                cache_note = ""
            elif ms_iter >= dram_floor * 0.7:
                cache_note = "✓ DRAM-bound"
            elif ms_iter >= l3_floor * 0.7:
                cache_note = "~ L3/LLC"
            else:
                cache_note = "✗ L1/L2 cache"

            # Target assessment
            if ms_iter < 1.0:
                target_note = "WAY TOO FAST"
            elif ms_iter < 5.0:
                target_note = "too fast"
            elif ms_iter < 8.0:
                target_note = "slightly low"
            elif ms_iter <= 15.0:
                target_note = "OK"
            elif ms_iter <= 30.0:
                target_note = "slightly slow"
            else:
                target_note = "TOO SLOW"

            print(f"  size={size:>12,}  {wb_mb:6.1f}MB/iter  {ms_iter:7.2f}ms/iter  "
                  f"total={total_s:.3f}s  {cache_note}  [{target_note}]")
            sizes_result[size] = ms_iter

        results[pid] = sizes_result

        # Restore default-size binary
        default_src = REPO_ROOT / "loops" / f"loop_{num}.c"
        m = re.search(r"#define SIZE\s+(\d+)", default_src.read_text()) if default_src.exists() else None
        if m:
            compile_at_size(handle, num, target, int(m.group(1)))
        print()

    # Summary: how many loops are in the right range?
    print(f"\n{'='*90}")
    print("SUMMARY (largest PERF_SIZE):")
    print(f"{'loop':<35} {'size':>12} {'ms/iter':>10} {'status'}")
    print(f"{'-'*70}")
    ok_count = too_fast = too_slow = failed = 0
    for pid, sz_map in sorted(results.items()):
        from eval.config import load_problem_sizes
        _, perf_sizes = load_problem_sizes(pid)
        if not perf_sizes:
            continue
        largest = max(perf_sizes)
        ms = sz_map.get(largest)
        if ms is None:
            status = "FAILED"
            failed += 1
        elif ms < 5.0:
            status = "TOO FAST  ← increase size"
            too_fast += 1
        elif ms > 30.0:
            status = "TOO SLOW  ← decrease size"
            too_slow += 1
        else:
            status = "OK"
            ok_count += 1
        ms_str = f"{ms:.2f}" if ms is not None else "N/A"
        print(f"  {pid:<33} {largest:>12,} {ms_str:>10} ms/iter  {status}")

    print(f"\n  OK: {ok_count}  too-fast: {too_fast}  too-slow: {too_slow}  failed: {failed}")
    print(f"{'='*90}\n")

    return results


def _is_reachable(handle) -> bool:
    try:
        rc, _, _ = handle.run("echo ok", timeout=10)
        return rc == 0
    except Exception:
        return False


if __name__ == "__main__":
    main()
