"""
scripts/collect_baselines_ncnn.py — Collect baseline timings for ncnn kernels.

For each problem in starter/ncnn/problems.json, builds the perf_candidate_{name}
and perf_baseline_{name} CMake targets on a remote Arm instance, runs the
matching test binaries once as a correctness gate, then times each perf binary
n times and records the results to baselines/ncnn.json:

  { "conv": { "candidate_ms": 12.3, "baseline_ms": 4.5, "ref_ms": null }, ... }

Where:
  candidate_ms = ms/invocation of perf_candidate_{name}
                 (scalar C kernel in candidates_src/ncnn/{kernel}.cpp)
  baseline_ms  = ms/invocation of perf_baseline_{name}
                 (ARM-heavy-optimized reference:
                  CPU-Kernel-Baseline/ncnn/arm-heavy-optimized/conv/{kernel}_arm.cpp)

NCNNTools.submit() consumes these: an agent that beats candidate_ms earns
level 2; beats baseline_ms earns level 3.

Assumes the top-level CMake model from arm-bench/CMakeLists.txt, which lives
at ~/arm-bench/ on the remote (see NCNNTools for the expected layout).

Usage (from arm-bench/):
    # Provision a remote instance first (or reuse running one):
    python eval/provision.py --instance c7g.large

    # Collect all kernels:
    python -m scripts.collect_baselines_ncnn --isa sve --n 10

    # Or a single one:
    python -m scripts.collect_baselines_ncnn --problem conv --isa sve
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import NCNN_PROBLEMS_JSON, BASELINES_DIR
from eval.provision import get_running_instance

DEFAULT_N = 10
DEFAULT_ISA = "sve"

REMOTE_ARM_BENCH = "~/arm-bench"
REMOTE_BUILD     = "~/arm-bench/build"
REMOTE_NCNN_BASE = "$HOME/ncnn"


def _run(handle, cmd: str, timeout: int = 120):
    return handle.run(cmd, timeout=timeout)


def setup_cmake(handle, isa: str) -> bool:
    """Configure the top-level arm-bench CMake once."""
    march = {
        "neon": "armv8.2-a+fp16+dotprod",
        "sve":  "armv8.2-a+fp16+dotprod+sve",
        "sve2": "armv8.2-a+fp16+dotprod+sve2",
    }.get(isa, "armv8.2-a+fp16+dotprod")

    cmd = (
        f"cmake -S {REMOTE_ARM_BENCH} -B {REMOTE_BUILD} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DBASE_ROOT={REMOTE_NCNN_BASE} "
        f"-DCMAKE_CXX_FLAGS='-march={march}' 2>&1"
    )
    rc, output, _ = _run(handle, cmd, timeout=120)
    if rc != 0:
        print(f"    cmake configure failed:\n{output[:500]}")
        return False
    return True


def build_targets(handle, name: str) -> bool:
    """Incrementally build one kernel's test + perf binaries on both sides."""
    cmd = (
        f"cmake --build {REMOTE_BUILD} "
        f"--target test_candidate_{name} --target perf_candidate_{name} "
        f"--target test_baseline_{name}  --target perf_baseline_{name} "
        f"-j$(nproc) 2>&1"
    )
    rc, output, _ = _run(handle, cmd, timeout=240)
    if rc != 0:
        errs = "\n".join(l for l in output.splitlines() if "error:" in l.lower())
        print(f"    cmake build failed:\n{errs or output[:500]}")
        return False
    return True


def check_correct(handle, binary: str) -> bool:
    """Run the test binary once; passes iff rc==0 and stdout has no FAIL."""
    rc, stdout, _ = _run(handle, binary, timeout=60)
    return rc == 0 and "FAIL" not in stdout


def time_binary(handle, binary: str, n: int) -> float | None:
    """Run binary n times back-to-back; return ms/invocation or None."""
    cmd = (
        f"t0=$(date +%s%N); "
        f"for i in $(seq 1 {n}); do {binary}; rc=$?; "
        f"if [ $rc -ne 0 ]; then exit $rc; fi; done; "
        f"t1=$(date +%s%N); "
        f'echo "TIME_NS=$((t1-t0))"'
    )
    rc, stdout, _ = _run(handle, cmd, timeout=600)
    if rc != 0:
        return None
    m = re.search(r"TIME_NS=(\d+)", stdout)
    return round(int(m.group(1)) / 1e6 / n, 3) if m else None


def collect_one(handle, problem: dict, n: int) -> dict | None:
    name = problem["id"]  # conv / conv1d / convdw / deconv / deconvdw
    print(f"  [{name}]")
    if not build_targets(handle, name):
        return None

    test_cand = f"{REMOTE_BUILD}/test_candidate_{name}"
    test_base = f"{REMOTE_BUILD}/test_baseline_{name}"
    perf_cand = f"{REMOTE_BUILD}/perf_candidate_{name}"
    perf_base = f"{REMOTE_BUILD}/perf_baseline_{name}"

    # Correctness gate on both sides before we trust any timing
    if not check_correct(handle, test_cand):
        print(f"    [candidate] correctness check failed — skipping")
        return None
    if not check_correct(handle, test_base):
        print(f"    [baseline]  correctness check failed — skipping")
        return None

    candidate_ms = time_binary(handle, perf_cand, n)
    baseline_ms  = time_binary(handle, perf_base, n)
    if candidate_ms is None and baseline_ms is None:
        print(f"    timing failed on both sides — skipping")
        return None

    print(f"    candidate_ms={candidate_ms}   baseline_ms={baseline_ms}")
    return {
        "candidate_ms": candidate_ms,
        "baseline_ms":  baseline_ms,
        "ref_ms":       None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect candidate/baseline timings for ncnn kernels"
    )
    parser.add_argument("--problem", help="Single problem ID (default: all)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Invocations per binary (default: {DEFAULT_N})")
    parser.add_argument("--isa", default=DEFAULT_ISA, choices=["neon", "sve", "sve2"],
                        help=f"ISA tier (default: {DEFAULT_ISA})")
    args = parser.parse_args()

    if not NCNN_PROBLEMS_JSON.exists():
        print(f"ERROR: {NCNN_PROBLEMS_JSON} not found.")
        sys.exit(1)

    problems_raw = json.loads(NCNN_PROBLEMS_JSON.read_text())
    if args.problem:
        problems_raw = [p for p in problems_raw if p["id"] == args.problem]
        if not problems_raw:
            print(f"Problem {args.problem!r} not found in {NCNN_PROBLEMS_JSON}")
            sys.exit(1)

    handle = get_running_instance(args.isa)
    if handle is None:
        print(f"No running instance for ISA={args.isa}. "
              f"Run: python eval/provision.py --instance c7g.large")
        sys.exit(1)

    print(f"\nConfiguring cmake on remote instance...")
    if not setup_cmake(handle, args.isa):
        print("cmake setup failed — aborting")
        sys.exit(1)

    print(f"\nCollecting baselines for {len(problems_raw)} kernels "
          f"(n={args.n}, isa={args.isa})")
    print(f"{'='*60}\n")

    results = {}
    for prob in problems_raw:
        entry = collect_one(handle, prob, args.n)
        if entry is None:
            print(f"    SKIPPED\n")
            continue
        results[prob["id"]] = entry
        print()

    BASELINES_DIR.mkdir(exist_ok=True)
    out_path = BASELINES_DIR / "ncnn.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))

    print(f"{'='*60}")
    print(f"Wrote {len(results)} entries to {out_path}\n")
    print(f"{'Problem':<15} {'candidate_ms':<15} {'baseline_ms':<15}")
    print("-" * 50)
    for pid, entry in results.items():
        c = entry.get("candidate_ms", "FAIL")
        b = entry.get("baseline_ms",  "N/A")
        print(f"{pid:<15} {str(c):<15} {str(b):<15}")


if __name__ == "__main__":
    main()