#!/usr/bin/env python3
"""Correctness check of the baseline-sme2 simd-loop solutions without SME hardware.

No AWS instance type has SME, and the sme2 tier targets Apple M4. This runs the
solutions on an emulated SME2 CPU instead: inside Linux aarch64 with clang and
QEMU user-mode (>= 10.1 for the SME2 ZA instructions), started as

    qemu-aarch64 -cpu max,sme-default-vector-length=64 \\
        $(which python3) analysis/sme_qemu_check.py [--max-elems N] [loop_216 ...]

where 64 bytes is M4's streaming vector length. Each solution is compiled the
way the harness compiles it and checked with the harness's own evaluator on the
definition's workloads small enough to emulate. Evaluation runs in-process: the
harness's isolation subprocess would be a new exec and run outside the emulator.
Only pass/fail is meaningful; emulated timings are not reported.

clang emits calls to the SME ABI routine __arm_get_current_vg for locally
streaming functions. macOS provides it; Ubuntu's compiler-rt does not, so this
script loads a one-function shim globally before any solution is dlopened.
"""
import argparse
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.config import EvalConfig  # noqa: E402
from bench.data.trace import EvaluationStatus  # noqa: E402
from bench.data.trace_set import TraceSet  # noqa: E402
from bench.runner import _run_solution_on_workloads_direct  # noqa: E402

AUTHOR = "baseline-sme2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("loops", nargs="*", help="loop ids (default: every baseline-sme2 solution)")
    ap.add_argument("--max-elems", type=int, default=13_000_000,
                    help="skip workloads whose largest input has more elements than this")
    args = ap.parse_args()

    _load_sme_abi_shim()
    ts = TraceSet.from_path(ROOT / "bench-trace")
    cfg = EvalConfig(warmup=0, repeat=1, inner_iters=1)
    results = {}
    for name, definition in sorted(ts.definitions.items()):
        if args.loops and name not in args.loops:
            continue
        solution = ts.get_baseline_solution(name, AUTHOR)
        if solution is None:
            continue
        workloads = [
            w for w in ts.get_workloads(name)
            if max(_elems(definition, w, t) for t in definition.inputs) <= args.max_elems
        ]
        traces = _run_solution_on_workloads_direct(
            definition, solution, workloads, is_baseline=True, cfg=cfg,
        )
        statuses = [t.evaluation.status for t in traces]
        ok = bool(statuses) and all(s == EvaluationStatus.PASSED for s in statuses)
        results[name] = ok
        sizes = ", ".join(f"{_axes(t.workload.axes)}={t.evaluation.status.value}" for t in traces)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {sizes}", flush=True)
        if not ok:
            log = next((t.evaluation.log for t in traces
                        if t.evaluation.status != EvaluationStatus.PASSED), "")
            print("      " + (log or "").strip().replace("\n", "\n      ")[:1500], flush=True)

    passed = sorted(k for k, v in results.items() if v)
    failed = sorted(k for k, v in results.items() if not v)
    print(f"\n{len(passed)}/{len(results)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed or not results else 0)


# __arm_get_current_vg returns VG (the vector length in 64-bit granules) of the
# current mode. The emulated CPU has non-streaming SVE as well as SME, so CNTD is
# valid, and correct, in either mode.
_SME_ABI_SHIM = """
__attribute__((naked)) unsigned long __arm_get_current_vg(void) {
  __asm__ volatile("cntd x0\\n ret");
}
"""


def _load_sme_abi_shim() -> None:
    build = Path(tempfile.mkdtemp(prefix="sme_abi_shim_"))
    (build / "shim.c").write_text(_SME_ABI_SHIM)
    subprocess.run(["clang", "-shared", "-fPIC", "-march=armv9-a+sve2+sme2", "-o",
                    str(build / "shim.so"), str(build / "shim.c")], check=True)
    ctypes.CDLL(str(build / "shim.so"), mode=ctypes.RTLD_GLOBAL)


def _elems(definition, workload, tensor: str) -> int:
    shape = definition.inputs[tensor].shape or []
    n = 1
    for axis in shape:
        n *= workload.axes.get(axis) or definition.axes[axis].value
    return n


def _axes(axes: dict) -> str:
    return "x".join(f"{k}{v}" for k, v in axes.items())


if __name__ == "__main__":
    main()
