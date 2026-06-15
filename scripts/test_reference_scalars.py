#!/usr/bin/env python3
"""
Quick correctness smoke-test: compile and run every reference-scalar solution
against the Python reference for all registered simd-loop problems.
"""
import ctypes, shutil, sys, pathlib
import numpy as np

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from bench.data import Solution, Definition, Workload
from bench.compile.builders.simd_loop import SimdLoopBuilder
from bench.datasets.simd_loop import SimdLoopDataset
from bench.runtime.inputs import gen_inputs_for_workload

DEFS_DIR  = REPO / "bench-trace/definitions/simd-loop"
WL_DIR    = REPO / "bench-trace/workloads/simd-loop"
SOL_DIR   = REPO / "bench-trace/solutions/simd-loop/reference-scalar"

def run_one(loop_id: str) -> tuple[int, int, list[str]]:
    """Returns (passed, total, [error_messages])."""
    def_path = DEFS_DIR / f"{loop_id}.json"
    wl_path  = WL_DIR  / f"{loop_id}.jsonl"
    sol_path = SOL_DIR / loop_id / f"reference-scalar_{loop_id}.json"

    if not all(p.exists() for p in [def_path, wl_path, sol_path]):
        return 0, 0, [f"missing files for {loop_id}"]

    defn = Definition.model_validate_json(def_path.read_text())
    sol  = Solution.model_validate_json(sol_path.read_text())
    workloads = [Workload.model_validate_json(l)
                 for l in wl_path.read_text().splitlines() if l.strip()]

    builder = SimdLoopBuilder()
    try:
        compiled = builder.build(defn, sol)
    except Exception as e:
        return 0, len(workloads), [f"compile error: {e}"]

    errors = []
    passed = 0
    try:
        lib = ctypes.CDLL(str(compiled.so_path))
        sym = getattr(lib, f"armbench_entry_{loop_id}")
        sym.restype = ctypes.c_int
        sym._lib = lib

        ds = SimdLoopDataset()
        exec_ns: dict = {}
        exec(defn.reference, exec_ns)
        ref_fn = exec_ns["run"]

        for wl in workloads:
            np_inputs = gen_inputs_for_workload(defn, wl)
            ctx = ds.wrap_inputs(np_inputs, loop_id, lib, definition=defn)
            rc = sym(*ctx.entry_args)
            if rc != 0:
                errors.append(f"N={wl.axes['N']}: runtime error rc={rc}")
                ds.release(ctx)
                continue

            out = ds.unwrap_output(ctx)
            ref_val = ref_fn(**np_inputs)
            ref_arr = np.asarray([ref_val]) if isinstance(ref_val, np.generic) else np.asarray(ref_val)
            got_arr = np.asarray(out).flatten()[:ref_arr.size]

            # Hybrid abs+rel tolerance (same as correctness.py defaults)
            abs_tol, rel_tol = 1e-3, 1e-3
            diff = np.abs(got_arr.astype(np.float64) - ref_arr.astype(np.float64).flatten())
            tol  = abs_tol + rel_tol * np.abs(ref_arr.astype(np.float64).flatten())
            if np.any(diff > tol):
                idx = int(np.argmax(diff > tol))
                errors.append(
                    f"N={wl.axes['N']}: FAIL max_abs={diff.max():.3e} "
                    f"max_rel={float(diff[idx]/max(abs(float(ref_arr.flat[idx])),1e-12)):.3e} "
                    f"got={float(got_arr.flat[idx]):.6f} ref={float(ref_arr.flat[idx]):.6f}"
                )
            else:
                passed += 1
            ds.release(ctx)
    finally:
        shutil.rmtree(compiled.build_dir, ignore_errors=True)

    return passed, len(workloads), errors


def main():
    loop_ids = sorted(p.stem for p in DEFS_DIR.glob("loop_*.json"))
    total_pass = total_wl = 0
    any_fail = False

    for loop_id in loop_ids:
        p, t, errs = run_one(loop_id)
        total_pass += p
        total_wl   += t
        status = "PASS" if not errs else "FAIL"
        if errs:
            any_fail = True
        print(f"  [{status}] {loop_id:<12} {p}/{t}")
        for e in errs:
            print(f"           {e}")

    print(f"\n{total_pass}/{total_wl} workloads passed across {len(loop_ids)} loops")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
