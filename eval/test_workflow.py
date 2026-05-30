"""
eval/test_workflow.py — End-to-end smoke test for the AWS eval pipeline.

Provisions a c7g.large instance, injects a known-good scalar implementation
as the candidate, then exercises compile → run → perf → disassemble → submit.
No LLM involved.

Usage:
    python -m eval.test_workflow [--teardown] [--problem loop_001] [--isa sve]
"""

import argparse
import json
import sys

from eval.provision import get_or_provision, teardown as do_teardown
from eval.tools import SIMDTools, NCNNTools

# ---------------------------------------------------------------------------
# Dummy candidate: scalar FP32 inner product (loop_001, always correct)
# ---------------------------------------------------------------------------

DUMMY_CANDIDATES = {
    "loop_001": """\
static void inner_loop_001(struct loop_001_data *restrict data) {
    float *a = data->a;
    float *b = data->b;
    int n = data->n;
    float res = 0.0f;
    for (int i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    data->res = res;
}
""",
    # NB: "conv" (conv2d) was migrated to bench-trace/ in Phase 2; the legacy
    # starter/tests/perf/candidates_src files are gone, so no dummy candidate
    # is supplied here. Use --problem conv1d / convdw / deconv / deconvdw for
    # the still-legacy ops, or run bench-trace via `python -m bench.cli bench`.
}

DEFAULT_PROBLEM = "loop_001"
DEFAULT_ISA = "sve"


def run_smoke_test(problem_id: str, isa: str, teardown: bool):
    print(f"\n{'='*60}")
    print(f"  arm-bench smoke test")
    print(f"  problem={problem_id}  isa={isa}")
    print(f"{'='*60}\n")

    # 1. Provision (or reuse) instance
    print("[1/5] Provisioning instance...")
    handle = get_or_provision(isa)
    print(f"      Host: {handle.host}\n")

    candidate = DUMMY_CANDIDATES.get(problem_id)
    if candidate is None:
        print(f"No dummy candidate defined for {problem_id}. Add one to DUMMY_CANDIDATES.")
        sys.exit(1)

    ToolsCls = NCNNTools if problem_id.startswith(("conv", "deconv")) else SIMDTools
    tools = ToolsCls(handle=handle, problem_id=problem_id, isa=isa)

    # 2. Compile
    print("[2/5] compile()...")
    cr = tools.compile(candidate)
    print(f"      success={cr.success}")
    if not cr.success:
        print(f"      ERRORS:\n{cr.errors}")
        sys.exit(1)
    if cr.warnings:
        print(f"      warnings: {cr.warnings}")
    print()

    # 3. Run
    print("[3/5] run(n=10)...")
    rr = tools.run(n=10)
    print(f"      correct={rr.correct}  runtime_ms={rr.runtime_ms}")
    if not rr.correct:
        print(f"      output: {rr.output}")
        sys.exit(1)
    print()

    # 4. Perf
    print("[4/5] perf(n=10)...")
    pr = tools.perf(n=10)
    print(f"      cycles={pr.cycles}  instructions={pr.instructions}  ipc={pr.ipc}  cache_misses/iter={pr.cache_misses_per_iter}  task_clock_ms={pr.task_clock_ms}")
    print()

    # 5. Disassemble
    fn = "convolution_kernel" if problem_id == "conv" else f"inner_loop_{tools.loop_num}"
    print(f"[5/5] disassemble(fn='{fn}')...")
    dr = tools.disassemble(fn=fn)
    lines = dr.asm.splitlines()
    preview = "\n      ".join(lines[:20])
    print(f"      {preview}")
    if len(lines) > 20:
        print(f"      ... ({len(lines)} lines total)")
    print()

    # Summary
    print(f"{'='*60}")
    print(f"  Smoke test PASSED")
    print(f"  runtime_ms : {rr.runtime_ms}")
    print(f"  cycles     : {pr.cycles}")
    print(f"  instructions: {pr.instructions}")
    print(f"  IPC        : {pr.ipc}")
    print(f"  LLC misses/iter: {pr.cache_misses_per_iter}")
    print(f"  task_clock : {pr.task_clock_ms} ms/iter")
    print(f"{'='*60}\n")

    if teardown:
        print("[teardown] Destroying instance...")
        do_teardown()
        print("[teardown] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for arm-bench AWS pipeline")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--isa", default=DEFAULT_ISA, choices=["neon", "sve", "sve2", "sme2"])
    parser.add_argument("--teardown", action="store_true", help="Destroy instance after test")
    args = parser.parse_args()

    run_smoke_test(args.problem, args.isa, args.teardown)
