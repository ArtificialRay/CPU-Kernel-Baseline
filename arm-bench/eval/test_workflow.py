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
from eval.tools import SIMDTools

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
    "conv": "convolution.cpp"
}

#DEFAULT_PROBLEM = "loop_001"
DEFAULT_PROBLEM = "conv"
DEFAULT_ISA = "sve"


def run_smoke_test(problem_id: str, isa: str, teardown: bool):
    print(f"\n{'='*60}")
    print(f"  arm-bench smoke test")
    print(f"  problem={problem_id}  isa={isa}")
    print(f"{'='*60}\n")

    # # 1. Provision (or reuse) instance
    # print("[1/5] Provisioning instance...")
    # handle = get_or_provision(isa)
    # print(f"      Host: {handle.host}\n")

    candidate = DUMMY_CANDIDATES.get(problem_id)
    if candidate is None:
        print(f"No dummy candidate defined for {problem_id}. Add one to DUMMY_CANDIDATES.")
        sys.exit(1)

    tools = SIMDTools(handle=None, problem_id=problem_id, isa=isa)

    # 1. Upload ncnn source tree + starter files to remote work dir
    print("[1/4]  Uploading ncnn tree...")
    tools.upload_ncnn_tree()
    print(f"      Synced to {tools.remote_project_root}\n")

    # 2. Compile
    print("[2/4] compile()...")
    cr = tools.compile(candidate)
    print(f"      success={cr.success}")
    if not cr.success:
        print(f"      ERRORS:\n{cr.errors}")
        sys.exit(1)
    if cr.warnings:
        print(f"      warnings: {cr.warnings}")
    print()

    # 3. Run (candidate + baseline)
    print("[3/4] run()...")
    rr = tools.run(n=1)
    print(f"      correct={rr.correct}  candidate_runtime_ms={rr.candidate_runtime_ms}  baseline_runtime_ms={rr.baseline_runtime_ms}")
    print(f"      {rr.output}")
    if not rr.correct:
        sys.exit(1)
    print()

    # 4. Perf (candidate + baseline, output returns only the candidate perf info)
    print("[4/4] perf()...")
    pr_candidate,pr_baseline = tools.perf(n=1)
    print(f"      Candidates: cycles={pr_candidate.cycles}  instructions={pr_candidate.instructions}  ipc={pr_candidate.ipc}  l1d_miss%={pr_candidate.l1d_miss_pct}")
    print(f"      Baseline: cycles={pr_baseline.cycles}  instructions={pr_baseline.instructions}  ipc={pr_baseline.ipc}  l1d_miss%={pr_baseline.l1d_miss_pct}")
    print()

    # # 5. Disassemble
    # fn = f"inner_loop_{tools.loop_num}"
    # print(f"[5/5] disassemble(fn='{fn}')...")
    # dr = tools.disassemble(fn=fn)
    # lines = dr.asm.splitlines()
    # preview = "\n      ".join(lines[:20])
    # print(f"      {preview}")
    # if len(lines) > 20:
    #     print(f"      ... ({len(lines)} lines total)")
    # print()

    # Summary
    print(f"{'='*60}")
    print(f"  Smoke test PASSED")
    print(f"  Candidate runtime_ms : {rr.candidate_runtime_ms}")
    print(f"  Candidate cycles     : {pr_candidate.cycles}")
    print(f"  Candidate instructions: {pr_candidate.instructions}")
    print(f"  Candidate IPC        : {pr_candidate.ipc}")
    print(f"  Candidate L1D miss % : {pr_candidate.l1d_miss_pct}")
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
