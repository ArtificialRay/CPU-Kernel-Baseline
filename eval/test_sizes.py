"""
eval/test_sizes.py — Integration test for the variable-size testing machinery.

Tests:
  1. run(size=...)  — explicit size on the run tool
  2. perf(size=...) — explicit size on the perf tool
  3. submit() edge-size correctness — EDGE_SIZES checked against c-scalar
  4. submit() perf-size collection  — PERF_SIZES timed and stored in perf_by_size
  5. Broken candidate detection     — tail-bug candidate caught by EDGE_SIZES

Uses loop_001 (FP32 inner product) with a known-good scalar candidate and a
deliberately broken one (ignores tail when n is not a multiple of 8).

Usage:
    python -m eval.test_sizes [--isa sve]
"""

import sys
from eval.provision import get_or_provision
from eval.tools import SIMDTools

# ── Candidates ────────────────────────────────────────────────────────────────

# Correct: scalar inner product, handles any n
GOOD_CANDIDATE = """\
static void inner_loop_001(struct loop_001_data *restrict data) {
    float *a = data->a;
    float *b = data->b;
    int n = data->n;
    float res = 0.0f;
    for (int i = 0; i < n; i++)
        res += a[i] * b[i];
    data->res = res;
}
"""

# Broken: skips the tail when n % 8 != 0 — wrong at sizes like 1, 7, 9999
BROKEN_CANDIDATE = """\
static void inner_loop_001(struct loop_001_data *restrict data) {
    float *a = data->a;
    float *b = data->b;
    int n = data->n;
    float res = 0.0f;
    int aligned = n - (n % 8);   /* BUG: tail is silently dropped */
    for (int i = 0; i < aligned; i++)
        res += a[i] * b[i];
    data->res = res;
}
"""

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}" + (f"  — {detail}" if detail else ""))
    return ok

def run_tests(isa: str):
    print(f"\n{'='*62}")
    print(f"  arm-bench size-testing integration test  (isa={isa})")
    print(f"{'='*62}\n")

    handle = get_or_provision(isa)
    print(f"  Instance: {handle.host}\n")

    tools = SIMDTools(handle=handle, problem_id="loop_001", isa=isa)

    # Show configured sizes for visibility
    print(f"  EDGE_SIZES : {tools._edge_sizes}")
    print(f"  PERF_SIZES : {tools._perf_sizes}\n")

    all_pass = True

    # ── 1. Compile good candidate ────────────────────────────────────────────
    print("── 1. compile(good candidate) ──")
    cr = tools.compile(GOOD_CANDIDATE)
    all_pass &= check("compile succeeded", cr.success, cr.errors if not cr.success else "")
    if not cr.success:
        print("\nCannot continue — compile failed.")
        sys.exit(1)
    print()

    # ── 2. run(size=...) explicit sizes ──────────────────────────────────────
    # correct=True means "ran without abort/crash" (hardcoded checksum is only
    # valid at the default size, so we don't use it for size-specific runs).
    print("── 2. run(size=...) ──")
    for size in [1, 7, 100, 50_000]:
        rr = tools.run(n=3, size=size)
        all_pass &= check(
            f"run(size={size:>7}) ran_ok={rr.correct}  runtime_ms={rr.runtime_ms}",
            rr.correct,
            rr.output[:80] if not rr.correct else "",
        )
    print()

    # ── 3. perf(size=...) at a large size ────────────────────────────────────
    print("── 3. perf(size=...) ──")
    for size in [10_000, 500_000]:
        pr = tools.perf(n=20, size=size)
        ran = pr.cycles is not None or "correct" in pr.raw_output.lower()
        all_pass &= check(
            f"perf(size={size:>7}) ran without error",
            not pr.raw_output.startswith("size=") and "failed" not in pr.raw_output,
            f"cycles={pr.cycles}",
        )
    print()

    # ── 4. submit() — good candidate: edge + perf sizes ─────────────────────
    print("── 4. submit(good candidate) ──")
    er = tools.submit(GOOD_CANDIDATE)
    all_pass &= check("correct",          er.correct)
    all_pass &= check("level >= 1",       er.level >= 1, f"level={er.level}")
    all_pass &= check("runtime_ms set",   er.runtime_ms is not None, f"{er.runtime_ms} ms")
    all_pass &= check(
        "perf_by_size populated",
        er.perf_by_size is not None and len(er.perf_by_size) > 0,
        str(er.perf_by_size),
    )
    if er.perf_by_size:
        for size, ms in sorted(er.perf_by_size.items()):
            label = f"  perf_by_size[{size:>9}]"
            all_pass &= check(label, ms is not None, f"{ms} ms/iter" if ms else "None (failed)")
    print()

    # ── 5. submit() — broken candidate: caught by edge sizes ─────────────────
    print("── 5. submit(broken candidate — tail bug) ──")
    tools2 = SIMDTools(handle=handle, problem_id="loop_001", isa=isa)
    er2 = tools2.submit(BROKEN_CANDIDATE)
    all_pass &= check(
        "broken candidate rejected",
        not er2.correct,
        er2.compile_error[:100] if er2.compile_error else "(no error message)",
    )
    if not er2.correct and er2.compile_error:
        print(f"         reason: {er2.compile_error}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*62}")
    if all_pass:
        print(f"  \033[32mAll tests passed.\033[0m")
    else:
        print(f"  \033[31mSome tests FAILED.\033[0m")
    print(f"{'='*62}\n")

    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2"])
    args = parser.parse_args()
    ok = run_tests(args.isa)
    sys.exit(0 if ok else 1)
