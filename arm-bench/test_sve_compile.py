"""
test_sve_compile.py — One-shot test: ask LLM for an SVE kernel, then compile it.

Usage (from arm-bench/):
    python test_sve_compile.py [--model anthropic/claude-sonnet-4-6] [--problem conv2d]

The script:
  1. Reads the current candidate source for the problem
  2. Calls the LLM once, asking for a minimal SVE optimisation
  3. Extracts the code, compiles it, and runs correctness + timing check
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm

from eval.config import REPO_ROOT
from eval.tools import SIMDTools
from eval.evaluator import load_starter_problems

load_dotenv(REPO_ROOT / ".env")

STARTER_DIR = REPO_ROOT / "starter"


def extract_code(text: str) -> str | None:
    """Pull the first ```cpp ... ``` block out of an LLM response."""
    m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def run(problem_id: str, model: str, isa: str) -> None:
    problems = load_starter_problems()
    if problem_id not in problems:
        print(f"Unknown problem: {problem_id}. Available: {list(problems)}")
        sys.exit(1)
    problem = problems[problem_id]

    candidate_src_path = REPO_ROOT.parent / "ncnn" / problem["candidate_source"]
    if not candidate_src_path.exists():
        print(f"Candidate source not found: {candidate_src_path}")
        sys.exit(1)
    candidate_source = candidate_src_path.read_text()

    # ── Step 1: single LLM call ──────────────────────────────────────────────
    print(f"[1/3] Calling {model} for SVE optimisation of {problem_id}...")
    prompt = f"""\
You are an AArch64 expert. Below is an ncnn kernel in C++.
Add a minimal SVE optimisation to the inner loop using ARM SVE ACLE intrinsics
(svfloat32_t, svdup_f32, svmla_f32_z, etc.).

Return ONLY the complete modified C++ source in a single ```cpp block.

Current source:
```cpp
{candidate_source}
```
"""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    reply = response.choices[0].message.content
    print(f"   LLM replied ({len(reply)} chars)")

    code = extract_code(reply)
    if code is None:
        print("   ERROR: no ```cpp block found in response")
        print("   Raw reply:\n", reply[:500])
        sys.exit(1)
    print(f"   Extracted code: {len(code)} chars\n")

    # ── Step 2: compile ──────────────────────────────────────────────────────
    tools = SIMDTools(handle=None, problem_id=problem_id, isa=isa)
    tools.upload_ncnn_tree()

    # Write the LLM code to the candidate source path on the remote
    import tempfile, os, shutil
    remote_path = f"{tools.remote_ncnn_root}/{problem['candidate_source']}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        tools._upload(tmp, remote_path)
    finally:
        os.unlink(tmp)

    print(f"[2/3] Compiling with isa={isa}...")
    cr = tools.compile(problem["starter"])
    if cr.success:
        print("   compile: OK")
        if cr.warnings:
            print(f"   warnings: {cr.warnings}")
    else:
        print("   compile: FAILED")
        print(f"   errors:\n{cr.errors}")
        sys.exit(1)

    # ── Step 3: run correctness check ────────────────────────────────────────
    print(f"\n[3/3] Running correctness + timing check (n=5)...")
    rr = tools.run(n=5)
    if rr.correct:
        print("   correctness: PASSED")
        print(f"   candidate runtime : {rr.candidate_runtime_ms} ms (total 5 iters)")
        print(f"   baseline runtime  : {rr.baseline_runtime_ms} ms (total 5 iters)")
        if rr.candidate_runtime_ms and rr.baseline_runtime_ms and rr.candidate_runtime_ms > 0:
            speedup = round(rr.baseline_runtime_ms / rr.candidate_runtime_ms, 2)
            print(f"   speedup vs baseline: {speedup}x")
    else:
        print("   correctness: FAILED")
        print(f"   output:\n{rr.output[:600]}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="conv2d")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2"])
    args = parser.parse_args()
    run(args.problem, args.model, args.isa)
