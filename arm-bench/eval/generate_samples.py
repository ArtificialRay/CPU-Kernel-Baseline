"""
eval/generate_samples.py — Single-shot LLM generation (KernelBench-compatible).

Calls the LLM once per problem and saves the generated C code to generations/.
Mirrors KernelBench's generate_samples.py structure so existing tooling works.

Usage:
    python eval/generate_samples.py --problem loop_001 --isa sve2 --model claude-opus-4-6
    python eval/generate_samples.py --all --isa sve2 --model claude-opus-4-6
    python eval/generate_samples.py --all --isa sve2 --model claude-opus-4-6 --workers 4
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

from eval.config import load_problems, ISA_INSTANCE_DESC, REPO_ROOT

GENERATIONS_DIR = REPO_ROOT / "generations"

SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. You will be given a scalar C loop
implementation and asked to rewrite it using a specific SIMD instruction set.

Rules:
  1. Preserve the exact function signature.
  2. The `res` checksum field must produce the same result as the scalar version.
  3. Output ONLY the C function — no explanation, no markdown fences, no includes
     beyond what is necessary. Do not include the struct definition.
  4. You may use inline assembly or ACLE intrinsics.
"""

USER_PROMPT_TEMPLATE = """\
Target ISA: {isa_upper} on {isa_desc}

Struct definition:
```c
{struct_def}
```

Scalar implementation (optimize this):
```c
{scalar_code}
```

Write an optimized {isa_upper} implementation. Output only the C function body.
"""


def generate_one(problem_id: str, isa: str, model: str) -> str:
    """
    Call the LLM once and return the generated C code string.
    """
    problems = load_problems()
    problem = problems[problem_id]

    prompt = USER_PROMPT_TEMPLATE.format(
        isa_upper=isa.upper(),
        isa_desc=ISA_INSTANCE_DESC.get(isa, isa),
        struct_def=problem.get("struct_def", ""),
        scalar_code=problem.get("scalar_code", ""),
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def extract_c_code(text: str) -> str:
    """
    Strip markdown fences if the model wrapped the output, return raw C.
    """
    # Try to extract content between ```c ... ``` or ``` ... ```
    m = re.search(r"```(?:c|cpp)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def save_generation(problem_id: str, isa: str, code: str, run_id: int = 0):
    """Save generated code to generations/<problem_id>_<isa>_run<N>.c"""
    GENERATIONS_DIR.mkdir(exist_ok=True)
    path = GENERATIONS_DIR / f"{problem_id}_{isa}_run{run_id}.c"
    path.write_text(code)
    return path


def main():
    parser = argparse.ArgumentParser(description="Single-shot LLM generation for simd-loops")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--problem", help="Problem ID, e.g. loop_001")
    grp.add_argument("--all", action="store_true", help="Generate for all problems")
    parser.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
    parser.add_argument("--model", required=True,
                        help="LiteLLM model string, e.g. anthropic/claude-opus-4-6")
    parser.add_argument("--workers", type=int, default=1, help="Parallel generation workers")
    parser.add_argument("--run-id", type=int, default=0, help="Run index suffix for output file")
    args = parser.parse_args()

    problems = load_problems()

    if args.problem:
        problem_ids = [args.problem]
    else:
        # Filter to problems matching the requested ISA tier
        problem_ids = [
            pid for pid, p in problems.items()
            if p.get("isa_target") == args.isa
        ]
        print(f"Generating for {len(problem_ids)} problems (ISA: {args.isa})")

    def _generate(pid):
        print(f"  Generating {pid} ...")
        try:
            raw = generate_one(pid, args.isa, args.model)
            code = extract_c_code(raw)
            path = save_generation(pid, args.isa, code, args.run_id)
            print(f"  Saved: {path}")
            return pid, True
        except Exception as e:
            print(f"  FAILED {pid}: {e}")
            return pid, False

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_generate, pid): pid for pid in problem_ids}
            for f in as_completed(futures):
                pid, ok = f.result()
    else:
        for pid in problem_ids:
            _generate(pid)

    print("Done.")


if __name__ == "__main__":
    main()
