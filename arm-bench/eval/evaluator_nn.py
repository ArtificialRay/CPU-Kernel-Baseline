"""
eval/evaluator_nn.py — Two-phase LLM optimization workflow for ncnn kernels.

Unlike the free-form agentic loop in evaluator.py, each pass here follows a
fixed deterministic pipeline:

  1. Compile the current candidate code
  2. Profile it (run timing + perf hardware counters)
  3. Planner LLM: receives profiling data + source → detailed optimization plan
  4. Generator LLM: receives plan + source → new C++ code
  5. Repeat up to max_passes; finalize with submit()

Compatible with any LiteLLM-supported model.

CLI integration note: add --workflow nn to run_benchmark.py to call run_nn_eval
instead of run_agentic_eval.
"""

import json
import re
import time

from dotenv import load_dotenv
import litellm

from eval.config import REPO_ROOT, ISA_INSTANCE_DESC
from eval.evaluator import (
    load_starter_problems,
    load_arm_baselines,
    _write_candidate_source,
    _force_final_score,
)
from eval.tools import SIMDTools, EvalResult

load_dotenv(REPO_ROOT / ".env")


# ─── System prompts ──────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are an expert AArch64 performance analyst specializing in ARM NEON/SVE
neural network kernels. Your role is to analyze profiling data and produce a
clear, actionable optimization plan.

You will be given:
  - Problem description and ISA target
  - Current candidate source code
  - Candidate and baseline performance metrics (runtime, cycles, IPC, L1D miss rate)
  - Compile status (success or errors)

Your output must be a numbered list of specific optimization steps to improve
the candidate kernel. Reason about bottlenecks using the profiling data.

Rules:
  - Do NOT write any code. Only produce a plan in prose.
  - Be specific: name relevant NEON/SVE intrinsics, loop transformations, or
    memory access patterns to apply.
  - If the code failed to compile, identify and explain the likely root cause
    and how to fix it before optimizing.
  - Keep the plan concise and actionable — 3 to 8 numbered steps.
"""

GENERATOR_SYSTEM_PROMPT = """\
You are an expert AArch64 C++ programmer specializing in ARM NEON/SVE
optimizations for neural network kernels. Your role is to implement an
optimization plan as a complete, compilable C++ source file.

You will be given:
  - A numbered optimization plan
  - The current candidate C++ source code

Your output must be a single complete C++ source file in a ```cpp code block.
No prose, no explanation — only the code block. The file must be a full
replacement for the candidate source (include all headers, namespaces, class
methods, and helper functions).
"""


# ─── Prompt builders ─────────────────────────────────────────────────────────

def build_planner_prompt(
    problem: dict,
    candidate_source: str,
    baselines: dict,
    isa: str,
    compile_result,
    run_result,
    perf_pair,
) -> str:
    """
    Build the user message for the planner LLM.

    compile_result, run_result, and perf_pair may be None (first pass, or
    when compile failed and profiling was skipped).
    """
    pid = problem["id"]
    isa_desc = ISA_INSTANCE_DESC.get(isa, isa)
    baseline = baselines.get(pid, {})

    # Baseline performance section
    baseline_section = ""
    if baseline:
        baseline_section = (
            f"\nBaseline performance (target to beat):\n"
            f"  Runtime: {baseline.get('baseline_ms', 'N/A')} ms/iter\n"
        )
        perf = baseline.get("perf", {})
        if perf:
            baseline_section += (
                f"  Cycles: {perf.get('cycles', 'N/A')}\n"
                f"  Instructions: {perf.get('instructions', 'N/A')}\n"
                f"  IPC: {perf.get('ipc', 'N/A')}\n"
                f"  L1D miss rate: {perf.get('l1d_miss_pct', 'N/A')}%\n"
            )

    # Compile status
    if compile_result is None:
        compile_section = "\nCompile status: not yet attempted (first pass)\n"
    elif compile_result.success:
        compile_section = "\nCompile status: SUCCESS\n"
    else:
        errors_preview = (compile_result.errors or "")[:800]
        compile_section = (
            f"\nCompile status: FAILED\n"
            f"Errors:\n{errors_preview}\n"
        )

    # Candidate performance (only available after a successful compile)
    candidate_section = ""
    if run_result is not None:
        c_ms = run_result.candidate_runtime_ms
        b_ms = run_result.baseline_runtime_ms
        correct = run_result.correct
        candidate_section = (
            f"\nCandidate performance:\n"
            f"  Correct: {correct}\n"
            f"  Runtime: {c_ms} ms  (baseline: {b_ms} ms)\n"
        )
        if perf_pair is not None:
            cand_perf, _ = perf_pair
            candidate_section += (
                f"  Cycles: {cand_perf.cycles}\n"
                f"  Instructions: {cand_perf.instructions}\n"
                f"  IPC: {cand_perf.ipc}\n"
                f"  L1D miss rate: {cand_perf.l1d_miss_pct}%\n"
            )

    return (
        f"Problem: {problem['name']}\n"
        f"Description: {problem.get('description', '')}\n"
        f"ISA target: {isa.upper()} on {isa_desc}\n"
        f"Candidate source file: {problem['candidate_source']}\n"
        f"Starter test file: {problem['starter']}\n"
        f"{baseline_section}"
        f"{compile_section}"
        f"{candidate_section}"
        f"\nCurrent candidate source:\n"
        f"```cpp\n{candidate_source}\n```\n"
        f"\nYour task: provide a numbered optimization plan (no code). "
        f"If the code failed to compile, lead with steps to fix the build error."
    )


def build_generator_prompt(plan: str, candidate_source: str) -> str:
    """Build the user message for the generator LLM."""
    return (
        f"Optimization plan:\n{plan}\n\n"
        f"Current candidate source:\n"
        f"```cpp\n{candidate_source}\n```\n\n"
        f"Implement the plan above. Return only a complete C++ source file "
        f"in a single ```cpp code block. No prose."
    )


# ─── LLM utilities ───────────────────────────────────────────────────────────

def _llm_call(messages: list[dict], model: str, temperature: float = 0.3) -> str:
    """Call the LLM (no tools) with retry on rate limit. Returns text content."""
    for retry in range(6):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except litellm.RateLimitError as e:
            wait = 30 * (2 ** retry)
            print(f"  [rate limit] sleeping {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("Exceeded retry budget for rate limit")


def _extract_code_from_response(text: str) -> str | None:
    """
    Extract C++ source from a markdown code block in the LLM response.

    Tries ```cpp first, then plain ```. Returns None if no valid C++ found.
    """
    # Try ```cpp ... ``` first
    m = re.search(r"```cpp\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        # Fallback: any ``` ... ``` block
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return None

    code = m.group(1).strip()
    # Sanity-check: must look like C++
    cpp_markers = ("#include", "namespace", "class ", "void ", "int ", "inline ")
    if not any(marker in code for marker in cpp_markers):
        return None
    return code


# ─── Main evaluation loop ────────────────────────────────────────────────────

def run_nn_eval(
    problem_id: str,
    isa: str,
    model: str,
    handle,
    max_passes: int = 5,
    verbose: bool = True,
) -> EvalResult:
    """
    Run the two-phase LLM optimization workflow for an ncnn kernel problem.

    Each pass: compile → profile → plan (LLM 1) → implement (LLM 2).
    Finalizes with submit() after max_passes.

    Args:
        problem_id: e.g. "conv2d"
        isa: "neon", "sve", or "sve2"
        model: LiteLLM model string, e.g. "anthropic/claude-opus-4-6"
        handle: SSH InstanceHandle or None for local mode
        max_passes: Number of plan-implement iterations
        verbose: Print per-pass status

    Returns:
        EvalResult from submit() or forced final scoring
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    problems = load_starter_problems()
    if problem_id not in problems:
        raise KeyError(f"Problem {problem_id!r} not found")
    problem = problems[problem_id]

    baselines = load_arm_baselines()

    candidate_src_path = REPO_ROOT.parent / "ncnn" / problem["candidate_source"]
    if not candidate_src_path.exists():
        raise FileNotFoundError(f"Candidate source not found: {candidate_src_path}")
    current_code = candidate_src_path.read_text()

    tools = SIMDTools(handle=handle, problem_id=problem_id, isa=isa)
    tools.upload_ncnn_tree()

    starter = problem["starter"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[nn-eval] Problem: {problem_id} ({problem['name']})")
        print(f"ISA: {isa} | Model: {model} | Passes: {max_passes}")
        baseline = baselines.get(problem_id, {})
        if baseline:
            print(f"ARM baseline: {baseline.get('baseline_ms')} ms/iter")
        print(f"{'='*60}")

    # ── Pass loop ─────────────────────────────────────────────────────────────
    for pass_num in range(max_passes):
        if verbose:
            print(f"\n[Pass {pass_num + 1}/{max_passes}]")

        # Step 1: compile
        _write_candidate_source(tools, problem, current_code)
        compile_result = tools.compile(starter)

        if verbose:
            status = "OK" if compile_result.success else "FAILED"
            print(f"  compile: {status}")
            if not compile_result.success:
                print(f"  {compile_result.errors[:200]}")

        # Step 2: profile (only if compile succeeded)
        run_result = None
        perf_pair = None
        if compile_result.success:
            run_result = tools.run(n=3)
            perf_pair = tools.perf(n=1)
            if verbose:
                c_ms = run_result.candidate_runtime_ms
                b_ms = run_result.baseline_runtime_ms
                speedup = round(b_ms / c_ms, 2) if c_ms and b_ms and c_ms > 0 else None
                print(
                    f"  run: correct={run_result.correct}, "
                    f"candidate={c_ms} ms, baseline={b_ms} ms, "
                    f"speedup={speedup}"
                )

        # Step 3: plan
        planner_msgs = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_planner_prompt(
                    problem, current_code, baselines, isa,
                    compile_result, run_result, perf_pair,
                ),
            },
        ]
        plan = _llm_call(planner_msgs, model, temperature=0.3)
        if verbose:
            plan_preview = plan[:300].replace("\n", " ")
            print(f"  plan: {plan_preview}...")

        # Step 4: implement
        impl_msgs = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generator_prompt(plan, current_code),
            },
        ]
        impl_response = _llm_call(impl_msgs, model, temperature=0.2)
        new_code = _extract_code_from_response(impl_response)

        if new_code is not None:
            if verbose:
                print(f"  implement: extracted {len(new_code)} chars of C++")
            current_code = new_code
        else:
            if verbose:
                print("  implement: no valid code extracted — keeping previous code")

    # ── Finalization ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[Finalizing — submitting after {max_passes} passes]")

    _write_candidate_source(tools, problem, current_code)
    submit_result = tools.submit(starter)
    submit_result.tool_calls = tools._tool_calls

    if verbose:
        print(f"\n[Final Result]")
        print(json.dumps(submit_result.to_dict(), indent=2))

    return submit_result
