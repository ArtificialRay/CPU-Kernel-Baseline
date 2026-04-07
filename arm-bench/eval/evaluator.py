"""
eval/evaluator.py — Agentic LLM evaluation orchestrator for simd-loops.

Runs an agent loop where the LLM iteratively uses compile/run/perf/disassemble
tools over SSH, then scores the final submission against pre-collected baselines.

Compatible with any LiteLLM-supported model.
"""

import copy
import json
import os
import time
from pathlib import Path

import litellm

from eval.config import DATASET_PATH, load_problems, ISA_INSTANCE_DESC
from eval.provision import InstanceHandle, get_or_provision
from eval.tools import SIMDTools, EvalResult

REPO_ROOT = Path(__file__).parent.parent

# System prompt for the LLM agent
SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.

You have access to four tools:
  - compile(code): Inject and compile your C implementation. Check for errors.
  - run(n): Run the compiled binary (n iterations). Check correctness (checksum).
  - perf(n): Collect hardware PMU counters (cycles, IPC, L1D miss rate).
  - disassemble(fn): See the generated assembly for a specific function.
  - submit(code): Submit your final implementation for scoring.

Guidelines:
  - The function signature must be preserved exactly.
  - The `res` field in the data struct is the checksum — it must match the scalar output.
  - Use compile() first, then run() to verify correctness, then optimize.
  - Use disassemble() to inspect generated instructions and verify vectorization.
  - Call submit() when you are satisfied — this triggers final scoring.
  - Be efficient: fewer tool calls is better, but correctness comes first.
"""

# One-shot example shown in the user prompt
ONE_SHOT_EXAMPLE = """\
Example — FP32 SAXPY optimized with SVE:

  // Scalar reference:
  void inner_loop_saxpy(struct saxpy_data *d) {
      for (int i = 0; i < d->n; i++) d->y[i] += d->a * d->x[i];
      float res = 0.f; for (int i = 0; i < d->n; i++) res += d->y[i]; d->res = res;
  }

  // SVE implementation:
  #include <arm_sve.h>
  void inner_loop_saxpy(struct saxpy_data *d) {
      svbool_t pg;
      svfloat32_t va = svdup_f32(d->a);
      int i = 0;
      for (; svptest_first(svptrue_b32(), pg = svwhilelt_b32(i, d->n)); i += svcntw())
          svst1(pg, d->y + i, svmla_m(pg, svld1(pg, d->y + i), va, svld1(pg, d->x + i)));
      svfloat32_t acc = svdup_f32(0.f);
      for (i = 0; svptest_first(svptrue_b32(), pg = svwhilelt_b32(i, d->n)); i += svcntw())
          acc = svadd_m(pg, acc, svld1(pg, d->y + i));
      d->res = svaddv(svptrue_b32(), acc);
  }
"""


def _compress_history(messages: list[dict], keep_full_turns: int = 2) -> list[dict]:
    """
    Compress old turns to keep context size bounded.

    The last `keep_full_turns` complete assistant+tool pairs are kept verbatim.
    Older turns have large payloads replaced with compact summaries:
      - compile/submit code: replaced with placeholder IF the compile succeeded.
        Failed compile code is kept verbatim so the model remembers what to avoid.
      - disassemble asm: always compressed (large, not needed after inspection)
      - run/perf results: already small, always kept verbatim

    Message structure is preserved exactly (tool_call_ids remain valid).
    messages[0] = system, messages[1] = initial user — always kept verbatim.
    """
    # Find indices of all assistant messages (each marks the start of a turn)
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    if len(assistant_indices) <= keep_full_turns:
        return messages  # nothing old enough to compress

    # Build map: tool_call_id → compile success (True/False/None for non-compile)
    compile_success: dict[str, bool] = {}
    for msg in messages:
        if msg["role"] == "tool":
            try:
                content = json.loads(msg["content"])
                if "success" in content:  # compile result
                    compile_success[msg["tool_call_id"]] = content["success"]
            except (json.JSONDecodeError, KeyError):
                pass

    # Everything from this index onward is kept verbatim
    keep_from = assistant_indices[-keep_full_turns]

    result = []
    for i, msg in enumerate(messages):
        if i < keep_from and i >= 2:  # compress; never touch system or initial user
            msg = copy.deepcopy(msg)
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] in ("compile", "submit"):
                        # Only compress code if the compile succeeded — failed
                        # attempts must stay visible so the model doesn't repeat them
                        if compile_success.get(tc["id"], True):
                            try:
                                args = json.loads(tc["function"]["arguments"])
                                code = args.get("code", "")
                                if len(code) > 100:
                                    args["code"] = (
                                        "/* [prior successful attempt: "
                                        f"{len(code)} chars omitted — "
                                        "do not resubmit this placeholder] */"
                                    )
                                    tc["function"]["arguments"] = json.dumps(args)
                            except (json.JSONDecodeError, KeyError):
                                pass
            elif msg["role"] == "tool":
                try:
                    content = json.loads(msg["content"])
                    if "asm" in content and len(content["asm"]) > 100:
                        lines = content["asm"].count("\n")
                        content["asm"] = f"[{lines} lines — omitted from history]"
                        msg["content"] = json.dumps(content)
                except (json.JSONDecodeError, KeyError):
                    pass
        result.append(msg)
    return result


def build_user_prompt(problem: dict, isa: str) -> str:
    """Build the initial user message shown to the LLM."""
    scalar_code = problem.get("scalar_code", "")
    struct_def = problem.get("struct_def", "")
    description = problem.get("description", "")
    isa_desc = ISA_INSTANCE_DESC.get(isa, isa)

    neon_code = problem.get("neon_code", "")
    neon_section = (
        f"\nNEON reference implementation (shows vectorisation structure):\n"
        f"```c\n{neon_code}\n```\n"
        if neon_code else ""
    )

    return f"""\
Problem: {problem["name"]}
Purpose: {description}
ISA target: {isa.upper()} on {isa_desc}

Struct definition (data layout):
```c
{struct_def}
```

Scalar reference implementation (your task: replace with {isa.upper()}):
```c
{scalar_code}
```
{neon_section}
{ONE_SHOT_EXAMPLE}

Write an optimized {isa.upper()} implementation. Start by calling compile() with your first attempt.
"""


def run_agentic_eval(
    problem_id: str,
    isa: str,
    model: str,
    handle: InstanceHandle,
    max_turns: int = 20,
    verbose: bool = True,
) -> EvalResult:
    """
    Run one agentic evaluation session.

    The LLM gets tools (compile, run, perf, disassemble, submit) and iterates
    until it calls submit() or hits max_turns.

    Args:
        problem_id: e.g. "loop_001"
        isa: "neon", "sve", "sve2", or "sme2"
        model: LiteLLM model string, e.g. "anthropic/claude-opus-4-6"
        handle: SSH handle to the provisioned instance
        max_turns: Maximum agent turns before forced submit
        verbose: Print conversation turns

    Returns:
        EvalResult from the submit() call (or a failed result if max_turns hit)
    """
    problems = load_problems()
    if problem_id not in problems:
        raise KeyError(f"Problem {problem_id!r} not found in problems.json")
    problem = problems[problem_id]

    tools = SIMDTools(handle=handle, problem_id=problem_id, isa=isa)
    schemas = SIMDTools.tool_schemas()

    isa_desc = ISA_INSTANCE_DESC.get(isa, isa)
    system = SYSTEM_PROMPT.format(isa_desc=isa_desc)
    user_msg = build_user_prompt(problem, isa)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem_id} | ISA: {isa} | Model: {model}")
        print(f"{'='*60}")

    final_result: EvalResult | None = None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Turn {turn+1}/{max_turns}]")

        compressed = _compress_history(messages)
        for _retry in range(6):
            try:
                response = litellm.completion(
                    model=model,
                    messages=compressed,
                    tools=schemas,
                    tool_choice="auto",
                    temperature=0.2,
                )
                break
            except litellm.RateLimitError as e:
                wait = 30 * (2 ** _retry)
                if verbose:
                    print(f"  [rate limit] sleeping {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError("Exceeded retry budget for rate limit")
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # No tool calls → agent is done (or confused)
        if not msg.tool_calls:
            if verbose:
                print(f"  Agent: {msg.content}")
            break

        # Execute each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                arg_preview = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                               for k, v in fn_args.items()}
                print(f"  → {fn_name}({arg_preview})")

            result_dict = tools.dispatch_tool_call(fn_name, fn_args)

            if verbose:
                if fn_name == "submit":
                    print(f"  ← {result_dict}")
                elif fn_name == "compile":
                    status = "OK" if result_dict.get("success") else "FAILED"
                    print(f"  ← compile: {status}")
                    if not result_dict.get("success"):
                        err_preview = result_dict.get("errors", "")[:200]
                        print(f"     {err_preview}")
                elif fn_name == "run":
                    correct = result_dict.get("correct")
                    ms = result_dict.get("runtime_ms")
                    print(f"  ← run: correct={correct}, {ms}ms")
                elif fn_name == "perf":
                    ipc = result_dict.get("ipc")
                    miss = result_dict.get("l1d_miss_pct")
                    print(f"  ← perf: IPC={ipc}, L1D_miss={miss}%")
                else:
                    print(f"  ← {fn_name}: {str(result_dict)[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict),
            })

            # Capture submit result
            if fn_name == "submit":
                er = EvalResult(**{k: result_dict[k] for k in EvalResult.__dataclass_fields__
                                   if k in result_dict})
                er.tool_calls = tools._tool_calls
                final_result = er

    # If agent never called submit, force a final run with last compiled code
    if final_result is None:
        if verbose:
            print("\n[Max turns reached — forcing final scoring run]")
        # Try a no-op submit with the current compiled state
        rr = tools.run(n=1000)
        from eval.config import load_baselines, ISA_TIER
        tier = ISA_TIER.get(isa, "c7g")
        baselines = load_baselines(tier)
        baseline = baselines.get(problem_id, {})
        scalar_ms = baseline.get("scalar_ms")
        autovec_ms = baseline.get("autovec_ms")
        ref_ms = baseline.get("ref_ms")

        speedup_vs_scalar = None
        speedup_vs_autovec = None
        speedup_vs_ref = None
        level = 0

        if rr.correct:
            level = 1
            if rr.runtime_ms and scalar_ms:
                speedup_vs_scalar = round(scalar_ms / rr.runtime_ms, 2)
                if speedup_vs_scalar > 1.0:
                    level = 2
            if rr.runtime_ms and autovec_ms:
                speedup_vs_autovec = round(autovec_ms / rr.runtime_ms, 2)
                if level >= 2 and speedup_vs_autovec > 1.0:
                    level = 3
            if rr.runtime_ms and ref_ms:
                speedup_vs_ref = round(ref_ms / rr.runtime_ms, 2)
                if level >= 3 and speedup_vs_ref > 1.0:
                    level = 4

        final_result = EvalResult(
            correct=rr.correct,
            speedup_vs_scalar=speedup_vs_scalar,
            speedup_vs_autovec=speedup_vs_autovec,
            speedup_vs_ref=speedup_vs_ref,
            level=level,
            runtime_ms=rr.runtime_ms,
            tool_calls=tools._tool_calls,
        )

    if verbose:
        print(f"\n[Final Result]")
        print(json.dumps(final_result.to_dict(), indent=2))

    return final_result
