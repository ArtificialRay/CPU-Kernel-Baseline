"""
eval/evaluator.py — Agentic LLM evaluation orchestrator for arm-bench.

Runs an agent loop where the LLM iteratively uses compile/run/perf/disassemble
tools over SSH, then scores the final submission against pre-collected baselines.

Compatible with any LiteLLM-supported model.
"""

import copy
import json
import os
import time
from datetime import datetime, timezone
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

You have access to these tools:
  - compile(code): Inject and compile your C implementation. Check for errors.
  - run(n): Run the compiled binary (n iterations). Verifies correctness via checksum.
  - perf(n, size): Collect hardware PMU counters: cycles, IPC, cache_misses_per_iter, task_clock_ms.
  - disassemble(fn): See the generated AArch64 assembly for a specific function.
  - submit(code, explanation): Submit your final implementation for scoring.

Workflow — follow this order:
  1. compile() your first attempt.
  2. run() to verify correctness. Fix any checksum failures before continuing.
  3. perf() after every correct implementation — use a large size (e.g. size=500000) to ensure
     data spills out of cache and you are measuring real memory bandwidth, not cache hits.
     - IPC < 1.5 on a SIMD kernel usually means poor vectorization or memory bottleneck.
     - cache_misses_per_iter > 1000 at large size means significant LLC misses — bandwidth-bound.
     - task_clock_ms is on-CPU time per iteration — use it to compare implementations.
  4. disassemble() if you want to confirm which SVE instructions were generated.
  5. Iterate: if IPC is low or task_clock_ms is high, try a different approach and perf() again.
  6. submit() once you have a correct implementation you are happy with.

Key rules:
  - The function signature must be preserved exactly.
  - The `res` field in the data struct is the checksum — it must match the scalar output.
  - Always call perf() after confirming correctness — do not submit without profiling.
  - Compare task_clock_ms across attempts to pick the fastest correct version.
  - Do NOT invent preprocessor macros or pragmas for feature detection (e.g. no
    #pragma GCC target(...) or #ifdef __ARM_FEATURE_SVE). Assume the build system
    provides all required compiler flags — just include <arm_sve.h> and use the
    intrinsics directly.
  - When submitting, include an explanation of which SVE instructions you used,
    why you chose them, and what perf() results you observed.

Reasoning requirement — BEFORE every tool call, write a short paragraph covering:
  1. Observation: what the last result told you (perf numbers, correctness, asm pattern).
  2. Hypothesis: what specific bottleneck or opportunity you are targeting next.
  3. Change: for compile(), describe the exact code change you are making and why it should help.
     e.g. "Switching from 4 to 8 accumulators because FMA latency is 4 cycles and IPC=1.26
     suggests the pipeline is stalled waiting for accumulator writeback."
  Keep it to 3-5 sentences — be precise, not verbose. This reasoning is recorded as part of
  the benchmark audit trail so clarity matters.
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


def _compress_history(
    messages: list[dict],
    keep_full_turns: int = 2,
    code_versions: list[dict] | None = None,
) -> list[dict]:
    """
    Compress old turns to keep context size bounded.

    The last `keep_full_turns` complete assistant+tool pairs are kept verbatim.
    Older turns have large payloads replaced with compact summaries:
      - compile/submit code: replaced with placeholder IF the compile succeeded.
        Failed compile code is kept verbatim so the model remembers what to avoid.
      - disassemble asm: always compressed (large, not needed after inspection)
      - run/perf results: already small, always kept verbatim

    code_versions: if provided, a version history table is injected into the recap
    so the agent can see all attempts and their metrics at a glance.

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

    recap_parts = ["[History compressed — earlier turns omitted to save context.]"]

    # Version history table — richer than the old last_correct_run/last_perf recap
    if code_versions:
        correct_versions = [v for v in code_versions if v.get("correct")]
        if correct_versions:
            best = min(correct_versions, key=lambda v: v.get("ms_per_iter") or float("inf"))
            recap_parts.append("All correct implementations found so far:")
            for v in correct_versions:
                ms = v.get("ms_per_iter")
                ms_str = f"{ms:.4f}ms/iter" if ms is not None else "timing unknown"
                perf = v.get("perf") or {}
                ipc_str = f", IPC={perf['ipc']}" if perf.get("ipc") else ""
                tc_str = f", task_clock={perf['task_clock_ms']}ms" if perf.get("task_clock_ms") else ""
                best_marker = " ← BEST" if v is best else ""
                recap_parts.append(
                    f"  v{v['version']} [turn {v['turn']}]: {ms_str}{ipc_str}{tc_str}{best_marker}"
                )
            recap_parts.append(
                f"Best so far: v{best['version']} ({best.get('ms_per_iter', '?'):.4f}ms/iter). "
                "Try to beat it, or submit it if you can't improve further."
            )
        elif code_versions:
            # Compiles attempted but none correct yet
            recap_parts.append(
                f"{len(code_versions)} compile attempt(s) so far — none produced a correct result yet."
            )
    else:
        # Fallback to old-style recap if no version tracking available
        last_correct_run: dict | None = None
        last_perf: dict | None = None
        for msg in messages[:keep_from]:
            if msg["role"] == "tool":
                try:
                    content = json.loads(msg["content"])
                    if "correct" in content and content.get("correct"):
                        last_correct_run = content
                    if "ipc" in content:
                        last_perf = content
                except (json.JSONDecodeError, KeyError):
                    pass
        if last_correct_run:
            recap_parts.append(
                f"Your last correct run: runtime_ms={last_correct_run.get('runtime_ms')}."
            )
        if last_perf:
            recap_parts.append(
                f"Your last perf result: IPC={last_perf.get('ipc')}, "
                f"task_clock_ms={last_perf.get('task_clock_ms')}, "
                f"cache_misses_per_iter={last_perf.get('cache_misses_per_iter')}."
            )

    recap_parts.append(
        "The most recently compiled binary is still active on the remote — "
        "call run() or perf() to test it, or compile() a new version to improve."
    )
    recap_msg = {"role": "user", "content": "\n".join(recap_parts)}

    result = []
    recap_inserted = False
    for i, msg in enumerate(messages):
        # Insert the recap just before the verbatim section starts
        if i == keep_from and not recap_inserted:
            result.append(recap_msg)
            recap_inserted = True
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
    tools_class=None,
    problem_override: dict | None = None,
    system_prompt_override: str | None = None,
    user_prompt_builder=None,
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
        tools_class: Tools class to use (default: SIMDTools). Must implement the
                     same interface (compile/run/perf/disassemble/submit/tool_schemas).
        problem_override: If provided, use this problem dict instead of loading
                          from problems.json. Useful for ncnn and other non-loop modes.
        system_prompt_override: If provided, use this system prompt template
                                instead of SYSTEM_PROMPT. Must accept {isa_desc}.
        user_prompt_builder: If provided, use fn(problem, isa) -> str to build
                             the initial user message instead of build_user_prompt.

    Returns:
        EvalResult from the submit() call (or a failed result if max_turns hit)
    """
    _ToolsClass = tools_class or SIMDTools

    if problem_override is not None:
        problem = problem_override
    else:
        problems = load_problems()
        if problem_id not in problems:
            raise KeyError(f"Problem {problem_id!r} not found in problems.json")
        problem = problems[problem_id]

    tools = _ToolsClass(handle=handle, problem_id=problem_id, isa=isa)
    schemas = _ToolsClass.tool_schemas()

    isa_desc = ISA_INSTANCE_DESC.get(isa, isa)
    sys_prompt_template = system_prompt_override or SYSTEM_PROMPT
    system = sys_prompt_template.format(isa_desc=isa_desc)
    _build_prompt = user_prompt_builder or build_user_prompt
    user_msg = _build_prompt(problem, isa)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem_id} | ISA: {isa} | Model: {model}")
        print(f"{'='*60}")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    final_result: EvalResult | None = None
    trace: list[dict] = []       # per-turn record of reasoning + tool usage
    code_versions: list[dict] = []  # every attempted version: code + metrics
    best_correct: dict | None = None  # {version, code, ms_per_iter} — fastest correct seen
    agent_submitted_code: str | None = None  # code the agent explicitly submitted

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Turn {turn+1}/{max_turns}]")

        # Warn agent when turns are running low so it submits rather than iterating
        turns_left = max_turns - turn
        if turns_left == 3 and any(m["role"] == "tool" and
                                    json.loads(m.get("content", "{}")).get("correct")
                                    for m in messages):
            messages.append({
                "role": "user",
                "content": (
                    f"[{turns_left} turns remaining] You have a correct implementation compiled. "
                    "Call submit() now with your best code and an explanation — "
                    "do not spend remaining turns on further optimisation."
                ),
            })

        compressed = _compress_history(messages, code_versions=code_versions)
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

        # Capture the agent's reasoning text (content before the tool calls)
        reasoning_text = msg.content or ""

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
                    expl = result_dict.get("explanation", "")
                    if expl:
                        print(f"  ← explanation: {expl[:200]}")
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
                    miss = result_dict.get("cache_misses_per_iter")
                    task_ms = result_dict.get("task_clock_ms")
                    print(f"  ← perf: IPC={ipc}, LLC_misses/iter={miss}, task_clock={task_ms}ms/iter")
                else:
                    print(f"  ← {fn_name}: {str(result_dict)[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict),
            })

            # Build a compact result summary (omit large fields like raw_output/asm)
            result_summary = {k: v for k, v in result_dict.items()
                              if k not in ("raw_output", "asm", "warnings", "output", "trace")}

            # Build a compact args summary (truncate large code strings)
            args_summary = {
                k: (v[:300] + f"... [{len(v)} chars]" if isinstance(v, str) and len(v) > 300 else v)
                for k, v in fn_args.items()
            }

            # For compile calls, compute a line-level diff against the previous version
            code_diff = None
            if fn_name == "compile":
                new_code = fn_args.get("code", "")
                prev_code = (code_versions[-1]["code"] if code_versions else None)
                if prev_code and prev_code != new_code:
                    import difflib
                    diff_lines = list(difflib.unified_diff(
                        prev_code.splitlines(),
                        new_code.splitlines(),
                        lineterm="",
                        n=2,  # 2 lines of context
                    ))
                    code_diff = "\n".join(diff_lines[:80])  # cap at 80 diff lines
                    if len(diff_lines) > 80:
                        code_diff += f"\n... ({len(diff_lines) - 80} more diff lines)"

            trace.append({
                "turn": turn + 1,
                "tool": fn_name,
                "reasoning": reasoning_text,
                "code_diff": code_diff,
                "args": args_summary,
                "result": result_summary,
            })
            # Only emit reasoning once per turn (may have multiple tool calls)
            reasoning_text = ""

            # ── Version tracking ─────────────────────────────────────────────
            if fn_name == "compile" and result_dict.get("success"):
                code_versions.append({
                    "version": len(code_versions) + 1,
                    "turn": turn + 1,
                    "code": fn_args.get("code", ""),
                    "correct": False,
                    "ms_per_iter": None,
                    "perf": None,
                })
            elif fn_name == "run" and result_dict.get("correct") and code_versions:
                n = fn_args.get("n", 100)
                total_ms = result_dict.get("runtime_ms")
                ms_per_iter = round(total_ms / n, 6) if total_ms and n else None
                code_versions[-1]["correct"] = True
                code_versions[-1]["ms_per_iter"] = ms_per_iter
                # Update best_correct if this is the fastest correct version seen
                if ms_per_iter is not None:
                    if best_correct is None or ms_per_iter < best_correct["ms_per_iter"]:
                        best_correct = {
                            "version": code_versions[-1]["version"],
                            "code": tools._last_candidate_code or fn_args.get("code", ""),
                            "ms_per_iter": ms_per_iter,
                        }
            elif fn_name == "perf" and code_versions:
                code_versions[-1]["perf"] = {
                    "ipc": result_dict.get("ipc"),
                    "task_clock_ms": result_dict.get("task_clock_ms"),
                    "cache_misses_per_iter": result_dict.get("cache_misses_per_iter"),
                }

            # ── Capture submit result ─────────────────────────────────────────
            if fn_name == "submit":
                agent_submitted_code = fn_args.get("code", "")
                er = EvalResult(**{k: result_dict[k] for k in EvalResult.__dataclass_fields__
                                   if k in result_dict})
                er.tool_calls = tools._tool_calls
                if not er.compile_error:  # ignore failed submits (e.g. placeholder)
                    final_result = er

    # If agent never submitted successfully, force a final run with last compiled code
    if final_result is None:
        if verbose:
            print("\n[Max turns reached — forcing final scoring run]")
        # Use a tools-class-specific iteration count (NCNNTools uses fewer)
        _autofail_n = getattr(tools, "_autofail_n", 1000)
        rr = tools.run(n=_autofail_n)
        # Use tools-class-specific baseline loader if available (e.g. NCNNTools)
        if hasattr(tools, "_load_baseline_for_problem"):
            baseline = tools._load_baseline_for_problem()
        else:
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

    # ── Auto-resubmit best version if it differs from what the agent submitted ──
    # The agent may have submitted a later (worse) version, or may not have submitted
    # at all. Re-score the best correct version seen and use it if faster.
    if best_correct and best_correct.get("code") and best_correct["code"] != agent_submitted_code:
        if verbose:
            print(f"\n[Auto-submit] Re-scoring best version "
                  f"(v{best_correct['version']}, {best_correct['ms_per_iter']:.4f}ms/iter during session)...")
        try:
            better = tools.submit(
                best_correct["code"],
                explanation=(
                    f"[auto-submitted: v{best_correct['version']} had best runtime "
                    f"({best_correct['ms_per_iter']:.4f}ms/iter) during session]"
                ),
            )
            better.tool_calls = tools._tool_calls
            if better.correct and (
                final_result is None
                or final_result.runtime_ms is None
                or (better.runtime_ms and better.runtime_ms < final_result.runtime_ms)
            ):
                if verbose:
                    print(f"  → v{best_correct['version']} scores better "
                          f"({better.runtime_ms}ms vs {final_result.runtime_ms if final_result else '?'}ms) — using it")
                final_result = better
            else:
                if verbose:
                    print(f"  → Agent's submission was already optimal — keeping it")
        except Exception as e:
            if verbose:
                print(f"  [auto-submit failed: {e}]")

    final_result.timestamp = run_timestamp
    final_result.trace = trace
    final_result.code_versions = code_versions

    if verbose:
        print(f"\n[Final Result]")
        print(json.dumps(final_result.to_dict(), indent=2))

    return final_result
