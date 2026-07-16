"""
eval/evaluator.py — Agentic LLM evaluation orchestrator for arm-bench.

Runs an agent loop where the LLM iteratively uses compile/evaluate/disassemble/submit
tools over SSH on a Graviton instance, then persists the best solution to bench-trace/.

Compatible with any LiteLLM-supported model.
"""

import copy
import json
import os
import time
from datetime import datetime, timezone

import litellm

from eval.remote import InstanceHandle


AGENT_SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. Your task: write an optimized
{op_type} kernel for {isa_desc}.

Tools:
  compile(code)              — compile kernel.cpp on the remote ARM instance
  evaluate(measure=true)     — run all workloads: correctness + timing
  disassemble(fn=None)       — view AArch64 assembly (up to 300 lines)
  submit(explanation=...)    — finalize and persist the best version from this session

Workflow:
  1. compile() your first attempt.
  2. evaluate(measure=false) — fast correctness check only.
  3. evaluate(measure=true)  — collect timing and cycle speedup.
  4. disassemble()           — inspect assembly when IPC is low or speedup is unexpectedly poor.
  5. Iterate: compile → evaluate → improve.
  6. submit() when satisfied.

Metrics from evaluate(measure=true):
  time_speedup_geomean   — wall-time speedup vs {baseline_label} (geomean across workloads; >1.0 = faster than baseline)
  cycle_speedup_geomean  — cycle count speedup vs {baseline_label} (geomean)
  ipc_mean               — mean IPC across workloads
  cache_misses_mean      — mean LLC misses

Before every tool call, write 3–5 sentences:
  1. Observation: what the last result revealed (speedup numbers, asm pattern, error).
  2. Hypothesis: the specific bottleneck or opportunity you are targeting.
  3. Change: for compile(), exactly what you changed and why it should help.
     e.g. "Switching to 8 accumulators because FMA latency is 4 cycles and IPC=1.3 suggests
     the pipeline stalls waiting for accumulator writeback."

Key rules:
  - The harness files (.h and the entry .cpp) are provided automatically — write only kernel.cpp.
  - Use {isa_name} intrinsics freely; the build system passes the correct -march flag.
  - Can write asm directly to your implementation if you think it may bring performance gain
  - Always verify correctness before profiling: evaluate(measure=false) first.
  - Do NOT submit without at least one evaluate(measure=true) showing a speedup.
"""

_AGENT_ISA_LABELS: dict[str, str] = {
    "c7g": "Graviton3 (SVE, 256-bit vector length)",
    "c8g": "Graviton4 (SVE2, 128-bit vector length)",
}

# Maps EC2 family → the ISA intrinsic family the agent should use.
_AGENT_ISA_NAMES: dict[str, str] = {
    "c7g": "SVE",
    "c8g": "SVE2",
}


def build_user_prompt(definition, ref_solution) -> str:
    parts = [f"Definition: {definition.name}  (op_type: {definition.op_type})"]

    if ref_solution is not None:
        header = next(
            (s for s in ref_solution.sources if s.path.endswith(".h")), None
        )
        kernel = next(
            (s for s in ref_solution.sources if s.path == "kernel.cpp"), None
        )
        if header:
            parts.append(
                f"\nHeader (shows the function signature you must implement):\n"
                f"```cpp\n{header.content}\n```"
            )
        if kernel:
            parts.append(
                f"\nReference scalar kernel (your task: replace with optimized implementation):\n"
                f"```cpp\n{kernel.content}\n```"
            )

    parts.append(
        "\nStart with compile(). Use evaluate(measure=false) to check correctness, "
        "then evaluate(measure=true) for speedup metrics."
    )
    return "\n".join(parts)


def _compress_history(
    messages: list[dict],
    keep_full_turns: int = 2,
    version_history: list[dict] | None = None,
) -> list[dict]:
    """Compress old turns for the AgentTools loop (compile/evaluate/disassemble/submit)."""
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(assistant_indices) <= keep_full_turns:
        return messages

    # Track which tool-call IDs correspond to successful compiles
    compile_success: dict[str, bool] = {}
    for msg in messages:
        if msg["role"] == "tool":
            try:
                content = json.loads(msg["content"])
                compile_success[msg["tool_call_id"]] = content.get("status") == "OK"
            except (json.JSONDecodeError, KeyError):
                pass

    keep_from = assistant_indices[-keep_full_turns]
    recap_parts = ["[History compressed — earlier turns summarized below.]"]
    # version_history: info for kernels at each turn, it would be like:
    #     [History compressed — earlier turns summarized below.]
    # Versions that passed correctness checks:
    #   v1 [turn 2]: time_speedup=1.234, cycle_speedup=1.189
    #   v3 [turn 5]: time_speedup=1.891, cycle_speedup=1.763  ← BEST
    # Best so far: v3 (time_speedup=1.891). Submit if you can't improve further.
    # The most recently compiled binary is still active on the remote — ...
    if version_history:
        passed = [v for v in version_history if v.get("passed")]
        if passed:
            best = max(passed, key=lambda v: v.get("time_speedup") or 0.0)
            recap_parts.append("Versions that passed correctness checks:")
            for v in passed:
                ts = v.get("time_speedup")
                cs = v.get("cycle_speedup")
                ts_str = f"time_speedup={ts:.3f}" if ts is not None else "correctness only"
                cs_str = f", cycle_speedup={cs:.3f}" if cs is not None else ""
                best_marker = " ← BEST" if v is best else ""
                recap_parts.append(
                    f"  v{v['version']} [turn {v['turn']}]: {ts_str}{cs_str}{best_marker}"
                )
            best_ts = best.get("time_speedup")
            best_ts_str = f"{best_ts:.3f}" if best_ts is not None else "?"
            recap_parts.append(
                f"Best so far: v{best['version']} "
                f"(time_speedup={best_ts_str}). "
                "Submit if you can't improve further."
            )
        else:
            recap_parts.append(
                f"{len(version_history)} compile attempt(s) — none passed correctness yet."
            )

    recap_parts.append(
        "The most recently compiled binary is still active on the remote — "
        "call evaluate() to test it, or compile() a new version."
    )
    recap_msg = {"role": "user", "content": "\n".join(recap_parts)}

    # messages: complete chat history at each runs
    result = []
    recap_inserted = False
    for i, msg in enumerate(messages):
        if i == keep_from and not recap_inserted:
            result.append(recap_msg)
            recap_inserted = True
        if i < keep_from and i >= 2:
            msg = copy.deepcopy(msg)
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] in ("compile", "submit"):
                        if compile_success.get(tc["id"], True):
                            try:  # role=="assistant" + tool_call=="compile" or "submit" + compile success
                                args = json.loads(tc["function"]["arguments"])
                                code = args.get("code", "")
                                if len(code) > 100:
                                    args["code"] = (
                                        f"/* [prior version: {len(code)} chars omitted] */"
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


def run_agentic_eval(
    definition,
    trace_set,
    author: str,
    model: str,
    handle: InstanceHandle,
    *,
    dataset: str = "ncnn",
    bench_cfg=None,
    max_turns: int = 20,
    verbose: bool = True,
) -> dict:
    """Run one agentic optimization session using the AgentTools ecosystem.

    The LLM receives compile/evaluate/disassemble/submit tools backed by a real
    Graviton instance via SSH. Iterates until the agent calls submit() or max_turns
    is reached (triggering auto-submit of the best correct version found).

    Args:
        definition: bench Definition object for the target op.
        trace_set: TraceSet used for solution persistence and baseline lookup.
        author: Solution author label (e.g. "claude-opus-4-8").
        model: LiteLLM model string (e.g. "anthropic/claude-opus-4-8").
        handle: SSH handle to the provisioned Graviton instance.
        dataset: Dataset key for resolve_tools() dispatch (default "ncnn").
        bench_cfg: Optional BenchmarkConfig override (baselines, perf counter settings).
        max_turns: Maximum agent turns before auto-submit.
        verbose: Print turn-by-turn progress.

    Returns:
        dict with keys: status, time_speedup, cycle_speedup, timestamp, version_history
    """
    from eval.agent_tools import resolve_tools

    ToolsCls = resolve_tools(dataset)
    tools = ToolsCls(handle, definition, trace_set, author, bench_cfg=bench_cfg)
    schemas = [{"type": "function", "function": s} for s in ToolsCls.tool_schemas()]

    baseline_author = bench_cfg.baseline_author if bench_cfg else "reference-scalar"
    ref_author = "reference-scalar" if dataset == "ncnn" or "llama.cpp" else baseline_author # TODO: use reference-scalar solution for all other dataset, simd-loop has no reference-scalar currently
    # Starter code shown to the agent = the baseline solution for this dataset
    # (author varies: reference-scalar/reference/baseline-llamacpp-arm).
    ref_solution = trace_set.get_baseline_solution(definition.name, ref_author)

    family = handle.instance_type.split(".")[0] if handle.instance_type else ""
    isa_desc = _AGENT_ISA_LABELS.get(family, handle.instance_type or "AArch64")
    isa_name = _AGENT_ISA_NAMES.get(family, "SVE2")

    # ISA-ablation override (env ARMBENCH_ISA_MODE): must match agent_tools.base's
    # _ISA_MODES so the prompt's allowed-ISA matches the actual compile flags.
    # "portable" = no hand-written SIMD (compiler auto-vec still allowed).
    _ISA_MODE_PROMPT = {
        "portable": ("AArch64 (portable C++ only — do NOT use NEON or SVE "
                     "intrinsics; rely on clean, compiler-vectorizable C++)",
                     "portable C++ (no SIMD intrinsics)"),
        "sve":      ("Graviton3 with SVE (SVE1, 128-bit) — use SVE intrinsics "
                     "only, NOT SVE2", "SVE"),
        "sve2":     ("Graviton4 with SVE2 (128-bit)", "SVE2"),
    }
    _mode = os.environ.get("ARMBENCH_ISA_MODE", "").strip().lower()
    if _mode in _ISA_MODE_PROMPT:
        isa_desc, isa_name = _ISA_MODE_PROMPT[_mode]

    _BASELINE_LABELS = {
        "baseline-ncnn-arm":     "hand-optimized ncnn ARM baseline",
        "reference-scalar":      "reference scalar implementation",
        "reference":             "reference scalar implementation",
        "baseline-llamacpp-arm": "llama.cpp (ggml) baseline",
    }
    baseline_label = _BASELINE_LABELS.get(baseline_author, baseline_author)

    system = AGENT_SYSTEM_PROMPT.format(
        op_type=definition.op_type,
        isa_desc=isa_desc,
        isa_name=isa_name,
        baseline_label=baseline_label,
    )
    user_msg = build_user_prompt(definition, ref_solution)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    final_result: dict | None = None
    version_history: list[dict] = []
    best_version: dict | None = None
    agent_submitted_code: str | None = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Definition: {definition.name} | Model: {model}")
        print(f"{'='*60}")

    try:
        for turn in range(max_turns):
            if verbose:
                print(f"\n[Turn {turn+1}/{max_turns}]")

            turns_left = max_turns - turn
            if turns_left == 3 and any(v.get("passed") for v in version_history):
                messages.append({
                    "role": "user",
                    "content": (
                        f"[{turns_left} turns remaining] You have a correct implementation. "
                        "Call submit() now — do not spend more turns optimizing."
                    ),
                })

            compressed = _compress_history(messages, version_history=version_history)
            completion_kwargs: dict = {
                "model": model,
                "messages": compressed,
                "tools": schemas,
                "tool_choice": "required",
            }
            if "opus-4-7" not in model and "opus-4-8" not in model:
                completion_kwargs["temperature"] = 0.2

            for _retry in range(6):
                try:
                    response = litellm.completion(**completion_kwargs)
                    break
                except litellm.RateLimitError as e:
                    wait = 30 * (2 ** _retry)
                    if verbose:
                        print(f"  [rate limit] sleeping {wait}s: {e}")
                    time.sleep(wait)
                except (litellm.InternalServerError, litellm.APIConnectionError,
                        litellm.ServiceUnavailableError) as e:
                    wait = 30 * (2 ** _retry)
                    if verbose:
                        print(f"  [server error] sleeping {wait}s: {type(e).__name__}: {e}")
                    time.sleep(wait)
                except litellm.BadRequestError as e:
                    if "temperature" in completion_kwargs and "temperature" in str(e).lower():
                        if verbose:
                            print(f"  [retry] dropping temperature: {e}")
                        completion_kwargs.pop("temperature")
                        continue
                    raise
            else:
                raise RuntimeError("Exceeded retry budget for rate/server errors")

            msg = response.choices[0].message
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                if verbose:
                    print(f"  Agent (no tool call): {msg.content}")
                    print("  [warning] expected a tool call — continuing loop")
                continue

            reasoning_text = msg.content or ""

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if verbose:
                    arg_preview = {
                        k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                        for k, v in fn_args.items()
                    }
                    print(f"  → {fn_name}({arg_preview})")

                result_dict = tools.dispatch_tool_call(fn_name, fn_args)

                if verbose:
                    if fn_name == "compile":
                        status = result_dict.get("status", "?")
                        print(f"  ← compile: {status}")
                        if status != "OK":
                            print(f"     {str(result_dict.get('error', ''))[:300]}")
                    elif fn_name == "evaluate":
                        status = result_dict.get("status", "?")
                        perf = result_dict.get("performance", {})
                        correctness = result_dict.get("correctness", {})
                        ts = perf.get("time_speedup_geomean")
                        cs = perf.get("cycle_speedup_geomean")
                        mae = correctness.get("max_absolute_error")
                        mre = correctness.get("max_relative_error")
                        perf_str = (
                            f", time_speedup={ts:.3f}, cycle_speedup={cs:.3f}"
                            if ts is not None else ""
                        )
                        correct_str = (
                            f", max_absolute_error={mae:.2e}, max_relative_error={mre:.2e}"
                            if mae is not None else ""
                        )
                        print(f"  ← evaluate: {status}{perf_str}{correct_str}")
                        if status != "PASSED":
                            wl = result_dict.get("failed_workload", "")
                            log = str(result_dict.get("log", ""))[:200]
                            print(f"     failed_workload={wl}  {log}")
                    elif fn_name == "disassemble":
                        lines = result_dict.get("asm", "").count("\n")
                        print(f"  ← disassemble: {lines} lines")
                    elif fn_name == "submit":
                        print(f"  ← submit: {result_dict}")
                    else:
                        print(f"  ← {fn_name}: {str(result_dict)[:100]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result_dict),
                })

                # ── Version tracking ──────────────────────────────────────────
                if fn_name == "compile" and result_dict.get("status") == "OK":
                    version_history.append({
                        "version": result_dict.get("version", len(version_history) + 1),
                        "turn": turn + 1,
                        "code": fn_args.get("code", ""),
                        "reasoning": reasoning_text,
                        "passed": False,
                        "time_speedup": None,
                        "cycle_speedup": None,
                    })

                elif fn_name == "evaluate" and version_history:
                    if result_dict.get("status") == "PASSED":
                        perf = result_dict.get("performance", {})
                        ts = perf.get("time_speedup_geomean")
                        cs = perf.get("cycle_speedup_geomean")
                        version_history[-1]["passed"] = True
                        if ts is not None:
                            version_history[-1]["time_speedup"] = ts
                            version_history[-1]["cycle_speedup"] = cs
                            if best_version is None or ts > (best_version.get("time_speedup") or 0.0):
                                best_version = {
                                    "version": version_history[-1]["version"],
                                    "code": version_history[-1]["code"],
                                    "time_speedup": ts,
                                    "cycle_speedup": cs,
                                }

                # ── Capture submit ────────────────────────────────────────────
                if fn_name == "submit" and result_dict.get("status") == "PASSED":
                    agent_submitted_code = fn_args.get("code", "")
                    final_result = {
                        **result_dict,
                        "timestamp": run_timestamp,
                        "version_history": version_history,
                    }

                reasoning_text = ""  # emit reasoning only on the first tool call per turn

            if final_result is not None:
                break

        # ── Auto-submit if agent never produced a successful submit ────────────
        if final_result is None and best_version and best_version.get("code"):
            reason = "max turns reached" if not agent_submitted_code else "submit did not pass"
            if verbose:
                ts = best_version.get("time_speedup", "?")
                print(f"\n[Auto-submit] {reason} — submitting "
                      f"v{best_version['version']} (time_speedup={ts})...")
            try:
                result = tools.submit(
                    explanation=(
                        f"[auto-submitted: v{best_version['version']} had best "
                        f"time_speedup={best_version.get('time_speedup', '?')}]"
                    ),
                    source_version=best_version["version"],
                )
                if result.get("status") == "PASSED":
                    final_result = {
                        **result,
                        "timestamp": run_timestamp,
                        "version_history": version_history,
                        "auto_submitted": True,
                    }
            except Exception as e:
                if verbose:
                    print(f"  [auto-submit failed: {e}]")

        if final_result is None:
            if verbose:
                print("\n[Max turns reached with no passing version — recording failure]")
            final_result = {
                "status": "NO_SUBMIT",
                "timestamp": run_timestamp,
                "version_history": version_history,
            }

    finally:
        tools.cleanup()

    if verbose:
        summary = {k: v for k, v in final_result.items() if k != "version_history"}
        print(f"\n[Final Result]\n{json.dumps(summary, indent=2)}")

    return final_result
