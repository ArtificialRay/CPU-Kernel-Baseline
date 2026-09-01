"""
eval/evaluator.py — Agentic LLM evaluation orchestrator for arm-bench.

Runs an agent loop where the LLM iteratively uses compile/evaluate/disassemble
tools against a real Graviton instance over MCP (eval/mcp_client.py — the same
mcp_app/server.py nanobot/Claude Code drive). There is no submit tool —
evaluate() persists the best solution to bench-trace/ automatically.

Compatible with any LiteLLM-supported model.
"""

import copy
import json
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import litellm

from contracts import AGENT_KERNEL_FILENAME, AGENT_LOOP_DEFAULTS, REFERENCE_SCALAR_AUTHORS

if TYPE_CHECKING:
    from eval.mcp_client import MCPKernelClient


AGENT_SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. Your task: write an optimized
{op_type} kernel for {isa_desc}.

Tools: compile, evaluate, disassemble — each takes `definition` (always
"{definition_name}" for this session) and, except compile, `version` (the
number compile() returned for the version you want to act on).

There is no separate submit tool — evaluate() automatically persists your
best result so far to bench-trace whenever it beats your previous best
here. Just keep iterating; nothing is lost even if you never explicitly
finalize anything.

Workflow:
  1. compile() your first attempt.
  2. evaluate()     — checks correctness first (fail-fast); if that passes, also
                       measures timing and cycle speedup in the same call.
  3. disassemble()  — inspect assembly when IPC is low or speedup is unexpectedly poor.
  4. Iterate: compile → evaluate → improve, using the full turn budget to explore.

Metrics from evaluate():
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
"""

# isa string -> (isa_desc, isa_name) shown to the agent in the system prompt.
# Keyed by the SAME `isa` string the caller's mcp_app.server session was
# started with (its --isa flag)
_ISA_PROMPT_INFO: dict[str, tuple[str, str]] = {
    "neon":     ("Arm Neoverse V1 (AWS Graviton3, NEON 128-bit)", "NEON"),
    "sve":      ("Graviton3 with SVE (SVE1, 256-bit)", "SVE"),
    "sve2":     ("Graviton4 with SVE2 (128-bit)", "SVE2"),
    "sme2":     ("Graviton4 with SME2", "SME2"),
    "portable": (
        "AArch64 (portable C++ only — do NOT use NEON or SVE intrinsics; "
        "rely on clean, compiler-vectorizable C++)",
        "portable C++ (no SIMD intrinsics)",
    ),
}


def build_user_prompt(definition, ref_solution) -> str:
    parts = [f"Definition: {definition.name}  (op_type: {definition.op_type})"]

    if ref_solution is not None:
        header = next(
            (s for s in ref_solution.sources if s.path.endswith(".h")), None
        )
        kernel = next(
            (s for s in ref_solution.sources if s.path == AGENT_KERNEL_FILENAME), None
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
    """Compress old turns for the AgentTools loop (compile/evaluate/disassemble)."""
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
                "It's already persisted — keep iterating to try to beat it."
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
                    if tc["function"]["name"] == "compile":
                        if compile_success.get(tc["id"], True):
                            try:  # role=="assistant" + tool_call=="compile" + compile success
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
    mcp_client: "MCPKernelClient",
    isa: str,
    *,
    dataset: str = "ncnn",
    bench_cfg=None,
    max_turns: int = 20,
    verbose: bool = True,
) -> dict:
    """Run one agentic optimization session, tools backed by mcp_app/server.py
    over a shared MCP session (eval/mcp_client.py) — the same server
    nanobot/Claude Code drive.

    Args:
        definition: bench Definition object for the target op.
        trace_set: TraceSet used for solution persistence and baseline lookup.
        author: Solution author label (e.g. "claude-opus-4-8").
        model: LiteLLM model string (e.g. "anthropic/claude-opus-4-8").
        mcp_client: Shared MCP session (eval/mcp_client.py::attach()) —
            one MCPKernelClient is meant to be shared across every
            definition in a run (see its own docstring), so it's passed in
            already connected, not built here.
        isa: The SAME isa string the mcp_app.server session was started
            with (e.g. "sve2", "portable") — drives the system prompt's
            isa_desc/isa_name via `_ISA_PROMPT_INFO` so the agent is never
            told a different ISA than what the server actually compiles
            with. Not derived from instance type.
        dataset: Dataset key, used for REFERENCE_SCALAR_AUTHORS lookup below
            (the MCP server itself already knows its own dataset).
        bench_cfg: Optional BenchmarkConfig override (baselines, perf counter settings).
        max_turns: Maximum agent turns before auto-submit.
        verbose: Print turn-by-turn progress.

    Returns:
        dict with keys: status, time_speedup, cycle_speedup, timestamp, version_history
    """
    tools = mcp_client.tools_for(definition.name)
    schemas = mcp_client.tool_schemas()

    # Optional persistent scratchpad (ARMBENCH_NOTEPAD=1): survives history
    # compression so the agent can track what it has already tried over long runs.
    _use_notepad = os.environ.get("ARMBENCH_NOTEPAD", "").strip() == "1"
    if _use_notepad:
        schemas.append({"type": "function", "function": {
            "name": "notepad",
            "description": ("Append a note to your persistent notepad. Old turns get "
                            "summarized away, but your FULL notepad is shown to you every "
                            "turn. Use it to record which optimizations you tried and their "
                            "measured speedup, what to try next, and your plan."),
            "parameters": {"type": "object",
                           "properties": {"note": {"type": "string", "description": "Note to append."}},
                           "required": ["note"]}}})
    notepad: list[str] = []

    baseline_author = bench_cfg.baseline_author if bench_cfg else "reference-scalar"
    ref_author = REFERENCE_SCALAR_AUTHORS.get(dataset, "reference-scalar")
    ref_solution = trace_set.get_baseline_solution(definition.name, ref_author)

    isa_desc, isa_name = _ISA_PROMPT_INFO.get(isa, (isa or "AArch64", "SVE2"))

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
        definition_name=definition.name,
    )
    user_msg = build_user_prompt(definition, ref_solution)
    if os.environ.get("ARMBENCH_PUSH_ITER", "").strip() == "1":
        user_msg += ("\n\nIMPORTANT: Keep going until you have compiled AND measured "
                     "(evaluate()) at least 5 GENUINELY DIFFERENT implementations "
                     "and can no longer beat your best measured time_speedup. Each new version "
                     "must try a distinct strategy — not a small tweak of the previous one.")

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    final_result: dict | None = None
    version_history: list[dict] = []
    best_version: dict | None = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Definition: {definition.name} | Model: {model}")
        print(f"{'='*60}")

    try:
        for turn in range(max_turns):
            if verbose:
                print(f"\n[Turn {turn+1}/{max_turns}]")

            compressed = _compress_history(messages, version_history=version_history)
            if _use_notepad and notepad:
                compressed = compressed + [{"role": "user", "content":
                    "[Your notepad — persists across turns]\n"
                    + "\n".join(f"- {n}" for n in notepad)}]
            completion_kwargs: dict = {
                "model": model,
                "messages": compressed,
                "tools": schemas,
                "tool_choice": "required",
                # litellm defaults to 600s; large reasoning responses over
                # OpenRouter can exceed that, so give more headroom.
                "timeout": AGENT_LOOP_DEFAULTS["completion_timeout_s"],
            }
            if not any(m in model for m in AGENT_LOOP_DEFAULTS["models_without_temperature"]):
                completion_kwargs["temperature"] = AGENT_LOOP_DEFAULTS["temperature"]

            for _retry in range(AGENT_LOOP_DEFAULTS["retry_max_attempts"]):
                try:
                    response = litellm.completion(**completion_kwargs)
                    break
                except litellm.RateLimitError as e:
                    wait = AGENT_LOOP_DEFAULTS["retry_base_wait_s"] * (2 ** _retry)
                    if verbose:
                        print(f"  [rate limit] sleeping {wait}s: {e}")
                    time.sleep(wait)
                except (litellm.InternalServerError, litellm.APIConnectionError,
                        litellm.ServiceUnavailableError, litellm.Timeout) as e:
                    wait = AGENT_LOOP_DEFAULTS["retry_base_wait_s"] * (2 ** _retry)
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
            dumped = msg.model_dump()
            # Sanitize tool-call arguments to valid JSON before storing in history:
            # cheap/flaky models emit valid-JSON-plus-trailing-junk, and the provider
            # rejects the whole request next turn ("function.arguments must be valid
            # JSON"). Salvage the leading object (or {}), so the round-trip is clean.
            for _tc in (dumped.get("tool_calls") or []):
                _raw = (_tc.get("function") or {}).get("arguments", "")
                try:
                    json.loads(_raw)
                except (json.JSONDecodeError, TypeError):
                    try:
                        _obj, _ = json.JSONDecoder().raw_decode((_raw or "").strip())
                    except json.JSONDecodeError:
                        _obj = {}
                    _tc["function"]["arguments"] = json.dumps(_obj)
            messages.append(dumped)

            if not msg.tool_calls:
                if verbose:
                    print(f"  Agent (no tool call): {msg.content}")
                    print("  [warning] expected a tool call — continuing loop")
                continue

            reasoning_text = msg.content or ""

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                # Cheap/flaky models sometimes emit valid JSON followed by trailing
                # junk ("Extra data") or minor malformation. Salvage the leading
                # object; if totally unparseable, fall back to empty args so
                # dispatch returns a normal error the agent can retry — rather than
                # a bare json.loads raising and killing the entire run.
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    try:
                        fn_args, _ = json.JSONDecoder().raw_decode(tc.function.arguments.strip())
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"  [warn] unparseable tool args for {fn_name}; using empty args")
                        fn_args = {}

                if verbose:
                    arg_preview = {
                        k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                        for k, v in fn_args.items()
                    }
                    print(f"  → {fn_name}({arg_preview})")

                if fn_name == "notepad":
                    notepad.append(str(fn_args.get("note", ""))[:2000])
                    result_dict = {"status": "OK", "notes_saved": len(notepad)}
                else:
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
                                    "correctness": result_dict.get("correctness", {}),
                                }

                reasoning_text = ""  # emit reasoning only on the first tool call per turn

        # ── Report the best version seen this session ────────────────────────────
        if best_version and best_version.get("code"):
            if verbose:
                ts = best_version.get("time_speedup", "?")
                print(f"\n[Result] max turns reached — best was "
                      f"v{best_version['version']} (time_speedup={ts})")
            final_result = {
                "status": "PASSED",
                "time_speedup": best_version.get("time_speedup"),
                "cycle_speedup": best_version.get("cycle_speedup"),
                "correctness": best_version.get("correctness", {}),
                "explanation": f"[best of session: v{best_version['version']}]",
                "timestamp": run_timestamp,
                "version_history": version_history,
                "auto_submitted": True,
            }

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
