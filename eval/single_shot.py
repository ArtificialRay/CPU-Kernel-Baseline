"""
eval/single_shot.py — one-shot, no-tools kernel generation.

The complement to eval/evaluator.py::run_agentic_eval. That loop lets a model
compile, measure, disassemble and iterate for up to `max_turns`; this one gives
it a single completion with no tools at all, takes whatever kernel.cpp comes
back, and measures it once. The difference between the two numbers is the value
of the iteration loop itself.

The model is handed exactly the same task information run_agentic_eval's first
turn gets — the entry signature from the definition's header plus the
reference-scalar kernel to replace (build_user_prompt, shared) — so "may I
iterate" is the only variable between the two.

Measurement goes through the same MCP compile/evaluate the tool loop uses, so
the resulting time_speedup_geomean is computed by the same evaluator against
the same baseline and is directly comparable to a multi-turn run's.

`samples` independent generations are drawn per definition at temperature 1.0
(the default of 0 would return near-identical text every time and waste the
repeats). A sample that fails to compile or fails correctness is kept as a
failed sample and NOT redrawn — the pass rate is half the result here, so
silently retrying until something compiles would report the wrong thing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from contracts import AGENT_KERNEL_FILENAME, REFERENCE_SCALAR_AUTHORS
from eval.evaluator import _ISA_PROMPT_INFO, build_user_prompt
from eval.llm_call import call_llm
from eval.llm_providers import resolve_completion_kwargs

if TYPE_CHECKING:
    from eval.mcp_client import MCPKernelClient


SINGLE_SHOT_SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. Your task: write an optimized
{op_type} kernel for {isa_desc}.

You get ONE attempt. There are no tools, no compiler, no profiler and no
second turn — whatever you write is compiled and benchmarked exactly as-is.
Spend your reasoning budget accordingly: think the design through before
you write, because you will not see an error message or a measurement.

Output format — this is parsed mechanically, so it matters:
  - Reply with exactly ONE ```cpp code block containing the complete
    contents of {kernel_filename}.
  - No second code block, and no prose outside it that could be mistaken
    for one.

Key rules:
  - The harness files (the .h and the entry .cpp) are provided automatically
    and must NOT be redefined — write only {kernel_filename}.
  - Define exactly the one function the reference kernel below defines: same
    name, same signature, C linkage. Nothing else — in particular the harness
    already defines the armbench_entry_* symbol, so defining it yourself fails
    to link, even though the header declares it.
  - Use {isa_name} intrinsics freely; the build system passes the correct
    -march flag. Inline asm is allowed if you think it helps.
  - NO threading of any kind: #pragma omp, omp.h, std::thread, pthread.h and
    fork are all rejected by a source scan before compilation. Single core only.
  - Correctness is checked first against a reference implementation; a kernel
    that is fast but wrong scores nothing.

You are measured on wall-time speedup against the {baseline_label}, as a
geometric mean over every workload of this definition.
"""

_CLOSING = (
    "\nWrite the optimized {kernel_filename} now, as a single ```cpp block. "
    "Remember: one attempt, no feedback, no second chance."
)

# ``` fences, optionally tagged cpp/c++/c. Non-greedy body, DOTALL.
_FENCE_RE = re.compile(r"```(?P<tag>[A-Za-z+#]*)\s*\n(?P<body>.*?)```", re.DOTALL)


def extract_kernel_code(text: str) -> Optional[str]:
    """Pull kernel.cpp out of a completion.

    Prefers a cpp/c++/c-tagged fence; falls back to an untagged one. When
    several qualify, the longest wins — a model that narrates with a short
    snippet before the real kernel would otherwise hand us the snippet.
    """
    if not text:
        return None
    blocks = [(m.group("tag").lower(), m.group("body")) for m in _FENCE_RE.finditer(text)]
    if not blocks:
        return None
    tagged = [b for tag, b in blocks if tag in ("cpp", "c++", "cc", "c")]
    pool = tagged or [b for _, b in blocks]
    best = max(pool, key=len).strip()
    return best or None


def run_single_shot(
    definition,
    trace_set,
    author: str,
    model: str,
    mcp_client: "MCPKernelClient",
    isa: str,
    *,
    dataset: str = "ncnn",
    bench_cfg=None,
    samples: int = 3,
    temperature: float = 1.0,
    timeout: float = 900.0,
    verbose: bool = True,
) -> dict:
    """Draw `samples` one-shot kernels for `definition` and measure each once.

    Returns a summary dict: every sample's outcome, plus the best passing
    time_speedup_geomean and how many samples passed.
    """
    baseline_author = bench_cfg.baseline_author if bench_cfg else "reference-scalar"
    ref_author = REFERENCE_SCALAR_AUTHORS.get(dataset, "reference-scalar")
    ref_solution = trace_set.get_baseline_solution(definition.name, ref_author)

    isa_desc, isa_name = _ISA_PROMPT_INFO.get(isa, (isa or "AArch64", "SVE2"))
    baseline_label = {
        "baseline-ncnn-arm":     "hand-optimized ncnn ARM baseline",
        "reference-scalar":      "reference scalar implementation",
        "reference":             "reference scalar implementation",
        "baseline-llamacpp-arm": "llama.cpp (ggml) baseline",
    }.get(baseline_author, baseline_author)

    system = SINGLE_SHOT_SYSTEM_PROMPT.format(
        op_type=definition.op_type,
        isa_desc=isa_desc,
        isa_name=isa_name,
        baseline_label=baseline_label,
        kernel_filename=AGENT_KERNEL_FILENAME,
    )
    user_msg = build_user_prompt(
        definition, ref_solution,
        closing=_CLOSING.format(kernel_filename=AGENT_KERNEL_FILENAME),
    )

    tools = mcp_client.tools_for(definition.name)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: list[dict] = []

    if verbose:
        print(f"\n{'='*60}\nsingle-shot  {definition.name}  ({samples} samples, {model})\n{'='*60}")

    for i in range(samples):
        row: dict = {"sample": i + 1, "status": "GENERATION_ERROR"}
        try:
            completion_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temperature,
                "timeout": timeout,
                **resolve_completion_kwargs(model),
            }
            message = call_llm(completion_kwargs)
            code = extract_kernel_code(getattr(message, "content", "") or "")
        except Exception as e:  # noqa: BLE001 — one bad draw must not kill the batch
            row["error"] = f"{type(e).__name__}: {e}"
            results.append(row)
            if verbose:
                print(f"  sample {i+1}: generation failed — {row['error'][:160]}")
            continue

        if not code:
            row["status"] = "NO_CODE_BLOCK"
            results.append(row)
            if verbose:
                print(f"  sample {i+1}: no ```cpp block in the reply")
            continue

        row["code_chars"] = len(code)
        compiled = tools.dispatch_tool_call(
            "compile", {"definition": definition.name, "code": code}
        )
        if compiled.get("status") != "OK":
            row["status"] = compiled.get("status", "COMPILE_ERROR")
            row["error"] = str(compiled.get("error", ""))[:400]
            results.append(row)
            if verbose:
                print(f"  sample {i+1}: {row['status']} — {row['error'][:160]}")
            continue

        row["version"] = compiled.get("version")
        evaluated = tools.dispatch_tool_call(
            "evaluate", {"definition": definition.name, "version": row["version"]}
        )
        perf = evaluated.get("performance") or {}
        row["status"] = evaluated.get("status", "?")
        row["time_speedup_geomean"] = perf.get("time_speedup_geomean")
        row["cycle_speedup_geomean"] = perf.get("cycle_speedup_geomean")
        if row["status"] != "PASSED":
            row["error"] = str(evaluated.get("log", ""))[:400]
            row["failed_workload"] = evaluated.get("failed_workload")
        results.append(row)
        if verbose:
            ts = row["time_speedup_geomean"]
            ts_s = f" time_speedup={ts:.3f}" if ts is not None else ""
            print(f"  sample {i+1}: {row['status']}{ts_s}")

    passed = [r for r in results if r["status"] == "PASSED" and r.get("time_speedup_geomean")]
    speedups = [r["time_speedup_geomean"] for r in passed]
    summary = {
        "definition": definition.name,
        "dataset": dataset,
        "isa": isa,
        "model": model,
        "author": author,
        "mode": "single-shot",
        "samples": samples,
        "timestamp": run_timestamp,
        "results": results,
        "n_passed": len(passed),
        "pass_rate": len(passed) / samples if samples else 0.0,
        "best_time_speedup": max(speedups) if speedups else None,
        "worst_time_speedup": min(speedups) if speedups else None,
    }
    if verbose:
        best = summary["best_time_speedup"]
        best_s = f"{best:.3f}" if best is not None else "n/a"
        print(f"  -> {len(passed)}/{samples} passed, best time_speedup={best_s}")
    return summary
