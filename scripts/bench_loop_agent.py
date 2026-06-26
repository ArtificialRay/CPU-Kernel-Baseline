#!/usr/bin/env python3
"""Local agentic loop: an LLM iterates on a SIMD-loop kernel, scored by the
**same** evaluation path as the rest of the benchmark.

Unlike the SSH eval (eval/run_benchmark.py), this runs entirely in-process: the
LLM generates a kernel, and bench compiles + scores it. Crucially, scoring goes
through `bench.runner.run_solution_on_workloads` → `DefaultEvaluator` — the exact
pipeline behind `bench.cli bench` and the `reference`/`autovec` baselines. So the
agent measures correctness (hybrid abs+rel tolerance, `bench/runtime/correctness`)
and timing (CPU-pinned, perf counters, `bench/runtime/timing`) identically to
everything else; it does not reimplement any of it.

Because the evaluator is dataset-driven, the agent now works for *every* loop
shape (scalar-output, array-output, in-place sort) — any loop_id that has a
`reference` + `autovec` solution in the warehouse.

Usage (run ON the target machine — Graviton4 for real SVE2 timing):
    OPENROUTER_API_KEY=sk-or-... python3 scripts/bench_loop_agent.py --loop loop_001
    python3 scripts/bench_loop_agent.py --loops loop_001,loop_028 --max-turns 6
    python3 scripts/bench_loop_agent.py --all-loops --self-test   # no LLM/key needed
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.config import BenchmarkConfig, EvalConfig
from bench.data.solution import Solution, SourceFile, SupportedDatasets
from bench.data.trace import EvaluationStatus
from bench.data.trace_set import TraceSet
from bench.runner import run_solution_on_workloads

BENCH_TRACE = ROOT / "bench-trace"


# ── Warehouse helpers ─────────────────────────────────────────────────────────

def _solution_by_author(ts: TraceSet, loop_id: str, author: str) -> Optional[Solution]:
    for s in ts.solutions.get(loop_id, []):
        if s.author == author:
            return s
    return None


def _available_loops(ts: TraceSet) -> List[str]:
    """Loop ids that have both a `reference` (for the prompt) and an `autovec`
    (build template) solution — i.e. everything the agent can target."""
    out = []
    for loop_id in sorted(ts.definitions):
        if _solution_by_author(ts, loop_id, "reference") and _solution_by_author(ts, loop_id, "autovec"):
            out.append(loop_id)
    return out


def _extract_braced_block(text: str, start_idx: int) -> Optional[str]:
    """Return text[start_idx:] up to and including the brace-balanced block."""
    open_brace = text.find("{", start_idx)
    if open_brace == -1:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                if end < len(text) and text[end] == ";":
                    end += 1
                return text[start_idx:end]
    return None


def _loop_context(ts: TraceSet, loop_id: str) -> Optional[dict]:
    """Prompt material for `loop_id`, extracted from its `reference` solution:
    the struct (from `loop_NNN.h`) and the scalar baseline (from `kernel.cpp`)."""
    ref = _solution_by_author(ts, loop_id, "reference")
    if ref is None:
        return None
    srcs = {s.path: s.content for s in ref.sources}
    hdr = srcs.get(f"{loop_id}.h", "")
    kern = srcs.get("kernel.cpp", "")

    si = hdr.find(f"struct {loop_id}_data")
    struct_def = _extract_braced_block(hdr, si) if si != -1 else None

    func = f"inner_{loop_id}"
    # Use the FULL scalar kernel.cpp (it carries its own includes — e.g.
    # <algorithm> for sort loops, <cstring> for string loops), so it compiles
    # as-is and shows the agent exactly what to beat.
    scalar_code = kern.strip()
    if not struct_def or not scalar_code or func not in kern:
        return None

    d = ts.definitions.get(loop_id)
    return {
        "name": (d.name if d else loop_id),
        "purpose": (d.description if d and d.description else "Vectorize the scalar loop below."),
        "struct_def": struct_def.strip(),
        "scalar_code": scalar_code,
        "func": func,
    }


def _build_agent_solution(ts: TraceSet, loop_id: str, kernel_code: str) -> Solution:
    """Build a self-contained candidate Solution: the `autovec` baseline's harness
    + spec (same -O3 -march=native build), with `kernel.cpp` swapped for the LLM's
    kernel. Building on the autovec template means the agent competes on identical
    compile footing with the baseline it's trying to beat."""
    template = _solution_by_author(ts, loop_id, "autovec")
    assert template is not None, f"no autovec template for {loop_id}"
    # Generous, always-safe include prelude so candidate kernels for any loop
    # shape compile without the LLM having to remember headers: SIMD intrinsics,
    # plus <algorithm>/<cstring> for sort/string loops. (kernel_code may repeat
    # any of these — header guards make that harmless.)
    wrapped = (
        f'#include "{loop_id}.h"\n'
        f"#include <stdint.h>\n#include <cstdint>\n#include <cstring>\n#include <algorithm>\n"
        f"#include <arm_neon.h>\n"
        f"#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n\n"
        f"{kernel_code}\n"
    )
    sources = [
        SourceFile(path=s.path, content=(wrapped if s.path == "kernel.cpp" else s.content))
        for s in template.sources
    ]
    return Solution(
        name=f"{loop_id}_agent",
        definition=loop_id,
        dataset=SupportedDatasets.SIMD_LOOP,
        author="agent",
        spec=template.spec,
        sources=sources,
    )


# ── Evaluation (delegated entirely to the benchmark runner) ───────────────────

def _eval_solution(ts: TraceSet, loop_id: str, kernel_code: str, cfg: EvalConfig,
                   *, perf: bool = True) -> dict:
    """Compile + score a candidate kernel through the standard runner/evaluator.

    Returns {"status": passed|incorrect|compile_error|error, "results": [...],
    "message": str}. `results` carry per-workload N, source tag, status, min_ns.
    """
    d = ts.definitions.get(loop_id)
    if d is None:
        return {"status": "error", "message": f"no definition {loop_id}", "results": []}
    wls = list(ts.workloads.get(loop_id, []))
    if not perf:
        wls = [w for w in wls if w.tags.get("source") == "edge"]

    sol = _build_agent_solution(ts, loop_id, kernel_code)
    try:
        traces = run_solution_on_workloads(d, sol, wls, cfg=cfg, trace_set=ts)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"runner raised: {e}", "results": []}

    if traces and all(t.evaluation.status == EvaluationStatus.COMPILE_ERROR for t in traces):
        return {"status": "compile_error", "message": traces[0].evaluation.log, "results": []}

    results, overall_ok = [], True
    for t in sorted(traces, key=lambda t: t.workload.axes.get("N", 0)):
        ev = t.evaluation
        st = ev.status
        row = {
            "N": t.workload.axes.get("N"),
            "source": t.workload.tags.get("source"),
            "status": "passed" if st == EvaluationStatus.PASSED else st.value,
            "min_ns": ev.performance.min_ns if (st == EvaluationStatus.PASSED and ev.performance) else None,
            "log": "" if st == EvaluationStatus.PASSED else (ev.log or "")[:200],
        }
        if st != EvaluationStatus.PASSED:
            overall_ok = False
        results.append(row)

    return {"status": "passed" if overall_ok else "incorrect", "results": results, "message": ""}


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(messages: list, model: str, api_key: str) -> str:
    import litellm
    response = litellm.completion(
        model=model, messages=messages, api_key=api_key,
        temperature=0.2, max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _extract_code(text: str) -> str:
    text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
    return text


def _is_arm() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


# ── Main agentic loop ─────────────────────────────────────────────────────────

def run_agent(loop_id: str, model: str, api_key: str, max_turns: int,
              ts: TraceSet, cfg: EvalConfig) -> None:
    ctx = _loop_context(ts, loop_id)
    if ctx is None:
        print(f"  SKIP {loop_id}: no reference/autovec solution to build from")
        return
    isa_desc = "Arm Neoverse V2 (Graviton4, SVE2 128-bit)" if _is_arm() else "Apple Silicon (ARM NEON)"
    isa_upper = "SVE2" if _is_arm() else "NEON"

    system_msg = {
        "role": "system",
        "content": (
            f"You are an expert AArch64 SIMD programmer targeting {isa_desc}. "
            f"Write an optimized C++ kernel for the given loop problem. "
            f'Rules: define `extern "C" void {ctx["func"]}(struct {loop_id}_data *data)` '
            f"(not static). The struct header and <arm_neon.h>/<arm_sve.h> are already "
            f"included — do not redefine the struct. Your output must match the scalar "
            f"baseline's result exactly. Output ONLY the C++ function — no markdown."
        ),
    }
    first_user = (
        f"Problem: {ctx['name']}\nPurpose: {ctx['purpose']}\nTarget: {isa_upper} on {isa_desc}\n\n"
        f"Struct (available via the header — do not redefine):\n```c\n{ctx['struct_def']}\n```\n\n"
        f"Scalar baseline to beat:\n```c\n{ctx['scalar_code']}\n```\n\n"
        f"Write an optimized {isa_upper} implementation. extern \"C\", no static."
    )

    messages = [system_msg, {"role": "user", "content": first_user}]
    best_ns: Optional[int] = None
    best_kernel: Optional[str] = None

    print(f"\n{'='*60}")
    print(f"Agent: {loop_id} — {ctx['name']} on {isa_desc}")
    print(f"Model: {model}  max_turns: {max_turns}")
    print(f"{'='*60}\n")

    for turn in range(1, max_turns + 1):
        print(f"── Turn {turn}/{max_turns} ──────────────────────────")
        raw = _call_llm(messages, model, api_key)
        kernel = _extract_code(raw)
        messages.append({"role": "assistant", "content": raw})

        print(f"Kernel preview: {kernel[:120].replace(chr(10), ' ')}...")
        result = _eval_solution(ts, loop_id, kernel, cfg)
        status = result["status"]

        if status == "compile_error":
            msg = result["message"][:800]
            feedback = f"Compile error:\n{msg}\n\nFix the compilation error and rewrite the function."
            print(f"  COMPILE ERROR: {msg[:120]}")
        elif status == "error":
            feedback = f"Error: {result['message']}\n\nRewrite the function."
            print(f"  ERROR: {result['message'][:120]}")
        elif status == "incorrect":
            bad = [r for r in result["results"] if r["status"] != "passed"]
            lines = "\n".join(f"  N={r['N']} ({r['source']}): {r['status']} {r.get('log','')}" for r in bad[:3])
            feedback = (f"Result does not match the scalar reference:\n{lines}\n\n"
                        f"Fix correctness and rewrite.")
            print(f"  INCORRECT on {len(bad)} workload(s)")
            for r in result["results"]:
                print(f"    N={r['N']:>9}: {r['status']}")
        elif status == "passed":
            # Optimize on PERF latency: the largest-N timed workload (not the tiny
            # edge case, which is ~call overhead).
            timed = [r for r in result["results"] if r.get("min_ns")]
            perf_r = max(timed, key=lambda r: r["N"]) if timed else None
            perf_ns = perf_r["min_ns"] if perf_r else None
            n_perf = perf_r["N"] if perf_r else 0
            print(f"  PASSED — timing:")
            for r in result["results"]:
                ns_str = f"{r['min_ns']} ns" if r.get("min_ns") else "n/a"
                print(f"    N={r['N']:>9}: {r['status']}  {ns_str} [{r.get('source')}]")

            if perf_ns is not None and (best_ns is None or perf_ns < best_ns):
                best_ns, best_kernel = perf_ns, kernel
                print(f"  → New best (perf N={n_perf}): {best_ns} ns")

            if turn < max_turns:
                feedback = (
                    f"Correct! Best timing so far: {perf_ns} ns at N={n_perf}.\n"
                    f"Optimize further — wider accumulators, more unrolling, better "
                    f"scheduling. Rewrite for maximum throughput."
                )
            else:
                break
        else:
            feedback = f"Unexpected status {status}."

        if turn < max_turns:
            messages.append({"role": "user", "content": feedback})

    print(f"\n{'='*60}\nFinal result: {loop_id}")
    if best_kernel:
        print(f"Best timing: {best_ns} ns")
        print(f"Best kernel:\n{best_kernel[:500]}")
    else:
        print("No correct solution found.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_self_test(loops: List[str], ts: TraceSet, cfg: EvalConfig) -> int:
    """Score each loop's own scalar baseline through the runner (no LLM/key). If
    the standard pipeline can build+run+score a loop, the agent can too. Uses edge
    workloads only for speed. Returns the number of loops that failed."""
    failed = 0
    for loop_id in loops:
        ctx = _loop_context(ts, loop_id)
        if ctx is None:
            print(f"  {loop_id:10s} SKIP  (no reference/autovec)")
            failed += 1
            continue
        result = _eval_solution(ts, loop_id, ctx["scalar_code"], cfg, perf=False)
        ok = result["status"] == "passed"
        print(f"  {loop_id:10s} {'OK ' if ok else 'FAIL'}  {result['status']}"
              + ("" if ok else f"  {result.get('message','')[:80]}"))
        if not ok:
            failed += 1
    print(f"\nself-test: {len(loops) - failed}/{len(loops)} loops wired correctly")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default="loop_001")
    ap.add_argument("--all-loops", action="store_true")
    ap.add_argument("--loops", default="", help="comma-separated loop ids")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4-6")
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--self-test", action="store_true",
                    help="validate loop wiring via the scalar baseline (no LLM/key)")
    args = ap.parse_args()

    ts = TraceSet.from_path(str(BENCH_TRACE))
    cfg = BenchmarkConfig().resolve_eval_config()

    if args.loops:
        loops = [l.strip() for l in args.loops.split(",") if l.strip()]
    elif args.all_loops:
        loops = _available_loops(ts)
    else:
        loops = [args.loop]

    unknown = [l for l in loops if l not in ts.definitions]
    if unknown:
        print(f"ERROR: unknown loop(s): {unknown}")
        sys.exit(1)

    if args.self_test:
        sys.exit(1 if _run_self_test(loops, ts, cfg) else 0)

    if not args.key:
        print("ERROR: set OPENROUTER_API_KEY or pass --key")
        sys.exit(1)

    for loop_id in loops:
        run_agent(loop_id, args.model, args.key, args.max_turns, ts, cfg)


if __name__ == "__main__":
    main()
