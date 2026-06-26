#!/usr/bin/env python3
"""Local agentic loop: LLM iterates on a SIMD loop problem using the bench harness.

Unlike the SSH eval (eval/run_benchmark.py), this runs entirely in-process:
the LLM generates a kernel, bench compiles and runs it locally (or on the
current machine if deployed to Graviton), and any compile/correctness failures
go straight back to the LLM as feedback.

Usage (run ON the target machine — Graviton4 for SVE2 timing):
    OPENROUTER_API_KEY=sk-or-... python3 scripts/bench_loop_agent.py --loop loop_001
    python3 scripts/bench_loop_agent.py --loop loop_001 --max-turns 6 --model openrouter/anthropic/claude-opus-4-6

The 'perf' displayed is min_ns on whatever machine this runs on.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.compile.builders.simd_loop import SimdLoopBuilder
from bench.data.definition import Definition
from bench.data.solution import Solution, SolutionSpec, SourceFile, SupportedDatasets
from bench.data.trace_set import TraceSet
from bench.data.workload import Workload
from bench.datasets.simd_loop import SimdLoopDataset
from bench.runtime.inputs import gen_inputs_for_workload
from bench.runtime.timing import time_callable

BENCH_TRACE = ROOT / "bench-trace"

# ── Compiler detection ────────────────────────────────────────────────────────

def _find_compiler() -> str:
    for name in ("clang++-18", "clang++-17", "clang++"):
        if shutil.which(name):
            return name
    raise RuntimeError("No clang++ found on PATH")


def _patch_compiler(compiler: str) -> None:
    """Monkey-patch subprocess.run so SimdLoopBuilder uses the right compiler."""
    import bench.compile.builders.simd_loop as _mod
    orig_build = _mod.SimdLoopBuilder.build

    def _patched_build(self, definition, solution):
        # Temporarily replace "clang++" with the detected compiler.
        import bench.compile.builder as b
        orig_run = subprocess.run
        def _run(cmd, **kw):
            if cmd and cmd[0] == "clang++":
                cmd = [compiler] + cmd[1:]
            return orig_run(cmd, **kw)
        subprocess.run = _run
        try:
            return orig_build(self, definition, solution)
        finally:
            subprocess.run = orig_run

    _mod.SimdLoopBuilder.build = _patched_build


# ── Problem table ─────────────────────────────────────────────────────────────

PROBLEMS = {
    "loop_001": {
        "name": "FP32 inner product", "purpose": "Use fp32 FMA instruction",
        "struct_def": "struct loop_001_data { float *a; float *b; int n; float res; };",
        "scalar_code": (
            'extern "C" void inner_loop_001(struct loop_001_data *data) {\n'
            '  float res = 0.0f;\n'
            '  for (int i = 0; i < data->n; i++) res += data->a[i] * data->b[i];\n'
            '  data->res = res;\n}'
        ),
        "header": "loop_001.h", "func": "inner_loop_001",
    },
    "loop_002": {
        "name": "UINT32 inner product", "purpose": "Use u32 MLA instruction",
        "struct_def": "struct loop_002_data { uint32_t *a; uint32_t *b; int n; uint32_t res; };",
        "scalar_code": (
            'extern "C" void inner_loop_002(struct loop_002_data *data) {\n'
            '  uint32_t res = 0;\n'
            '  for (int i = 0; i < data->n; i++) res += data->a[i] * data->b[i];\n'
            '  data->res = res;\n}'
        ),
        "header": "loop_002.h", "func": "inner_loop_002",
    },
    "loop_003": {
        "name": "FP64 inner product", "purpose": "Use fp64 FMA instruction",
        "struct_def": "struct loop_003_data { double *a; double *b; int n; double res; };",
        "scalar_code": (
            'extern "C" void inner_loop_003(struct loop_003_data *data) {\n'
            '  double res = 0.0;\n'
            '  for (int i = 0; i < data->n; i++) res += data->a[i] * data->b[i];\n'
            '  data->res = res;\n}'
        ),
        "header": "loop_003.h", "func": "inner_loop_003",
    },
}

# ── Auto-extend the problem table from bench-trace reference solutions ─────────
# The three loops above are hand-curated. Every other scalar-output loop has a
# `reference` solution whose harness sources already contain exactly what the
# agent prompt needs: the struct (in `loop_NNN.h`) and the scalar inner loop
# (in `kernel.cpp`). Extract them so the agent can target many loops without
# hand-writing each one. Loops the extractor can't cleanly parse are skipped.

# Scalar-output reduction loops whose eval shape matches `_eval_kernel`
# (input `a` present, single scalar `res` output).
_AUTO_SCALAR_LOOPS = [
    "loop_004", "loop_008", "loop_010", "loop_024",
    "loop_032", "loop_033", "loop_126", "loop_127",
]


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
                # include trailing ';' for structs
                end = i + 1
                if end < len(text) and text[end] == ";":
                    end += 1
                return text[start_idx:end]
    return None


def _load_problem_from_trace(loop_id: str) -> Optional[dict]:
    """Build a PROBLEMS-style entry from the `reference` solution sources."""
    import glob
    sol_glob = str(BENCH_TRACE / "solutions" / "simd-loop" / "reference" / loop_id / "*.json")
    matches = glob.glob(sol_glob)
    if not matches:
        return None
    try:
        sol = json.loads(Path(matches[0]).read_text())
        srcs = {s["path"]: s["content"] for s in sol.get("sources", [])}
        hdr = srcs.get(f"{loop_id}.h", "")
        kern = srcs.get("kernel.cpp", "")

        si = hdr.find(f"struct {loop_id}_data")
        struct_def = _extract_braced_block(hdr, si) if si != -1 else None

        func = f"inner_{loop_id}"
        fi = kern.find(f"void {func}")
        scalar_code = None
        if fi != -1:
            # back up to the start of the declaration (extern "C" ...)
            line_start = kern.rfind("\n", 0, fi) + 1
            block = _extract_braced_block(kern, line_start)
            if block:
                scalar_code = block.strip()

        if not struct_def or not scalar_code:
            return None

        dpath = BENCH_TRACE / "definitions" / "simd-loop" / f"{loop_id}.json"
        name = loop_id
        purpose = "Vectorize the scalar reduction below for SVE2/NEON."
        if dpath.exists():
            d = json.loads(dpath.read_text())
            name = d.get("name", loop_id)
            purpose = d.get("description", purpose)

        return {
            "name": name, "purpose": purpose,
            "struct_def": struct_def.strip(),
            "scalar_code": scalar_code,
            "header": f"{loop_id}.h", "func": func,
        }
    except (OSError, ValueError, KeyError):
        return None


for _lid in _AUTO_SCALAR_LOOPS:
    if _lid not in PROBLEMS:
        _entry = _load_problem_from_trace(_lid)
        if _entry is not None:
            PROBLEMS[_lid] = _entry


# ── Compile + eval ────────────────────────────────────────────────────────────

def _eval_kernel(kernel_code: str, prob: dict, ts: TraceSet,
                 edge_only: bool = True) -> dict:
    """Compile kernel_code and run edge workloads. Returns result dict."""
    op = list(prob["header"].replace(".h", "").split("/"))[-1]
    op_type = op  # e.g. "loop_001"
    header = prob["header"]
    func = prob["func"]

    full_src = f'#include "{header}"\n#include <stdint.h>\n#include <arm_neon.h>\n\n{kernel_code}\n'
    sol = Solution(
        name=f"{op_type}_agent",
        definition=op_type,
        dataset=SupportedDatasets.SIMD_LOOP,
        author="agent",
        spec=SolutionSpec(
            target_hardware=["aarch64"],
            entry_point=f"kernel.cpp::{func}",
            compile_flags=["-O3", "-march=armv9-a+sve2", "-std=c++14"],
        ),
        sources=[SourceFile(path="kernel.cpp", content=full_src)],
    )

    d = ts.definitions.get(op_type)
    if d is None:
        return {"status": "error", "message": f"Definition {op_type!r} not in bench-trace"}

    builder = SimdLoopBuilder()
    try:
        compiled = builder.build(d, sol)
    except Exception as e:
        return {"status": "compile_error", "message": str(e)}

    try:
        lib = ctypes.CDLL(str(compiled.so_path))
        sym = getattr(lib, f"armbench_entry_{op_type}")
        sym.restype = ctypes.c_int
        sym._lib = lib

        ds = SimdLoopDataset()
        import numpy as np
        # Correctness is checked on the small EDGE workloads (fast feedback); the
        # reported latency comes from the large PERF workloads (N in the millions)
        # — same scale the timing sweep / autovec baseline use. Timing only the
        # edge workloads is meaningless: max N≈10k is dominated by call overhead.
        edge_wls = sorted((w for w in ts.workloads.get(op_type, [])
                           if w.tags.get("source") == "edge"), key=lambda w: w.axes["N"])
        perf_wls = sorted((w for w in ts.workloads.get(op_type, [])
                           if w.tags.get("source") == "perf"), key=lambda w: w.axes["N"])
        workloads = edge_wls if edge_only else (edge_wls + perf_wls)

        results = []
        all_passed = True
        for wl in workloads:
            # Don't run a huge perf input through a kernel that already failed edge.
            if wl.tags.get("source") == "perf" and not all_passed:
                break
            np_inputs = gen_inputs_for_workload(d, wl)
            ctx = ds.wrap_inputs(np_inputs, op_type, lib, definition=d)
            rc = sym(*ctx.entry_args)
            if rc != 0:
                results.append({"N": wl.axes["N"], "status": "runtime_error", "rc": rc})
                all_passed = False
                ds.release(ctx)
                continue

            out = ds.unwrap_output(ctx)
            ref_run_ns: dict = {}
            # Compute reference
            exec_ns: dict = {}
            exec(d.reference, exec_ns)
            ref_val = exec_ns["run"](**{k: v for k, v in np_inputs.items()})
            import numpy as np
            ref_arr = np.asarray([ref_val]) if isinstance(ref_val, np.generic) else ref_val
            if hasattr(ref_arr, "detach"):
                ref_arr = ref_arr.detach().numpy()

            ref_scalar = float(ref_arr.flat[0])
            diff = abs(float(out[0]) - ref_scalar)
            # Hybrid abs+rel tolerance: at perf N (millions) float reductions
            # differ from numpy purely by summation order, so a fixed abs tol
            # would false-fail correct kernels. Integers must be near-exact.
            if d.inputs["a"].dtype.value in ("float32", "float64"):
                ok = diff <= max(1e-2, 1e-2 * abs(ref_scalar))
            else:
                ok = diff <= 1
            tol = None

            # Time it (only if correct)
            min_ns = None
            if ok:
                try:
                    timing = time_callable(lambda: sym(*ctx.entry_args), warmup=5, repeat=20)
                    min_ns = timing.min_ns
                except Exception:
                    pass

            results.append({
                "N": wl.axes["N"],
                "source": wl.tags.get("source"),
                "status": "passed" if ok else "incorrect",
                "diff": diff,
                "min_ns": min_ns,
            })
            if not ok:
                all_passed = False
            ds.release(ctx)

        return {
            "status": "passed" if all_passed else "incorrect",
            "results": results,
        }
    finally:
        shutil.rmtree(compiled.build_dir, ignore_errors=True)


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _call_llm(messages: list, model: str, api_key: str) -> str:
    import litellm
    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _extract_code(text: str) -> str:
    text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
    return text


def _is_arm() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


# ── Main agentic loop ─────────────────────────────────────────────────────────

def run_agent(loop_id: str, model: str, api_key: str, max_turns: int, compiler: str) -> None:
    prob = PROBLEMS[loop_id]
    isa_desc = "Arm Neoverse V2 (Graviton4, SVE2 128-bit)" if _is_arm() else "Apple Silicon (ARM NEON)"
    isa_upper = "SVE2" if _is_arm() else "NEON"

    ts = TraceSet.from_path(str(BENCH_TRACE))

    system_msg = {
        "role": "system",
        "content": (
            f"You are an expert AArch64 SIMD programmer targeting {isa_desc}. "
            f"Write an optimized C++ kernel for the given loop problem. "
            f"Rules: use extern \"C\" (not static). Include <arm_neon.h> or "
            f"<arm_sve.h> as needed. The `res` field must match the scalar output. "
            f"Output ONLY the C++ function — no markdown, no explanation."
        ),
    }

    first_user = (
        f"Problem: {prob['name']}\nPurpose: {prob['purpose']}\nTarget: {isa_upper} on {isa_desc}\n\n"
        f"Struct (already available via the header — do not redefine):\n```c\n{prob['struct_def']}\n```\n\n"
        f"Scalar baseline to beat:\n```c\n{prob['scalar_code']}\n```\n\n"
        f"Write an optimized {isa_upper} implementation. extern \"C\", no static."
    )

    messages = [system_msg, {"role": "user", "content": first_user}]
    best_ns: Optional[int] = None
    best_kernel: Optional[str] = None

    print(f"\n{'='*60}")
    print(f"Agent: {loop_id} — {prob['name']} on {isa_desc}")
    print(f"Model: {model}  max_turns: {max_turns}")
    print(f"{'='*60}\n")

    for turn in range(1, max_turns + 1):
        print(f"── Turn {turn}/{max_turns} ──────────────────────────")
        raw = _call_llm(messages, model, api_key)
        kernel = _extract_code(raw)
        messages.append({"role": "assistant", "content": raw})

        print(f"Kernel preview: {kernel[:120].replace(chr(10), ' ')}...")
        result = _eval_kernel(kernel, prob, ts, edge_only=False)
        status = result["status"]

        if status == "compile_error":
            msg = result["message"][:800]
            feedback = f"Compile error:\n{msg}\n\nFix the compilation error and rewrite the function."
            print(f"  COMPILE ERROR: {msg[:120]}")

        elif status == "incorrect":
            bad = [r for r in result["results"] if r["status"] == "incorrect"]
            lines = "\n".join(f"  N={r['N']}: diff={r['diff']:.3e}" for r in bad[:3])
            feedback = f"Correctness failed:\n{lines}\n\nThe result `res` does not match the scalar reference. Fix and rewrite."
            print(f"  INCORRECT on {len(bad)} workload(s)")
            for r in result["results"]:
                print(f"    N={r['N']:6d}: {r['status']}  diff={r.get('diff', '?'):.2e}")

        elif status == "passed":
            # Optimize on the PERF latency: the largest-N timed workload, not the
            # min (which would be the tiny edge case ≈ call overhead).
            timed = [r for r in result["results"] if r.get("min_ns")]
            perf_r = max(timed, key=lambda r: r["N"]) if timed else None
            perf_ns = perf_r["min_ns"] if perf_r else None
            n_perf = perf_r["N"] if perf_r else 0
            print(f"  PASSED — timing:")
            for r in result["results"]:
                ns_str = f"{r['min_ns']} ns" if r.get("min_ns") else "n/a"
                tag = f" [{r.get('source')}]" if r.get("source") else ""
                print(f"    N={r['N']:>9d}: {r['status']}  {ns_str}{tag}")

            if perf_ns is not None and (best_ns is None or perf_ns < best_ns):
                best_ns = perf_ns
                best_kernel = kernel
                print(f"  → New best (perf N={n_perf}): {best_ns} ns")

            if turn < max_turns:
                ns_perf = perf_ns
                feedback = (
                    f"Correct! Best timing so far: {ns_perf} ns at N={n_perf}.\n"
                    f"Can you optimize further? Try unrolling more, using wider accumulators, "
                    f"or better instruction scheduling. Rewrite for maximum throughput."
                )
            else:
                break
        else:
            feedback = f"Error: {result.get('message', status)}"
            print(f"  ERROR: {feedback}")

        if turn < max_turns:
            messages.append({"role": "user", "content": feedback})

    print(f"\n{'='*60}")
    print(f"Final result: {loop_id}")
    if best_kernel:
        print(f"Best timing: {best_ns} ns")
        print(f"Best kernel:\n{best_kernel[:500]}")
    else:
        print("No correct solution found.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_self_test(loops: list, compiler: str) -> int:
    """Compile + run each loop's scalar baseline through the eval path.

    Validates that every selected loop is wired end-to-end (struct extraction,
    definition lookup, input gen, correctness, timing) WITHOUT calling any LLM —
    so the overnight agent run can't be blocked by a mis-wired loop. Returns the
    number of loops that failed.
    """
    ts = TraceSet.from_path(str(BENCH_TRACE))
    failed = 0
    for loop_id in loops:
        prob = PROBLEMS[loop_id]
        result = _eval_kernel(prob["scalar_code"], prob, ts)
        status = result["status"]
        ok = status == "passed"
        print(f"  {loop_id:10s} {'OK ' if ok else 'FAIL'}  {status}"
              + ("" if ok else f"  {result.get('message','')[:80]}"))
        if not ok:
            failed += 1
    print(f"\nself-test: {len(loops) - failed}/{len(loops)} loops wired correctly")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default="loop_001", choices=list(PROBLEMS))
    ap.add_argument("--all-loops", action="store_true")
    ap.add_argument("--loops", default="", help="comma-separated loop ids, e.g. loop_001,loop_024")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--model", default="openrouter/anthropic/claude-opus-4-6")
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--self-test", action="store_true",
                    help="validate loop wiring via the scalar baseline (no LLM/key needed)")
    args = ap.parse_args()

    if args.loops:
        loops = [l.strip() for l in args.loops.split(",") if l.strip()]
        unknown = [l for l in loops if l not in PROBLEMS]
        if unknown:
            print(f"ERROR: unknown loop(s): {unknown}\nknown: {sorted(PROBLEMS)}")
            sys.exit(1)
    elif args.all_loops:
        loops = list(PROBLEMS)
    else:
        loops = [args.loop]

    compiler = _find_compiler()
    print(f"Using compiler: {compiler}")
    _patch_compiler(compiler)

    if args.self_test:
        sys.exit(1 if _run_self_test(loops, compiler) else 0)

    if not args.key:
        print("ERROR: set OPENROUTER_API_KEY or pass --key")
        sys.exit(1)

    for loop_id in loops:
        run_agent(loop_id, args.model, args.key, args.max_turns, compiler)


if __name__ == "__main__":
    main()
