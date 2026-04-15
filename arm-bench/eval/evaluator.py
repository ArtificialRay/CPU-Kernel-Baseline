"""
eval/evaluator.py — Agentic LLM evaluation orchestrator for ncnn kernel optimization.

Runs an agent loop where the LLM iteratively reads, modifies, and benchmarks
ncnn kernel implementations (candidate vs ARM baseline) using
compile/run/perf/disassemble tools.

Compatible with any LiteLLM-supported model.
"""

import copy
import datetime
import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
import litellm

from eval.config import REPO_ROOT, ISA_INSTANCE_DESC
from eval.tools import SIMDTools, EvalResult
from eval.context import _compress_history_with_code,_compress_history

# Load .env from arm-bench/ root
load_dotenv(REPO_ROOT / ".env")

STARTER_DIR = REPO_ROOT / "starter"
STARTER_PROBLEMS_JSON = STARTER_DIR / "problems.json"
BASELINES_FILE = REPO_ROOT / "baselines" / "arm_baselines.json"


# ─── Problem / baseline loaders ──────────────────────────────────────────────

def load_starter_problems() -> dict[str, dict]:
    """Load starter/problems.json, keyed by problem ID."""
    raw = json.loads(STARTER_PROBLEMS_JSON.read_text())
    return {p["id"]: p for p in raw}


def load_arm_baselines() -> dict:
    """Load baselines/arm_baselines.json."""
    if BASELINES_FILE.exists():
        return json.loads(BASELINES_FILE.read_text())
    return {}


# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AArch64 programmer specializing in ARM NEON/SVE optimizations
for neural network kernels. Your task is to optimize an ncnn kernel implementation
to run faster than the hand-tuned ARM baseline.

You are given:
  - The current candidate kernel source code (a base C++ implementation)
  - ARM baseline performance numbers (the target to beat)
  - A test harness that validates correctness and measures performance

You have access to five tools:
  - compile(code): Replace the candidate kernel source with your optimized code,
    rebuild the cmake libraries, and compile the test binary. The code must be a
    complete replacement for the candidate source file.
  - run(n): Run both candidate and baseline binaries (n iterations). Check
    correctness and compare timing.
  - perf(n): Collect hardware PMU counters (cycles, instructions, IPC, L1D miss
    rate) for both candidate and baseline.
  - submit(code): Submit your final optimized implementation for scoring.

Guidelines:
  - Correctness first: your optimized kernel must produce the same output as the
    reference implementation within tolerance (1e-3).
  - Start by calling compile() with the original source to establish a baseline,
    then iterate.
  - Use perf() to understand bottlenecks and verify vectorization.
  - Common optimization strategies: NEON intrinsics, loop tiling, cache-friendly
    access patterns, packing weights, reducing branches.
  - Call submit() when you are satisfied with your optimization.
  - Be efficient: fewer tool calls is better, but correctness comes first.
"""
# TODO: disassemble() is deprecated in nn-bench, will integrate it later
# Prompt to use disassemble: - Use perf() and disassemble() to understand bottlenecks and verify vectorization.
# use disassemble:   - disassemble(fn): Inspect generated AArch64 assembly for a specific function.


# ─── User prompt builder ────────────────────────────────────────────────────

def build_user_prompt(problem: dict, candidate_source: str, baselines: dict, isa: str) -> str:
    """Build the initial user message shown to the LLM."""
    pid = problem["id"]
    isa_desc = ISA_INSTANCE_DESC.get(isa, isa)
    baseline = baselines.get(pid, {})

    baseline_section = ""
    if baseline:
        baseline_section = (
            f"\nARM baseline performance (the target to beat):\n"
            f"  - Runtime: {baseline.get('baseline_ms', 'N/A')} ms/iter\n"
        )
        perf = baseline.get("perf", {})
        if perf:
            baseline_section += (
                f"  - Cycles: {perf.get('cycles', 'N/A')}\n"
                f"  - Instructions: {perf.get('instructions', 'N/A')}\n"
                f"  - IPC: {perf.get('ipc', 'N/A')}\n"
                f"  - L1D miss rate: {perf.get('l1d_miss_pct', 'N/A')}%\n"
            )

    return f"""\
Problem: {problem["name"]}
Description: {problem.get("description", "")}
ISA target: {isa.upper()} on {isa_desc}
Candidate source file: {problem["candidate_source"]}
Starter test file: {problem["starter"]}
{baseline_section}
Current candidate kernel implementation:
```cpp
{candidate_source}
```

Your task: optimize this kernel implementation to be faster than the ARM baseline.
Start by calling compile() with the source code above (unmodified) to verify the
build works, then iterate with optimizations. Call submit() when done.

Optimization checklist you may find interesting:
#### Essential Optimizations(apply first):

- [ ] **Channel Packing** : Group channels into SIMD-width bundles (pack4 for fp32, pack8 for fp16). Kernel weights and output accumulators must match the same packing scheme.  
- [ ] **SIMD Intrinsics** : Replace scalar arithmetic with NEON/SVE/SME intrinsics . Every FP op in the hot loop must be vectorized.
- [ ] **Inline Assembly for Hot Paths**: Use inline asm for the widest unroll tiers to control register assignment, instruction ordering, and load/compute interleaving explicitly.
- [ ] **OpenMP Parallelism**: Parallelize the outermost embarrassingly-parallel dimension  (output channels) with `#pragma omp parallel for num_threads(opt.num_threads)`.  
- [ ] **Norm or Activation Fusion** : Fuse the activation function (ReLU, Sigmoid, etc.) into the end of the compute kernel, before storing to memory.

####  **Performance Optimizations (Apply as Needed)**

- [ ] **Loop-Invariant Weight Hoisting**: Load all kernel weights into SIMD registers before the spatial loop. 
- [ ] **Bias Pre-fill**: Initialize the output tensor with bias via `out.fill(_bias)` before the accumulation loop. 
- [ ] **Output Channel Tiling (Register Blocking)**: On aarch64 (32 SIMD registers), process 2 output channel groups simultaneously (`pp * 2`).  Fall back to 1-group on armv7 (16 registers).
- [ ] **Spatial Loop Unrolling (Graduated Widths)**: Use greedy remainder reduction for the output width loop: process `j+=8` → `j+=4` → `j+=2` → `j+=1`. Use inline asm for j>=2 paths; intrinsics are sufficient for the j=1 tail.
"""


# ─── Perf history persistence ────────────────────────────────────────────────

HISTORY_ROOT = REPO_ROOT / "history"


def _init_history_dir(problem_id: str) -> Path:
    """
    Create and return a local history directory for this eval run.

    Directory name: ``{HISTORY_ROOT}/{problem_id}-{timestamp}``
    e.g. ``arm-bench/history/conv2d-20260415_143022``
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = HISTORY_ROOT / f"{problem_id}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_code_snapshot(
    history_dir: Path,
    problem: dict,
    code: str,
    action: str,
    result_dict: dict,
    speedup: float | None,
    turn: int,
) -> None:
    """
    Save candidate source and result metadata to history_dir.

    ``action`` is one of ``"compiled"``, ``"run"``, or ``"perf"``.

    Filename stem: ``{pid}_turn{N}_{action}`` or ``{pid}_turn{N}_{action}_{speedup}x``
    e.g. ``conv2d_turn3_run_1.42x.cpp``
    """
    pid = problem["id"]
    speedup_suffix = f"_{speedup:.2f}x" if speedup is not None else ""
    stem = f"{pid}_turn{turn}_{action}{speedup_suffix}"

    entry: dict = {
        "turn": turn,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "problem_id": pid,
        "action": action,
        "speedup_vs_baseline": speedup,
    }
    if action == "run":
        entry["candidate_runtime_ms"] = result_dict.get("candidate runtime_ms")
        entry["baseline_runtime_ms"] = result_dict.get("baseline runtime_ms")
        entry["correct"] = result_dict.get("correct")
    elif action == "perf":
        if "candidate" in result_dict:
            cp = result_dict["candidate"]
            entry["candidate_cycles"] = cp.get("cycles")
            entry["candidate_ipc"] = cp.get("ipc")
            entry["candidate_l1d_miss_pct"] = cp.get("l1d_miss_pct")
        if "baseline" in result_dict:
            bp = result_dict["baseline"]
            entry["baseline_cycles"] = bp.get("cycles")
            entry["baseline_ipc"] = bp.get("ipc")

    (history_dir / f"{stem}.json").write_text(json.dumps(entry, indent=2))
    (history_dir / f"{stem}.cpp").write_text(code)


# ─── Tool dispatch with candidate source injection ─────────────────────────

def _write_candidate_source(tools: SIMDTools, problem: dict, code: str) -> None:
    """
    Write the agent's optimized code to the candidate source file on the remote.

    This updates e.g. ncnn/mapped/convolution/convolution.cpp so the next cmake
    rebuild picks up the changes.
    """
    candidate_rel = problem["candidate_source"]
    remote_path = f"{tools.remote_ncnn_root}/{candidate_rel}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        tools._upload(tmp_path, remote_path)
    finally:
        os.unlink(tmp_path)


def dispatch_tool(tools: SIMDTools, problem: dict, name: str, args: dict) -> dict:
    """
    Dispatch a tool call, injecting candidate source for compile/submit.

    For compile(code) and submit(code):
      1. Write the agent's code to the remote candidate source file
      2. Trigger cmake rebuild + starter compile via SIMDTools
    Other tools delegate directly to SIMDTools.
    """
    starter = problem["starter"]

    if name == "compile":
        code = args.get("code", "")
        _write_candidate_source(tools, problem, code)
        return tools.compile(starter).to_tool_result()

    elif name == "submit":
        code = args.get("code", "")
        _write_candidate_source(tools, problem, code)
        return tools.submit(starter).to_dict()

    elif name == "run":
        return tools.run(args.get("n", 1)).to_tool_result()

    elif name == "perf":
        result = tools.perf(args.get("n", 1))
        if isinstance(result, tuple):
            candidate_perf, baseline_perf = result
            return {
                "candidate": candidate_perf.to_tool_result(),
                "baseline": baseline_perf.to_tool_result(),
            }
        return result.to_tool_result()

    elif name == "disassemble":
        return tools.disassemble(args.get("fn")).to_tool_result()

    else:
        return {"error": f"Unknown tool: {name}"}


# ─── Tool schemas ───────────────────────────────────────────────────────────

def tool_schemas() -> list[dict]:
    """Return OpenAI-compatible function tool definitions for the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "compile",
                "description": (
                    "Replace the candidate kernel source file with your optimized code, "
                    "rebuild the cmake libraries, and compile both candidate and baseline "
                    "test binaries. Returns whether compilation succeeded and any errors."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Your complete optimized C++ source file for the candidate "
                                "kernel. Must be a full replacement — include all headers, "
                                "namespace, class methods, etc."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run",
                "description": (
                    "Run both candidate and baseline binaries and check correctness + timing. "
                    "Must call compile() successfully first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "description": "Number of iterations (default 1; more = more stable timing).",
                            "default": 1,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "perf",
                "description": (
                    "Run perf stat on both candidate and baseline binaries to collect "
                    "hardware PMU counters: cycles, instructions, IPC, L1D cache miss rate."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "description": "Number of iterations.",
                            "default": 1,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "disassemble",
                "description": (
                    "Disassemble the compiled candidate binary. Filter to a specific "
                    "function to see the generated AArch64 instructions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fn": {
                            "type": "string",
                            "description": (
                                "Function name to filter to. "
                                "If omitted, returns full disassembly (may be large)."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": (
                    "Submit your final optimized implementation for scoring. "
                    "Compiles, verifies correctness, runs timing comparison against "
                    "ARM baseline, and computes speedup. Call when satisfied."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Your final optimized C++ kernel source.",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
    ]


# ─── Main agent loop ────────────────────────────────────────────────────────

def run_agentic_eval(
    problem_id: str,
    isa: str,
    model: str,
    handle,
    max_turns: int = 20,
    verbose: bool = True,
) -> EvalResult:
    """
    Run one agentic evaluation session for an ncnn kernel optimization problem.

    The LLM gets the candidate kernel source, ARM baseline perf numbers, and
    tools (compile, run, perf, disassemble, submit) to iteratively optimize.

    Args:
        problem_id: e.g. "conv2d"
        isa: "neon", "sve", or "sve2"
        model: LiteLLM model string, e.g. "anthropic/claude-opus-4-6"
        handle: SSH handle to the provisioned instance, or None for local
        max_turns: Maximum agent turns before forced submit
        verbose: Print conversation turns

    Returns:
        (EvalResult, perf_speedup_buffer) where perf_speedup_buffer is a list of
        {"turn": int, "speedup_vs_baseline": float} entries, one per perf() call
        that produced a measurable speedup.
    """
    # Load problem metadata
    problems = load_starter_problems()
    if problem_id not in problems:
        raise KeyError(f"Problem {problem_id!r} not found in {STARTER_PROBLEMS_JSON}")
    problem = problems[problem_id]

    # Load ARM baselines
    baselines = load_arm_baselines()

    # Read the candidate source file
    candidate_src_path = REPO_ROOT.parent / "ncnn" / problem["candidate_source"]
    if not candidate_src_path.exists():
        raise FileNotFoundError(f"Candidate source not found: {candidate_src_path}")
    candidate_source = candidate_src_path.read_text()

    # Initialize tools and upload source tree
    tools = SIMDTools(handle=handle, problem_id=problem_id, isa=isa)

    # Local history directory for this run, e.g. history/conv2d-20260415_143022/
    history_dir = _init_history_dir(problem_id)

    schemas = tool_schemas()

    system = SYSTEM_PROMPT
    user_msg = build_user_prompt(problem, candidate_source, baselines, isa)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem_id} ({problem['name']})")
        print(f"ISA: {isa} | Model: {model}")
        print(f"Candidate: {problem['candidate_source']}")
        print(f"Starter: {problem['starter']}")
        baseline = baselines.get(problem_id, {})
        if baseline:
            print(f"ARM baseline: {baseline.get('baseline_ms')}ms/iter")
        print(f"{'='*60}")

    final_result: EvalResult | None = None

    # Latest compiled code; updated on every compile call so perf always snapshots the right code.
    # Also substituted into the compressed user prompt after each perf call.
    latest_compiled_code: str = candidate_source

    # State from the most recent tool call; injected into the user prompt each turn
    # so the LLM can see what happened last (e.g. a compile failure with the failing code).
    last_round_state: dict | None = None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Turn {turn+1}/{max_turns}]")

        compressed = _compress_history_with_code(
            messages,
            latest_compiled_code=latest_compiled_code,
            last_round_state=last_round_state,
        )
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
                arg_preview = {
                    k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                    for k, v in fn_args.items()
                }
                print(f"  -> {fn_name}({arg_preview})")

            result_dict = dispatch_tool(tools, problem, fn_name, fn_args)

            if verbose:
                _print_tool_result(fn_name, result_dict)

            # Keep latest_compiled_code current so run/perf always snapshot the right code
            if fn_name == "compile":
                latest_compiled_code = fn_args.get("code", "")
                success = bool(result_dict.get("success"))
                last_round_state = {
                    "action": "compile",
                    "turn": turn + 1,
                    "success": success,
                    "code": latest_compiled_code if not success else "",
                    "errors": result_dict.get("errors", "") if not success else "",
                }
                # if success:
                #     _save_code_snapshot(history_dir, problem, latest_compiled_code, "compiled", result_dict, None, turn + 1)
                #     if verbose:
                #         print(f"  [code snapshot saved: compiled, turn {turn + 1}]")
                _save_code_snapshot(history_dir, problem, latest_compiled_code, "compiled", result_dict, None, turn + 1)
                if verbose:
                    print(f"  [code snapshot saved: compiled, turn {turn + 1}]")

            # On run: compute runtime speedup, persist snapshot, record state
            elif fn_name == "run":
                speedup: float | None = None
                c_ms = result_dict.get("candidate runtime_ms")
                b_ms = result_dict.get("baseline runtime_ms")
                if c_ms and b_ms and c_ms > 0:
                    speedup = round(b_ms / c_ms, 3)
                last_round_state = {
                    "action": "run",
                    "turn": turn + 1,
                    "correct": result_dict.get("correct"),
                    "candidate_ms": c_ms,
                    "baseline_ms": b_ms,
                    "speedup": speedup,
                }
                _save_code_snapshot(history_dir, problem, latest_compiled_code, "run", result_dict, speedup, turn + 1)
                if verbose:
                    print(f"  [code snapshot saved: run, speedup_vs_baseline={speedup}]")

            # On perf: compute cycle-based speedup, persist history, update buffer, record state
            elif fn_name == "perf":
                speedup = None
                if "candidate" in result_dict and "baseline" in result_dict:
                    c_cycles = result_dict["candidate"].get("cycles")
                    b_cycles = result_dict["baseline"].get("cycles")
                    if c_cycles and b_cycles and c_cycles > 0:
                        speedup = round(b_cycles / c_cycles, 3)
                last_round_state = {
                    "action": "perf",
                    "turn": turn + 1,
                    "speedup": speedup,
                }
                if speedup is not None:
                    tools.perf_speedup_buffer.append({"turn": turn + 1, "speedup_vs_baseline": speedup})
                _save_code_snapshot(history_dir, problem, latest_compiled_code, "perf", result_dict, speedup, turn + 1)
                if verbose:
                    print(f"  [code snapshot saved: perf, speedup_vs_baseline={speedup}]")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict),
            })

            # Capture submit result
            if fn_name == "submit":
                er = EvalResult(**{
                    k: result_dict[k]
                    for k in EvalResult.__dataclass_fields__
                    if k in result_dict
                })
                er.tool_calls = tools._tool_calls
                final_result = er

    # If agent never called submit, force final scoring
    if final_result is None:
        if verbose:
            print("\n[Max turns reached — forcing final scoring]")
        final_result = _force_final_score(tools, problem, baselines)

    # Scan history JSONs to surface the best intermediate perf result.
    # Useful when the final compile failed and final_result carries no speedup.
    best_speedup: float | None = None
    best_json_file: str | None = None
    for f in history_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            s = data.get("speedup_vs_baseline")
            if s is not None and (best_speedup is None or s > best_speedup):
                best_speedup = s
                best_json_file = f.name
        except (json.JSONDecodeError, OSError):
            pass
    final_result.speedup_vs_ref = best_speedup
    final_result.best_history_file = best_json_file

    if verbose:
        print(f"\n[Final Result]")
        print(json.dumps(final_result.to_dict(), indent=2))

    return final_result


def _print_tool_result(fn_name: str, result_dict: dict) -> None:
    """Pretty-print a tool result for verbose output."""
    if fn_name == "submit":
        print(f"  <- {result_dict}")
    elif fn_name == "compile":
        status = "OK" if result_dict.get("success") else "FAILED"
        print(f"  <- compile: {status}")
        if not result_dict.get("success"):
            err = result_dict.get("errors", "")[:200]
            print(f"     {err}")
    elif fn_name == "run":
        correct = result_dict.get("correct")
        c_ms = result_dict.get("candidate runtime_ms")
        b_ms = result_dict.get("baseline runtime_ms")
        print(f"  <- run: correct={correct}, candidate={c_ms}ms, baseline={b_ms}ms")
    elif fn_name == "perf":
        if "candidate" in result_dict:
            cp = result_dict["candidate"]
            bp = result_dict.get("baseline", {})
            print(f"  <- perf: candidate IPC={cp.get('ipc')}, baseline IPC={bp.get('ipc')}")
        else:
            print(f"  <- perf: IPC={result_dict.get('ipc')}")
    else:
        print(f"  <- {fn_name}: {str(result_dict)[:120]}")


def _force_final_score(
    tools: SIMDTools,
    problem: dict,
    baselines: dict,
) -> EvalResult:
    """Force a final scoring run when the agent hits max_turns without submitting."""
    if not tools._last_compile_ok: # check if last compiled code succeed
        return EvalResult(correct=False, level=0, tool_calls=tools._tool_calls)

    rr = tools.run(n=10)
    if not rr.correct: # check if last compiled code is current
        return EvalResult(correct=False, level=0, tool_calls=tools._tool_calls)

    pid = problem["id"]
    baseline = baselines.get(pid, {})
    baseline_ms = baseline.get("baseline_ms")

    speedup_vs_ref = None
    level = 1  # correct
    if rr.candidate_runtime_ms and baseline_ms and rr.candidate_runtime_ms > 0:
        # Normalize to per-iteration (run returns total for n iterations)
        candidate_per_iter = rr.candidate_runtime_ms / 10
        speedup_vs_ref = round(baseline_ms / candidate_per_iter, 2)
        if speedup_vs_ref > 1.0:
            level = 2

    return EvalResult(
        correct=True,
        speedup_vs_ref=speedup_vs_ref,
        level=level,
        runtime_ms=rr.candidate_runtime_ms,
        tool_calls=tools._tool_calls,
    )
