# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**CPU-Kernel-Baseline** evaluates LLMs on their ability to write optimized AArch64
SIMD kernels for ncnn / llama.cpp / synthetic simd-loop benchmarks. Three evaluation
paths, all built on the same `bench/` harness, the same `bench-trace/` warehouse,
and — as of the eval/mcp_client.py migration — the same `mcp_app/server.py` tool
execution surface for every agent-driven path:

- **Path 1 — In-repo litellm agent loop** (`eval/`): a self-contained litellm
  agent loop (`eval/evaluator.py::run_agentic_eval`) drives compile/evaluate/
  disassemble/submit as an MCP client of `mcp_app/server.py` (`eval/mcp_client.py`),
  started on a provisioned Graviton instance over an SSH-tunneled MCP session —
  the same server nanobot/Claude Code drive in Path 3, just with this repo's
  own litellm loop as the client instead of an external harness. Driven via
  `test_scripts/bench_fleet.py --harness own` — the same batch-fleet entry
  point Path 3's harnesses use (`OwnHarnessAdapter` in
  `test_scripts/harness_adapters.py` calls `run_agentic_eval` in-process,
  no external CLI subprocess). (Previously had its own independent
  SSH-based tool system, `eval/agent_tools/`, then its own standalone CLI,
  `eval/run_benchmark.py` — both retired once bench_fleet.py covered the
  same ground; see `eval/mcp_client.py`'s module docstring.)
- **Path 2 — Local `bench/` harness**: compiles solutions into `.so` files with
  clang++ locally, dlopens them, runs correctness + timing without SSH or any
  agent loop. Works on any machine; produces real SVE2 numbers on Graviton.
  Also the library every other path calls into.
- **Path 3 — MCP-server-driven eval** (`mcp_app/` + `skills/`): the same
  compile/evaluate/disassemble/submit surface exposed as an MCP server
  (`mcp_app/server.py`) so an external agent harness (nanobot, Claude Code)
  drives the session instead of an in-repo agent loop. `skills/launch/`
  provisions an instance and starts the server; `skills/<harness>/` holds
  that harness's own `SKILL.md`.

Top-level framework directories (`ncnn/`, `ggml/`, `vllm/`, `paddleLite/`, `tnn/`)
are read-only reference baselines. `ncnn/` is NOT in this repo — clone separately
for ncnn baseline builds (see below).

---

## System architecture: key internals

### Build pipeline: two paths, one builder registry

`bench/compile/registry.py` has a `BuilderRegistry` singleton that dispatches
based on `is_baseline` (whether `solution.author == baseline_author`):

- **`CandidateBuilder`** (`bench/compile/builders/candidate.py`): the raw `float*`
  ABI path for ALL non-baseline solutions. No ncnn dependency. The solution's
  own sources define `armbench_entry_<op>` — the only shared code is what ships
  in the solution JSON. Uses `-O3 -march=armv8.2-a+sve -std=c++14` for SVE.
- **`NcnnBuilder`** (`bench/compile/builders/ncnn.py`): baseline-only for ncnn
  dataset. Links against real `libncnn.a` from `ncnn/build/src/`. Each baseline
  solution ships its own `binding.cpp` (defines `armbench_entry_<op>` with
  constexpr-baked params and `ncnn::Mat`/`Option` shim) + `kernel.cpp` (delegates
  to ncnn's kernel layer) + a contract header.
- **`SimdLoopBuilder`** / **`LlamaCppBuilder`**: dataset-specific paths for simd-loop
  and llama.cpp solutions. Self-contained solutions (harness sources fused into
  solution JSON).

Both GCC (`g++-13`, `g++-14`, `g++`) and clang (`clang++-18`, `clang++`, `clang++-17`)
are detected at build time (see `_resolve_cxx` in `bench/compile/builder.py`).

### Candidate kernel ABI (raw float* path)

For candidate (LLM-generated) kernels in the ncnn dataset, the contract is:

```c
// conv2d_depthwise.h (per-op header, embedded in solution JSON)
void inner_conv2d_depthwise(
    const float* input, float* output,
    const float* weight, const float* bias,
    int N, int C, int H, int W, int H_out, int W_out);

// binding.cpp (in solution JSON) computes H_out/W_out and calls inner_...:
int armbench_entry_conv2d_depthwise(
    const float* input, float* output,
    const float* weight, const float* bias,
    int N, int C, int H, int W);
```

Key constraints for candidate kernels:
- Must be `extern "C"` (not `static`).
- **No OpenMP** — `#pragma omp` and any threading APIs are detected by source-pattern
  scanning in `bench/config.py` (`DEFAULT_DISALLOWED_SOURCE_PATTERNS`) and will be
  rejected by the evaluator.
- `Kernel.cpp` is the expected file path for the LLM's implementation. The entry
  symbol name matches the op type (e.g., `inner_conv2d_depthwise` for conv2d_depthwise).
- Per-definition constants (`Kh`, `Kw`, `Sh`, `Sw`, `Dh`, `Dw`, `pad`) are baked
  into a namespace (`conv2d_depthwise_def`) in the solution's `.h` file.

### Evaluation pipeline

`bench/runner.py::run_solution_on_workloads` runs inside an **isolated subprocess**
(`bench/runtime/isolation.py`) — a hanging or crashing kernel can't take the
runner/server down. The subprocess:

1. Compiles via `BuilderRegistry` → `.so`
2. Dlopens and binds `armbench_entry_<op>` via ctypes
3. Iterates workloads: for each, calls `bench/evaluators/default.py::DefaultEvaluator`
   which runs correctness (hybrid abs+rel tolerance AND comparison) + timing
   (CPU-pinned via `os.sched_setaffinity`, hardware perf counters via
   `perf_event_open`, `min_ns` + `p5_ns` as the metrics)
4. Reports speedup factors vs the competitive baseline

Timing (`bench/runtime/timing.py`):
- Default: warmup=5, repeat=50, inner_iters=1, watchdog=30s
- `min_ns` is the canonical metric (one per workload), `p5_ns` as jitter proxy
- Hardware perf counters: CYCLES, INSTRUCTIONS, CACHE_MISSES via Linux
  `perf_event_open(2)` — best-effort (silently skips where unavailable)
- CPU pinning on Linux via `os.sched_setaffinity`

Correctness (`bench/runtime/correctness.py`):
- An element FAILS only if BOTH `|got-ref| > abs_tol` AND `|got-ref|/|ref| > rel_tol`
  (AND condition — more lenient than old `diff > abs_tol + rel_tol * |ref|`)
- Default tolerances: abs=1e-3, rel=1e-3, required_matched_ratio=1.0
- Per-op-type overrides in `config/kernel_contracts.yaml` (gemm: 2e-3/1e-2/0.98,
  moe: 1e-2/5e-2/0.95, mha: same as default, gqa/mla refine: custom)

### MCP server internals (`mcp_app/`)

`mcp_app/server.py` starts an MCP server that exposes `compile`/`evaluate`/
`disassemble`/`submit` as MCP tools. Key design:

- **Zero coupling to `eval/` or `skills/`**: never provisions instances, never
  imports from either. Uses `contracts.py` from repo root for shared naming.
- **Long-lived process**: one server serves every definition in `--dataset`.
  `compile()` takes `definition` string as per-call argument.
- **`KernelSession` ABC** (`mcp_app/agent_tools/base.py`): subclasses per dataset
  (`ncnn.py`, `simd_loop.py`, `llama_cpp.py`). `DispatcherKernelSession`
  (`dispatcher.py`) routes across multiple datasets when a run spans them.
- **Lazy baseline collection** (`baseline_readiness.py`): first `compile()` call
  triggers baseline check-then-collect for that definition.
- **Isolated evaluation**: `evaluate_kernel()` in `ops.py` wraps everything in
  `run_in_subprocess` (timeout=750s, under the MCP client's 900s tool timeout).
- ISA verification: `verify_isa_available()` checks `/proc/cpuinfo` at startup.
  `march_for_isa()` is the compile-flag authority.
- Multi-dataset routing: `DispatcherKernelSession` wraps multiple per-dataset
  sessions and routes `compile`/`evaluate` by looking up which dataset owns
  the given definition name.
- MCP Resources: `resources.py` exposes each version's source/disassembly/
  trajectory as scoped resources under the `run_dir`. `trajectory.py` maintains
  a live append-only audit trail per definition.
- Transport: supports both `stdio` (local) and `streamable-http` (remote, over
  SSH local-port-forward).

### Workloads and definitions

Each definition JSON (`bench-trace/definitions/<op_type>/<name>.json`) specifies:
- `axes`: const (fixed values like Kh=5, Sh=2) or var (per-workload like C, H, W)
- `inputs`/`outputs`: tensor specs with shape expressions referencing axes
- `constraints`: formulas like H_out = (H + 2*pad - Dh*(Kh-1) - 1) / Sh + 1
- `reference`: Python reference implementation (usually torch/numpy, run as subprocess)

Each definition has a set of workloads (`bench-trace/workloads/<op_type>/<name>.jsonl`),
one JSON object per line with concrete axis values and input generation tags
(e.g., `"type": "random"`).

---

## Repository layout

```
bench/                          # Local harness Python package
  compile/
    builder.py                  # Builder ABC + CompileError/CompileResult
    registry.py                 # BuilderRegistry singleton, build cache (hash-keyed)
    builders/
      candidate.py              # CandidateBuilder: raw float* for LLM candidates
      ncnn.py                   # NcnnBuilder: ncnn::Mat for ncnn baselines (links libncnn.a)
      simd_loop.py              # SimdLoopBuilder: from solution sources
      llama_cpp.py              # LlamaCppBuilder: ggml static libs
      candidate_harness/        # C shims for candidate kernels
      ncnn_harness/             # C shims for ncnn baseline kernels
      simd_loop_harness/        # Legacy on-disk copies (fallback only)
  datasets/
    ncnn.py                     # NcnnDataset adapter
    raw.py                      # RawDataset for candidates
    simd_loop.py                # SimdLoopDataset
    llama_cpp.py                # LlamaCppDataset
  evaluators/
    evaluator.py                # Evaluator ABC + BoundKernel (carries Definition)
    default.py                  # DefaultEvaluator (correctness + timing + speedup)
    sqnr.py                     # SqnrEvaluator (for quantized/q8_0 MoE defs)
    registry.py                 # resolve_evaluator() — first-match dispatch
  runner.py                     # compile-once → BoundKernel → evaluator per workload
  benchmark.py                  # Benchmark orchestration (build lifecycle, candidate-vs-baseline)
  cli.py                        # CLI entry points (bench, list-definitions, list-solutions)
  config.py                     # BenchmarkConfig + EvalConfig (tolerances, source patterns)
  data/
    definition.py               # Definition + SimdLoopMeta + DType (incl. unsigned)
    solution.py                 # Solution, SourceFile, SolutionSpec, SupportedDatasets
    trace.py                    # Trace, Evaluation, Environment, EvaluationStatus
    trace_set.py                # TraceSet — in-memory warehouse (load/query/persist)
    workload.py                 # Workload + Axes
    utils.py                    # BaseModelWithDocstrings helpers
  runtime/
    inputs.py                   # Deterministic input generators
    timing.py                   # ns timing with CPU pinning + perf counters
    correctness.py              # Hybrid abs+rel tolerance AND comparison
    isolation.py                # Subprocess isolation (timeout, crash containment)
    perf_counters.py            # Linux perf_event_open wrapper

bench-trace/                    # On-disk warehouse (TraceSet root) — .gitignored
  definitions/<op_type>/
  workloads/<op_type>/
  solutions/<dataset>/<author>/<op_type>/
  traces/<op_type>/

eval/                           # In-repo litellm agent loop (Path 1)
  provision.py                  # Terraform lifecycle for Graviton EC2 instances
  evaluator.py                  # run_agentic_eval turn loop (prompts, retries, history compression)
  mcp_client.py                 # MCP client bridge — attach()es to an already-running
                                 #   mcp_app/server.py session (compile/evaluate/disassemble/submit),
                                 #   same server Path 3 drives; provisioning/tunnel lifecycle is the
                                 #   caller's (test_scripts/bench_fleet.py's) job, not this module's
  remote.py                     # InstanceHandle — SSH/rsync to a provisioned instance
  eval_config.json              # SSH connection info — copy from .example

mcp_app/                        # MCP server for Path 3
  server.py                     # MCP server (--transport stdio|streamable-http)
  session.py                    # SessionConfig + build_tools() bootstrap
  resources.py                  # MCP Resources over run_dir (source/disasm/trajectory)
  agent_tools/
    base.py                     # KernelSession ABC
    dispatcher.py               # DispatcherKernelSession (multi-dataset routing)
    ncnn.py, simd_loop.py, llama_cpp.py
    ops.py                      # compile_kernel/evaluate_kernel/disassemble_so
    isa.py                      # march_for_isa() + verify_isa_available()
    trajectory.py               # per-definition audit trail writer
    baseline_readiness.py       # Lazy baseline check-then-collect
    registry.py                 # resolve_tools(dataset) -> Type[KernelSession]

skills/                         # Harness-agnostic session launch
  launch/launch_session.py      # provision/prepare-session/sync-results/status/teardown
  nanobot/nanobot-kernel-session/
    SKILL.md, README.md

test_scripts/                   # Batch-fleet entry point, shared across all three paths
  bench_fleet.py                 # `--harness {claude-code,nanobot,own}` — compute author/label
                                 #   once, provision, prepare_session, retry/log/sync loop over
                                 #   every definition matching --dataset
  harness_adapters.py           # Per-harness HarnessAdapter (ClaudeCodeAdapter/NanobotAdapter/
                                 #   OwnHarnessAdapter) — what's genuinely harness-specific: how
                                 #   each is invoked, how it connects to the MCP endpoint

scripts/
  gen_definitions.py            # Regenerate ncnn definitions+workloads from test files
  gen_simd_loop_harness.py      # Code-gen all simd-loop harnesses + bench-trace artifacts
  bench_loop_agent.py           # Local iterative LLM agent (Path 2)
  test_reference_scalars.py     # Correctness smoke-test for all reference-scalar solutions

case-study/                     # Per-definition optimization write-ups
terraform/                      # Graviton EC2 Terraform config
config/kernel_contracts.yaml    # Single source of truth: ISA mappings, evaluator defaults,
                                #   baseline/reference-scalar authors per dataset
```

---

## Key commands

### Provision & teardown
```bash
python eval/provision.py --isa sve2       # Graviton4 c8g.large (SVE2=128-bit)
python eval/provision.py --isa sve        # Graviton3 c7g.large (SVE=256-bit)
python eval/provision.py --teardown
```

### In-repo litellm agent loop (Path 1 — requires Graviton instance)
```bash
python3 test_scripts/bench_fleet.py --harness own \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-6
# Narrow to specific definitions (space-separated or a JSON array); empty --definitions = every
# definition matching --dataset
python3 test_scripts/bench_fleet.py --harness own \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-6 --definitions loop_001
```

### Local iterative LLM agent (Path 2 — run on target machine for real timing)
```bash
python scripts/bench_loop_agent.py --loop loop_001 --max-turns 6 --model openrouter/anthropic/claude-opus-4-6
```

### Validate via CLI (bench/cli.py)
```bash
python -m bench.cli list-definitions          # list all definitions
python -m bench.cli bench --definition loop_001 --solution reference-scalar_loop_001
python -m bench.cli bench --definition conv2d_depthwise_fp32_kh5_kw5_sh2_sw2_dh1_dw1_p2 \
  --solution reference-scalar_conv2d_depthwise_fp32_kh5_kw5_sh2_sw2_dh1_dw1_p2
```

### Correctness smoke-test
```bash
python scripts/test_reference_scalars.py      # should print all workloads PASSED
```

### MCP session (Path 3)
```bash
# MCP server over SSH stdio (connect Claude Code MCP config to it)
python3 -m mcp_app.server --dataset ncnn --author claude-code --isa sve \
  --run-dir ~/arm-bench/agent-runs-mcp/claude-code --transport stdio \
  --baseline-author baseline-ncnn-arm

# Launch session (skills/launch/)
python3 launch_session.py launch --isa sve --dataset ncnn
```

### Regenerate definitions/workloads
```bash
python scripts/gen_simd_loop_harness.py       # simd-loop (idempotent)
python scripts/gen_definitions.py             # ncnn definitions + workloads
```

### Sync local code to remote instance
```bash
./sync_remote.sh                              # rsync the repo
./sync_remote.sh --mirror                     # force mirror (rm remote, then copy)
HOST=1.2.3.4 ./sync_remote.sh                 # different instance
```

---

## ISA targets and instance types

| ISA | Instance | March flag | Notes |
|-----|----------|------------|-------|
| NEON | c7g.large | `-march=armv8-a` | 128-bit NEON only |
| SVE | c7g.large | `-march=armv8.2-a+sve` | Graviton3, Neoverse V1, 256-bit SVE |
| SVE2 | c8g.large | `-march=armv9-a+sve2` | Graviton4, Neoverse V2, 128-bit SVE2 |
| SME2 | — | `-march=armv9-a+sve2+sme2` | No AWS instance supports SME2 yet |

---

## Important constraints

1. **No OpenMP in candidate kernels** — `#pragma omp`, `omp.h`, `std::thread`,
   `pthread.h`, `fork`, etc. are all blocked by source-pattern scanning in
   `bench/config.py::DEFAULT_DISALLOWED_SOURCE_PATTERNS`. Rejected at eval time.
2. **Single-core timing** — `num_threads=1` assumed. The timer uses CPU pinning
   and does not support multi-threaded timing.
3. **Candidate kernel entry points must be `extern "C"`** — the dlopen/cffi
   binding looks them up by mangled name.
4. **Solution JSONs are self-contained** — each embeds its own harness sources
   (`.h` + `.cpp` + `kernel.cpp`). The builder materializes them to a temp dir
   and compiles them together.
5. **`ncnn/` is NOT in this repo** — must be cloned and built separately for
   ncnn baseline evaluation (see ncnn builder section above).
6. **Subprocess isolation** — every evaluation spawns a child process for crash
   protection. Timeout defaults to 750s in MCP mode, configurable per eval.
7. **ISA mismatch at MCP startup** — `verify_isa_available()` reads
   `/proc/cpuinfo` and rejects an ISA the hardware doesn't support (safety
   check, not compile-flag authority — that's `march_for_isa()`).
