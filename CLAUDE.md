# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

**CPU-Kernel-Baseline** evaluates LLMs on their ability to write optimized AArch64
SIMD kernels for ncnn / llama.cpp / synthetic simd-loop benchmarks. Three evaluation
paths, all built on the same `bench/` harness and the same `bench-trace/` warehouse:

- **Path 1 — Agentic SSH eval** (`eval/`): a self-contained litellm agent loop
  (`eval/run_benchmark.py`) drives compile/evaluate/disassemble/submit tools
  (`eval/agent_tools/`) over SSH against a provisioned Graviton instance.
- **Path 2 — Local `bench/` harness**: compiles solutions into `.so` files with
  clang++ locally, dlopens them, runs correctness + timing without SSH or any
  agent loop. Works on any machine; produces real SVE2 numbers on Graviton.
  Also the library every other path calls into.
- **Path 3 — MCP-server-driven eval** (`mcp_app/` + `skills/`): the same
  compile/evaluate/disassemble/submit surface exposed as an MCP server
  (`mcp_app/server.py`) so an external agent harness (nanobot today; Claude
  Code/Gemini CLI planned) drives the session instead of an in-repo agent
  loop. `skills/launch/` provisions an instance and starts the server;
  `skills/<harness>/` holds that harness's own `SKILL.md`.

Top-level framework directories (`ncnn/`, `ggml/`, `vllm/`, `paddleLite/`, `tnn/`)
are read-only reference baselines. `ncnn/` is NOT in this repo — clone separately
for ncnn baseline builds (see below).

---

## System walkthrough

**Shared warehouse (`bench-trace/`)**: every path reads/writes the same on-disk
store — `definitions/<op_type>/*.json` (op shape + `simd_loop_meta`),
`workloads/<op_type>/*.jsonl` (concrete input shapes, append-only),
`solutions/<dataset>/<author>/<op_type>/*.json` (kernel source + metadata),
`traces/<op_type>/*.json` (correctness/timing results). `bench/runner.py`
compiles a solution once via `BuilderRegistry`, binds it into a `BoundKernel`,
and hands it to an `Evaluator` per workload.

**Path 1 flow**: `run_benchmark.py` → `eval.evaluator.run_agentic_eval` → a
litellm agent loop calling `eval/agent_tools/*` (compile/evaluate/disassemble/
submit) over SSH → results land in `bench-trace/` and `results/`/`traces/`.

**Path 3 flow**: `skills/launch/launch_session.py` provisions/reaches an
instance via `eval/provision.py` (called only as a subprocess, never
imported — both sides share the same `eval/eval_config.json` "what's up"
record), rsyncs the repo, builds the dataset's native lib, then starts
`mcp_app.server` in streamable-http mode over an SSH local-port-forward. The
external harness (e.g. nanobot, via
`skills/nanobot/nanobot-kernel-session/SKILL.md`) connects and calls
`compile`/`evaluate`/`disassemble`/`submit`. `mcp_app.session` builds one
`KernelSession` per dataset (`mcp_app/agent_tools/{ncnn,simd_loop,llama_cpp}.py`,
all subclassing the `KernelSession` ABC in `base.py`); when a run spans
multiple datasets, `DispatcherKernelSession` (`dispatcher.py`) wraps them and
routes each call by looking up which dataset owns the given `definition`.
`evaluate` runs in an isolated subprocess (`bench/runtime/isolation.py`) so
one crashing kernel can't take down the whole server. Every definition's
`reference-scalar-kernel.cpp` is written out as an MCP Resource at startup
(`session.py`); `resources.py` exposes each version's source/disassembly/
trajectory as further resources, scoped per `run_dir`. `trajectory.py` writes
a live, append-only turn-by-turn audit trail per definition. After the run,
`launch_session.py sync-results` pulls results back to the local checkout.

**Other directories**: `dataset/problems/` — 75 raw simd-loop problem specs,
the source `scripts/gen_simd_loop_harness.py` reads from; `case-study/` —
human-readable write-ups of one definition's optimization trajectory
(produced by the `case-study` skill); `terraform/` — the Graviton EC2 config
`eval/provision.py` drives; `agent-runs*/` — historical run artifacts per
path, gitignored.

---

## Repository layout (after PR #15 flattening)

```
bench/                          # Local harness Python package
  compile/
    builder.py                  # Builder ABC + CompileResult
    registry.py                 # BuilderRegistry (singleton, build cache)
    builders/
      candidate.py              # CandidateBuilder: raw float* for LLM candidates
      ncnn.py                   # NcnnBuilder: ncnn::Mat for ncnn baselines
      simd_loop.py              # SimdLoopBuilder: reads harness from solution sources
      candidate_harness/        # C shims for candidate kernels
      ncnn_harness/             # C shims for ncnn baseline kernels
      simd_loop_harness/        # Legacy on-disk copies (fallback only; fused into solution JSON)
  datasets/
    ncnn.py                     # NcnnDataset adapter
    raw.py                      # RawDataset for candidates
    simd_loop.py                # SimdLoopDataset — derives all metadata from Definition
  evaluators/
    evaluator.py                # BoundKernel (carries Definition), RefBaseline, Evaluator ABC
    default.py                  # DefaultEvaluator (correctness + timing + speedup)
  runner.py                     # compile-once → BoundKernel → evaluator per workload
  data/
    definition.py               # Definition + SimdLoopMeta + DType (incl. unsigned)
    solution.py                 # Solution, SourceFile, SupportedDatasets
    ...                         # Workload, Trace, TraceSet Pydantic schemas
  runtime/
    inputs.py                   # Deterministic input generators
    timing.py                   # ns timing with CPU pinning + perf counters
    correctness.py              # Hybrid abs+rel tolerance comparison

bench-trace/                    # On-disk warehouse (TraceSet root) — gitignored, generated
  definitions/<op_type>/        # Definition JSONs (include simd_loop_meta)
  workloads/<op_type>/          # Workload JSONLs (append-only)
  solutions/<dataset>/<author>/<op_type>/
  traces/<op_type>/

eval/                           # Agentic SSH eval (Path 1) — also owns provisioning for Path 3
  provision.py                  # Terraform lifecycle for Graviton EC2 instances
  run_benchmark.py              # LLM agent loop (SSH path)
  agent_tools/                  # compile/evaluate/disassemble/submit tools for the SSH litellm loop
  eval_config.json              # SSH connection info — copy from .example; shared "what's up" record

mcp_app/                        # MCP server for Path 3 (nanobot etc.) — no imports from eval/ or skills/
  server.py                     # the MCP server process (--transport stdio|streamable-http)
  session.py                    # SessionConfig + build_tools() — server-side bootstrap
  resources.py                  # MCP Resources over a session's run_dir (source/disasm/trajectory)
  agent_tools/
    base.py                     # KernelSession ABC — compile/evaluate/disassemble/submit
    dispatcher.py                # DispatcherKernelSession — routes calls across multiple datasets
    baseline_readiness.py       # lazy per-definition baseline check-then-collect
    ncnn.py, simd_loop.py, llama_cpp.py   # per-dataset KernelSession subclasses
    trajectory.py                # per-definition audit trail writer

skills/                         # Harness-agnostic session launch + per-harness SKILL.md files
  launch/launch_session.py      # provision/prepare-session/sync-results/status/teardown (Path 3)
  nanobot/nanobot-kernel-session/SKILL.md   # nanobot's own optimization workflow doc

scripts/
  gen_definitions.py            # Regenerate ncnn definitions+workloads from test files
  gen_simd_loop_harness.py      # Code-gen all simd-loop harnesses + bench-trace artifacts
  bench_loop_agent.py           # Local iterative LLM agent (Path 2, works locally + on Graviton)
  test_reference_scalars.py     # Correctness smoke-test: all reference-scalar solutions vs Python ref

dataset/problems/               # 75 raw simd-loop problem specs (source for gen_simd_loop_harness.py)
case-study/                     # Per-definition optimization write-ups (case-study skill output)
terraform/                      # Graviton EC2 Terraform config, driven by eval/provision.py
```

---

## Key commands

### Provision & teardown
```bash
python eval/provision.py --isa sve2       # Graviton4 c8g.large
python eval/provision.py --teardown
```

### Agentic SSH eval (Path 1 — requires Graviton)
```bash
python -m eval.test_workflow --isa sve2
python eval/run_benchmark.py --problem loop_001 --isa sve2 --model anthropic/claude-opus-4-6
./sync_remote.sh && python eval/run_benchmark.py --problem conv --mode ncnn --isa sve2 --model anthropic/claude-opus-4-6
```

### Local iterative LLM agent (Path 2)
```bash
# Run on Mac for NEON dev/correctness, or on Graviton for real SVE2 timing
OPENROUTER_API_KEY=sk-or-... python scripts/bench_loop_agent.py --loop loop_001
python scripts/bench_loop_agent.py --all-loops --max-turns 4 \
  --model openrouter/anthropic/claude-opus-4-6
```

### Validate via CLI (bench/cli.py)
```bash
python -m bench.cli list-definitions          # list all 20 simd-loop + ncnn definitions
python -m bench.cli bench --definition loop_001 --solution reference-scalar_loop_001
```

### Correctness smoke-test (all 20 reference-scalar solutions)
```bash
python scripts/test_reference_scalars.py      # should print 121/121 workloads passed
```

### Regenerate simd-loop harnesses + bench-trace
```bash
python scripts/gen_simd_loop_harness.py       # idempotent; only writes on content change
```

### Regenerate ncnn definitions
```bash
python scripts/gen_definitions.py   # → bench-trace/definitions/ and bench-trace/workloads/
```

---

## bench/ harness — two datasets

### simd-loop (fully tested on Graviton4 SVE2)

**20 loops** across three patterns (121/121 workload correctness, Mac + Graviton4):
- **Scalar-output**: 001-004, 008, 010, 024, 032, 033, 126, 127 — reduction → single value
- **Array-output**: 027, 028, 029, 035, 113, 128 — element-wise → N-element output
- **Inplace-sort**: 120, 121, 122 — data sorted in-place

**75 total loops in `dataset/problems/`; 55 not yet integrated:**

| Reason skipped | Count | Examples |
|----------------|-------|---------|
| SME2/MOPA matmul — no AWS instance yet | ~25 | 200-series, 130, 135-137 |
| Non-trivial multi-ptr structs (linked list, sparse, indirect) | ~15 | 009, 019, 023, 036, 102, 104 |
| Multi-axis matmul needing m/n/k | ~5 | 025 |
| String/char* ops | ~5 | 005, 006, 022, 031, 034 |
| Complex C types (cuint32_t etc.) | ~5 | 037, 109, 110, 112 |
| Scalar-only struct (no array ptr) | ~2 | 040, 012 |

All 75 loops have SVE/NEON implementations — the blockers are ABI complexity, not ISA.

**To add more loops**: add the loop ID to `TARGET_LOOP_IDS` in `scripts/gen_simd_loop_harness.py`
and run it. The generator handles the three patterns automatically. Non-standard cases:
- Sort/inplace: add to `_SORT_LOOPS` dict (lists scratch field names)
- Custom kernel: add to `_CUSTOM_SCALAR_KERNELS` (e.g. loop_001 uses double accumulation)
- Non-trivial reference: add to `_CUSTOM_REFS`
- Array padding: add to `_LOOP_META_OVERRIDES` (e.g. `"array_pad": 2` for loop_113)

**Architecture (self-contained solutions):**
- Each solution JSON embeds its harness sources (`loop_NNN.h`, `loop_NNN.cpp`, `kernel.cpp`)
- `SimdLoopBuilder` compiles directly from solution sources — no separate harness directory needed
- `SimdLoopDataset` derives all adapter metadata from `Definition.simd_loop_meta` (written by
  the generator into each definition JSON) — no hard-coded `_LOOP_META` or `SIGNATURES` dicts
- `DType` enum includes unsigned types (uint8/16/32/64) needed for integer accumulation loops
- `bench/cli.py` works for all simd-loop solutions without setting `is_baseline`

**Candidate convention:** `inner_loop_NNN` must be `extern "C"` (not `static`).

**Graviton4 results (clang++-18, -O3 -march=armv9-a+sve2):**
- Claude Opus generates correct SVE2 on turn 1 for all 4 loops
- Iterative agent: best timing stable within 4 turns (~552–561 ns small N, ~1590–1620 ns N=10K)
- Self-corrects mixed SVE+NEON compile errors on the next turn

### ncnn (fully tested on Graviton4 — compile + link verified for all 5 op_types)

Conv2d (existing) + conv1d, conv2d_depthwise, deconv2d, deconv2d_depthwise (new).
114 total definitions across all 5 op_types.

- `NcnnBuilder` — links against `<repo_root>/ncnn/build/src/libncnn.a`
- ncnn/ is NOT in this repo — clone and build it first:

```bash
sudo apt-get install clang-18 libomp-18-dev cmake   # libomp required for linking
git clone --depth=1 https://github.com/Tencent/ncnn.git ncnn
cd ncnn && cmake -B build \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
cmake --build build -j$(nproc) ncnn
```

---

## Adding a new simd-loop problem

The generator handles everything — just run it after editing `TARGET_LOOP_IDS`:

```bash
python scripts/gen_simd_loop_harness.py
```

This writes (idempotently):
1. `bench/compile/builders/simd_loop_harness/loop_NNN.{h,cpp}` — on-disk copies (legacy fallback)
2. `bench-trace/definitions/simd-loop/loop_NNN.json` — includes `simd_loop_meta` for the adapter
3. `bench-trace/workloads/simd-loop/loop_NNN.jsonl`
4. `bench-trace/solutions/simd-loop/reference-scalar/loop_NNN/reference-scalar_loop_NNN.json`
   — sources include fused `loop_NNN.h` + `loop_NNN.cpp` + `kernel.cpp`

No Python files need manual editing to add a supported loop pattern.

## Instance types

| ISA  | Instance  | Notes                                |
|------|-----------|--------------------------------------|
| SVE  | c7g.large | Graviton3, Neoverse V1, 256-bit SVE  |
| SVE2 | c8g.large | Graviton4, Neoverse V2, 128-bit SVE2 |
| SME2 | —         | No AWS instance supports SME2 yet    |
