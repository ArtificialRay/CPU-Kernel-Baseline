# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

**CPU-Kernel-Baseline** evaluates LLMs on their ability to write optimized AArch64
SIMD kernels. Two evaluation paths:

- **Path 1 — Agentic SSH eval** (`eval/`): LLM gets tools (compile, run, perf,
  disassemble, submit) over SSH on a provisioned Graviton instance. Driven by
  `eval/run_benchmark.py`.
- **Path 2 — Local `bench/` harness**: Compiles solutions into `.so` files using
  clang++ locally, dlopens them, runs correctness + timing without SSH. Works on
  any machine; produces real SVE2 numbers when run on Graviton.

Top-level framework directories (`ncnn/`, `ggml/`, `vllm/`, `paddleLite/`, `tnn/`)
are read-only reference baselines. `ncnn/` is NOT in this repo — clone separately
for ncnn baseline builds (see below).

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
      simd_loop.py              # SimdLoopBuilder: flat-array for loop_* baselines
      candidate_harness/        # C shims for candidate kernels
      ncnn_harness/             # C shims for ncnn baseline kernels
      simd_loop_harness/        # C shims for simd-loop baseline kernels
  datasets/
    ncnn.py                     # NcnnDataset adapter + SIGNATURES
    raw.py                      # RawDataset for candidates
    simd_loop.py                # SimdLoopDataset + SIGNATURES
  evaluators/
    evaluator.py                # BoundKernel, RefBaseline, Evaluator ABC
    default.py                  # DefaultEvaluator (correctness + timing + speedup)
  runner.py                     # compile-once → BoundKernel → evaluator per workload
  data/                         # Pydantic schemas: Definition, Solution, Workload, Trace
  runtime/
    inputs.py                   # Deterministic input generators
    timing.py                   # ns timing with CPU pinning + perf counters
    correctness.py              # Hybrid abs+rel tolerance comparison

bench-trace/                    # On-disk warehouse (TraceSet root)
  definitions/<op_type>/        # Definition JSONs
  workloads/<op_type>/          # Workload JSONLs (append-only)
  solutions/<dataset>/<author>/<op_type>/
  traces/<op_type>/

eval/                           # Agentic SSH eval (Path 1)
  provision.py                  # Terraform lifecycle for Graviton EC2 instances
  run_benchmark.py              # LLM agent loop (SSH path)
  eval_config.json              # SSH connection info — copy from .example

scripts/
  gen_definitions.py            # Regenerate ncnn definitions+workloads from test files
  gen_simd_loop_harness.py      # Code-gen all simd-loop harnesses + bench-trace artifacts
  bench_loop_agent.py           # Local iterative LLM agent (Path 2, works locally + on Graviton)
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

### Regenerate ncnn definitions
```bash
python scripts/gen_definitions.py   # → bench-trace/definitions/ and bench-trace/workloads/
```

---

## bench/ harness — two datasets

### simd-loop (fully tested on Graviton4 SVE2)

**20 loops** across three patterns (80/80 edge-workload correctness on Mac):
- **Scalar-output**: 001-004, 008, 010, 024, 032, 033, 126, 127 — reduction → single value
- **Array-output**: 027, 028, 029, 035, 113, 128 — element-wise → N-element output
- **Inplace-sort**: 120, 121, 122 — data array sorted in-place

**To add more loops**: edit `TARGET_LOOP_IDS` in `scripts/gen_simd_loop_harness.py` and run it. New patterns may require adding a `_SORT_LOOPS` entry (inplace) or a custom ref/kernel.

Both local dev and Graviton use the same code path; platform detected at runtime for ISA-appropriate prompting.

- `SimdLoopBuilder` — no framework deps, just clang++ + harness shim
- Harness: `bench/compile/builders/simd_loop_harness/loop_NNN.{h,cpp}`
- Adapter: `bench/datasets/simd_loop.py`
- Data: `bench-trace/definitions/simd-loop/` + `bench-trace/workloads/simd-loop/`

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

1. `bench/compile/builders/simd_loop_harness/<op>.{h,cpp}` — struct + entry shim
2. `bench/datasets/simd_loop.py` — add SIGNATURES entry + _RESULT_DTYPE
3. `bench-trace/definitions/simd-loop/<op>.json` + `bench-trace/workloads/simd-loop/<op>.jsonl`
4. `_scalar_args_for` in `bench/evaluators/default.py` picks it up automatically

## Instance types

| ISA  | Instance  | Notes                                |
|------|-----------|--------------------------------------|
| SVE  | c7g.large | Graviton3, Neoverse V1, 256-bit SVE  |
| SVE2 | c8g.large | Graviton4, Neoverse V2, 128-bit SVE2 |
| SME2 | —         | No AWS instance supports SME2 yet    |

## In-flight PRs (as of 2026-06-05)

| Branch                    | Status         | Notes                                       |
|---------------------------|----------------|---------------------------------------------|
| `feat/simd-loop-harness`  | Ready to merge | PR 1 — simd-loop harness, gitignore         |
| `feat/ncnn-4ops-harness`  | Ready to merge | PR 2 — stacked on PR 1, ncnn 4ops + agent  |
| `feat/collect-ncnn-*`     | Stale (pre-#15)| Superseded by PR 2                          |
| `chore/small-refactor-*`  | Stale (pre-#15)| Superseded by PR #15                        |
