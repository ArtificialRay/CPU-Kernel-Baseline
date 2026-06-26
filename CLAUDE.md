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

eval/                           # Agentic SSH eval (Path 1)
  provision.py                  # Terraform lifecycle for Graviton EC2 instances
  run_benchmark.py              # LLM agent loop (SSH path)
  eval_config.json              # SSH connection info — copy from .example

scripts/
  gen_definitions.py            # Regenerate ncnn definitions+workloads from test files
  gen_simd_loop_harness.py      # Code-gen all simd-loop harnesses + bench-trace artifacts
  bench_loop_agent.py           # Local iterative LLM agent (Path 2, works locally + on Graviton)
  test_reference_scalars.py     # Correctness smoke-test: reference/autovec baseline solutions vs Python ref
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
python -m bench.cli list-definitions          # list all simd-loop + ncnn definitions
python -m bench.cli bench --definition loop_001 --solution reference_loop_001
python -m bench.cli bench --definition loop_001 --solution autovec_loop_001
```

Each simd-loop has two baseline solutions from the same kernel source:
`reference` (scalar, `-fno-vectorize`) and `autovec` (`-O3 -march=native`).

### Correctness smoke-test (all reference baseline solutions)
```bash
python scripts/test_reference_scalars.py          # `reference` author; prints N/N workloads passed
python scripts/test_reference_scalars.py autovec  # smoke-test the autovec baseline instead
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

**23 loops** across three patterns (140/140 workload correctness, Mac + Graviton4):
- **Scalar-output**: 001-004, 008, 010, 024, 032, 033, 126, 127 — reduction → single value
- **Array-output**: 027, 028, 029, 035, 108, 113, 128 — element-wise → N-element output
- **Inplace-sort**: 120, 121, 122, 123, 124 — data sorted in-place

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
4. `bench-trace/solutions/simd-loop/{reference,autovec}/loop_NNN/{reference,autovec}_loop_NNN.json`
   — two baseline authors from the same fused sources (`loop_NNN.h` + `loop_NNN.cpp` +
   `kernel.cpp`); `reference` is scalar (`-fno-vectorize`), `autovec` is `-O3 -march=native`

No Python files need manual editing to add a supported loop pattern.

## Instance types

| ISA  | Instance  | Notes                                |
|------|-----------|--------------------------------------|
| SVE  | c7g.large | Graviton3, Neoverse V1, 256-bit SVE  |
| SVE2 | c8g.large | Graviton4, Neoverse V2, 128-bit SVE2 |
| SME2 | —         | No AWS instance supports SME2 yet    |

## In-flight PRs (as of 2026-06-14)

| Branch                                | PR  | Status | Notes                                                              |
|---------------------------------------|-----|--------|--------------------------------------------------------------------|
| `feat/simd-loop-20-loops`             | #21 | Merged | 23 loops total (scalar/array/inplace-sort); self-contained solutions |
| `fix/no-candidate-harness`            | #22 | Merged | Bind candidate + simd-loop harness to solution sources; remove all framework hard-codes |
| `feat/simd-loop-sort-123-124`         | #23 | Draft  | Loops 123/124 (bitonic + radix sort) + loop_108 (pixel/LD4); needs rebase on main |
| `feat/update-load-workloads`          | #24 | Open   | Workload format: `scalar_inputs` → explicit `inputs` dict; uuid-seeded random generation; AND correctness formula |

---

## API consistency rules — ncnn conv2d is ground truth

The ncnn conv2d pipeline is the canonical reference for how datasets, adapters, and
solutions should be structured. When adding or modifying simd-loop code, match it.

**The interface contract (both ncnn and simd-loop must follow this):**
- `wrap_inputs(np_inputs, op_type, lib, *, definition)` — no `scalar_args`, `definition` is keyword-only
- Solutions are fully self-contained: harness `.h` + `.cpp` + `kernel.cpp` embedded in solution JSON
- No per-loop or per-op hardcodes in any framework Python file (`datasets/`, `evaluators/`, `runner.py`)
- Per-loop config lives in the generator (`gen_simd_loop_harness.py`) and gets baked into definition/solution JSON — the framework never sees it
- `_DTYPE_MAP` in each dataset adapter covers all dtypes that dataset actually uses

**What must NOT exist in simd-loop that doesn't exist in ncnn:**
- `sig_from_definition()` — removed in PR #22; do not re-add; argtypes are derived from definition in `wrap_inputs`
- `scalar_args` / `{"N": n}` passed to `wrap_inputs` — n is now derived from input array shape
- Per-loop input generation hacks in `bench/runtime/inputs.py` — input generation must be type-driven, not loop-ID-driven

---

## Open TODOs

### Resolved (was "block PR #23 merge"; all verified done 2026-06-26)
- [x] Rebased on origin/main (branch is 0 commits behind).
- [x] `scripts/test_reference_scalars.py` no longer imports `sig_from_definition`; uses the new `wrap_inputs` signature.
- [x] `uint8`/`uint16` present in `_DTYPE_MAP` (`bench/datasets/simd_loop.py`).
- [x] Workloads regenerated to `inputs: Dict[str, WorkloadInput]` (PR #24 merged).
- [x] uint32 LCG ramp removed from `bench/runtime/inputs.py` — generation is type-driven
  (`_gen_random_tensor` by dtype, `_gen_byte_buffer` for the `bytes` type); the
  `make_*_ramp` helpers that remain are ncnn's, not a simd-loop hack.

### Longer term
- [ ] Add `baseline-sve` solutions to the trace (extract `HAVE_SVE_INTRINSICS` block from `loops/loop_NNN.c` via a new `_extract_sve_kernel()` in the generator) — ncnn has `baseline-ncnn-arm` as its expert ceiling; simd-loop needs the equivalent so agent speedup is measured against both scalar and SVE baselines
- [ ] Run `bench_loop_agent.py` against all 23 loops on Graviton4 and commit timing traces — agent performance has only been validated for the original 4 loops
- [ ] Wire `bench/` evaluator output into agent loop (Issue #17) so the agent sees correctness + speedup per turn

### Trace / HuggingFace
- Simd-loop definitions, workloads, and `reference`/`autovec` baseline solutions are uploaded to `arm-bench/arm-bench-trace` on HuggingFace
- After adding `baseline-sve` solutions, upload those too
- ncnn data on HF is pre-PR#15 (conv2d only) — needs refresh with 5-op layout + self-contained solutions
