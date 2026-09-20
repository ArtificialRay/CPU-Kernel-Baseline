# E2E: agent-optimized kernels → end-to-end tokens/s on Qwen3.5-4B (llama.cpp, Graviton4)

Branch `feat/e2e-qwen35`. Status: **scaffold, locally tested** (templates/repack/override hook on macOS arm64 against v0.4.1; nothing run on a Graviton yet) — nothing has been launched or spent.

## Question
Take one real model, let the agent optimize every kernel family that matters for
its CPU inference, splice the kernels back into the runtime, and measure the
change in tokens/s. Per-kernel speedups (the rest of the benchmark) say what the
agent can do to a kernel; this says what that is worth end to end.

## Fixed choices
| item | choice | why |
|---|---|---|
| runtime | llama.cpp **v0.4.1** (2026-09-14), CPU backend only | actively maintained (Arm SVE/SME work through Aug 2026), every kernel is C/C++ source, already the source of the `llama.cpp` dataset |
| model | **Qwen3.5-4B**, `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (2.74 GB) | dense 4B, Feb 2026, hybrid 3×GatedDeltaNet:1×attention → more kernel diversity than a plain transformer; fits an xlarge |
| measurement box | c8g.4xlarge (16 vCPU, $0.64/h on-demand) | 60 tok/s-class for a 4B Q4; xlarge (4 vCPU) for agent runs |
| agent | nanobot + `openai/gpt-5.6-sol` via OpenRouter ($2/$10 per M) | strongest OpenAI tier; Luna's 1.34 geomean would not move tokens/s |
| protocol | S0: `--min-iterations 40 --max-iterations 50`, 3 seeds when budget allows | same as the rest of the paper |

Pinning lives in `config/dataset_builds.json` (`llama.cpp` and `llama.cpp-e2e` entries).

## What runs per token (from the GGUF header, `scripts/e2e/qwen35_inventory.py`)
Decode on a 4B Q4 model is memory-bound, so per-token weight bytes ≈ time share for the mul_mat kernels.

| n | tensor (role) | type | shape [K, N] | MB/token | share |
|---|---|---|---|---|---|
| 1 | token_embd (tied lm_head) | Q6_K | 2560 × 248320 | 521.5 | 19.1% |
| 32 | ffn_gate | Q4_K | 2560 × 9216 | 424.7 | 15.6% |
| 32 | ffn_up | Q4_K | 2560 × 9216 | 424.7 | 15.6% |
| 24 | attn_qkv (GDN in-proj) | Q5_K | 2560 × 8192 | 346.0 | 12.7% |
| 16 | ffn_down | Q6_K | 9216 × 2560 | 309.7 | 11.3% |
| 16 | ffn_down | Q4_K | 9216 × 2560 | 212.3 | 7.8% |
| 24 | ssm_out (GDN out-proj) | Q5_K | 4096 × 2560 | 173.0 | 6.3% |
| 24 | attn_gate (GDN z) | Q4_K | 2560 × 4096 | 141.6 | 5.2% |
| 8 | attn_q (+gate) | Q4_K | 2560 × 8192 | 94.4 | 3.5% |
| 8 | attn_output | Q4_K | 4096 × 2560 | 47.2 | 1.7% |
| 8 | attn_k | Q4_K | 2560 × 1024 | 11.8 | 0.4% |
| 5+3 | attn_v | Q6_K / Q4_K | 2560 × 1024 | 15.2 | 0.6% |
| | everything else (norms, conv1d, ssm α/β, A, dt) | F32/Q8_0 | | 8 | 0.3% |

By type: Q4_K 49.9%, Q6_K 30.8%, Q5_K 19.0%. Total 2.73 GB/token → the
bandwidth floor on a c8g.4xlarge (~60 GB/s usable) is ~22 ms/token ≈ 45 tok/s
for pure weight streaming; the stock build's measured tg128 will show how far
above that floor it sits (that gap is what kernels can recover).

**Kernel definitions (11 unique shapes, packed ggml block ABI, names `gemm_ggml_<type>_n<N>_k<K>`):**

| definition | share | roles |
|---|---|---|
| gemm_ggml_q4_K_n9216_k2560 | 31.1% | ffn_gate ×32, ffn_up ×32 |
| gemm_ggml_q6_K_n248320_k2560 | 19.1% | token_embd (tied lm_head) |
| gemm_ggml_q5_K_n8192_k2560 | 12.7% | attn_qkv (GDN in-proj) ×24 |
| gemm_ggml_q6_K_n2560_k9216 | 11.3% | ffn_down ×16 |
| gemm_ggml_q4_K_n2560_k9216 | 7.8% | ffn_down ×16 |
| gemm_ggml_q5_K_n2560_k4096 | 6.3% | ssm_out ×24 |
| gemm_ggml_q4_K_n4096_k2560 | 5.2% | attn_gate (GDN z) ×24 |
| gemm_ggml_q4_K_n8192_k2560 | 3.5% | attn_q ×8 |
| gemm_ggml_q4_K_n2560_k4096 | 1.7% | attn_output ×8 |
| gemm_ggml_q4_K_n1024_k2560 | 0.6% | attn_k ×8, attn_v ×3 |
| gemm_ggml_q6_K_n1024_k2560 | 0.4% | attn_v ×5 |

plus the existing `rms_norm_fp32_d2560`. **ABI decision:** these definitions hand the kernel
the raw ggml block rows (`B` uint8 `[N, K/256·{144,176,210}]`), not the benchmark's flat
nibble/scale layout — flat Q5_K/Q6_K would read 45–55% more bytes per token than stock
ggml and sink the memory-bound decode; with the packed ABI the agent kernel is spliced
into llama.cpp with zero data conversion (only the f32→bf16 activation cast). Kernel
entry: `armbench_entry_gemm(const uint16_t* A_bf16, float* out, const uint8_t* B_blocks, int M)`,
N and K baked. Workloads: M ∈ {1, 2, 4, 8, 16, 32} (M=8 max for the lm_head, which
llama.cpp evaluates for the last token only) — decode-sized like the existing gemm
definitions; prefill (pp512) is measured end to end only. The numpy references dequantize
in row chunks (lm_head reference: ~7 s/workload, 2 GB peak).

Not in scope for phase 1 (no dataset op type yet, ~0.3% of decode bytes but
compute-visible in prefill): `gated_delta_net` (ggml's CPU version is explicitly
a basic vector implementation), `ssm_conv`, `l2_norm`, gated RMSNorm, rope,
flash-attention over the 8 attention layers, SiLU-gate. `scripts/e2e/profile_ops.sh`
measures their real share with perf; add them as phase 2 if prefill share is material.

## Pipeline
1. **Inventory** — `qwen35_inventory.py <gguf header> --json inventory.json`.
2. **Definitions + baselines** — `gen_qwen35_definitions.py inventory.json` writes
   definitions / workloads / `reference-scalar` + `baseline-llamacpp-arm` solutions into
   bench-trace (gitignored; push to HF additively). Q5_K/Q6_K templates: `kquant_templates.py`,
   adapter repack in `bench/datasets/llama_cpp.py` (tests: `test_kquant.py`).
3. **Agent runs** — the usual driver, e.g.
   `bench_fleet.py --harness nanobot --dataset llama.cpp --isa sve2 --instance c8g.xlarge --on-demand --model openai/gpt-5.6-sol --min-iterations 40 --max-iterations 50 --definitions "<13 gemms> rms_norm_fp32_d2560" --wandb-group nanobot__gpt-5.6-sol__llama.cpp__sve2__E2E`
   with `NANOBOT_CONFIG_BASE` pointing at a private copy of `skills/nanobot/nanobot-kernel-session/config.openrouter.json` carrying the OpenRouter key (the adapter switches provider automatically for `vendor/model` ids).
4. **Splice** — each submitted `kernel.cpp` is compiled exactly as the harness compiled it
   (`bench/compile/builders/llama_cpp.py` flags) into a `.so`; `override/manifest.json` maps
   (op, type, K, N) → `.so` + symbol. The override hook (`scripts/e2e/override/`) is patched
   into ggml-cpu's `GGML_OP_MUL_MAT` dispatch; unmatched shapes fall through to stock ggml.
5. **Measure** — `provision_e2e.py --label <box> --agent-build` builds three trees
   (`build_agent_llamacpp.sh`): **stock** (upstream, repack on — what users run), **norepack**
   (upstream with `GGML_CPU_REPACK=OFF`/`KLEIDIAI=OFF` — the same dispatch path the agent
   kernels replace), **agent** (norepack + override hook + manifest). Then
   `measure_e2e.py --build stock=… --build norepack=… --build agent=… --overrides agent=manifest.json --threads 1 4 16 --reps 5`
   (interleaved runs, medians, plus `--perplexity` as the acceptance check). Report tg128 and
   pp512 separately and `-t 1` as the apples-to-apples row (override kernels are single-threaded);
   expect tg to compress toward the bandwidth floor. Verified in v0.4.1: NEON+i8mm repacks every
   q4_K weight with N % 8 == 0 and repacked tensors never reach the dispatch switch, so the
   stock-vs-agent gap includes ggml's interleaved fast path — the norepack row isolates the kernel effect.
6. **Attribution** — rerun with the manifest restricted to one quant type / one role at a
   time to get per-kernel end-to-end contributions (Amdahl check against the byte shares).

## Budget (estimate, not measured)
- Agent: ~110 tool calls/kernel; Sol ≈ $6–8/kernel at OpenRouter rates (reasoning may push it to
  $10–15). 14 kernels × 1 seed ≈ $150–200; 3 seeds ≈ $450–600. First run gives the real number.
- AWS: agent runs on c8g.xlarge ($0.16/h, ~1.5 h/kernel across 3 lanes); measurement on a
  c8g.4xlarge for ~2 h ($1.30). Nothing is launched without an explicit OK.

## Caveats to state in the paper
- Override kernels run single-threaded inside ggml's thread pool (the standalone ABI has no
  threading); llama-bench with `-t 1` is the apples-to-apples number, `-t 16` shows what the
  gap costs. Multi-threading the override (splitting N across ggml's threads) is a later step.
- Stock llama.cpp on Graviton may route Q4_K through repacked/KleidiAI paths; the measurement
  builds and flags are documented in `scripts/e2e/override/README.md`.
