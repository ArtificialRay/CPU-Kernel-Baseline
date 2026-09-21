# E2E: agent-optimized kernels → end-to-end tokens/s on Qwen (llama.cpp, Graviton4)

Models are entries in `config/e2e_models.json`; everything below is derived from the GGUF header, so the pipeline is the same for each. Primary: **Qwen3.8-27B** (this branch); the 4B walkthrough below is the worked example.

Branches `feat/e2e-qwen35` (4B) → `feat/e2e-qwen38-27b` (adds the 27B + model registry). Status: **4B run complete (2026-09-21)** — see Results at the end.

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

## Qwen3.8-27B (branch feat/e2e-qwen38-27b)
Newest dense Qwen (2026-08-13, Apache-2.0, `bartowski/Qwen3.8-27B-GGUF` Q4_K_M, 17 GB). Same
`qwen35` architecture as the 4B: 64 decode layers (+1 MTP layer llama.cpp skips), d=5120,
ff=17408, GDN inner 6144, 24/4 attention heads, untied Q6_K lm_head. Qwen3.6-27B is
shape-identical; Qwen3.6-35B-A3B (MoE) is registered but not supported — its 8-of-256 expert
`mul_mat_id` ops carry ~30% of decode bytes and the override hook only intercepts 2-D mul_mat.

Per-token weight traffic **16.5 GB** (Q4_K 65%, Q6_K 25%, Q8_0 8.5%, Q5_K 0.7%): bandwidth-bound
decode ≈ 3–4 tok/s on a c8g.4xlarge (32 GB RAM needed for the 17 GB file; **xlarge cannot run it**).
Kernel evaluation boxes never load the model, so agent runs still use c8g.xlarge; only
measurement needs the 4xlarge (tg128 ≈ 40 s/run → `--reps 5` over three builds ≈ 15 min/thread-count).

| definition | share | roles |
|---|---|---|
| gemm_ggml_q4_K_n17408_k5120 | 33.5% | ffn_gate ×63, ffn_up ×47 |
| gemm_ggml_q4_K_n5120_k17408 | 18.3% | ffn_down ×60 |
| gemm_ggml_q6_K_n17408_k5120 | 7.5% | ffn_up ×16, ffn_gate ×1 |
| gemm_ggml_q4_K_n10240_k5120 | 6.4% | attn_qkv ×36 |
| gemm_ggml_q6_K_n248320_k5120 | 6.3% | output (lm_head) |
| gemm_ggml_q6_K_n5120_k6144 | 5.6% | ssm_out ×36 |
| gemm_ggml_q4_K_n6144_k5120 | 3.8% | attn_gate ×35 |
| gemm_ggml_q8_0_n5120_k6144 | 5.7% | attn_output ×16, ssm_out ×12 |
| gemm_ggml_q4_K_n12288_k5120 | 3.2% | attn_q ×15 |
| gemm_ggml_q6_K_n10240_k5120 | 2.9% | attn_qkv ×11 |
| gemm_ggml_q6_K_n6144_k5120 | 1.9% | attn_gate ×12 |
| others (Q8_0 k/v/up/down, Q5_K up, Q6_K q/down) | <1% each | |

Cost scales with the number of distinct shapes (≈20 definitions here vs 11 for the 4B), not
with model size: ≈ $200–300 per seed at Sol rates. lm_head reference: 248320×5120 chunked.

Runbook: `provision_e2e.py --label <4xl box> --model qwen3.8-27b --agent-build`, then `measure_e2e.py`
with `--threads 1 16`. Regenerate definitions: `qwen35_inventory.py <header> --json inv.json && gen_qwen35_definitions.py inv.json --model-tag qwen3.8-27b`.

## Results — Qwen3.5-4B, Claude Code + Claude Fable 5.1, 2026-09-21

**Agent runs.** One lane, c8g.xlarge, Claude Code headless (Max plan), `--min-iterations 40 --max-iterations 50`,
isa sve2, 11 packed-ABI definitions (99.7% of decode bytes), ~45 min/kernel, ~10.5 h total. Every kernel
met the 40-call floor after the prompt fix (see `test_scripts/harness_adapters.py`). Per-kernel best vs
`baseline-llamacpp-arm` (= `ggml_mul_mat` on the plain layout, i.e. ggml's generic dot-product path):

| definition | share | best |
|---|---|---|
| gemm_ggml_q4_K_n9216_k2560 (ffn gate/up) | 31.1% | 3.00x |
| gemm_ggml_q6_K_n248320_k2560 (lm_head) | 19.1% | 2.25x |
| gemm_ggml_q5_K_n8192_k2560 (GDN qkv) | 12.7% | 3.42x |
| gemm_ggml_q6_K_n2560_k9216 (ffn down) | 11.3% | 2.74x |
| gemm_ggml_q4_K_n2560_k9216 (ffn down) | 7.8% | 2.87x |
| gemm_ggml_q5_K_n2560_k4096 (GDN out) | 6.3% | 3.39x |
| gemm_ggml_q4_K_n4096_k2560 (GDN z) | 5.2% | 3.03x |
| gemm_ggml_q4_K_n8192_k2560 (attn q) | 3.5% | 2.99x |
| gemm_ggml_q4_K_n2560_k4096 (attn out) | 1.7% | 2.95x |
| gemm_ggml_q4_K_n1024_k2560 (attn k/v) | 0.6% | 3.11x |
| gemm_ggml_q6_K_n1024_k2560 (attn v) | 0.4% | 2.70x |

Geomean 2.93x. Kernels: HF `solutions/llama.cpp/claude-code-claude-fable-5-1-sve2/`; runs under `runs/e2e-qwen3.5-4b/`.
Every kernel is a genuine Q*_K × Q8 int8 matmul (i8mm `usmmla` tiles + dynamic activation quantization) — the
same idea as llama.cpp's own repack path, rediscovered for the unpacked layout.

**End-to-end** (c8g.4xlarge, llama.cpp v0.4.1, Q4_K_M, llama-bench pp512/tg128, medians of 5 interleaved reps,
spreads < 1%; `stock` = upstream with repack ON (what users run), `norepack` = upstream with
`GGML_CPU_REPACK=OFF`/`KLEIDIAI=OFF` (the dispatch path the agent kernels replace), `agent` = norepack + override
hook + the 11 kernels with the threaded `entry_rows` dispatch, `agent1t` = same kernels forced to one thread):

| threads | kind | stock | norepack | agent | agent1t | agent/stock | agent/norepack |
|---|---|---|---|---|---|---|---|
| 1 | pp512 | 13.7 | 6.6 | **19.6** | 19.7 | **1.43x** | 2.98x |
| 1 | tg128 | 5.31 | 4.01 | **5.78** | 5.78 | **1.09x** | 1.44x |
| 4 | pp512 | 53.3 | 26.2 | **75.3** | 27.5 | **1.41x** | 2.87x |
| 4 | tg128 | 17.7 | 13.1 | **19.1** | 6.1 | **1.08x** | 1.46x |
| 16 | pp512 | 158.0 | 86.0 | **213.1** | 23.3 | **1.35x** | 2.48x |
| 16 | tg128 | 38.7 | 33.9 | **41.9** | 5.0 | **1.08x** | 1.24x |

Reading: (1) the agent kernels beat the code path users actually run (stock, repack on) at every thread count:
+35–43% prefill, +8–9% decode. (2) Against the generic path they replace, 2.5–3.0x prefill and 1.24–1.46x decode.
(3) The 2.93x per-kernel geomean compresses to 1.24–1.46x on decode because decode is DRAM-bandwidth bound
(2.73 GB of weights per token; 16-thread stock already streams at ~106 GB/s-equivalent) — the microbenchmark's
hot-L2, single-thread setting overstates what a memory-bound loop can deliver. (4) `agent1t` shows why the
override had to be threaded: single-threaded kernels cap decode at 5–6 tok/s regardless of thread count.
(5) Caveats: only mul_mat is overridden (norms, GDN recurrence, attention, activations stay stock — ~0.3% of
decode bytes but more of prefill compute); the override casts activations f32→bf16 per call.
Raw numbers: HF `runs/e2e-qwen3.5-4b/measurements/`, W&B run `e2e_qwen3.5-4b_fable_c8g4xl_20260921T1845Z`.

### Perplexity acceptance check — FAILED for prefill, marginal for decode
wikitext-2 test, 8 chunks, 16 threads: stock **10.06** ± 0.61, norepack 10.07, **agent 14.45** ± 0.90, agent1t 14.58.
Attribution (override restricted to one kernel at a time, ub=512): ffn gate/up q4_K 11.51, GDN in-proj q5_K 11.23,
GDN z-gate q4_K 10.87, attn q 10.23; all others ≤ 10.14 (lm_head 10.09, every Q6_K clean). The three costly kernels
are exactly the ones fed the normalized hidden state, where LLM activations carry large channel outliers.
Per-call M dependence (2 chunks, stock 8.56): M=1 → 8.80 (+3%), M=32 → 12.7 (+49%), M=2…8 → ~85 (catastrophic).
**Control:** the reference-scalar kernels spliced through the same override reproduce stock (8.54/8.55 at ub32/ub512),
so the splice is correct and the defect is in the agent kernels' numerics. On the Mac the Fable q4_K kernel is
correct at every M on random data (41 dB SQNR) but drops to 17 dB with channel outliers because it uses one
activation scale per row (ggml: one per 256-element block); the M=2…8 collapse is not reproduced with synthetic
data and its mechanism is still open.

**What this means.** The per-kernel harness — random inputs, 20 dB SQNR gate, M ≤ 32 — accepted kernels that are
unusable in the real model. Paper-safe statement today: decode (M=1) +8% tokens/s at a +1–3% perplexity cost;
the prefill numbers are not valid. Fixes for the benchmark: (a) workloads drawn from real activations (dump
from llama.cpp), (b) a gate relative to the baseline's own SQNR instead of an absolute 20 dB, (c) end-to-end
perplexity as the acceptance test for any e2e claim. Attribution logs: HF `runs/e2e-qwen3.5-4b/measurements/attrib/`.


### Root cause — the M>=2 batched path requantizes the Q8_K block sums (2026-09-21, reproduced locally)

Reproduced end to end on an Apple M2 Pro with no cloud box. Recipe: clone llama.cpp
v0.4.1, `scripts/e2e/override/apply.sh`, cmake with `-DGGML_METAL=OFF
-DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF`, and rebuild the Fable kernels with the
new `build_manifest.py --march='-march=armv8.6-a+i8mm+bf16'`. The two q6_K kernels use
SVE only for a 4-byte sign-extending load
(`svget_neonq_s32(svld1sb_s32(svptrue_b32(), p))`); substituting
`vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(p))))` makes all 11 build and run on NEON+i8mm.
wikitext-2 now comes from `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
(the old S3 link is dead). All perplexities below: 2 chunks, `-c 512`, `-t 8`.

Every Fable gemm has two paths. `M == 1` keeps the activation block sums **exact**,
splitting each int16 sum into lo/hi bytes and recombining with `kMinMul {1,128,1,128}`.
`M >= 2` instead requantizes those sums to **int8 under a single shared exponent `sh`
computed from `maxabs` over the whole batch** (`mscale = 1 << sh`), so the Q4_K/Q5_K
min-correction term fits one `usmmla`. That shared exponent is the defect: it is a
batch-global scale applied to a quantity whose per-row spread is large on real hidden
states.

Proof by construction. Mechanically rewriting `if (M == 1) { ... }` into a per-row loop
in all 11 kernels (M=1 semantics at every M) restores perplexity:

| configuration | ub512 | ub64 |
|---|---|---|
| stock (norepack) | 8.59 | 8.59 |
| Fable, per-row loop (exact block sums) | 8.77 | 8.81 |
| Fable as submitted (batched path) | 12.68 | 12.89 |

The residual +2.2% of the per-row variant is the *second*, smaller defect: one activation
scale per row of K elements where ggml's Q8_K uses one per 256-element block. In isolation
that costs ~3 dB SQNR on activations with channel outliers.

Per-kernel attribution (that kernel batched, the other ten row-at-a-time, ub512; the
deltas sum to +3.87 against an observed +3.91, so contributions are additive):

| kernel | role | PPL | delta |
|---|---|---|---|
| gemm_ggml_q4_K_n9216_k2560 | ffn gate/up | 10.45 | +1.67 |
| gemm_ggml_q5_K_n8192_k2560 | GDN in-projection | 9.97 | +1.19 |
| gemm_ggml_q4_K_n4096_k2560 | GDN z-gate | 9.38 | +0.61 |
| gemm_ggml_q4_K_n8192_k2560 | attention qkv | 8.99 | +0.22 |
| the other seven | | <= 8.87 | <= +0.10 each |

Three kernels carry 89% of the damage, and they are exactly the three fed the normalized
hidden state. This matches the c8g.4xlarge attribution independently.

**The fix is cheap and was verified.** Storing the block sums as exact lo/hi (32 B per
pair-block instead of 16), two `usmmla` plus `vmlaq_n_s32(mlo, mhi, 128)`, and
`mscale = 1`, applied to `gemm_ggml_q4_K_n9216_k2560` alone:

| manifest | PPL (ub512) |
|---|---|
| ten row-at-a-time + this kernel fixed | 8.79 (vs 8.77 all row-at-a-time) |
| ten as submitted + this kernel fixed | 10.82 (vs 12.68 all as submitted) |

so the fix removes that kernel's entire contribution while keeping the batched i8mm path.
It costs about 15% of pp512 on this host (105 -> 89 tok/s, whole model, two interleaved
5-rep runs), and nothing on decode, which never leaves the M=1 path.

**The 80-87 perplexities in `attrib/ppl_diag2.log` do not reproduce.** With all 11 kernels
locally, every ubatch from 4 to 512 lands between 8.81 and 12.89. That diagnostic's script
died with its box; treat those three numbers as an artifact of the run, not a finding. The
reproducible claim is: decode (M=1) +2.6% perplexity, prefill (M>=2) +30-50%.

**What this says about the benchmark.** The submitted and the fixed kernel differ by 0.3 dB
SQNR on the harness's random-input workloads, and by 3.9 perplexity in the model. A fixed
20 dB SQNR floor over synthetic activations cannot separate them. Before any re-run:
workloads built from activations dumped out of llama.cpp, a gate set relative to the
baseline's own SQNR rather than an absolute floor, and end-to-end perplexity as the
acceptance test for any model-level claim.
