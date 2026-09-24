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

### Gate v2 — what changed, and the evidence it works (2026-09-21)

Three changes, all needed; any one alone still lets the bad kernel through.

1. **Real activations.** New workload input type `{"type": "tensor", "path": ...}` loads a
   tensor from the warehouse. `ARMBENCH_DUMP_DIR` on the override hook captures what
   llama.cpp actually computes (`ARMBENCH_DUMP_STRIDE` spreads captures over the layer
   stack, since the same shape recurs once per layer), and
   `scripts/e2e/dump_to_workloads.py` slices those into per-M workloads. 5.3 MB for all 11
   definitions.
2. **Realistic weights.** `_gen_ggml_kquant_rows` used to draw the quant integers uniform
   and the per-sub-block scale and min independently from one distribution. Real Q4_K has
   `dmin` about 8x `d` and 6-bit mins near 46/63, because the sub-block min carries most of
   the weight value; the old generator gave `dmin/d = 1.0` and mins of 37, which made the
   min-correction term a minor correction and hid any kernel that mishandled it. It now
   draws Gaussian weights and quantizes them the way ggml does, which reproduces the real
   tensor's statistics to three digits.

   | | d | dmin | dmin/d | mean 6-bit min |
   |---|---|---|---|---|
   | real Qwen3.5-4B tensor | 5.0e-5 | 4.04e-4 | 8.08 | 46.3 |
   | old generator | 2.8e-4 | 2.8e-4 | 1.00 | 37.0 |
   | new generator | 4.9e-5 | 4.04e-4 | 8.19 | 46.6 |

3. **A baseline-relative floor.** `EvalConfig.sqnr_margin_db` (default 1.0) plus a
   per-workload `baseline_sqnr_db` tag written by `scripts/e2e/calibrate_sqnr_floor.py`.
   The candidate is held to what the reference implementation itself scores on the same
   inputs. On these definitions that floor is 48-51 dB, against the old absolute 20 dB.

**Evidence.** On `gemm_ggml_q4_K_n9216_k2560` the submitted kernel and the same kernel with
exact block sums were 0.3 dB apart under the old gate and both passed. Under gate v2 the
submitted kernel scores 27-33 dB against a 49-51 dB floor and fails every M>=2 workload;
the exact-bsums version passes all six. Across the nine kernels that compile on a NEON host
(the two SVE q6_K kernels need a Graviton), gate v2 rejects exactly the ones that hurt the
model:

| kernel | workloads passing gate v2 | measured e2e PPL cost |
|---|---|---|
| gemm_ggml_q4_K_n9216_k2560 | 1/6 | +1.67 |
| gemm_ggml_q5_K_n8192_k2560 | 0/6 | +1.19 |
| gemm_ggml_q4_K_n4096_k2560 | 1/6 | +0.61 |
| gemm_ggml_q4_K_n8192_k2560 | 1/6 | +0.22 |
| gemm_ggml_q5_K_n2560_k4096 | 1/6 | +0.00 |
| gemm_ggml_q4_K_n1024_k2560 | 6/6 | +0.10 |
| gemm_ggml_q4_K_n2560_k9216 | 6/6 | +0.05 |
| gemm_ggml_q4_K_n2560_k4096 | 6/6 | -0.05 |
| gemm_ggml_q6_K_n1024_k2560 | 6/6 | +0.01 |

The single workload the failing kernels still pass is M=1, whose code path keeps the block
sums exact -- which is also the only regime where the end-to-end perplexity cost was
acceptable. `gemm_ggml_q5_K_n2560_k4096` is a conservative rejection: it takes the same
shortcut but sits on a tensor where it happens not to matter. Erring that way is correct for
a gate.

All 30 packed-ggml definitions (the 11 Qwen3.5-4B shapes and the 19 Qwen3.8-27B shapes) are
calibrated, 176 workloads. The 27B workloads still use random activations -- only the 4B
model was dumped -- so their floors are relative but not yet outlier-aware.

### Results under gate v2 — the shortcut bought almost nothing (2026-09-22)

c8g.4xlarge, medians of 5 interleaved repetitions, perplexity on 8 chunks of wikitext-2
from the same binaries in the same run. `norepack` is stock code with ggml's repack path
disabled, i.e. the generic kernels the override actually replaces.

| build | pp512 t=16 | tg128 t=16 | speedup pp / tg | perplexity |
|---|---|---|---|---|
| stock | 157.9 | 38.00 | 1.00x / 1.00x | 10.06 |
| norepack | 85.8 | 33.18 | 0.54x / 0.87x | 10.07 |
| Fable as submitted | 212.5 | 41.74 | 1.35x / 1.10x | 14.45 (+43.6%) |
| Fable repaired | 209.3 | 41.72 | 1.33x / 1.10x | 10.27 (+2.1%) |

At 1 and 4 threads the same holds: 1.47x vs 1.45x prefill, and 1.41x vs 1.37x, with decode
identical to two decimals.

**The batch-shared exponent on the block sums bought about 1.5% of prefill throughput and
cost 41 points of perplexity.** It was never a speed-versus-accuracy tradeoff worth making;
the old gate simply could not price it. The repaired set keeps essentially all of the
speedup and lands within 2.1% of stock perplexity, which is the same cost decode already
paid and comes from the kernels' per-row activation scale rather than anything in the
batched path.

Per-kernel under gate v2 on the same box, the repaired set passes 63 of 64 workloads. The
single miss is the q5_K n8192 M=1 workload at 0.1 dB under its floor, the per-row
activation scale again, reproducing the Mac result exactly.

Paper-safe claim: **+33% prefill and +10% decode over stock llama.cpp at +2.1% perplexity**,
with the whole set passing a gate calibrated against the reference implementation's own
numerics on real activations.

### The Sol arm, and a kernel that submitted the baseline (2026-09-22)

`gpt-5.6-sol` (nanobot, OpenAI key, budget 50/60, 3 lanes) produced all 11 kernels under
gate v2 in about 2.5 h. Per-kernel harness speedups: 1.01x to 1.75x, geometric mean 1.20x.

Its `gemm_ggml_q5_K_n8192_k2560` does not implement a kernel. It calls
`ggml_vec_dot_q5_K_q8_K` -- ggml's own reference routine. That compiles and scores 1.01x
because the evaluation harness links ggml, so the symbol resolves: the kernel *is* the
baseline. Spliced into llama.cpp it cannot be dlopened, and because the override loader
disables the whole manifest when any library fails, the first Sol measurement silently ran
the no-kernel build and produced a plausible table of nothing (identical to `norepack`,
perplexity 10.067). `build_manifest.py` now checks each built .so for undefined `ggml_*`
symbols and skips with a reason; the deeper fix is to reject this at submission time so the
agent is told it has submitted the baseline. Exactly one kernel of the 22 across both sets
does this.

With that kernel dropped (its shape falls back to stock ggml, which is what it does anyway),
the Sol set measured on a c8g.4xlarge:

| build | pp512 t=16 | tg128 t=16 | speedup pp / tg | perplexity |
|---|---|---|---|---|
| stock | 157.7 | 38.05 | 1.00x / 1.00x | 10.06 |
| sol (10 kernels) | 92.4 | 34.74 | 0.59x / 0.91x | 10.06 |
| Fable repaired | 209.1 | 41.35 | 1.33x / 1.09x | 10.27 |

**Sol's kernels are numerically clean and too slow to matter.** Perplexity is
indistinguishable from stock, so nothing like the Fable defect is present. But at 0.59x of
stock they lose badly: they beat the generic ggml path they replace by only about 1.08x,
while ggml's own repack fast path beats that generic path by roughly 1.85x. A kernel set has
to clear the repack path, not the generic one, to be worth splicing.

The repaired Fable set was carried in this run as a cross-check and reproduced across two
separate instances: 1.33x vs 1.32x prefill, 1.09x vs 1.10x decode, perplexity 10.271 both
times.

### Does a better gate produce better work? (2026-09-22, complete)

Re-ran all five kernels that carried the batch-shared-exponent defect with Fable 5.1 under
gate v2, budget 33/42 (the previous sweep showed every one reached 99.6-100% of its own best
by turn 33).

| kernel | with the shortcut | under gate v2 | change |
|---|---|---|---|
| gemm_ggml_q5_K_n2560_k4096 | 3.39x | **3.601x** | +6.2% |
| gemm_ggml_q5_K_n8192_k2560 | 3.42x | **3.573x** | +4.5% |
| gemm_ggml_q4_K_n4096_k2560 | 3.03x | 3.026x | -0.1% |
| gemm_ggml_q4_K_n9216_k2560 | 3.00x | 2.967x | -1.1% |
| gemm_ggml_q4_K_n8192_k2560 | 2.99x | 2.927x | -2.1% |
| **geometric mean** | **3.160x** | **3.205x** | **+1.4%** |

**Told the truth about precision, the model finds the same speed honestly -- slightly more
of it.** The geometric mean is 1.4% *above* the shortcut versions, with every submission
clearing a floor set at the reference implementation's own SQNR on real activations. The two
Q5_K kernels gain 4-6%; the three Q4_K kernels give up 0.1-2.1%. Taken with the end-to-end
result -- the shortcut bought 1.5% of prefill and cost 41 points of perplexity -- the trade
the old gate induced was never a trade at all. The benchmark priced precision at zero, so
the optimizer spent it.

Stored as solution author `claude-code-claude-fable-5-1-gatev2` (HF + local backup in
`sweep_backups/2026-09-22_fable_gatev2/`). Cost $70 of Fable, about a quarter of the
eleven-kernel 2026-09-21 sweep.

**All 11 kernels are now Fable-authored**: 6 never had the defect and are the original
submissions untouched, 5 were re-optimized here under gate v2. No hand-patched kernel remains
in the set. Total cost across both rounds ~$137 of Fable, about half the 2026-09-21
eleven-kernel sweep.

**The hand-patched `fable-exact-bsums` set is retired as a result.** It did its job: it is
how the defect was proven (mechanically removing the shared exponent, changing nothing else,
moved perplexity from 14.45 to 10.27 while prefill went only 1.35x -> 1.33x), and that
established the shortcut was never a real tradeoff. But it is a hand edit of the model's
output, not something an agent produced, so it cannot carry a claim in a paper about what
agents produce. It stays in the root-cause section as evidence and is dropped from the
results tables, the kernel-level comparison and the headline. The Fable-authored gate-v2
set supersedes it.

### A measurement that measured the wrong kernels (2026-09-22)

The first end-to-end run of the complete gate-v2 set reported 1.35x prefill at perplexity
**14.449** — matching the known-broken set's 14.4493 to four digits. It was not a coincidence.
The manifest had been built with `build_manifest.py --solutions
claude-code-claude-fable-5-1-gatev2`, and that solution folder had been assembled by hand by
copying JSONs off the controller *after* a local→controller `bench-trace/` rsync had already
replaced them with the previous round's submissions. Ten of eleven entries were the original
kernels; only `gemm_ggml_q5_K_n2560_k4096` was genuinely new.

Nothing in the pipeline objected, and nothing should have: the splice was correct, 11/11 shapes
were intercepted, every `.so` was readable, every gate passed. **The plumbing checks verify that
kernels run, not that they are the kernels you meant.** The only tell was a perplexity that
happened to be recognisable.

Two things came out of it:

- `scripts/e2e/verify_kernel_set.py` — compares a manifest's actual compiled sources against a
  reference run set and fails *before* provisioning if a kernel that should be new is
  byte-identical to the old one (`--expect-changed`), if a deliberately reused kernel has
  drifted (`--expect-same`), or if a known-bad idiom appears anywhere (`--forbid 'maxabs >> sh'`).
  Sources are compared through `kernel_rows_abi_text()`, since `build_manifest` leaves a
  *rewritten* kernel.cpp in its build dir and raw run-dir sources never match it byte for byte.
  Run against the bad set it reports "11 reused, 0 new" and refuses.
- `scripts/e2e/solutions_from_runs.py` — builds solution JSONs from the agent run directories,
  which is the authoritative record (`trajectory.jsonl` names the submitted version, `v<N>.cpp`
  holds it). The folder was assembled by hand because no tool did this. The regenerated folder
  has been re-pushed to HF and verified against the originals in both directions.

Auditing the rest turned up one more instance of the same contamination: the controller's
`claude-code-claude-fable-5-1-sve2` copy of `gemm_ggml_q5_K_n2560_k4096` had been overwritten
with the gate-v2 kernel, plus a stray `..._current.json` that `TraceSet` would have loaded as a
twelfth solution. Restored from the clean local copy; HF was never affected (checked file by
file against local).

The corrected measurement builds the manifest with `--runs <original-runs> <gatev2-runs>`
(later roots win, which is exactly the 6-reused + 5-re-optimized shape) and runs the guard
before provisioning.

**Invalidated by this**: the `fable_gatev2` row of the 2026-09-22 table, and the claim that the
fully Fable-authored set "matches the cheating original's 1.35x prefill exactly" — it *was* the
cheating original, measured twice. The `stock`, `norepack` and `fable_repaired` rows stand, and
`fable_repaired` has now reproduced 1.32x at PPL 10.27 on three separate instances. The
kernel-level gate-v2 speedups (3.601x / 3.573x / 3.026x / 2.967x / 2.927x) were read from the
trajectories, not from the copied files, and are unaffected.

### End to end, fully Fable-authored (2026-09-22, corrected run)

Manifest built from the agent run directories (`build_manifest.py --runs <original> <gate-v2>`,
later roots win) and checked by `verify_kernel_set.py` before the box came up: 5 verified new,
6 verified reused, the shared-exponent idiom absent from all 11. c8g.4xlarge, medians of 5,
perplexity over 8 wikitext-2 chunks from the same binaries.

| build | pp512 t=16 | tg128 t=16 | speedup pp / tg | perplexity |
|---|---|---|---|---|
| stock | 158.1 | 39.17 | 1.00x / 1.00x | 10.062 |
| norepack | 86.1 | 34.14 | 0.54x / 0.87x | 10.067 |
| **Fable, gate v2** | **210.4** | **42.26** | **1.33x / 1.08x** | **10.162 (+1.0%)** |

At lower thread counts the margin is wider — 1.41x prefill at t=1 and 1.39x at t=4 — so 1.33x
is the conservative end of the range, taken where ggml's own repack path is strongest.

**Headline claim: +33% prefill and +8% decode over stock llama.cpp at +0.62% perplexity, with
every kernel written by the agent.** (The +1.0% figure this table's 8-chunk perplexity column
gives is an artifact of too small a sample; the 64-chunk re-measurement below puts it at
+0.62% with 3x tighter error bars.) Against the same set's shortcut-taking predecessor
(1.35x prefill, perplexity 14.45), being honest about precision cost about 1.5% of prefill
and recovered 43 points of perplexity.

The hand-patched set, measured alongside as the last use of it, reached the same speed (1.33x)
at **+2.1%** perplexity against the re-optimized set's **+1.0%**. So the re-run bought more
than authorship: re-optimizing under a real precision constraint produced kernels that are
measurably more accurate than mechanically patching the shortcut out of the old ones. That
question is now answered and the hand-patched set is retired.

### Kernel level, under gate v2 (2026-09-22)

Both kernel sets run on one c8g.xlarge — the instance type the agents ran on — against ggml's
own kernel as baseline, on the gate-v2 workloads (`scripts/e2e/speed_compare.py`). Six of the
eleven definitions were never re-optimized, so the two columns there are the *same bytes*
and measure nothing but run-to-run noise:

| definition | original submission | gate-v2 | Δ |
|---|---|---|---|
| q4_K n1024 k2560 | 3.145x | 3.147x | +0.05% |
| q4_K n2560 k4096 | 2.901x | 2.885x | −0.56% |
| q4_K n2560 k9216 | 2.880x | 2.874x | −0.21% |
| q6_K n1024 k2560 | 2.713x | 2.704x | −0.35% |
| q6_K n248320 k2560 | 2.264x | 2.265x | +0.04% |
| q6_K n2560 k9216 | 2.715x | 2.728x | +0.47% |
| **geomean** | **2.756x** | **2.753x** | **−0.09%** |

**The measurement's noise floor is 0.09% in the geomean and 0.56% on the worst single
kernel.** Any difference larger than that in the table below is real.

The five re-optimized definitions:

| definition | original submission | gate-v2 |
|---|---|---|
| q4_K n4096 k2560 | fails gate (1/6 workloads) | 2.950x |
| q4_K n8192 k2560 | fails gate (1/6) | 2.909x |
| q4_K n9216 k2560 | fails gate (1/6) | 2.908x |
| q5_K n2560 k4096 | fails gate (1/6) | 3.568x |
| q5_K n8192 k2560 | fails gate (**0/6**) | 3.625x |
| **geomean** | — | **3.175x** |

**The original submissions have no speed on this axis, because they are not admissible.**
Each one passes at most the single M=1 workload and fails every batched one with
`INCORRECT_NUMERICAL`; the q5_K n8192 kernel fails all six. Their old headline numbers
(2.99x-3.42x) were measured against a gate that could not see the defect, so there is nothing
to compare them to here — which is the point. Under a gate that prices precision, 5 of the
11 submissions from the first sweep simply do not count.

Across all eleven, the gate-v2 set is **2.938x, with 11/11 admissible**; the original set is
**6/11 admissible**. Beware the naive per-author geomean here: the shortcut set scores 2.756x
only because its five hardest kernels are missing from the column. `speed_compare.py` prints
the common-definition geomean alongside it for that reason.

### What the kernels' quality cost is worth, measured (2026-09-22)

A quality cost is unreadable without a scale, so the same binary, text and thread count were
run across the Qwen3.5-4B quantization ladder (`scripts/e2e/quant_reference.py`).

**Measured twice.** The first pass used the same 8 chunks (~4k tokens) as the kernel
measurement and produced a ladder that was *non-monotonic* — Q4_K_S scored better than
Q4_K_M and Q5_K_M scored worse — which cannot be true and showed the sample could not
resolve differences below about ±0.3%. Conclusions drawn from it were wrong: it put the
kernels' cost at +1.00% and 4-bit quantization's at +0.57%, i.e. the kernels costing nearly
twice the quantization. Re-run at 64 chunks, with error bars 3x tighter (±0.20 vs ±0.61),
the ladder is monotonic and the relationship reverses. **The 8-chunk numbers are recorded
here only as the reason to distrust them; the 64-chunk numbers are the result.**

| model | size | PPL (64 chunks) | vs Q4_K_M |
|---|---|---|---|
| Q3_K_M | 2.29G | 9.8459 | +3.61% |
| **Q4_K_M (stock)** | 2.74G | **9.5029** | — |
| *Q4_K_M + our kernels* | *2.74G* | *9.5617 ± 0.201* | *+0.62%* |
| Q6_K | 3.53G | 9.4364 | −0.70% |
| BF16 | 8.42G | 9.3841 | −1.25% |

- **Quantizing this model to 4 bits costs +1.27%** (BF16 → Q4_K_M). The kernels add
  **+0.62%** on top — **about half** what the quantization they run on already costs.
- Dropping a whole level, Q4_K_M → Q3_K_M, costs **+3.61%**. The kernels cost **17% of one
  step down the ladder**.
- Against unquantized BF16, the whole stack — 4-bit weights plus our kernels — is +1.89%.

**Paper-safe statement: +33% prefill and +8% decode for a perplexity cost of +0.62%, which is
half the cost of the 4-bit quantization it runs on and a sixth of one quantization level.**

The residual is the known second-order defect: the kernels carry one activation scale per row
of K elements where ggml uses a Q8_K scale per 256-element block. It is not the
shared-exponent bug — that one is gone — and closing it would need another gate change and
another re-optimization round.

**Methodological note worth keeping:** an 8-chunk perplexity sample is enough to detect a
catastrophic regression (the broken kernels' +44% was never in doubt) and not enough to size
a small one. A ladder that fails to order itself monotonically is the cheap tell that the
sample is too small, and it should disqualify the comparison before conclusions are drawn
from it, not after.

### The headline was a single operating point (2026-09-24)

Every end-to-end number above was measured at one configuration: `pp512`, llama-bench's default
micro-batch of 512, and `-fa auto`. Sweeping the axes we had never swept changed two of the
three conclusions.

**Prompt length, and the flash-attention trap.** Prefill speedup falls with context — but far
less than the first sweep suggested, because `-fa auto` selects flash attention, which on this
Graviton4 is roughly **2x slower** than no-FA at long prompts (stock 138 vs 67 tok/s at pp8192).
The first curve was measured on the slow path:

| prompt | `-fa off` | `-fa on` |
|---|---|---|
| 512 | **1.373x** | 1.328x |
| 2048 | **1.350x** | 1.240x |
| 4096 | **1.321x** | 1.178x |
| 8192 | **1.277x** | 1.119x |

A control run reproduced the `-fa on` column exactly against the original sweep
(1.328/1.240/1.178/1.119 vs 1.328/1.237/1.175/1.119), so the difference is the attention path
and nothing else. **On the path you would actually deploy, the speedup holds: 1.28x at 8k.**

**Micro-batch — where it was catastrophic.** `-ub` sets the M dimension the kernels see, and
M>=2 is where the original precision shortcut lived, so it is the obvious axis to sweep. It had
never been swept:

| `-ub` | before | after |
|---|---|---|
| 1 | 1.078x | 1.086x |
| **2** | **0.213x** | **1.439x** |
| **8** | **0.382x** | **1.339x** |
| 64 | 1.060x | 1.059x |
| 512 | 1.324x | 1.324x |

At micro-batch 2 the spliced build ran **4.7x slower than stock**. Not the agent's kernels —
`armbench_override.c` branched on `M == 1` for N-split and M-split everything else, so at M=2
with 16 threads exactly 2 threads worked and 14 idled while stock ggml parallelises over N.
Fixed by N-splitting whenever `M < nth` (commit 33f370c); small batches are now the *best*
cells, since small M is where ggml's own kernels are weakest.

**Decode is barely context-sensitive** — 1.074x at depth 0, 1.041x at depth 32768 — and
quantized KV (q8_0) changes nothing (1.053x vs 1.059x). A combined `-pg 2048,128` turn measures
1.206x, which matches what the per-rate arithmetic predicted, so that method is sound at
moderate lengths; only the long-context extrapolation was wrong.

**Revised honest claim: ~1.37x prefill at short context, 1.28x at 8k, 1.32-1.44x across batch
shapes, at +0.62% perplexity** — better than the original claim once measured on the right
attention path and with the dispatch bug fixed.

**Methods note, and it is the same lesson the rest of this document is about.** Three separate
errors in one evening, all from treating a tool default as a property of the system: a latency
table extrapolated from a single pp512 rate (predicted 29% at 8k, truth 12%); a "decays to
parity" conclusion drawn on an attention path `auto` had chosen; and four prior end-to-end
measurements that all ran at `ub=512` and so never saw a 4.7x regression sitting one axis away.
Pin `-fa`, sweep `-ub`, and never extrapolate a rate measured at one shape.

*Unresolved:* the decode control returned 1.025x on the second box against 1.074x on the first.
The patch is functionally identical at M=1, so this is probably cross-box variation, but it has
not been shown, and no decode figure should be quoted until it is.
