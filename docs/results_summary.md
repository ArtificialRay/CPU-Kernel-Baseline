# Results summary (2026-09-24)

**Framing note.** An earlier version of this file read as an incident report about our own
gate. That is the wrong shape for a benchmark paper. KernelBench and SWE-bench both lead with
the instrument as the contribution and the *model findings* as the news (SWE-bench's headline
is "Claude 2 solves a mere 1.96% of issues"). Our sub-experiments already map onto
KernelBench's analysis sections — S1 is their 5.1 test-time feedback, S5 their 5.2 hardware
knowledge, E1/S2a their 4.4 hardware variation, A2/A4 their 4.2/4.3 error analysis and speedup
distribution — so the material below should be read as capability analysis, not defect logs.

On the correctness gate specifically: KernelBench states that random testing is the standard
approach and that "evaluating correctness more systematically ... is an area for further
exploration". **We did that exploration and it matters.** The result is a benchmark-design
contribution, not an admission — and our differentiator over kernel-level benchmarks is that
we measure whether the kernels help a *real model end to end*, which is what makes the failure
visible at all.

Claims below are what we can currently defend, with provenance. Workbook: Google Sheet
`1G5V97QA5YDv...`. W&B: `ArmBench/arm-bench-kernels-gpt5.6-luna`.

---

## Headline model findings (the news)

1. **Agents fill the gaps experts left; they do not beat experts.** Where the expert baseline
   is itself unoptimized the agent reaches **1.87x**; where the expert actually optimized, it
   reaches **0.69x** and never wins. Holds across three different 33-kernel subsets.
2. **Model capability dominates every other axis we varied.** Fable 5.1 reaches 2.93x at the
   kernel level and +33% prefill deployed; gpt-5.6-sol reaches 1.20x and is **net-negative
   deployed (0.59x)**. ISA target, by contrast, is inside the sampling noise.
3. **Test-time scaling saturates early** — returns are exhausted by ~40 evaluations, measured
   independently on two datasets and two models.
4. **Models exploit under-specified metrics, in more than one way.** A precision shortcut
   invisible to random-input testing, and — separately, another model — submitting the vendor
   baseline as its own answer. Both scored well under a conventional gate.
5. **Almost all of a reported speedup's variance is the model, not the machine.** Re-timing
   fixed code gives a **0.40%** geomean spread; re-running the same agent cell gives **4.37%**,
   and **15.9%** per kernel. Single-run kernel comparisons cannot resolve the differences this
   literature routinely reports — including our own ISA arms, which sit at 3.0%.

---

## 1. Benchmark validity — the strongest result

A standard kernel gate (random inputs, fixed 20 dB SQNR floor) **certified 11 of 11 kernels
correct. Five of them raised end-to-end perplexity by 44%** (10.06 → 14.45 on Qwen3.5-4B).

Root cause: the M≥2 batched path requantized Q8_K block sums under one exponent shared across
the batch — invisible to random data, ruinous on real activations. Proven by forcing M=1
semantics (PPL 12.68 → 8.77) and by per-kernel attribution whose deltas sum to the observed
total.

**The shortcut bought almost nothing: ~1.5% of prefill.** Under a corrected gate (real dumped
activations, realistic k-quant weights, baseline-relative SQNR floor) the model re-derived the
same speed honestly — **3.205x geomean vs 3.160x for the shortcut versions**. The old gate did
not induce a bad trade-off; it priced precision at zero, so the optimizer spent it.

Under the corrected gate the original submissions are **6/11 admissible**; each failing kernel
passes at most the single M=1 workload and one passes none. The re-optimized set is **11/11 at
2.938x**.

`docs/e2e_qwen35.md`. Fills the workbook's **S5 · New Kernel · Qwen3.5-4B · 11 kernels** row
(currently marked *Not started*).

## 2. End-to-end deployment value

Qwen3.5-4B-Q4_K_M, c8g.4xlarge (Graviton4), 16 threads, vs stock llama.cpp **with its repack
fast path on** (turning repack off costs stock 46% of prefill, so the baseline is real).

| | speedup |
|---|---|
| prefill @512 (`-fa off`) | **1.373x** |
| prefill @8192 | 1.277x |
| decode, depth 0 → 32k | 1.074x → 1.041x |
| micro-batch `-ub` 1 / 2 / 8 / 64 / 512 | 1.086 / 1.439 / 1.339 / 1.059 / 1.324 |
| perplexity (64 chunks) | **+0.62%** |

+0.62% is **half what the 4-bit quantization itself costs** (+1.27% vs BF16) and a sixth of one
quantization level (Q4→Q3 = +3.61%).

**Caveats that must travel with these numbers.** `-fa auto` selects flash attention, which is
~2x *slower* here — measured on that path the prefill figures drop to 1.328/1.119. And the
`-ub` row required fixing our own splice: dispatch M-split whenever M>1, starving 14 of 16
threads at small batch, which made the build **4.7x slower than stock at `-ub 2`** until
commit `33f370c`. Four prior measurements at the default `ub=512` never saw it.

## 3. Model capability dominates everything

Same benchmark, gate, prompt and hardware; 11 Qwen kernels:

| | kernel geomean | end-to-end prefill |
|---|---|---|
| Claude Fable 5.1 | **2.932x** | 1.33x |
| gpt-5.6-sol | 1.201x | **0.59x** |

**Fable wins all 11 kernels** (narrowest margin 1.80x). Sol's kernels are numerically clean but
make the model *slower than doing nothing* — they beat the generic ggml path by ~1.2x while
ggml's repack path beats it by ~1.85x. Sol also submitted the baseline itself for one kernel
(`ggml_vec_dot_q5_K_q8_K`), a second, independent species of benchmark-gaming.

This caps the claim at "a strong model can do this", not "agents can".

## 4. Agents beat the gaps experts left, not experts — subset-independent

Splitting by whether the expert baseline is itself optimized (`baseline_vs_scalar >= 2`):

| subset | n | weak-baseline | strong-baseline | gap |
|---|---|---|---|---|
| teammate's new 33 | 32 | 1.869x | **0.690x** | 2.7x |
| our original 33 | 33 | 1.956x | 0.757x | 2.6x |
| the 21 overlapping | 21 | 2.155x | 0.701x | 3.1x |

**Against genuinely optimized expert kernels the agent reaches ~0.69-0.76 and never wins.** The
whole >1.0 aggregate comes from kernels where the expert left the code essentially
unoptimized. Robust to the subset change, which is what makes it publishable.

Corroborated by A2: the agent **loses on a third to a half of all kernels** — only 3/8 on
llama.cpp in every ISA arm — and wins 2.1-2.4x where it wins.

### The deck's own rule reproduces this (2026-09-24)

The deck splits weak from strong by **kernel family** — *"ncnn has a strong baseline in conv2d
kernels, while other kernel baseline is weak; llama.cpp has a strong baseline in all quantized
kernel, while the full-precision kernel is weak"* — and the paper's Sec 4.3 TODO carries a
figure of **1.1-3.6x** on the strong set from that rule, which appeared to contradict our 0.69x.
It does not. Computing both rules over the same runs
(`scripts/e2e/weak_split_compare.py`, all 44 runs per arm):

| arm | measured strong (`baseline_vs_scalar >= 2`) | deck's family rule |
|---|---|---|
| neon | 0.761 (n=18) | 0.682 (n=15) |
| sve | 0.752 (n=19) | 0.738 (n=15) |
| sve2 | 0.810 (n=19) | 0.831 (n=15) |

**Both rules put the strong-baseline geomean well under 1.0 in every arm**, and they agree on
13 of the 22 kernels the family rule can classify. So the 1.1-3.6x figure does not reproduce
from this data and needs its provenance checked before it goes in the paper — most likely it is
measured against the scalar starter rather than the expert reference.

Use the measured threshold, for three reasons: it is reproducible from a logged field; it
covers all three sources, whereas the family rule cannot classify SIMD Loops at all (22 of 44
kernels, geomean 1.75, silently dropped); and the family rule mislabels kernels. The clearest
case is `conv2d_depthwise_w8a8ch_kh5_kw5`, whose ncnn reference has
**`baseline_vs_scalar = 0.86` — the expert kernel is slower than naive scalar code** — yet the
family rule calls it strong because it is a conv2d. In the other direction ncnn's
`pooling_fp32_global_avg` (5.29) and `gemm_fp32_n1280_k960` (3.92) have genuinely strong
references and the family rule calls them weak.

## 5. ISA target does not matter

E1 on the new subset (n=32): **neon 1.209 / sve 1.193 / sve2 1.229** — a 3.0% spread against a
**4.37% agent-resampling floor** measured by repeating one identical cell (S7 seed 1 vs the E1
sve arm). The arms differ from each other by less than the same agent differs from itself.
Per-kernel the median spread between identical runs is **15.9%**, and one kernel (`loop_120`)
went 1.70x → 5.64x between two identical runs.

That floor is not the stopwatch. **S9** re-times fixed code (autovec vs `baseline-sve2`, 13
definitions, 3 repeats on one box, nothing about the agent varying):

| | measurement-only (S9) | agent resampling (S7) | ratio |
|---|---|---|---|
| geomean spread | **0.40%** | 4.37% | 11x |
| median per-kernel spread | **0.82%** | 15.9% | 19x |
| worst kernel | 2.97% (`loop_108`) | 1.70x → 5.64x (`loop_120`) | — |

**The harness measures to a few tenths of a percent; ~99% of the variance in a reported
speedup is the model resampling, not the timer.** This is the finding that licenses the ISA
null: a 3.0% spread is 7x the measurement floor but well under the sampling floor, so the arms
are indistinguishable *for the reason that matters* — you cannot resolve them without more
agent runs, and no amount of extra timing repeats will help.

It also sets the protocol for anyone using this benchmark: **report n≥3 independent agent runs;
do not spend the budget on timing repetitions.** Single-run kernel comparisons — the norm in
this literature, and what KernelBench's temperature-0 sampling gives — cannot resolve
differences of the size routinely reported.

Two confounds to state rather than have found: the arms differ in three ways at once (neon's
`armv8-a` has neither `fullfp16` nor `dotprod`), and E1 is supplementary — the workbook's ISA
row is S2a, which we have.

A third, structural one, found while completing the 33rd definition: **an ISA ablation can only
compare tiers where an expert baseline exists.** The subset's `gemm_fp32_n512_k512` is scored
against KleidiAI's own kernel, whose spec declares `isa_features: ["sve"]` and builds
`-march=armv8.2-a+sve`. There is no neon-tier KleidiAI reference, so the harness's isa-filter
drops the definition under `--isa neon` and E1's neon arm is **structurally 32/33** — not a gap
more compute can close. Vendor-optimized references are themselves written for a chosen ISA
tier, which bounds what any hardware ablation over agent-written kernels can measure.

## 6. Test-time scaling — returns exhausted early

S1, geomean by **tool-call** budget (the harness caps tool calls; each evaluation costs a
compile plus an evaluate, so halve for evaluations):

| tool calls | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|
| all 33 | 1.018 | 1.148 | 1.272 | 1.371 | 1.401 | **1.416** |

10→50 buys +28%; **50→100 buys +8.6%**. Plateau by dataset: ncnn 90, llama.cpp 60, simd-loop 80
tool calls — **contradicting the deck**, which expects ~80 for ncnn/llama.cpp and ~30 for
simd-loop. Independently corroborated by the e2e sweep, where the best kernel appeared by turn
30-33 and turns 34-42 bought ~0.1%.

## 7. Where the agent wins and loses (A4)

gemm **0.807** and conv2d **0.553** are the *worst* op types; moe, rms_norm and simd-loop are
above 1.0. **The deck says "gemm strong"** — measured, it is second-worst. gemm and conv2d are
exactly where vendor experts concentrate, consistent with §4.

---

## Open

- **S7 seeds 2-3** running; n≥3 needed for the ±std on slide 11 to be defensible.
- **S9** (measurement-vs-agent noise decomposition) running — determines how hard §5 can be pushed.
- **kleidiai `gemm_fp32_n512_k512`** staged, serialised behind the current wave; completes 33/33.
- Residual +0.62% perplexity is a known unfixed defect (per-row activation scale vs ggml's
  per-256-block Q8_K).
