# Derived analyses (workbook sheet C) — no new runs

Produced 2026-09-24 from existing S1 / S2a / E1 logs. Sources: W&B project
`ArmBench/arm-bench-kernels-gpt5.6-luna`; S1 trajectories in
`sweep_backups/2026-09-17_s2a_s1/agent-runs-s1/`.

**Unit warning.** The workbook says "steps". In this harness `--max-iterations` caps
*tool calls*, and each evaluation costs a compile plus an evaluate, so 110 tool calls buys
~55 evaluations. Every table below is in **tool calls**; halve for evaluations.

## A1 — convergence per dataset (S1, n=33)

| tool calls | ncnn | llama.cpp | simd-loop | all |
|---|---|---|---|---|
| 10 | 0.667 | 0.924 | 1.597 | 1.018 |
| 20 | 0.796 | 1.140 | 1.617 | 1.148 |
| 40 | 0.985 | 1.252 | 1.626 | 1.272 |
| 60 | 1.160 | 1.294 | 1.659 | 1.371 |
| 80 | 1.166 | 1.295 | 1.743 | 1.401 |
| 100 | 1.188 | 1.295 | 1.757 | 1.416 |

Plateau (within 1% of the 100-call value): **ncnn 90, llama.cpp 60, simd-loop 80** tool calls
— i.e. ~45 / ~30 / ~40 evaluations.

**This contradicts the deck**, which expected ~80 steps for ncnn/llama.cpp and ~30 for
simd-loop. llama.cpp converges *faster* than expected (~30 evaluations) and simd-loop
*slower* (~40). Whichever unit the deck meant, simd-loop is not the quick one.

## A2 — speedup after dropping kernels the baseline already wins

| arm | dataset | kept/total | all | kept only |
|---|---|---|---|---|
| sve (E1) | ncnn | 7/12 | 1.015 | 1.813 |
| sve (E1) | llama.cpp | 3/8 | 1.083 | 4.261 |
| sve (E1) | simd-loop | 10/13 | 1.609 | 2.001 |
| sve (E1) | **ALL** | **20/33** | **1.236** | **2.165** |
| neon (E1) | ALL | 18/33 | 1.270 | 2.399 |
| sve2 (S2a) | ALL | 20/33 | 1.299 | 2.130 |
| sve 110-call (S1) | ALL | 22/33 | 1.435 | 2.201 |

**The agent loses to the expert baseline on a third to a half of all kernels**, and on
llama.cpp it wins only 3 of 8 in every arm. Where it wins it wins big (2.1-2.4x), which is
the shape of "fills gaps experts left" rather than "beats experts". Doubling the budget
(S1) converts two extra kernels from loss to win, 20/33 -> 22/33.

## A4 — geomean by op_type (E1 sve arm)

| op_type | n | geomean | min | max |
|---|---|---|---|---|
| rms_norm | 1 | 5.415 | | |
| mha | 1 | 1.884 | | |
| simd-loop | 13 | 1.609 | 0.605 | 29.443 |
| conv2d_depthwise | 4 | 1.407 | 0.468 | 3.960 |
| pooling | 2 | 1.319 | 1.257 | 1.383 |
| moe | 3 | 1.142 | 0.443 | 7.582 |
| **gemm** | 4 | **0.807** | 0.341 | 3.395 |
| **conv2d** | 5 | **0.553** | 0.245 | 1.303 |

**This contradicts the deck too.** The deck has "gemm strong; moe and q4_k_m weak"; measured,
**gemm is one of the two worst op types (0.807)** and conv2d is the worst (0.553), while moe
is above 1.0. gemm and conv2d are exactly where vendor experts concentrate their effort, which
is consistent with A2: the agent loses where the baseline is strong.

Caveat: n is small per op_type (1-5 outside simd-loop), and the simd-loop column contains
`loop_121` at 29.4x, an outlier that dominates any aggregate containing it.
