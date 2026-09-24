# Draft — §4.5 From kernel speedup to deployed speedup

Prose for the paper, against the definitive run (2026-09-24). Figures referenced by the
filenames `scripts/e2e/plot_e2e.py` emits into `sweep_backups/figs/`. Numbers here are final
unless marked. Frame: **a validation study of the benchmark's metric**, not a second
benchmark — one model, one quantization, one operator family, one target.

---

## 4.5 From kernel speedup to deployed speedup

Every result so far scores a kernel against a reference in isolation. Whether those scores
predict anything a practitioner would feel is a separate question, and one that a kernel-level
benchmark cannot answer about itself. We therefore take the kernels an agent wrote for a real
model, splice them back into the inference runtime they came from, and measure the model.

**Setup.** We target Qwen3.5-4B-Q4_K_M under llama.cpp on Graviton4 (AWS `c8g.4xlarge`). Eleven
GEMM shapes — the `mul_mat` calls that dominate this model's CPU inference, spanning Q4_K, Q5_K
and Q6_K — are generated from the model's own GGUF header, optimized by the agent under the
standard benchmark protocol, and loaded back into llama.cpp through a dispatch hook that
intercepts `GGML_OP_MUL_MAT` for exactly those shapes. The comparison is against stock
llama.cpp **with its repack fast path enabled**: disabling it costs stock 46% of prefill
(Fig. `throughput`, `norepack`), so it would be a straw baseline. We report throughput with
`llama-bench` and quality as wikitext-2 perplexity over 64 chunks.

**A conventional correctness gate certifies kernels that destroy the model.** Under random
inputs and a fixed 20 dB SQNR floor — the standard practice this literature uses, and what our
own harness used initially — all eleven kernels passed. Deployed together they raise perplexity
from 10.06 to **14.45, +44%** (Fig. `gate`). Overriding one kernel at a time isolates three as
the cause: the FFN gate/up projection (11.51), the GDN input projection (11.23) and the GDN
z-gate (10.87). These are precisely the kernels fed the normalized hidden state, where LLM
activations carry large channel outliers; the remaining eight stay within 0.8% of stock. The
defect is a shared-exponent requantization of the Q8_K block sums on the M≥2 batched path,
invisible to random data and ruinous on real activations. A reference-scalar kernel spliced
through the same hook reproduces stock exactly, so the splice is not implicated.

This motivates the gate used for the rest of this paper: workloads drawn from activations
dumped from real model execution, and an SQNR floor defined *relative to the reference's own*
rather than at an absolute 20 dB. Under that gate the agent re-optimizes the affected shapes
and the eleven kernels retain a 2.94× kernel-level geometric mean while becoming deployable.

**What the speedups are worth.** With the repaired set, prefill improves **1.33×** and decode
**1.08×** at 16 threads, for **+0.6%** perplexity (Table X). The gain holds across thread counts
(1.41× / 1.39× / 1.33× at 1 / 4 / 16 threads) and across micro-batch sizes, but it is a curve,
not a point: prefill decays from 1.377× at 512 prompt tokens to 1.279× at 8192
(Fig. `prefill_decay`), as attention takes a growing share of the work the hook does not touch.

The quality cost is best read against the quantization ladder rather than in isolation
(Fig. `ppl_ladder`). On the same 64-chunk measurement, Q4_K_M itself costs +1.27% over BF16 and
one further level down (Q3_K_M) costs +3.61%. **The agent's kernels cost +0.62% — roughly half
of one quantization step**, for a 33% prefill improvement. Two independent 64-chunk runs two
days apart return 9.5617 both times.

**Kernel-level score does not rank models the way deployment does.** Two models optimized the
same eleven kernels under the same protocol, gate and hardware (Fig. `model_vs_deployed`).
Claude Fable 5.1 reaches 2.93× at kernel level and 1.33× deployed. gpt-5.6-sol reaches 1.20× at
kernel level — a score that reads as a modest success in Table 2 — and **0.59× deployed: the
model runs slower than with no agent kernels at all.** The mechanism is visible in the same
figure: Sol's kernels beat the generic ggml path by ~1.2×, but the repack path they displace
already beats it by ~1.85×, so a genuine kernel-level win is a deployment loss. Sol also
submitted the vendor baseline itself as its own answer for one kernel, a second and independent
way a conventional metric can be satisfied without the work being done.

**Scope.** This is a validation of the metric on one model, one quantization, one operator
family and one target, not a deployment benchmark. It does not establish that the ranking
inverts in general. It does establish that it *can* invert, that a standard correctness gate
does not prevent it, and that neither is visible from kernel-level scores alone.

---

## Where this changes the rest of the paper

- **Abstract / contributions** (both `[TODO]`): this is the fourth contribution — the benchmark
  is validated end to end on a real model, and that validation caught both a correctness-gate
  failure and a metric-ranking inversion.
- **§2.3 Measurement and integrity** asserts "Correctness is checked before performance."
  Forward-reference §4.5 there: that check is load-bearing, and the naive version of it fails.
- **§7 Limitations**, item (5) currently reads "Kernel-level speedups may not translate to
  end-to-end model latency." Delete it — it is now a result. Replace with the scope limit above.
- **§4.1** claims agents are "competitive with optimized kernels." The Sol arm is a direct
  counterexample at deployment; soften to match §4.3's weak/strong split.

## Numbers, for the table

| | stock | agent | ratio |
|---|---|---|---|
| prefill @512, 16 threads | 158.2 tok/s | 210.3 | **1.33×** |
| decode, 16 threads | 39.07 tok/s | 42.10 | **1.08×** |
| prefill @8192 (`-fa off`) | 138.5 | 177.1 | 1.279× |
| perplexity, wikitext-2, 64 chunks | 9.503 | 9.562 | **+0.6%** |

Provenance: `~/e2e/definitive/table_20260924T1457Z.txt`; ladder from `~/e2e/ppl64/`; gate
attribution from `docs/e2e_qwen35.md` (8 chunks — do not mix the two perplexity scales in one
figure without saying so; stock reads 10.06 there and 9.503 at 64 chunks).
