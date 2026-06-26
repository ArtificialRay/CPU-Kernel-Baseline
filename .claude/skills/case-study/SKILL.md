---
name: case-study
description: Write a performance case study for an agent run on a specific definition; explains why speedup improved across versions, where the baseline bottlenecks, and which definition parameters are hardcoded
---

## case-study

Produce a self-contained English markdown document in `case-study/<def_name>.md` that
explains the full performance trajectory of an agent run: the performance relationship
between reference-scalar and ncnn-baseline and why it goes the direction it does, what
limits the ncnn baseline itself, why the agent versions improve, and which definition
parameters are hardcoded in the agent code.

---

## What to extract from the user's message

- **Definition name** — the specific problem (e.g. `conv1d_kw1_sw1_dw1_cout512_p0`).
  Derive the agent-run directory as `agent-runs/<def_name>/`.
  If not mentioned, ask before proceeding.
- **Op type** — inferred from the definition name prefix (e.g. `conv1d`, `conv2d`).
- **Focus** — all four sections by default; user may ask for a subset.

Valid invocations look like:
```
Write a case study for conv1d_kw1_sw1_dw1_cout512_p0
Case study the agent run for conv2d_kh3_kw3_sh1_sw1_cout256
```

---

## Data sources to read before writing

### 1. Agent run directory — `agent-runs/<def_name>/`

- **`trajectory.jsonl`** — one JSON object per agent tool call; each line has `turn`,
  `tool`, and `metrics`. Key metrics fields:
  - `time_speedup_geomean`, `cycle_speedup_geomean` — performance vs baseline, from `evaluate` turns
  - `ipc_mean` — instructions per cycle, from `evaluate` turns
  - `cache_misses_mean` — from `evaluate` turns
  - `status` — `"PASSED"` / `"COMPILE_ERROR"` / `"FAILED"`, from `compile` / `evaluate` turns
  - `source_file` — which `vN.cpp` was compiled, from `compile` turns
  - `time_speedup`, `cycle_speedup` — final submitted numbers, from `submit` turns

- **`vN.cpp` files** — the agent's kernel source for each version. Read all versions to
  understand the optimisation progression.

- **`vN.s` files** — disassembly the agent inspected; useful for confirming what the
  compiler actually generated.

### 2. ncnn ARM source — `ncnn/src/layer/arm/`

Find the ARM implementation for the op type:

| Op type            | Packed kernel header                           |
|--------------------|------------------------------------------------|
| conv1d             | `convolution1d_packed.h`                       |
| conv2d             | `convolution_packed.h`                         |
| conv2d_depthwise   | `convolutiondepthwise_packed.h`                |
| deconv2d           | `deconvolution_packed.h`                       |
| deconv2d_depthwise | `deconvolutiondepthwise_packed.h`              |

Read the relevant packed header to understand:
- The inner oc-tile loop structure (oc8, oc4, oc1 kernels)
- How the input is loaded (whether `elempack==1` forces gather via `vsetq_lane_f32`,
  or `elempack==4` allows contiguous `vld1q_f32`)
- Where `kptr` is defined and whether it is reset per output position

### 3. Solution header — baked into the solution JSON

The agent receives a `conv1d.h` (or equivalent) with constexpr parameters for this
definition:

```bash
python3 -c "
import json
with open('bench-trace/solutions/ncnn/claude-sonnet-4-6/<op_type>/<def_name>.json') as f:
    sol = json.load(f)
for s in sol['sources']:
    print('=== FILE:', s['path'], '===')
    print(s['content'][:500])
"
```

This shows exactly which constants the agent could exploit: `Cout`, `Kw`, `Sw`, etc.

---

## Four mandatory sections

### Section 1 — Reference-scalar vs ncnn-baseline: performance direction and cause

**First, determine the direction.** Read `trajectory.jsonl` for the reference-scalar entry
or check `bench-trace/solutions/ncnn/reference-scalar/<op_type>/` against the baseline
speedup recorded in the run. State the ratio explicitly: either "ncnn is N× faster than
scalar" or "scalar is N× faster than ncnn". Both are valid outcomes — op type and
definition parameters determine which way it goes.

---

**If ncnn is faster than scalar** — focus on what ncnn does right:

- `create_pipeline` pre-packs weights into interleaved format (e.g., `(Cout/8, Cin/8, 8×8)`)
  so that `kptr` walks sequentially with no gather during inference
- The oc-tile NEON kernel (oc8, oc4) uses `vfmaq_laneq_f32` or `vmlaq_laneq_f32` to
  process multiple output channels simultaneously
- Quote the inner FMA loop from the ncnn packed header with actual line numbers
- State the theoretical throughput multiplier (e.g., 8 OC × 4 floats = 32×) and the
  measured ratio (e.g., "scalar speedup = 0.174×, i.e., ncnn is 5.7× faster")

---

**If scalar is faster than ncnn** — identify ncnn's algorithmic inefficiency:

Common patterns that cause this:

1. **Output-space traversal with stride filtering** (typical in deconv with stride > 1):
   ncnn iterates over output pixels and checks each kernel position for phase alignment
   (`sys % stride_h != 0`). For stride S, only 1 in S² kernel positions contributes per
   output pixel — the rest hit `continue`. Compute the waste ratio: for kh=kw=sh=sw=2,
   75% of inner iterations are discarded. Quote the relevant ncnn source (file + line range).

2. **No SIMD in the relevant elempack path**: verify which `elempack` branch the
   benchmarking harness exercises. If it falls into `elempack == 1` and that path is
   purely scalar, state this explicitly.

3. **Expensive per-iteration integer arithmetic**: integer division and modulo (e.g.,
   `sys / stride_h`, `sys % stride_h`) inside the inner loop add ~10–20 cycles per
   iteration on ARM even when the branch is not taken.

For each inefficiency, quantify: what fraction of iterations are wasted, and how does
that map to the measured scalar-vs-ncnn ratio.

### Section 2 — ncnn-baseline's own bottlenecks

Identify what limits ncnn from reaching its theoretical peak. Common patterns:

**Gather load** (present when `elempack == 1`):
```cpp
// ncnn uses vsetq_lane_f32 to build input vectors lane-by-lane:
_r0 = vsetq_lane_f32(r0[0],     _r0, 0);   // depends on previous _r0
_r0 = vsetq_lane_f32(r0[N],     _r0, 1);   // serialised: must wait ≥3 cycles
_r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
_r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
```
This creates a 4-deep dependency chain (~12 cycle latency) that stalls all downstream FMAs.

**kptr reset** (present when the output-position loop is outermost):
```cpp
for (int j = 0; j < outw; j++) {
    const float* kptr = weight_data_tm.channel(p / 8);  // reset every j
    // ...
    kptr += 64;   // walked to end; repeated W_out times total
}
```
The weight block is re-walked `W_out` times, adding L1 pressure alongside the gather loads.

Not all bottlenecks appear in every op type — read the actual ncnn source and only
describe what is genuinely present for this definition's loop structure.

### Section 3 — Why the agent's best version beats/loses to ncnn-baseline

**Clarify the metric direction first.** `time_speedup = ncnn_baseline_time / candidate_time`:
a value > 1 means the agent is faster than ncnn; < 1 means the agent is slower. State
this outcome explicitly in the opening sentence so the reader cannot misread the number
(e.g., "Agent vN achieves `time_speedup = 0.64`, meaning it is 1.56× *slower* than ncnn").

**Step 1 — Identify the algorithm ncnn uses for this definition.**

Read the op type's `_arm.cpp` `create_pipeline` to find which dispatch branch fires for
this definition's exact parameters. The branch condition often keys on `kernel_w`,
`kernel_h`, `stride_w`, `dilation_w`, `num_input`, `num_output`. Name the algorithm
(e.g., Winograd, im2col+GEMM, direct packed) and in one sentence state why it reduces
arithmetic work or memory traffic relative to the naive direct approach the agent uses.

**Step 2 — Compute the arithmetic ceiling.**

Calculate multiply-add counts for the agent's direct approach vs ncnn's algorithm on a
representative workload from `bench-trace/workloads/<op_type>/<def_name>.jsonl`. If ncnn's
algorithm has its own overhead (transform passes, tile padding waste, workspace allocation),
note how that overhead changes across the workload range — small spatial sizes often close
the gap because overhead becomes a larger fraction of total work.

Express the ceiling as `direct_muls / ncnn_algorithm_muls`: this is the maximum speedup
achievable by pure SIMD/cache optimisation without changing the algorithm.

**Step 3 — Explain each agent version's contribution.**

Walk through each version using `cache_misses_mean` and `ipc_mean` from `trajectory.jsonl`.
Label each fix by its mechanism (vectorisation axis choice, loop reorder, register tiling,
boundary precompute, explicit register unrolling, etc.) and tie it to a concrete metric
change. Make explicit whether each gain comes from reducing arithmetic count, or from
improving execution/memory efficiency on the same arithmetic.
Also, explain any IPC drops between versions (e.g., a larger tile opens more concurrent
load/store ops, shifting the bottleneck from FMA throughput to memory bandwidth).

Show the version progression using numbers from `trajectory.jsonl`:

| Version | OC_TILE | W_out unroll | IPC  | Time speedup |
|---------|---------|--------------|------|-------------|
| v1      | ...     | ...          | ...  | ...×         |
| v2      | ...     | ...          | ...  | ...×         |
| vN      | ...     | ...          | ...  | **...×**     |

**Step 4 — State the remaining gap and what would be needed to close it.**

End with a concrete statement of the arithmetic ceiling and whether algorithmic parity is
achievable through further SIMD tuning alone, or whether it requires adopting the same
algorithm ncnn uses (e.g., implementing Winograd transforms, im2col, weight packing).

### Section 4 — Hardcoded parameters in the agent solution

Check each parameter from the definition name (Kw, Sw, Dw, pad, Cout) against the agent
code. For each one, determine:

1. **Is it hardcoded?** Look for:
   - A missing loop dimension (e.g., no `kw` loop → hardcodes Kw=1)
   - Using a derived size as a stride where the raw dimension should be used
     (e.g., `input + ic * W_out` instead of `input + ic * W` → hardcodes Sw=1 when W_out==W)
   - No tail handler for an OC tile (e.g., `oc += 16` with no remainder → hardcodes `Cout % 16 == 0`)
   - Parameter is genuinely irrelevant (Dw doesn't matter when Kw=1; pad is handled by harness)

2. **What breaks if changed?** Distinguish:
   - *Algorithmic*: the entire kernel structure is wrong (e.g., missing kw loop for Kw=1 assumption)
   - *Correctness bug*: wrong pointer arithmetic (e.g., `W_out` vs `W` as row stride)
   - *Minor*: out-of-bounds on edge case (e.g., missing tail for non-divisible Cout)
   - *Not hardcoded*: parameter handled externally or irrelevant

End with a summary table:

| Parameter | Hardcode type | Effect of changing |
|-----------|--------------|-------------------|
| Kw=1      | Algorithmic  | Wrong results; missing kw loop |
| ...       | ...          | ... |

---

## Output format

File: `case-study/<def_name>.md`

- Write in English
- Begin with a header table showing the speedup trajectory from `trajectory.jsonl`
- Each section uses `##` headings matching the four topics above
- Every claim about performance must cite a number from `trajectory.jsonl`
- Every claim about code behaviour must include the relevant code snippet with a file
  reference (filename and approximate line range); never describe code without quoting it
- Do not pad with generic micro-architecture background that isn't specific to this run

---

## Common investigation patterns

| Observation | Where to look |
|-------------|--------------|
| Large speedup gap between v1 and v2 | Compare OC_TILE, W_out unroll, and IPC between turns |
| IPC drops from vN to vN+1 despite faster time | New bottleneck (e.g., memory BW after FMA freed); explain |
| Low IPC in all versions | May indicate gather load stall even in agent code — check pointer arithmetic |
| `cache_misses_mean` spikes in one version | Tiling change broke cache locality; compare W_TILE or OC_TILE |
| Agent disassembles after evaluating | Inspecting compiler output to diagnose; note what the agent found and how it changed the next version |
| Multiple `evaluate` calls per compile | Agent re-measured for stability; use the second measurement as the canonical number |