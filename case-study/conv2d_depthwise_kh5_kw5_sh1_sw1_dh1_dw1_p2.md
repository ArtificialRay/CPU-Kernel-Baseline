# Case Study: conv2d_depthwise_kh5_kw5_sh1_sw1_dh1_dw1_p2

5×5 depthwise conv2d, stride 1, dilation 1, padding 2. Output size equals input size (H_out = H, W_out = W). Benchmarked on Graviton4 (Neoverse V2, 128-bit SVE2, clang++-18 -O3 -march=armv9-a+sve2).

## Performance Trajectory

| Turn | Event | Version | time_speedup | cycle_speedup | IPC  | cache_misses |
|------|-------|---------|-------------|--------------|------|-------------|
| —    | Reference scalar | —  | 0.125×      | 0.136×       | —    | —           |
| 1–2  | Compile error    | —  | —           | —            | —    | —           |
| 3    | Compile OK       | v1 | —           | —            | —    | —           |
| 5    | Timing           | v1 | **0.673×**  | 0.703×       | 4.29 | 979         |
| 8    | Compile OK       | v2 | —           | —            | —    | —           |
| 9    | Correctness only | v2 | PASSED      | —            | —    | —           |
| 11   | Compile OK       | v3 | —           | —            | —    | —           |
| 15   | Timing           | v3 | 0.579×      | 0.626×       | 4.31 | 2703        |
| 17   | Compile OK       | v4 | —           | —            | —    | —           |
| 18   | Timing           | v4 | 0.653×      | 0.703×       | 4.44 | 2609        |
| 19   | **Submit**       | v4 | **0.653×**  | **0.703×**   | —    | —           |

All speedup values are `ncnn_baseline_time / candidate_time`. A value < 1 means the agent is **slower** than ncnn. v4 achieves `time_speedup = 0.653`, meaning it runs at 65.3% of ncnn's throughput — 1.53× slower. The submitted solution never surpasses ncnn, but it reaches 5.2× over the reference scalar.

---

## Section 1 — Reference Scalar vs. ncnn Baseline

**Direction: ncnn is 8× faster than reference scalar** (0.125× time speedup = ncnn baseline / scalar = 8.0×; 0.136× cycle speedup = ncnn is 7.4× faster in cycles).

### Why reference scalar is so slow

The scalar implementation in `bench-trace/solutions/ncnn/reference-scalar/conv2d_depthwise/conv2d_depthwise_kh5_kw5_sh1_sw1_dh1_dw1_p2.json` is a straightforward triple-nested loop:

```cpp
for (int kh = 0; kh < Kh; ++kh) {
    int ih = oh * Sh - pad + kh * Dh;
    if (ih < 0 || ih >= H) continue;
    for (int kw = 0; kw < Kw; ++kw) {
        int iw = ow * Sw - pad + kw * Dw;
        if (iw >= 0 && iw < W)
            sum += input[...] * weight[kh * Kw + kw];
    }
}
```

Three properties prevent auto-vectorization and generate purely serial code:
1. **Boundary branches on every element** — `if (ih < 0 || ih >= H)` and `if (iw >= 0 && iw < W)` introduce data-dependent control flow the compiler cannot eliminate without proving all accesses are interior.
2. **Outermost loop is over output pixels** — the innermost `kw` loop only runs 5 iterations and has a conditional write, blocking auto-vectorization.
3. **No SIMD** — 25 scalar FMAs per output pixel, completely serial.

For a representative workload (N=1, C=64, H=56, W=56): H_out × W_out × 25 × C = 56 × 56 × 25 × 64 = 5,017,600 scalar multiply-adds, all with branch overhead.

### What ncnn does instead

ncnn dispatches at `convolutiondepthwise_arm.cpp:372–374` (for C % 4 == 0):
```cpp
else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    convdw5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);
```

The harness (`baseline-ncnn-arm/.../kernel.cpp`) calls `convert_packing(bottom_blob, bottom_pack4, 4, opt4)` when C % 4 == 0 (all workloads except C=11). This rearranges the input from `(C, H, W)` to `(C/4, H, W, 4)` — four channels packed contiguously at every spatial position.

`convdw5x5s1_pack4_neon` (`convolutiondepthwise_5x5_pack4.h:4–740`) then:
1. **Processes 4 channels simultaneously** — each `vld1q_f32(r0)` loads 4 channels from one spatial location; `vmlaq_f32(_sum00, _k00, _r00)` accumulates all 4 channels at once.
2. **Dual-row tiling** — outer `i += 2` loop (`convolutiondepthwise_5x5_pack4.h:42`) processes `_sum00.._sum03` (row 0) and `_sum10.._sum13` (row 1) simultaneously, reusing input data across both output rows.
3. **4-column unroll** — inner `j += 4` (`convolutiondepthwise_5x5_pack4.h:46`) computes 4 output columns per iteration, making use of the 8 pre-loaded input vectors.
4. **No boundary checks** — the input is pre-padded by ncnn's `copy_make_border`, so the hot loop has no if-branches.

For C % 4 != 0 (workload C=11), it falls back to `convdw5x5s1_neon` (pack1), which uses hand-written AArch64 inline assembly (`convolutiondepthwise_5x5.h:75–378`) with `ext` instructions to generate all 5 horizontal shifts from two 4-element loads, processing 8 output columns × 2 output rows per loop iteration.

The effective SIMD parallelism of pack4: 4 channels × 4 columns × 2 rows = 32 output values per inner-tile. This is why ncnn is 8× faster than the fully serial scalar.

---

## Section 2 — ncnn Baseline's Own Bottlenecks

ncnn's pack4 path is genuinely efficient for this operator configuration. The main remaining bottleneck is the **pack4 re-packing overhead** at inference time: `convert_packing` must rearrange C × H × W floats from planar to channel-packed layout before computation, and then the output is re-packed back to planar (`convert_packing(top_pack4, local_top, 1, opt)`). For C=64, H=56, W=56 this moves 64 × 56 × 56 × 4 = 802,816 bytes in each direction.

A second bottleneck is the **dual-row tail and column tail**: the outer loop uses `i += 2` with a separate scalar `i += 1` remainder (`convolutiondepthwise_5x5_pack4.h:742`), and the inner loop uses `j += 4`, `j += 2`, `j += 1` tiers. For common workload sizes (H=56, W=56), H_out/2 and W_out/4 are exact, so these tails are small.

The pack1 fallback (`convdw5x5s1_neon`, used only for C=11) is inherently limited: `ext` produces shifted windows from 4-element vectors, but the AArch64 assembly only processes 4 floats per output element (no pack-4 channel fusion). For the C=11 workload its effective throughput is much lower, which pulls down the geomean.

There is no gather-load bottleneck because pack4 layout keeps channels contiguous: `vld1q_f32(r0)` is a single sequential load.

---

## Section 3 — Why the Agent Reaches 0.653× and No Further

Agent v4 achieves `time_speedup = 0.653`, i.e., it is 1.53× slower than ncnn. `cycle_speedup = 0.703` means it uses 42% more cycles for the same work.

### Algorithm used by ncnn for this definition

ncnn's pack4 path vectorises over the **channel dimension** (4 channels per NEON lane), then over output rows (2 at a time), then output columns (4 at a time). One inner-tile iteration computes 4C × 2H × 4W = 32 output values with 25 × 8 = 200 `vmlaq_f32` operations, where every weight vector `_k00` is shared across all 4 output column accumulators.

The agent's approach vectorises over the **width dimension** (vl=4 columns per SVE element). One SVE iteration computes 1C × 1H × 4W = 4 output values with 25 `svmla_f32_m`. ncnn therefore processes 8× more output values per similarly-sized instruction block.

### Arithmetic ceiling

For the representative workload (N=1, C=64, H=56, W=56), interior pixels (interior rows oh=2..53, interior cols ow=2..53):
- Interior output pixels per channel: 52 × 52 = 2704
- Total interior pixels: 64 × 2704 = 173,056
- Border pixels: 64 × (56 × 56 − 52 × 52) = 64 × 432 = 27,648 (14% of total)

The SVE loop handles 86% of output via 4-wide SIMD; 14% is scalar. A simple model:
```
1 / (0.86 × (1/4) + 0.14 × 1) = 1 / (0.215 + 0.14) = 2.8×  over reference scalar
```
This matches the observed 0.653 / 0.125 = **5.2×** — the model underestimates because the scalar branch elimination in the SVE path provides an additional throughput gain beyond just lane count.

The arithmetic ceiling against ncnn is determined by the channel-dimension advantage: ncnn processes 4 channels simultaneously with no per-channel loop overhead, while the agent loops over each channel sequentially. Even with perfect SVE execution for the interior, the agent's theoretical speedup cap is:
```
agent_peak / ncnn_peak ≈ (vl=4) / (pack4 × 2-row × 4-col efficiency) ≈ 4 / (4 × 2 × 4) = ~0.5×
```
The actual ratio is 0.65×, suggesting the agent partially compensates through Graviton4's out-of-order execution hiding the channel-loop overhead.

### Version-by-version analysis

**v1** (`v1.cpp:38–109`, turn 5: time=0.673×, cycle=0.703×, IPC=4.29, cache_misses=979)

The core SVE loop processes `vl=4` output columns per iteration with 25 `svmla_f32_m`:

```cpp
// v1.cpp:75–93
for (; ow + (int)vl + 2 <= W_out; ow += vl) {
    svfloat32_t acc = vbias;
    for (int kh = 0; kh < 5; ++kh) {
        if (!rows[kh]) continue;                           // null check per kh
        const float* in_row = rows[kh];
        acc = svmla_f32_m(pg_all, acc, svld1_f32(pg_all, in_row + ow - 2), svdup_f32(wvals[kh*5+0]));
        // ... 4 more kw offsets
    }
    svst1_f32(pg_all, out_row + ow, acc);
}
```

The `if (!rows[kh]) continue` is per-row (not per-pixel), and for interior rows (oh=2..H_out-2, ~87% of compute) all five `rows[kh]` are non-null. The branch predictor eliminates this check's cost on interior rows. `svdup_f32(wvals[...])` inside the ow loop is inlined into the instruction stream as an immediate — clang hoists these to constant registers. IPC=4.29 confirms the FMA units are nearly saturated. Cache miss count (979 geomean) is low because the access pattern follows a strict oh=0..H_out-1 sequential order, which the hardware prefetcher predicts well.

Left/right border columns (ow=0,1 and ow≥W_out-2) fall through to scalar, covering 4/W_out = 4/56 ≈ 7% of columns.

**v2** (turn 8–9, no timing recorded)

The agent compiled v2 and verified correctness only. The subsequent COMPILE_ERROR (turn 10) suggests v2.5 introduced an intrinsic incompatibility. No timing data is available.

**v3** (`v3.cpp:74–160`, turn 15: time=0.579×, cycle=0.626×, IPC=4.31, cache_misses=2703)

v3 is structurally almost identical to v1 for the interior SVE path — same 25 `svmla_f32_m` per iteration, same vl=4 — but makes two changes:

1. **Border row separation**: `scalar_row` lambda is called for oh=0,1 then oh=H_out-2,H_out-1 first, before the interior loop. This changes the input access order: border rows are touched twice (once in the scalar_row call, once as interior rows reference them as r0/r1/r3/r4). For C=64, H=56: border traversal touches `4 × W × C × sizeof(float) = 4 × 56 × 64 × 4 = 57,344 bytes` extra, polluting L1/L2 before interior computation starts.

2. **`svdup_f32` inside ow loop** (`v3.cpp:113–141`): Unlike v4, v3 recreates all 25 `svdup_f32` values on every ow iteration. With the compiler unable to prove these are loop-invariant (weight values come from local float variables w0..w24, which are loop-constant but may not be proven so after the lambda capture), some compilers recompute these inside the loop.

The result: cache_misses jump from 979 to 2703 (+2.8×) due to the double-touching of border rows across channels, degrading the geomean by 14%.

**v4** (`v4.cpp:130–189`, turn 18: time=0.653×, cycle=0.703×, IPC=4.44, cache_misses=2609)

v4 fixes the weight broadcast regression by explicitly hoisting all 25 weight vectors outside the ow loop:

```cpp
// v4.cpp:130–155
svfloat32_t vw0  = svdup_f32(wf[0]);   // hoisted once per output row
svfloat32_t vw1  = svdup_f32(wf[1]);
// ... 23 more
svfloat32_t vbias = svdup_f32(bias_val);

for (; ow + (int)vl <= ow_sve_end; ow += vl) {
    svfloat32_t acc = vbias;
    acc = svmla_f32_m(pg_all, acc, svld1_f32(pg_all, r0+ow-2), vw0);
    // ...
```

IPC rises to 4.44 (+3.5% over v1) confirming the compiler can now schedule these as register operands throughout. The 4-row tile structure adds an outer `t` loop (`v4.cpp:92–208`) but the actual per-row SVE code is unchanged from the single-row approach. cache_misses drop slightly to 2609 but remain elevated vs. v1's 979 because v4 retains the border-row separation, continuing to double-touch edge rows.

Time speedup is 0.653× vs. v1's 0.673×: v4's 4-row tile bookkeeping adds loop control overhead for small-H workloads (e.g., H=28 → interior only covers 24 rows, tile overhead matters more).

### Summary of remaining gap and how to close it

| Version | Channel SIMD | Row tile | Width SIMD | IPC  | Time speedup |
|---------|-------------|----------|-----------|------|-------------|
| ref-scalar | 1 | 1 | 1 | — | 0.125× |
| v1 | 1 | 1 | SVE vl=4 | 4.29 | 0.673× |
| v3 | 1 | 1 | SVE vl=4 | 4.31 | 0.579× |
| v4 (submitted) | 1 | 4 | SVE vl=4 | 4.44 | **0.653×** |
| ncnn pack4 | 4 (NEON) | 2 | 4 | — | 1.000× |

Closing the remaining 35% gap requires one or more of:
1. **Channel tiling**: process 4 channels per SVE vector (requires interleaved channel layout, matching ncnn's pack4 re-packing cost).
2. **Winograd F(2,3) or F(4,3)**: reduces 25 multiplies per output to 9 (F(2,3)) at the cost of transform passes. At H_out=W_out=56 the transform overhead amortises favourably.
3. **`ext`-based width sliding**: replace 5 overlapping `svld1` loads per kw row with 1 base load + 4 `svext` shifts, halving load-port pressure for the interior kw dimension.

SIMD tuning alone (hoisting, unrolling) cannot bridge the gap — the architectural advantage of 4-channel parallelism is fundamental.

---

## Section 4 — Hardcoded Parameters in the Submitted Solution

The agent receives the following constexpr header (`conv2d_depthwise.h`):
```cpp
namespace conv2d_depthwise_def {
constexpr int Kh  = 5;  constexpr int Kw  = 5;
constexpr int Sh  = 1;  constexpr int Sw  = 1;
constexpr int Dh  = 1;  constexpr int Dw  = 1;
constexpr int pad = 2;
}
```

These are available as named constants. The agent code uses them via `using namespace conv2d_depthwise_def;`, but then proceeds to hardcode several parameters structurally rather than deriving them from the constexpr values.

### Kh = 5 and Kw = 5 — Algorithmic hardcode

The 5×5 kernel shape is embedded in every structural decision:
- `float wf[25]` (`v4.cpp:49`) — array size 25 = Kh × Kw
- Five explicit input row pointers `r0..r4` (`v4.cpp:94–98`) — the loop `for (int i = 0; i < tile + 4; ++i)` relies on exactly 5 valid rows
- 25 explicit `svmla_f32_m` calls (`v4.cpp:160–188`) — the entire inner loop body is unrolled for exactly Kh=5, Kw=5
- The border scalar computations (`v4.cpp:102–119`) hard-enumerate weights `wf[2]`, `wf[3]`, `wf[4]`, etc. for ow=0 and ow=1, which are specific to the kernel shape and pad

**Effect of changing**: Any other kernel size requires rewriting the entire inner loop structure.

### pad = 2 — Algorithmic hardcode

Pad=2 is embedded in three places:
- `ow - 2` in all `svld1_f32` load addresses (`v4.cpp:160,161,...`) — the -2 offset is pad, not derived from `pad`
- `oh_int_start = 2` and `oh_int_end = H_out - 2` (`v4.cpp:75–76`) — the 2 is pad, selecting which rows have all kernel rows valid
- The scalar ow=0 and ow=1 border accumulators (`v4.cpp:102–119`) hard-code which weights apply at those positions for a 5×5 kernel with pad=2

**Effect of changing**: Wrong load addresses (ow - 2 would be off-by-(pad-2)) and wrong border row range.

### Sh = 1 and Sw = 1 — Implicit hardcode

With pad=2, Kh=5, Sh=1: H_out = (H + 4 − 5)/1 + 1 = H. So H_out == H and W_out == W. The code uses `W_out` as the output stride consistently (`v4.cpp:93,127`), which is correct. However, `r0 = in_c + (oh - 2) * W` derives input row offset using `W` (input width), which equals `W_out` only when Sw=1. If Sw > 1, H_out < H and the pointer arithmetic would read the wrong input row.

**Effect of changing**: Wrong output if Sw ≠ 1 (reads wrong input rows via `(oh - 2) * W` where `W` is input width but the stride should be `(oh * Sh - pad) / Sh * W`).

### Dh = 1 and Dw = 1 — Algorithmic hardcode

Dilation is fixed at 1 everywhere: `in_row + ow - 2` loads positions `(ow-2, ow-1, ow, ow+1, ow+2)`, which are the kernel positions for dilation=1. For dilation=2 the offsets should be `(ow-4, ow-2, ow, ow+2, ow+4)`.

**Effect of changing**: Wrong results; the 5 loads per kh row access the wrong input positions.

### Summary Table

| Parameter | Hardcode type | Effect if changed |
|-----------|--------------|-------------------|
| Kh = 5 | Algorithmic | Wrong results; array size, row count, inner loop all assume 5 |
| Kw = 5 | Algorithmic | Wrong results; 25-element unroll, weight indexing assume 5 |
| pad = 2 | Algorithmic | Wrong load offsets (`ow - 2`), wrong interior row range (`[2, H_out-2)`) |
| Sh = 1 | Correctness bug | Wrong input row pointer for interior rows when H_out ≠ H |
| Sw = 1 | Correctness bug | Same pointer bug applies to column offsets in border scalar path |
| Dh = 1 | Algorithmic | Load offsets miss dilated positions |
| Dw = 1 | Algorithmic | Same as Dh |
