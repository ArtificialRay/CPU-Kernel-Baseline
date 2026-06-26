# Case Study: `deconv2d_kh3_kw3_sh1_sw1_cout256`

**Speedup trajectory** (all values relative to ncnn-baseline; second `evaluate` call per compile is canonical):

| Version | Time speedup | Cycle speedup | IPC  | Cache misses (mean) | Key change |
|---------|-------------|---------------|------|---------------------|-----------|
| reference-scalar | 1.785× | 1.785× | — | — | Scatter-write, scalar C++ |
| v1 | 0.717× | 0.713× | 2.21 | 323.8 M | SVE attempted, inner co-loop stays scalar |
| v2 | 8.694× | 8.669× | 2.88 | 7.97 M | Weight transpose + local acc + SVE FMLA |
| v3 | 8.762× | 8.731× | 3.32 | 4.52 M | OW\_TILE=4 |
| **v4 (submitted)** | **11.134×** | **11.101×** | 2.35 | 2.38 M | OW\_TILE=8 + 4-vector FMA unroll |

---

## 1. Reference-scalar vs ncnn-baseline: scalar is 1.785× faster than ncnn

The ncnn baseline for this definition dispatches to `deconv3x3s1_neon`
([`ncnn/src/layer/arm/deconvolution_arm.cpp:629–637`](../ncnn/src/layer/arm/deconvolution_arm.cpp#L629-L637)):

```cpp
if (elempack == 1 && out_elempack == 1)
{
    if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && ...)
    {
        deconv3x3s1_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, opt);
```

`deconv3x3s1_neon` is in [`deconvolution_3x3.h:4–128`](../ncnn/src/layer/arm/deconvolution_3x3.h#L4-L128).
Its loop order is **`outch → inch → h → w (4-wide NEON)`**. For each group of 4 input
pixels, it scatters updates to 3×3 output positions using sequential load-modify-store:

```cpp
// deconvolution_3x3.h:56-95 (inner j loop, one row shown)
float32x4_t _v = vld1q_f32(r0);                                     // inputs j..j+3

float32x4_t _out00 = vld1q_f32(outptr0 + 0);                        // output[j..j+3]
_out00 = vmlaq_lane_f32(_out00, _v, vget_low_f32(_k0), 0);
vst1q_f32(outptr0 + 0, _out00);                                      // ← STORE to j..j+3

float32x4_t _out01 = vld1q_f32(outptr0 + 1);   // output[j+1..j+4] ← OVERLAPS previous store!
_out01 = vmlaq_lane_f32(_out01, _v, vget_low_f32(_k0), 1);
vst1q_f32(outptr0 + 1, _out01);                                      // ← STORE to j+1..j+4

float32x4_t _out02 = vld1q_f32(outptr0 + 2);   // output[j+2..j+5] ← OVERLAPS previous two stores
_out02 = vmlaq_lane_f32(_out02, _v, vget_high_f32(_k0), 0);
vst1q_f32(outptr0 + 2, _out02);
// ... repeated for outptr1 (row i+1) and outptr2 (row i+2)
```

This is semantically correct: `_out01` must read the value just written by `_out00` so
that position j+1 accumulates the kernel-col-0 contribution from input pixel j+1 **and**
the kernel-col-1 contribution from input pixel j. However, it forces
**store-to-load forwarding** on every overlapping element. On Neoverse V2, store-to-load
forwarding takes ~4 cycles, serializing the 9-deep chain of `{load → FMA → store}`
operations within each j-block (3 stores per kernel row × 3 kernel rows). Out-of-order
execution cannot overlap these because each load partially aliases the immediately
preceding store.

The second structural issue is that the outer loop is **output-channel-first
(`outch=256`)**: for each output channel `p`, all `inch=64` input channels are visited,
re-reading the full H×W input planes each time. While individual input planes (~12 KB)
stay in L1 between `q` iterations, traversing 256 output channels sequentially means the
256 output buffers (~3.5 MB total) cycle through L3 rather than L2.

The reference scalar ([`bench-trace/solutions/ncnn/reference-scalar/deconv2d/deconv2d_kh3_kw3_sh1_sw1_cout256.json`](../bench-trace/solutions/ncnn/reference-scalar/deconv2d/deconv2d_kh3_kw3_sh1_sw1_cout256.json))
loops as **`ci → ih → iw → co → kh → kw`**:

```cpp
for (int co = 0; co < Cout; ++co) {
    float* out_co = out_n + (long)co * H_out * W_out;
    const float* w_co_ci = weight + ((long)co * C_in + ci) * Kh * Kw;
    for (int kh = 0; kh < Kh; ++kh)
        for (int kw = 0; kw < Kw; ++kw)
            out_co[(ih+kh)*W_out + (iw+kw)] += in_val * w_co_ci[kh*Kw+kw];
}
```

The inner `kh, kw` writes are to a **single output channel** (no overlapping stores across
iterations), and the 9 memory locations are independent of one another. The CPU can
retire them in parallel. Weight accesses `w_co_ci[0..8]` are sequential (9 floats). Both
factors avoid the serialization that plagues the ncnn NEON code.

**Result**: scalar executes the same arithmetic but with independent, parallelisable
stores, achieving 1.785× the throughput of ncnn's overlapping-store NEON.

---

## 2. ncnn-baseline's own bottlenecks

### 2a. Store-to-load forwarding chains

As shown above, `deconv3x3s1_neon` interleaves stores and overlapping loads 9 times per
4-pixel j-block. Neoverse V2 store-to-load forwarding latency is ~4 cycles; with FMA
latency also ~4 cycles, each chained step costs ~8 cycles. The three kernel-row chains
(`outptr0`, `outptr1`, `outptr2`) are independent of each other and can be partially
overlapped by OOO execution, but within each row there is a 2-deep forwarding dependency
(`+0 → +1 → +2`), serialising ~8 + 4 cycles of work that could otherwise overlap.

### 2b. No output-channel SIMD

The NEON lane used for vectorisation is **input width (4-wide)**, not output channels.
With Cout=256, there is no mechanism in `deconv3x3s1_neon` to process multiple output
channels simultaneously; each `p` iteration runs completely independently, serialised by
the outer OpenMP loop (single-threaded in the benchmark). An 8-wide SVE instruction over
the Cout dimension would compute 8× more output channels per FMA without any overlapping
store dependency, but ncnn's `deconv3x3s1_neon` does not exploit this axis.

### 2c. kptr re-walk per output channel

For each output channel `p`, the weight pointer `kernel0 = kernel + p * inch * 9 + q * 9`
advances linearly through `inch × 9` floats per `q` loop. With `inch=64` and `Kh×Kw=9`,
the weight data for one output channel is 64 × 9 × 4 = 2.3 KB. For 256 channels this
totals 589 KB — comfortably in L2 — so weight re-loading is not a primary bottleneck,
but it contributes to pressure alongside the store-forwarding chains.

---

## 3. Agent version progression

Both the agent's direct algorithm and ncnn's `deconv3x3s1_neon` compute the same number
of multiply-accumulates. For the largest workload (N=1, C_in=128, H=28, W=28):

```
FMAs = Cout × C_in × Kh × Kw × H × W = 256 × 128 × 9 × 28 × 28 ≈ 231 M
```

All speedup comes from **execution efficiency**, not reduced arithmetic.

### v1 — SVE attempted, but co loop remains scalar (0.717×)

v1 reorganises to loop order `n → ci → ih → iw → kh → kw → co`, putting `co` innermost:

```cpp
// v1.cpp:85-88
const float* w_base = weight + w_ci_kh_kw;
for (int co = 0; co < Cout; ++co) {
    out_ptr[co * HW_out] += in_val * w_base[co * w_stride];
}
```

Weight stride = `C_in × Kh × Kw = 576 floats`; output stride = `HW_out ≈ 3364 floats`.
Both are large scatter accesses. The comment in v1 reads: *"Actually, let's just do
scalar for now."* No SVE instruction actually appears in the inner loop. Cache misses
spike to **323.8 M** — over 40× higher than v2 — because every co iteration chases two
widely-separated cache lines. v1 is 1.39× slower than the reference scalar (and 0.717×
relative to ncnn), purely from poor memory access patterns.

### v2 — Weight transpose + local accumulator + SVE FMLA (8.694×)

This version identifies and fixes the root cause. Two structural changes:

**① Weight transpose** `(Cout, C_in, Kh, Kw) → (C_in, Kh, Kw, Cout)`:

```cpp
// v2.cpp:53-60
for (int co = 0; co < Cout; ++co) {
    for (int idx = 0; idx < CinKhKw; ++idx) {
        weight_T[idx * Cout + co] = weight[co * CinKhKw + idx];
    }
}
```

After transposition, `weight_T[(ci*KhKw + kh*Kw + kw) * Cout + co]` is **contiguous in
`co`** for any fixed `(ci, kh, kw)`. A single `svld1_f32` with `svptrue_b32()` now loads
8 consecutive weight values across 8 output channels.

**② Local accumulator + SVE FMLA**:

```cpp
// v2.cpp:111-140  (per output position oh, ow)
float acc[256] __attribute__((aligned(64)));
// ... init from bias ...
for (int ci = 0; ci < C_in; ++ci) {
    float in_val = ...;
    svfloat32_t vin = svdup_f32(in_val);              // broadcast 1 scalar
    const float* wptr2 = weight_T + (ci * KhKw + kh * Kw + kw) * Cout;
    for (int co = 0; co < Cout; co += 8) {            // step = svcntw() = 8
        svfloat32_t vacc = svld1_f32(svptrue_b32(), acc + co);
        svfloat32_t vw   = svld1_f32(svptrue_b32(), wptr2 + co); // contiguous!
        vacc = svmla_f32_m(svptrue_b32(), vacc, vin, vw);         // 8-wide FMA
        svst1_f32(svptrue_b32(), acc + co, vacc);
    }
}
// scatter acc → output only once per (oh, ow)
for (int co = 0; co < Cout; ++co)
    out_n[(long)co * HW_out + out_pos] = acc[co];
```

The inner SVE loop accesses `acc[co]` and `weight_T[...co]` both sequentially. The 256-float
`acc` array (1 KB) stays in L1 across all `(ci, kh, kw)` iterations for one `(oh, ow)`.
The single scatter-write to output runs once per output position, amortising its cost
over the full `C_in × Kh × Kw = 576` FMA iterations.

**Effect**: cache misses drop from 323.8 M to **7.97 M** (40×); IPC rises from 2.21 to
2.88; time speedup jumps from 0.717× to **8.694×**.

### v3 — OW\_TILE=4 (8.762×)

v3 processes 4 output columns simultaneously, with separate accumulators `acc0..acc3`:

```cpp
// v3.cpp:92-105
float acc0[256], acc1[256], acc2[256], acc3[256]; // 4 × 1 KB in L1
// load from output (already has bias)
for (int co = 0; co < Cout; ++co) {
    long base = (long)co * HW_out + oh * W_out;
    acc0[co] = out_n[base + ow];   acc1[co] = out_n[base + ow + 1];
    acc2[co] = out_n[base + ow + 2]; acc3[co] = out_n[base + ow + 3];
}
// inner (kh, kw, ci) loop: same weight_T slice applied to 4 input values
if (v0) { svfloat32_t vin0 = svdup_f32(in_row[iw0]); /* FMLA into acc0 */ }
if (v1) { svfloat32_t vin1 = svdup_f32(in_row[iw1]); /* FMLA into acc1 */ }
// ...
```

For a fixed `(ci, kh, kw)` the weight slice `weight_T[... + co]` is loaded once and
reused for up to 4 input values. The weight-to-FMA ratio improves 4×. Cache misses fall
to **4.52 M**; IPC rises to 3.32. Time speedup: **8.762×** — only +0.7% over v2 because
the improvement is partially cancelled by the overhead of conditional `if (vN)` branches.

### v4 — OW\_TILE=8 + 4-vector FMA unroll (11.134×)

v4 doubles the tile width to 8 and unrolls the `co` loop 4 ways:

```cpp
// v4.cpp:139-172  (inner co chunk, per kh/kw/ci iteration)
for (int co = 0; co < 256; co += 32) {
    // Load 4 weight vectors (each covers 8 co values = 1 SVE register)
    svfloat32_t vw0 = svld1_f32(pg, wp + co);
    svfloat32_t vw1 = svld1_f32(pg, wp + co + 8);
    svfloat32_t vw2 = svld1_f32(pg, wp + co + 16);
    svfloat32_t vw3 = svld1_f32(pg, wp + co + 24);

    // Apply same 4 weight vectors to all 8 tile columns via DO_TILE macro
    DO_TILE(acc0, 0)  DO_TILE(acc1, 1)  DO_TILE(acc2, 2)  DO_TILE(acc3, 3)
    DO_TILE(acc4, 4)  DO_TILE(acc5, 5)  DO_TILE(acc6, 6)  DO_TILE(acc7, 7)
}
// DO_TILE expands to: 4 loads from acc, 4 FMLA (with preloaded vw0..vw3), 4 stores
```

Each pass through the outer `co += 32` loop:
- loads 4 weight vectors (32 floats, one `svld1_f32` each)
- applies them to up to 8 accumulators via 32 SVMLA instructions (4 per tile column × 8 columns)

This gives a **weight-load to FMA ratio of 4 loads : 32 FMAs = 1:8**, vs v3's 1:1 ratio
(one weight load per FMA). The 4-vector unroll also chains four independent FMA
dependency chains, hiding the 4-cycle FMA latency on Neoverse V2.

The IPC **drops** from 3.32 to 2.35 despite faster wallclock time. The wider tile (8 ×
1 KB accumulators = 8 KB) fills more of L1 and increases load/store port pressure; the
execution bottleneck shifts from the FMA pipeline to memory bandwidth. However, the
reduction in total loop iterations (8× fewer outer ow steps) and the drop in cache
misses to **2.38 M** (−47% vs v3) more than compensate.

**Version summary**:

| Version | OW\_TILE | co unroll | IPC  | Cache misses | Time speedup |
|---------|----------|-----------|------|-------------|-------------|
| v1 | 1 | 1 (scalar) | 2.21 | 323.8 M | 0.717× |
| v2 | 1 | 1 (SVE ×8) | 2.88 | 7.97 M | 8.694× |
| v3 | 4 | 1 (SVE ×8) | 3.32 | 4.52 M | 8.762× |
| **v4** | **8** | **4 (SVE ×32)** | 2.35 | 2.38 M | **11.134×** |

### Remaining gap

The arithmetic ceiling is 1.0 — both the agent and ncnn execute the same number of FMAs.
The 11.1× speedup over ncnn is purely execution efficiency: the agent eliminates the
ncnn NEON store-forwarding hazards and achieves much wider SIMD utilisation (8-wide SVE
over Cout vs 4-wide NEON over width). Further gains could come from:

- Wider OW\_TILE (16 or 32): amortises scatter write and weight load further, if L1 can
  absorb 16+ × 1 KB accumulators
- Software-prefetch of weight\_T: for large C\_in (128), weight\_T for one (kh,kw) slice
  = 128 × 256 × 4 = 128 KB, which spills L1; prefetching could reduce the remaining 2.38 M
  cache misses
- No algorithmic change is needed — unlike Winograd convolutions, deconv3×3s1 with
  direct output-channel accumulation is already the optimal algorithm for this kernel.

---

## 4. Hardcoded parameters in the submitted kernel (v4)

### Cout = 256

Every version uses the `Cout` constant from the header namespace for loop bounds.
However, v4's inner co loop hardcodes the literal `256` and step `32`:

```cpp
// v4.cpp:139
for (int co = 0; co < 256; co += 32) {
```

The accumulator arrays are also fixed-size:
```cpp
// v4.cpp:81-88
float acc0[256] __attribute__((aligned(64)));
// ... × 8
```

If `Cout` changes, the loop terminates at the wrong place and the stack arrays overflow.
v2 avoids this by using `Cout` from the namespace and stepping by `svcntw()`, but v4
trades generality for throughput.

### Sh = 1 and Sw = 1 (algorithmic)

The coordinate mapping in all versions assumes unit strides:

```cpp
// v4.cpp:105, 109 (representative)
for (int kh = 0; kh < Kh; ++kh) {
    int ih = oh - kh;          // correct only when Sh = 1
    ...
    int iw_base = ow - kw;     // correct only when Sw = 1
```

For stride S, the correct formula is: given output position oh, input pixel ih
contributes if `ih * Sh + kh == oh` ↔ `ih = (oh - kh) / Sh` with remainder check
`(oh - kh) % Sh == 0`. This check is absent in all versions. Changing Sh or Sw to any
value > 1 produces wrong results. (The harness definition header exposes `Sh` and `Sw`
as constexpr constants so the fix is straightforward.)

### Dh = 1 and Dw = 1 (implicit)

None of the versions include a dilation factor in the coordinate computation. For
dilation `Dh > 1`, the kernel extent grows to `Dh * (Kh - 1) + 1` and the input
coordinate formula becomes `ih = (oh - kh * Dh) / Sh`. This is not a concern for
`deconv2d_kh3_kw3_sh1_sw1_cout256` (where `Dh = Dw = 1`), but the kernel would be
algorithmically wrong on any definition with dilation > 1.

### SVE vector width = 256-bit (svcntw() == 8)

v3 and v4 offset weight vectors by multiples of 8 floats and step the co loop by 32:

```cpp
// v4.cpp:141-144
svfloat32_t vw1 = svld1_f32(pg, wp + co + 8);   // assumes 8 floats per vector
svfloat32_t vw2 = svld1_f32(pg, wp + co + 16);
svfloat32_t vw3 = svld1_f32(pg, wp + co + 24);
```

This is correct only when `svcntw() == 8` (256-bit SVE). On a machine with 128-bit
SVE2 (`svcntw() == 4`), adjacent offset groups would overlap (e.g., `wp + co` covers
[co..co+3] and `wp + co + 8` starts at co+8, skipping [co+4..co+7]). The kernel would
produce wrong results silently. v2 avoids this by stepping the co loop by `svcntw()`.

### Summary

| Parameter | Hardcode type | Effect of changing |
|-----------|--------------|-------------------|
| Cout = 256 | Size (v3/v4) | Wrong results; stack arrays may overflow; loop stops early or late |
| Sh = 1 | Algorithmic (all versions) | Wrong input coordinate mapping; silent incorrect output |
| Sw = 1 | Algorithmic (all versions) | Wrong input coordinate mapping; silent incorrect output |
| Dh = Dw = 1 | Algorithmic (all versions) | Wrong if dilation > 1 |
| SVE width = 256-bit | Structural (v3/v4) | Wrong output on 128-bit SVE hardware |
| Kh = 3, Kw = 3 | **Not hardcoded** | Loop bounds use `Kh`/`Kw` from namespace |
| N, C_in, H, W, H_out, W_out | **Not hardcoded** | All variable-dim arguments handled correctly |
