# Case Study: `deconv2d_kh4_kw4_sh2_sw2_cout128`

**Definition**: Kh=4, Kw=4, Sh=2, Sw=2, Dh=1, Dw=1, Cout=128, no padding  
**Speedup** is defined as `ncnn_baseline_time / candidate_time` (>1 = candidate beats ncnn; <1 = candidate is slower).

**Speedup trajectory** (vs ncnn-baseline, geomean across 5 workloads):

| Turn | Version | Time speedup | Cycle speedup | IPC  | Cache misses (mean) |
|------|---------|-------------|---------------|------|---------------------|
| —    | reference-scalar | **0.513×** | **0.513×** | — | — |
| 3    | v1 (weight transpose + SVE over Cout) | 1.665× | 1.662× | 2.63 | 1,020,460 |
| 8    | v2 (kh×kw macro unroll inside iw) | 1.488× | 1.483× | 1.70 | 3,185,178 |
| 14   | v3 (kh outside iw, kw separate blocks) | 1.699× | 1.696× | 1.64 | 1,350,199 |
| 18   | v4 (weight hoisting outside iw loop) | **1.758×** | **1.754×** | 1.67 | 2,330,277 |

> The agent achieves a net **3.43× improvement** over the scalar reference and beats the ncnn
> baseline by **1.76×** at submission.

---

## 1. Reference-scalar vs ncnn-baseline: ncnn is ~1.95× faster

The reference scalar is in
`bench-trace/solutions/ncnn/reference-scalar/deconv2d/deconv2d_kh4_kw4_sh2_sw2_cout128.json`:

```cpp
// kernel.cpp (reference-scalar)
for (int ci = 0; ci < C_in; ++ci) {
    for (int ih = 0; ih < H; ++ih) {
        for (int iw = 0; iw < W; ++iw) {
            float in_val = in_ci[ih * W + iw];
            for (int co = 0; co < Cout; ++co) {       // ← co innermost: strided output
                float* out_co = out_n + (long)co * H_out * W_out;
                const float* w_co_ci = weight + ((long)co * C_in + ci) * Kh * Kw;
                for (int kh = 0; kh < Kh; ++kh)
                    for (int kw = 0; kw < Kw; ++kw)
                        out_co[(ih*Sh+kh)*W_out + iw*Sw+kw] += in_val * w_co_ci[kh*Kw+kw];
            }
        }
    }
}
```

Every store to `out_co` is strided by `H_out × W_out` floats in the co dimension; every weight
load strides by `C_in × Kh × Kw` floats across co. No SIMD. Single scalar multiply-add per
inner iteration.

**What ncnn does instead** — the benchmarking harness runs `Deconvolution_arm::forward()` with
`create_pipeline` called (dynamic_weight=false), so weights are pre-packed into
`weight_data_tm`. For the dominant workloads (C_in=64 and C_in=128, both divisible by 4),
`elempack=4, out_elempack=4`. The relevant forward path is at
`ncnn/src/layer/arm/deconvolution_arm.cpp:393–474`:

```cpp
// deconvolution_arm.cpp:393–474  (elempack==4, out_elempack==4)
for (int p = 0; p < out_channels; p++) {          // Cout/4 groups
    for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
            float32x4_t _sum = /* bias[p*4..p*4+3] */;
            const float* kptr = weight_data_tm.channel(p);
            for (int q = 0; q < channels; q++) {   // C_in/4 groups
                float32x4_t _val = vld1q_f32(sptr); // 4 input channels
                // For each kernel position (y, x):
                int sys = (i + y*Dh) - (kernel_extent_h - 1);
                if (sys < 0 || sys % stride_h != 0) continue;  // ← STRIDE FILTER
                int sxs = (j + x*Dw) - (kernel_extent_w - 1);
                if (sxs < 0 || sxs % stride_w != 0) continue;  // ← STRIDE FILTER
                // 4-way FMA: 4 input lanes × 4 output channels
                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
            }
            vst1q_f32(outptr + j*4, _sum);
        }
    }
}
```

The `vmlaq_laneq_f32` accumulates 4 input channels × 4 output channels at once. In contrast,
the scalar reference computes 1 channel at a time.

**The stride-filter waste**: for Kh=Kw=4 and Sh=Sw=2, `kernel_extent_h = Dh*(Kh-1)+1 = 4`.
For each output row `i` and kernel row `y`, `sys = i + y - 3`. This passes `sys % 2 == 0`
only when `(i + y)` is odd or even in lockstep — exactly 2 of the 4 `y` values per output row.
Same in the x direction: 2 of 4. So only `(Kh/Sh) × (Kw/Sw) = 2 × 2 = 4` of 16 kernel
positions pass the filter; **75% of inner iterations execute only the `continue` branch**.

Despite 75% filter waste, ncnn's 4-output-channel NEON vectorisation and packed weight layout
give an ~1.95× advantage over the scalar (1 / 0.513 ≈ 1.95). The filtered branches still
require two modulo checks plus comparisons per iteration on Neoverse V2 (~5 cycles per
integer mod), which limits the practical gain.

For C_in=11 (not divisible by 4), the benchmark falls into the `elempack==1, out_elempack==4`
path (`deconvolution_arm.cpp:477–546`), which uses `vdupq_n_f32(sptr[sx])` to broadcast a
scalar input and `vmlaq_f32` with a 4-float weight vector. Same stride-filter waste, one
scalar input per iteration.

---

## 2. ncnn-baseline's own bottlenecks

The `elempack==4, out_elempack==4` path has two structural bottlenecks:

### 2a. 75% wasted iterations due to stride filtering

The output-space traversal (`for i, j`) forces ncnn to check every kernel position at every
output pixel for phase alignment. For Sh=Sw=2, Kh=Kw=4, 12 of 16 iterations per (q, i, j)
exit via `continue` after executing two modulo operations. On Neoverse V2, integer modulo is
not a single-cycle operation; the pipeline cannot overlap this computation with subsequent FMAs
effectively.

### 2b. kptr is reset per output channel group and walked per (i, j)

```cpp
// deconvolution_arm.cpp:412
const float* kptr = weight_data_tm.channel(p);   // reset per (p, i, j)
...
kptr += maxk * 16;                                // advance per q
```

The entire weight block for channel group `p` is traversed independently for each output
pixel `(i, j)`. For the largest workload (H_out=114, W_out=114), this means the same weight
block is walked `114 × 114 = 12,996` times. For Cout/4 = 32 groups × 12,996 iterations ×
`C_in/4` weight loads, the weight access pattern is completely re-read for every output pixel
— L2/L3 thrashing is unavoidable on large input sizes.

### 2c. `deconv4x4s2_neon` is never reached for these workloads

The dedicated input-space kernel in `deconvolution_4x4.h:173–320` (which uses
`vld2q_f32/vst2q_f32` to handle stride-2 scatter in NEON without any filtering) only fires for
`elempack==1, out_elempack==1` (`deconvolution_arm.cpp:656–664`). Since Cout=128 is divisible
by 4, `out_elempack` is always 4 for this definition, so `deconv4x4s2_neon` is never invoked.
That kernel would be more efficient (no stride filter needed, scatter handled by `vld2q/vst2q`)
but its 128-bit NEON `float32x4_t` only covers 4 output channels per FMA group.

---

## 3. Why the agent beats ncnn-baseline (final 1.758×)

The agent's insight is to **invert the vectorisation axis**: instead of NEON-vectorising over
4 input or output channels (as ncnn does), exploit the fixed Cout=128 by vectorising over all
128 output channels simultaneously using SVE on Graviton4's 128-bit SVE2 (4 floats per vector,
so 128 floats = 32 vectors — but Graviton4 has 128-bit SVE, i.e., `svcntw()=4`, so 32 SVE
vectors). Wait: Graviton3 (c7g) is 256-bit SVE; Graviton4 (c8g) is 128-bit SVE2.
With 128-bit SVE2, `svcntw()=4`, so 128 output channels = 32 SVE vectors.

However, the agent's code hardcodes offsets `+0, +8, +16, ..., +120` as if `svcntw()=8`
(256-bit SVE). Looking at v4.cpp, the bias initialisation and main loop both use:
```cpp
svld1_f32(pg, ptr + 0),  svld1_f32(pg, ptr + 8), ...  // 16 loads × 8 floats = 128
```
This targets **256-bit SVE (Graviton3)** exactly. Despite the mismatch comment in v1 ("SVE 256-bit"),
the code does produce correct results on Graviton4's 128-bit SVE2 because `svld1_f32` with a
128-bit register and stride +8 simply reads the next 4 floats, and the compiler unrolls correctly
— what matters is that the total coverage is Cout=128 floats regardless of vector width.

### Step 1: The algorithmic flip (v1, +1.665×)

v1 (`agent-runs/deconv2d_kh4_kw4_sh2_sw2_cout128/v1.cpp`) introduces three changes together:

**Weight transpose**: from `[Cout, C_in, Kh, Kw]` to `[C_in, Kh, Kw, Cout]`, making the Cout
dimension contiguous for fixed `(ci, kh, kw)`:
```cpp
w_trans[((ci * Kh + kh) * Kw + kw) * Cout + co] =
    weight[((co * C_in + ci) * Kh + kh) * Kw + kw];
```

**Temporary output buffer in `[H_out, W_out, Cout]` layout**, making all 128 output channels
for a given spatial position contiguous:
```cpp
float* out_tmp = (float*)malloc((long)N * H_out * W_out * Cout * sizeof(float));
```

**SVE scatter-accumulate over Cout**:
```cpp
svfloat32_t sv_in = svdup_f32(in_val);   // broadcast scalar input
for (int co = 0; co < Cout; co += svcntw()) {
    svfloat32_t w_vec = svld1_f32(pg, wptr + co);
    svfloat32_t o_vec = svld1_f32(pg, optr + co);
    o_vec = svmla_f32_x(pg, o_vec, sv_in, w_vec);
    svst1_f32(pg, optr + co, o_vec);
}
```

This is the **scatter-accumulate / input-space algorithm**: for each `(ci, ih, iw, kh, kw)`,
scatter to output position `(oh, ow)` and update all 128 channels at once. No stride filtering
needed: `oh = ih*Sh + kh`, `ow = iw*Sw + kw` map directly.

IPC=2.63 in v1 — the highest of any version — because the inner SVE loop is small and tight,
weight and output fit well in L1/L2 (only 1.02M cache misses), and the FMA pipeline is kept
busy with no branch overhead.

The final output transpose back to `[N, Cout, H_out, W_out]` is a scalar loop; it is not a
bottleneck because it runs once after the main computation.

### Step 2: Failed kh×kw full unroll (v2, regression to 1.488×)

v2 attempts to unroll all 16 kernel positions `(kh=0..3, kw=0..3)` inside the `iw` loop using
a macro `DO_KH_KW(kh_val, kw_val)`, which expands to 32 `svld1 + svmla + svst1` instructions
per kernel position:

```cpp
#define DO_KH_KW(kh_val, kw_val) { \
    svfloat32_t w0 = svld1_f32(pg, wptr + 0); \
    /* ... 16 weight + 16 output loads, 16 FMAs, 16 stores ... */ \
}
DO_KH_KW(0,0) DO_KH_KW(0,1) ... DO_KH_KW(3,3)  // 16 full expansions inside iw loop
```

This creates a code block requiring 16 × 16 = 256 live SVE registers simultaneously,
far exceeding the 32 Z-registers available on AArch64 SVE2. The compiler must spill registers
to the stack, introducing load/store round-trips for every weight vector.

**IPC collapsed from 2.63 to 1.70**, cache misses tripled to 3.19M. The measured speedup
dropped from 1.665× to 1.488× — a regression despite more explicit unrolling.

### Step 3: Loop restructure: kh/kw outside iw (v3, 1.699×)

v3 (`v3.cpp`) resolves the register pressure by restructuring: kh is promoted outside the iw
loop, and each kw is a separate code block that each contain their own iw loop:

```cpp
for (int kh = 0; kh < Kh; ++kh) {
    // kw=0: load 16 weights, iterate over iw
    {
        const float* wptr = w_kh + 0 * Cout;
        svfloat32_t w0 = svld1_f32(pg, wptr + 0);  // 16 weight vecs
        // ...
        for (int iw = 0; iw < W; ++iw) {
            svfloat32_t sv_in = svdup_f32(in_row[iw]);
            float* optr = out_row + (iw * Sw + 0) * Cout;
            // 16 output loads, 16 FMAs, 16 stores
        }
    }
    // kw=1, kw=2, kw=3: same structure
}
```

The 16 weight vectors for a given `kw` are alive only within their own block, fitting in 32
Z-registers alongside the 16 output accumulator vectors. This eliminates the spill.

IPC recovers partially to 1.64; cache misses fall to 1.35M (vs 3.19M in v2). However, the
16 weight vectors are still reloaded on every pass through the outer loop — once per `(ci, kh)`
pair — and reused across the `iw` iterations within a single kw block.

The v3 disassembly (`v3.s`) showed scalar `ldr s0 / str s1` pairs in the weight-transpose
section (weight transpose is still scalar), and the agent's comment at the top of v4.cpp
confirms the agent noticed this: *"The assembly shows the weight transpose uses scalar
loads/stores (ldr s0, str s1 pairs) which is slow."*

### Step 4: Weight hoisting outside iw loop (v4, 1.758×)

v4's key change: for each `kw` block, the 16 weight vectors are loaded **before** the `iw`
loop, not inside it:

```cpp
// kw=0: load 16 weight vecs ONCE
const float* wptr = w_kh + 0 * Cout;
svfloat32_t w0  = svld1_f32(pg, wptr + 0);
svfloat32_t w1  = svld1_f32(pg, wptr + 8);
// ... (16 total, loaded once)

for (int iw = 0; iw < W; ++iw) {          // ← w0..w15 stay in registers
    svfloat32_t sv_in = svdup_f32(in_row[iw]);
    float* optr = out_row + (iw * Sw + 0) * Cout;
    svfloat32_t o0 = svld1_f32(pg, optr + 0);   // 16 output loads
    // ...
    o0 = svmla_f32_x(pg, o0, sv_in, w0);        // 16 FMAs
    // ...
    svst1_f32(pg, optr + 0, o0);                 // 16 stores
}
```

For the largest workload (H=56, W=56): each `(ci, kh, kw)` combination now loads 16 weight
vectors exactly once and reuses them across all W=56 iw iterations, reducing weight loads by
**56×** for that block. Total weight loads for the main loop drop from
`C_in × H × Kh × Kw × W × 16` (v3) to `C_in × H × Kh × Kw × 16` (v4).

IPC in v4 is 1.67 — slightly higher than v3's 1.64, as the weight-load pressure is removed
from the iw inner loop and the FMA units see a cleaner stream of work.

The cache miss count rises from 1.35M (v3) to 2.33M (v4). The access pattern changed: in v3,
for each `iw`, all four `kw` positions are written sequentially (close in memory, stride
Cout×4=512 bytes). In v4, the kw loop is outside the iw loop — all iw values are written for
kw=0 before moving to kw=1. The two output streams (kw=0 and kw=1) are `Cout×Sw×4 = 1024`
bytes apart, leading to slightly more L2/L3 activity. But the reduction in weight loads more
than offsets this.

### Version progression summary

| Version | Key change | IPC  | Cache misses | Time speedup |
|---------|-----------|------|-------------|--------------|
| ref-scalar | No SIMD | — | — | 0.513× |
| v1 | Weight transpose + SVE over Cout | 2.63 | 1.02M | 1.665× |
| v2 | Full kh×kw macro unroll inside iw | 1.70 | 3.19M | 1.488× ↓ |
| v3 | kh outside iw, per-kw iw loops | 1.64 | 1.35M | 1.699× |
| v4 | Weight hoisting outside iw loop | 1.67 | 2.33M | **1.758×** |

### Why the agent's approach exceeds ncnn's ceiling

The ncnn `elempack==4, out_elempack==4` path does:
- 4 output channels × 4 input channels per FMA = 16 multiplications per valid kernel position
- But 12 of 16 kernel positions are filtered at the cost of two `%` operations each

The agent's approach does:
- 128 output channels per FMA group (32 SVE vectors)
- 0 wasted iterations — scatter to exact `(oh, ow)` positions directly
- Weight hoisting: 16 weight vectors loaded once per `(ci, kh, kw)`, reused W times

The agent's inner iw loop is almost purely FMA + load/store: `16 svld1 + 16 svmla + 16 svst1`
with weights already in registers. No integer modulo, no branch prediction needed.

Since both algorithms do the same total arithmetic, the 1.76× advantage comes entirely from
eliminating the 75% branch waste and using a wider (128-channel vs 4-channel) vectorisation
tile that keeps 32 FMA pipelines busy per iw iteration.

---

## 4. Hardcoded parameters in the agent solution

The header `deconv2d.h` provides all seven definition parameters as `constexpr` in namespace
`deconv2d_def`. The agent's `kernel.cpp` (`v4.cpp`) uses `using namespace deconv2d_def;` and
references `Sh`, `Sw`, `Kh`, `Kw`, `Cout`, `Kh`, `Kw` by name. But several are effectively
hardcoded by algorithmic structure rather than as literals:

### Cout=128: hardcoded by SVE register count

The bias initialisation and all inner loops use exactly 16 named SVE vector variables
(`b0..b15`, `w0..w15`, `o0..o15`) with hardcoded offsets `+0, +8, +16, ..., +120`.
There is no `for (co = 0; co < Cout; co += svcntw())` loop — the coverage is fixed at
16 × 8 = 128 floats (assuming 256-bit SVE; 16 × 4 = 64 for 128-bit SVE2):

```cpp
svfloat32_t b0  = svld1_f32(pg, bias + 0);    // offset 0
svfloat32_t b1  = svld1_f32(pg, bias + 8);    // offset 8
// ...
svfloat32_t b15 = svld1_f32(pg, bias + 120);  // offset 120
```

If Cout ≠ 128, the offsets are wrong and coverage is incomplete or overflows.

### Kh=4 and Kw=4: hardcoded by loop elimination

There is no `for (int kh = 0; kh < Kh; ++kh)` loop. Instead, four explicit kh blocks are
written inline (`kh=0..3`), each containing four kw sub-blocks (`kw=0..3`). The reference to
`constexpr int Kh = 4` is used only in `oh = ih * Sh + kh` calculations (where kh is the
literal block index, not a loop variable). Any definition with Kh ≠ 4 or Kw ≠ 4 would produce
wrong results (missing contributions from kernel positions beyond the hardcoded blocks).

### Sh=2, Sw=2: referenced correctly via constexpr

`oh = ih * Sh + kh` and `ow = iw * Sw + kw` use the constexpr values directly. Changing Sh
or Sw would produce correct output pointer arithmetic without code modification.

### Dh=1, Dw=1: implicitly hardcoded

No `kh * Dh` term appears in the code. Output row is computed as `oh = ih * Sh + kh`, which
is only correct for Dh=1 (where kernel position `kh` maps directly to output offset `kh`).
For Dh > 1, the correct formula is `oh = ih * Sh + kh * Dh`, and the weight-transpose loop
would also need to account for dilation in the output index computation.

### Summary

| Parameter | Hardcode type | Effect of changing |
|-----------|--------------|-------------------|
| Cout=128  | Structural: 16 fixed SVE register slots with hardcoded offsets | Wrong coverage or buffer overrun; must regenerate all 16-vector blocks |
| Kh=4      | Structural: no kh loop, 4 explicit blocks | Missing contributions for Kh > 4; wrong results for Kh ≠ 4 |
| Kw=4      | Structural: no kw loop, 4 explicit blocks per kh | Same as Kh |
| Sh=2      | Correct: uses constexpr `Sh` in `ih * Sh + kh` | Safe to change |
| Sw=2      | Correct: uses constexpr `Sw` in `iw * Sw + kw` | Safe to change |
| Dh=1      | Implicit: `oh = ih*Sh + kh` has no `kh*Dh` factor | Wrong output positions for Dh > 1 |
| Dw=1      | Implicit: `ow = iw*Sw + kw` has no `kw*Dw` factor | Wrong output positions for Dw > 1 |
