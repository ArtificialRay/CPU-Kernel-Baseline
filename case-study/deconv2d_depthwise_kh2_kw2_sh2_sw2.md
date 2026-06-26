# Case Study: deconv2d_depthwise_kh2_kw2_sh2_sw2

Depthwise transposed 2D conv, kernel 2×2, stride 2×2, dilation 1×1, no padding.
Axes: `N=1`, `C` varies (11–128), `H`/`W` vary (28–56). One agent version submitted.

## Speedup Trajectory

| Turn | Tool | Status | time_speedup | cycle_speedup | IPC | cache_misses |
|------|------|--------|-------------|---------------|-----|-------------|
| 1 | compile v1.cpp | OK | — | — | — | — |
| 2 | evaluate | PASSED | — | — | — | — |
| 3 | evaluate | PASSED | 37.950× | 33.603× | 2.993 | 3932 |
| 7 | submit | PASSED | **37.828×** | **33.707×** | — | — |

**Reference-scalar vs ncnn-baseline: scalar is 20.385× faster (time), 19.930× (cycle).**

---

## Section 1 — Reference-scalar vs ncnn-baseline: scalar is 20× faster

For this definition, scalar is already faster than ncnn. The cause is an algorithmic mismatch in ncnn's implementation for deconv with stride > 1.

ncnn's `elempack==1` path (`deconvolutiondepthwise_arm.cpp:325–385`) uses **output-space traversal**: it iterates over every output pixel and, for each, tests all kh×kw=4 kernel positions for stride-phase alignment:

```cpp
// deconvolutiondepthwise_arm.cpp:340–381
for (int i = 0; i < outh; i++) {
    for (int j = 0; j < outw; j++) {
        for (int y = 0; y < kernel_h; y++) {
            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
            if (sys < 0 || sys % stride_h != 0) continue;   // ← phase filter
            int sy = sys / stride_h;                          // ← integer division
            if (sy >= h) continue;
            for (int x = 0; x < kernel_w; x++) {
                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                if (sxs < 0 || sxs % stride_w != 0) continue;
                int sx = sxs / stride_w;
                if (sx >= w) continue;
                sum += sptr[sx] * kptr[y * kernel_w + x];
            }
        }
        outptr[j] = sum;
    }
}
```

For kh=kw=sh=sw=2, a kernel position (y, x) contributes only when *both* `sys % 2 == 0` and `sxs % 2 == 0`. For any output pixel exactly one of the four (y, x) pairs passes both checks; the other three hit `continue`. **75% of inner-loop iterations are discarded**, yet each still pays:

1. Two integer modulo operations (`sys % stride_h`, `sxs % stride_w`) — ~4–8 cycles each on ARM
2. One integer division (`sys / stride_h`) — ~10–20 cycles
3. Branch misprediction (~3–5 cycles) when the alternating taken/not-taken pattern fools the predictor

Total loop iterations: outh×outw×4 = **16HW**, of which **4HW** do useful FMA and **12HW** pay overhead only.

Reference-scalar uses **input-space scatter**: for each input pixel (ih, iw), it directly computes the four output addresses `oh = ih*Sh + kh`, `ow = iw*Sw + kw` and accumulates — no conditionals, no modulo, no division. Total: **4HW FMA** in **4HW iterations**. The algorithmic ratio is 16/4 = **4×**, and the per-wasted-iteration arithmetic overhead (mod/div) multiplies it further to the observed **20.385×** gap.

Additionally, the `elempack==1` path contains no SIMD; all computation is scalar `float sum += val * w`.

---

## Section 2 — ncnn-baseline's own bottlenecks

With the `elempack==1` path purely scalar and 75% of its iterations wasted, ncnn's structural limiters are:

**Division and modulo in the hot loop.** ARM lacks fast hardware integer divide; `sdiv` takes 12–20 cycles. `sys % stride_h` and `sxs % stride_w` appear in the innermost body, so every iteration — including the 75% that produce no output — pays their latency before the branch resolves.

**Branch-prediction pressure.** The filter `sys % stride_h != 0` fires on ~50% of `y` iterations (for sh=2, every other output row misses). The `sxs % stride_w != 0` filter fires independently on ~50% of `x` iterations. These form an alternating pattern that the branch predictor cannot fully capture, adding repeated flush penalties.

There is an `elempack==4` NEON path (`deconvolutiondepthwise_arm.cpp:264–322`) that processes 4 channels at once via `vmlaq_f32`. However, it carries the same output-space loop with identical mod/div/branch structure. The 75% wasted iterations still occur; only the 25% useful iterations gain 4-channel NEON instead of scalar FMA. Effective gain: `1 / (0.75 + 0.25/4) ≈ 1.23×` — insufficient to close the algorithmic gap. There is no `deconvolutiondepthwise_packed.h`; ncnn has no tiled or weight-packed kernel for this op type.

---

## Section 3 — Why v1 achieves 37.828× (1.85× over reference-scalar)

v1 is the only submitted version. It achieves **37.828× over ncnn** (time), which is **~1.85× over reference-scalar** (37.828 / 20.385 ≈ 1.85).

### Algorithmic gain — matching reference-scalar's 20×

v1 adopts input-space scatter. The kernel loop is fully eliminated: the four weights are unpacked to scalar variables once per channel:

```cpp
// v1.cpp:56–60
float w00 = w_c[0], w01 = w_c[1], w10 = w_c[2], w11 = w_c[3];
svfloat32_t vw00 = svdup_f32(w00);
svfloat32_t vw01 = svdup_f32(w01);
svfloat32_t vw10 = svdup_f32(w10);
svfloat32_t vw11 = svdup_f32(w11);
```

Output row pointers are precomputed per `ih`:

```cpp
// v1.cpp:65–66
float* out_row0 = out_c + (long)(ih * 2)     * W_out;
float* out_row1 = out_c + (long)(ih * 2 + 1) * W_out;
```

No conditionals, no modulo, no division at runtime.

### SVE gain — ~1.85× over scalar

v1 vectorises the `iw` loop with SVE, processing vl=4 inputs per iteration (128-bit SVE2 on Graviton4):

```cpp
// v1.cpp:84–107
svfloat32_t vin   = svld1_f32(pg, in_row + iw);
svfloat32_t prod00 = svmul_f32_x(pg, vin, vw00);   // 4× val*w00
svfloat32_t prod01 = svmul_f32_x(pg, vin, vw01);   // 4× val*w01
svfloat32_t zip0_lo = svzip1_f32(prod00, prod01);   // interleave → stride-2 layout
svfloat32_t zip0_hi = svzip2_f32(prod00, prod01);
// load-add-store 2×vl=8 row0 outputs; same for row1
```

The compiler recognises the `svzip + svst1` pattern and emits `ld2w`/`st2w` structured load/store instructions that deinterleave and interleave directly in memory (confirmed in `v1.s:0xaec–0xb2c`):

```asm
; v1.s:0xaec–0xb2c — hot loop body (~11 instructions for vl=4 pixels)
ld1w  z16,        [x2, x18, lsl #2]     ; load 4 inputs
ld2w  {z17, z18}, [x0, x9, lsl #2]      ; load 2×vl row0 outputs, auto-deinterleaved
fmla  z19.s, z16.s, z4.s                ; row0_even += input × w00
fmla  z20.s, z16.s, z5.s                ; row0_odd  += input × w01
st2w  {z19, z20}, [x0, x9, lsl #2]      ; store 2×vl row0 outputs, auto-interleaved
ld2w  {z17, z18}, [x1, x9, lsl #2]      ; load 2×vl row1 outputs
fmla  z19.s, z16.s, z6.s
fmla  z20.s, z16.s, z7.s
st2w  {z19, z20}, [x1, x9, lsl #2]      ; store 2×vl row1 outputs
```

The `ld2w/st2w` absorb the zip overhead at no extra instruction cost; the C-level `svzip1/svzip2` compile away entirely.

**Why ~1.85× not ~4×:** the bottleneck shifts to the load port. Each hot-loop iteration issues 5 load μops (1 ld1w + 2×ld2w, where each ld2w fills two registers = 2 μops) on a 2-port load unit, requiring ≥2.5 cycles. Scalar for 4 equivalent pixels issues ~12 load μops at ≥6 cycles. Throughput ratio: 6/2.5 = **2.4×**, consistent with the observed 1.85× (the tail scalar path and bias-fill loop pull the geometric mean below the hot-loop ceiling).

The IPC of 2.993 (~3.0 out of 4-wide issue) confirms the core is ~75% utilised; the remaining 25% gap is load-port stall.

| Version | Algorithm | Vectorisation | IPC  | Time speedup |
|---------|-----------|---------------|------|-------------|
| v1      | input-space scatter | SVE iw-dim (ld2w/st2w) | 2.993 | **37.828×** |

No further gap remains versus ncnn: ncnn's approach is algorithmically inferior to scalar, and v1 additionally vectorises what scalar does.

---

## Section 4 — Hardcoded parameters in v1

**Kh=2, Kw=2 — Algorithmic**
The kernel dimensions are never looped over. `v1.cpp:56–60` unpacks exactly four weights (`w00, w01, w10, w11`) with fixed indices. `v1.cpp:65–66` hardcodes two output rows (`out_row0`, `out_row1`). Changing to Kh=3 would silently access `w_c[4]` (out of bounds) and miss the third kernel row entirely.

**Sh=2, Sw=2 — Algorithmic**
Output row addresses use literal `* 2` (`v1.cpp:65–66`). The `svzip1/svzip2` interleaving pattern — and the `ld2w/st2w` the compiler emits — assume Sw=2: each pair of output columns for adjacent `iw` values lies at consecutive addresses. Any other stride breaks the interleave layout and produces wrong output.

**Dh=1, Dw=1 — Not hardcoded (irrelevant)**
Dilation=1 means kernel taps land at consecutive input positions, which the code handles correctly without explicitly referencing dilation. For Dh>1, the output addresses remain unchanged but the input sampling would require gaps between taps — not modeled. Because all definitions in this family have Dh=Dw=1, this is not a live risk.

| Parameter | Hardcode type | Effect of changing |
|-----------|--------------|-------------------|
| Kh=2, Kw=2 | Algorithmic | Wrong results; third kernel row never accessed |
| Sh=2, Sw=2 | Algorithmic | Wrong output addresses; zip/ld2w interleave pattern invalid for other strides |
| Dh=1, Dw=1 | Not hardcoded | Irrelevant at Dh=Dw=1; silently wrong for Dh>1 |
