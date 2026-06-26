# Case Study: `conv2d_kh1_kw1_sh1_sw1_dh1_dw1_cout256`

**Definition**: Kh=1, Kw=1, Sh=1, Sw=1, Dh=1, Dw=1, Cout=256, pad=(0,0)
(1×1 pointwise convolution; C_in and spatial dims vary per workload)

**Speedup trajectory** (vs ncnn-baseline, geomean across 6 workloads):

| Version | Time speedup | Cycle speedup | IPC   | Cache misses (mean) |
|---------|-------------|---------------|-------|---------------------|
| reference-scalar | 0.057× | 0.061× | — | — |
| agent v1 (OC=4, 1 HW vec) | 0.357× | 0.368× | 1.837 | 255,645 |
| agent v2 (OC=8, 2 HW vecs) | 0.430× | 0.443× | 1.594 | 400,620 |
| **agent v3 (OC=8, 4 HW vecs)** | **0.550×** | **0.548×** | **2.153** | 614,330 |

> All speedup values are **ncnn\_time / candidate\_time**. Values below 1.0 mean the
> candidate is slower than ncnn. Agent v3 (0.550×) is still 1.82× slower than ncnn;
> it does not beat the baseline on this problem.

---

## 1. Why reference-scalar is 17.5× slower than ncnn-baseline

### ncnn's 1×1 routing — unconditional GEMM path

`convolution_arm.cpp:259` hard-routes 1×1 convolutions through a dedicated GEMM path
regardless of any other heuristics:

```cpp
// convolution_arm.cpp:259
if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
{
    convolution_im2col_gemm_transform_kernel(weight_data, weight_sgemm_data,
                                             num_input, num_output,
                                             kernel_w, kernel_h, opt);
    ...
    return 0;
}
```

No 1×1 convolution ever falls through to the generic loop path.

### Weight pre-packing at model load time

`convolution_im2col_gemm.h:6749` transforms weights from `[Cout, Cin, 1, 1]` into a
blocked `[TILE_K × TILE_M, K/TILE_K, M/TILE_M]` layout **once** during `create_pipeline`:

```cpp
// convolution_im2col_gemm.h:6772-6774 — for 1×1, reshape is a no-op
if (maxk == 1)
{
    A_data = kernel.reshape(maxk * inch, outch);  // [Cin, Cout] view, no copy
}

// convolution_im2col_gemm.h:6801-6816 — pack into blocked tile layout
AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);
for (int ppj = 0; ppj < nn_M; ppj++) {
    for (int k = 0; k < K; k += TILE_K) {
        convolution_im2col_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
    }
}
```

`convolution_im2col_pack_A_tile` (line 4) interleaves 8 output channels × 8 input channels
into a single 64-float contiguous block using `transpose8x8_ps`:

```cpp
// convolution_im2col_gemm.h:26-61 — pack 8 OC rows × 8 IC cols → 64 sequential floats
float32x4_t _r0l = vld1q_f32(p0);  float32x4_t _r0h = vld1q_f32(p0 + 4);
float32x4_t _r1l = vld1q_f32(p1);  float32x4_t _r1h = vld1q_f32(p1 + 4);
// ... p2..p7 (8 OC rows)
transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h,
                _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h);
vst1q_f32(pp,      _r0l);  vst1q_f32(pp + 4,  _r0h);  // pp[0..7]  = oc0..7 for ic=k
vst1q_f32(pp + 8,  _r1l);  vst1q_f32(pp + 12, _r1h);  // pp[8..15] = oc0..7 for ic=k+1
// ...
pp += 64;  // 8 OC × 8 IC = 64 floats, all contiguous
```

After packing, the GEMM kernel's weight pointer (`pA`) steps through memory sequentially:
one `ld1 {v4.4s}` + `ld1 {v5.4s}` loads 8 weight values for all 8 output channels
simultaneously, with guaranteed L1 hits.

### Hand-tuned aarch64 GEMM kernel — 8×12 tiling with prefetch

The inner kernel (`convolution_im2col_gemm.h:281`) uses inline assembly to process
8 OC × 12 spatial positions per inner loop iteration, interleaving loads and FMAs:

```asm
// convolution_im2col_gemm.h:281-323 — inner loop body (one K-step, 8×12 tile)
"4:                                  \n"
"ldr    d1, [%2], #8                \n"   // ← load B[spatial+4..7] (interleaved)
"fmla   v8.4s,  v4.4s, v0.s[0]     \n"   // OC[0..3] × spatial[0], 4 FMAs
"ldr    x21, [%2], #8               \n"
"fmla   v9.4s,  v4.4s, v0.s[1]     \n"   // OC[0..3] × spatial[1]
"ins    v5.d[1], x25                \n"
"fmla   v10.4s, v4.4s, v0.s[2]     \n"   // OC[0..3] × spatial[2]
"ldr    d2, [%2], #8                \n"
"fmla   v11.4s, v4.4s, v0.s[3]     \n"   // OC[0..3] × spatial[3]
...
"fmla   v20.4s, v5.4s, v0.s[0]     \n"   // OC[4..7] × spatial[0]
...
"prfm   pldl1keep, [%2, #512]       \n"   // ← prefetch B 512 bytes ahead
"prfm   pldl1keep, [%1, #512]       \n"   // ← prefetch A 512 bytes ahead
```

- `v4.4s` / `v5.4s`: 8 sequential weight values from AT, loaded with two `ld1`s
- `v0.s[0]` .. `v0.s[3]`: 4 input scalars broadcast across 4 OC lanes each
- 24 `fmla` instructions per 4 K-steps → **96 FMAs per inner iteration**, dual-issue
- `prfm` hides DRAM latency; on Neoverse V1 this prevents L2 stalls

**Reference scalar** has none of this: one multiply per element, row-major weight access
(`weight[oc * C_in + ic]`), no blocking, no prefetch → `0.057×` of ncnn speed.

---

## 2. ncnn-baseline's own bottlenecks

### Bottleneck 1 — Runtime input (B matrix) packing every inference

Weights are packed once offline. But the input tensor must be repacked into BT format
**on every forward pass** before the GEMM can run:

```cpp
// convolution_im2col_gemm.h:6838 — fresh allocation and fill on every call
Mat BT(TILE_K * TILE_N,
        (K + TILE_K - 1) / TILE_K,
        (N + TILE_N - 1) / TILE_N,
        4u, opt.workspace_allocator);
// ... then BT is filled from bottom_blob via convolution_im2col_input_tile
```

For the largest workload (Cin=64, H=W=224):
- B matrix = C_in × HW = 64 × 50,176 ≈ 12.8 M floats = **51 MB**
- Every inference: 51 MB read (from input) + 51 MB write (into BT) before any GEMM

This is pure overhead that a candidate kernel avoiding im2col does not pay.

### Bottleneck 2 — 128-bit NEON ceiling on a 256-bit SVE machine

The GEMM kernel uses NEON `fmla .4s`, which operates on 128-bit vectors (4 floats):

```asm
"fmla   v8.4s, v4.4s, v0.s[0]     \n"   // 4 FMAs per instruction (128-bit NEON)
```

Graviton3 supports 256-bit SVE (8 floats per vector). The ncnn codebase has no SVE
code path for this kernel. Every NEON instruction leaves 4 FMA slots idle that SVE
would have used. The theoretical FMA throughput gap is 2×.

### Bottleneck 3 — L3 traffic for large-HW workloads

For the 224×224 workload, the N dimension (HW = 50,176) far exceeds what fits in L2
with K=64. TILE_N is computed from L2 cache size (`convolution_im2col_gemm.h:5937`);
on Graviton3 (1 MB L2/core):

```
TILE_K × TILE_N × 4 bytes ≤ L2 budget
256 × TILE_N × 4 ≤ ~900 KB  →  TILE_N ≈ 900
```

The BT matrix is walked in ~56 outer panels (50,176 / 900). Each panel brings in
~3.5 MB of BT from L3. For a single inference the GEMM generates ~200 MB of L3 read
traffic just from the B matrix, independent of compute.

---

## 3. How agent improved from 0.057× to 0.550×

*Agent v3 did not beat ncnn. The analysis below explains why the 9.6× gain over
reference-scalar occurred and why the remaining 1.82× gap to ncnn persists.*

### Root cause: register-blocking eliminates partial-sum spill

v1 and v2 share the same fatal structure: partial sums are read from and written to
memory on **every IC iteration**.

**v1 (0.357×): OC=4, output spills on every IC step**

```cpp
// v1.cpp:66-96
for (int ic = 0; ic < C_in; ++ic) {                // outer loop over input channels
    svfloat32_t sw0 = svdup_f32(w0[ic]);            // broadcast weight scalar

    for (int hw2 = 0; hw2 < HW; hw2 += vl) {
        svbool_t pg = svwhilelt_b32(hw2, HW);
        svfloat32_t vin = svld1_f32(pg, in_ic + hw2);

        svfloat32_t vout0 = svld1_f32(pg, out0 + hw2);  // ← LOAD partial sum (cache)
        svfloat32_t vout1 = svld1_f32(pg, out1 + hw2);
        svfloat32_t vout2 = svld1_f32(pg, out2 + hw2);
        svfloat32_t vout3 = svld1_f32(pg, out3 + hw2);

        vout0 = svmla_f32_m(pg, vout0, vin, sw0);        // FMA
        vout1 = svmla_f32_m(pg, vout1, vin, sw1);
        vout2 = svmla_f32_m(pg, vout2, vin, sw2);
        vout3 = svmla_f32_m(pg, vout3, vin, sw3);

        svst1_f32(pg, out0 + hw2, vout0);  // ← STORE partial sum back (cache)
        svst1_f32(pg, out1 + hw2, vout1);
        svst1_f32(pg, out2 + hw2, vout2);
        svst1_f32(pg, out3 + hw2, vout3);
    }
}
// For Cin=64: each output element is loaded and stored 64 times
```

For workload (Cin=64, H=W=224): 4 OC × 50,176 HW × 64 IC = **12.8 M spurious load/store
pairs** that carry no information — they exist only to preserve partial sums between IC
steps. The SVE FMAs are correct and fast; the surrounding memory traffic overwhelms them.
IPC is 1.837, memory-bound.

**v2 (0.430×): OC=8, 2 HW vectors — same spill, larger working set**

```cpp
// v2.cpp:35-88 — process_oc8_hw: 8 OC, 2×vl HW unroll
for (; hw < hw2; hw += 2 * vl) {
    svfloat32_t vin0 = svld1_f32(pg, in_ic + hw);
    svfloat32_t vin1 = svld1_f32(pg, in_ic + hw + vl);

    svfloat32_t vo00 = svld1_f32(pg, out0 + hw);    // ← still loading partial sums
    svfloat32_t vo01 = svld1_f32(pg, out0 + hw + vl);
    svfloat32_t vo10 = svld1_f32(pg, out1 + hw);
    // ... 16 svld1 calls per hw chunk

    vo00 = svmla_f32_x(pg, vo00, vin0, sw0);
    // ... 16 svmla calls

    svst1_f32(pg, out0 + hw, vo00);                 // ← still storing partial sums
    // ... 16 svst1 calls
}
```

Doubling OC from 4 to 8 doubled the output working set. IPC dropped from 1.837 to
1.594 and cache misses rose from 255K to 400K: the larger tile evicts more from L1D
than the additional FMAs can compensate.

**v3 (0.550×): 32 accumulators live in SVE registers across the entire IC loop**

```cpp
// v3.cpp:44-139
// Tiling: HW_TILE = 4×vl = 32 spatial positions; OC_TILE = 8 channels
for (int hw_start = 0; hw_start < HW; hw_start += HW_TILE) {
    for (int oc = 0; oc < Cout; oc += 8) {

        // 8 OC × 4 HW vectors = 32 SVE accumulators, all in registers
        svfloat32_t acc00, acc01, acc02, acc03;   // OC[oc+0], HW[0..3]
        svfloat32_t acc10, acc11, acc12, acc13;   // OC[oc+1], HW[0..3]
        // ...
        svfloat32_t acc70, acc71, acc72, acc73;   // OC[oc+7], HW[0..3]

        acc00 = acc01 = acc02 = acc03 = svdup_f32(0.0f);  // v3.cpp:69
        acc10 = acc11 = acc12 = acc13 = svdup_f32(0.0f);
        // ... zero-init all 32, no memory writes

        // Full IC reduction — no memory store of partial sums
        for (int ic = 0; ic < C_in; ++ic) {               // v3.cpp:79
            const float* in_ic = in_n + (long)ic * HW + hw_start;

            // 4 input loads — contiguous, ~4 cycles each
            svfloat32_t vin0 = svld1_f32(pg0, in_ic + 0);
            svfloat32_t vin1 = svld1_f32(pg1, in_ic + vl);
            svfloat32_t vin2 = svld1_f32(pg2, in_ic + 2*vl);
            svfloat32_t vin3 = svld1_f32(pg3, in_ic + 3*vl);

            // 8 weight scalars (stride C_in — non-contiguous, see §4)
            const float* wbase = weight + (long)oc * C_in + ic;
            float w0 = wbase[0 * C_in];  // weight[oc+0][ic]
            float w1 = wbase[1 * C_in];  // weight[oc+1][ic]  (+256 bytes from w0)
            float w2 = wbase[2 * C_in];
            float w3 = wbase[3 * C_in];
            float w4 = wbase[4 * C_in];
            float w5 = wbase[5 * C_in];
            float w6 = wbase[6 * C_in];
            float w7 = wbase[7 * C_in];  // (+1792 bytes from w0)

            // 32 FMAs — all operands in registers, zero memory traffic
            acc00 = svmla_n_f32_x(pg, acc00, vin0, w0);  // v3.cpp:100
            acc01 = svmla_n_f32_x(pg, acc01, vin1, w0);
            acc02 = svmla_n_f32_x(pg, acc02, vin2, w0);
            acc03 = svmla_n_f32_x(pg, acc03, vin3, w0);
            // ... (32 total, through acc73)
            acc73 = svmla_n_f32_x(pg, acc73, vin3, w7);  // v3.cpp:138
        }

        // Write output exactly once per (hw_tile × oc_tile) block — v3.cpp:151-189
        svst1_f32(pg0, out0 + 0,    acc00);
        svst1_f32(pg1, out0 + vl,   acc01);
        svst1_f32(pg2, out0 + 2*vl, acc02);
        svst1_f32(pg3, out0 + 3*vl, acc03);
        // ... (32 stores total, output written only once)
    }
}
```

Output partial sums are never written to memory during the IC loop. The 12.8 M spurious
load/store pairs from v1 collapse to 32 SVE stores per `(hw_tile, oc_tile)` block.
IPC jumps to 2.153 — the FMA units are now the bottleneck, not the load/store ports.

### IPC progression

| Version | IPC  | Cache misses | Bottleneck |
|---------|------|-------------|------------|
| v1 | 1.837 | 255K | Output partial sums spill; 4 loads + 4 stores per FMA iteration |
| v2 | 1.594 | 400K | Doubled OC tile → doubled output working set → more L1D evictions |
| v3 | 2.153 | 614K | FMA-bound; higher cache misses from larger input tile, but zero spill |

v2's IPC *regression* (1.837 → 1.594) is the clearest signal: expanding OC without
eliminating the spill makes things worse. v3's cache miss count is highest of all three,
yet IPC is highest too — proof that the misses in v3 are pure input reads rather than
redundant output re-reads.

### Why the remaining 1.82× gap to ncnn persists

**Weight layout: 8 scattered loads vs. 2 sequential vector loads**

v3 loads 8 weight scalars per IC step, each at stride `C_in × 4 = 256 bytes`:

```
weight[oc+0][ic] → address A
weight[oc+1][ic] → address A + 256   (different cache line)
weight[oc+2][ic] → address A + 512
...
weight[oc+7][ic] → address A + 1792  (8 different cache lines, 4 bytes used per line)
```

ncnn packs all 8 weight values for a given IC position into consecutive memory:

```cpp
// convolution_im2col_gemm.h:73-81 — scalar path of pack_A_tile
pp[0] = p0[0]; pp[1] = p1[0]; pp[2] = p2[0]; pp[3] = p3[0];
pp[4] = p4[0]; pp[5] = p5[0]; pp[6] = p6[0]; pp[7] = p7[0];
pp += 8;  // all 8 OC weights for this IC position now at consecutive addresses
```

During the GEMM kernel, `ld1 {v4.4s}` and `ld1 {v5.4s}` fetch all 8 weights in 2
vector loads from sequential memory. v3 needs 8 scalar loads spanning 1792 bytes.

**SVE 2× width advantage partially cancelled by weight access cost**

v3 uses 256-bit SVE (vl=8 floats) vs. ncnn's 128-bit NEON (4 floats). This should
give 2× FMA throughput. But each of v3's 8 weight scalar loads can miss L1 independently,
stalling the FMA pipeline while ncnn's 2 sequential vector loads hit L1 reliably.
The effective gap between SVE FMA speed and weight load latency prevents v3 from
extracting the theoretical 2× advantage.

---

## 4. Hard-coded parameters in agent v3

### Kh=1, Kw=1 — algorithmic dependency, completely wrong for larger kernels

v3 flattens the 4D conv output into 2D by treating H×W as a single `HW` dimension:

```cpp
// v3.cpp:25
const int HW = H_out * W_out;
// Implicit: kernel covers exactly one input position per output.
// No loop over (kh, kw) exists anywhere in the file.
```

For a 3×3 convolution each output `(oh, ow)` must accumulate over 9 kernel taps:

```cpp
// Required for general (Kh, Kw) — entirely absent in v3
for (int kh = 0; kh < Kh; ++kh)
    for (int kw = 0; kw < Kw; ++kw)
        output[n][oc][oh][ow] += weight[oc][ic][kh][kw]
                                * input[n][ic][oh*Sh + kh*Dh][ow*Sw + kw*Dw];
```

With Kh=3, Kw=3, the agent would compute only `(kh=0, kw=0)` tap and produce
completely wrong results — no compile error, no assertion, just silently wrong numbers.

### Cout % 8 == 0 — missing tail handler

The outer loop steps by 8 and never checks whether a partial group remains:

```cpp
// v3.cpp:43 — no tail
for (int oc = 0; oc < Cout; oc += 8) {
    // Unconditionally accesses weight[oc+0..7][ic] and output[oc+0..7][hw]
    float* out7 = out_n + (long)(oc + 7) * HW + hw_start;  // v3.cpp:149
    // ...
    svst1_f32(pg3, out7 + 3*vl, acc73);  // write to out[oc+7]
}
```

For `Cout=100`, the iteration at `oc=96` writes to `output[103..107]` — four elements
past the end of the allocated tensor. For `Cout=512` this is fine (512 % 8 == 0).
A tail loop for the last `Cout % 8` channels is needed to generalize.

### Sh=1, Sw=1 — silent correctness bug (wrong input stride for Sh > 1)

v3 computes `HW = H_out * W_out` at line 25 and reuses it as the input channel stride:

```cpp
// v3.cpp:25
const int HW = H_out * W_out;

// v3.cpp:80 — BUG: input channel stride should be H * W, not H_out * W_out
const float* in_ic = in_n + (long)ic * HW + hw_start;
```

The input tensor is laid out as `[N, C_in, H, W]`, so channel `ic` starts at
`in_n + ic * H * W`. For `Sh=Sw=1` with `Kh=Kw=1` and `pad=0`,
`H_out == H` and `W_out == W`, so `HW == H * W` and the pointer is correct.

For `Sh=2` (e.g. `H=56, W=56` → `H_out=28, W_out=28`):

```
correct input stride:  ic * H * W      = ic * 56 * 56 = ic * 3136
v3 input stride:       ic * H_out*W_out = ic * 28 * 28 = ic * 784  ← 4× too small
```

Every channel beyond `ic=0` reads from the wrong address. Output values are garbage;
no assertion or compile error fires. The harness signature passes both `H, W` and
`H_out, W_out` to `inner_conv2d`, but `H` and `W` are accepted and then ignored
throughout v3's body — a single character change (`HW` → `H * W`) on line 80 fixes it.

### Dh=1, Dw=1 — irrelevant for 1×1 kernels

Dilation controls the spacing between kernel taps. With a single-tap kernel (Kh=Kw=1)
there are no inter-tap spacings to dilate. Any Dh/Dw value produces identical results.

### pad=(0,0) — handled by harness

Padding is applied to the input tensor by the harness before calling `inner_conv2d`.
The kernel receives already-padded dimensions; it never needs to know `pad_top` or
`pad_left`.

---

## Summary

| Parameter | Status in v3 | Effect of changing |
|-----------|-------------|-------------------|
| Kh=1, Kw=1 | **Algorithmic constraint** — GEMM collapses all kernel taps | Wrong results; missing `(kh, kw)` loops |
| Cout % 8 == 0 | **Missing tail handler** — always processes groups of 8 | Out-of-bounds write for non-multiple Cout |
| Sh=1, Sw=1 | Bug — input stride uses `H_out×W_out` instead of `H×W` | Wrong input pointer; silent garbage for Sh > 1 |
| Dh, Dw | Not hardcoded (irrelevant for 1×1) | No effect |
| pad | Not hardcoded (harness-applied) | No effect |
