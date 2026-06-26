# Case Study: `conv1d_kw1_sw1_dw1_cout512_p0`

**Definition**: Kw=1, Sw=1, Dw=1, Cout=512, pad=0  
**Speedup trajectory** (vs ncnn-baseline, geomean across workloads):

| Version | Time speedup | Cycle speedup | IPC  |
|---------|-------------|---------------|------|
| reference-scalar | 0.174× | 0.173× | ~1.0 |
| agent v1 (OC\_TILE=4)  | 1.52× | 1.50× | 1.44 |
| agent v2 (OC\_TILE=8)  | 2.25× | 2.20× | 2.11 |
| agent v3 (OC\_TILE=16) | **2.26×** | **2.22×** | 1.83 |

---

## 1. Why reference-scalar is 5.7× slower than ncnn-baseline

### What ncnn does right: weight pre-packing + NEON oc8 kernel

Before any inference, ncnn's `create_pipeline` calls
`convolution1d_transform_kernel_packed` to repack the weight tensor from the natural
`(Cout, Cin, Kw)` layout into an interleaved format:

```
weight_data_tm layout for one oc8 block:
  [w_oc0_ic0, w_oc1_ic0, ..., w_oc7_ic0,   ← 8 channels, ic=0
   w_oc0_ic1, w_oc1_ic1, ..., w_oc7_ic1,   ← 8 channels, ic=1
   ...
   w_oc0_icN, w_oc1_icN, ..., w_oc7_icN]   ← 8 channels, ic=C_in-1
```

This makes `kptr` walk sequentially through memory during inference.

During inference the oc8 NEON kernel (from `convolution1d_packed.h`) uses
`vfmaq_laneq_f32` to accumulate 8 output channels simultaneously:

```cpp
// convolution1d_packed.h — oc8 inner loop (Kw=1, one q-block-of-8)
float32x4_t _w0 = vld1q_f32(kptr);       // oc0~3 weights for ic q~q+7
float32x4_t _w1 = vld1q_f32(kptr + 4);   // oc4~7 weights
// ...16 weight vectors total

_sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);  // oc0~3 FMA with input lane 0
_sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 0);  // oc4~7 FMA with input lane 0
_sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 1);  // oc0~3 FMA with input lane 1
// ...
kptr += 64;
```

One iteration computes 8 output channels × 8 input channels = 64 FMAs. The reference
scalar computes 1 FMA per cycle; the NEON kernel issues up to 16 FMAs per cycle
(theoretical 32× speedup). Measured speedup is 5.7× — the gap is explained in section 2.

---

## 2. ncnn-baseline's own bottlenecks

Despite the NEON advantage, ncnn's `forward()` path has two structural performance costs.

### Bottleneck 1 — Gather load: `vsetq_lane_f32` dependency chain

When the input tensor has `elempack=1` (the raw `float*` layout used in this benchmark),
reading 8 elements from 8 different channel rows requires a lane-by-lane insert:

```cpp
// convolution1d_packed.h:563-574
_r0 = float32x4_t();
_r0 = vsetq_lane_f32(r0[0],     _r0, 0);  // channel q,   position j
_r0 = vsetq_lane_f32(r0[N],     _r0, 1);  // channel q+1, position j  (N = W_out)
_r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);  // channel q+2, position j
_r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);  // channel q+3, position j
// _r1: same for q+4 .. q+7
```

`vsetq_lane_f32(scalar, vec, lane)` returns a **new** vector with one lane changed — each
call reads the previous `_r0` as input, forming a 4-deep dependency chain. On Neoverse V2
each `INS Vd.S[n], Wn` has ~3 cycle latency and cannot overlap with the next:

```
_r0 = [?,?,?,?]
  ↓ INS lane0  (3 cycles)
_r0 = [ch0_j, ?, ?, ?]
  ↓ INS lane1  (3 cycles, blocked on previous)
_r0 = [ch0_j, ch1_j, ?, ?]
  ↓ INS lane2  (3 cycles)
_r0 = [ch0_j, ch1_j, ch2_j, ?]
  ↓ INS lane3  (3 cycles)
_r0 = [ch0_j, ch1_j, ch2_j, ch3_j]   ← ready after ≥12 cycles
```

All 16 subsequent `vfmaq_laneq_f32` FMAs depend on `_r0` / `_r1` and are fully stalled
until the chain completes. Compare to a single `vld1q_f32` which takes ~4 cycles with no
dependency chain. The gather raises the effective load cost by ~3×, reducing FMA pipeline
utilisation to under 40%.

### Bottleneck 2 — kptr reset: weight block re-read every output position

`kptr` is reset to the start of the weight block at the beginning of every `j` iteration:

```cpp
// convolution1d_packed.h — oc8 kernel structure
for (int j = 0; j < outw; j++)                          // W_out iterations
{
    const float* kptr = weight_data_tm.channel(p / 8);  // ← reset to start every j

    for (int q = 0; q + 7 < inh; q += 8)
    {
        // ... FMA work ...
        kptr += 64;    // advance through weight block
    }
    // kptr is at the end; next j resets it
}
```

The weight block for one oc8 group is `8 × C_in × 4` bytes (16 KB for C_in=512). This
entire block is walked **W_out times** — once per output position. The gather loads from
the input simultaneously compete for L1D bandwidth, creating cache pressure that can evict
weight cache lines even when the block nominally fits in L1.

---

## 3. Why agent v3 beats ncnn-baseline by 2.26×

### Root cause: loop order inversion (outer-product GEMM)

ncnn uses an **inner-product** structure — the outermost loop is over output positions `j`;
for each `j`, the full dot product over all `ic` is computed for 8 output channels.

The agent inverts this to an **outer-product** structure — the outer loop is over `ic`; for
each `ic`, all output positions `ow` are updated for all 16 output channels at once:

```
ncnn (inner-product):
  for j in [0, W_out):           ← output position is outermost
    for q in [0, C_in) step 8:   ← accumulate dot product
      FMA: output[:,j] += weight[:,q:q+8] × input[q:q+8,j]
      → weight re-read W_out times; input gathered per j

agent v3 (outer-product):
  for ic in [0, C_in):           ← input channel is outermost
    for ow in [0, W_out) by vl:  ← SVE sweeps all output positions
      FMA: output[oc0..15,ow:ow+vl] += input[ic,ow:ow+vl] × weight[oc0..15,ic]
      → weight loaded once; input streamed sequentially
```

This inversion creates two independent amortisation benefits.

### Amortisation axis 1 — Weight: one scalar load covers all W_out positions

```cpp
// v3.cpp:64-83
for (int ic = 0; ic < C_in; ++ic) {
    const float* in_row = input + (long)ic * W_out;

    svfloat32_t vw0  = svdup_f32(w0[ic]);   // 1 scalar load + register broadcast
    svfloat32_t vw1  = svdup_f32(w1[ic]);   // no cache pressure, ~2 cycles each
    // ... vw2 ~ vw15

    for (int ow = 0; ow + 2*vl <= W_out; ow += 2*vl) {
        // vw0..vw15 stay in registers for the entire W_out sweep
        FMA2(o0, vw0); FMA2(o1, vw1); ... FMA2(o15, vw15);
    }
}
```

`svdup_f32(scalar)` is a pure register operation — it copies one scalar into all SVE lanes
with no memory access. The single scalar load `w0[ic]` is amortised over all `W_out / vl`
inner-loop iterations. For large W_out this cost approaches zero per output element.

### Amortisation axis 2 — Input: one vector load fans out to 16 OC

```cpp
// v3.cpp:87-117 — inner loop body
svfloat32_t vin0 = svld1_f32(pg, in_row + ow);        // contiguous load, ~4 cycle, no chain
svfloat32_t vin1 = svld1_f32(pg, in_row + ow + vl);   // next vl elements

#define FMA2(op, vw) \
    do { \
        svfloat32_t va = svld1_f32(pg, op + ow); \
        svfloat32_t vb = svld1_f32(pg, op + ow + vl); \
        va = svmla_f32_x(pg, va, vin0, vw); \   // vin0 reused — stays in register
        vb = svmla_f32_x(pg, vb, vin1, vw); \   // vin1 reused — stays in register
        svst1_f32(pg, op + ow, va); \
        svst1_f32(pg, op + ow + vl, vb); \
    } while(0)

FMA2(o0, vw0); FMA2(o1, vw1); ... FMA2(o15, vw15);
// 2 input loads → 32 FMAs (16 OC × 2 unroll)
```

`vin0` and `vin1` are loaded once into SVE registers and used by all 16 `FMA2` calls
before the loop advances. The effective arithmetic intensity is:

```
32 FMAs per (2 input loads + 32 output load/store pairs)
vs ncnn: 16 FMAs per (8 gather inserts + 16 weight loads)
```

### IPC progression confirms the analysis

| Version  | OC\_TILE | W\_out unroll | IPC  | Interpretation |
|----------|---------|--------------|------|----------------|
| v1       | 4       | ×1           | 1.44 | FMA pipeline ~40% full; too few independent ops per ic step |
| v2       | 8       | ×2           | 2.11 | 8 oc × 2 unroll = 16 independent FMAs; pipeline fed well |
| v3       | 16      | ×2           | 1.83 | 32 independent FMAs; now memory BW (output load/store) is the new ceiling |

v3's IPC slightly drops from v2 because OC\_TILE=16 opens 32 output load/store operations
per inner iteration — the memory subsystem becomes the bottleneck rather than the FMA
pipes.

---

## 4. Hard-coded parameters in the agent solution

### Kw=1 — algorithmic dependency (breaks completely for Kw > 1)

The entire kernel is structured as a 2D GEMM with no loop over the kernel width:

```cpp
// v3.cpp:205-209
// Main GEMM loop: output += weight * input
// Cout=512 is divisible by 16
for (int oc = 0; oc < Cout; oc += 16) {
    gemm_oc16(input, output, weight, C_in, W_out, oc);
}
```

`gemm_oc16` only loops over `ic` and `ow`. A correct general kernel would need:

```cpp
// Required for Kw > 1 — completely absent in agent code
for (int kw = 0; kw < Kw; ++kw)
    for (int ic = 0; ic < C_in; ++ic)
        for (int ow = 0; ow < W_out; ++ow)
            output[oc][ow] += weight[oc][ic][kw] * input[ic][ow * Sw + kw * Dw];
```

For Kw=2 or Kw=3 the agent code produces incorrect results — it ignores the extra kernel
positions entirely. This is not a coding shortcut; the GEMM structure itself only holds
when Kw=1.

### Sw=1 — correctness bug (silent wrong answers for Sw ≠ 1)

The function signature receives both `W` (actual input width) and `W_out` (output width):

```cpp
// v3.cpp:183-186
extern "C" void inner_conv1d(
    const float* input, float* output,
    const float* weight, const float* bias,
    int C_in, int W, int W_out)   // both W and W_out passed in
```

But inside `gemm_oc16`, the input row pointer uses `W_out` as the row stride:

```cpp
// v3.cpp:65  — BUG: should be `ic * W`, not `ic * W_out`
const float* in_row = input + (long)ic * W_out;
```

The actual input memory layout is `(C_in, W)`, so the correct row stride is `W`. This
works only because for Sw=1, Kw=1, pad=0 we have `W_out == W`. The parameter `W` is never
used anywhere in the function body. For Sw=2, `W_out ≈ W/2`, and `input + ic * W_out`
points to the wrong address — the kernel reads garbage and writes garbage with no runtime
error. v1 (which was the first version) correctly used `W` as the row stride.

### Cout % 16 == 0 — minor (out-of-bounds for non-divisible Cout)

```cpp
// v3.cpp:207 — no tail handler
for (int oc = 0; oc < Cout; oc += 16) {
    gemm_oc16(input, output, weight, C_in, W_out, oc);  // always processes 16 channels
}
```

`gemm_oc16` unconditionally dereferences 16 output and weight row pointers. For
`Cout=100`, the iteration with `oc=96` would access `output[100..111]` and
`weight[100..111]` — out-of-bounds. A five-line fix suffices:

```cpp
for (int oc = 0; oc < Cout; oc += 16) {
    if (oc + 16 <= Cout) gemm_oc16(input, output, weight, C_in, W_out, oc);
    else                 gemm_tail(input, output, weight, C_in, W_out, oc, Cout - oc);
}
```

### Dw=1 and pad=0 — not hardcoded

- **Dw**: dilation is irrelevant when Kw=1 (only one kernel tap; no inter-tap spacing).
  The agent code is correct for any Dw value.
- **pad**: padding is applied by the harness before calling `inner_conv1d`; the kernel
  only sees the post-padded input and the pre-computed `W_out`. No hardcode.

### Summary

| Parameter | Hardcode type | Effect of changing |
|-----------|--------------|-------------------|
| Kw=1 | Algorithmic — GEMM structure | Wrong results; missing kw loop |
| Sw=1 | Correctness bug — wrong row stride | Wrong results; silent; one-line fix (`W_out` → `W`) |
| Cout % 16 == 0 | Missing tail handler | Out-of-bounds write; five-line fix |
| Dw=1 | Not hardcoded | No effect |
| pad=0 | Not hardcoded (harness handles) | No effect |
