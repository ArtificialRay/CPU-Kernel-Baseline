# Case Study: conv2d_depthwise_kh3_kw3_sh1_sw1_dh1_dw1_p1

3×3 depthwise conv2d, stride 1, dilation 1, padding 1. Output size equals input size (H_out = H, W_out = W). Benchmarked on Graviton4 (Neoverse V2, 128-bit SVE2, clang++-18 -O3 -march=armv9-a+sve2).

## Performance Trajectory

| Turn | Event | Version | time_speedup | cycle_speedup | IPC   | cache_misses |
|------|-------|---------|-------------|--------------|-------|-------------|
| —    | Reference scalar | — | 0.218×    | 0.232×       | —     | —           |
| 1    | Compile OK  | v1 | —          | —            | —     | —           |
| 2    | Correctness | v1 | PASSED     | —            | —     | —           |
| 3    | **Timing**  | v1 | **2.035×** | **1.842×**   | 3.347 | 912.4       |
| 4–5  | read_code, disassemble v1.s | — | — | — | — | —       |
| 6    | Compile OK  | v2 | —          | —            | —     | —           |
| 7    | Correctness | v2 | PASSED     | —            | —     | —           |
| 8    | Compile OK  | v3 | —          | —            | —     | —           |
| 9    | Correctness | v3 | PASSED     | —            | —     | —           |
| 10   | Timing      | v3 | 1.877×     | 1.725×       | 3.386 | 1241.3      |
| 11–13 | read_code v3, v1.s, v1.cpp | — | — | — | — | —      |
| 14   | Compile OK  | v4 | —          | —            | —     | —           |
| 15   | Correctness | v4 | PASSED     | —            | —     | —           |
| 16   | Timing      | v4 | 1.795×     | 1.681×       | 3.483 | 1071.4      |
| 17   | read_code v4 | — | —          | —            | —     | —           |
| 18   | **Submit**  | v1 | **2.041×** | **1.841×**   | —     | —           |

All speedup values are `ncnn_baseline_time / candidate_time`. Values > 1 mean the agent kernel is **faster** than ncnn. v1 achieves `time_speedup = 2.041×` — the agent is 2.04× faster than ncnn. The agent tried three successor versions, found all inferior, and correctly submitted the first version.

---

## Section 1 — Reference Scalar vs. ncnn Baseline

**Direction: ncnn is 4.6× faster than reference scalar** (0.218× time speedup → 1/0.218 = 4.59×; 0.232× cycle speedup → 4.3× in cycles).

### Why the scalar is slow

The reference-scalar kernel (`bench-trace/solutions/ncnn/reference-scalar/conv2d_depthwise/conv2d_depthwise_kh3_kw3_sh1_sw1_dh1_dw1_p1.json`) is a six-level nested loop with per-pixel boundary branches:

```cpp
for (int oh = 0; oh < H_out; ++oh)
  for (int ow = 0; ow < W_out; ++ow) {
    float sum = bias[c];
    for (int kh = 0; kh < Kh; ++kh)
      for (int kw = 0; kw < Kw; ++kw) {
        int ih = oh * Sh - pad + kh * Dh;
        int iw = ow * Sw - pad + kw * Dw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
          sum += in_c[ih * W + iw] * w_c[kh * Kw + kw];
      }
    out_c[oh * W_out + ow] = sum;
  }
```

Every output pixel requires 9 multiply-adds with four conditional checks (`ih >= 0`, `ih < H`, `iw >= 0`, `iw < W`) inside the kh/kw loops. These branches are data-dependent on the loop indices, preventing auto-vectorization of the inner body. The compiler emits scalar FMAs with branches. On a representative workload (N=1, C=64, H=W=112): 64 × 112 × 112 × 9 = 7,225,344 scalar multiply-adds, all with branch overhead per element.

### What ncnn does instead

`create_pipeline` in `convolutiondepthwise_arm.cpp:120–124` packs weights into **pack4 format** — interleaving 4 channels' 3×3 kernels consecutively so that the weight tensor strides are cache-friendly for the NEON kernel:

```cpp
// convolutiondepthwise_arm.cpp:122-124
Mat weight_data_r2 = weight_data.reshape(maxk, group);
convert_packing(weight_data_r2, weight_data_tm, 4, opt);
```

The forward pass then dispatches to `convdw3x3s1_pack4_neon` (`convolutiondepthwise_arm.cpp:352`), a hand-written AArch64 assembly kernel (`convolutiondepthwise_3x3_pack4.h`). The inner loop (lines 46–205) tiles 2 output rows × 4 output columns and computes 4 channels per NEON instruction:

```asm
// convolutiondepthwise_3x3_pack4.h ~line 73
fmla   v16.4s, %15.4s, v10.4s      // row1, col0: 4 channels at once
fmla   v17.4s, %15.4s, v11.4s      // row1, col1
fmla   v18.4s, %15.4s, v12.4s      // row1, col2
fmla   v19.4s, %15.4s, v13.4s      // row1, col3
```

Each asm block produces **2 rows × 4 cols × 4 channels = 32 output values** per invocation, against the scalar's 1 output per 9 FMAs. This 32× theoretical throughput multiplier, moderated by memory traffic, produces the measured 4.6× speedup over scalar.

---

## Section 2 — ncnn Baseline's Own Bottlenecks

Despite the efficient pack4 NEON kernel, the **benchmark harness wrapper** surrounding ncnn's forward pass adds three full-tensor memory traversals that the agent's CHW-native kernel avoids entirely.

### Pack/unpack overhead

The baseline kernel (`bench-trace/solutions/ncnn/baseline-ncnn-arm/conv2d_depthwise/conv2d_depthwise_kh3_kw3_sh1_sw1_dh1_dw1_p1.json`, `kernel.cpp`) executes this sequence on every benchmark call:

```cpp
// Step 1: CHW → pack4 format (full tensor read + write)
convert_packing(bottom_blob, bottom_pack4, 4, opt4);

// Step 2: repack weights into pack4 format
conv.create_pipeline(opt4);

// Step 3: pack4 NEON depthwise convolution
conv.forward(bottom_pack4, top_pack4, opt4);

// Step 4: pack4 → CHW format (full tensor read + write)
convert_packing(top_pack4, local_top, 1, opt);

// Step 5: copy to destination (full tensor read + write)
for (int c = 0; c < C; ++c)
    std::memcpy((float*)top_blob.channel(c), (const float*)local_top.channel(c), ...);
```

Steps 1, 4, and 5 each traverse the entire input or output tensor with no arithmetic — they are pure memory-bandwidth cost. For the C=64, H=W=112 workload (input = 64 × 112 × 112 × 4B = **3.2 MB**, output = 3.2 MB):

- Step 1: read 3.2 MB + write 3.2 MB = 6.4 MB of pack overhead
- Step 4: read 3.2 MB + write 3.2 MB = 6.4 MB of unpack overhead
- Step 5: read + write 3.2 MB = 6.4 MB of copy overhead

The actual depthwise convolution reads each input element 9 times (once per kernel position) → ~28.8 MB of input reads for the compute. The 12.8 MB of pack/unpack alone is **44%** of the compute-read volume, added purely as format-conversion overhead.

### Bordered blob allocation

Inside `conv.forward`, ncnn allocates and zero-fills a padded "bordered" input blob of size (H+2) × (W+2) per channel before the NEON kernel runs. For H=W=112, this is a 114×114 tensor — a full allocation and memset per channel group that adds latency before any compute starts.

The combined effect: ncnn's pack4 NEON kernel is wrapped by 3 extra full-tensor passes, capping its effective advantage to 4.6× over scalar despite the 32× theoretical throughput of the NEON inner loop.

---

## Section 3 — Why v1.cpp Beats ncnn

**v1 achieves `time_speedup = 2.041×`, meaning it is 2.04× faster than ncnn** (`trajectory.jsonl`, turn 18).

### Algorithm: CHW-native SVE, vectorised over output width

v1 works directly in the benchmark's native CHW format with zero data-format conversion. The 9 weights are broadcast to SVE registers once per channel (`svdup_f32`), then the interior of each output row is processed with SVE loads and FMAs (`v1.cpp:38–126`):

```cpp
// v1.cpp:38-41 — weights live in SVE registers for the entire (oh, ow) loop
svfloat32_t vw00 = svdup_f32(w00), vw01 = svdup_f32(w01), vw02 = svdup_f32(w02);
svfloat32_t vw10 = svdup_f32(w10), vw11 = svdup_f32(w11), vw12 = svdup_f32(w12);
svfloat32_t vw20 = svdup_f32(w20), vw21 = svdup_f32(w21), vw22 = svdup_f32(w22);

// v1.cpp:86-126 — interior loop: 4 output positions per SVE vector
for (int ow = 1; ow < W_out - 1; ) {
    svbool_t pg = svwhilelt_b32(ow, ow_end);
    svfloat32_t acc = vbias;
    if (row0) {
        acc = svmla_f32_m(pg, acc, svld1(pg, row0 + ow - 1), vw00);
        acc = svmla_f32_m(pg, acc, svld1(pg, row0 + ow),     vw01);
        acc = svmla_f32_m(pg, acc, svld1(pg, row0 + ow + 1), vw02);
    }
    // ... same for row1, row2
    svst1(pg, out_row + ow, acc);
    ow += vl;  // vl = svcntw() = 4 on Graviton4
}
```

Boundary pixels (ow=0, ow=W_out−1) are handled with scalar code that uses the correct subset of kernel weights (`v1.cpp:66-140`), completely removing bounds-checking branches from the hot vector path.

### Arithmetic

Both v1 and ncnn's pack4 kernel compute the same `N × C × H_out × W_out × 9` FMAs. There is no algorithmic difference. The speedup comes entirely from eliminating the pack/unpack passes. On the C=64, H=W=112 workload, ncnn incurs ~12.8 MB of format-conversion memory traffic that v1 does not — approximately 44% of the compute-phase input-read volume, translating directly to a 2× throughput advantage at this problem scale.

### Version progression and why later versions regressed

| Version | Strategy | time_speedup | cycle_speedup | IPC   | cache_misses |
|---------|----------|-------------|--------------|-------|-------------|
| v1      | 1 SVE accumulator, per-row | **2.035×** | **1.842×** | 3.347 | 912         |
| v2      | 4-row H-tile, 1 accumulator | (not timed) | — | — | —       |
| v3      | 2 SVE accumulators, per-row | 1.877×  | 1.725×       | 3.386 | 1241        |
| v4      | 4 SVE accumulators + 2-acc tail | 1.795× | 1.681×    | 3.483 | 1071        |

**v2 (H-tiling, turns 6–7)**: The agent added `TILE_OH=4`, prefetching 6 input row pointers into a local array and processing 4 output rows per outer iteration. This was correctness-checked (PASSED) but never timed — the agent moved immediately to v3, apparently expecting accumulator unrolling to have larger impact.

**v3 (2 accumulators, turns 8–10)**: Adding a second independent accumulator chain raised IPC slightly (3.386 vs 3.347) but pushed `cache_misses_mean` from 912 to **1241** — a 36% increase. The 2-accumulator inner loop issues 18 `svld1` loads per iteration (9 for acc1, 9 for acc2) accessing 18 distinct memory addresses. On Graviton4 (2 load pipelines), this doubles the concurrent load-stream count compared to v1's 9, causing more L1D pressure and evictions. The cache miss penalty outweighed the FMA-latency improvement.

**v4 (4 accumulators, turns 14–16)**: Further unrolling to 4 accumulators raised IPC to 3.483 (the highest of all versions) but dropped time_speedup to 1.795×. Each inner-loop iteration issues 36 `svld1` + 36 `svmla` instructions. On a 2-load-port machine, 36 loads saturate the load ports at 18 cycles minimum, making the kernel **load-bandwidth limited**: IPC rises (the OOO engine finds more independent FMA work) but total cycle count rises with it because loads now gate forward progress.

The diagnostic pattern: IPC increasing while time_speedup decreases signals a bottleneck shift from FMA throughput to load-port saturation. v1's single-accumulator chain, despite lower IPC, lets the OOO engine naturally interleave the next iteration's loads behind the current FMA chain — a tight, cache-friendly loop that fits the Graviton4's 2-FMA / 2-load throughput balance.

After disassembling v1 (296 lines of SVE assembly, turn 5) and re-reading v3.cpp and v1.s on turns 11–13, the agent recognised that the accumulator unrolling was counter-productive and submitted v1.

### Remaining gap

The 2.04× advantage captures essentially all available gain from eliminating ncnn's format-conversion overhead. To improve further would require reducing the memory-bandwidth footprint of the convolution itself — for example, fusing bias addition or an activation function so the output tensor is consumed before it is written to L2/DRAM. Pure SIMD tuning (more accumulators, H-tiling) cannot close the gap because load ports are already near saturation at v1's packing level.

---

## Section 4 — Hardcoded Parameters in v1.cpp

### Kh=3, Kw=3 — Algorithmic

The 3×3 kernel is fully unrolled into 9 named scalar weight variables and 9 SVE broadcast registers; there is no kh or kw loop:

```cpp
// v1.cpp:33-40
float w00 = w_c[0], w01 = w_c[1], w02 = w_c[2];
float w10 = w_c[3], w11 = w_c[4], w12 = w_c[5];
float w20 = w_c[6], w21 = w_c[7], w22 = w_c[8];

svfloat32_t vw00 = svdup_f32(w00); // ... through vw22
```

If Kh or Kw changes, the code loads wrong weights and applies wrong spatial offsets. The entire unrolled FMA body would need to be replaced with explicit kh/kw loops.

### Sh=1, Sw=1 — Algorithmic

Input row indices are computed without stride multiplication:

```cpp
// v1.cpp:45-47
int ih0 = oh - 1;   // should be oh * Sh - pad + 0 * Dh
int ih1 = oh;       // should be oh * Sh - pad + 1 * Dh
int ih2 = oh + 1;   // should be oh * Sh - pad + 2 * Dh
```

For Sh=2, ih0/ih1/ih2 would be wrong, producing incorrect output-row–to–input-row mappings. Column offsets `ow − 1`, `ow`, `ow + 1` in the SVE loads (`v1.cpp:96–126`) similarly assume Sw=1.

### Dh=1, Dw=1 — Algorithmic

The three input rows are always consecutive (`oh − 1, oh, oh + 1`) and the three input columns within each row differ by exactly 1 (`ow − 1, ow, ow + 1`). For dilation Dh > 1, the row separation would need to be `Dh` instead of 1; for Dw > 1, the column separation would need to be `Dw`.

### pad=1 — Structural

The boundary peeling always strips exactly one pixel from each edge. The interior SVE loop is hardcoded to `ow_start = 1, ow_end = W_out − 1` (v1.cpp:83-84). The scalar boundary handlers use the fixed weight indices that correspond to pad=1:

```cpp
// v1.cpp:72-74 — left boundary (ow=0): iw=-1 is out-of-bounds,
// so only kw=1 (w01) and kw=2 (w02) contribute
if (row0) { sum += row0[0] * w01; if (W > 1) sum += row0[1] * w02; }
```

For pad=0 no boundary pixels exist; for pad=2 the first two and last two columns need special-casing with different valid weight subsets.

### Summary

| Parameter | Status | Effect of changing |
|-----------|--------|-------------------|
| Kh=3      | Hardcoded (algorithmic) | Wrong results; must add kh loop and generalise row offsets |
| Kw=3      | Hardcoded (algorithmic) | Wrong results; must add kw loop and generalise column offsets |
| Sh=1      | Hardcoded (algorithmic) | Wrong row indices; needs `oh × Sh` factor |
| Sw=1      | Hardcoded (algorithmic) | Wrong column offsets in SVE loads; needs `ow × Sw` factor |
| Dh=1      | Hardcoded (algorithmic) | Wrong row spacing; needs `kh × Dh` factor |
| Dw=1      | Hardcoded (algorithmic) | Wrong column spacing in loads; needs `kw × Dw` factor |
| pad=1     | Structural | Wrong boundary region and weight-index subsets |
