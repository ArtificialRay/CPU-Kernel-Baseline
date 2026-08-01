# Case Study: `conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0`

**Definition**: Kh=1, Kw=1, Sh=1, Sw=1, Dh=1, Dw=1, pad_top=pad_left=0
(1×1 pointwise convolution; N, C_in, C_out, H, W all vary per workload — 20 workloads,
`bench-trace/workloads/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.jsonl`)

**Speedup trajectory** (`agent-runs/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0/trajectory.jsonl`,
all values are `ncnn-baseline_time / candidate_time` — **below 1.0 means the candidate is
slower than ncnn**):

| Version | Structure | Time speedup (geomean) | Cycle speedup (geomean) | IPC (mean) | Cache misses (mean) |
|---------|-----------|------------------------|--------------------------|------------|----------------------|
| v1 | CO_BLOCK=4, no HW tiling | 0.313× | 0.342× | 1.674 | 1,898,177 |
| v2 | CO_BLOCK=8, no HW tiling | 0.347× | 0.381× | 1.747 | 20,527,467 |
| **v3 (submitted)** | CO_BLOCK=8 + HW_TILE=512 | **0.371×** | **0.402×** | **1.967** | 19,002,224 |

Submitted result (turn 18): `time_speedup=0.3707`, `cycle_speedup=0.4022`,
`max_relative_error=4.94` / `max_absolute_error=5.72e-5` (PASSED — the huge relative
error comes from near-zero reference elements, absolute error is within tolerance).
v3 never beats ncnn on any version; it tops out **2.7× slower** than the baseline.

---

## 1. Reference-scalar vs ncnn-baseline: ncnn is faster, by roughly an order of magnitude

The exact reference-scalar trace for this specific (flattened) definition isn't present
in `bench-trace/traces/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.jsonl` — that file
only contains the agent's own v3 submission (its aggregate matches the trajectory
`time_speedup_geomean=0.3707` exactly, confirming the file is the candidate trace, not a
baseline trace). The pre-flattening sibling definition
`conv2d_kh1_kw1_sh1_sw1_dh1_dw1_cout256` (identical Kh=Kw=Sh=Sw=Dh=Dw=1, pad=0 shape,
just a fixed Cout=256 instead of Cout varying) has a directly measured number in
`case-study/conv2d_kh1_kw1_sh1_sw1_dh1_dw1_cout256.md`:

> reference-scalar: **0.057×** vs ncnn-baseline → **ncnn is ~17.5× faster than scalar**.

Two pieces of evidence confirm the same direction holds for this run:

**(a) The agent's own hand-vectorized SVE code never beats ncnn.** v3 uses 256-bit SVE
(`svcntw()` = 8 floats/vector), 8-way output-channel blocking, and cache tiling, yet its
best `time_speedup` is 0.371× (`agent-runs/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0/trajectory.jsonl`
turn 13/18). Since the unvectorized reference-scalar kernel has strictly less exploitable
parallelism than v3, it cannot close a gap v3 itself couldn't close — it can only be
further behind.

**(b) The reference-scalar kernel has no SIMD and a per-pixel bounds branch in the
innermost loop:**

```cpp
// bench-trace/solutions/ncnn/reference-scalar/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.json
// kernel.cpp
for (int ci = 0; ci < C_in; ++ci) {
    const float* in_c = in_n + (long)ci * H * W;
    const float* w_c  = w_co + (long)ci * Kh * Kw;
    for (int kh = 0; kh < Kh; ++kh) {
        for (int kw = 0; kw < Kw; ++kw) {
            int ih = oh * Sh - pad_top  + kh * Dh;
            int iw = ow * Sw - pad_left + kw * Dw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += in_c[ih * W + iw] * w_c[kh * Kw + kw];
        }
    }
}
```

Every single multiply-add is preceded by a data-independent-but-unpredicted bounds check
and reads one scalar float at a time — no vector loads, no register blocking, no weight
reuse across output channels. Combined with (a), the direction is unambiguous: **ncnn is
faster than reference-scalar**, on the order of the 17.5× measured on the shape-identical
sibling definition.

---

## 2. ncnn-baseline's own bottlenecks

### 2.1 — 1×1 convolutions are unconditionally routed to the im2col+GEMM path

`convolution_arm.cpp:259` (and the mirrored check at `:543` in `forward()`) hard-routes
any 1×1 kernel to the GEMM path regardless of the `prefer_sgemm` heuristic:

```cpp
// ncnn/src/layer/arm/convolution_arm.cpp:259
if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
{
    convolution_im2col_gemm_transform_kernel(weight_data, weight_sgemm_data,
                                              num_input, num_output, kernel_w, kernel_h, opt);
    ...
    return 0;
}
```

For Kh=Kw=1, "im2col" is a no-op reshape (`convolution_im2col_gemm.h:6772-6774`,
`maxk == 1` branch) — the routine is really a straight `[Cout,Cin] x [Cin,HW] -> [Cout,HW]`
GEMM, packed and tiled for cache.

### 2.2 — Input (B-matrix) must be repacked from scratch on every forward call

Weights (`A`) are packed once in `create_pipeline` (`convolution_im2col_gemm.h:6749`,
`convolution_im2col_gemm_transform_kernel`). The input tensor is **not** — it is
repacked into the `BT` tile buffer on every inference:

```cpp
// ncnn/src/layer/arm/convolution_im2col_gemm.h:6838
Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
...
// ncnn/src/layer/arm/convolution_im2col_gemm.h:6844-6860
#pragma omp parallel for num_threads(nT)
for (int ppjk = 0; ppjk < nn_NK; ppjk++) {
    ...
    convolution_im2col_input_tile(bottom_blob, BT_tile, j, max_jj, k, max_kk,
                                   kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}
```

For workload 4 (`C_in=64, HW=65536`, the largest in this definition's workload set) this
reads and rewrites `C_in * HW * 4 bytes = 16 MB` before a single FMA runs — pure overhead
that never shows up in the FLOP count.

### 2.3 — L2-sized K/N tiling still means the largest workloads spill to L3/DRAM

`convolution_im2col_gemm_get_optimal_tile_mnk` (`convolution_im2col_gemm.h:5937-6035`)
sizes `TILE_K` and `TILE_N` from `get_cpu_level2_cache_size()` so a `TILE_K x TILE_N` `BT`
panel plus the `AT` tile fit L2. On Neoverse V1 (Graviton3, c7g.large — this run's target
per the SVE width in the agent's own comments, `vl = svcntw() = 8`, i.e. 256-bit SVE) L2
is 1 MB/core. For `K=64` this keeps `TILE_K=K=64` (no K-split) and bounds `TILE_N` to a
few thousand columns — the panel is re-streamed from L3/DRAM roughly `HW / TILE_N` times
per M-tile pass. This cost is small relative to §2.2 for this op family, but it means
ncnn is not immune to memory-bandwidth limits on the largest spatial workloads either
(consistent with the ncnn-relative agent numbers still climbing, not just flatlining, as
`HW` grows in the workload set).

---

## 3. Why agent v3 (`time_speedup=0.371`) is still 2.7× slower than ncnn

`time_speedup = ncnn_time / candidate_time`; v3's 0.371 means the candidate takes
`1/0.371 ≈ 2.7×` longer than ncnn-baseline on the geomean across the 20 workloads.

### Step 1 — Algorithm: both sides run the *same* GEMM, so there is no algorithmic ceiling to close

Unlike 3×3/5×5/7×7 conv2d definitions where ncnn may dispatch to Winograd or a
depth-varying packed kernel, `create_pipeline` shows 1×1 kernels are **always** GEMM
(§2.1) — and for Kh=Kw=1 im2col is a no-op reshape, so ncnn performs exactly the same
`M×N×K` multiply-adds as the agent's direct loop nest. For workload 4
(`M=Cout=256, N=HW=65536, K=Cin=64`, from
`bench-trace/workloads/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.jsonl` row 4):

```
FLOPs = 2 * M * N * K = 2 * 256 * 65536 * 64 ≈ 2.15 GFLOP   (identical on both sides)
```

**`direct_muls / ncnn_algorithm_muls = 1.0×`** — the entire 2.7× gap is execution
efficiency (memory traffic, register use), not a missed algorithmic optimization. This
is the key structural fact distinguishing this definition from Winograd-eligible ones.

### Step 2 — Where the execution gap comes from: partial sums round-trip through memory in the agent's kernel, but stay in registers in ncnn's

**v3's inner loop** (`agent-runs/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0/v3.cpp:105-135`)
reloads the 8 output accumulators from memory, FMAs, and stores them back — **on every
single `ci` iteration**:

```cpp
// v3.cpp:105-135 (inside the CO_BLOCK=8, HW_TILE loop)
for (int hw = hw_base; hw < hw_end; hw += vl) {
    svfloat32_t vin = svld1_f32(pg, in_ci + hw);
    svfloat32_t vout0 = svld1_f32(pg, out_co0 + hw);   // ← reload partial sum
    ...
    svfloat32_t vout7 = svld1_f32(pg, out_co7 + hw);
    vout0 = svmla_f32_x(pg, vout0, vin, vw0);
    ...
    svst1_f32(pg, out_co0 + hw, vout0);                // ← store partial sum back
    ...
    svst1_f32(pg, out_co7 + hw, vout7);
}
```

The generated assembly confirms this literally: for every `ci`, 8 `ld1w` + 8 `fmla` +
8 `st1w` (`agent-runs/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0/v3.s:232-258`). For workload
4 (`C_in=64`), each output element is loaded and stored **64 times** to accumulate one
final value — 64× more memory traffic than necessary per element.

**ncnn's microkernel never does this.** `convolution_gemm_transB_packed_tile`
(`convolution_im2col_gemm.h:178-330`) keeps an 8×12 output tile resident in 24 NEON
vector registers (`v8`–`v31`) for the *entire* K-reduction, writing to memory only once
at `k_end`:

```cpp
// convolution_im2col_gemm.h:200-263 — accumulators zero-initialized (or loaded from bias) once...
"mov v9.16b, v8.16b \n" ... "mov v19.16b, v8.16b \n"   // 11 more copies, all-register init
...
// convolution_im2col_gemm.h:279-317 — K-loop body: only A/B loads and FMAs, no C traffic
"ldr d1, [%2], #8 \n"
"fmla v8.4s, v4.4s, v0.s[0] \n"
"fmla v9.4s, v4.4s, v0.s[1] \n"
...   // 24 fmla per 4-K-step iteration, zero ld/st on the C accumulators
```

Output (`outptr0`) is only touched after the full `max_kk` reduction for a tile
completes. This single difference — accumulate-in-registers vs. accumulate-in-memory —
is the dominant reason ncnn's *identical* FLOP count executes faster.

### Step 3 — Version-by-version: v1→v2→v3 traded output tile size against cache pressure

| Version | OC_TILE | HW handling | IPC | Cache misses (mean) | Time speedup |
|---------|---------|-------------|-----|----------------------|-------------|
| v1 | 4 | full HW per pass | 1.674 | 1,898,177 | 0.313× |
| v2 | 8 | full HW per pass | 1.747 | 20,527,467 | 0.347× |
| v3 | 8 | HW tiled (HW_TILE=512) | 1.967 | 19,002,224 | **0.371×** |

- **v1→v2 (CO_BLOCK 4→8):** doubling the output-channel tile amortizes each SVE weight
  broadcast (`svdup_f32`) over twice as many FMA lanes and roughly halves the number of
  `ci`-loop passes needed to cover `C_out`. IPC rises slightly (1.674→1.747) and
  `time_speedup` improves (0.313→0.347) — but `cache_misses_mean` jumps **10.8×**
  (1.9M→20.5M). Doubling `CO_BLOCK` doubles the number of live output planes touched per
  `ci` step (from 4×`HW` floats to 8×`HW` floats); for the workload set's largest-`HW`
  entries this working set stops fitting in L1/L2, and the mean is dominated by those
  outliers. The net win happens only because most of the 20 workloads have small `HW`
  and are FMA-bound, not memory-bound, so the amortization gain outweighs the cache
  regression on average.

- **v2→v3 (add HW_TILE=512):** v3's own header comment explains the motivation directly
  (`v3.cpp:7-10`): *"Large workloads (256x256, 128x128) have terrible IPC (0.66-0.95) and
  massive cache misses... because we're reading output[co,hw] for 8 different co channels
  repeatedly for each ci."* Tiling `HW` into 512-element chunks bounds the working set
  per `(co_block, hw_tile)` pass to `8 * 512 * 4B = 16 KB` output + `512 * 4B = 2 KB`
  input per `ci` — intended to fit L1. IPC improves further (1.747→1.967) and
  `cache_misses_mean` drops modestly (20.5M→19.0M, ~7%).

  **But the per-workload trace
  (`bench-trace/traces/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.jsonl`) shows the
  fix only partially worked**: workload 4 (`HW=65536, Cin=64, Cout=256`) still has
  `ipc=0.931` and `cache_misses=244,801,676` — the single largest outlier in the entire
  set, essentially unchanged in character from the pre-tiling problem the comment
  describes. The reason HW-tiling can't fix it: v3's loop order is
  `co_block (outer) → hw_tile → ci (reduction)` (`v3.cpp:51-62,87`). Fixing the *output*
  working set per tile does nothing about the *input* — for each of the
  `C_out/CO_BLOCK = 256/8 = 32` output-channel blocks, the tiled loop still re-scans the
  **entire** input tensor once. Total input bytes streamed for workload 4:

  ```
  C_in * HW * 4B * (C_out / CO_BLOCK) = 64 * 65536 * 4 * 32 ≈ 512 MB
  ```

  512 MB streamed from a machine with a 1 MB L2/core is why `cache_misses` stays at
  ~245M despite the output-side fix. ncnn avoids exactly this: `convolution_im2col_gemm`
  (`convolution_im2col_gemm.h:6844-6860`) packs each `(j,k)` `BT` panel **once**, before
  the M-loop (`:6870-6898`) even starts, and the M-loop (which iterates over `Cout`
  blocks) reuses the already-packed panels — ncnn's input traffic is `O(C_in * HW)`
  once, not `O(C_in * HW * C_out/CO_BLOCK)`.

### Step 4 — Remaining gap and what would close it

The arithmetic ceiling is 1.0× (Step 1) — there is no Winograd-style algorithmic
shortcut available for a 1×1 kernel, so **pure SIMD/cache tuning is, in principle,
sufficient to reach parity with ncnn** on this definition; it does not require adopting
a different algorithm. What v3 is missing relative to ncnn's microkernel is specifically:
(1) keeping the `CO_BLOCK` accumulators resident in registers across the full `ci`
reduction instead of round-tripping through memory every step, and (2) restructuring the
loop nest so each input element is read once per forward call regardless of how many
`CO_BLOCK` groups exist (e.g. `ci` outer to `co_block`, or packing/caching the input tile
once and reusing it across all `co_block` passes) — the two mechanisms ncnn's packed GEMM
already uses. Closing (1) alone would remove the per-`ci` memory round trip seen in
`v3.s:232-258`; closing (2) would remove the 32× input over-read on workload 4 specifically.

---

## 4. Hardcoded parameters in the agent solution

The kernel signature it was given never receives `Kh, Kw, Sh, Sw, Dh, Dw, pad_top,
pad_left` — only `N, C_in, H, W, C_out, H_out, W_out`
(`bench-trace/solutions/ncnn/claude-sonnet-4-6/conv2d/conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0.json`,
`conv2d.h`). v3's structure encodes the Kh=Kw=Sh=Sw=1 assumption directly into the loop
nest rather than reading it as data.

**Kh=1, Kw=1 — algorithmic, no kernel-tap loop exists**

```cpp
// v3.cpp:64 — weight treated as a flat (C_out, C_in) matrix
const float* w_row = weight + (long)(co) * C_in + ci;
float w0 = w_row[(long)0 * C_in];
```

Weight layout is `(C_out, C_in, Kh, Kw)`; indexing it as `co * C_in + ci` with no `kh,kw`
term is only correct because `Kh=Kw=1` collapses the last two axes to size 1. For any
`Kh,Kw > 1` this reads the wrong weight entirely and never accumulates over kernel taps
— silently wrong output, no crash.

**Sh=1, Sw=1 — silent correctness bug, not a crash, if reused elsewhere**

```cpp
// v3.cpp:20,63 — HW used as the input channel stride
const int HW = H_out * W_out;
...
const float* in_ci = in_n + (long)ci * HW + hw_base; /* via out_co pointers */
```

Input is laid out `[N, C_in, H, W]`, so channel `ci` should start at `ci * H * W`. v3
uses `ci * H_out * W_out` instead. For this definition `Sh=Sw=1, Kh=Kw=1, pad=0` forces
`H_out==H, W_out==W`, so `HW == H*W` and the code is correct — but only as a consequence
of this specific definition's fixed axes, not because the code checks them. The harness
passes `H` and `W` into `inner_conv2d` and v3 never reads them for indexing.

**Cout % 8 == 0 — correctly handled, not hardcoded**

Unlike some sibling runs on this op family, this run's v3 *does* include a remainder
loop for `Cout % CO_BLOCK != 0`:

```cpp
// v3.cpp:141-165
for (; co < C_out; ++co) {          // remaining channels after the CO_BLOCK=8 loop
    float* out_co0 = out_n + (long)co * HW;
    ...
}
```

All 20 workloads' `C_out` values (16, 72, 120, 240, 256, 24, 480, 672, 40, 512, 960,
80, 128, 112, 2048) are exercised regardless of divisibility by 8, and this path is
exercised for e.g. `C_out=120` (workload 2) and `C_out=40` (workload 8). No out-of-bounds
risk here.

**Dh=1, Dw=1 / pad_top=pad_left=0 — genuinely irrelevant for Kh=Kw=1**

Dilation only affects spacing between kernel taps; with a single tap there is nothing to
dilate. Padding is a property of how the *harness* computes `H_out/W_out` before calling
`inner_conv2d` (`bench-trace/solutions/ncnn/reference-scalar/conv2d/.../conv2d.cpp`
computes `H_out` from `pad_top` before the call); the kernel itself never needs the pad
values.

### Summary

| Parameter | Hardcode type | Effect of changing |
|-----------|---------------|---------------------|
| Kh=1, Kw=1 | Algorithmic | No kernel-tap loop; wrong results for any Kh,Kw > 1 |
| Sh=1, Sw=1 | Correctness bug (dormant) | Input channel stride uses `H_out*W_out` instead of `H*W`; only correct because this definition forces `H_out==H, W_out==W` |
| Cout % 8 | Not hardcoded | Remainder loop present and exercised by non-multiple-of-8 workloads |
| Dh=1, Dw=1 | Not hardcoded — irrelevant | No effect (no kernel taps to space) |
| pad_top=0, pad_left=0 | Not hardcoded — handled by harness | No effect (padding resolved before `inner_conv2d` is called) |
