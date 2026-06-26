# Case Study: `conv2d_kh3_kw3_sh1_sw1_dh1_dw1_cout128`

**Definition**: Kh=3, Kw=3, Sh=1, Sw=1, Dh=1, Dw=1, Cout=128, pad=1 (same padding)  
**Speedup** is defined as `ncnn_baseline_time / candidate_time` (>1 = candidate beats ncnn; <1 = candidate is slower than ncnn).

**Speedup trajectory** (vs ncnn-baseline, geomean across workloads):

| Turn | Version | Time speedup | Cycle speedup | IPC  | Cache misses (mean) |
|------|---------|-------------|---------------|------|---------------------|
| —    | reference-scalar | 0.119× | 0.120× | ~1.0 | — |
| 2    | v1 (vectorize over oc — wrong layout) | INCORRECT | — | — | — |
| 4–5  | v2 (SVE over ow, valid-range precompute) | 0.452× | 0.453× | 2.96 | 2,801,130 |
| 9–10 | v3 (OC\_BLOCK=4) | 0.572× | 0.573× | 3.49 | 1,285,563 |
| 15–16 | v4 (OC\_BLOCK=8, explicit unroll) | **0.639×** | **0.641×** | 1.65 | 746,497 |

> **Important**: even the best agent version (v4, 0.639×) is **1.56× slower** than the ncnn
> baseline. The gap is algorithmic, not micro-architectural.

---

## 1. Why reference-scalar is ~8.4× slower than ncnn-baseline

### What ncnn does right: Winograd F(6,3) algorithm

For 3×3 convolution with stride=1, dilation=1, and `num_input >= 8 || num_output >= 8`, ncnn's
`create_pipeline` selects Winograd F(6,3) unconditionally
(`convolution_arm.cpp:204–219`):

```cpp
// convolution_arm.cpp:204
bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution
                     || opt.use_winograd63_convolution) && (num_input >= 8 || num_output >= 8);

if (opt.use_winograd_convolution && prefer_winograd
        && kernel_w == 3 && kernel_h == 3
        && dilation_w == 1 && dilation_h == 1
        && stride_w == 1 && stride_h == 1)
{
    // F(6,3): valid for num_input <= 128 && num_output <= 128
    if (opt.use_winograd63_convolution && (num_input <= 128 && num_output <= 128))
        conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data,
                                              num_input, num_output, opt);
    ...
    return 0;   // kernel transform is offline; forward() does input-transform + GEMM + output-transform
}
```

This problem (Cout=128, Cin≤128) hits the F(6,3) branch every time.

#### What Winograd F(6,3) actually does

Winograd transforms the convolution into a GEMM in a transform domain where fewer
multiplications are needed.  The key formula: **F(m, r)** produces `m` outputs from an
`r`-tap kernel using only `m + r - 1` multiplications instead of `m × r`.

For F(6, 3): `m=6` outputs from a `r=3` kernel → uses **8 multiplications** instead of 18.

Applied in 2D, one 8×8 input tile → 6×6 output tile:
- Direct conv: 6 × 6 × 9 = **324 multiply-adds** per (tile, ic)
- Winograd F(6,3): 8 × 8 = **64 multiply-adds** per (tile, ic) in transform domain

This is a **5.06× reduction in multiply-add count** before any SIMD is applied.

The three stages at runtime (`conv3x3s1_winograd63`, `convolution_3x3_winograd.h:9304`):

```cpp
// convolution_3x3_winograd.h:9304
static int conv3x3s1_winograd63(const Mat& bottom_blob, Mat& top_blob,
                                 const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    // pad to 6n+2, winograd F(6,3)
    int w_tiles = (outw + 5) / 6;
    int h_tiles = (outh + 5) / 6;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;   // = Cout
    const int N = tiles;                              // = number of 6x6 output tiles
    const int K = bottom_blob.c * bottom_blob.elempack; // = Cin
    const int B = 64;                                 // = 8x8 transform size

    // Stage 1: transform input tiles into transform domain  →  BT[B, N, K]
    conv3x3s1_winograd63_transform_input_tile(...);

    // Stage 2: B independent GEMMs  →  top_tile = AT × BT  (AT pre-transformed at create_pipeline)
    conv3x3s1_winograd_gemm_transB_packed_tile(...);

    // Stage 3: inverse-transform output tiles back to spatial domain
    conv3x3s1_winograd63_transform_output_tile(...);
}
```

#### The input transform matrix (itm[8][8]):

```cpp
// convolution_3x3_winograd.h:7495–7503
// const float itm[8][8] = {
//     {1.0f, 0.0f,-5.25f, 0.00f, 5.25f, 0.00f,-1.0f, 0.0f},
//     {0.0f, 1.0f, 1.00f,-4.25f,-4.25f, 1.00f, 1.0f, 0.0f},
//     {0.0f,-1.0f, 1.00f, 4.25f,-4.25f,-1.00f, 1.0f, 0.0f},
//     {0.0f, 0.5f, 0.25f,-2.50f,-1.25f, 2.00f, 1.0f, 0.0f},
//     {0.0f,-0.5f, 0.25f, 2.50f,-1.25f,-2.00f, 1.0f, 0.0f},
//     {0.0f, 2.0f, 4.00f,-2.50f,-5.00f, 0.50f, 1.0f, 0.0f},
//     {0.0f,-2.0f, 4.00f, 2.50f,-5.00f,-0.50f, 1.0f, 0.0f},
//     {0.0f,-1.0f, 0.00f, 5.25f, 0.00f,-5.25f, 0.0f, 1.0f}
// };
```

ncnn implements this transform with `float32x4_t` NEON intrinsics (two 4-wide lanes covering
the 8 elements), exploiting the many zeros and symmetries to minimise actual multiplications.

#### The weight transform matrix (ktm[8][3]):

```cpp
// convolution_3x3_winograd.h:7390–7398
// const float ktm[8][3] = {
//     {1.0f, 0.0f, 0.0f},
//     {-2.0f/9,  -2.0f/9,  -2.0f/9},
//     {-2.0f/9,   2.0f/9,  -2.0f/9},
//     {1.0f/90,  1.0f/45,  2.0f/45},
//     {1.0f/90, -1.0f/45,  2.0f/45},
//     {1.0f/45,  1.0f/90,  1.0f/180},
//     {1.0f/45, -1.0f/90,  1.0f/180},
//     {0.0f, 0.0f, 1.0f}
// };
```

This transform is applied **offline** in `create_pipeline` and stored in `weight_winograd63_data`,
so it has zero runtime cost.

#### Tiled L2-cache-aware GEMM

After input transform, the GEMM runs `B=64` independent matrix multiplications (one per
transform-domain position).  ncnn selects tile sizes at runtime to stay within L2 cache:

```cpp
// convolution_3x3_winograd.h:4451–4478
static void conv3x3s1_winograd_get_optimal_tile_mnk(int M, int N, int K, int B,
                                                      int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    int l2_cache_size_fp32 = get_cpu_level2_cache_size() / sizeof(float);
    // ...
    int tile_size = (l2_cache_size_fp32 - 32) / 12;  // aarch64 12-wide output tile
    TILE_K = std::max(8, tile_size / 8 * 8);
    // ...
}
```

The inner GEMM kernel uses hand-written AArch64 inline assembly with `ld4`, `uzp1/uzp2`,
prefetch (`prfm pldl1keep`), and a 12×8 output register block to maximise FMA throughput.

### What reference-scalar does wrong

```cpp
// bench-trace/solutions/ncnn/reference-scalar/conv2d/.../kernel.cpp
extern "C" void inner_conv2d(
    const float* input, float* output, const float* weight,
    int N, int C_in, int H, int W, int H_out, int W_out)
{
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < Cout; ++oc) {          // oc outermost
            float* outc = out_n + (long)oc * H_out * W_out;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < C_in; ++ic) {
                        const float* inc = in_n + (long)ic * H * W;
                        for (int kh = 0; kh < Kh; ++kh) {
                            for (int kw = 0; kw < Kw; ++kw) {
                                int ih = oh * Sh - pad_top  + kh * Dh;
                                int iw = ow * Sw - pad_left + kw * Dw;
                                float px = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                           ? inc[ih * W + iw] : 0.0f;  // branch per pixel
                                int widx = ((oc * C_in + ic) * Kh + kh) * Kw + kw;
                                sum += px * weight[widx];               // scalar FMA
                            }
                        }
                    }
                    outc[oh * W_out + ow] = sum;
                }
            }
        }
    }
}
```

Three compounding problems:

| Problem | Effect |
|---------|--------|
| No SIMD | 1/8 of peak float throughput vs 8-wide SVE/NEON |
| `oc` outermost loop | Every input pixel re-read 128 times (Cout=128), thrashing cache |
| Boundary `if` in innermost loop | 1 branch per multiply-add, destroys branch predictor |

---

## 2. The ncnn baseline's own bottleneck

Winograd F(6,3) cuts multiply-add count by ~4.4–5×, but it introduces structural costs that
make it 1.56× slower than optimal for this problem.  There are three compounding issues.

### 2a. Three-pass memory traffic — the unavoidable Winograd tax

Direct convolution can be computed in a single streaming pass over input and output:
```
input → [FMA loop] → output      (1 read of input, 1 write of output)
```

Winograd F(6,3) requires three separate passes over the spatial data:
```
Pass 1 (input transform):   input  → BT       (read Cin×H×W, write 64×tiles×Cin)
Pass 2 (GEMM):              AT×BT  → top_tile  (read AT+BT, write 64×tiles×Cout)
Pass 3 (output transform):  top_tile → output  (read top_tile, write Cout×H_out×W_out)
```

Where `B=64` comes from the 8×8 transform tile size (`convolution_3x3_winograd.h:9317`):
```cpp
const int B = 64;   // 8x8 Winograd transform domain
```

The intermediate tensors are large.  On the actual workloads in this definition
(Graviton3 L2 = 1 MB per core):

| Cin | H×W | AT (weights, KB) | BT (transformed input, KB) | Total working set |
|-----|-----|-------------------|---------------------------|-------------------|
| 64  | 224×224 | 2048 | 23,104 | 71,360 KB — **70× L2** |
| 128 | 56×56   | 4096 |  3,200 | 10,496 KB — **10× L2** |
| 64  | 28×28   | 2048 |    400 |  3,248 KB — **3× L2** |
| 64  | 14×14   | 2048 |    144 |  2,480 KB — **2.4× L2** |
| 11  | 28×28   |  352 |     69 |  1,221 KB — **1.2× L2** |

**AT (the pre-transformed weight tensor, `64 × Cout × Cin` floats) is the key constraint.**
Even for the smallest typical workload (Cin=64): AT = 64 × 128 × 64 × 4 = 2 MB — already
**2× the L2 size**.  Every GEMM call starts by reading AT cold from L3.  ncnn's tiling
(`conv3x3s1_winograd_get_optimal_tile_mnk`) partitions AT into `TILE_M × TILE_K` blocks that
*do* fit in L2, but this means the BT column tiles must be re-read for each TILE_M stripe —
the total DRAM traffic is multiplied by the number of TILE_M stripes.

ncnn mitigates this with cache-prefetch instructions in the inline assembly GEMM kernel:

```cpp
// convolution_3x3_winograd.h:99–103 (inside pack_B_tile transpose)
asm volatile(
    "prfm   pldl1keep, [%0, #512]       \n"
    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
    ...
```

But prefetch can only hide, not eliminate, L3 latency.

### 2b. Tile padding wastes computation at small spatial sizes

```cpp
// convolution_3x3_winograd.h:9309–9312
// pad to 6n+2, winograd F(6,3)
int w_tiles = (outw + 5) / 6;   // ceiling division — always rounds up
int h_tiles = (outh + 5) / 6;
int tiles   = w_tiles * h_tiles;
```

F(6,3) works on 6×6 output tiles, which means the spatial dimensions must be padded to a
multiple of 6.  Any partial tile on the border must still be computed in full and then
discarded.  The fraction wasted depends on alignment:

| H=W  | h_tiles | Covered | Waste |
|------|---------|---------|-------|
| 224  | 38      | 228     | 3.5%  |
| 56   | 10      | 60      | 12.9% |
| 28   | 5       | 30      | 12.9% |
| **14**   | **3** | **18**  | **39.5%** |

For H=W=14 — a common feature map size in ResNet stages — **39.5% of all three passes
(transform, GEMM, inverse-transform) operate on padding pixels that are immediately discarded**.
This severely limits Winograd's advantage at the tail of the network.

Winograd's theoretical multiply-add reduction for H=W=14:

```
direct_muls  = 128 × 64 × 14 × 14 × 9  = 14.5M
winograd_muls = 64 × 128 × 9 × 64  = 4.7M      (9 tiles, not 14×14/36 ≈ 5.4 equivalent)

Effective speedup: 14.5M / 4.7M ≈ 3.1×   (vs 4.4× at H=56)
```

### 2c. The output transform is a separate write-back pass

After the GEMM, each 8×8 transform-domain output tile must be inverse-transformed back to a
6×6 spatial tile.  The output transform matrix (`convolution_3x3_winograd.h:8455`):

```cpp
static inline void conv3x3s1_winograd63_transform_output_tile(
    const Mat& top_tile, Mat& top_blob, const Mat& bias,
    int i, int max_ii, int j, int max_jj)
{
    // Applies inverse transform:
    // Each 8-element transform-domain column → 6 spatial outputs
    // Uses ~10 add/sub operations per element (no multiplications)
    ...
}
```

The inverse transform itself uses only additions (a key Winograd property), but it still
requires reading the entire `top_tile` tensor (64 × tiles × Cout floats) from memory and
writing the result back to `top_blob` in a different layout.  For H=56, W=56, Cout=128:
- `top_tile` = 64 × 100 × 128 × 4 = 3.2 MB — again exceeds L2
- This is a third, separate L3 read on top of the two passes in §2a

In summary: ncnn's Winograd bottleneck is **memory bandwidth**, not arithmetic.  All three
intermediate tensors (BT, top_tile, and AT itself) exceed L2 on every realistic workload in
this definition.  The GEMM arithmetic is compute-efficient, but it is fed by L3-bandwidth-
limited loads, not L1/L2 cache hits.

---

## 3. Why the agent improved from 0.119× to 0.639× — but never beat ncnn

> **Key clarification**: the agent's best version (v4, 0.639×) is still **1.56× slower** than
> ncnn.  The agent never beats ncnn in this problem.  The 0.119× → 0.639× improvement is the
> journey from "terrible scalar" to "decent SIMD direct convolution" — a separate story from
> "beating Winograd".

The agent improved by fixing three independent problems, each contributing a multiplicative gain:

| Fix | v2 → v3 / v3 → v4 gain | Primary mechanism |
|-----|------------------------|-------------------|
| Vectorize over `ow` (8-wide SVE) | ~8× FP throughput | Replace scalar FMA with `svmla_f32_m` |
| Reorder loops, pre-compute valid range | Eliminate per-pixel branch | `ow_start/ow_end` computed once per (kh,kw) |
| OC\_BLOCK=4 → 8 register blocking | 4× input load amortisation | One `svld1_f32` serves 8 FMAs |

### Fix 1: SVE vectorization over `ow` (reference-scalar → v2, ×3.8)

The critical insight was choosing the **right dimension to vectorize**.  The first attempt (v1)
tried to vectorize over `oc`, but weights for different `oc` at the same `(ic,kh,kw)` position
are separated by `C_in × Kh × Kw` floats — they are **not contiguous**.  Output elements for
the same `oc` but different `ow` are contiguous: `output[oc, oh, ow]` → trivially vectorizable.

```cpp
// v2.cpp — loop order: oc → ic → kh → kw → oh → ow(SVE)
for (int oc = 0; oc < Cout; ++oc) {
    for (int ic = 0; ic < C_in; ++ic) {
        const float* w_base = weight + ((long)oc * C_in + ic) * Kh * Kw;
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                float wval = w_base[kh * Kw + kw];
                svfloat32_t vw = svdup_f32(wval);  // broadcast scalar weight

                for (int oh = 0; oh < H_out; ++oh) {
                    // Pre-compute valid ow range — eliminates boundary branch
                    int ow_start = pad_left - kw; if (ow_start < 0) ow_start = 0;
                    int ow_end   = W - kw + pad_left; if (ow_end > W_out) ow_end = W_out;

                    const float* in_row  = in_ic + ih * W + (kw - pad_left);
                    float*       out_row = out_oc + oh * W_out;

                    for (int ow = ow_start; ow < ow_end; ) {
                        svbool_t pg = svwhilelt_b32(ow, ow_end);
                        svfloat32_t vin  = svld1_f32(pg, in_row  + ow);  // 8 floats
                        svfloat32_t vout = svld1_f32(pg, out_row + ow);
                        vout = svmla_f32_m(pg, vout, vin, vw);           // 8-wide FMA
                        svst1_f32(pg, out_row + ow, vout);
                        ow += svcntw();  // advance by SVE vector width (8 at 256-bit)
                    }
                }
            }
        }
    }
}
```

The boundary branch is gone: `ow_start/ow_end` clip the range once per `(kh,kw)` tuple; the
inner loop body is branch-free.  Cache misses after this fix: **2,801,130**.

The remaining problem: for each `(ic, kh, kw)`, the same `in_row` is read **128 times** (once
per `oc`), because `oc` is the outermost loop.  This causes repeated L2/L3 pressure.

### Fix 2: OC\_BLOCK=4 register tiling (v2 → v3, ×1.27)

Reorder the outer loops so that 4 output channels share the same `(ic, kh, kw)` pass, reading
`in_row` once and sending it to 4 FMAs:

```cpp
// v3.cpp — loop order: oc_block4 → ic → kh → kw → oh → [4 oc × ow(SVE)]
const int OC_BLOCK = 4;
for (int oc_base = 0; oc_base < Cout; oc_base += OC_BLOCK) {
    float* out_oc[4] = { out_n + (long)(oc_base+0)*H_out_W_out,
                         out_n + (long)(oc_base+1)*H_out_W_out,
                         out_n + (long)(oc_base+2)*H_out_W_out,
                         out_n + (long)(oc_base+3)*H_out_W_out };

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                float wvals[4];
                for (int i = 0; i < 4; i++)
                    wvals[i] = weight[((oc_base+i)*C_in + ic)*KhKw + kh*Kw + kw];

                for (int oh = 0; oh < H_out; ++oh) {
                    const float* in_row = in_ic + ih * W + (kw - pad_left);

                    // Same in_row used for all 4 oc — 4× amortisation
                    for (int i = 0; i < oc_count; i++) {
                        svfloat32_t vw = svdup_f32(wvals[i]);
                        float* out_row = out_oc[i] + oh * W_out;
                        int ow = ow_start;
                        while (ow < ow_end) {
                            svbool_t pg = svwhilelt_b32(ow, ow_end);
                            svfloat32_t vin  = svld1_f32(pg, in_row  + ow); // same pointer
                            svfloat32_t vout = svld1_f32(pg, out_row + ow);
                            vout = svmla_f32_m(pg, vout, vin, vw);
                            svst1_f32(pg, out_row + ow, vout);
                            ow += svcntw();
                        }
                    }
                }
            }
        }
    }
}
```

Cache misses drop from 2,801,130 to **1,285,562** (−54%).  IPC rises from 2.96 to **3.49**
(the processor is better saturated with independent instructions).

### Fix 3: OC\_BLOCK=8 with explicit register unrolling (v3 → v4, ×1.12)

Double the block to 8 output channels and pull all 8 `vw` and 8 `vout` vectors into named
local variables, preventing the compiler from rematerialising them across the store/load sequence:

```cpp
// v4.cpp — the fast path (oc_count == OC_BLOCK == 8)
svfloat32_t vw0 = svdup_f32(wvals[0]);
svfloat32_t vw1 = svdup_f32(wvals[1]);
// ... vw2..vw7

float* out_row0 = out_oc[0] + oh * W_out;
// ... out_row1..out_row7

int ow = ow_start;
while (ow < ow_end) {
    svbool_t pg = svwhilelt_b32(ow, ow_end);
    svfloat32_t vin = svld1_f32(pg, in_row + ow);  // ONE load serves all 8 FMAs

    svfloat32_t vout0 = svld1_f32(pg, out_row0 + ow);
    // ... vout1..vout7

    vout0 = svmla_f32_m(pg, vout0, vin, vw0);
    vout1 = svmla_f32_m(pg, vout1, vin, vw1);
    // ... 8 FMAs total

    svst1_f32(pg, out_row0 + ow, vout0);
    // ... 8 stores total

    ow += vl;
}
```

Cache misses: 1,285,562 → **746,497** (−42%).  However, IPC drops from 3.49 to **1.65** —
the store bandwidth is now saturated (8 concurrent 256-bit stores per loop iteration exceeds
the L2 store bandwidth for large spatial sizes).

### Why 0.639× is the ceiling for direct convolution against Winograd

Even with perfect SIMD execution, direct 3×3 convolution always performs:

```
multiply-adds = N × Cout × Cin × H_out × W_out × Kh × Kw
              = N × 128  × Cin × H_out × W_out × 9
```

Winograd F(6,3) performs (ignoring transform overhead):

```
multiply-adds = N × (Cout × Cin × 64) × tiles
              = N × (Cout × Cin × 64) × ⌈H_out/6⌉ × ⌈W_out/6⌉
              ≈ N × 128 × Cin × H_out × W_out × (64/36)  =  1.78× fewer
```

For large spatial sizes where transform overhead is negligible, Winograd always wins by ~5×
in raw multiply-add count.  The agent's register blocking reduces cache pressure but cannot
reduce the number of arithmetic operations.

---

## 4. Hard-coded assumptions in agent v4 — and what breaks if you change them

All per-definition constants are supplied via `conv2d.h`, which is auto-generated per
definition:

```cpp
// conv2d.h (auto-generated for this definition)
namespace conv2d_def {
constexpr int Cout            = 128;
constexpr int Kh              = 3;
constexpr int Kw              = 3;
constexpr int Sh              = 1;
constexpr int Sw              = 1;
constexpr int Dh              = 1;
constexpr int Dw              = 1;
constexpr int pad_top         = 1;
constexpr int pad_left        = 1;
constexpr int activation_type = 0;
}
```

v4.cpp encodes additional structural assumptions that rely on these values:

### 4a. `OC_BLOCK=8` assumes `Cout` is divisible by 8

```cpp
// v4.cpp:39
const int OC_BLOCK = 8;
for (int oc_base = 0; oc_base < Cout; oc_base += OC_BLOCK) {
    int oc_end   = oc_base + OC_BLOCK;
    int oc_count = oc_end - oc_base;  // = 8 for Cout=128 (always a full block)
    ...
    if (oc_count == OC_BLOCK) {  // explicit 8-way unroll — fast path
        // 8 svdup, 8 output row pointers, 8 svld1, 8 svmla, 8 svst1
    } else {
        for (int i = 0; i < oc_count; i++) { ... }  // generic fallback
    }
}
```

With Cout=128 and OC\_BLOCK=8: 128/8=16 full blocks, the fast path is always taken.  
With **Cout=100**: 12 full blocks + 1 remainder block of 4 → fast path taken 12/13 of the
time, acceptable.  
With **Cout=5**: only 0 full blocks; all iterations fall back to the generic path — the explicit
8-way unroll is dead code, and performance collapses to v2-level.

### 4b. Valid range formula assumes `Sh=1`, `Sw=1`, `Dh=1`, `Dw=1`

```cpp
// v4.cpp:64–68
int ow_start = pad_left - kw;
if (ow_start < 0) ow_start = 0;
int ow_end = W - kw + pad_left;
if (ow_end > W_out) ow_end = W_out;

// v4.cpp:72
int ih = oh - pad_top + kh;   // assumes Dh=1, Sh=1: ih = oh*Sh - pad_top + kh*Dh
```

These are the stride=1, dilation=1 specialisations of the general formulas:
```
ih = oh * Sh - pad_top + kh * Dh
iw = ow * Sw - pad_left + kw * Dw
```

If **stride=2** (`Sh=2`): `ih = oh*2 - pad_top + kh`, but v4 computes `ih = oh - pad_top + kh`
→ wrong output rows are read → incorrect results.  
If **dilation=2** (`Dh=2`): `ih = oh - pad_top + kh*2`, but v4 computes `kh` not `kh*2`
→ wrong receptive field → incorrect results.

### 4c. Weight indexing assumes `OIHW` layout and specific `Kh`, `Kw`

```cpp
// v4.cpp:58–60
int widx = ((oc_base + i) * C_in + ic) * KhKw + kh * Kw + kw;
wvals[i] = weight[widx];
```

`KhKw = Kh * Kw = 9` is computed from the constexpr at compile time.  If `Kh` or `Kw`
changes (e.g., 5×5 kernel), `KhKw` correctly becomes 25 via the header — the formula
generalises correctly.  The layout assumption (`OIHW`) is the real constraint: if ncnn's
repacked layout were used instead, `widx` would be wrong.

### 4d. `OC_BLOCK=8` is tuned to SVE 256-bit (8 floats per vector)

```cpp
// v4.cpp:29
const int vl = svcntw();  // = 8 on Graviton3 (256-bit SVE)
```

On a hypothetical 128-bit SVE machine (e.g., a mobile Cortex-A76 with SVE), `vl=4`.  With
`OC_BLOCK=8` and `vl=4`, each inner loop iteration performs 8 stores of 4 floats = 32 float
stores, while loading only 1 × 4 floats.  The 8:1 store-to-load ratio would saturate store
bandwidth even faster, reducing IPC further.  Optimal `OC_BLOCK` on 128-bit SVE would likely
be 4.

### Summary of hard-coded constraints

| Assumption | Where in v4.cpp | Breaks if changed to |
|-----------|-----------------|----------------------|
| `Cout % OC_BLOCK == 0` | `oc_count == OC_BLOCK` fast path | Cout not divisible by 8 → falls back to scalar inner loop |
| `Sh=1`, `Sw=1` | `ih = oh - pad_top + kh` | stride > 1 → wrong input rows → wrong output |
| `Dh=1`, `Dw=1` | `ih = oh - pad_top + kh` | dilation > 1 → wrong receptive field → wrong output |
| `OC_BLOCK=8 ≈ vl` | explicit 8-way SVE unroll | 128-bit SVE (vl=4) → 8 stores per 4-float load → bandwidth bottleneck worsens |
| `OIHW` weight layout | `widx = ((oc*Cin+ic)*KhKw + kh*Kw + kw)` | ncnn repacked layout → wrong weights → wrong output |
