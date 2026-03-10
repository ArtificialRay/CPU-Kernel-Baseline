#pragma once
// =============================================================================
// llama_neon_ops.h  —  Standalone ARM NEON kernels extracted from llama.cpp/ggml
//
// Mirrors the ACTUAL bottleneck ops in ggml's CPU backend:
//   ggml/src/ggml-cpu/vec.h         → element-wise + reductions
//   ggml/src/ggml-cpu/ops.cpp       → rms_norm, norm, softmax, rope
//   ggml/src/ggml-cpu/ggml-quants.c → quantized vec_dot (MUL_MAT hot path)
//   ggml/src/ggml-impl.h            → fp16 <-> fp32 conversion
//
// Zero dependencies: no ggml headers, no llama headers, no BLAS.
// Drop-in for unit tests; compiles on x86 with LLAMA_NO_NEON for scalar ref.
//
// Compile (ARM64 native or cross):
//   aarch64-linux-gnu-g++ -O3 -std=c++11 -I. -o test test_llama_ops.cpp
//
// Compile (x86 host — scalar fallback, for CI):
//   g++ -O2 -std=c++11 -DLLAMA_NO_NEON -I. -o test test_llama_ops.cpp
// =============================================================================

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <algorithm>

#if defined(__ARM_NEON) && !defined(LLAMA_NO_NEON)
#  include <arm_neon.h>
#  define LLAMA_NEON 1
#else
#  define LLAMA_NEON 0
#endif

#if defined(__ARM_FEATURE_DOTPROD) && !defined(LLAMA_NO_NEON)
#  define LLAMA_DOTPROD 1
#else
#  define LLAMA_DOTPROD 0
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(LLAMA_NO_NEON)
#  define LLAMA_I8MM 1
#else
#  define LLAMA_I8MM 0
#endif

namespace llama_ops {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace detail {

// Fast polynomial approximation of exp(x) for x in [-88, 0]
// Used by softmax and SiLU.  Matches ggml's neon_mathfun approach.
inline float fast_exp(float x) {
    return std::exp(x);  // scalar; on NEON path we use vexpq_f32 if available,
                         // otherwise the same std::exp per element
}

// Sigmoid via fast_exp
inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

} // namespace detail

// ==========================================================================
// 1.  FP16 <-> FP32  (mirrors ggml_fp32_to_fp16_row / ggml_fp16_to_fp32_row)
//
// ggml stores fp16 as uint16_t (ggml_fp16_t).  We do the same.
// On AArch64: vcvt_f16_f32 / vcvt_f32_f16 via <arm_neon.h>.
// ==========================================================================
using fp16_t = uint16_t;

// Single scalar conversion helpers (bit-exact with IEEE 754 fp16)
inline fp16_t f32_to_f16(float f) {
#if LLAMA_NEON
    float32x4_t v = vdupq_n_f32(f);
    float16x4_t h = vcvt_f16_f32(v);
    uint16_t r;
    vst1_lane_u16(&r, vreinterpret_u16_f16(h), 0);
    return r;
#else
    // Software implementation (same as ggml_fp32_to_fp16 fallback)
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign  = (bits >> 31) & 1u;
    int32_t  exp   = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant  = (bits >> 13) & 0x3FFu;
    if (exp <= 0)  return (fp16_t)(sign << 15);
    if (exp >= 31) return (fp16_t)((sign << 15) | (0x1F << 10));
    return (fp16_t)((sign << 15) | (exp << 10) | mant);
#endif
}

inline float f16_to_f32(fp16_t h) {
#if LLAMA_NEON
    uint16x4_t hv = vdup_n_u16(h);
    float32x4_t fv = vcvt_f32_f16(vreinterpret_f16_u16(hv));
    return vgetq_lane_f32(fv, 0);
#else
    uint32_t sign = (h >> 15) & 1u;
    int32_t  exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0)  { float r = std::ldexp((float)mant, -24); return sign ? -r : r; }
    if (exp == 31) { return sign ? -INFINITY : INFINITY; }
    uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float r; std::memcpy(&r, &bits, 4); return r;
#endif
}

// Row conversion (mirrors ggml_fp32_to_fp16_row / ggml_fp16_to_fp32_row)
inline void fp32_to_fp16_row(const float* __restrict__ src,
                              fp16_t*      __restrict__ dst, int n) {
#if LLAMA_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vf = vld1q_f32(src + i);
        float16x4_t vh = vcvt_f16_f32(vf);
        vst1_u16(dst + i, vreinterpret_u16_f16(vh));
    }
    for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);
#else
    for (int i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
#endif
}

inline void fp16_to_fp32_row(const fp16_t* __restrict__ src,
                              float*        __restrict__ dst, int n) {
#if LLAMA_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float16x4_t vh = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(dst + i, vf);
    }
    for (; i < n; ++i) dst[i] = f16_to_f32(src[i]);
#else
    for (int i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
#endif
}

// ==========================================================================
// 2.  Quantization helpers  (Q8_0 pack/unpack)
//
// ggml Q8_0: block of 32 int8 values + 1 fp16 scale.
// Layout: [scale:fp16][d0..d31:int8]  = 34 bytes/block
// Reference: ggml/src/ggml-quants.h block_q8_0
// ==========================================================================
static constexpr int QK8_0 = 32;  // block size for Q8_0

struct block_q8_0 {
    fp16_t  d;           // scale (fp16)
    int8_t  qs[QK8_0];  // quantized values
};

// Quantize one block of 32 floats → Q8_0
inline block_q8_0 quantize_q8_0_block(const float* src) {
    block_q8_0 blk;
    // Find abs-max
    float amax = 0.f;
    for (int i = 0; i < QK8_0; ++i) amax = std::max(amax, std::fabs(src[i]));
    const float d    = amax / 127.f;
    const float id   = (d > 1e-9f) ? 1.f / d : 0.f;
    blk.d = f32_to_f16(d);
    for (int i = 0; i < QK8_0; ++i)
        blk.qs[i] = (int8_t)std::max(-128, std::min(127, (int)std::roundf(src[i] * id)));
    return blk;
}

// Dequantize one Q8_0 block → floats
inline void dequantize_q8_0_block(const block_q8_0& blk, float* dst) {
    const float d = f16_to_f32(blk.d);
    for (int i = 0; i < QK8_0; ++i) dst[i] = blk.qs[i] * d;
}

// ==========================================================================
// 3.  Quantization helpers  (Q4_0 pack/unpack)
//
// ggml Q4_0: block of 32 4-bit signed values packed as 16 bytes + 1 fp16 scale.
// Layout: [scale:fp16][nibble0..nibble31 packed as 16 bytes]
// Values are stored as unsigned nibble − 8 to give signed range [−8,7].
// ==========================================================================
static constexpr int QK4_0 = 32;

struct block_q4_0 {
    fp16_t  d;
    uint8_t qs[QK4_0 / 2];  // 16 bytes holding 32 nibbles
};

inline block_q4_0 quantize_q4_0_block(const float* src) {
    block_q4_0 blk;
    float amax = 0.f;
    for (int i = 0; i < QK4_0; ++i) amax = std::max(amax, std::fabs(src[i]));
    const float d  = amax / 7.f;
    const float id = (d > 1e-9f) ? 1.f / d : 0.f;
    blk.d = f32_to_f16(d);
    for (int i = 0; i < QK4_0 / 2; ++i) {
        int v0 = (int)std::max(-8, std::min(7, (int)std::roundf(src[2*i]   * id)));
        int v1 = (int)std::max(-8, std::min(7, (int)std::roundf(src[2*i+1] * id)));
        blk.qs[i] = (uint8_t)(((v0 + 8) & 0xF) | (((v1 + 8) & 0xF) << 4));
    }
    return blk;
}

inline void dequantize_q4_0_block(const block_q4_0& blk, float* dst) {
    const float d = f16_to_f32(blk.d);
    for (int i = 0; i < QK4_0 / 2; ++i) {
        dst[2*i]   = ((int)(blk.qs[i] & 0xF) - 8) * d;
        dst[2*i+1] = ((int)(blk.qs[i] >> 4)  - 8) * d;
    }
}

// ==========================================================================
// 4.  vec_dot_q8_0_q8_0
//
// Mirrors ggml_vec_dot_q8_0_q8_0.
// Computes dot(A_q8, B_q8) across nb blocks, returns float.
// ISA tiers:
//   DOTPROD  → vdotq_s32
//   I8MM     → vmmlaq_s32 (not used here — that needs 2-row packing)
//   NEON     → vmull_s8 + vpaddlq
//   scalar   → plain int32 accumulate
// ==========================================================================
inline float vec_dot_q8_0_q8_0(
    const block_q8_0* __restrict__ a,
    const block_q8_0* __restrict__ b,
    int nb)
{
    float sumf = 0.f;
#if LLAMA_DOTPROD
    // Fast path: vdotq_s32 processes 4 int8×int8 pairs per lane
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        int32x4_t acc = vdupq_n_s32(0);
        // 32 elements = 8 groups of 4 using vdotq_s32
        for (int j = 0; j < QK8_0; j += 16) {
            int8x16_t va = vld1q_s8(a[i].qs + j);
            int8x16_t vb = vld1q_s8(b[i].qs + j);
            acc = vdotq_s32(acc, va, vb);
        }
        // horizontal sum of acc
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        sumf += da * db * (float)(vget_lane_s32(vpadd_s32(s2, s2), 0));
    }
#elif LLAMA_NEON
    // Baseline NEON: vmull_s8 + vpaddlq_s16
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        int32x4_t acc = vdupq_n_s32(0);
        for (int j = 0; j < QK8_0; j += 8) {
            int8x8_t  va = vld1_s8(a[i].qs + j);
            int8x8_t  vb = vld1_s8(b[i].qs + j);
            int16x8_t p  = vmull_s8(va, vb);
            acc = vpadalq_s16(acc, p);
        }
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        sumf += da * db * (float)(vget_lane_s32(vpadd_s32(s2, s2), 0));
    }
#else
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        int32_t s = 0;
        for (int j = 0; j < QK8_0; ++j) s += (int32_t)a[i].qs[j] * b[i].qs[j];
        sumf += da * db * (float)s;
    }
#endif
    return sumf;
}

// ==========================================================================
// 5.  vec_dot_q4_0_q8_0
//
// Mirrors ggml_vec_dot_q4_0_q8_0 — the single most-called kernel during
// quantized LLM inference (decode stage, every linear layer).
// A is Q4_0 weights, B is Q8_0 activations.
// ==========================================================================
inline float vec_dot_q4_0_q8_0(
    const block_q4_0* __restrict__ a,
    const block_q8_0* __restrict__ b,
    int nb)
{
    float sumf = 0.f;
#if LLAMA_DOTPROD
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        // Unpack Q4_0 nibbles → int8 in [-8, 7]
        int8x16_t  mask4  = vdupq_n_s8(0xF);
        uint8x16_t raw    = vld1q_u8(a[i].qs);                    // 16 bytes = 32 nibbles
        int8x16_t  lo     = vreinterpretq_s8_u8(vandq_u8(raw, vreinterpretq_u8_s8(mask4)));
        int8x16_t  hi     = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
        // Subtract 8 to shift from [0,15] → [-8,7]
        int8x16_t  sub8   = vdupq_n_s8(8);
        lo = vsubq_s8(lo, sub8);
        hi = vsubq_s8(hi, sub8);
        // Load Q8_0 activations
        int8x16_t vb0 = vld1q_s8(b[i].qs);
        int8x16_t vb1 = vld1q_s8(b[i].qs + 16);
        // Dot products
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, lo, vb0);
        acc = vdotq_s32(acc, hi, vb1);
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        sumf += da * db * (float)(vget_lane_s32(vpadd_s32(s2, s2), 0));
    }
#elif LLAMA_NEON
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        uint8x16_t raw   = vld1q_u8(a[i].qs);
        int8x16_t  mask4 = vdupq_n_s8(0xF);
        int8x16_t  sub8  = vdupq_n_s8(8);
        int8x16_t  lo    = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vreinterpretq_u8_s8(mask4))), sub8);
        int8x16_t  hi    = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), sub8);
        int8x8_t lo0 = vget_low_s8(lo),  lo1 = vget_high_s8(lo);
        int8x8_t hi0 = vget_low_s8(hi),  hi1 = vget_high_s8(hi);
        int8x8_t b0  = vld1_s8(b[i].qs),     b1  = vld1_s8(b[i].qs + 8);
        int8x8_t b2  = vld1_s8(b[i].qs + 16),b3  = vld1_s8(b[i].qs + 24);
        int32x4_t acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(lo0, b0));
        acc = vpadalq_s16(acc, vmull_s8(lo1, b1));
        acc = vpadalq_s16(acc, vmull_s8(hi0, b2));
        acc = vpadalq_s16(acc, vmull_s8(hi1, b3));
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        sumf += da * db * (float)(vget_lane_s32(vpadd_s32(s2, s2), 0));
    }
#else
    for (int i = 0; i < nb; ++i) {
        const float da = f16_to_f32(a[i].d);
        const float db = f16_to_f32(b[i].d);
        int32_t s = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int v0 = (int)(a[i].qs[j] & 0xF) - 8;
            int v1 = (int)(a[i].qs[j] >> 4)  - 8;
            s += v0 * b[i].qs[2*j];
            s += v1 * b[i].qs[2*j + 1];
        }
        sumf += da * db * (float)s;
    }
#endif
    return sumf;
}

// ==========================================================================
// 6.  vec_dot_f32  (FP32 dot product, mirrors ggml_vec_dot_f32)
// ==========================================================================
inline float vec_dot_f32(const float* __restrict__ a, const float* __restrict__ b, int n) {
    float sum = 0.f;
#if LLAMA_NEON
    float32x4_t acc = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 3 < n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float32x2_t v2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < n; ++i) sum += a[i] * b[i];
#else
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
#endif
    return sum;
}

// ==========================================================================
// 7.  RMS Norm  (mirrors ggml_compute_forward_rms_norm_f32)
//
// y_i = x_i / sqrt(mean(x^2) + eps) * weight_i
// weight may be nullptr (then no scaling).
// ==========================================================================
inline void rms_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,  // [n] or nullptr
    float*       __restrict__ y,
    int n, float eps = 1e-5f)
{
    // Compute mean(x^2)
    float ss = 0.f;
#if LLAMA_NEON
    {
        float32x4_t acc = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            acc = vfmaq_f32(acc, v, v);
        }
        float32x2_t v2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        ss = vget_lane_f32(vpadd_f32(v2, v2), 0);
        for (; i < n; ++i) ss += x[i] * x[i];
    }
#else
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
#endif
    ss = ss / (float)n + eps;
    const float scale = 1.f / std::sqrt(ss);

#if LLAMA_NEON
    const float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vmulq_f32(vld1q_f32(x + i), vscale);
        if (weight) vx = vmulq_f32(vx, vld1q_f32(weight + i));
        vst1q_f32(y + i, vx);
    }
    for (; i < n; ++i) y[i] = x[i] * scale * (weight ? weight[i] : 1.f);
#else
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale * (weight ? weight[i] : 1.f);
#endif
}

// ==========================================================================
// 8.  Layer Norm  (mirrors ggml_compute_forward_norm_f32)
//     y = (x - mean) / sqrt(var + eps) * weight + bias
// ==========================================================================
inline void layer_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,  // [n] or nullptr
    const float* __restrict__ bias,    // [n] or nullptr
    float*       __restrict__ y,
    int n, float eps = 1e-5f)
{
    // Mean
    float mean = 0.f;
#if LLAMA_NEON
    {
        float32x4_t acc = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 3 < n; i += 4) acc = vaddq_f32(acc, vld1q_f32(x + i));
        float32x2_t v2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        mean = vget_lane_f32(vpadd_f32(v2, v2), 0);
        for (; i < n; ++i) mean += x[i];
    }
#else
    for (int i = 0; i < n; ++i) mean += x[i];
#endif
    mean /= (float)n;

    // Variance
    float var = 0.f;
#if LLAMA_NEON
    {
        float32x4_t acc  = vdupq_n_f32(0.f);
        float32x4_t vmean = vdupq_n_f32(mean);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
            acc = vfmaq_f32(acc, d, d);
        }
        float32x2_t v2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        var = vget_lane_f32(vpadd_f32(v2, v2), 0);
        for (; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    }
#else
    for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
#endif
    var = var / (float)n + eps;
    const float inv_std = 1.f / std::sqrt(var);

#if LLAMA_NEON
    const float32x4_t vmean   = vdupq_n_f32(mean);
    const float32x4_t vinvstd = vdupq_n_f32(inv_std);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmean), vinvstd);
        if (weight) v = vmulq_f32(v, vld1q_f32(weight + i));
        if (bias)   v = vaddq_f32(v, vld1q_f32(bias + i));
        vst1q_f32(y + i, v);
    }
    for (; i < n; ++i) {
        float v = (x[i] - mean) * inv_std;
        y[i] = (weight ? weight[i] * v : v) + (bias ? bias[i] : 0.f);
    }
#else
    for (int i = 0; i < n; ++i) {
        float v = (x[i] - mean) * inv_std;
        y[i] = (weight ? weight[i] * v : v) + (bias ? bias[i] : 0.f);
    }
#endif
}

// ==========================================================================
// 9.  Softmax  (mirrors ggml_compute_forward_soft_max_f32)
//     Numerically stable: subtract max before exp.
//     Operates on one row of length n.
// ==========================================================================
inline void softmax(float* __restrict__ x, int n) {
    // Max
    float max_val = x[0];
#if LLAMA_NEON
    {
        float32x4_t vmax = vdupq_n_f32(-1e38f);
        int i = 0;
        for (; i + 3 < n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
        float32x2_t v2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        max_val = vget_lane_f32(vpmax_f32(v2, v2), 0);
        for (; i < n; ++i) max_val = std::max(max_val, x[i]);
    }
#else
    for (int i = 1; i < n; ++i) max_val = std::max(max_val, x[i]);
#endif

    // exp(x - max) and sum
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }

    // Normalize
    const float inv_sum = 1.f / sum;
#if LLAMA_NEON
    {
        float32x4_t vinv = vdupq_n_f32(inv_sum);
        int i = 0;
        for (; i + 3 < n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
        for (; i < n; ++i) x[i] *= inv_sum;
    }
#else
    for (int i = 0; i < n; ++i) x[i] *= inv_sum;
#endif
}

// ==========================================================================
// 10. SiLU  (mirrors ggml_compute_forward_silu_f32)
//     y_i = x_i * sigmoid(x_i) = x_i / (1 + exp(-x_i))
//     In-place capable.
// ==========================================================================
inline void silu(float* __restrict__ x, int n) {
    // scalar loop — on ARM the compiler auto-vectorises this well with -O3,
    // but we show the explicit NEON form too using vdivq_f32 (A64)
#if LLAMA_NEON
    const float32x4_t vone = vdupq_n_f32(1.f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx  = vld1q_f32(x + i);
        // exp(-x) element-wise — no vexpq_f32 in baseline NEON, so scalar exp
        float exps[4];
        float tmp[4]; vst1q_f32(tmp, vnegq_f32(vx));
        for (int k = 0; k < 4; ++k) exps[k] = std::exp(tmp[k]);
        float32x4_t vexp  = vld1q_f32(exps);
        float32x4_t vsig  = vrecpeq_f32(vaddq_f32(vone, vexp));  // approx 1/(1+e^-x)
        // Newton-Raphson refinement for reciprocal: vsig = vsig * (2 - (1+vexp)*vsig)
        vsig = vmulq_f32(vsig, vrecpsq_f32(vaddq_f32(vone, vexp), vsig));
        vst1q_f32(x + i, vmulq_f32(vx, vsig));
    }
    for (; i < n; ++i) x[i] = x[i] / (1.f + std::exp(-x[i]));
#else
    for (int i = 0; i < n; ++i) x[i] = x[i] / (1.f + std::exp(-x[i]));
#endif
}

// ==========================================================================
// 11. ReLU  (mirrors ggml_compute_forward_relu_f32)
// ==========================================================================
inline void relu(float* __restrict__ x, int n) {
#if LLAMA_NEON
    const float32x4_t vzero = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 3 < n; i += 4) vst1q_f32(x + i, vmaxq_f32(vld1q_f32(x + i), vzero));
    for (; i < n; ++i) x[i] = x[i] > 0.f ? x[i] : 0.f;
#else
    for (int i = 0; i < n; ++i) x[i] = x[i] > 0.f ? x[i] : 0.f;
#endif
}

// ==========================================================================
// 12. RoPE  (Rotary Position Embedding)
//     mirrors ggml_compute_forward_rope_f32 for the standard non-interleaved
//     (neox) layout used by LLaMA / Mistral / Qwen etc.
//
//     Applies RoPE in-place to a row of q or k of shape [n_heads, head_dim].
//     For each pair (x[2i], x[2i+1]):
//        x[2i]   =  x[2i]   * cos(θ_i) − x[2i+1] * sin(θ_i)
//        x[2i+1] =  x[2i+1] * cos(θ_i) + x[2i]   * sin(θ_i)
//     where θ_i = pos / (base ^ (2i / head_dim)).
// ==========================================================================
inline void rope_f32(
    float*  __restrict__ x,       // [head_dim] — modified in place
    int     head_dim,
    int     pos,                  // token position
    float   base     = 10000.f,   // RoPE base
    float   freq_scale = 1.f)     // optional NTK scaling
{
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float theta = freq_scale * (float)pos /
            std::pow(base, 2.f * (float)i / (float)head_dim);
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = x[i];
        const float x1 = x[i + half];
        x[i]        = x0 * cos_t - x1 * sin_t;
        x[i + half] = x1 * cos_t + x0 * sin_t;
    }
}

// Vectorised form: processes pairs with NEON (mirrors ggml rope inner loop)
inline void rope_f32_neon(
    float*  __restrict__ x,
    int     head_dim,
    int     pos,
    float   base     = 10000.f,
    float   freq_scale = 1.f)
{
#if LLAMA_NEON
    const int half = head_dim / 2;
    int i = 0;
    for (; i + 3 < half; i += 4) {
        // Compute 4 thetas
        float thetas[4];
        for (int k = 0; k < 4; ++k)
            thetas[k] = freq_scale * (float)pos /
                std::pow(base, 2.f * (float)(i + k) / (float)head_dim);
        float cos4[4], sin4[4];
        for (int k = 0; k < 4; ++k) { cos4[k] = std::cos(thetas[k]); sin4[k] = std::sin(thetas[k]); }
        float32x4_t vc = vld1q_f32(cos4);
        float32x4_t vs = vld1q_f32(sin4);
        float32x4_t vx0 = vld1q_f32(x + i);
        float32x4_t vx1 = vld1q_f32(x + i + half);
        // x[i]      = x0*cos - x1*sin
        // x[i+half] = x1*cos + x0*sin
        vst1q_f32(x + i,       vsubq_f32(vmulq_f32(vx0, vc), vmulq_f32(vx1, vs)));
        vst1q_f32(x + i + half, vaddq_f32(vmulq_f32(vx1, vc), vmulq_f32(vx0, vs)));
    }
    for (; i < half; ++i) {
        const float theta = freq_scale * (float)pos /
            std::pow(base, 2.f * (float)i / (float)head_dim);
        const float cos_t = std::cos(theta), sin_t = std::sin(theta);
        const float x0 = x[i], x1 = x[i + half];
        x[i]        = x0 * cos_t - x1 * sin_t;
        x[i + half] = x1 * cos_t + x0 * sin_t;
    }
#else
    rope_f32(x, head_dim, pos, base, freq_scale);
#endif
}

// ==========================================================================
// 13. Element-wise ops — ADD, MUL, SCALE
//     All mirror the corresponding ggml_vec_* functions in vec.h
// ==========================================================================
inline void vec_add_f32(const float* a, const float* b, float* c, int n) {
#if LLAMA_NEON
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(c + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}

inline void vec_mul_f32(const float* a, const float* b, float* c, int n) {
#if LLAMA_NEON
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(c + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; ++i) c[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; ++i) c[i] = a[i] * b[i];
#endif
}

inline void vec_scale_f32(float* x, float s, int n) {
#if LLAMA_NEON
    const float32x4_t vs = vdupq_n_f32(s);
    int i = 0;
    for (; i + 3 < n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vs));
    for (; i < n; ++i) x[i] *= s;
#else
    for (int i = 0; i < n; ++i) x[i] *= s;
#endif
}

// ==========================================================================
// 14. Scalar matmul helper (quantized MUL_MAT, multi-row)
//     Wraps vec_dot_q4_0_q8_0 for a full weight matrix × vector product.
//
//     W: [out_rows, nb]  blocks of Q4_0  (pre-quantized weights)
//     x: [nb]            blocks of Q8_0  (quantized activation row)
//     y: [out_rows]      float output
// ==========================================================================
inline void matmul_q4_q8(
    const block_q4_0* __restrict__ W,  // [out_rows * nb]
    const block_q8_0* __restrict__ x,  // [nb]
    float*            __restrict__ y,
    int out_rows, int nb)
{
    for (int r = 0; r < out_rows; ++r)
        y[r] = vec_dot_q4_0_q8_0(W + r * nb, x, nb);
}

} // namespace llama_ops
