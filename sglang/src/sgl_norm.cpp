// sgl_norm.cpp — RMSNorm kernels, zero framework dependency
// Compile: g++ -O3 -march=native -mavx512f -mavx512bf16 -fopenmp -std=c++17
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Internal: compute rms scale factor for one row of hidden_dim elements
// ---------------------------------------------------------------------------
static inline float compute_rms_scale(const bf16_t* __restrict__ row,
                                       int hidden_dim, float eps) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    // Process 32 bf16 per iteration (two __m512 f32 worth)
    for (; i + SGL_VEC_BF16_W <= hidden_dim; i += SGL_VEC_BF16_W) {
        __m512i v = sgl_load_bf16(row + i);
        acc = sgl_sq_acc_bf16(acc, v);
    }
    float sq_sum = sgl_hsum_f32(acc);
    // Scalar tail
    for (; i < hidden_dim; ++i) {
        float v = bf16_to_f32(row[i]);
        sq_sum += v * v;
    }
    float rms = sqrtf(sq_sum / (float)hidden_dim + eps);
    return 1.0f / rms;
}

// ---------------------------------------------------------------------------
// Internal: apply rms_scale * weight[j] to each element, store bf16
// ---------------------------------------------------------------------------
static inline void apply_rms_norm_row(
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ weight,
        int hidden_dim, float rms_scale) {

    __m512 vscale = _mm512_set1_ps(rms_scale);
    int i = 0;
    for (; i + SGL_VEC_BF16_W <= hidden_dim; i += SGL_VEC_BF16_W) {
        __m512i vx = sgl_load_bf16(x + i);
        __m512i vw = sgl_load_bf16(weight + i);

        __m512 xlo = sgl_cvt_bf16_lo_f32(vx);
        __m512 xhi = sgl_cvt_bf16_hi_f32(vx);
        __m512 wlo = sgl_cvt_bf16_lo_f32(vw);
        __m512 whi = sgl_cvt_bf16_hi_f32(vw);

        __m512 rlo = _mm512_mul_ps(_mm512_mul_ps(xlo, vscale), wlo);
        __m512 rhi = _mm512_mul_ps(_mm512_mul_ps(xhi, vscale), whi);

        sgl_store_bf16(out + i, sgl_cvt_2xf32_bf16(rlo, rhi));
    }
    // Scalar tail
    for (; i < hidden_dim; ++i) {
        float v = bf16_to_f32(x[i]) * rms_scale * bf16_to_f32(weight[i]);
        out[i] = f32_to_bf16(v);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void sgl_rms_norm_bf16(
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ weight,
        int n_tokens, int hidden_dim, float eps) {

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tokens; ++t) {
        const bf16_t* row_x   = x   + (size_t)t * hidden_dim;
        bf16_t*       row_out = out + (size_t)t * hidden_dim;

        float scale = compute_rms_scale(row_x, hidden_dim, eps);
        apply_rms_norm_row(row_out, row_x, weight, hidden_dim, scale);
    }
}

extern "C" void sgl_add_rms_norm_bf16(
        bf16_t*       __restrict__ residual,
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ weight,
        int n_tokens, int hidden_dim, float eps) {

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tokens; ++t) {
        bf16_t*       res_row = residual + (size_t)t * hidden_dim;
        bf16_t*       out_row = out      + (size_t)t * hidden_dim;
        const bf16_t* x_row   = x        + (size_t)t * hidden_dim;

        // Step 1: residual += x  (vectorized, in-place)
        int i = 0;
        for (; i + SGL_VEC_BF16_W <= hidden_dim; i += SGL_VEC_BF16_W) {
            __m512i vr = sgl_load_bf16(res_row + i);
            __m512i vx = sgl_load_bf16(x_row + i);

            __m512 rlo = sgl_cvt_bf16_lo_f32(vr);
            __m512 rhi = sgl_cvt_bf16_hi_f32(vr);
            __m512 xlo = sgl_cvt_bf16_lo_f32(vx);
            __m512 xhi = sgl_cvt_bf16_hi_f32(vx);

            __m512 slo = _mm512_add_ps(rlo, xlo);
            __m512 shi = _mm512_add_ps(rhi, xhi);

            __m512i packed = sgl_cvt_2xf32_bf16(slo, shi);
            sgl_store_bf16(res_row + i, packed);
        }
        for (; i < hidden_dim; ++i) {
            float v = bf16_to_f32(res_row[i]) + bf16_to_f32(x_row[i]);
            res_row[i] = f32_to_bf16(v);
        }

        // Step 2: rms_norm(residual) → out
        float scale = compute_rms_scale(res_row, hidden_dim, eps);
        apply_rms_norm_row(out_row, res_row, weight, hidden_dim, scale);
    }
}
