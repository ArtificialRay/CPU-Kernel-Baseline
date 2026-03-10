// sgl_activation.cpp — SiLU-and-mul, zero framework dependency
// Compile: g++ -O3 -march=native -mavx512f -mavx512bf16 -fopenmp -std=c++17
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>

// ---------------------------------------------------------------------------
// Internal: process one token row of length d
//   out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
static inline void silu_and_mul_row(
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ gate,
        const bf16_t* __restrict__ up,
        int d) {

    int i = 0;
    for (; i + SGL_VEC_BF16_W <= d; i += SGL_VEC_BF16_W) {
        __m512i vg = sgl_load_bf16(gate + i);
        __m512i vu = sgl_load_bf16(up   + i);

        // Convert gate to f32 (2 x __m512)
        __m512 glo = sgl_cvt_bf16_lo_f32(vg);
        __m512 ghi = sgl_cvt_bf16_hi_f32(vg);
        // Convert up to f32
        __m512 ulo = sgl_cvt_bf16_lo_f32(vu);
        __m512 uhi = sgl_cvt_bf16_hi_f32(vu);

        // Apply SiLU to gate
        __m512 slo = sgl_silu_f32(glo);
        __m512 shi = sgl_silu_f32(ghi);

        // Multiply by up-proj
        __m512 rlo = _mm512_mul_ps(slo, ulo);
        __m512 rhi = _mm512_mul_ps(shi, uhi);

        sgl_store_bf16(out + i, sgl_cvt_2xf32_bf16(rlo, rhi));
    }
    // Scalar tail
    for (; i < d; ++i) {
        float g = bf16_to_f32(gate[i]);
        float u = bf16_to_f32(up[i]);
        float silu_g = g / (1.0f + expf(-g));
        out[i] = f32_to_bf16(silu_g * u);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void sgl_silu_and_mul_bf16(
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ y,
        int n_tokens, int d) {

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tokens; ++t) {
        silu_and_mul_row(
            out + (size_t)t * d,
            x   + (size_t)t * d,
            y   + (size_t)t * d,
            d
        );
    }
}

// In-place: buf is [n_tokens, 2*d], gate = buf[:d], up = buf[d:2d]
// Output overwrites gate half [n_tokens, d]
extern "C" void sgl_silu_and_mul_inplace_bf16(
        bf16_t* __restrict__ buf,
        int n_tokens, int d) {

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tokens; ++t) {
        bf16_t* gate = buf + (size_t)t * 2 * d;
        bf16_t* up   = gate + d;
        silu_and_mul_row(gate, gate, up, d);
    }
}
