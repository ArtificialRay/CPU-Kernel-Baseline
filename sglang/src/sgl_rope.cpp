// sgl_rope.cpp — Rotary Positional Embedding, zero framework dependency
// Neox layout: rotate first head_dim/2 elements of each head.
// Compile: g++ -O3 -march=native -mavx512f -mavx512bf16 -fopenmp -std=c++17
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>

// ---------------------------------------------------------------------------
// Internal: apply RoPE to one head (head_dim elements)
//   x_out[i]           = x[i]*cos[i] - x[i+half]*sin[i]
//   x_out[i+half]      = x[i]*sin[i] + x[i+half]*cos[i]
//   for i in [0, half)
// ---------------------------------------------------------------------------
static inline void rope_head(
        bf16_t*       __restrict__ out,
        const bf16_t* __restrict__ x,
        const float*  __restrict__ cos_row,   // [head_dim/2]
        const float*  __restrict__ sin_row,   // [head_dim/2]
        int head_dim) {

    int half = head_dim / 2;
    int i = 0;

    // Process SGL_VEC_F32_W = 16 f32 pairs per iteration
    for (; i + SGL_VEC_F32_W <= half; i += SGL_VEC_F32_W) {
        // Load 16 bf16 from first half and second half of head
        __m256i lo_raw = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i hi_raw = _mm256_loadu_si256((const __m256i*)(x + i + half));

        // Convert bf16 → f32
        __m512 xlo = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(lo_raw), 16));
        __m512 xhi = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi_raw), 16));

        __m512 vc  = _mm512_loadu_ps(cos_row + i);
        __m512 vs  = _mm512_loadu_ps(sin_row + i);

        // Rotated pairs
        __m512 r0 = _mm512_fmsub_ps(xlo, vc, _mm512_mul_ps(xhi, vs));  // x*cos - x2*sin
        __m512 r1 = _mm512_fmadd_ps(xlo, vs, _mm512_mul_ps(xhi, vc));  // x*sin + x2*cos

        // Convert back to bf16 and store
        __m256i out0 = sgl_cvt_f32_to_bf16_256(r0);
        __m256i out1 = sgl_cvt_f32_to_bf16_256(r1);
        _mm256_storeu_si256((__m256i*)(out + i),        out0);
        _mm256_storeu_si256((__m256i*)(out + i + half), out1);
    }
    // Scalar tail
    for (; i < half; ++i) {
        float xi  = bf16_to_f32(x[i]);
        float xi2 = bf16_to_f32(x[i + half]);
        float c   = cos_row[i];
        float s   = sin_row[i];
        out[i]        = f32_to_bf16(xi * c - xi2 * s);
        out[i + half] = f32_to_bf16(xi * s + xi2 * c);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void sgl_rope_neox_bf16(
        bf16_t*        __restrict__ q_out,
        bf16_t*        __restrict__ k_out,
        const bf16_t*  __restrict__ q,
        const bf16_t*  __restrict__ k,
        const float*   __restrict__ cos_table,
        const float*   __restrict__ sin_table,
        const int32_t* __restrict__ positions,
        int n_tokens,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int max_pos) {

    (void)max_pos;  // bounds checking omitted for hot path
    int half = head_dim / 2;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tokens; ++t) {
        int pos = positions[t];
        const float* cos_row = cos_table + (size_t)pos * half;
        const float* sin_row = sin_table + (size_t)pos * half;

        // Rotate query heads
        for (int h = 0; h < n_q_heads; ++h) {
            const bf16_t* qin  = q     + (size_t)t * n_q_heads * head_dim + h * head_dim;
            bf16_t*       qout = q_out + (size_t)t * n_q_heads * head_dim + h * head_dim;
            rope_head(qout, qin, cos_row, sin_row, head_dim);
        }

        // Rotate key heads
        for (int h = 0; h < n_kv_heads; ++h) {
            const bf16_t* kin  = k     + (size_t)t * n_kv_heads * head_dim + h * head_dim;
            bf16_t*       kout = k_out + (size_t)t * n_kv_heads * head_dim + h * head_dim;
            rope_head(kout, kin, cos_row, sin_row, head_dim);
        }
    }
}
