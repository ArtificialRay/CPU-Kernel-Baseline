// sgl_moe.cpp — Fused MoE forward, zero framework dependency
//
// Replaces:  sgl-kernel::fused_experts_cpu
//            sgl-kernel::shared_expert_cpu
//
// Algorithm (per token t, per selected expert e with weight w):
//   h1[2N]  = w1[e] @ x[t]          (GEMM, BF16)
//   gate[N] = silu(h1[:N]) * h1[N:] (SwiGLU)
//   h2[K]   = w2[e] @ gate          (GEMM, BF16)
//   out[t] += w * h2                (weighted accumulate in f32, then store bf16)
//
// Parallelism strategy:
//   Outer parallel: tokens chunked across OMP threads.
//   Per token: sequential over topk experts (small topk, usually 2-8).
//   Expert GEMM uses the inner sgl_gemm helper (single-threaded variant).
//
// Compile: g++ -O3 -march=native -mavx512f -mavx512bf16 -fopenmp -std=c++17
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Internal: single-token single-expert GEMM  C[M=1, N] = A[1,K] * W[N,K]^T
// (M=1 specialisation — avoids full tiling overhead for decode-phase)
// ---------------------------------------------------------------------------
static void matvec_bf16_f32(
        float*        __restrict__ out_f32,  // [N]
        const bf16_t* __restrict__ x,        // [K]
        const bf16_t* __restrict__ W,        // [N, K] row-major
        int N, int K) {

    for (int n = 0; n < N; ++n) {
        const bf16_t* Wrow = W + (size_t)n * K;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 32 <= K; k += 32) {
            __m512i va = sgl_load_bf16(x    + k);
            __m512i vb = sgl_load_bf16(Wrow + k);
            acc = sgl_dpbf16_lo(acc, va, vb);
        }
        float sum = sgl_hsum_f32(acc);
        for (; k < K; ++k)
            sum += bf16_to_f32(x[k]) * bf16_to_f32(Wrow[k]);
        out_f32[n] = sum;
    }
}

// ---------------------------------------------------------------------------
// Internal: process one token against one expert.
//   Writes result into out_f32[K], scaled by expert_weight.
//   buf must be at least max(2N, N, K) floats.
// ---------------------------------------------------------------------------
static void expert_forward_token(
        float*        __restrict__ out_f32,    // [K] accumulate target (f32)
        const bf16_t* __restrict__ x,          // [K]
        const bf16_t* __restrict__ w1,         // [2N, K]
        const bf16_t* __restrict__ w2,         // [K, N]
        float                      ew,         // expert weight scalar
        float*        __restrict__ buf,        // scratch [2N + N + K]
        int K, int N) {

    float* h1   = buf;           // [2N] — gate+up activations
    float* gate = buf + 2 * N;   // [N]  — after SwiGLU

    // 1. h1 = w1 @ x  (matrix [2N x K] times vector [K])
    matvec_bf16_f32(h1, x, w1, 2 * N, K);

    // 2. gate[i] = silu(h1[i]) * h1[i + N]  (vectorized f32)
    {
        int i = 0;
        for (; i + SGL_VEC_F32_W <= N; i += SGL_VEC_F32_W) {
            __m512 vg = _mm512_loadu_ps(h1 + i);
            __m512 vu = _mm512_loadu_ps(h1 + i + N);
            __m512 vs = sgl_silu_f32(vg);
            __m512 vr = _mm512_mul_ps(vs, vu);
            _mm512_storeu_ps(gate + i, vr);
        }
        for (; i < N; ++i) {
            float g = h1[i];
            float u = h1[i + N];
            gate[i] = (g / (1.0f + expf(-g))) * u;
        }
    }

    // 3. h2 = w2 @ gate  (matrix [K x N] times vector [N])
    //    We need to convert gate (f32) back to bf16 for the GEMM intrinsic.
    //    Alternatively: inline f32 matvec here.
    //    We use an f32-input matvec to avoid round-trip conversion.
    {
        for (int k = 0; k < K; ++k) {
            const bf16_t* Wrow = w2 + (size_t)k * N;
            __m512 acc = _mm512_setzero_ps();
            int n = 0;
            // dot product gate_f32 · w2_bf16
            for (; n + SGL_VEC_F32_W <= N; n += SGL_VEC_F32_W) {
                __m512 vg = _mm512_loadu_ps(gate + n);
                // Convert 16 bf16 to f32
                __m256i raw = _mm256_loadu_si256((const __m256i*)(Wrow + n));
                __m512 vw = _mm512_castsi512_ps(
                    _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                acc = _mm512_fmadd_ps(vg, vw, acc);
            }
            float sum = sgl_hsum_f32(acc);
            for (; n < N; ++n)
                sum += gate[n] * bf16_to_f32(Wrow[n]);
            // Weighted accumulate into output
            out_f32[k] += ew * sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void sgl_fused_experts_bf16(
        bf16_t*        __restrict__ out,
        const bf16_t*  __restrict__ hidden_states,
        const bf16_t*  __restrict__ w1,
        const bf16_t*  __restrict__ w2,
        const float*   __restrict__ topk_weights,
        const int32_t* __restrict__ topk_ids,
        int n_tokens, int n_experts, int K, int N, int topk) {

    (void)n_experts;  // bounds-check omitted in hot path

    size_t w1_expert_stride = (size_t)2 * N * K;   // bytes in bf16
    size_t w2_expert_stride = (size_t)K * N;

    // Scratch per thread: 2N + N f32 = 3N
    int buf_size = 3 * N;

    #pragma omp parallel
    {
        float* buf     = (float*)aligned_alloc(64, sizeof(float) * buf_size);
        float* out_f32 = (float*)aligned_alloc(64, sizeof(float) * K);

        #pragma omp for schedule(dynamic, 4)
        for (int t = 0; t < n_tokens; ++t) {
            const bf16_t* x = hidden_states + (size_t)t * K;

            // Zero f32 accumulator for this token
            memset(out_f32, 0, sizeof(float) * K);

            for (int k = 0; k < topk; ++k) {
                int   eid = topk_ids[t * topk + k];
                float ew  = topk_weights[t * topk + k];

                const bf16_t* ew1 = w1 + eid * w1_expert_stride;
                const bf16_t* ew2 = w2 + eid * w2_expert_stride;

                expert_forward_token(out_f32, x, ew1, ew2, ew, buf, K, N);
            }

            // Store f32 → bf16
            bf16_t* out_row = out + (size_t)t * K;
            int i = 0;
            for (; i + SGL_VEC_BF16_W <= K; i += SGL_VEC_BF16_W) {
                __m512 lo = _mm512_loadu_ps(out_f32 + i);
                __m512 hi = _mm512_loadu_ps(out_f32 + i + 16);
                sgl_store_bf16(out_row + i, sgl_cvt_2xf32_bf16(lo, hi));
            }
            for (; i < K; ++i) out_row[i] = f32_to_bf16(out_f32[i]);
        }

        free(buf);
        free(out_f32);
    }
}

extern "C" void sgl_shared_expert_bf16(
        bf16_t*        __restrict__ out,
        const bf16_t*  __restrict__ hidden_states,
        const bf16_t*  __restrict__ w1,            // [2N, K]
        const bf16_t*  __restrict__ w2,            // [K, N]
        float          scale,
        int n_tokens, int K, int N) {

    int buf_size = 3 * N;

    #pragma omp parallel
    {
        float* buf     = (float*)aligned_alloc(64, sizeof(float) * buf_size);
        float* out_f32 = (float*)aligned_alloc(64, sizeof(float) * K);

        #pragma omp for schedule(static)
        for (int t = 0; t < n_tokens; ++t) {
            const bf16_t* x = hidden_states + (size_t)t * K;

            memset(out_f32, 0, sizeof(float) * K);
            expert_forward_token(out_f32, x, w1, w2, scale, buf, K, N);

            bf16_t* out_row = out + (size_t)t * K;
            int i = 0;
            for (; i + SGL_VEC_BF16_W <= K; i += SGL_VEC_BF16_W) {
                __m512 lo = _mm512_loadu_ps(out_f32 + i);
                __m512 hi = _mm512_loadu_ps(out_f32 + i + 16);
                sgl_store_bf16(out_row + i, sgl_cvt_2xf32_bf16(lo, hi));
            }
            for (; i < K; ++i) out_row[i] = f32_to_bf16(out_f32[i]);
        }

        free(buf);
        free(out_f32);
    }
}
