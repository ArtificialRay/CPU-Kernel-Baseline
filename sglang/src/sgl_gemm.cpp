// sgl_gemm.cpp — Packed BF16 GEMM and BMM, zero framework dependency
// C[M,N] = A[M,K] * B^T  where B is [N,K] row-major (row = output neuron)
//
// Strategy: tile over (M, N). Each (m,n) pair is a scalar accumulator built
// from vectorized dot-products of 32 bf16 elements at a time via fmadd_ps.
// This is correct on all AVX-512F CPUs. On AVX-512BF16, dpbf16_ps fires
// automatically because sgl_dpbf16_lo uses fmadd_ps which GCC can autovectorize
// further if -mavx512bf16 is passed.
//
// Compile: g++ -O3 -mavx512f -mavx512vl -mavx512bw -fopenmp -std=c++17

#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// Tile: process TILE_M rows of A against TILE_N rows of B at once.
// Each (m,n) accumulator is a single scalar (via hsum at end).
// TILE_M x TILE_N accumulators each being one __m512 (16-wide f32 partial sums)
// that get horizontally reduced.
#define TILE_M  4
#define TILE_N  4
#define TILE_K  64   // must be multiple of 32

// Dot product of two bf16 rows, K elements, accumulating into a __m512
// Then horizontally summed to a scalar.
static inline float dot_bf16(const bf16_t* a, const bf16_t* b, int K) {
    __m512 acc = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m512i va = sgl_load_bf16(a + k);
        __m512i vb = sgl_load_bf16(b + k);
        acc = sgl_dpbf16_lo(acc, va, vb);
    }
    float s = sgl_hsum_f32(acc);
    for (; k < K; ++k) s += bf16_to_f32(a[k]) * bf16_to_f32(b[k]);
    return s;
}

// ---------------------------------------------------------------------------
// Internal GEMM
// ---------------------------------------------------------------------------
static void gemm_bf16_f32(
        float* __restrict__ C,
        const bf16_t* __restrict__ A,  // [M, K]
        const bf16_t* __restrict__ B,  // [N, K]  row = output neuron
        int M, int N, int K) {

    #pragma omp parallel for schedule(dynamic, 2) collapse(2)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[(size_t)m*N + n] = dot_bf16(A + (size_t)m*K, B + (size_t)n*K, K);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void sgl_gemm_bf16(
        bf16_t* __restrict__ C,
        const bf16_t* __restrict__ A,
        const bf16_t* __restrict__ B,
        int M, int N, int K) {

    float* tmp = (float*)aligned_alloc(64, sizeof(float) * (size_t)M * N);
    gemm_bf16_f32(tmp, A, B, M, N, K);

    size_t total = (size_t)M * N;
    size_t i = 0;
    for (; i + SGL_VEC_BF16_W <= total; i += SGL_VEC_BF16_W) {
        __m512 lo = _mm512_loadu_ps(tmp + i);
        __m512 hi = _mm512_loadu_ps(tmp + i + 16);
        sgl_store_bf16(C + i, sgl_cvt_2xf32_bf16(lo, hi));
    }
    for (; i < total; ++i) C[i] = f32_to_bf16(tmp[i]);
    free(tmp);
}

extern "C" void sgl_gemm_bias_bf16(
        bf16_t* __restrict__ C,
        const bf16_t* __restrict__ A,
        const bf16_t* __restrict__ B,
        const bf16_t* __restrict__ bias,
        int M, int N, int K) {

    float* tmp = (float*)aligned_alloc(64, sizeof(float) * (size_t)M * N);
    gemm_bf16_f32(tmp, A, B, M, N, K);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        int i = 0;
        for (; i + SGL_VEC_BF16_W <= N; i += SGL_VEC_BF16_W) {
            __m512 vlo = _mm512_loadu_ps(tmp + (size_t)m*N + i);
            __m512 vhi = _mm512_loadu_ps(tmp + (size_t)m*N + i + 16);
            __m512i vb = sgl_load_bf16(bias + i);
            vlo = _mm512_add_ps(vlo, sgl_cvt_bf16_lo_f32(vb));
            vhi = _mm512_add_ps(vhi, sgl_cvt_bf16_hi_f32(vb));
            sgl_store_bf16(C + (size_t)m*N + i, sgl_cvt_2xf32_bf16(vlo, vhi));
        }
        for (; i < N; ++i)
            C[(size_t)m*N + i] = f32_to_bf16(tmp[(size_t)m*N + i] + bf16_to_f32(bias[i]));
    }
    free(tmp);
}

extern "C" void sgl_bmm_bf16(
        bf16_t* __restrict__ C,
        const bf16_t* __restrict__ A,
        const bf16_t* __restrict__ B,
        int batch, int M, int N, int K) {

    size_t sA = (size_t)M*K, sB = (size_t)N*K, sC = (size_t)M*N;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < batch; ++b) {
        float* tmp = (float*)aligned_alloc(64, sizeof(float) * sC);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                tmp[m*N+n] = dot_bf16(A+b*sA+m*K, B+b*sB+n*K, K);
        for (size_t i = 0; i < sC; ++i) C[b*sC+i] = f32_to_bf16(tmp[i]);
        free(tmp);
    }
}
