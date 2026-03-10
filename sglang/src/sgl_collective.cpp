// sgl_collective.cpp — CPU AllReduce for single-node Tensor Parallelism
// Uses a shared buffer + OpenMP barrier protocol.
// Zero framework dependency — no MPI, no NCCL, no ATen.
//
// Protocol:
//   Each TP rank holds a shard of the full buffer (or the full buffer if
//   using replicated layout). All n_ranks threads must call this function
//   with the SAME pointer and n_elements. A two-phase (reduce + broadcast)
//   approach using OMP atomic + barrier achieves correctness.
//   For high throughput, replace with a ring-allreduce variant.
//
// Compile: g++ -O3 -march=native -mavx512f -fopenmp -std=c++17
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// We use a statically allocated staging area per-reduce call.
// For production, pass a pre-allocated workspace instead.

extern "C" void sgl_allreduce_sum_f32(
        float*  __restrict__ buf,    // [n_elements], shared by all ranks
        int     n_elements,
        int     n_ranks,
        int     rank_id) {

    // Temporary buffer allocated once per call group.
    // In a real system, pass this in as a workspace to avoid malloc.
    static float* scratch = nullptr;
    static int    scratch_n = 0;

    // Thread 0 allocates / resizes scratch
    #pragma omp single
    {
        if (n_elements > scratch_n) {
            free(scratch);
            scratch   = (float*)aligned_alloc(64, sizeof(float) * n_elements);
            scratch_n = n_elements;
            memset(scratch, 0, sizeof(float) * n_elements);
        } else {
            memset(scratch, 0, sizeof(float) * n_elements);
        }
    }
    // Barrier: all ranks finished zeroing
    #pragma omp barrier

    // Phase 1: each rank atomically adds its contribution to scratch
    // (This works because all n_ranks threads share the same scratch ptr)
    int i = 0;
    for (; i + SGL_VEC_F32_W <= n_elements; i += SGL_VEC_F32_W) {
        __m512 v = _mm512_loadu_ps(buf + i);
        // Atomic f32 add per lane — no vectorized atomic in AVX-512,
        // so we loop over lanes. For best perf use a per-rank scratch
        // buffer and do a final reduction in one thread instead.
        for (int lane = 0; lane < SGL_VEC_F32_W; ++lane) {
            float val;
            _mm_store_ss(&val, _mm512_extractf32x4_ps(v, lane / 4));
            // Approximation: extract scalar lane
            float arr[16];
            _mm512_storeu_ps(arr, v);
            #pragma omp atomic
            scratch[i + lane] += arr[lane];
        }
    }
    for (; i < n_elements; ++i) {
        #pragma omp atomic
        scratch[i] += buf[i];
    }

    // Barrier: all ranks done writing partial sums
    #pragma omp barrier

    // Phase 2: broadcast final sum to all ranks' buf
    int j = 0;
    for (; j + SGL_VEC_F32_W <= n_elements; j += SGL_VEC_F32_W) {
        __m512 v = _mm512_loadu_ps(scratch + j);
        _mm512_storeu_ps(buf + j, v);
    }
    for (; j < n_elements; ++j) buf[j] = scratch[j];

    // Final barrier so no rank reads before broadcast is complete
    #pragma omp barrier

    (void)n_ranks;
    (void)rank_id;
}
