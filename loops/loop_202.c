/*----------------------------------------------------------------------------
#
#   Loop 202: FP32 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of fp32 MOPA (or MLA) instructions.
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "helpers.h"
#include "loops.h"

/*
  Data format -
    A: column-major
    B: row-major
    C: row-major
  Constraints -
    M: multiple of SVLh
    N: multiple of SVLh
    K: even
*/

struct loop_202_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float *restrict a;
  float *restrict b;
  float *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_202(struct loop_202_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(__ARM_FEATURE_SME2)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(__ARM_NEON)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif



void matmul_fp32(uint64_t, uint64_t, uint64_t, float *restrict, float *restrict,
                 float *restrict) LOOP_ATTR;

#if !defined(HAVE_CANDIDATE)
static void inner_loop_202(struct loop_202_data *data)
LOOP_ATTR
{
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) || defined(__ARM_FEATURE_SME) || \
defined(__ARM_FEATURE_SVE) || defined(__ARM_NEON)
  matmul_fp32(data->m, data->n, data->k, data->a, data->b, data->c);
#else
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
#endif
}
#endif /* !HAVE_CANDIDATE */


// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 256KiB equates to original problem size (M=128, K=256, N=128)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 256
#endif

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(float))

LOOP_DECL(202, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLh
  uint64_t K = 0;  // even

  // For this loop, K should remain as 2*M, M and N must be equal
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m;
    uint64_t k = m * 2;
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_202_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_float(data.a, M * K);
  fill_float(data.b, K * N);

  inner_loops_202(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                                   \
  {                                                                   \
    float d = 0.0f;                                                   \
    for (int k = 0; k < K; k++)                                       \
      d += data.a[k * M + (x)] * data.b[k * N + (y)];                 \
    checksum += (int)!check_float(d, data.c[(x) * N + (y)], 1e-3f);   \
  }
#ifdef FULL_CHECK
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(202, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
