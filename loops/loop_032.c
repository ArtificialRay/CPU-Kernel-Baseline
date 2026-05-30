/*----------------------------------------------------------------------------
#
#   Loop 032: FP64 banded linear equations
#
#   Purpose:
#     Use of strided gather and INC instructions.
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


struct loop_032_data {
  double *restrict a;
  double *restrict b;
  int n;
  double res;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_032(struct loop_032_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_032(struct loop_032_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int n = input->n;

  double res = 0.0;
  int lw = 0;
  for (int j = 4; j < n; j = j + 5) {
    res -= a[lw] * b[j];
    lw++;
  }
  input->res = res;
}
#elif defined(HAVE_SVE_INTRINSICS)
static void inner_loop_032(struct loop_032_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int n = input->n;

  svfloat64_t c = svdup_f64(0.0);
  svuint64_t idx = svindex_u64(4, 5);

  int m = (n - 4) / 5;

  svbool_t p;
  FOR_LOOP_64(int, i, 0, m, p) {
    svfloat64_t a_vec = svld1(p, a + i);
    svfloat64_t b_vec = svld1_gather_index(p, b + 5 * i, idx);
    c = svmls_m(p, c, a_vec, b_vec);
  }

  input->res = svaddv(svptrue_b64(), c);
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_032(struct loop_032_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int n = input->n;

  double res = 0.0;
  int64_t lw = 0;
  int64_t lmt = (n - 4) / 5;

  asm volatile(
      "       mov     z0.d, #0                              \n"
      "       index   z4.d, #4, #5                          \n"
      "       whilele p0.d, xzr, %[lmt]                     \n"
      "1:     ld1d    {z2.d}, p0/z, [%[a], %[lw], lsl #3]   \n"
      "       incd    %[lw]                                 \n"
      "       ld1d    {z1.d}, p0/z, [%[b], z4.d, lsl #3]    \n"
      "       incb    %[b], all, mul #5                     \n"
      "       fmls    z0.d, p0/m, z1.d, z2.d                \n"
      "       whilele p0.d, %[lw], %[lmt]                   \n"
      "       b.any   1b                                    \n"
      "       ptrue   p0.d                                  \n"
      "       faddv   %d[res], p0, z0.d                     \n"
      // output operands, source operands, and clobber list
      : [a] "+&r"(a), [b] "+&r"(b), [res] "+&w"(res), [lw] "+&r"(lw)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "v4", "p0", "cc", "memory");

  input->res = res;
}
#elif defined(__ARM_NEON)
static void inner_loop_032(struct loop_032_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int n = input->n;

  double res = 0.0;
  double *lmt0 = b + n - 10;
  double *lmt = b + n;
  double *pb = b + 4;

  asm volatile(
      "       add     %[lmt], %[lmt], #0x5      \n"
      "       movi    v1.16b, #0                \n"
      "1:     ldr     d0, [%[pb]]               \n"
      "       ldr     d3, [%[pb], #0x28]        \n"
      "       add     %[pb], %[pb], #0x50       \n"
      "       ldr     q2, [%[a]], #16           \n"
      "       mov     v0.d[1], v3.d[0]          \n"
      "       fmls    v1.2d, v2.2d, v0.2d       \n"
      "       cmp     %[pb], %[lmt0]            \n"
      "       b.lt    1b                        \n"
      "       faddp   v1.2d, v1.2d, v1.2d       \n"
      "       fmov    %d[res], d1               \n"
      "2:     ldr     d1, [%[pb]]               \n"
      "       add     %[pb], %[pb], #0x28       \n"
      "       ldr     d2, [%[a]], #8            \n"
      "       fmsub   %d[res], d2, d1, %d[res]  \n"
      "       cmp     %[pb], %[lmt]             \n"
      "       b.lt    2b                        \n"
      // output operands, source operands, and clobber list
      : [a] "+&r"(a), [pb] "+&r"(pb), [res] "+&w"(res), [lmt] "+&r"(lmt)
      : [n] "r"(n), [lmt0] "r"(lmt0)
      : "v0", "v1", "v2", "v3", "cc", "memory");

  input->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_032(struct loop_032_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int n = input->n;

  double res = 0.0;
  int64_t j = 4;

  asm volatile(
      "1:     ldr     d1, [%[b], %[j], lsl #3]  \n"
      "       add     %[j], %[j], #0x5          \n"
      "       ldr     d2, [%[a]], #8            \n"
      "       fmsub   %d[res], d2, d1, %d[res]  \n"
      "       cmp     %[j], %x[n]               \n"
      "       b.lt    1b                        \n"
      // output operands, source operands, and clobber list
      : [a] "+&r"(a), [b] "+&r"(b), [j] "+&r"(j), [res] "+&w"(res)
      : [n] "r"(n)
      : "v1", "v2", "cc", "memory");

  input->res = res;
}
#else
static void inner_loop_032(struct loop_032_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(032, NS_SVE_LOOP_ATTR)
{
  struct loop_032_data data = { .n = SIZE, .res = DBL_MAX, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_double(data.a, SIZE);
  fill_double(data.b, SIZE);

  inner_loops_032(iters, &data);

  double res = data.res;
  bool passed = check_double(res, -496.05, 1.0);
#ifndef STANDALONE
  FINALISE_LOOP_F(32, passed, "%9.6f", -496.05, 1.0, res)
#endif
  return passed ? 0 : 1;
}
