/*----------------------------------------------------------------------------
#
#   Loop 033: FP64 Inner product
#
#   Purpose:
#     Use of a tight loop with INC and WHILE instructions.
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


struct loop_033_data {
  double *restrict a;
  double *restrict b;
  int64_t n;
  double res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_033(struct loop_033_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_033(struct loop_033_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  input->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_033(struct loop_033_data *restrict input)
LOOP_ATTR
{
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  svfloat64_t c = svdup_f64(0.0);

  svbool_t p;
  FOR_LOOP_64(int64_t, i, 0, n, p) {
    svfloat64_t a_vec = svld1(p, a + i);
    svfloat64_t b_vec = svld1(p, b + i);
    c = svmla_m(p, c, a_vec, b_vec);
  }

  input->res = svaddv(svptrue_b64(), c);
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))
static void inner_loop_033(struct loop_033_data *restrict input)
LOOP_ATTR
{
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  double res = 0.0;
  int64_t i = 0;

  asm volatile(
      "       mov     z0.d, #0                              \n"
      "       whilele p0.d, xzr, %[n]                       \n"
      "1:     ld1d    {z1.d}, p0/z, [%[a], %[i], lsl #3]    \n"
      "       ld1d    {z2.d}, p0/z, [%[b], %[i], lsl #3]    \n"
      "       incd    %[i]                                  \n"
      "       fmla    z0.d, p0/m, z1.d, z2.d                \n"
      "       whilelt p0.d, %[i], %[n]                      \n"
      "       b.any   1b                                    \n"
      "       ptrue   p0.d                                  \n"
      "       faddv   %d[res], p0, z0.d                     \n"
      // output operands, source operands, and clobber list
      : [res] "+&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "v0", "v1", "v2", "p0", "cc", "memory");

  input->res = res;
}
#elif defined(__ARM_NEON)
static void inner_loop_033(struct loop_033_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  double res = 0.0;
  int64_t i = 0;
  int64_t lmt0 = (n * 8) - 16;
  int64_t lmt = n * 8;

  asm volatile(
      "       movi    v0.16b, #0                \n"
      "1:     ldr     q1, [%[a], %[i]]          \n"
      "       ldr     q2, [%[b], %[i]]          \n"
      "       add     %[i], %[i], #0x10         \n"
      "       fmla    v0.2d, v1.2d, v2.2d       \n"
      "       cmp     %[i], %[lmt0]             \n"
      "       b.lt    1b                        \n"
      "       faddp   v0.2d, v0.2d, v0.2d       \n"
      "       fmov    %d[res], d0               \n"
      "2:     ldr     d1, [%[a], %[i]]          \n"
      "       ldr     d2, [%[b], %[i]]          \n"
      "       add     %[i], %[i], #0x8          \n"
      "       fmadd   %d[res], d1, d2, %d[res]  \n"
      "       cmp     %[i], %[lmt]              \n"
      "       b.lt    2b                        \n"
      // output operands, source operands, and clobber list
      : [res] "+&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [lmt] "r"(lmt), [lmt0] "r"(lmt0)
      : "v0", "v1", "v2", "cc", "memory");

  input->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_033(struct loop_033_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  double res = 0.0;
  int64_t i = 0;
  int64_t lmt = n * 8;

  asm volatile(
      "1:     ldr     d1, [%[a], %[i]]          \n"
      "       ldr     d2, [%[b], %[i]]          \n"
      "       add     %[i], %[i], #0x8          \n"
      "       fmadd   %d[res], d1, d2, %d[res]  \n"
      "       cmp     %[i], %[lmt]              \n"
      "       b.lt    1b                        \n"
      // output operands, source operands, and clobber list
      : [a] "+&r"(a), [b] "+&r"(b), [res] "+&w"(res), [i] "+&r"(i)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "cc", "memory");

  input->res = res;
}
#else
static void inner_loop_033(struct loop_033_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 3000
#endif

LOOP_DECL(033, SC_SVE_LOOP_ATTR)
{
  struct loop_033_data data = { .n = SIZE, .res = DBL_MAX, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_double(data.a, SIZE);
  fill_double(data.b, SIZE);

  inner_loops_033(iters, &data);

  double res = data.res;
  bool passed = check_double(res, 755.5, 1.0);
#ifndef STANDALONE
  FINALISE_LOOP_F(33, passed, "%9.6f", 755.5, 1.0, res)
#endif
  return passed ? 0 : 1;
}
