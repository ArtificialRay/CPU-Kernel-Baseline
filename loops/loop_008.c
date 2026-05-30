/*----------------------------------------------------------------------------
#
#   Loop 008: Precise fp64 add reduction
#
#   Purpose:
#     Use of FADDA instructions.
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


struct loop_008_data {
  double *a;
  int n;
  double res;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_008(struct loop_008_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;
  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += a[i];
  }
  data->res = res;
}
#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;

  double res = 0.0;
  svbool_t p;
  FOR_LOOP_64(int, i, 0, n, p) {
    res = svadda(p, res, svld1(p, &a[i]));
  }

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;

  double res = 0.0f;
  int i = 0;
  asm volatile(
      "   whilelt pn8.d, %x[i], %x[n], vlx4                 \n"
      "   b.none  2f                                        \n"
      "1:                                                   \n"
      "   ld1d    {z0.d-z3.d}, pn8/z, [%[a], %x[i], lsl #3] \n"
      "   pext    {p0.d,p1.d}, pn8[0]                       \n"
      "   pext    {p2.d,p3.d}, pn8[1]                       \n"
      "   incd    %x[i], all, mul #4                        \n"
      "   fadda   %d[res], p0, %d[res], z0.d                \n"
      "   fadda   %d[res], p1, %d[res], z1.d                \n"
      "   fadda   %d[res], p2, %d[res], z2.d                \n"
      "   fadda   %d[res], p3, %d[res], z3.d                \n"
      "   whilelt pn8.d, %x[i], %x[n], vlx4                 \n"
      "   b.first 1b                                        \n"
      "2:                                                   \n"
      : [res] "+&w"(res), [i] "+&r"(i)
      : [n] "r"(n), [a] "r"(a)
      : "z0", "z1", "z2", "z3", "p0", "p1", "p2", "p3", "p8", "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;

  double *lmt = a + (n - (n % 8));
  double res;

  asm volatile(
      "       fmov    %d[res], xzr                        \n"
      "       ptrue   p0.d                                \n"
      "       b       2f                                  \n"

      "1:     ld1d    {z1.d}, p0/z, [%[a]]                \n"
      "       ld1d    {z2.d}, p0/z, [%[a], #1, mul vl]    \n"
      "       ld1d    {z3.d}, p0/z, [%[a], #2, mul vl]    \n"
      "       ld1d    {z4.d}, p0/z, [%[a], #3, mul vl]    \n"

      "       incb    %[a], all, mul #4                   \n"

      "       fadda   %d[res], p0, %d[res], z1.d          \n"
      "       fadda   %d[res], p0, %d[res], z2.d          \n"
      "       fadda   %d[res], p0, %d[res], z3.d          \n"
      "       fadda   %d[res], p0, %d[res], z4.d          \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [a] "+&r"(a)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "p0", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i];
  }
  data->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;

  double *lmt = a + (n - (n % 8));
  double res;

  asm volatile(
      "       fmov    %d[res], xzr                        \n"
      "       b       2f                                  \n"

      "1:     ldp     d1,  d2,  [%[a]]                    \n"
      "       ldp     d3,  d4,  [%[a], #16]               \n"
      "       ldp     d5,  d6,  [%[a], #32]               \n"
      "       ldp     d7,  d8,  [%[a], #48]               \n"

      "       add     %[a], %[a], #64                     \n"

      "       fadd    %d[res], %d[res], d1                \n"
      "       fadd    %d[res], %d[res], d2                \n"
      "       fadd    %d[res], %d[res], d3                \n"
      "       fadd    %d[res], %d[res], d4                \n"
      "       fadd    %d[res], %d[res], d5                \n"
      "       fadd    %d[res], %d[res], d6                \n"
      "       fadd    %d[res], %d[res], d7                \n"
      "       fadd    %d[res], %d[res], d8                \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [a] "+&r"(a)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i];
  }
  data->res = res;
}
#else
static void inner_loop_008(struct loop_008_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(008, NS_SVE_LOOP_ATTR)
{
  struct loop_008_data data = { .n = SIZE, .res = DBL_MAX, };
  ALLOC_64B(data.a, SIZE, "input array");
  fill_double(data.a, SIZE);

  inner_loops_008(iters, &data);

  double res = data.res;
  bool passed = check_exact_double(res, 0x40b37aa777747742L);
#ifndef STANDALONE
  union {
    uint64_t u;
    double d;
  } bits;
  bits.d = res;
  FINALISE_LOOP_I(8, passed, "0x%016"PRIx64, (uint64_t) 0x40b37aa777747742, bits.u)
#endif
  return passed ? 0 : 1;
}
