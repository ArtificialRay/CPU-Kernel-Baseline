/*----------------------------------------------------------------------------
#
#   Loop 105: Cascade summation
#
#   Purpose:
#     Use of pairwise FP add instruction.
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
   https://en.wikipedia.org/wiki/Pairwise_summation
*/

struct loop_105_data {
  float *restrict a;
  float *restrict b;
  int n;
  float res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_105(struct loop_105_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) || defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS) ||\
          defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME) || defined(__ARM_FEATURE_SVE) || defined(__ARM_NEON)
static float cascade_summation_16(float *restrict a)
LOOP_ATTR
{
  float t0 = a[0] + a[1];
  float t1 = a[2] + a[3];
  float t2 = a[4] + a[5];
  float t3 = a[6] + a[7];
  float t4 = a[8] + a[9];
  float t5 = a[10] + a[11];
  float t6 = a[12] + a[13];
  float t7 = a[14] + a[15];
  float t10 = t0 + t1;
  float t11 = t2 + t3;
  float t12 = t4 + t5;
  float t13 = t6 + t7;
  float t100 = t10 + t11;
  float t101 = t12 + t13;
  return t100 + t101;
}
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n) {
  if (n == 16) {
    return cascade_summation_16(a);
  }

  int half = n / 2;
  for (int i = 0; i < half; i++) {
    b[i] = a[2 * i] + a[2 * i + 1];
  }

  return cascade_summation(b, b + half, half);
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n)
LOOP_ATTR
{
  if (n == 16) {
    return cascade_summation_16(a);
  }

  float *s = a;
  float *d = b;
  float *lmt = a + n;

  svbool_t p_all = svptrue_b32();
  svuint32_t idx = svuzp1(svindex_u32(0, 1), svindex_u32(1, 1));
#if defined(__ARM_FEATURE_SVE2p1)
#define LOAD_PAIR lda = svld1_x2(svptrue_c32(), s)
#else
#define LOAD(p) svld1_vnum(p_all, s, p)
#define LOAD_PAIR lda = svcreate2(LOAD(0), LOAD(1))
#endif

  while (s < lmt) {
    svfloat32x2_t lda = LOAD_PAIR;
    svfloat32_t b_vec = svaddp_x(p_all, svget2(lda, 0), svget2(lda, 1));
    svfloat32_t b_res = svtbl(b_vec, idx);
    svst1(p_all, d, b_res);
    s += svcntw() * 2;
    d += svcntw();
  }

  int half = n / 2;
  return cascade_summation(b, b + half, half);
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n)
LOOP_ATTR
{
  if (n == 16) {
    return cascade_summation_16(a);
  }

  float *s = a;
  float *d = b;
  float *lmt = a + n;

  asm volatile(
      "       ptrue   p0.s                              \n"
      "       index   z3.s, #0, #1                      \n"
      "       index   z4.s, #1, #1                      \n"
      "       uzp1    z2.s, z3.s, z4.s                  \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "       ptrue   pn8.s                             \n"
      "1:     ld1w    {z0.s-z1.s}, pn8/z, [%[s]]        \n"
#else
      "1:     ld1w    {z0.s}, p0/z, [%[s]]              \n"
      "       ld1w    {z1.s}, p0/z, [%[s], #1, mul vl]  \n"
#endif
      "       incb    %[s], all, mul #2                 \n"
      "       faddp   z0.s, p0/m, z0.s, z1.s            \n"
      "       tbl     z0.s, {z0.s}, z2.s                \n"
      "       st1w    {z0.s}, p0, [%[d]]                \n"
      "       incb    %[d]                              \n"
      "       cmp     %[s], %[lmt]                      \n"
      "       b.lt    1b                                \n"  // loop back
      // output operands, source operands, and clobber list
      : [s] "+&r"(s), [d] "+&r"(d)
      : [lmt] "r"(lmt)
      : "z0", "z1", "z2", "z3", "z4", "p0", "p8", "cc", "memory");

  int half = n / 2;
  return cascade_summation(b, b + half, half);
}
#elif defined(__ARM_FEATURE_SVE)

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n)
LOOP_ATTR
{
  if (n == 16) {
    return cascade_summation_16(a);
  }

  float *s = a;
  float *d = b;
  float *lmt = a + n;

  asm volatile(
      "       ptrue   p0.s                        \n"
      "1:     ld2w    {z0.s, z1.s}, p0/z, [%[s]]  \n"
      "       incb    %[s], all, mul #2           \n"
      "       fadd    z0.s, z0.s, z1.s            \n"
      "       st1w    {z0.s}, p0, [%[d]]          \n"
      "       incb    %[d]                        \n"
      "       cmp     %[s], %[lmt]                \n"
      "       b.lt    1b                          \n"  // loop back
      // output operands, source operands, and clobber list
      : [s] "+&r"(s), [d] "+&r"(d)
      : [lmt] "r"(lmt)
      : "z0", "z1", "p0", "cc", "memory");

  int half = n / 2;
  return cascade_summation(b, b + half, half);
}
#elif defined(__ARM_NEON)

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n) {
  if (n == 16) {
    return cascade_summation_16(a);
  }

  float *s = a;
  float *d = b;
  float *lmt = a + n;

  asm volatile(
      "1:     ldp     q0, q1, [%[s]], #32     \n"
      "       faddp   v0.4s, v0.4s, v1.4s     \n"
      "       str     q0, [%[d]], #16         \n"
      "       cmp     %[s], %[lmt]            \n"
      "       b.lt    1b                      \n"  // loop back
      // output operands, source operands, and clobber list
      : [s] "+&r"(s), [d] "+&r"(d)
      : [lmt] "r"(lmt)
      : "v0", "v1", "cc", "memory");

  int half = n / 2;
  return cascade_summation(b, b + half, half);
}
#else

static float NOINLINE cascade_summation(float *restrict a, float *restrict b,
                                        int n) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

static void inner_loop_105(struct loop_105_data *restrict input)
LOOP_ATTR
{
  float *a = input->a;
  float *b = input->b;
  int n = input->n;
  input->res = cascade_summation(a, b, n);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 4096  // Must be a power of two
#endif

LOOP_DECL(105, SC_SVE_LOOP_ATTR)
{
  struct loop_105_data data = { .n = SIZE, .res = FLT_MAX, };

  ALLOC_64B(data.a, SIZE, "input data");
  ALLOC_64B(data.b, SIZE, "output buffer");

  fill_float_range(data.a, SIZE, -1.0f, 1.0f);
  fill_float_range(data.b, SIZE, -1.0f, 1.0f);

  iters *= 2; // Multiply iters by 2 to increase work

  inner_loops_105(iters, &data);

  float res = data.res;
  bool passed = check_exact_float(res, 0xc1e34698);
#ifndef STANDALONE
  union {
    uint32_t u;
    float f;
  } bits;
  bits.f = res;
  FINALISE_LOOP_I(105, passed, "0x%08"PRIx32, 0xc1e34698, bits.u)
#endif
  return passed ? 0 : 1;
}
