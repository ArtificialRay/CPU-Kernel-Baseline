/*----------------------------------------------------------------------------
#
#   Loop 114: Auto-correlation
#
#   Purpose:
#     Use of shifts, widening mult and load-replicate instructions.
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


struct loop_114_data {
  int16_t *restrict data;
  int16_t *restrict res;
  int32_t n;
  int32_t lags;
  int16_t scale;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_114(struct loop_114_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_114(struct loop_114_data *restrict input) {
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  for (int lag = 0; lag < lags; lag++) {
    int32_t acc = 0;
    int lmt = n - lag;
    for (int i = 0; i < lmt; i++) {
      acc += ((int32_t)data[i] * (int32_t)data[i + lag]) >> scale;
    }

    res[lag] = (int16_t)(acc >> 16);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_114(struct loop_114_data *restrict input)
LOOP_ATTR
{
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  svbool_t all = svptrue_b8();
  svuint32_t scales = svdup_u32(scale);

  svbool_t pgl;
  FOR_LOOP_16(int32_t, lag, 0, lags, pgl) {
    svint32_t acc_lo = svdup_s32(0);
    svint32_t acc_hi = svdup_s32(0);

    int16_t *lmt = data + n;
    for (int16_t *p0 = data, *p1 = data + lag; p0 < lmt; p0++, p1++) {
      svint16_t vec_cur = svdup_s16_x(all, p0[0]);
      svint16_t vec_lag = svld1_s16(pgl, p1);
      svint32_t lo = svasr_x(all, svmullb(vec_cur, vec_lag), scales);
      svint32_t hi = svasr_x(all, svmullt(vec_cur, vec_lag), scales);
      acc_lo = svadd_x(all, acc_lo, lo);
      acc_hi = svadd_x(all, acc_hi, hi);
    }

    svint16_t half_lo = svreinterpret_s16(acc_lo);
    svint16_t half_hi = svreinterpret_s16(acc_hi);
    svint16_t zipped = svtrn2_s16(half_lo, half_hi);

    svst1(pgl, res + lag, zipped);
  }
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))

static void inner_loop_114(struct loop_114_data *restrict input)
LOOP_ATTR
{
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  int64_t step = get_sve_vl() / 16;

  for (int64_t lag = 0; lag < lags; lag += step) {
    int16_t *lmt = data + n;
    int16_t *p0 = data;
    int16_t *p1 = data + lag;
    int16_t *dst = res + lag;

    asm volatile(
        "   ptrue   p0.b                    \n"
        "   whilelo p1.h, xzr, %x[lags]     \n"
        "   dup     z5.s, %w[scale]         \n"
        "   mov     z0.s, #0x0              \n"
        "   mov     z1.s, #0x0              \n"
        "1: ld1rh   z2.h, p1/z, [%[p0]]     \n"
        "   ld1h    z3.h, p1/z, [%[p1]]     \n"
        "   add     %[p0], %[p0], #2        \n"
        "   add     %[p1], %[p1], #2        \n"
        "   smullb  z4.s, z2.h, z3.h        \n"
        "   smullt  z3.s, z2.h, z3.h        \n"
        "   asr     z4.s, p0/m, z4.s, z5.s  \n"
        "   asr     z3.s, p0/m, z3.s, z5.s  \n"
        "   add     z0.s, z4.s, z0.s        \n"
        "   add     z1.s, z3.s, z1.s        \n"
        "   cmp     %[p0], %[lmt]           \n"
        "   b.ne    1b                      \n"
        "   trn2    z0.h, z0.h, z1.h        \n"
        "   st1h    z0.h, p1, [%[dst]]      \n"
        // output operands, source operands, and clobber list
        : [p0] "+&r"(p0), [p1] "+&r"(p1)
        : [scale] "r"(scale), [dst] "r"(dst), [lmt] "r"(lmt), [lags] "r"(lags)
        : "v0", "v1", "v2", "v3", "v4", "v5", "p0", "p1", "memory", "cc");
  }
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_114(struct loop_114_data *restrict input)
LOOP_ATTR
{
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  int64_t step = get_sve_vl() / 16;

  for (int64_t lag = 0; lag < lags; lag += step) {
    int16_t *lmt = data + n;
    int16_t *p0 = data;
    int16_t *p1 = data + lag;
    int16_t *dst = res + lag;

    asm volatile(
        "   ptrue   p0.b                    \n"
        "   whilelo p1.h, xzr, %x[lags]     \n"
        "   dup     z5.s, %w[scale]         \n"
        "   mov     z0.s, #0x0              \n"
        "   mov     z1.s, #0x0              \n"
        "1: ld1rh   z2.h, p1/z, [%[p0]]     \n"
        "   ld1h    z3.h, p1/z, [%[p1]]     \n"
        "   add     %[p0], %[p0], #2        \n"
        "   add     %[p1], %[p1], #2        \n"
        "   movprfx z4.h, p0/z, z3.h        \n"
        "   mul     z4.h, p0/m, z4.h, z2.h  \n"
        "   smulh   z3.h, p0/m, z3.h, z2.h  \n"
        "   zip1    z2.h, z4.h, z3.h        \n"
        "   zip2    z3.h, z4.h, z3.h        \n"
        "   asr     z2.s, p0/m, z2.s, z5.s  \n"
        "   asr     z3.s, p0/m, z3.s, z5.s  \n"
        "   add     z0.s, z2.s, z0.s        \n"
        "   add     z1.s, z3.s, z1.s        \n"
        "   cmp     %[p0], %[lmt]           \n"
        "   b.ne    1b                      \n"
        "   uzp2    z0.h, z0.h, z1.h        \n"
        "   st1h    z0.h, p1, [%[dst]]      \n"
        // output operands, source operands, and clobber list
        : [p0] "+&r"(p0), [p1] "+&r"(p1)
        : [scale] "r"(scale), [dst] "r"(dst), [lmt] "r"(lmt), [lags] "r"(lags)
        : "v0", "v1", "v2", "v3", "v4", "v5", "p0", "p1", "memory", "cc");
  }
}
#elif defined(__ARM_NEON)

static void inner_loop_114(struct loop_114_data *restrict input) {
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  int32_t ns = -scale;
  for (int lag = 0; lag < lags; lag += 8) {
    int16_t *lmt = data + n;
    int16_t *p0 = data;
    int16_t *p1 = data + lag;
    int16_t *dst = res + lag;

    asm volatile(
        "   dup     v5.4s, %[ns].s[0]   \n"
        "   movi    v0.4s, #0x0         \n"
        "   movi    v1.4s, #0x0         \n"
        "1: ld1r    { v2.8h }, [%[p0]]  \n"
        "   ldr     q3, [%[p1]]         \n"
        "   add     %[p0], %[p0], #2    \n"
        "   add     %[p1], %[p1], #2    \n"
        "   smull   v4.4s, v2.4h, v3.4h \n"
        "   smull2  v3.4s, v2.8h, v3.8h \n"
        "   sshl    v4.4s, v4.4s, v5.4s \n"
        "   sshl    v3.4s, v3.4s, v5.4s \n"
        "   add     v0.4s, v4.4s, v0.4s \n"
        "   add     v1.4s, v3.4s, v1.4s \n"
        "   cmp     %[p0], %[lmt]       \n"
        "   b.ne    1b                  \n"
        "   uzp2    v0.8h, v0.8h, v1.8h \n"
        "   str     q0, [%[dst]]        \n"
        // output operands, source operands, and clobber list
        : [p0] "+&r"(p0), [p1] "+&r"(p1)
        : [ns] "w"(ns), [dst] "r"(dst), [lmt] "r"(lmt)
        : "v0", "v1", "v2", "v3", "v4", "v5", "memory", "cc");
  }
}
#else

static void inner_loop_114(struct loop_114_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#define DATA 256
#define LAGS 64
#define SCALE 3

_Static_assert(LAGS % 8 == 0, "LAGS must be a multiple of 8");

LOOP_DECL(114, SC_SVE_LOOP_ATTR)
{
  struct loop_114_data data = { .n = DATA, .lags = LAGS, .scale = SCALE, };

  ALLOC_64B(data.data, DATA + LAGS, "input data");
  ALLOC_64B(data.res, LAGS, "result buffer");

  fill_int16(data.data, DATA);  // Leave LAGS padding at the end as 0
  fill_int16(data.res, LAGS);

  inner_loops_114(iters, &data);

  int res = 0;
  for (int i = 0; i < LAGS; i++) {
    res += i * data.res[i];
  }

  bool passed = res == 4194938;
#ifndef STANDALONE
  FINALISE_LOOP_I(114, passed, "%d", 4194938, res)
#endif
  return passed ? 0 : 1;
}
