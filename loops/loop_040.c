/*----------------------------------------------------------------------------
#
#   Loop 040: Clamp operation
#
#   Purpose:
#     Use of simd clamp (or min, max and shift) instructions.
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


struct loop_040_data {
  int32_t count;
  int32_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLAMP(v, a, b) MIN(MAX(v, b), a)

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_040(struct loop_040_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_040(struct loop_040_data *restrict input) {
  int32_t count = input->count;
  int32_t value = count / 2;
  int32_t result = 0;

  for (int32_t i = 0; i < count; i++) {
    result += CLAMP(value, 2 * i, i);
  }
  input->res = result;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_040(struct loop_040_data *restrict input)
LOOP_ATTR
{
  uint64_t count = input->count;

  svbool_t  p_all = svptrue_b32();
  svint32_t val_2 = svdup_s32((int32_t)count / 2);
  svint32_t acc_0 = svdup_s32(0);
  svint32_t acc_1 = svdup_s32(0);
  svint32_t min_0 = svindex_s32(0, 1);
  svint32_t min_1 = svadd_n_s32_x(p_all, min_0, svcntw());

  svint32_t max_0, max_1;
  svint32_t val_0, val_1;
  svbool_t p0, p1;

  for (uint64_t i = 0; i < count; i += svcnth()) {
    max_0 = svlsl_x(p_all, min_0, 1);
    max_1 = svlsl_x(p_all, min_1, 1);
#if defined(__ARM_FEATURE_SVE2p1)
    svboolx2_t pn = svwhilelt_b32_x2(i, count);
    p0 = svget2(pn, 0);
    p1 = svget2(pn, 1);
    val_0 = svclamp(val_2, min_0, max_0);
    val_1 = svclamp(val_2, min_1, max_1);
#else
    p0 = svwhilelt_b32(i + svcntw() * 0, count);
    p1 = svwhilelt_b32(i + svcntw() * 1, count);
    val_0 = svmax_m(p_all, val_2, min_0);
    val_1 = svmax_m(p_all, val_2, min_1);
    val_0 = svmin_m(p_all, val_0, max_0);
    val_1 = svmin_m(p_all, val_1, max_1);
#endif
    acc_0 = svadd_m(p0, acc_0, val_0);
    acc_1 = svadd_m(p1, acc_1, val_1);
    min_0 = svadd_x(p_all, min_0, svcnth());
    min_1 = svadd_x(p_all, min_1, svcnth());
  }

  input->res = svaddv_s32(p_all, svadd_x(p_all, acc_0, acc_1));
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_040(struct loop_040_data *restrict input)
LOOP_ATTR
{
  uint64_t count = input->count;
  uint64_t index = 0;
  int32_t value = count / 2;
  int32_t result = 0;

  asm volatile(
      "   whilelt {p0.s, p1.s}, xzr, %x[n]      \n"
      "   b.none  2f                            \n"
      "   ptrue   p2.s                          \n"
      "   dup     z8.s, %w[val]                 \n"
      "   index   z4.s, #0, #1                  \n"
      "   mov     z0.s, #0                      \n"
      "   mov     z5.d, z4.d                    \n"
      "   mov     z1.s, #0                      \n"
      "   incw    z5.s                          \n"
      "1:                                       \n"
      "   lsl     z6.s, z4.s, #1                \n"
      "   lsl     z7.s, z5.s, #1                \n"
      "   inch    %[i]                          \n"
      "   movprfx z2, z8                        \n"
      "   sclamp  z2.s, z4.s, z6.s              \n"
      "   movprfx z3, z8                        \n"
      "   sclamp  z3.s, z5.s, z7.s              \n"
      "   incw    z4.s, all, mul #2             \n"
      "   incw    z5.s, all, mul #2             \n"
      "   add     z0.s, p0/m, z0.s, z2.s        \n"
      "   add     z1.s, p1/m, z1.s, z3.s        \n"
      "   whilelt {p0.s, p1.s}, %x[i], %x[n]    \n"
      "   b.first 1b                            \n"
      "   add     z9.s, z0.s, z1.s              \n"
      "   saddv   %d[res], p2, z9.s             \n"
      "2:                                       \n"
      : [res] "+&w"(result), [i] "+&r"(index)
      : [n] "r"(count), [val] "r"(value)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "p0", "p1", "p2", "p8", "cc");

  input->res = result;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_040(struct loop_040_data *restrict input)
LOOP_ATTR
{
  int32_t count = input->count;
  int32_t value = count / 2;
  int32_t result = 0;
  int32_t iters = count / svcnth();

  svbool_t p_all = svptrue_b32();
  svint32_t vvalue = svdup_s32(value);
  svint32_t acc0 = svdup_s32(0);
  svint32_t acc1 = svdup_s32(0);
  svint32_t min0 = svindex_s32(0, 1);
  svint32_t min1 = svadd_n_s32_x(p_all, min0, svcntw());
  svint32_t wcnt2 = svdup_s32(svcnth());

  int it = iters;

  asm volatile(
      "1:     subs    %w[it], %w[it], #1                  \n"
      "       movprfx z2,   %[vvalue]                     \n"
      "       smax    z2.s, %[p_all]/m, z2.s, %[min0].s   \n"
      "       movprfx z3,   %[vvalue]                     \n"
      "       smax    z3.s, %[p_all]/m, z3.s, %[min1].s   \n"
      "       lsl     z0.s, %[min0].s, #1                 \n"
      "       add     %[min0].s, %[min0].s, %[wcnt2].s    \n"
      "       lsl     z1.s, %[min1].s, #1                 \n"
      "       add     %[min1].s, %[min1].s, %[wcnt2].s    \n"
      "       smin    z0.s, %[p_all]/m, z0.s, z2.s        \n"
      "       smin    z1.s, %[p_all]/m, z1.s, z3.s        \n"
      "       add     %[acc0].s, %[acc0].s, z0.s          \n"
      "       add     %[acc1].s, %[acc1].s, z1.s          \n"
      "       b.ne    1b                                  \n"
      // output operands, source operands, and clobber list
      : [it] "+&r"(it), [min0] "+&w"(min0), [min1] "+&w"(min1), [acc0] "+&w"(acc0),
        [acc1] "+&w"(acc1)
      : [p_all] "Upl"(p_all), [iters] "r"(iters), [vvalue] "w"(vvalue),
        [wcnt2] "w"(wcnt2)
      : "z0", "z1", "z2", "z3", "cc");

  acc0 = svadd_x(p_all, acc0, acc1);
  result = svaddv_s32(p_all, acc0);

  for (int32_t i = iters * svcnth(); i < count; i++) {
    result += CLAMP(value, i, 2 * i);
  }
  input->res = result;
}
#elif defined(__ARM_NEON)
static void inner_loop_040(struct loop_040_data *restrict input) {
  int32_t count = input->count;
  int32_t value = count / 2;
  int32_t result = 0;
  int32_t iters = count / 8;

  int32x4_t vvalue = vdupq_n_s32(value);
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t min0 = {0, 1, 2, 3};
  int32x4_t min1 = {4, 5, 6, 7};
  int32x4_t wcnt2 = vdupq_n_s32(8);

  int it = iters;

  asm volatile(
      "1:     subs    %w[it], %w[it], #1                    \n"
      "       smax    v2.4s, %[vvalue].4s, %[min0].4s       \n"
      "       smax    v3.4s, %[vvalue].4s, %[min1].4s       \n"
      "       shl     v0.4s, %[min0].4s, #1                 \n"
      "       add     %[min0].4s, %[min0].4s, %[wcnt2].4s   \n"
      "       shl     v1.4s, %[min1].4s, #1                 \n"
      "       add     %[min1].4s, %[min1].4s, %[wcnt2].4s   \n"
      "       smin    v0.4s, v2.4s, v0.4s                   \n"
      "       smin    v1.4s, v3.4s, v1.4s                   \n"
      "       add     %[acc0].4s, %[acc0].4s, v0.4s         \n"
      "       add     %[acc1].4s, %[acc1].4s, v1.4s         \n"
      "       b.ne    1b                                    \n"
      // output operands, source operands, and clobber list
      : [it] "+&r"(it), [min0] "+&w"(min0), [min1] "+&w"(min1), [acc0] "+&w"(acc0),
        [acc1] "+&w"(acc1)
      : [vvalue] "w"(vvalue), [wcnt2] "w"(wcnt2)
      : "v0", "v1", "v2", "v3", "cc");

  acc0 = vaddq_s32(acc0, acc1);
  result = vaddvq_s32(acc0);

  for (int32_t i = iters * 8; i < count; i++) {
    result += CLAMP(value, i, 2 * i);
  }
  input->res = result;
}
#else
static void inner_loop_040(struct loop_040_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#define COUNT 10000

LOOP_DECL(040, SC_SVE_LOOP_ATTR)
{
  struct loop_040_data data = { .count = COUNT, .res = 0, };
  inner_loops_040(iters, &data);

  int32_t res = data.res;
  int32_t expected = 56245000;
  bool passed = res == expected;
#ifndef STANDALONE
  FINALISE_LOOP_I(40, passed, "%d", expected, res)
#endif
  return passed ? 0 : 1;
}
