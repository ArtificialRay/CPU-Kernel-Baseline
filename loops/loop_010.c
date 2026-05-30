/*----------------------------------------------------------------------------
#
#   Loop 010: Conditional reduction (fp)
#
#   Purpose:
#     Use of CLAST (SIMD&FP scalar) instructions.
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


struct loop_010_data {
  float *a;
  uint64_t n;
  int res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_010(struct loop_010_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_010(struct loop_010_data *restrict data) {
  float *a = data->a;
  uint64_t n = data->n;

  bool any = 0;
  bool all = 1;

  for (int i = 0; i < n; i++) {
    if (a[i] < 0.0f) {
      any = 1;
    } else {
      all = 0;
    }
  }
  data->res = all ? 1 : any ? 2 : 3;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_010(struct loop_010_data *restrict data)
LOOP_ATTR
{
  float *a = data->a;
  uint64_t n = data->n;

  int any = 0;
  int all = 1;
  svint32_t any_vec = svdup_s32(0);
  svint32_t all_vec = svdup_s32(1);

  svbool_t p0;
  FOR_LOOP_32(uint64_t, i, 0, n, p0) {
    svfloat32_t a_vec = svld1(p0, &a[i]);
    svbool_t p1 = svcmplt(p0, a_vec, 0.0f);
    any = svclastb(p1, any, all_vec);
    svbool_t p2 = svnot_z(p0, p1);
    all = svclastb(p2, all, any_vec);
  }

  data->res = all ? 1 : any ? 2 : 3;
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))

static void inner_loop_010(struct loop_010_data *restrict data)
LOOP_ATTR
{
  float *a = data->a;
  uint64_t n = data->n;

  uint64_t i = 0;
  int any = 0;
  int all = 1;

  asm volatile(
      "   mov     z1.s, #1                            \n"
      "   mov     z2.s, #0                            \n"
      "   b       2f                                  \n"
      "1: ld1w    {z0.s}, p0/z, [%[a], %[i], lsl #2]  \n"
      "   incw    %[i]                                \n"
      "   fcmlt   p1.s, p0/z, z0.s, #0.0              \n"
      "   clastb  %s[any], p1, %s[any], z1.s          \n"
      "   not     p2.b, p0/z, p1.b                    \n"
      "   clastb  %s[all], p2, %s[all], z2.s          \n"
      "2: whilelo p0.s, %[i], %[n]                    \n"
      "   b.any   1b                                  \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [any] "+&w"(any), [all] "+&w"(all)
      : [a] "r"(a), [n] "r"(n)
      : "v0", "v1", "v2", "p0", "p1", "cc", "memory");

  data->res = all ? 1 : any ? 2 : 3;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

// Scalar and Neon version (can't do better with Neon)
static void inner_loop_010(struct loop_010_data *restrict data) {
  float *a = data->a;
  uint64_t n = data->n;

  bool any = 0;
  bool all = 1;

  asm volatile(
      "1:   ldr	s0, [%[a]], #4                    \n"
      "     fcmpe	s0, #0.0                        \n"
      "     csel	%w[all], %w[all], wzr, mi       \n"
      "     csel	%w[any], %w[one], %w[any], mi   \n"
      "     cmp	%[a], %[lmt]                      \n"
      "     b.ne	1b                              \n"
      // output operands, source operands, and clobber list
      : [any] "+&r"(any), [all] "+&r"(all), [a] "+&r"(a)
      : [lmt] "r"(a + n), [one] "r"(1)
      : "v0", "memory", "cc");

  data->res = all ? 1 : any ? 2 : 3;
}
#else
static void inner_loop_010(struct loop_010_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(010, SC_SVE_LOOP_ATTR)
{
  struct loop_010_data data = { .n = SIZE, .res = 0, };
  ALLOC_64B(data.a, SIZE, "input array");
  fill_float(data.a, SIZE);

  inner_loops_010(iters, &data);

  int res = data.res;
  bool passed = res == 3;
#ifndef STANDALONE
  FINALISE_LOOP_I(10, passed, "%d", 3, res)
#endif
  return passed ? 0 : 1;
}
