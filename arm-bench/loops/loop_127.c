/*----------------------------------------------------------------------------
#
#   Loop 127: loop early exit
#
#   Purpose:
#     Use of simd loop with early exit.
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


struct loop_127_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  int n;
  uint32_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_127(struct loop_127_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_127(struct loop_127_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  uint32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
    if (a[i] == 512) {
      break;
    }
  }
  data->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_127(struct loop_127_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  svuint32_t res_vec = svdup_u32(0);
  svuint32_t pred_vec = svdup_u32(512);
  svbool_t p0;
  svbool_t p1;

  FOR_LOOP_32(int32_t, i, 0, n, p0) {
    svuint32_t a_vec = svld1(p0, a);
    svuint32_t b_vec = svld1(p0, b);
    p1 = svcmpeq(p0, a_vec, pred_vec);
    if (svptest_any(p0, p1)) {
      p1 = svbrka_z(p0, p1);
      res_vec = svmla_m(p1, res_vec, a_vec, b_vec);
      break;
    } else {
      res_vec = svmla_m(p0, res_vec, a_vec, b_vec);
      a += svcntw();
      b += svcntw();
    }
  }
  data->res = svaddv(svptrue_b32(), res_vec);
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))
static void inner_loop_127(struct loop_127_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  uint32_t res = 0;
  uint32_t idx = 0;

  asm volatile(
      "       ptrue   p2.b                                  \n"
      "       pfalse  p3.b                                  \n"
      "       mov     z3.s,   #0                            \n"
      "       mov     z4.s,   #128                          \n"
      "       lsl     z4.s,   z4.s,   #2                    \n"
      "       b       2f                                    \n"

      "1:     ptest   p2, p3.b                              \n"
      "       ld1w    {z1.s}, p0/z, [%[a], %x[idx], lsl #2] \n"
      "       ld1w    {z2.s}, p0/z, [%[b], %x[idx], lsl #2] \n"
      "       cmpeq   p1.s,  p0/z, z1.s, z4.s               \n"
      "       b.any   3f                                    \n"
      "       mla     z3.s,  p0/m, z1.s, z2.s               \n"
      "       incw    %x[idx]                               \n"

      "2:     whilelt p0.s,  %w[idx], %w[n]                 \n"
      "       b.first 1b                                    \n"  // loop back
      "       b       4f                                    \n"  // loop back

      "3:     brka    p1.b,  p0/z, p1.b                     \n"
      "       mla     z3.s,  p1/m, z1.s, z2.s               \n"

      "4:     uaddv   %d[res], p2, z3.s                     \n"
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [idx] "+&r"(idx)
      : [n] "r"(n), [a] "r"(a), [b] "r"(b)
      : "v1", "v2", "v3", "v4", "p0", "p1", "p2", "p3", "cc", "memory");
  data->res = res;
}
#else
static void inner_loop_127(struct loop_127_data *restrict data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(127, SC_SVE_LOOP_ATTR)
{
  struct loop_127_data data = { .n = SIZE, .res = 0, };

  ALLOC_64B(data.a, SIZE, "1st operand array");
  ALLOC_64B(data.b, SIZE, "2nd operand array");

  fill_uint32(data.a, SIZE);
  fill_uint32(data.b, SIZE);

  for (int i = 0; i < SIZE; i++) {
    data.a[i] %= 600;
    data.b[i] %= 600;
  }

  iters *= 2; // Multiply iters by 2 to increase work

  inner_loops_127(iters, &data);

  uint32_t res = data.res;
  bool passed = res == 0x03005c9d;
#ifndef STANDALONE
  FINALISE_LOOP_I(127, passed, "0x%08"PRIx32, 0x03005c9d, res)
#endif
  return passed ? 0 : 1;
}
