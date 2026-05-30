/*----------------------------------------------------------------------------
#
#   Loop 128: alias in contiguous access
#
#   Purpose:
#     Use of simd loop with possible alias in contiguous mem access
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

#include <assert.h>

struct loop_128_data {
  uint32_t *restrict a;
  uint32_t *b;
  uint32_t *c;
  int n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_128(struct loop_128_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_128(struct loop_128_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;
  int n = data->n;

  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_128(struct loop_128_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;
  int n = data->n;

  uint32_t *lmt = a + n;
  uint32_t cnt = 0;
  svbool_t p0, p2;
  svbool_t p1 = svwhilewr(b, c);
  while (a < lmt) {
    p0 = svwhilelt_b32((uint64_t)a, (uint64_t)lmt);
    p2 = svand_z(p0, p1, p0);
    svuint32_t a_vec = svld1(p2, a);
    svuint32_t b_vec = svld1(p2, b);
    a_vec = svadd_x(p2, a_vec, b_vec);
    svst1(p2, c, a_vec);
    cnt = svcntp_b32(p2, p2);
    a += cnt;
    b += cnt;
    c += cnt;
  }
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))
static void inner_loop_128(struct loop_128_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;
  int n = data->n;

  uint32_t cnt = 0;
  uint32_t *lmt = a + n;
  asm volatile(
      "       whilewr p1.s, %[b], %[c]                             \n"
      "       b       2f                                           \n"

      "1:     and     p2.b, p0/z, p1.b, p0.b                       \n"
      "       ld1w    {z1.s}, p2/z, [%[a]]                         \n"
      "       ld1w    {z2.s}, p2/z, [%[b]]                         \n"
      "       add     z1.s, z1.s, z2.s                             \n"
      "       st1w    {z1.s}, p2, [%[c]]                           \n"
      "       cntp    %x[cnt], p2, p2.s                            \n"
      "       lsl     %x[cnt], %x[cnt], #2                         \n"
      "       add     %[a], %[a], %x[cnt]                          \n"
      "       add     %[b], %[b], %x[cnt]                          \n"
      "       add     %[c], %[c], %x[cnt]                          \n"

      "2:     whilelt p0.s, %[a], %[lmt]                           \n"
      "       b.first 1b                                           \n"  // loop
                                                                        // back
      // output operands, source operands, and clobber list
      : [c] "+&r"(c), [a] "+&r"(a), [b] "+&r"(b), [cnt] "+&r"(cnt)
      : [lmt] "r"(lmt)
      : "z1", "z2", "z3", "p0", "p1", "p2", "cc", "memory");
}
#else
static void inner_loop_128(struct loop_128_data *restrict data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(128, SC_SVE_LOOP_ATTR)
{
  struct loop_128_data data;

  ALLOC_64B(data.a, SIZE, "1st operand array");
  ALLOC_64B(data.b, SIZE, "2nd operand array");

  fill_uint32(data.a, SIZE);
  fill_uint32(data.b, SIZE);

  for (int i = 0; i < SIZE; i++) {
    data.a[i] %= 100;
    data.b[i] %= 100;
  }

  uint32_t offset = rand_uint32() % 10;
  data.c = data.b + offset;
  data.n = SIZE - offset;

  inner_loops_128(iters, &data);

  uint32_t res = data.b[SIZE - offset];
  bool passed = res == 0x0001de18;
#ifndef STANDALONE
  FINALISE_LOOP_I(128, passed, "0x%08"PRIx32, 0x0001de18, res)
#endif
  return passed ? 0 : 1;
}
