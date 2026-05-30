/*----------------------------------------------------------------------------
#
#   Loop 035: Array addition
#
#   Purpose:
#     Use of WHILE for loop control.
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


struct loop_035_data {
  float *restrict a;
  float *restrict b;
  float *restrict c;
  int64_t n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_035(struct loop_035_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_035(struct loop_035_data *restrict input) {
  float *restrict a = input->a;
  float *restrict b = input->b;
  float *restrict c = input->c;
  int64_t n = input->n;

  for (int64_t i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_035(struct loop_035_data *restrict input)
LOOP_ATTR
{
  float *restrict a = input->a;
  float *restrict b = input->b;
  float *restrict c = input->c;
  int64_t n = input->n;

  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, n, p) {
    svfloat32_t a_vec = svld1(p, a + i);
    svfloat32_t b_vec = svld1(p, b + i);
    svfloat32_t c_vec = svadd_x(p, a_vec, b_vec);
    svst1(p, c + i, c_vec);
  }
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))
static void inner_loop_035(struct loop_035_data *restrict input)
LOOP_ATTR
{
  float *restrict a = input->a;
  float *restrict b = input->b;
  float *restrict c = input->c;
  int64_t n = input->n;

  int64_t i = 0;

  asm volatile(
      "       whilelo p0.s, xzr, %[n]                       \n"
      "1:     ld1w    {z0.s}, p0/z, [%[a], %[i], lsl #2]    \n"
      "       ld1w    {z1.s}, p0/z, [%[b], %[i], lsl #2]    \n"
      "       fadd    z0.s, p0/m, z0.s, z1.s                \n"
      "       st1w    {z0.s}, p0, [%[c], %[i], lsl #2]      \n"
      "       incw    %[i]                                  \n"
      "       whilelo p0.s, %[i], %[n]                      \n"
      "       b.any   1b                                    \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [c] "r"(c), [n] "r"(n)
      : "v0", "v1", "p0", "memory", "cc");
}
#elif defined(__ARM_NEON)
static void inner_loop_035(struct loop_035_data *restrict input) {
  float *restrict a = input->a;
  float *restrict b = input->b;
  float *restrict c = input->c;
  int64_t n = input->n;

  int64_t lmt = n - (n % 16);
  int64_t offset;

  asm volatile(
      "       mov     %[off], xzr             \n"
      "1:     ldr     q0, [%[a], %[off]]      \n"
      "       ldr     q1, [%[b], %[off]]      \n"
      "       fadd    v0.4s, v0.4s, v1.4s     \n"
      "       str     q0, [%[c], %[off]]      \n"
      "       add     %[off], %[off], #16     \n"
      "       cmp     %[off], %[lmt]          \n"
      "       b.ne    1b                      \n"
      // output operands, source operands, and clobber list
      : [off] "=&r"(offset)
      : [a] "r"(a), [b] "r"(b), [c] "r"(c), [lmt] "r"(4 * lmt)
      : "v0", "v1", "memory", "cc");

  for (int i = lmt; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}
#else
static void inner_loop_035(struct loop_035_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(035, SC_SVE_LOOP_ATTR)
{
  struct loop_035_data data = { .n = SIZE };

  ALLOC_64B(data.a, SIZE, "1st operand array");
  ALLOC_64B(data.b, SIZE, "2nd operand array");
  ALLOC_64B(data.c, SIZE, "result buffer");

  fill_float(data.a, SIZE);
  fill_float(data.b, SIZE);
  fill_float(data.c, SIZE);

  inner_loops_035(iters, &data);

  float res = 0.0f;
  for (int64_t i = 0; i < SIZE; i++) {
    res += i * data.c[i];
  }

  bool passed = check_float(res, 49439168.0f, 10.0f);
#ifndef STANDALONE
  FINALISE_LOOP_F(35, passed, "%9.6f", 49439168.0f, 10.0, res)
#endif
  return passed ? 0 : 1;
}
