/*----------------------------------------------------------------------------
#
#   Loop 001: FP32 inner product
#
#   Purpose:
#     Use of fp32 MLA instruction.
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


struct loop_001_data {
  float *restrict a;
  float *restrict b;
  int n;
  float res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_001(struct loop_001_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_001(struct loop_001_data *restrict data) {
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  float res = 0.0f;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_001(struct loop_001_data *restrict data)
LOOP_ATTR
{
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  svfloat32_t res_vec = svdup_f32(0);
  svbool_t p;
  FOR_LOOP_32(int32_t, i, 0, n, p) {
    svfloat32_t a_vec = svld1(p, a + i);
    svfloat32_t b_vec = svld1(p, b + i);
    res_vec = svmla_m(p, res_vec, a_vec, b_vec);
  }

  data->res = svaddv(svptrue_b32(), res_vec);
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_001(struct loop_001_data *restrict data)
LOOP_ATTR
{
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  float res = 0.0f;
  int i = 0;
  asm volatile(
      "   ptrue   p0.s                                        \n"
      "   mov     z10.s, #0                                   \n"
      "   mov     z11.s, #0                                   \n"
      "   mov     z12.s, #0                                   \n"
      "   mov     z13.s, #0                                   \n"
      "   whilelt pn8.s, %x[i], %x[n], vlx4                   \n"
      "   b.none  2f                                          \n"
      "1:                                                     \n"
      "   ld1w    {z0.s-z3.s}, pn8/z, [%[a], %x[i], lsl #2]   \n"
      "   ld1w    {z4.s-z7.s}, pn8/z, [%[b], %x[i], lsl #2]   \n"
      "   incw    %x[i], all, mul #4                          \n"
      "   fmla    z10.s, p0/m, z0.s, z4.s                     \n"
      "   fmla    z11.s, p0/m, z1.s, z5.s                     \n"
      "   fmla    z12.s, p0/m, z2.s, z6.s                     \n"
      "   fmla    z13.s, p0/m, z3.s, z7.s                     \n"
      "   whilelt pn8.s, %x[i], %x[n], vlx4                   \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"
      "   fadd    z10.s, z10.s, z11.s                         \n"
      "   fadd    z12.s, z12.s, z13.s                         \n"
      "   fadd    z10.s, z10.s, z12.s                         \n"
      "   faddv   %s[res], p0, z10.s                          \n"
      : [res] "=&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "p0", "p8", "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_001(struct loop_001_data *restrict data)
LOOP_ATTR
{
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  int pad = get_sve_vl() <= 512 ? 64 : 256;
  float *lmt = a + (n - (n % pad));
  float res = 0.0;

  asm volatile(
      "       ptrue   p0.s                                  \n"
      "       mov     z10.s, #0                             \n"
      "       mov     z11.s, #0                             \n"
      "       mov     z12.s, #0                             \n"
      "       mov     z13.s, #0                             \n"
      "       b       2f                                    \n"

      "1:     ld1w    {z1.s}, p0/z, [%[a]]                  \n"
      "       ld1w    {z5.s}, p0/z, [%[b]]                  \n"
      "       ld1w    {z2.s}, p0/z, [%[a], #1, mul vl]      \n"
      "       ld1w    {z6.s}, p0/z, [%[b], #1, mul vl]      \n"
      "       ld1w    {z3.s}, p0/z, [%[a], #2, mul vl]      \n"
      "       ld1w    {z7.s}, p0/z, [%[b], #2, mul vl]      \n"
      "       ld1w    {z4.s}, p0/z, [%[a], #3, mul vl]      \n"
      "       ld1w    {z8.s}, p0/z, [%[b], #3, mul vl]      \n"

      "       incb    %[a], all, mul #4                     \n"
      "       incb    %[b], all, mul #4                     \n"

      "       fmla    z10.s, p0/m, z1.s, z5.s               \n"
      "       fmla    z11.s, p0/m, z2.s, z6.s               \n"
      "       fmla    z12.s, p0/m, z3.s, z7.s               \n"
      "       fmla    z13.s, p0/m, z4.s, z8.s               \n"

      "2:     cmp     %[a], %[lmt]                          \n"
      "       b.lt    1b                                    \n"  // loop back

      "       fadd    z10.s, z10.s, z11.s                   \n"
      "       fadd    z12.s, z12.s, z13.s                   \n"
      "       fadd    z1.s,  z10.s, z12.s                   \n"
      "       faddv   %s[res], p0, z1.s                     \n"
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12",
        "v13", "p0", "cc", "memory");

  for (int i = 0; i < (n % pad); i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif defined(__ARM_NEON)
static void inner_loop_001(struct loop_001_data *restrict data) {
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  float *lmt = a + (n - (n % 16));
  float res = 0.0;

  asm volatile(
      "       movi    v10.4s, #0                          \n"
      "       movi    v11.4s, #0                          \n"
      "       movi    v12.4s, #0                          \n"
      "       movi    v13.4s, #0                          \n"
      "       b       2f                                  \n"

      "1:     ldp     q1, q2, [%[a]]                      \n"
      "       ldp     q5, q6, [%[b]]                      \n"
      "       ldp     q3, q4, [%[a], #32]                 \n"
      "       ldp     q7, q8, [%[b], #32]                 \n"

      "       add     %[a], %[a], #64                     \n"
      "       add     %[b], %[b], #64                     \n"

      "       fmla    v10.4s, v1.4s, v5.4s                \n"
      "       fmla    v11.4s, v2.4s, v6.4s                \n"
      "       fmla    v12.4s, v3.4s, v7.4s                \n"
      "       fmla    v13.4s, v4.4s, v8.4s                \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       fadd    v10.4s, v10.4s, v11.4s              \n"
      "       fadd    v12.4s, v12.4s, v13.4s              \n"
      "       fadd    v1.4s,  v10.4s, v12.4s              \n"
      "       faddp   v1.4s, v1.4s, v1.4s                 \n"
      "       faddp   v1.4s, v1.4s, v1.4s                 \n"
      "       fmov    %w[res], s1                         \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12",
        "v13", "cc", "memory");

  for (int i = 0; i < (n % 16); i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_001(struct loop_001_data *restrict data) {
  float *a = data->a;
  float *b = data->b;
  int n = data->n;

  float *lmt = a + (n - (n % 8));
  float res = 0.0f;

  asm volatile(
      "       fmov    s17, wzr                            \n"
      "       fmov    s18, wzr                            \n"
      "       fmov    s19, wzr                            \n"
      "       fmov    s20, wzr                            \n"
      "       fmov    s21, wzr                            \n"
      "       fmov    s22, wzr                            \n"
      "       fmov    s23, wzr                            \n"
      "       fmov    s24, wzr                            \n"
      "       b       2f                                  \n"

      "1:     ldp     s1,  s2,  [%[a]]                    \n"
      "       ldp     s3,  s4,  [%[a], #8]                \n"
      "       ldp     s9,  s10, [%[b]]                    \n"
      "       ldp     s11, s12, [%[b], #8]                \n"
      "       ldp     s5,  s6,  [%[a], #16]               \n"
      "       ldp     s7,  s8,  [%[a], #24]               \n"
      "       ldp     s13, s14, [%[b], #16]               \n"
      "       ldp     s15, s16, [%[b], #24]               \n"

      "       add     %[a], %[a], #32                     \n"
      "       add     %[b], %[b], #32                     \n"

      "       fmadd   s17, s1, s9, s17                    \n"
      "       fmadd   s18, s2, s10, s18                   \n"
      "       fmadd   s19, s3, s11, s19                   \n"
      "       fmadd   s20, s4, s12, s20                   \n"
      "       fmadd   s21, s5, s13, s21                   \n"
      "       fmadd   s22, s6, s14, s22                   \n"
      "       fmadd   s23, s7, s15, s23                   \n"
      "       fmadd   s24, s8, s16, s24                   \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       fadd    s17, s17, s18                       \n"
      "       fadd    s19, s19, s20                       \n"
      "       fadd    s21, s21, s22                       \n"
      "       fadd    s23, s23, s24                       \n"
      "       fadd    s17, s17, s19                       \n"
      "       fadd    s21, s21, s23                       \n"
      "       fadd    %s[res], s17, s21                   \n"
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v24", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#else
static void inner_loop_001(struct loop_001_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(001, SC_SVE_LOOP_ATTR)
{
  struct loop_001_data data = { .n = SIZE, .res = FLT_MAX, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_float(data.a, SIZE);
  fill_float(data.b, SIZE);

  inner_loops_001(iters, &data);

  float res = data.res;
  bool passed = check_float(res, 2448.1f, 0.1f);
#ifndef STANDALONE
  FINALISE_LOOP_F(1, passed, "%9.6f", 2448.1f, 0.1f, res)
#endif
  return passed ? 0 : 1;
}
