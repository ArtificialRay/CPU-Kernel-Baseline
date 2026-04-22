/*----------------------------------------------------------------------------
#
#   Loop 003: FP64 inner product
#
#   Purpose:
#     Use of fp64 MLA instruction.
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


struct loop_003_data {
  double *restrict a;
  double *restrict b;
  int n;
  double res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_003(struct loop_003_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_003(struct loop_003_data *restrict data) {
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_003(struct loop_003_data *restrict data)
LOOP_ATTR
{
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  double res = 0;
  svfloat64_t res_vec = svdup_f64(res);
  svbool_t p;
  FOR_LOOP_64(int32_t, i, 0, n, p) {
    svfloat64_t a_vec = svld1(p, a + i);
    svfloat64_t b_vec = svld1(p, b + i);
    res_vec = svmla_m(p, res_vec, a_vec, b_vec);
  }
  data->res = svaddv(svptrue_b64(), res_vec);
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_003(struct loop_003_data *restrict data)
LOOP_ATTR
{
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  double res = 0.0;
  int i = 0;
  asm volatile(
      "   ptrue   p0.d                                        \n"
      "   mov     z10.d, #0                                   \n"
      "   mov     z11.d, #0                                   \n"
      "   mov     z12.d, #0                                   \n"
      "   mov     z13.d, #0                                   \n"
      "   whilelt pn8.d, %x[i], %x[n], vlx4                   \n"
      "   b.none  2f                                          \n"
      "1:                                                     \n"
      "   ld1d    {z0.d-z3.d}, pn8/z, [%[a], %x[i], lsl #3]   \n"
      "   ld1d    {z4.d-z7.d}, pn8/z, [%[b], %x[i], lsl #3]   \n"
      "   incd    %x[i], all, mul #4                          \n"
      "   fmla    z10.d, p0/m, z0.d, z4.d                     \n"
      "   fmla    z11.d, p0/m, z1.d, z5.d                     \n"
      "   fmla    z12.d, p0/m, z2.d, z6.d                     \n"
      "   fmla    z13.d, p0/m, z3.d, z7.d                     \n"
      "   whilelt pn8.d, %x[i], %x[n], vlx4                   \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"
      "   fadd    z10.d, z10.d, z11.d                         \n"
      "   fadd    z12.d, z12.d, z13.d                         \n"
      "   fadd    z10.d, z10.d, z12.d                         \n"
      "   faddv   %d[res], p0, z10.d                          \n"
      : [res] "=&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "p0", "p8", "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_003(struct loop_003_data *restrict data)
LOOP_ATTR
{
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  int pad = get_sve_vl() <= 512 ? 32 : 128;
  double *lmt = a + (n - (n % pad));
  double res = 0.0;

  asm volatile(
      "       ptrue   p0.d                                  \n"
      "       mov     z10.d, #0                             \n"
      "       mov     z11.d, #0                             \n"
      "       mov     z12.d, #0                             \n"
      "       mov     z13.d, #0                             \n"
      "       b       2f                                    \n"

      "1:     ld1d    {z1.d}, p0/z, [%[a]]                  \n"
      "       ld1d    {z5.d}, p0/z, [%[b]]                  \n"
      "       ld1d    {z2.d}, p0/z, [%[a], #1, mul vl]      \n"
      "       ld1d    {z6.d}, p0/z, [%[b], #1, mul vl]      \n"
      "       ld1d    {z3.d}, p0/z, [%[a], #2, mul vl]      \n"
      "       ld1d    {z7.d}, p0/z, [%[b], #2, mul vl]      \n"
      "       ld1d    {z4.d}, p0/z, [%[a], #3, mul vl]      \n"
      "       ld1d    {z8.d}, p0/z, [%[b], #3, mul vl]      \n"

      "       incb    %[a], all, mul #4                     \n"
      "       incb    %[b], all, mul #4                     \n"

      "       fmla    z10.d, p0/m, z1.d, z5.d               \n"
      "       fmla    z11.d, p0/m, z2.d, z6.d               \n"
      "       fmla    z12.d, p0/m, z3.d, z7.d               \n"
      "       fmla    z13.d, p0/m, z4.d, z8.d               \n"

      "2:     cmp     %[a], %[lmt]                          \n"
      "       b.lt    1b                                    \n"  // loop back

      "       fadd    z10.d, z10.d, z11.d                   \n"
      "       fadd    z12.d, z12.d, z13.d                   \n"
      "       fadd    z1.d,  z10.d, z12.d                   \n"
      "       faddv   %d[res], p0, z1.d                     \n"
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
static void inner_loop_003(struct loop_003_data *restrict data) {
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  double *lmt = a + (n - (n % 8));
  double res = 0.0;

  asm volatile(
      "       movi    v10.16b, #0                         \n"
      "       movi    v11.16b, #0                         \n"
      "       movi    v12.16b, #0                         \n"
      "       movi    v13.16b, #0                         \n"
      "       b       2f                                  \n"

      "1:     ldp     q1, q2, [%[a]]                      \n"
      "       ldp     q5, q6, [%[b]]                      \n"
      "       ldp     q3, q4, [%[a], #32]                 \n"
      "       ldp     q7, q8, [%[b], #32]                 \n"

      "       add     %[a], %[a], #64                     \n"
      "       add     %[b], %[b], #64                     \n"

      "       fmla    v10.2d, v1.2d, v5.2d                \n"
      "       fmla    v11.2d, v2.2d, v6.2d                \n"
      "       fmla    v12.2d, v3.2d, v7.2d                \n"
      "       fmla    v13.2d, v4.2d, v8.2d                \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       fadd    v10.2d, v10.2d, v11.2d              \n"
      "       fadd    v12.2d, v12.2d, v13.2d              \n"
      "       fadd    v1.2d,  v10.2d, v12.2d              \n"
      "       faddp   v1.2d, v1.2d, v1.2d                 \n"
      "       fmov    %[res], d1                          \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12",
        "v13", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_003(struct loop_003_data *restrict data) {
  double *restrict a = data->a;
  double *restrict b = data->b;
  int n = data->n;

  double *lmt = a + (n - (n % 8));
  double res = 0.0;

  asm volatile(
      "       fmov    d17, xzr                            \n"
      "       fmov    d18, xzr                            \n"
      "       fmov    d19, xzr                            \n"
      "       fmov    d20, xzr                            \n"
      "       fmov    d21, xzr                            \n"
      "       fmov    d22, xzr                            \n"
      "       fmov    d23, xzr                            \n"
      "       fmov    d24, xzr                            \n"
      "       b       2f                                  \n"

      "1:     ldp     d1,  d2,  [%[a]]                    \n"
      "       ldp     d3,  d4,  [%[a], #16]               \n"
      "       ldp     d9,  d10, [%[b]]                    \n"
      "       ldp     d11, d12, [%[b], #16]               \n"
      "       ldp     d5,  d6,  [%[a], #32]               \n"
      "       ldp     d7,  d8,  [%[a], #48]               \n"
      "       ldp     d13, d14, [%[b], #32]               \n"
      "       ldp     d15, d16, [%[b], #48]               \n"

      "       add     %[a], %[a], #64                     \n"
      "       add     %[b], %[b], #64                     \n"

      "       fmadd   d17, d1, d9, d17                    \n"
      "       fmadd   d18, d2, d10, d18                   \n"
      "       fmadd   d19, d3, d11, d19                   \n"
      "       fmadd   d20, d4, d12, d20                   \n"
      "       fmadd   d21, d5, d13, d21                   \n"
      "       fmadd   d22, d6, d14, d22                   \n"
      "       fmadd   d23, d7, d15, d23                   \n"
      "       fmadd   d24, d8, d16, d24                   \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       fadd    d17, d17, d18                       \n"
      "       fadd    d19, d19, d20                       \n"
      "       fadd    d21, d21, d22                       \n"
      "       fadd    d23, d23, d24                       \n"
      "       fadd    d17, d17, d19                       \n"
      "       fadd    d21, d21, d23                       \n"
      "       fadd    %d[res], d17, d21                   \n"
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
static void inner_loop_003(struct loop_003_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 5000
#endif

LOOP_DECL(003, SC_SVE_LOOP_ATTR)
{
  struct loop_003_data data = { .n = SIZE, .res = DBL_MAX, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_double(data.a, SIZE);
  fill_double(data.b, SIZE);

  inner_loops_003(iters, &data);

  double res = data.res;
  bool passed = check_double(res, 1246.8, 0.1);
#ifndef STANDALONE
  FINALISE_LOOP_F(3, passed, "%9.6f", 1246.8, 0.1, res)
#endif
  return passed ? 0 : 1;
}
