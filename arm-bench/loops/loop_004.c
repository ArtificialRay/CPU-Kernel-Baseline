/*----------------------------------------------------------------------------
#
#   Loop 004: UINT64 inner product
#
#   Purpose:
#     Use of u64 MLA instruction.
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


struct loop_004_data {
  uint64_t *restrict a;
  uint64_t *restrict b;
  int n;
  uint64_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_004(struct loop_004_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_004(struct loop_004_data *restrict data) {
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  uint64_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_004(struct loop_004_data *restrict data)
LOOP_ATTR
{
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  uint64_t res = 0;
  svuint64_t res_vec = svdup_u64(res);
  svbool_t p;
  FOR_LOOP_64(int32_t, i, 0, n, p) {
    svuint64_t a_vec = svld1(p, a + i);
    svuint64_t b_vec = svld1(p, b + i);
    res_vec = svmla_m(p, res_vec, a_vec, b_vec);
  }
  data->res = svaddv(svptrue_b64(), res_vec);
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_004(struct loop_004_data *restrict data)
LOOP_ATTR
{
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  uint64_t res = 0;
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
      "   mla     z10.d, p0/m, z0.d, z4.d                     \n"
      "   mla     z11.d, p0/m, z1.d, z5.d                     \n"
      "   mla     z12.d, p0/m, z2.d, z6.d                     \n"
      "   mla     z13.d, p0/m, z3.d, z7.d                     \n"
      "   whilelt pn8.d, %x[i], %x[n], vlx4                   \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"
      "   add     z10.d, z10.d, z11.d                         \n"
      "   add     z12.d, z12.d, z13.d                         \n"
      "   add     z10.d, z10.d, z12.d                         \n"
      "   uaddv   %d[res], p0, z10.d                          \n"
      : [res] "=&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "p0", "p8", "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_004(struct loop_004_data *restrict data)
LOOP_ATTR
{
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  int pad = get_sve_vl() <= 512 ? 32 : 128;
  uint64_t *lmt = a + (n - (n % pad));
  uint64_t res = 0;

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

      "       mla     z10.d, p0/m, z1.d, z5.d               \n"
      "       mla     z11.d, p0/m, z2.d, z6.d               \n"
      "       mla     z12.d, p0/m, z3.d, z7.d               \n"
      "       mla     z13.d, p0/m, z4.d, z8.d               \n"

      "2:     cmp     %[a], %[lmt]                          \n"
      "       b.lt    1b                                    \n"  // loop back

      "       add     z10.d, z10.d, z11.d                   \n"
      "       add     z12.d, z12.d, z13.d                   \n"
      "       add     z1.d,  z10.d, z12.d                   \n"
      "       uaddv   %d[res], p0, z1.d                     \n"
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
// No 64-bit integer MLA in Neon
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_004(struct loop_004_data *restrict data) {
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  uint64_t *lmt = a + (n - (n % 8));
  uint64_t res = 0;

  asm volatile(
      "       mov     x17, xzr                            \n"
      "       mov     x26, xzr                            \n"
      "       mov     x19, xzr                            \n"
      "       mov     x20, xzr                            \n"
      "       mov     x21, xzr                            \n"
      "       mov     x22, xzr                            \n"
      "       mov     x23, xzr                            \n"
      "       mov     x24, xzr                            \n"
      "       b       2f                                  \n"

      "1:     ldp     x1,  x2,  [%[a]]                    \n"
      "       ldp     x3,  x4,  [%[a], #16]               \n"
      "       ldp     x9,  x10, [%[b]]                    \n"
      "       ldp     x11, x12, [%[b], #16]               \n"
      "       ldp     x5,  x6,  [%[a], #32]               \n"
      "       ldp     x7,  x8,  [%[a], #48]               \n"
      "       ldp     x13, x25, [%[b], #32]               \n"
      "       ldp     x15, x16, [%[b], #48]               \n"

      "       add     %[a], %[a], #64                     \n"
      "       add     %[b], %[b], #64                     \n"

      "       madd    x17, x1, x9, x17                    \n"
      "       madd    x26, x2, x10, x26                   \n"
      "       madd    x19, x3, x11, x19                   \n"
      "       madd    x20, x4, x12, x20                   \n"
      "       madd    x21, x5, x13, x21                   \n"
      "       madd    x22, x6, x25, x22                   \n"
      "       madd    x23, x7, x15, x23                   \n"
      "       madd    x24, x8, x16, x24                   \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       add     x17, x17, x26                       \n"
      "       add     x19, x19, x20                       \n"
      "       add     x21, x21, x22                       \n"
      "       add     x23, x23, x24                       \n"
      "       add     x17, x17, x19                       \n"
      "       add     x21, x21, x23                       \n"
      "       add     %[res], x17, x21                    \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
        "x12", "x13", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23",
        "x24", "x25", "x26", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i] * b[i];
  }
  data->res = res;
}
#else
static void inner_loop_004(struct loop_004_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 5000
#endif

LOOP_DECL(004, SC_SVE_LOOP_ATTR)
{
  struct loop_004_data data = { .n = SIZE, .res = 0, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_uint64(data.a, SIZE);
  fill_uint64(data.b, SIZE);

  inner_loops_004(iters, &data);

  uint64_t res = data.res;
  uint64_t correct = 0x94e0f14909c9e5faL;
  bool passed = res == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(4, passed, "0x%016"PRIx64, correct,  res)
#endif
  return passed ? 0 : 1;
}
