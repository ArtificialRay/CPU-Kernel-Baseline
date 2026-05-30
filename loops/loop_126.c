/*----------------------------------------------------------------------------
#
#   Loop 126: conditional update
#
#   Purpose:
#     Use of simd loop with conditional update.
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


struct loop_126_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  int n;
  uint32_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_126(struct loop_126_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_126(struct loop_126_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  uint32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
    if (res % 2) {
      res++;
    }
  }
  data->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_126(struct loop_126_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  int pad = get_sve_vl() <= 512 ? 64 : 256;
  uint32_t *lmt = a + (n - (n % pad));
  uint32_t res = 0;
  svuint32_t res_vec = svdup_u32(res);
  svuint32_t tmp_vec = svdup_u32(0);
  svuint32_t pred_vec = svdup_u32(1);
  while (a < lmt) {
    svuint32_t a_vec = svld1(svptrue_b32(), a);
    svuint32_t b_vec = svld1(svptrue_b32(), b);
    res_vec = svmla_x(svptrue_b32(), res_vec, a_vec, b_vec);
    tmp_vec = svand_x(svptrue_b32(), res_vec, pred_vec);
    res_vec = svadd_x(svptrue_b32(), res_vec, tmp_vec);
    a += svcntw();
    b += svcntw();
  }
  res = svaddv(svptrue_b32(), res_vec);
  for (int i = 0; i < (n % pad); i++) {
    res += a[i] * b[i];
    if (res % 2) {
      res++;
    }
  }
  data->res = res;
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_126(struct loop_126_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint64_t n = data->n;

  uint32_t res;
  uint64_t i = 0;

  asm volatile(
      "   ptrue   p0.s                                      \n"
      "   dup     z18.s, #1                                 \n"
      "   mov     z10.s, #0                                 \n"
      "   mov     z11.s, #0                                 \n"
      "   mov     z12.s, #0                                 \n"
      "   mov     z13.s, #0                                 \n"
      "   mov     z14.s, #0                                 \n"
      "   mov     z15.s, #0                                 \n"
      "   mov     z16.s, #0                                 \n"
      "   mov     z17.s, #0                                 \n"
      "   mov     %[i], #0                                  \n"
      "   whilelt pn8.s, %[i], %[n], vlx2                   \n"
      "   b.none  2f                                        \n"
      "1:                                                   \n"
      "   ld1w    {z0.s-z3.s}, pn8/z, [%[a], %[i], lsl #2]  \n"
      "   ld1w    {z4.s-z7.s}, pn8/z, [%[b], %[i], lsl #2]  \n"
      "   incw    %[i], all, mul #4                         \n"
      "   mla     z10.s, p0/m, z0.s, z4.s                   \n"
      "   mla     z11.s, p0/m, z1.s, z5.s                   \n"
      "   mla     z12.s, p0/m, z2.s, z6.s                   \n"
      "   mla     z13.s, p0/m, z3.s, z7.s                   \n"
      "   and     z14.d, z10.d, z18.d                       \n"
      "   and     z15.d, z11.d, z18.d                       \n"
      "   and     z16.d, z12.d, z18.d                       \n"
      "   and     z17.d, z13.d, z18.d                       \n"
      "   add     z10.s, z10.s, z14.s                       \n"
      "   add     z11.s, z11.s, z15.s                       \n"
      "   add     z12.s, z12.s, z16.s                       \n"
      "   add     z13.s, z13.s, z17.s                       \n"
      "   whilelt pn8.s, %[i], %[n], vlx2                   \n"
      "   b.first 1b                                        \n"
      "2:                                                   \n"
      "   add     z10.s, z10.s, z11.s                       \n"
      "   add     z12.s, z12.s, z13.s                       \n"
      "   add     z18.s, z10.s, z12.s                       \n"
      "   uaddv   %d[res], p0, z18.s                        \n"
      : [res] "=&w"(res), [a] "+&r"(a), [b] "+&r"(b), [i] "+&r"(i)
      : [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "p0", "p8", "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE2)
static void inner_loop_126(struct loop_126_data *restrict data)
LOOP_ATTR
{
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  int pad = get_sve_vl() <= 512 ? 64 : 256;
  uint32_t *lmt = a + (n - (n % pad));
  uint32_t res = 0;

  asm volatile(
      "       ptrue   p0.s                                  \n"
      "       mov     z10.s, #0                             \n"
      "       mov     z11.s, #0                             \n"
      "       mov     z12.s, #0                             \n"
      "       mov     z13.s, #0                             \n"
      "       dup     z14.s, #1                             \n"
      "       dup     z15.s, #1                             \n"
      "       dup     z16.s, #1                             \n"
      "       dup     z17.s, #1                             \n"
      "       mov     z18.s, #0                             \n"
      "       mov     z19.s, #0                             \n"
      "       mov     z20.s, #0                             \n"
      "       mov     z21.s, #0                             \n"

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

      "       mla    z10.s, p0/m, z1.s, z5.s                \n"
      "       mla    z11.s, p0/m, z2.s, z6.s                \n"
      "       mla    z12.s, p0/m, z3.s, z7.s                \n"
      "       mla    z13.s, p0/m, z4.s, z8.s                \n"
      "       and    z18.d, z14.d, z10.d                    \n"
      "       and    z19.d, z15.d, z11.d                    \n"
      "       and    z20.d, z16.d, z12.d                    \n"
      "       and    z21.d, z17.d, z13.d                    \n"
      "       add    z10.s, z10.s, z18.s                    \n"
      "       add    z11.s, z11.s, z19.s                    \n"
      "       add    z12.s, z12.s, z20.s                    \n"
      "       add    z13.s, z13.s, z21.s                    \n"

      "2:     cmp     %[a], %[lmt]                          \n"
      "       b.lt    1b                                    \n"  // loop back

      "       add     z10.s, z10.s, z11.s                   \n"
      "       add     z12.s, z12.s, z13.s                   \n"
      "       add     z0.s,  z10.s, z12.s                   \n"
      "       uaddv   %d[res], p0, z0.s                     \n"
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "p0",
        "cc", "memory");

  for (int i = 0; i < (n % pad); i++) {
    res += a[i] * b[i];
    if (res % 2) {
      res++;
    }
  }
  data->res = res;
}
#elif defined(__ARM_NEON)
static void inner_loop_126(struct loop_126_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  uint32_t *lmt = a + (n - (n % 16));
  uint32_t res = 0;

  asm volatile(
      "       movi    v10.4s, #0                          \n"
      "       movi    v11.4s, #0                          \n"
      "       movi    v12.4s, #0                          \n"
      "       movi    v13.4s, #0                          \n"
      "       movi    v14.4s, #1                          \n"
      "       movi    v15.4s, #1                          \n"
      "       movi    v16.4s, #1                          \n"
      "       movi    v17.4s, #1                          \n"
      "       movi    v18.4s, #0                          \n"
      "       movi    v19.4s, #0                          \n"
      "       movi    v20.4s, #0                          \n"
      "       movi    v21.4s, #0                          \n"
      "       b       2f                                  \n"

      "1:     ldp     q1, q2, [%[a]]                      \n"
      "       ldp     q5, q6, [%[b]]                      \n"
      "       ldp     q3, q4, [%[a], #32]                 \n"
      "       ldp     q7, q8, [%[b], #32]                 \n"

      "       add     %[a], %[a], #64                     \n"
      "       add     %[b], %[b], #64                     \n"

      "       mla     v10.4s, v1.4s, v5.4s                \n"
      "       mla     v11.4s, v2.4s, v6.4s                \n"
      "       mla     v12.4s, v3.4s, v7.4s                \n"
      "       mla     v13.4s, v4.4s, v8.4s                \n"
      "       and     v18.16b, v14.16b, v10.16b           \n"
      "       and     v19.16b, v15.16b, v11.16b           \n"
      "       and     v20.16b, v16.16b, v12.16b           \n"
      "       and     v21.16b, v17.16b, v13.16b           \n"
      "       add     v10.4s, v10.4s, v18.4s              \n"
      "       add     v11.4s, v11.4s, v19.4s              \n"
      "       add     v12.4s, v12.4s, v20.4s              \n"
      "       add     v13.4s, v13.4s, v21.4s              \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       add     v10.4s, v10.4s, v11.4s              \n"
      "       add     v12.4s, v12.4s, v13.4s              \n"
      "       add     v1.4s,  v10.4s, v12.4s              \n"
      "       addp    v1.4s, v1.4s, v1.4s                 \n"
      "       addp    v1.4s, v1.4s, v1.4s                 \n"
      "       fmov    %w[res], s1                         \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "cc", "memory");

  for (int i = 0; i < (n % 16); i++) {
    res += a[i] * b[i];
    if (res % 2) {
      res++;
    }
  }
  data->res = res;
}
#else
static void inner_loop_126(struct loop_126_data *restrict data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(126, SC_SVE_LOOP_ATTR)
{
  struct loop_126_data data = { .n = SIZE, .res = 0, };

  ALLOC_64B(data.a, SIZE, "1st operand array");
  ALLOC_64B(data.b, SIZE, "2nd operand array");

  fill_uint32(data.a, SIZE);
  fill_uint32(data.b, SIZE);

  for (int i = 0; i < SIZE; i++) {
    data.a[i] %= 100;
    data.b[i] %= 100;
  }

  inner_loops_126(iters, &data);

  uint32_t res = data.res;
  bool passed = res == 0x01761d22;
#ifndef STANDALONE
  FINALISE_LOOP_I(126, passed, "0x%08"PRIx32, 0x01761d22, res)
#endif
  return passed ? 0 : 1;
}
