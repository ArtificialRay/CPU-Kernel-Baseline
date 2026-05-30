/*----------------------------------------------------------------------------
#
#   Loop 002: UINT32 inner product
#
#   Purpose:
#     Use of u32 MLA instruction.
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


struct loop_002_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  int n;
  uint32_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_002(struct loop_002_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_002(struct loop_002_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
  input->res = res;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_002(struct loop_002_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t res = 0;
  svuint32_t res_vec = svdup_u32(res);
  svbool_t p;
  FOR_LOOP_32(int32_t, i, 0, n, p) {
    svuint32_t a_vec = svld1(p, a + i);
    svuint32_t b_vec = svld1(p, b + i);
    res_vec = svmla_m(p, res_vec, a_vec, b_vec);
  }

  input->res = svaddv(svptrue_b32(), res_vec);
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_002(struct loop_002_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t res = 0;
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
      "   mla     z10.s, p0/m, z0.s, z4.s                     \n"
      "   mla     z11.s, p0/m, z1.s, z5.s                     \n"
      "   mla     z12.s, p0/m, z2.s, z6.s                     \n"
      "   mla     z13.s, p0/m, z3.s, z7.s                     \n"
      "   whilelt pn8.s, %x[i], %x[n], vlx4                   \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"
      "   add     z10.s, z10.s, z11.s                         \n"
      "   add     z12.s, z12.s, z13.s                         \n"
      "   add     z10.s, z10.s, z12.s                         \n"
      "   uaddv   %d[res], p0, z10.s                          \n"
      : [res] "=&w"(res), [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "p0", "p8", "cc", "memory");

  input->res = res;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_002(struct loop_002_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  int pad = get_sve_vl() <= 512 ? 64 : 256;
  uint32_t *lmt = a + (n - (n % pad));
  uint32_t res = 0;

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

      "       mla    z10.s, p0/m, z1.s, z5.s               \n"
      "       mla    z11.s, p0/m, z2.s, z6.s               \n"
      "       mla    z12.s, p0/m, z3.s, z7.s               \n"
      "       mla    z13.s, p0/m, z4.s, z8.s               \n"

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
        "v13", "p0", "cc", "memory");

  for (int i = 0; i < (n % pad); i++) {
    res += a[i] * b[i];
  }
  input->res = res;
}
#elif defined(__ARM_NEON)
static void inner_loop_002(struct loop_002_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t *lmt = a + (n - (n % 16));
  uint32_t res = 0;

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

      "       mla     v10.4s, v1.4s, v5.4s                \n"
      "       mla     v11.4s, v2.4s, v6.4s                \n"
      "       mla     v12.4s, v3.4s, v7.4s                \n"
      "       mla     v13.4s, v4.4s, v8.4s                \n"

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
        "v13", "cc", "memory");

  for (int i = 0; i < (n % 16); i++) {
    res += a[i] * b[i];
  }
  input->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_002(struct loop_002_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t *lmt = a + (n - (n % 8));
  uint32_t res = 0;

  asm volatile(
      "       mov     w17, wzr                            \n"
      "       mov     w26, wzr                            \n"
      "       mov     w19, wzr                            \n"
      "       mov     w20, wzr                            \n"
      "       mov     w21, wzr                            \n"
      "       mov     w22, wzr                            \n"
      "       mov     w23, wzr                            \n"
      "       mov     w24, wzr                            \n"
      "       b       2f                                  \n"

      "1:     ldp     w1,  w2,  [%[a]]                    \n"
      "       ldp     w3,  w4,  [%[a], #8]                \n"
      "       ldp     w9,  w10, [%[b]]                    \n"
      "       ldp     w11, w12, [%[b], #8]                \n"
      "       ldp     w5,  w6,  [%[a], #16]               \n"
      "       ldp     w7,  w8,  [%[a], #24]               \n"
      "       ldp     w13, w25, [%[b], #16]               \n"
      "       ldp     w15, w16, [%[b], #24]               \n"

      "       add     %[a], %[a], #32                     \n"
      "       add     %[b], %[b], #32                     \n"

      "       madd    w17, w1, w9, w17                    \n"
      "       madd    w26, w2, w10, w26                   \n"
      "       madd    w19, w3, w11, w19                   \n"
      "       madd    w20, w4, w12, w20                   \n"
      "       madd    w21, w5, w13, w21                   \n"
      "       madd    w22, w6, w25, w22                   \n"
      "       madd    w23, w7, w15, w23                   \n"
      "       madd    w24, w8, w16, w24                   \n"

      "2:     cmp     %[a], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back

      "       add     w17, w17, w26                       \n"
      "       add     w19, w19, w20                       \n"
      "       add     w21, w21, w22                       \n"
      "       add     w23, w23, w24                       \n"
      "       add     w17, w17, w19                       \n"
      "       add     w21, w21, w23                       \n"
      "       add     %w[res], w17, w21                   \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [a] "+&r"(a), [b] "+&r"(b)
      : [lmt] "r"(lmt)
      : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
        "x12", "x13", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23",
        "x24", "x25", "x26", "cc", "memory");

  for (int i = 0; i < (n % 8); i++) {
    res += a[i] * b[i];
  }
  input->res = res;
}
#else
static void inner_loop_002(struct loop_002_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(002, SC_SVE_LOOP_ATTR)
{
  struct loop_002_data data = { .n = SIZE, .res = 0, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_uint32(data.a, SIZE);
  fill_uint32(data.b, SIZE);

  for (int i = 0; i < SIZE; i++) {
    data.a[i] %= 100;
    data.b[i] %= 100;
  }

  inner_loops_002(iters, &data);

  uint32_t res = data.res;
  bool passed = res == 0x01761385;
#ifndef STANDALONE
  FINALISE_LOOP_I(2, passed, "0x%08"PRIx32, 0x01761385, res)
#endif
  return passed ? 0 : 1;
}
