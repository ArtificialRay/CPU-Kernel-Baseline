/*----------------------------------------------------------------------------
#
#   Loop 024: Sum of abs diffs
#
#   Purpose:
#     Use of DOT instruction.
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


struct loop_024_data {
  uint8_t *restrict a;
  uint8_t *restrict b;
  int64_t n;
  uint32_t res;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_024(struct loop_024_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_024(struct loop_024_data *restrict data) {
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  uint32_t sum = 0;
  for (int i = 0; i < n; i++) {
    sum += __builtin_abs(a[i] - b[i]);
  }
  data->res = sum;
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_024(struct loop_024_data *restrict data)
LOOP_ATTR
{
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  svuint32_t acc = svdup_u32(0);
  svuint8_t ones = svdup_u8(1);

  svbool_t p;
  FOR_LOOP_8(int64_t, i, 0, n, p) {
    svuint8_t a_vec = svld1(p, a + i);
    svuint8_t b_vec = svld1(p, b + i);
    svuint8_t udiff = svabd_x(p, a_vec, b_vec);
    acc = svdot(acc, udiff, ones);
  }
  data->res = svaddv(svptrue_b8(), acc);
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_024(struct loop_024_data *restrict data)
LOOP_ATTR
{
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  uint32_t sum = 0;
  int64_t i = 0;
  asm volatile(
      "   ptrue   p0.b                                  \n"
      "   mov     z14.b, #1                             \n"
      "   whilelt pn8.b, %[i], %[n], vlx4               \n"
      "   b.none  2f                                    \n"
      "   mov     z10.s, #0                             \n"
      "   mov     z11.s, #0                             \n"
      "   mov     z12.s, #0                             \n"
      "   mov     z13.s, #0                             \n"
      "1:                                               \n"
      "   ld1b    {z0.b-z3.b}, pn8/z, [%[a], %[i]]      \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[b], %[i]]      \n"
      "   uabd    z0.b, p0/m, z0.b, z4.b                \n"
      "   uabd    z1.b, p0/m, z1.b, z5.b                \n"
      "   uabd    z2.b, p0/m, z2.b, z6.b                \n"
      "   uabd    z3.b, p0/m, z3.b, z7.b                \n"
      "   incb    %[i], all, mul #4                     \n"
      "   udot    z10.s, z14.b, z0.b                    \n"
      "   udot    z11.s, z14.b, z1.b                    \n"
      "   udot    z12.s, z14.b, z2.b                    \n"
      "   udot    z13.s, z14.b, z3.b                    \n"
      "   whilelt pn8.b, %[i], %[n], vlx4               \n"
      "   b.first 1b                                    \n"
      "   add     z10.s, z10.s, z11.s                   \n"
      "   add     z12.s, z12.s, z13.s                   \n"
      "   add     z14.s, z10.s, z12.s                   \n"
      "2:                                               \n"
      "   uaddv   %d[sum], p0, z14.s                    \n"
      : [i] "+&r"(i), [sum] "=&w"(sum)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "p0", "p8", "cc", "memory");

  data->res = sum;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_024(struct loop_024_data *restrict data)
LOOP_ATTR
{
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  int64_t i = 0;
  int vl = get_sve_vl();
  int pad = vl <= 256 ? 128 : vl <= 512 ? 256 : 1024;
  int64_t lmt = n - (n % pad);
  uint8_t *a0 = a;
  uint8_t *a1 = a;
  uint8_t *a2 = a;
  uint8_t *a3 = a;
  uint8_t *b0 = b;
  uint8_t *b1 = b;
  uint8_t *b2 = b;
  uint8_t *b3 = b;
  uint32_t sum = 0;

  asm volatile(
      "       ptrue   p0.b                                  \n"
      "       mov     z0.b, #1                              \n"
      "       mov     z10.s, #0                             \n"
      "       mov     z11.s, #0                             \n"
      "       mov     z12.s, #0                             \n"
      "       mov     z13.s, #0                             \n"
      "       incb    %[a1]                                 \n"
      "       incb    %[a2], all, mul #2                    \n"
      "       incb    %[a3], all, mul #3                    \n"
      "       incb    %[b1]                                 \n"
      "       incb    %[b2], all, mul #2                    \n"
      "       incb    %[b3], all, mul #3                    \n"
      "       b       2f                                    \n"
      "1:     ld1b    {z1.b}, p0/z, [%[a0], %[i]]           \n"
      "       ld1b    {z2.b}, p0/z, [%[b0], %[i]]           \n"
      "       ld1b    {z3.b}, p0/z, [%[a1], %[i]]           \n"
      "       ld1b    {z4.b}, p0/z, [%[b1], %[i]]           \n"
      "       ld1b    {z5.b}, p0/z, [%[a2], %[i]]           \n"
      "       ld1b    {z6.b}, p0/z, [%[b2], %[i]]           \n"
      "       ld1b    {z7.b}, p0/z, [%[a3], %[i]]           \n"
      "       ld1b    {z8.b}, p0/z, [%[b3], %[i]]           \n"
      "       incb    %[i], all, mul #4                     \n"
      "       uabd    z1.b, p0/m, z1.b, z2.b                \n"
      "       uabd    z3.b, p0/m, z3.b, z4.b                \n"
      "       uabd    z5.b, p0/m, z5.b, z6.b                \n"
      "       uabd    z7.b, p0/m, z7.b, z8.b                \n"
      "       udot    z10.s, z1.b, z0.b                     \n"
      "       udot    z11.s, z3.b, z0.b                     \n"
      "       udot    z12.s, z5.b, z0.b                     \n"
      "       udot    z13.s, z7.b, z0.b                     \n"
      "2:     cmp     %[i], %[lmt]                          \n"
      "       b.lt    1b                                    \n"  // loop back
      "       add     z10.s, z10.s, z11.s                   \n"
      "       add     z12.s, z12.s, z13.s                   \n"
      "       add     z0.s,  z10.s, z12.s                   \n"
      "       uaddv   %d[sum], p0, z0.s                     \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [sum] "=&w"(sum), [a0] "+&r"(a0), [a1] "+&r"(a1),
        [a2] "+&r"(a2), [a3] "+&r"(a3), [b0] "+&r"(b0), [b1] "+&r"(b1),
        [b2] "+&r"(b2), [b3] "+&r"(b3)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11",
        "v12", "v13", "p0", "cc", "memory");

  for (; i < n; i++) {
    sum += __builtin_abs(a[i] - b[i]);
  }
  data->res = sum;
}
#elif defined(__ARM_NEON)
static void inner_loop_024(struct loop_024_data *restrict data) {
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  int64_t i = 0;
  int64_t lmt = n - (n % 128);
  uint8_t *a0 = a;
  uint8_t *a1 = a;
  uint8_t *a2 = a;
  uint8_t *a3 = a;
  uint8_t *b0 = b;
  uint8_t *b1 = b;
  uint8_t *b2 = b;
  uint8_t *b3 = b;
  uint32_t sum = 0;

  asm volatile(
      "       movi    v0.16b, #1                          \n"
      "       movi    v10.4s, #0                          \n"
      "       movi    v11.4s, #0                          \n"
      "       movi    v12.4s, #0                          \n"
      "       movi    v13.4s, #0                          \n"
      "       add     %[a1], %[a1], #16                   \n"
      "       add     %[a2], %[a2], #32                   \n"
      "       add     %[a3], %[a3], #48                   \n"
      "       add     %[b1], %[b1], #16                   \n"
      "       add     %[b2], %[b2], #32                   \n"
      "       add     %[b3], %[b3], #48                   \n"
      "       b       2f                                  \n"
      "1:     ldr     q1, [%[a0], %[i]]                   \n"
      "       ldr     q2, [%[b0], %[i]]                   \n"
      "       ldr     q3, [%[a1], %[i]]                   \n"
      "       ldr     q4, [%[b1], %[i]]                   \n"
      "       ldr     q5, [%[a2], %[i]]                   \n"
      "       ldr     q6, [%[b2], %[i]]                   \n"
      "       ldr     q7, [%[a3], %[i]]                   \n"
      "       ldr     q8, [%[b3], %[i]]                   \n"
      "       add     %[i], %[i], 64                      \n"
      "       uabd    v1.16b, v1.16b, v2.16b              \n"
      "       uabd    v3.16b, v3.16b, v4.16b              \n"
      "       uabd    v5.16b, v5.16b, v6.16b              \n"
      "       uabd    v7.16b, v7.16b, v8.16b              \n"
      "       udot    v10.4s, v1.16b, v0.16b              \n"
      "       udot    v11.4s, v3.16b, v0.16b              \n"
      "       udot    v12.4s, v5.16b, v0.16b              \n"
      "       udot    v13.4s, v7.16b, v0.16b              \n"
      "2:     cmp     %[i], %[lmt]                        \n"
      "       b.lt    1b                                  \n"  // loop back
      "       add     v10.4s, v10.4s, v11.4s              \n"
      "       add     v12.4s, v12.4s, v13.4s              \n"
      "       add     v0.4s,  v10.4s, v12.4s              \n"
      "       addp    v0.4s, v0.4s, v0.4s                 \n"
      "       addp    v0.4s, v0.4s, v0.4s                 \n"
      "       fmov    %w[sum], s0                         \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [sum] "=&r"(sum), [a0] "+&r"(a0), [a1] "+&r"(a1),
        [a2] "+&r"(a2), [a3] "+&r"(a3), [b0] "+&r"(b0), [b1] "+&r"(b1),
        [b2] "+&r"(b2), [b3] "+&r"(b3)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11",
        "v12", "v13", "cc", "memory");

  for (; i < n; i++) {
    sum += __builtin_abs(a[i] - b[i]);
  }
  data->res = sum;
}
#else
static void inner_loop_024(struct loop_024_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 40000
#endif

LOOP_DECL(024, SC_SVE_LOOP_ATTR)
{
  struct loop_024_data data = { .n = SIZE, .res = 0, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");

  fill_uint8(data.a, SIZE);
  fill_uint8(data.b, SIZE);

  inner_loops_024(iters, &data);

  uint32_t res = data.res;
  bool passed = res == 0x003433c9;
#ifndef STANDALONE
  FINALISE_LOOP_I(24, passed, "0x%08"PRIx32, 0x003433c9, res)
#endif
  return passed ? 0 : 1;
}
