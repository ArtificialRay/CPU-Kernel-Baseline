/*----------------------------------------------------------------------------
#
#   Loop 031: small-lengths inline memcpy test
#
#   Purpose:
#     Use of simd-based memcpy for small lengths and at varied alignments.
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


struct loop_031_data {
  uint8_t *a;
  uint8_t *b;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_031(struct loop_031_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
//Do not define
//   static void inline_memcpy(uint8_t *restrict dst, uint8_t *restrict src,int64_t count)
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inline_memcpy(uint8_t *restrict dst, uint8_t *restrict src,
                          int64_t count)
LOOP_ATTR
{
  svbool_t p;
  FOR_LOOP_8(int64_t, i, 0, count, p)
  svst1(p, dst + i, svld1(p, src + i));
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))
static void inline_memcpy(uint8_t *restrict dst, uint8_t *restrict src,
                          int64_t count)
LOOP_ATTR
{
  int64_t i = 0;

  asm volatile(
      "   whilelo p0.b, %[i], %[count]                        \n"
      "1: ld1b    {z0.b}, p0/z, [%[src], %[i]]               \n"
      "   st1b    {z0.b}, p0, [%[dst], %[i]]                  \n"
      "   incb    %[i]                                        \n"
      "   whilelo p0.b, %[i], %[count]                        \n"
      "   b.any   1b                                          \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [dst] "r"(dst), [src] "r"(src), [count] "r"(count)
      : "v0", "p0", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inline_memcpy(uint8_t *restrict dst, uint8_t *restrict src,
                          int64_t count) {
  int64_t tmp1;
  int64_t A_l;
  int64_t B_l;
  int64_t A_h;
  int64_t srcend;
  int64_t dstend;

  asm volatile(
      "   add     %[srcend], %[src], %[count]     \n"
      "   add     %[dstend], %[dst], %[count]     \n"
      "   cmp     %[count], 16                    \n"
      "   b.ls    2f                              \n"
      "1: sub     %[count], %[count], 16          \n"
      "   cmp     %[count], 16                    \n"
      "   ldp     %[A_l], %[A_h], [%[src]], 16    \n"
      "   stp     %[A_l], %[A_h], [%[dst]], 16    \n"
      "   b.hi    1b                              \n"
      "2: cmp     %[count], 8                     \n"
      "   b.lo    1f                              \n"
      "   ldr     %[A_l], [%[src]]                \n"
      "   ldr     %[A_h], [%[srcend], -8]         \n"
      "   str     %[A_l], [%[dst]]                \n"
      "   str     %[A_h], [%[dstend], -8]         \n"
      "   b       2f                              \n"
      "1: tbz     %[count], 2, 1f                 \n"
      "   ldr     %w[A_l], [%[src]]               \n"
      "   ldr     %w[A_h], [%[srcend], -4]        \n"
      "   str     %w[A_l], [%[dst]]               \n"
      "   str     %w[A_h], [%[dstend], -4]        \n"
      "   b       2f                              \n"
      "1: cbz     %[count], 2f                    \n"
      "   lsr     %[tmp1], %[count], 1            \n"
      "   ldrb    %w[A_l], [%[src]]               \n"
      "   ldrb    %w[A_h], [%[srcend], -1]        \n"
      "   ldrb    %w[B_l], [%[src], %[tmp1]]      \n"
      "   strb    %w[A_l], [%[dst]]               \n"
      "   strb    %w[B_l], [%[dst], %[tmp1]]      \n"
      "   strb    %w[A_h], [%[dstend], -1]        \n"
      "2:                                         \n"
      // output operands, source operands, and clobber list
      : [tmp1] "=&r"(tmp1), [srcend] "=&r"(srcend), [dstend] "=&r"(dstend),
        [A_l] "=&r"(A_l), [B_l] "=&r"(B_l), [A_h] "=&r"(A_h), [dst] "+&r"(dst),
        [src] "+&r"(src), [count] "+&r"(count)
      :
      : "cc", "memory");
}
#endif

#if !defined(HAVE_CANDIDATE)
static size_t count[] = {0,  1,  2,  3,  4,  5,   6,   7,   8,   15,
                         16, 31, 64, 80, 96, 127, 128, 200, 255, 512};

#define CHUNKS 20

static void inner_loop_031(struct loop_031_data *restrict input)
LOOP_ATTR
{
  uint8_t *src = input->a;
  uint8_t *to = input->b;
  for (int j = 0; j < 10; j++) {
    for (int c = 0; c < CHUNKS; c++) {
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
      memcpy(to, src, count[c]);
#elif defined(__ARM_FEATURE_SVE) || defined(__aarch64__) && !defined(HAVE_AUTOVEC)
      inline_memcpy(to, src, count[c]);
#else
      memcpy(to, src, count[c]);
#endif
      src += count[c];
      to += count[c];
    }
  }
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 15600
#endif

LOOP_DECL(031, SC_SVE_LOOP_ATTR)
{
  struct loop_031_data data;

  ALLOC_64B(data.a, SIZE, "input data");
  ALLOC_64B(data.b, SIZE, "output buffer");

  fill_uint8(data.a, SIZE);
  fill_uint8(data.b, SIZE);

  inner_loops_031(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    checksum ^= data.b[i] << (8 * (i % 4));
  }

  uint32_t correct = 0xb2a9af85;
  bool passed = checksum == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(31, passed, "0x%08"PRIx32, correct, checksum)
#endif
  return passed ? 0 : 1;
}
