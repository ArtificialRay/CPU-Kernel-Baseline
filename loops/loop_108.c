/*----------------------------------------------------------------------------
#
#   Loop 108: Pixel manipulation
#
#   Purpose:
#     Use of LD4 with shift-accumulate instructions.
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

/*
  color pixels to grey scale
    Y = (R >> 2) + (G >> 1) + (G >> 3) + (B >> 3)
*/
struct loop_108_data {
  uint32_t *restrict rgba;
  uint8_t *restrict y;
  int64_t n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_108(struct loop_108_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_108(struct loop_108_data *restrict input) {
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    y[i] = (rgba[i] >> 24) >> 2;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 1;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 3;
    y[i] += ((rgba[i] >> 8) & 0xff) >> 3;
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_108(struct loop_108_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  svbool_t p;
  FOR_LOOP_8(int64_t, i, 0, n, p) {
    svuint8x4_t u = svld4(p, (uint8_t *)(rgba + i));
    svuint8_t v;
    v = svlsr_x(p, svget4(u, 3), 2);
    v = svsra(v, svget4(u, 2), 1);
    v = svsra(v, svget4(u, 2), 3);
    v = svsra(v, svget4(u, 1), 3);
    svst1(p, &(y[i]), v);
  }
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))

static void inner_loop_108(struct loop_108_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  int64_t i = 0;
  int64_t i4 = 0;

  asm volatile(
      "        whilelo  p0.b, %[i], %[n]                                  \n"
      "1:      ld4b     {z0.b, z1.b, z2.b, z3.b}, p0/z, [%[rgba], %[i4]]  \n"
      "        incb     %[i4], all, mul #4                                \n"
      "        lsr      z4.b, z3.b, #2                                    \n"
      "        usra     z4.b, z2.b, #1                                    \n"
      "        usra     z4.b, z2.b, #3                                    \n"
      "        usra     z4.b, z1.b, #3                                    \n"
      "        st1b     {z4.b}, p0, [%[y], %[i]]                          \n"
      "        incb     %[i]                                              \n"
      "        whilelo  p0.b, %[i], %[n]                                  \n"
      "        b.any    1b                                                \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [i4] "+&r"(i4)
      : [n] "r"(n), [rgba] "r"(rgba), [y] "r"(y)
      : "v0", "v1", "v2", "v3", "v4", "p0", "cc", "memory");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_108(struct loop_108_data *restrict input)
LOOP_ATTR
{
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  int64_t i = 0;
  int64_t i4 = 0;

  asm volatile(
      "        whilelo  p0.b, %[i], %[n]                                \n"
      "1:      ld4b     {z0.b, z1.b, z2.b, z3.b}, p0/z, [%[rgba], %[i4]]\n"
      "        incb     %[i4], all, mul #4                              \n"
      "        lsr      z3.b, z3.b, #2                                  \n"
      "        lsr      z4.b, z2.b, #3                                  \n"
      "        lsr      z2.b, z2.b, #1                                  \n"
      "        lsr      z1.b, z1.b, #3                                  \n"
      "        add      z3.b, z3.b, z2.b                                \n"
      "        add      z4.b, z4.b, z1.b                                \n"
      "        add      z3.b, z3.b, z4.b                                \n"
      "        st1b     {z3.b}, p0, [%[y], %[i]]                        \n"
      "        incb     %[i]                                            \n"
      "        whilelo  p0.b, %[i], %[n]                                \n"
      "        b.any    1b                                              \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [i4] "+&r"(i4)
      : [n] "r"(n), [rgba] "r"(rgba), [y] "r"(y)
      : "v0", "v1", "v2", "v3", "v4", "p0", "cc", "memory");
}
#elif defined(__ARM_NEON)

static void inner_loop_108(struct loop_108_data *restrict input) {
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  int64_t lmt = n - (n % 16);
  int64_t i = 0;
  uint32_t *read = rgba;

  asm volatile(
      "        b        2f                                                \n"
      "1:      ld4      {v0.16b, v1.16b, v2.16b, v3.16b}, [%[read]], #64  \n"
      "        ushr     v4.16b, v3.16b, #2                                \n"
      "        usra     v4.16b, v2.16b, #1                                \n"
      "        usra     v4.16b, v2.16b, #3                                \n"
      "        usra     v4.16b, v1.16b, #3                                \n"
      "        str      q4, [%[y], %[i]]                                  \n"
      "        add      %[i], %[i], #16                                   \n"
      "2:      cmp      %[i], %[lmt]                                      \n"
      "        b.lt     1b                                                \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [read] "+&r"(read)
      : [lmt] "r"(lmt), [y] "r"(y)
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");

  for (; i < n; i++) {
    y[i] = (rgba[i] >> 24) >> 2;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 1;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 3;
    y[i] += ((rgba[i] >> 8) & 0xff) >> 3;
  }
}
#else
static void inner_loop_108(struct loop_108_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(108, SC_SVE_LOOP_ATTR)
{
  struct loop_108_data data = { .n = SIZE };

  ALLOC_64B(data.rgba, SIZE, "input RGBA data");
  ALLOC_64B(data.y, SIZE, "output greyscale values");

  fill_uint32(data.rgba, SIZE);
  fill_uint8(data.y, SIZE);

  inner_loops_108(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    checksum ^= data.y[i] << (8 * (i % 4));
  }

  bool passed = checksum == 0x72831c31;
#ifndef STANDALONE
  FINALISE_LOOP_I(108, passed, "0x%08"PRIx32, 0x72831c31, checksum)
#endif
  return passed ? 0 : 1;
}
