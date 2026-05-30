/*----------------------------------------------------------------------------
#
#   Loop 025: FP32 small matrix-matrix multiply
#
#   Purpose:
#     Use of fp32 indexed MLA instruction.
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


struct loop_025_data {
  float *restrict a;
  float *restrict b;
  float *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_025(struct loop_025_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c) {
  memset(c, 0, sizeof(float) * 8 * 8);

  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      for (int i = 0; i < 8; i++) {
        c[col + row * 8] += a[i + row * 8] * b[col + i * 8];
      }
    }
  }
}
#elif defined(HAVE_SVE_INTRINSICS)
static void BLOCK_4x4_MM(uint32_t ar1, uint32_t ac1, uint32_t br1, uint32_t bc1,
                         uint32_t ar2, uint32_t ac2, uint32_t br2, uint32_t bc2,
                         uint32_t cr, uint32_t cc, float *restrict a,
                         float *restrict b, float *restrict c1) {
  float32_t *a1 = a + (ac1 - 1) * 4 + 32 * (ar1 - 1);
  float32_t *b1 = b + (bc1 - 1) * 4 + 32 * (br1 - 1);
  float32_t *a2 = a + (ac2 - 1) * 4 + 32 * (ar2 - 1);
  float32_t *b2 = b + (bc2 - 1) * 4 + 32 * (br2 - 1);
  float32_t *c = c1 + (cc - 1) * 4 + 32 * (cr - 1);

  svbool_t p0 = svptrue_b32();
  svfloat32_t z12 = svld1(p0, c);
  svfloat32_t z13 = svld1(p0, c + 8);
  svfloat32_t z14 = svld1(p0, c + 16);
  svfloat32_t z15 = svld1(p0, c + 24);

  svfloat32_t z4 = svld1(p0, a1);
  svfloat32_t z5 = svld1(p0, a1 + 8);
  svfloat32_t z6 = svld1(p0, a1 + 16);
  svfloat32_t z7 = svld1(p0, a1 + 24);

  svfloat32_t z8 = svld1(p0, b1);
  svfloat32_t z9 = svld1(p0, b1 + 8);
  svfloat32_t z10 = svld1(p0, b1 + 16);
  svfloat32_t z11 = svld1(p0, b1 + 24);

  svfloat32_t z0 = svld1(p0, a2);
  svfloat32_t z1 = svld1(p0, a2 + 8);
  svfloat32_t z2 = svld1(p0, a2 + 16);
  svfloat32_t z3 = svld1(p0, a2 + 24);

  svfloat32_t z16 = svld1(p0, b2);
  svfloat32_t z17 = svld1(p0, b2 + 8);
  svfloat32_t z18 = svld1(p0, b2 + 16);
  svfloat32_t z19 = svld1(p0, b2 + 24);

  z12 = svmla_lane(z12, z8, z4, 0);
  z13 = svmla_lane(z13, z8, z5, 0);
  z14 = svmla_lane(z14, z8, z6, 0);
  z15 = svmla_lane(z15, z8, z7, 0);

  z12 = svmla_lane(z12, z9, z4, 1);
  z13 = svmla_lane(z13, z9, z5, 1);
  z14 = svmla_lane(z14, z9, z6, 1);
  z15 = svmla_lane(z15, z9, z7, 1);

  z12 = svmla_lane(z12, z10, z4, 2);
  z13 = svmla_lane(z13, z10, z5, 2);
  z14 = svmla_lane(z14, z10, z6, 2);
  z15 = svmla_lane(z15, z10, z7, 2);

  z12 = svmla_lane(z12, z11, z4, 3);
  z13 = svmla_lane(z13, z11, z5, 3);
  z14 = svmla_lane(z14, z11, z6, 3);
  z15 = svmla_lane(z15, z11, z7, 3);

  z12 = svmla_lane(z12, z16, z0, 0);
  z13 = svmla_lane(z13, z16, z1, 0);
  z14 = svmla_lane(z14, z16, z2, 0);
  z15 = svmla_lane(z15, z16, z3, 0);

  z12 = svmla_lane(z12, z17, z0, 1);
  z13 = svmla_lane(z13, z17, z1, 1);
  z14 = svmla_lane(z14, z17, z2, 1);
  z15 = svmla_lane(z15, z17, z3, 1);

  z12 = svmla_lane(z12, z18, z0, 2);
  z13 = svmla_lane(z13, z18, z1, 2);
  z14 = svmla_lane(z14, z18, z2, 2);
  z15 = svmla_lane(z15, z18, z3, 2);

  z12 = svmla_lane(z12, z19, z0, 3);
  z13 = svmla_lane(z13, z19, z1, 3);
  z14 = svmla_lane(z14, z19, z2, 3);
  z15 = svmla_lane(z15, z19, z3, 3);

  svst1(p0, c, z12);
  svst1(p0, c + 8, z13);
  svst1(p0, c + 16, z14);
  svst1(p0, c + 24, z15);
}

static void matrix_multiply_8x8_sve128(float *restrict a, float *restrict b,
                                       float *restrict c1) {
  BLOCK_4x4_MM(1, 1, 1, 1, 1, 2, 2, 1, 1, 1, a, b, c1);
  BLOCK_4x4_MM(1, 1, 1, 2, 1, 2, 2, 2, 1, 2, a, b, c1);
  BLOCK_4x4_MM(2, 1, 1, 1, 2, 2, 2, 1, 2, 1, a, b, c1);
  BLOCK_4x4_MM(2, 1, 1, 2, 2, 2, 2, 2, 2, 2, a, b, c1);
}

static void BLOCK_8x4_MM(uint32_t ar1, uint32_t ac1, uint32_t br1, uint32_t bc1,
                         uint32_t ar2, uint32_t ac2, uint32_t br2, uint32_t bc2,
                         uint32_t cr, uint32_t cc, float *restrict a,
                         float *restrict b, float *restrict c1) {
  float32_t *a1 = a + (ac1 - 1) * 4 + 32 * (ar1 - 1);
  float32_t *b1 = b + (bc1 - 1) * 4 + 32 * (br1 - 1);
  float32_t *a2 = a + (ac2 - 1) * 4 + 32 * (ar2 - 1);
  float32_t *b2 = b + (bc2 - 1) * 4 + 32 * (br2 - 1);
  float32_t *c = c1 + (cc - 1) * 4 + 32 * (cr - 1);

  svbool_t p0 = svptrue_b32();
  svfloat32_t z12 = svld1(p0, c);
  svfloat32_t z13 = svld1(p0, c + 8);
  svfloat32_t z14 = svld1(p0, c + 16);
  svfloat32_t z15 = svld1(p0, c + 24);

  svfloat32_t z4 = svld1rq(p0, a1);
  svfloat32_t z5 = svld1rq(p0, a1 + 8);
  svfloat32_t z6 = svld1rq(p0, a1 + 16);
  svfloat32_t z7 = svld1rq(p0, a1 + 24);

  svfloat32_t z8 = svld1(p0, b1);
  svfloat32_t z9 = svld1(p0, b1 + 8);
  svfloat32_t z10 = svld1(p0, b1 + 16);
  svfloat32_t z11 = svld1(p0, b1 + 24);

  svfloat32_t z0 = svld1rq(p0, a2);
  svfloat32_t z1 = svld1rq(p0, a2 + 8);
  svfloat32_t z2 = svld1rq(p0, a2 + 16);
  svfloat32_t z3 = svld1rq(p0, a2 + 24);

  svfloat32_t z16 = svld1(p0, b2);
  svfloat32_t z17 = svld1(p0, b2 + 8);
  svfloat32_t z18 = svld1(p0, b2 + 16);
  svfloat32_t z19 = svld1(p0, b2 + 24);

  z12 = svmla_lane(z12, z8, z4, 0);
  z13 = svmla_lane(z13, z8, z5, 0);
  z14 = svmla_lane(z14, z8, z6, 0);
  z15 = svmla_lane(z15, z8, z7, 0);

  z12 = svmla_lane(z12, z9, z4, 1);
  z13 = svmla_lane(z13, z9, z5, 1);
  z14 = svmla_lane(z14, z9, z6, 1);
  z15 = svmla_lane(z15, z9, z7, 1);

  z12 = svmla_lane(z12, z10, z4, 2);
  z13 = svmla_lane(z13, z10, z5, 2);
  z14 = svmla_lane(z14, z10, z6, 2);
  z15 = svmla_lane(z15, z10, z7, 2);

  z12 = svmla_lane(z12, z11, z4, 3);
  z13 = svmla_lane(z13, z11, z5, 3);
  z14 = svmla_lane(z14, z11, z6, 3);
  z15 = svmla_lane(z15, z11, z7, 3);

  z12 = svmla_lane(z12, z16, z0, 0);
  z13 = svmla_lane(z13, z16, z1, 0);
  z14 = svmla_lane(z14, z16, z2, 0);
  z15 = svmla_lane(z15, z16, z3, 0);

  z12 = svmla_lane(z12, z17, z0, 1);
  z13 = svmla_lane(z13, z17, z1, 1);
  z14 = svmla_lane(z14, z17, z2, 1);
  z15 = svmla_lane(z15, z17, z3, 1);

  z12 = svmla_lane(z12, z18, z0, 2);
  z13 = svmla_lane(z13, z18, z1, 2);
  z14 = svmla_lane(z14, z18, z2, 2);
  z15 = svmla_lane(z15, z18, z3, 2);

  z12 = svmla_lane(z12, z19, z0, 3);
  z13 = svmla_lane(z13, z19, z1, 3);
  z14 = svmla_lane(z14, z19, z2, 3);
  z15 = svmla_lane(z15, z19, z3, 3);

  svst1(p0, c, z12);
  svst1(p0, c + 8, z13);
  svst1(p0, c + 16, z14);
  svst1(p0, c + 24, z15);
}

static void matrix_multiply_8x8_sve256(float *restrict a, float *restrict b,
                                       float *restrict c1) {
  BLOCK_8x4_MM(1, 1, 1, 1, 1, 2, 2, 1, 1, 1, a, b, c1);
  BLOCK_8x4_MM(2, 1, 1, 1, 2, 2, 2, 1, 2, 1, a, b, c1);
}

static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c1) {
  memset(c1, 0, sizeof(float) * 8 * 8);

  int vl = get_sve_vl();

  if (vl == 128) {
    matrix_multiply_8x8_sve128(a, b, c1);
  } else if (vl == 256) {
    matrix_multiply_8x8_sve256(a, b, c1);
  } else {
    printf("ABORT: disabled for VL > 32 bytes.\n");
    exit(2);
  }
}
#elif defined(__ARM_FEATURE_SVE)
static void matrix_multiply_8x8_sve128(float *restrict a, float *restrict b,
                                       float *restrict c) {
#ifdef BLOCK_4x4_MM
#error "BLOCK_4x4_MM defined"
#endif
#define BLOCK_4x4_MM(ar1, ac1, br1, bc1, ar2, ac2, br2, bc2, cr, cc)        \
  asm volatile(                                                             \
      "       ptrue   p0.s                                \n"               \
      "       mov     x7, #8                              \n"               \
      "       mov     x8, #16                             \n"               \
      "       mov     x9, #24                             \n"               \
                                                                            \
      "       ld1w    {z12.s}, p0/z, [%[c]]               \n"               \
      "       ld1w    {z13.s}, p0/z, [%[c], x7, lsl #2]   \n"               \
      "       ld1w    {z14.s}, p0/z, [%[c], x8, lsl #2]   \n"               \
      "       ld1w    {z15.s}, p0/z, [%[c], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1w    {z4.s}, p0/z, [%[a1]]               \n"               \
      "       ld1w    {z5.s}, p0/z, [%[a1], x7, lsl #2]   \n"               \
      "       ld1w    {z6.s}, p0/z, [%[a1], x8, lsl #2]   \n"               \
      "       ld1w    {z7.s}, p0/z, [%[a1], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1w    {z8.s},  p0/z, [%[b1]]              \n"               \
      "       ld1w    {z9.s},  p0/z, [%[b1], x7, lsl #2]  \n"               \
      "       ld1w    {z10.s}, p0/z, [%[b1], x8, lsl #2]  \n"               \
      "       ld1w    {z11.s}, p0/z, [%[b1], x9, lsl #2]  \n"               \
                                                                            \
      "       ld1w    {z0.s}, p0/z, [%[a2]]               \n"               \
      "       ld1w    {z1.s}, p0/z, [%[a2], x7, lsl #2]   \n"               \
      "       ld1w    {z2.s}, p0/z, [%[a2], x8, lsl #2]   \n"               \
      "       ld1w    {z3.s}, p0/z, [%[a2], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1w    {z16.s}, p0/z, [%[b2]]              \n"               \
      "       ld1w    {z17.s}, p0/z, [%[b2], x7, lsl #2]  \n"               \
      "       ld1w    {z18.s}, p0/z, [%[b2], x8, lsl #2]  \n"               \
      "       ld1w    {z19.s}, p0/z, [%[b2], x9, lsl #2]  \n"               \
                                                                            \
      "       fmla    z12.s, z8.s, z4.s[0]                \n"               \
      "       fmla    z13.s, z8.s, z5.s[0]                \n"               \
      "       fmla    z14.s, z8.s, z6.s[0]                \n"               \
      "       fmla    z15.s, z8.s, z7.s[0]                \n"               \
                                                                            \
      "       fmla    z12.s, z9.s, z4.s[1]                \n"               \
      "       fmla    z13.s, z9.s, z5.s[1]                \n"               \
      "       fmla    z14.s, z9.s, z6.s[1]                \n"               \
      "       fmla    z15.s, z9.s, z7.s[1]                \n"               \
                                                                            \
      "       fmla    z12.s, z10.s, z4.s[2]               \n"               \
      "       fmla    z13.s, z10.s, z5.s[2]               \n"               \
      "       fmla    z14.s, z10.s, z6.s[2]               \n"               \
      "       fmla    z15.s, z10.s, z7.s[2]               \n"               \
                                                                            \
      "       fmla    z12.s, z11.s, z4.s[3]               \n"               \
      "       fmla    z13.s, z11.s, z5.s[3]               \n"               \
      "       fmla    z14.s, z11.s, z6.s[3]               \n"               \
      "       fmla    z15.s, z11.s, z7.s[3]               \n"               \
                                                                            \
      "       fmla    z12.s, z16.s, z0.s[0]               \n"               \
      "       fmla    z13.s, z16.s, z1.s[0]               \n"               \
      "       fmla    z14.s, z16.s, z2.s[0]               \n"               \
      "       fmla    z15.s, z16.s, z3.s[0]               \n"               \
                                                                            \
      "       fmla    z12.s, z17.s, z0.s[1]               \n"               \
      "       fmla    z13.s, z17.s, z1.s[1]               \n"               \
      "       fmla    z14.s, z17.s, z2.s[1]               \n"               \
      "       fmla    z15.s, z17.s, z3.s[1]               \n"               \
                                                                            \
      "       fmla    z12.s, z18.s, z0.s[2]               \n"               \
      "       fmla    z13.s, z18.s, z1.s[2]               \n"               \
      "       fmla    z14.s, z18.s, z2.s[2]               \n"               \
      "       fmla    z15.s, z18.s, z3.s[2]               \n"               \
                                                                            \
      "       fmla    z12.s, z19.s, z0.s[3]               \n"               \
      "       fmla    z13.s, z19.s, z1.s[3]               \n"               \
      "       fmla    z14.s, z19.s, z2.s[3]               \n"               \
      "       fmla    z15.s, z19.s, z3.s[3]               \n"               \
                                                                            \
      "       st1w    {z12.s}, p0, [%[c]]                 \n"               \
      "       st1w    {z13.s}, p0, [%[c], x7, lsl #2]     \n"               \
      "       st1w    {z14.s}, p0, [%[c], x8, lsl #2]     \n"               \
      "       st1w    {z15.s}, p0, [%[c], x9, lsl #2]     \n"               \
                                                                            \
      :                                                                     \
      : [a1] "r"(a + (ac1 - 1) * 4 + 32 * (ar1 - 1)),                       \
        [b1] "r"(b + (bc1 - 1) * 4 + 32 * (br1 - 1)),                       \
        [a2] "r"(a + (ac2 - 1) * 4 + 32 * (ar2 - 1)),                       \
        [b2] "r"(b + (bc2 - 1) * 4 + 32 * (br2 - 1)),                       \
        [c] "r"(c + (cc - 1) * 4 + 32 * (cr - 1))                           \
      : "x7", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",   \
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
        "v18", "v19", "p0", "memory");

  BLOCK_4x4_MM(1, 1, 1, 1, 1, 2, 2, 1, 1, 1);
  BLOCK_4x4_MM(1, 1, 1, 2, 1, 2, 2, 2, 1, 2);
  BLOCK_4x4_MM(2, 1, 1, 1, 2, 2, 2, 1, 2, 1);
  BLOCK_4x4_MM(2, 1, 1, 2, 2, 2, 2, 2, 2, 2);
}

static void matrix_multiply_8x8_sve256(float *restrict a, float *restrict b,
                                       float *restrict c) {
#ifdef BLOCK_8x4_MM
#error "BLOCK_8x4_MM defined"
#endif
#define BLOCK_8x4_MM(ar1, ac1, br1, bc1, ar2, ac2, br2, bc2, cr, cc)        \
  asm volatile(                                                             \
      "       ptrue   p0.s                                \n"               \
      "       mov     x7, #8                              \n"               \
      "       mov     x8, #16                             \n"               \
      "       mov     x9, #24                             \n"               \
                                                                            \
      "       ld1w    {z12.s}, p0/z, [%[c]]               \n"               \
      "       ld1w    {z13.s}, p0/z, [%[c], x7, lsl #2]   \n"               \
      "       ld1w    {z14.s}, p0/z, [%[c], x8, lsl #2]   \n"               \
      "       ld1w    {z15.s}, p0/z, [%[c], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1rqw  {z4.s}, p0/z, [%[a1]]               \n"               \
      "       ld1rqw  {z5.s}, p0/z, [%[a1], x7, lsl #2]   \n"               \
      "       ld1rqw  {z6.s}, p0/z, [%[a1], x8, lsl #2]   \n"               \
      "       ld1rqw  {z7.s}, p0/z, [%[a1], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1w    {z8.s},  p0/z, [%[b1]]              \n"               \
      "       ld1w    {z9.s},  p0/z, [%[b1], x7, lsl #2]  \n"               \
      "       ld1w    {z10.s}, p0/z, [%[b1], x8, lsl #2]  \n"               \
      "       ld1w    {z11.s}, p0/z, [%[b1], x9, lsl #2]  \n"               \
                                                                            \
      "       ld1rqw  {z0.s}, p0/z, [%[a2]]               \n"               \
      "       ld1rqw  {z1.s}, p0/z, [%[a2], x7, lsl #2]   \n"               \
      "       ld1rqw  {z2.s}, p0/z, [%[a2], x8, lsl #2]   \n"               \
      "       ld1rqw  {z3.s}, p0/z, [%[a2], x9, lsl #2]   \n"               \
                                                                            \
      "       ld1w    {z16.s}, p0/z, [%[b2]]              \n"               \
      "       ld1w    {z17.s}, p0/z, [%[b2], x7, lsl #2]  \n"               \
      "       ld1w    {z18.s}, p0/z, [%[b2], x8, lsl #2]  \n"               \
      "       ld1w    {z19.s}, p0/z, [%[b2], x9, lsl #2]  \n"               \
                                                                            \
      "       fmla    z12.s, z8.s, z4.s[0]                \n"               \
      "       fmla    z13.s, z8.s, z5.s[0]                \n"               \
      "       fmla    z14.s, z8.s, z6.s[0]                \n"               \
      "       fmla    z15.s, z8.s, z7.s[0]                \n"               \
                                                                            \
      "       fmla    z12.s, z9.s, z4.s[1]                \n"               \
      "       fmla    z13.s, z9.s, z5.s[1]                \n"               \
      "       fmla    z14.s, z9.s, z6.s[1]                \n"               \
      "       fmla    z15.s, z9.s, z7.s[1]                \n"               \
                                                                            \
      "       fmla    z12.s, z10.s, z4.s[2]               \n"               \
      "       fmla    z13.s, z10.s, z5.s[2]               \n"               \
      "       fmla    z14.s, z10.s, z6.s[2]               \n"               \
      "       fmla    z15.s, z10.s, z7.s[2]               \n"               \
                                                                            \
      "       fmla    z12.s, z11.s, z4.s[3]               \n"               \
      "       fmla    z13.s, z11.s, z5.s[3]               \n"               \
      "       fmla    z14.s, z11.s, z6.s[3]               \n"               \
      "       fmla    z15.s, z11.s, z7.s[3]               \n"               \
                                                                            \
      "       fmla    z12.s, z16.s, z0.s[0]               \n"               \
      "       fmla    z13.s, z16.s, z1.s[0]               \n"               \
      "       fmla    z14.s, z16.s, z2.s[0]               \n"               \
      "       fmla    z15.s, z16.s, z3.s[0]               \n"               \
                                                                            \
      "       fmla    z12.s, z17.s, z0.s[1]               \n"               \
      "       fmla    z13.s, z17.s, z1.s[1]               \n"               \
      "       fmla    z14.s, z17.s, z2.s[1]               \n"               \
      "       fmla    z15.s, z17.s, z3.s[1]               \n"               \
                                                                            \
      "       fmla    z12.s, z18.s, z0.s[2]               \n"               \
      "       fmla    z13.s, z18.s, z1.s[2]               \n"               \
      "       fmla    z14.s, z18.s, z2.s[2]               \n"               \
      "       fmla    z15.s, z18.s, z3.s[2]               \n"               \
                                                                            \
      "       fmla    z12.s, z19.s, z0.s[3]               \n"               \
      "       fmla    z13.s, z19.s, z1.s[3]               \n"               \
      "       fmla    z14.s, z19.s, z2.s[3]               \n"               \
      "       fmla    z15.s, z19.s, z3.s[3]               \n"               \
                                                                            \
      "       st1w    {z12.s}, p0, [%[c]]                 \n"               \
      "       st1w    {z13.s}, p0, [%[c], x7, lsl #2]     \n"               \
      "       st1w    {z14.s}, p0, [%[c], x8, lsl #2]     \n"               \
      "       st1w    {z15.s}, p0, [%[c], x9, lsl #2]     \n"               \
                                                                            \
      :                                                                     \
      : [a1] "r"(a + (ac1 - 1) * 4 + 32 * (ar1 - 1)),                       \
        [b1] "r"(b + (bc1 - 1) * 4 + 32 * (br1 - 1)),                       \
        [a2] "r"(a + (ac2 - 1) * 4 + 32 * (ar2 - 1)),                       \
        [b2] "r"(b + (bc2 - 1) * 4 + 32 * (br2 - 1)),                       \
        [c] "r"(c + (cc - 1) * 4 + 32 * (cr - 1))                           \
      : "x7", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",   \
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
        "v18", "v19", "p0", "memory");

  BLOCK_8x4_MM(1, 1, 1, 1, 1, 2, 2, 1, 1, 1);
  BLOCK_8x4_MM(2, 1, 1, 1, 2, 2, 2, 1, 2, 1);
}

static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c) {
  memset(c, 0, sizeof(float) * 8 * 8);

  int vl = get_sve_vl();

  if (vl == 128) {
    matrix_multiply_8x8_sve128(a, b, c);
  } else if (vl == 256) {
    matrix_multiply_8x8_sve256(a, b, c);
  } else {
    printf("ABORT: disabled for VL > 32 bytes.\n");
    exit(2);
  }
}
#elif defined(__ARM_NEON)
static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c) {
  memset(c, 0, sizeof(float) * 8 * 8);

#ifdef BLOCK_4x4_MM
#error "BLOCK_4x4_MM defined"
#endif
#define BLOCK_4x4_MM(ar1, ac1, br1, bc1, ar2, ac2, br2, bc2, cr, cc)       \
  asm volatile(                                                            \
      "       ldr     q0, [%[c], #0]              \n"                      \
      "       ldr     q1, [%[c], #32]             \n"                      \
      "       ldr     q2, [%[c], #64]             \n"                      \
      "       ldr     q3, [%[c], #96]             \n"                      \
                                                                           \
      "       ldr     q4, [%[a1], #0]             \n"                      \
      "       ldr     q5, [%[a1], #32]            \n"                      \
      "       ldr     q6, [%[a1], #64]            \n"                      \
      "       ldr     q7, [%[a1], #96]            \n"                      \
                                                                           \
      "       ldr     q8,  [%[b1], #0]            \n"                      \
      "       ldr     q9,  [%[b1], #32]           \n"                      \
      "       ldr     q10, [%[b1], #64]           \n"                      \
      "       ldr     q11, [%[b1], #96]           \n"                      \
                                                                           \
      "       ldr     q12, [%[a2], #0]            \n"                      \
      "       ldr     q13, [%[a2], #32]           \n"                      \
      "       ldr     q14, [%[a2], #64]           \n"                      \
      "       ldr     q15, [%[a2], #96]           \n"                      \
                                                                           \
      "       ldr     q16, [%[b2], #0]            \n"                      \
      "       ldr     q17, [%[b2], #32]           \n"                      \
      "       ldr     q18, [%[b2], #64]           \n"                      \
      "       ldr     q19, [%[b2], #96]           \n"                      \
                                                                           \
      "       fmla    v0.4s, v8.4s,  v4.s[0]     \n"                       \
      "       fmla    v1.4s, v8.4s,  v5.s[0]     \n"                       \
      "       fmla    v2.4s, v8.4s,  v6.s[0]     \n"                       \
      "       fmla    v3.4s, v8.4s,  v7.s[0]     \n"                       \
                                                                           \
      "       fmla    v0.4s, v9.4s,  v4.s[1]     \n"                       \
      "       fmla    v1.4s, v9.4s,  v5.s[1]     \n"                       \
      "       fmla    v2.4s, v9.4s,  v6.s[1]     \n"                       \
      "       fmla    v3.4s, v9.4s,  v7.s[1]     \n"                       \
                                                                           \
      "       fmla    v0.4s, v10.4s, v4.s[2]     \n"                       \
      "       fmla    v1.4s, v10.4s, v5.s[2]     \n"                       \
      "       fmla    v2.4s, v10.4s, v6.s[2]     \n"                       \
      "       fmla    v3.4s, v10.4s, v7.s[2]     \n"                       \
                                                                           \
      "       fmla    v0.4s, v11.4s, v4.s[3]     \n"                       \
      "       fmla    v1.4s, v11.4s, v5.s[3]     \n"                       \
      "       fmla    v2.4s, v11.4s, v6.s[3]     \n"                       \
      "       fmla    v3.4s, v11.4s, v7.s[3]     \n"                       \
                                                                           \
      "       fmla    v0.4s, v16.4s, v12.s[0]    \n"                       \
      "       fmla    v1.4s, v16.4s, v13.s[0]    \n"                       \
      "       fmla    v2.4s, v16.4s, v14.s[0]    \n"                       \
      "       fmla    v3.4s, v16.4s, v15.s[0]    \n"                       \
                                                                           \
      "       fmla    v0.4s, v17.4s, v12.s[1]    \n"                       \
      "       fmla    v1.4s, v17.4s, v13.s[1]    \n"                       \
      "       fmla    v2.4s, v17.4s, v14.s[1]    \n"                       \
      "       fmla    v3.4s, v17.4s, v15.s[1]    \n"                       \
                                                                           \
      "       fmla    v0.4s, v18.4s, v12.s[2]    \n"                       \
      "       fmla    v1.4s, v18.4s, v13.s[2]    \n"                       \
      "       fmla    v2.4s, v18.4s, v14.s[2]    \n"                       \
      "       fmla    v3.4s, v18.4s, v15.s[2]    \n"                       \
                                                                           \
      "       fmla    v0.4s, v19.4s, v12.s[3]    \n"                       \
      "       fmla    v1.4s, v19.4s, v13.s[3]    \n"                       \
      "       fmla    v2.4s, v19.4s, v14.s[3]    \n"                       \
      "       fmla    v3.4s, v19.4s, v15.s[3]    \n"                       \
                                                                           \
      "       str     q0, [%[c], #0]              \n"                      \
      "       str     q1, [%[c], #32]             \n"                      \
      "       str     q2, [%[c], #64]             \n"                      \
      "       str     q3, [%[c], #96]             \n"                      \
                                                                           \
      :                                                                    \
      : [a1] "r"(a + (ac1 - 1) * 4 + 32 * (ar1 - 1)),                      \
        [b1] "r"(b + (bc1 - 1) * 4 + 32 * (br1 - 1)),                      \
        [a2] "r"(a + (ac2 - 1) * 4 + 32 * (ar2 - 1)),                      \
        [b2] "r"(b + (bc2 - 1) * 4 + 32 * (br2 - 1)),                      \
        [c] "r"(c + (cc - 1) * 4 + 32 * (cr - 1))                          \
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", \
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",     \
        "memory");

  BLOCK_4x4_MM(1, 1, 1, 1, 1, 2, 2, 1, 1, 1);
  BLOCK_4x4_MM(1, 1, 1, 2, 1, 2, 2, 2, 1, 2);
  BLOCK_4x4_MM(2, 1, 1, 1, 2, 2, 2, 1, 2, 1);
  BLOCK_4x4_MM(2, 1, 1, 2, 2, 2, 2, 2, 2, 2);
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void matrix_multiply_4x4(float *restrict a, float *restrict b,
                                float *restrict c, int width) {
  c[0 + 0 * width] += a[0 + 0 * width] * b[0 + 0 * width];
  c[0 + 0 * width] += a[1 + 0 * width] * b[0 + 1 * width];
  c[0 + 0 * width] += a[2 + 0 * width] * b[0 + 2 * width];
  c[0 + 0 * width] += a[3 + 0 * width] * b[0 + 3 * width];
  c[1 + 0 * width] += a[0 + 0 * width] * b[1 + 0 * width];
  c[1 + 0 * width] += a[1 + 0 * width] * b[1 + 1 * width];
  c[1 + 0 * width] += a[2 + 0 * width] * b[1 + 2 * width];
  c[1 + 0 * width] += a[3 + 0 * width] * b[1 + 3 * width];
  c[2 + 0 * width] += a[0 + 0 * width] * b[2 + 0 * width];
  c[2 + 0 * width] += a[1 + 0 * width] * b[2 + 1 * width];
  c[2 + 0 * width] += a[2 + 0 * width] * b[2 + 2 * width];
  c[2 + 0 * width] += a[3 + 0 * width] * b[2 + 3 * width];
  c[3 + 0 * width] += a[0 + 0 * width] * b[3 + 0 * width];
  c[3 + 0 * width] += a[1 + 0 * width] * b[3 + 1 * width];
  c[3 + 0 * width] += a[2 + 0 * width] * b[3 + 2 * width];
  c[3 + 0 * width] += a[3 + 0 * width] * b[3 + 3 * width];
  c[0 + 1 * width] += a[0 + 1 * width] * b[0 + 0 * width];
  c[0 + 1 * width] += a[1 + 1 * width] * b[0 + 1 * width];
  c[0 + 1 * width] += a[2 + 1 * width] * b[0 + 2 * width];
  c[0 + 1 * width] += a[3 + 1 * width] * b[0 + 3 * width];
  c[1 + 1 * width] += a[0 + 1 * width] * b[1 + 0 * width];
  c[1 + 1 * width] += a[1 + 1 * width] * b[1 + 1 * width];
  c[1 + 1 * width] += a[2 + 1 * width] * b[1 + 2 * width];
  c[1 + 1 * width] += a[3 + 1 * width] * b[1 + 3 * width];
  c[2 + 1 * width] += a[0 + 1 * width] * b[2 + 0 * width];
  c[2 + 1 * width] += a[1 + 1 * width] * b[2 + 1 * width];
  c[2 + 1 * width] += a[2 + 1 * width] * b[2 + 2 * width];
  c[2 + 1 * width] += a[3 + 1 * width] * b[2 + 3 * width];
  c[3 + 1 * width] += a[0 + 1 * width] * b[3 + 0 * width];
  c[3 + 1 * width] += a[1 + 1 * width] * b[3 + 1 * width];
  c[3 + 1 * width] += a[2 + 1 * width] * b[3 + 2 * width];
  c[3 + 1 * width] += a[3 + 1 * width] * b[3 + 3 * width];
  c[0 + 2 * width] += a[0 + 2 * width] * b[0 + 0 * width];
  c[0 + 2 * width] += a[1 + 2 * width] * b[0 + 1 * width];
  c[0 + 2 * width] += a[2 + 2 * width] * b[0 + 2 * width];
  c[0 + 2 * width] += a[3 + 2 * width] * b[0 + 3 * width];
  c[1 + 2 * width] += a[0 + 2 * width] * b[1 + 0 * width];
  c[1 + 2 * width] += a[1 + 2 * width] * b[1 + 1 * width];
  c[1 + 2 * width] += a[2 + 2 * width] * b[1 + 2 * width];
  c[1 + 2 * width] += a[3 + 2 * width] * b[1 + 3 * width];
  c[2 + 2 * width] += a[0 + 2 * width] * b[2 + 0 * width];
  c[2 + 2 * width] += a[1 + 2 * width] * b[2 + 1 * width];
  c[2 + 2 * width] += a[2 + 2 * width] * b[2 + 2 * width];
  c[2 + 2 * width] += a[3 + 2 * width] * b[2 + 3 * width];
  c[3 + 2 * width] += a[0 + 2 * width] * b[3 + 0 * width];
  c[3 + 2 * width] += a[1 + 2 * width] * b[3 + 1 * width];
  c[3 + 2 * width] += a[2 + 2 * width] * b[3 + 2 * width];
  c[3 + 2 * width] += a[3 + 2 * width] * b[3 + 3 * width];
  c[0 + 3 * width] += a[0 + 3 * width] * b[0 + 0 * width];
  c[0 + 3 * width] += a[1 + 3 * width] * b[0 + 1 * width];
  c[0 + 3 * width] += a[2 + 3 * width] * b[0 + 2 * width];
  c[0 + 3 * width] += a[3 + 3 * width] * b[0 + 3 * width];
  c[1 + 3 * width] += a[0 + 3 * width] * b[1 + 0 * width];
  c[1 + 3 * width] += a[1 + 3 * width] * b[1 + 1 * width];
  c[1 + 3 * width] += a[2 + 3 * width] * b[1 + 2 * width];
  c[1 + 3 * width] += a[3 + 3 * width] * b[1 + 3 * width];
  c[2 + 3 * width] += a[0 + 3 * width] * b[2 + 0 * width];
  c[2 + 3 * width] += a[1 + 3 * width] * b[2 + 1 * width];
  c[2 + 3 * width] += a[2 + 3 * width] * b[2 + 2 * width];
  c[2 + 3 * width] += a[3 + 3 * width] * b[2 + 3 * width];
  c[3 + 3 * width] += a[0 + 3 * width] * b[3 + 0 * width];
  c[3 + 3 * width] += a[1 + 3 * width] * b[3 + 1 * width];
  c[3 + 3 * width] += a[2 + 3 * width] * b[3 + 2 * width];
  c[3 + 3 * width] += a[3 + 3 * width] * b[3 + 3 * width];
}

// Block-based Matrix Multiply
//
//  {C11 C12}   {A11 A12}   {B11 B12}   {A11xB11+A12xB21  A11xB12+A12xB22}
//  {       } = {       } x {       } = {                                }
//  {C21 C22}   {A21 A22}   {B21 B22}   {A21xB11+A22xB21  A21xB12+A22xB22}

static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c) {
  memset(c, 0, sizeof(float) * 8 * 8);

#ifdef AT
#error "AT defined"
#endif
#define AT(matrix, row, col) (matrix + (col - 1) * 4 + 32 * (row - 1))

  matrix_multiply_4x4(AT(a, 1, 1), AT(b, 1, 1), AT(c, 1, 1), 8);
  matrix_multiply_4x4(AT(a, 1, 2), AT(b, 2, 1), AT(c, 1, 1), 8);
  matrix_multiply_4x4(AT(a, 1, 1), AT(b, 1, 2), AT(c, 1, 2), 8);
  matrix_multiply_4x4(AT(a, 1, 2), AT(b, 2, 2), AT(c, 1, 2), 8);
  matrix_multiply_4x4(AT(a, 2, 1), AT(b, 1, 1), AT(c, 2, 1), 8);
  matrix_multiply_4x4(AT(a, 2, 2), AT(b, 2, 1), AT(c, 2, 1), 8);
  matrix_multiply_4x4(AT(a, 2, 1), AT(b, 1, 2), AT(c, 2, 2), 8);
  matrix_multiply_4x4(AT(a, 2, 2), AT(b, 2, 2), AT(c, 2, 2), 8);

#undef AT
}

#else
static void inner_loop_025(struct loop_025_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_025(struct loop_025_data *restrict data) {
  float *a = data->a;
  float *b = data->b;
  float *c = data->c;
  for (int m = 0; m < 8; m++) {
    int offset = m * 8 * 8;
    matrix_multiply_8x8(a + offset, b + offset, c + offset);
  }
}
#endif /* !HAVE_CANDIDATE */

#define SIZE (8 * 8 * 8)

LOOP_DECL(025, NS_SVE_LOOP_ATTR)
{
  struct loop_025_data data;

  ALLOC_64B(data.a, SIZE, "A matrix");
  ALLOC_64B(data.b, SIZE, "A matrix");
  ALLOC_64B(data.c, SIZE, "A matrix");

  fill_float(data.a, SIZE);
  fill_float(data.b, SIZE);
  fill_float(data.c, SIZE);

  iters = iters * 4; // Multiply iters by 4 to increase work

  inner_loops_025(iters, &data);

  float checksum = 0.0f;
  for (int i = 0; i < SIZE; i++) {
    checksum += data.c[i] * (i % 10);
  }

  bool passed = check_float(checksum, 4365.9f, 0.1f);
#ifndef STANDALONE
  FINALISE_LOOP_F(25, passed, "%9.6f", 4365.9f, 0.1f, checksum)
#endif
  return passed ? 0 : 1;
}
