/*----------------------------------------------------------------------------
#
#   Loop 216: FP32 col-major matrix-vector multiply
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

/*
  Data format -
    A: column-major
    X: column-vector
    B: column-vector
  Constraints -
    M: multiple of 16*SVLs
    N: multiple of 4
*/

struct loop_216_data {
  uint64_t m;
  uint64_t n;
  float32_t *restrict a;
  float32_t *restrict x;
  float32_t *restrict b;
};


#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_216(struct loop_216_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(__ARM_FEATURE_SME2)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_216(struct loop_216_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  float32_t *restrict a = data->a;
  float32_t *restrict x = data->x;
  float32_t *restrict b = data->b;
  float32_t d;
  for (uint64_t i = 0; i < m; i++) {
    d = 0;
    for (uint64_t j = 0; j < n; j++) d += a[(m * j) + i] * x[j];
    b[i] = d;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_216(struct loop_216_data *data)
LOOP_ATTR
{
  uint64_t A_rows = data->m;
  uint64_t A_cols = data->n;

  uint64_t svlw = svcntw();
  uint64_t ZA_size = svlw * svcntb();
  uint64_t A_rows_left_vgx4 = A_rows / svlw / 4;
  uint64_t ZA_rows_vgx4 = svlw;
  float32_t *A_src = data->a;
  float32_t *X_src = data->x;
  float32_t *B_dst = data->b;

  float32_t *A_ptr_base, *A_ptr;
  uint64_t A_row_idx, A_col_idx, ZA_group_idx;

  svbool_t p_all = svptrue_b32();
  svcount_t c_all = svptrue_c32();

  svfloat32_t ldx;
  svfloat32x4_t lda_0, lda_1, lda_2, lda_3;
  svfloat32x4_t stb_0, stb_1, stb_2, stb_3;

  svfloat64x4_t conv_0, conv_1, conv_2, conv_3;
#define CAST(conv, i) svreinterpret_f32(svget4((conv), (i)))
#define TO_F32X4(conv) svcreate4(CAST((conv), 0), CAST((conv), 1), CAST((conv), 2), CAST((conv), 3))

#define LOAD(q) lda_##q = svld1_vnum_x4(c_all, A_ptr, q * 4);
#define LOAD_GROUP LOAD(0) LOAD(1) LOAD(2) LOAD(3)

#define FMLA(l, q) svmla_lane_za32_vg1x4(ZA_group_idx + (q), lda_##q, ldx, l);
#define FMLA_GROUP(l) FMLA(l, 0) FMLA(l, 1) FMLA(l, 2) FMLA(l, 3)

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (A_row_idx = 0; A_row_idx < A_rows; A_row_idx += ZA_size) {
#if !defined(__ARM_FEATURE_SME2p1)
    svzero_za();
#endif
    ZA_rows_vgx4 = A_rows_left_vgx4 < ZA_rows_vgx4 ? A_rows_left_vgx4 : ZA_rows_vgx4;
    for (A_col_idx = 0; A_col_idx < A_cols; A_col_idx += 4) {

      ldx = svld1rq(p_all, &X_src[A_col_idx]);

      A_ptr_base = &A_src[A_row_idx + A_col_idx * A_rows];
      for (ZA_group_idx = 0; ZA_group_idx < ZA_rows_vgx4; ZA_group_idx += 4) {
        A_ptr = A_ptr_base + ZA_group_idx * 4 * svlw;

        LOAD_GROUP;
        FMLA_GROUP(0);
        A_ptr += A_rows;
        LOAD_GROUP;
        FMLA_GROUP(1);
        A_ptr += A_rows;
        LOAD_GROUP;
        FMLA_GROUP(2);
        A_ptr += A_rows;
        LOAD_GROUP;
        FMLA_GROUP(3);
      }
    }
    A_rows_left_vgx4 -= ZA_rows_vgx4;

    for (ZA_group_idx = 0; ZA_group_idx < ZA_rows_vgx4; ZA_group_idx += 4) {
#if defined(__ARM_FEATURE_SME2p1)
      conv_0 = svreadz_za64_f64_vg1x4(ZA_group_idx + 0);
      stb_0 = TO_F32X4(conv_0);
      conv_1 = svreadz_za64_f64_vg1x4(ZA_group_idx + 1);
      stb_1 = TO_F32X4(conv_1);
      conv_2 = svreadz_za64_f64_vg1x4(ZA_group_idx + 2);
      stb_2 = TO_F32X4(conv_2);
      conv_3 = svreadz_za64_f64_vg1x4(ZA_group_idx + 3);
      stb_3 = TO_F32X4(conv_3);
#else
      conv_0 = svread_za64_f64_vg1x4(ZA_group_idx + 0);
      stb_0 = TO_F32X4(conv_0);
      conv_1 = svread_za64_f64_vg1x4(ZA_group_idx + 1);
      stb_1 = TO_F32X4(conv_1);
      conv_2 = svread_za64_f64_vg1x4(ZA_group_idx + 2);
      stb_2 = TO_F32X4(conv_2);
      conv_3 = svread_za64_f64_vg1x4(ZA_group_idx + 3);
      stb_3 = TO_F32X4(conv_3);
#endif
      svst1_vnum(c_all, B_dst, 0, stb_0);
      svst1_vnum(c_all, B_dst, 4, stb_1);
      svst1_vnum(c_all, B_dst, 8, stb_2);
      svst1_vnum(c_all, B_dst, 12, stb_3);
      B_dst += 16 * svlw;
    }
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_216(struct loop_216_data *data)
LOOP_ATTR
{
  uint64_t A_rows = data->m;
  uint64_t A_cols = data->n;

  float32_t *A_src = data->a;
  float32_t *X_src = data->x;
  float32_t *B_dst = data->b;

  float32_t *A_ptr;
  float32_t *B_ptr;

  uint64_t A_row_idx, A_col_idx;
  svbool_t p_all = svptrue_b32();

  svfloat32x4_t acc_0, acc_1, acc_2, acc_3;
  svfloat32x4_t lda;
  svfloat32_t ldx;

#define ZERO svdup_f32(0)
#define ZERO_QUAD(q) acc_##q = svcreate4(ZERO, ZERO, ZERO, ZERO)

#define GETB(q, p) svget4(acc_##q, p)

#define FMLA(q, p, l) svmla_lane(GETB(q, p), svget4(lda, p), ldx, l)
#define FMLA_LANE(q, l) acc_##q = \
  svcreate4(FMLA(q, 0, l), FMLA(q, 1, l), FMLA(q, 2, l), FMLA(q, 3, l))

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c32();
#define LOAD_QUAD(q) lda = svld1_vnum_x4(c_all, A_ptr, q * 4)
#define STORE_QUAD(q) svst1_vnum(c_all, B_ptr, q * 4, acc_##q)
#else
#define LOAD(q, p) svld1_vnum(p_all, A_ptr, q * 4 + p)
#define LOAD_QUAD(q) lda = \
  svcreate4(LOAD(q, 0), LOAD(q, 1), LOAD(q, 2), LOAD(q, 3))
#define STORE(q, p) svst1_vnum(p_all, B_ptr, q * 4 + p, GETB(q, p));
#define STORE_QUAD(q) STORE(q, 0) STORE(q, 1) STORE(q, 2) STORE(q, 3)
#endif

  for (A_row_idx = 0; A_row_idx < A_rows; A_row_idx += svcntw() * 16) {
    ZERO_QUAD(0);
    ZERO_QUAD(1);
    ZERO_QUAD(2);
    ZERO_QUAD(3);

    A_ptr = &A_src[A_row_idx];
    for (A_col_idx = 0; A_col_idx < A_cols; A_col_idx += 4) {
      ldx = svld1rq(p_all, &X_src[A_col_idx]);

      LOAD_QUAD(0);
      FMLA_LANE(0, 0);
      LOAD_QUAD(1);
      FMLA_LANE(1, 0);
      LOAD_QUAD(2);
      FMLA_LANE(2, 0);
      LOAD_QUAD(3);
      FMLA_LANE(3, 0);
      A_ptr += A_rows;

      LOAD_QUAD(0);
      FMLA_LANE(0, 1);
      LOAD_QUAD(1);
      FMLA_LANE(1, 1);
      LOAD_QUAD(2);
      FMLA_LANE(2, 1);
      LOAD_QUAD(3);
      FMLA_LANE(3, 1);
      A_ptr += A_rows;

      LOAD_QUAD(0);
      FMLA_LANE(0, 2);
      LOAD_QUAD(1);
      FMLA_LANE(1, 2);
      LOAD_QUAD(2);
      FMLA_LANE(2, 2);
      LOAD_QUAD(3);
      FMLA_LANE(3, 2);
      A_ptr += A_rows;

      LOAD_QUAD(0);
      FMLA_LANE(0, 3);
      LOAD_QUAD(1);
      FMLA_LANE(1, 3);
      LOAD_QUAD(2);
      FMLA_LANE(2, 3);
      LOAD_QUAD(3);
      FMLA_LANE(3, 3);
      A_ptr += A_rows;
    }

    B_ptr = &B_dst[A_row_idx];
    STORE_QUAD(0);
    STORE_QUAD(1);
    STORE_QUAD(2);
    STORE_QUAD(3);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_216(struct loop_216_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;

  register uint64_t A_col_ptr ;
  register uint64_t A_col_end = (uint64_t) &data->a[A_rows * A_cols];
  register uint64_t A_row_end = (uint64_t) &data->a[A_rows];
  register uint64_t A_ptr_base;
  register uint64_t X_ptr;
  register uint64_t ZA_size;
  register uint64_t svlw;
  register uint64_t A_ptr;
  register uint64_t A_rows_left_vgx4;
  register uint64_t ZA_rows_vgx4;
  // x9: slice index register for fmla and mov

  asm volatile("cntw %[svlw]\n"
               :[svlw] "=&r"(svlw)::);
  asm volatile("cntb %[ZA_size]\n"
               "mul  %[ZA_size], %[ZA_size], %[svlw]\n"
               :[ZA_size] "=&r"(ZA_size)
               :[svlw] "r"(svlw):);

  A_rows_left_vgx4 = A_rows / svlw / 4;
  ZA_rows_vgx4 = svlw;

  asm volatile(
      "   ptrue   p0.b                                                \n"
      "   ptrue   pn8.s                                               \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      // Loop rows
      "1:                                                             \n"
      "   mov     %[A_col_ptr], %[A_src]                              \n"  // column loop pointer
      "   mov     %[X_ptr], %[X_src]                                  \n"  // X pointer
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif

      "   cmp     %[A_rows_left_vgx4], %[ZA_rows_vgx4]                \n"
      "   csel    %[ZA_rows_vgx4], %[A_rows_left_vgx4], %[ZA_rows_vgx4], lt \n"

      // Loop ALL columns
      "2:                                                             \n"
      // Assumption N is multiple of 4
      "   ld1rqw  {z0.s}, p0/z, [%[X_ptr]]                            \n"

      "   mov     x9, #0                                              \n"  // ZA group loop counter
      "   mov     %[A_ptr_base], %[A_col_ptr]                         \n"  // A base pointer

      // Loop fill ZA rows, store after
      "3:                                                             \n"
      "   mov     %[A_ptr], %[A_ptr_base]                             \n"  // A pointer

      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], #4, mul vl]        \n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], #8, mul vl]        \n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], #12, mul vl]       \n"
      "   add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2               \n"

      "   fmla    za.s[w9, #0], {z16.s-z19.s}, z0.s[0]                \n"
      "   fmla    za.s[w9, #1], {z20.s-z23.s}, z0.s[0]                \n"
      "   fmla    za.s[w9, #2], {z24.s-z27.s}, z0.s[0]                \n"
      "   fmla    za.s[w9, #3], {z28.s-z31.s}, z0.s[0]                \n"

      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], #4, mul vl]        \n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], #8, mul vl]        \n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], #12, mul vl]       \n"
      "   add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2               \n"

      "   fmla    za.s[w9, #0], {z16.s-z19.s}, z0.s[1]                \n"
      "   fmla    za.s[w9, #1], {z20.s-z23.s}, z0.s[1]                \n"
      "   fmla    za.s[w9, #2], {z24.s-z27.s}, z0.s[1]                \n"
      "   fmla    za.s[w9, #3], {z28.s-z31.s}, z0.s[1]                \n"

      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], #4, mul vl]        \n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], #8, mul vl]        \n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], #12, mul vl]       \n"
      "   add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2               \n"

      "   fmla    za.s[w9, #0], {z16.s-z19.s}, z0.s[2]                \n"
      "   fmla    za.s[w9, #1], {z20.s-z23.s}, z0.s[2]                \n"
      "   fmla    za.s[w9, #2], {z24.s-z27.s}, z0.s[2]                \n"
      "   fmla    za.s[w9, #3], {z28.s-z31.s}, z0.s[2]                \n"

      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], #4, mul vl]        \n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], #8, mul vl]        \n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], #12, mul vl]       \n"

      "   fmla    za.s[w9, #0], {z16.s-z19.s}, z0.s[3]                \n"
      "   fmla    za.s[w9, #1], {z20.s-z23.s}, z0.s[3]                \n"
      "   fmla    za.s[w9, #2], {z24.s-z27.s}, z0.s[3]                \n"
      "   fmla    za.s[w9, #3], {z28.s-z31.s}, z0.s[3]                \n"

      // Loop ZA groups increment
      "   addvl   %[A_ptr_base], %[A_ptr_base], #16                   \n"
      "   add     x9, x9, #4                                          \n"
      "   cmp     x9, %[ZA_rows_vgx4]                                 \n"
      "   b.lt    3b                                                  \n"

      // loop columns increments
      "   add     %[X_ptr], %[X_ptr], #16                             \n"
      "   add     %[A_col_ptr], %[A_col_ptr], %[A_rows], lsl #4       \n" // progress 4 cols = 4 * words per row
      "   cmp     %[A_col_ptr], %[A_col_end]                          \n"
      "   b.lt    2b                                                  \n"

      // End loop columns, store results
      "   mov     x9, #0                                              \n"

      "4:                                                             \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   movaz   {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   movaz   {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   movaz   {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#else
      "   mov     {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   mov     {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   mov     {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   mov     {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#endif
      "   st1w    {z16.s-z19.s}, pn8, [%[B_dst]]                      \n"
      "   st1w    {z20.s-z23.s}, pn8, [%[B_dst], #4, mul vl]          \n"
      "   st1w    {z24.s-z27.s}, pn8, [%[B_dst], #8, mul vl]          \n"
      "   st1w    {z28.s-z31.s}, pn8, [%[B_dst], #12, mul vl]         \n"

      "   addvl   %[B_dst], %[B_dst], #16                             \n"
      "   add     x9, x9, #4                                          \n"
      "   cmp     x9, %[ZA_rows_vgx4]                                 \n"
      "   b.lt    4b                                                  \n"

      // loop rows increments
      "   sub     %[A_rows_left_vgx4], %[A_rows_left_vgx4], %[ZA_rows_vgx4] \n"
      "   add     %[A_src], %[A_src], %[ZA_size], lsl #2              \n"
      "   cmp     %[A_src], %[A_row_end]                              \n"
      "   b.lt 1b                                                     \n"

      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_col_ptr] "=&r"(A_col_ptr),
        [A_ptr_base] "=&r"(A_ptr_base), [X_ptr] "=&r"(X_ptr),
        [A_ptr] "=&r"(A_ptr), [A_rows_left_vgx4] "+&r"(A_rows_left_vgx4),
        [ZA_rows_vgx4] "+&r"(ZA_rows_vgx4)
      :
        [A_rows] "r"(A_rows), [A_cols] "r"(A_cols), [X_src] "r"(X_src),
        [A_row_end] "r"(A_row_end),
        [A_col_end] "r"(A_col_end), [ZA_size] "r"(ZA_size),
        [svlw] "r"(svlw)
      : "z0", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
        "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p8", "x9",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_216(struct loop_216_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;
  register uint64_t A_ptr = 0;
  register uint64_t X_ptr = 0;
  register uint64_t X_end = (uint64_t)&data->x[data->n];
  register uint64_t A_end = (uint64_t)&data->a[A_rows];

  asm volatile(
      // constants
      "   ptrue   p0.s                                                \n"
      "   ptrue   pn8.s                                               \n"

      // Row loop
      "1:                                                             \n"
      "   mov     z16.s, #0                                           \n"
      "   mov     z17.s, #0                                           \n"
      "   mov     z18.s, #0                                           \n"
      "   mov     z19.s, #0                                           \n"
      "   mov     z20.s, #0                                           \n"
      "   mov     z21.s, #0                                           \n"
      "   mov     z22.s, #0                                           \n"
      "   mov     z23.s, #0                                           \n"
      "   mov     z24.s, #0                                           \n"
      "   mov     z25.s, #0                                           \n"
      "   mov     z26.s, #0                                           \n"
      "   mov     z27.s, #0                                           \n"
      "   mov     z28.s, #0                                           \n"
      "   mov     z29.s, #0                                           \n"
      "   mov     z30.s, #0                                           \n"
      "   mov     z31.s, #0                                           \n"

      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_ptr], %[A_src]                                  \n"
      "2:                                                             \n"
      "    ld1rqw  {z0.s}, p0/z, [%[X_ptr]]                           \n"
      "    add     %[X_ptr], %[X_ptr], #16                            \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x0, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0x4, mul vl]     \n"
      "    fmla    z16.s, z8.s, z0.s[0]                               \n"
      "    fmla    z17.s, z9.s, z0.s[0]                               \n"
      "    fmla    z18.s, z10.s, z0.s[0]                              \n"
      "    fmla    z19.s, z11.s, z0.s[0]                              \n"
      "    fmla    z20.s, z12.s, z0.s[0]                              \n"
      "    fmla    z21.s, z13.s, z0.s[0]                              \n"
      "    fmla    z22.s, z14.s, z0.s[0]                              \n"
      "    fmla    z23.s, z15.s, z0.s[0]                              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x8, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0xc, mul vl]     \n"
      "    fmla    z24.s, z8.s, z0.s[0]                               \n"
      "    fmla    z25.s, z9.s, z0.s[0]                               \n"
      "    fmla    z26.s, z10.s, z0.s[0]                              \n"
      "    fmla    z27.s, z11.s, z0.s[0]                              \n"
      "    fmla    z28.s, z12.s, z0.s[0]                              \n"
      "    fmla    z29.s, z13.s, z0.s[0]                              \n"
      "    fmla    z30.s, z14.s, z0.s[0]                              \n"
      "    fmla    z31.s, z15.s, z0.s[0]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x0, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0x4, mul vl]     \n"
      "    fmla    z16.s, z8.s, z0.s[1]                               \n"
      "    fmla    z17.s, z9.s, z0.s[1]                               \n"
      "    fmla    z18.s, z10.s, z0.s[1]                              \n"
      "    fmla    z19.s, z11.s, z0.s[1]                              \n"
      "    fmla    z20.s, z12.s, z0.s[1]                              \n"
      "    fmla    z21.s, z13.s, z0.s[1]                              \n"
      "    fmla    z22.s, z14.s, z0.s[1]                              \n"
      "    fmla    z23.s, z15.s, z0.s[1]                              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x8, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0xc, mul vl]     \n"
      "    fmla    z24.s, z8.s, z0.s[1]                               \n"
      "    fmla    z25.s, z9.s, z0.s[1]                               \n"
      "    fmla    z26.s, z10.s, z0.s[1]                              \n"
      "    fmla    z27.s, z11.s, z0.s[1]                              \n"
      "    fmla    z28.s, z12.s, z0.s[1]                              \n"
      "    fmla    z29.s, z13.s, z0.s[1]                              \n"
      "    fmla    z30.s, z14.s, z0.s[1]                              \n"
      "    fmla    z31.s, z15.s, z0.s[1]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x0, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0x4, mul vl]     \n"
      "    fmla    z16.s, z8.s, z0.s[2]                               \n"
      "    fmla    z17.s, z9.s, z0.s[2]                               \n"
      "    fmla    z18.s, z10.s, z0.s[2]                              \n"
      "    fmla    z19.s, z11.s, z0.s[2]                              \n"
      "    fmla    z20.s, z12.s, z0.s[2]                              \n"
      "    fmla    z21.s, z13.s, z0.s[2]                              \n"
      "    fmla    z22.s, z14.s, z0.s[2]                              \n"
      "    fmla    z23.s, z15.s, z0.s[2]                              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x8, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0xc, mul vl]     \n"
      "    fmla    z24.s, z8.s, z0.s[2]                               \n"
      "    fmla    z25.s, z9.s, z0.s[2]                               \n"
      "    fmla    z26.s, z10.s, z0.s[2]                              \n"
      "    fmla    z27.s, z11.s, z0.s[2]                              \n"
      "    fmla    z28.s, z12.s, z0.s[2]                              \n"
      "    fmla    z29.s, z13.s, z0.s[2]                              \n"
      "    fmla    z30.s, z14.s, z0.s[2]                              \n"
      "    fmla    z31.s, z15.s, z0.s[2]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x0, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0x4, mul vl]     \n"
      "    fmla    z16.s, z8.s, z0.s[3]                               \n"
      "    fmla    z17.s, z9.s, z0.s[3]                               \n"
      "    fmla    z18.s, z10.s, z0.s[3]                              \n"
      "    fmla    z19.s, z11.s, z0.s[3]                              \n"
      "    fmla    z20.s, z12.s, z0.s[3]                              \n"
      "    fmla    z21.s, z13.s, z0.s[3]                              \n"
      "    fmla    z22.s, z14.s, z0.s[3]                              \n"
      "    fmla    z23.s, z15.s, z0.s[3]                              \n"
      "    ld1w    {z8.s-z11.s} , pn8/z, [%[A_ptr], #0x8, mul vl]     \n"
      "    ld1w    {z12.s-z15.s}, pn8/z, [%[A_ptr], #0xc, mul vl]     \n"
      "    fmla    z24.s, z8.s, z0.s[3]                               \n"
      "    fmla    z25.s, z9.s, z0.s[3]                               \n"
      "    fmla    z26.s, z10.s, z0.s[3]                              \n"
      "    fmla    z27.s, z11.s, z0.s[3]                              \n"
      "    fmla    z28.s, z12.s, z0.s[3]                              \n"
      "    fmla    z29.s, z13.s, z0.s[3]                              \n"
      "    fmla    z30.s, z14.s, z0.s[3]                              \n"
      "    fmla    z31.s, z15.s, z0.s[3]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"
      "    cmp     %[X_ptr], %[X_end]                                 \n"
      "    b.lt    2b                                                 \n"

      "    st1w    {z16.s-z19.s}, pn8, [%[B_dst], #0x0, mul vl]       \n"
      "    st1w    {z20.s-z23.s}, pn8, [%[B_dst], #0x4, mul vl]       \n"
      "    st1w    {z24.s-z27.s}, pn8, [%[B_dst], #0x8, mul vl]       \n"
      "    st1w    {z28.s-z31.s}, pn8, [%[B_dst], #0xc, mul vl]       \n"

      "    addvl   %[A_src], %[A_src], #16                            \n"
      "    addvl   %[B_dst], %[B_dst], #16                            \n"
      "    cmp     %[A_src], %[A_end]                                 \n"
      "    b.lt 1b                                                    \n"

      : [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst),
        [A_ptr] "=&r"(A_ptr), [X_ptr] "=&r"(X_ptr)
      : [X_src] "r"(X_src), [X_end] "r"(X_end), [A_end] "r"(A_end),
        [A_rows] "r"(A_rows)
      : "z0", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
        "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
        "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
        "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_216(struct loop_216_data *data)
LOOP_ATTR
{
  register uint64_t svlw = 0;

  asm volatile("cntw %[svlw]\n"
               :[svlw] "=&r"(svlw)::);

  register uint64_t A_rows = data->m;
  register uint64_t A_src = (uint64_t)&data->a[8 * svlw];
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)&data->b[8 * svlw];
  register uint64_t A_ptr;
  register uint64_t X_ptr;
  register uint64_t X_end = (uint64_t)&data->x[data->n];
  register uint64_t A_end = (uint64_t)&data->a[A_rows + 8 * svlw];

  asm volatile(
      // constants
      "   ptrue   p0.s                                                \n"

      // Row loop
      "1:                                                             \n"

      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_ptr], %[A_src]                                  \n"

      "   mov     z16.s, #0                                           \n"
      "   mov     z17.s, #0                                           \n"
      "   mov     z18.s, #0                                           \n"
      "   mov     z19.s, #0                                           \n"
      "   mov     z20.s, #0                                           \n"
      "   mov     z21.s, #0                                           \n"
      "   mov     z22.s, #0                                           \n"
      "   mov     z23.s, #0                                           \n"
      "   mov     z24.s, #0                                           \n"
      "   mov     z25.s, #0                                           \n"
      "   mov     z26.s, #0                                           \n"
      "   mov     z27.s, #0                                           \n"
      "   mov     z28.s, #0                                           \n"
      "   mov     z29.s, #0                                           \n"
      "   mov     z30.s, #0                                           \n"
      "   mov     z31.s, #0                                           \n"

      // Column loop
      // N must be multiple of 4
      // M must be multiple of 16 * SVLs
      "2:                                                             \n"

      "    ld1rqw  {z0.s}, p0/z, [%[X_ptr]]                           \n"
      "    add     %[X_ptr], %[X_ptr], #16                            \n"

      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #-8, mul vl]              \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #-7, mul vl]              \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #-6, mul vl]             \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #-5, mul vl]             \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #-4, mul vl]             \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #-3, mul vl]             \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #-2, mul vl]             \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #-1, mul vl]             \n"
      "    fmla    z16.s, z8.s, z0.s[0]                               \n"
      "    fmla    z17.s, z9.s, z0.s[0]                               \n"
      "    fmla    z18.s, z10.s, z0.s[0]                              \n"
      "    fmla    z19.s, z11.s, z0.s[0]                              \n"
      "    fmla    z20.s, z12.s, z0.s[0]                              \n"
      "    fmla    z21.s, z13.s, z0.s[0]                              \n"
      "    fmla    z22.s, z14.s, z0.s[0]                              \n"
      "    fmla    z23.s, z15.s, z0.s[0]                              \n"
      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #0, mul vl]               \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #1, mul vl]               \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #4, mul vl]              \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #5, mul vl]              \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #6, mul vl]              \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #7, mul vl]              \n"
      "    fmla    z24.s, z8.s, z0.s[0]                               \n"
      "    fmla    z25.s, z9.s, z0.s[0]                               \n"
      "    fmla    z26.s, z10.s, z0.s[0]                              \n"
      "    fmla    z27.s, z11.s, z0.s[0]                              \n"
      "    fmla    z28.s, z12.s, z0.s[0]                              \n"
      "    fmla    z29.s, z13.s, z0.s[0]                              \n"
      "    fmla    z30.s, z14.s, z0.s[0]                              \n"
      "    fmla    z31.s, z15.s, z0.s[0]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"

      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #-8, mul vl]              \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #-7, mul vl]              \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #-6, mul vl]             \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #-5, mul vl]             \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #-4, mul vl]             \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #-3, mul vl]             \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #-2, mul vl]             \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #-1, mul vl]             \n"
      "    fmla    z16.s, z8.s, z0.s[1]                               \n"
      "    fmla    z17.s, z9.s, z0.s[1]                               \n"
      "    fmla    z18.s, z10.s, z0.s[1]                              \n"
      "    fmla    z19.s, z11.s, z0.s[1]                              \n"
      "    fmla    z20.s, z12.s, z0.s[1]                              \n"
      "    fmla    z21.s, z13.s, z0.s[1]                              \n"
      "    fmla    z22.s, z14.s, z0.s[1]                              \n"
      "    fmla    z23.s, z15.s, z0.s[1]                              \n"
      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #0, mul vl]               \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #1, mul vl]               \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #4, mul vl]              \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #5, mul vl]              \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #6, mul vl]              \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #7, mul vl]              \n"
      "    fmla    z24.s, z8.s, z0.s[1]                               \n"
      "    fmla    z25.s, z9.s, z0.s[1]                               \n"
      "    fmla    z26.s, z10.s, z0.s[1]                              \n"
      "    fmla    z27.s, z11.s, z0.s[1]                              \n"
      "    fmla    z28.s, z12.s, z0.s[1]                              \n"
      "    fmla    z29.s, z13.s, z0.s[1]                              \n"
      "    fmla    z30.s, z14.s, z0.s[1]                              \n"
      "    fmla    z31.s, z15.s, z0.s[1]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"

      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #-8, mul vl]              \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #-7, mul vl]              \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #-6, mul vl]             \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #-5, mul vl]             \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #-4, mul vl]             \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #-3, mul vl]             \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #-2, mul vl]             \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #-1, mul vl]             \n"
      "    fmla    z16.s, z8.s, z0.s[2]                               \n"
      "    fmla    z17.s, z9.s, z0.s[2]                               \n"
      "    fmla    z18.s, z10.s, z0.s[2]                              \n"
      "    fmla    z19.s, z11.s, z0.s[2]                              \n"
      "    fmla    z20.s, z12.s, z0.s[2]                              \n"
      "    fmla    z21.s, z13.s, z0.s[2]                              \n"
      "    fmla    z22.s, z14.s, z0.s[2]                              \n"
      "    fmla    z23.s, z15.s, z0.s[2]                              \n"
      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #0, mul vl]               \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #1, mul vl]               \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #4, mul vl]              \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #5, mul vl]              \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #6, mul vl]              \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #7, mul vl]              \n"
      "    fmla    z24.s, z8.s, z0.s[2]                               \n"
      "    fmla    z25.s, z9.s, z0.s[2]                               \n"
      "    fmla    z26.s, z10.s, z0.s[2]                              \n"
      "    fmla    z27.s, z11.s, z0.s[2]                              \n"
      "    fmla    z28.s, z12.s, z0.s[2]                              \n"
      "    fmla    z29.s, z13.s, z0.s[2]                              \n"
      "    fmla    z30.s, z14.s, z0.s[2]                              \n"
      "    fmla    z31.s, z15.s, z0.s[2]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"

      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #-8, mul vl]              \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #-7, mul vl]              \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #-6, mul vl]             \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #-5, mul vl]             \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #-4, mul vl]             \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #-3, mul vl]             \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #-2, mul vl]             \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #-1, mul vl]             \n"
      "    fmla    z16.s, z8.s, z0.s[3]                               \n"
      "    fmla    z17.s, z9.s, z0.s[3]                               \n"
      "    fmla    z18.s, z10.s, z0.s[3]                              \n"
      "    fmla    z19.s, z11.s, z0.s[3]                              \n"
      "    fmla    z20.s, z12.s, z0.s[3]                              \n"
      "    fmla    z21.s, z13.s, z0.s[3]                              \n"
      "    fmla    z22.s, z14.s, z0.s[3]                              \n"
      "    fmla    z23.s, z15.s, z0.s[3]                              \n"
      "    ld1w    {z8.s}, p0/z, [%[A_ptr], #0, mul vl]               \n"
      "    ld1w    {z9.s}, p0/z, [%[A_ptr], #1, mul vl]               \n"
      "    ld1w    {z10.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z11.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    ld1w    {z12.s}, p0/z, [%[A_ptr], #4, mul vl]              \n"
      "    ld1w    {z13.s}, p0/z, [%[A_ptr], #5, mul vl]              \n"
      "    ld1w    {z14.s}, p0/z, [%[A_ptr], #6, mul vl]              \n"
      "    ld1w    {z15.s}, p0/z, [%[A_ptr], #7, mul vl]              \n"
      "    fmla    z24.s, z8.s, z0.s[3]                               \n"
      "    fmla    z25.s, z9.s, z0.s[3]                               \n"
      "    fmla    z26.s, z10.s, z0.s[3]                              \n"
      "    fmla    z27.s, z11.s, z0.s[3]                              \n"
      "    fmla    z28.s, z12.s, z0.s[3]                              \n"
      "    fmla    z29.s, z13.s, z0.s[3]                              \n"
      "    fmla    z30.s, z14.s, z0.s[3]                              \n"
      "    fmla    z31.s, z15.s, z0.s[3]                              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_rows], lsl #2              \n"

      "    cmp     %[X_ptr], %[X_end]                                 \n"
      "    b.lt    2b                                                 \n"

      "    st1w    {z16.s}, p0, [%[B_dst], #-8, mul vl]               \n"
      "    st1w    {z17.s}, p0, [%[B_dst], #-7, mul vl]               \n"
      "    st1w    {z18.s}, p0, [%[B_dst], #-6, mul vl]               \n"
      "    st1w    {z19.s}, p0, [%[B_dst], #-5, mul vl]               \n"
      "    st1w    {z20.s}, p0, [%[B_dst], #-4, mul vl]               \n"
      "    st1w    {z21.s}, p0, [%[B_dst], #-3, mul vl]               \n"
      "    st1w    {z22.s}, p0, [%[B_dst], #-2, mul vl]               \n"
      "    st1w    {z23.s}, p0, [%[B_dst], #-1, mul vl]               \n"
      "    st1w    {z24.s}, p0, [%[B_dst], #0, mul vl]                \n"
      "    st1w    {z25.s}, p0, [%[B_dst], #1, mul vl]                \n"
      "    st1w    {z26.s}, p0, [%[B_dst], #2, mul vl]                \n"
      "    st1w    {z27.s}, p0, [%[B_dst], #3, mul vl]                \n"
      "    st1w    {z28.s}, p0, [%[B_dst], #4, mul vl]                \n"
      "    st1w    {z29.s}, p0, [%[B_dst], #5, mul vl]                \n"
      "    st1w    {z30.s}, p0, [%[B_dst], #6, mul vl]                \n"
      "    st1w    {z31.s}, p0, [%[B_dst], #7, mul vl]                \n"

      "    addvl   %[B_dst], %[B_dst], #16                            \n"
      "    addvl   %[A_src], %[A_src], #16                            \n"
      "    cmp     %[A_src], %[A_end]                                 \n"
      "    b.lt 1b                                                    \n"

      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_ptr] "=&r"(A_ptr),
        [X_ptr] "=&r"(X_ptr)
      : [A_rows] "r"(A_rows), [X_src] "r"(X_src), [X_end] "r"(X_end),
        [A_end] "r"(A_end)
      : "p0",
        "z0", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
        "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
        "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
        "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_216(struct loop_216_data *data) {
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;

  register uint64_t A_end = (uint64_t)&data->a[A_rows];
  register uint64_t X_end = (uint64_t)&data->x[A_cols];
  register uint64_t A_ptr_base;
  register uint64_t A_ptr;
  register uint64_t X_ptr;
  register uint64_t A_row_inc = 16 * 4 * 4;
  register uint64_t A_rows_x3 = A_rows * 3;

  asm volatile(
      // Row loop head
      "1:                                                             \n"
      "   movi    v16.4s, #0                                          \n"
      "   movi    v17.4s, #0                                          \n"
      "   movi    v18.4s, #0                                          \n"
      "   movi    v19.4s, #0                                          \n"
      "   movi    v20.4s, #0                                          \n"
      "   movi    v21.4s, #0                                          \n"
      "   movi    v22.4s, #0                                          \n"
      "   movi    v23.4s, #0                                          \n"
      "   movi    v24.4s, #0                                          \n"
      "   movi    v25.4s, #0                                          \n"
      "   movi    v26.4s, #0                                          \n"
      "   movi    v27.4s, #0                                          \n"
      "   movi    v28.4s, #0                                          \n"
      "   movi    v29.4s, #0                                          \n"
      "   movi    v30.4s, #0                                          \n"
      "   movi    v31.4s, #0                                          \n"

      // Column loop
      // N must be multiple of 4
      // M must be multiple of 64
      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_ptr_base], %[A_src]                             \n"
      "2:                                                             \n"
      "   ldr     q0, [%[X_ptr]], #16                                 \n"
      "   mov     %[A_ptr], %[A_ptr_base]                             \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v16.4s, v1.4s, v0.s[0]                              \n"
      "   fmla    v17.4s, v2.4s, v0.s[0]                              \n"
      "   fmla    v18.4s, v3.4s, v0.s[0]                              \n"
      "   fmla    v19.4s, v4.4s, v0.s[0]                              \n"
      "   fmla    v20.4s, v5.4s, v0.s[0]                              \n"
      "   fmla    v21.4s, v6.4s, v0.s[0]                              \n"
      "   fmla    v22.4s, v7.4s, v0.s[0]                              \n"
      "   fmla    v23.4s, v8.4s, v0.s[0]                              \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v24.4s, v1.4s, v0.s[0]                              \n"
      "   fmla    v25.4s, v2.4s, v0.s[0]                              \n"
      "   fmla    v26.4s, v3.4s, v0.s[0]                              \n"
      "   fmla    v27.4s, v4.4s, v0.s[0]                              \n"
      "   fmla    v28.4s, v5.4s, v0.s[0]                              \n"
      "   fmla    v29.4s, v6.4s, v0.s[0]                              \n"
      "   fmla    v30.4s, v7.4s, v0.s[0]                              \n"
      "   fmla    v31.4s, v8.4s, v0.s[0]                              \n"
      "   add     %[A_ptr], %[A_ptr_base], %[A_rows], lsl #2          \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v16.4s, v1.4s, v0.s[1]                              \n"
      "   fmla    v17.4s, v2.4s, v0.s[1]                              \n"
      "   fmla    v18.4s, v3.4s, v0.s[1]                              \n"
      "   fmla    v19.4s, v4.4s, v0.s[1]                              \n"
      "   fmla    v20.4s, v5.4s, v0.s[1]                              \n"
      "   fmla    v21.4s, v6.4s, v0.s[1]                              \n"
      "   fmla    v22.4s, v7.4s, v0.s[1]                              \n"
      "   fmla    v23.4s, v8.4s, v0.s[1]                              \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v24.4s, v1.4s, v0.s[1]                              \n"
      "   fmla    v25.4s, v2.4s, v0.s[1]                              \n"
      "   fmla    v26.4s, v3.4s, v0.s[1]                              \n"
      "   fmla    v27.4s, v4.4s, v0.s[1]                              \n"
      "   fmla    v28.4s, v5.4s, v0.s[1]                              \n"
      "   fmla    v29.4s, v6.4s, v0.s[1]                              \n"
      "   fmla    v30.4s, v7.4s, v0.s[1]                              \n"
      "   fmla    v31.4s, v8.4s, v0.s[1]                              \n"
      "   add     %[A_ptr], %[A_ptr_base], %[A_rows], lsl #3          \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v16.4s, v1.4s, v0.s[2]                              \n"
      "   fmla    v17.4s, v2.4s, v0.s[2]                              \n"
      "   fmla    v18.4s, v3.4s, v0.s[2]                              \n"
      "   fmla    v19.4s, v4.4s, v0.s[2]                              \n"
      "   fmla    v20.4s, v5.4s, v0.s[2]                              \n"
      "   fmla    v21.4s, v6.4s, v0.s[2]                              \n"
      "   fmla    v22.4s, v7.4s, v0.s[2]                              \n"
      "   fmla    v23.4s, v8.4s, v0.s[2]                              \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v24.4s, v1.4s, v0.s[2]                              \n"
      "   fmla    v25.4s, v2.4s, v0.s[2]                              \n"
      "   fmla    v26.4s, v3.4s, v0.s[2]                              \n"
      "   fmla    v27.4s, v4.4s, v0.s[2]                              \n"
      "   fmla    v28.4s, v5.4s, v0.s[2]                              \n"
      "   fmla    v29.4s, v6.4s, v0.s[2]                              \n"
      "   fmla    v30.4s, v7.4s, v0.s[2]                              \n"
      "   fmla    v31.4s, v8.4s, v0.s[2]                              \n"
      "   add     %[A_ptr], %[A_ptr_base], %[A_rows_x3], lsl #2       \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v16.4s, v1.4s, v0.s[3]                              \n"
      "   fmla    v17.4s, v2.4s, v0.s[3]                              \n"
      "   fmla    v18.4s, v3.4s, v0.s[3]                              \n"
      "   fmla    v19.4s, v4.4s, v0.s[3]                              \n"
      "   fmla    v20.4s, v5.4s, v0.s[3]                              \n"
      "   fmla    v21.4s, v6.4s, v0.s[3]                              \n"
      "   fmla    v22.4s, v7.4s, v0.s[3]                              \n"
      "   fmla    v23.4s, v8.4s, v0.s[3]                              \n"
      "   ld1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[A_ptr]], #64          \n"
      "   ld1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[A_ptr]], #64          \n"
      "   fmla    v24.4s, v1.4s, v0.s[3]                              \n"
      "   fmla    v25.4s, v2.4s, v0.s[3]                              \n"
      "   fmla    v26.4s, v3.4s, v0.s[3]                              \n"
      "   fmla    v27.4s, v4.4s, v0.s[3]                              \n"
      "   fmla    v28.4s, v5.4s, v0.s[3]                              \n"
      "   fmla    v29.4s, v6.4s, v0.s[3]                              \n"
      "   fmla    v30.4s, v7.4s, v0.s[3]                              \n"
      "   fmla    v31.4s, v8.4s, v0.s[3]                              \n"

      "   add     %[A_ptr_base], %[A_ptr_base], %[A_rows], lsl #4     \n"
      "   cmp     %[X_ptr], %[X_end]                                  \n"
      "   b.lt    2b                                                  \n"

      // Store
      "   st1     {v16.4s,v17.4s,v18.4s,v19.4s}, [%[B_dst]], #64      \n"
      "   st1     {v20.4s,v21.4s,v22.4s,v23.4s}, [%[B_dst]], #64      \n"
      "   st1     {v24.4s,v25.4s,v26.4s,v27.4s}, [%[B_dst]], #64      \n"
      "   st1     {v28.4s,v29.4s,v30.4s,v31.4s}, [%[B_dst]], #64      \n"

      // M loop tail
      "   add     %[A_src], %[A_src], %[A_row_inc]                    \n"
      "   cmp     %[A_src], %[A_end]                                  \n"
      "   b.lt    1b                                                  \n"

      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_ptr] "=&r"(A_ptr),
        [A_ptr_base] "=&r"(A_ptr_base), [X_ptr] "=&r"(X_ptr)
      : [A_rows] "r"(A_rows), [A_cols] "r"(A_cols), [X_src] "r"(X_src),
        [A_row_inc] "r"(A_row_inc), [A_rows_x3] "r"(A_rows_x3),
        [X_end] "r"(X_end), [A_end] "r"(A_end)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31", "cc", "memory");
}

#else

static void inner_loop_216(struct loop_216_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}

#endif


// Ensure the maxSVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Dimensions
#ifndef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB and MAX_VL if it has been set to 0
// (indicates default problem size requested)
#if PROBLEM_SIZE_LIMIT_KIB == 0
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n) ((n)*((m)+1)*sizeof(float32_t))

LOOP_DECL(216, OUTER_LOOP_ATTR)
{
  uint64_t M = 0;  // multiple of 16*SVLs
  uint64_t N = 0;  // multiple of 4
  const uint64_t M_base =
      (uint64_t)(MAX_VL / 2);
  if(PROBLEM_SIZE_ACTUAL(2 * M_base, M_base / 2) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    uint64_t M_temp = 0;
    while (true) {
      M_temp += M_base;
      if (PROBLEM_SIZE_ACTUAL(M_temp, (M_temp / 4)) <= (PROBLEM_SIZE_LIMIT_KIB * 1024)) {
        M = M_temp;
        // N must a multiple of 4 (which it will implicitly will be as M is
        // guarenteed to be) and should be 4x smaller than M for this
        // loop's M-to-N ratio.
        N = M_temp / 4;
      } else {
        break;
      }
    }
  }
  // Try to get at least 2 row loops and 2 column loops
  else if(PROBLEM_SIZE_ACTUAL(2 * M_base, 8) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = M_base * 2;
      N = ((PROBLEM_SIZE_LIMIT_KIB * 1024) / 4) / (M_base * 2);
      N = N - 1; // for X vector
      N = N - N % 4;
  }
  // Try to get at least 2 column loops, 1 row loop
  else if(PROBLEM_SIZE_ACTUAL(M_base, 8) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    M = M_base;
    N = ((PROBLEM_SIZE_LIMIT_KIB * 1024) / 4) / M_base;
    N = N - 1; // for X vector
    N = N - N % 4;
  }
  // Try to get at least 1 column loops, 1 row loop
  else if(PROBLEM_SIZE_ACTUAL(M_base, 4) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    M = M_base;
    N = 4;
  }
  // Error
  else {
    printf("\
ERROR LOOP 216: need at least %" PRIu64 "KiB problem size to \
construct a matrix that satisfies the 16*svlw | M and 4 | N \
constraints (%" PRIu64 " X 4 matrix of 4 byte elements)",\
             M_base * 4 * 4 / 1024, M_base);
    return 1;
  }

  struct loop_216_data data = { .m = M, .n = N, };
  ALLOC_64B(data.a, M * N, "A matrix");
  ALLOC_64B(data.x, N * 1, "x vector");
  ALLOC_64B(data.b, M * 1, "b vector");

  fill_float(data.a, M * N);
  fill_float(data.x, N * 1);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x 1\n", M, N, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N)/1024.0f);
#endif

  inner_loops_216(iters, &data);

  int checksum = 0;
#define CHECK(i)                                                          \
  {                                                                       \
    float32_t d = 0;                                                      \
    for (int j = 0; j < N; j++) d += data.a[(M * j) + i] * data.x[j];     \
    checksum += check_float(d, data.b[i], 0.001f) ? 0 : 1;                \
  }
#ifdef FULL_CHECK
  for (int m = 0; m < M; m++) CHECK(m);
#else
  CHECK(0);
  CHECK(M - 1);
  CHECK(M / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(216, passed, "%d", 0, checksum)
#endif

  return passed ? 0 : 1;
}
