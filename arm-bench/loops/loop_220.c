/*----------------------------------------------------------------------------
#
#   Loop 220: FP32 row-major matrix-vector multiply
#
#   Purpose:
#     Use of fp32 MLA and ADDV instructions.
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
    A: row-major
    X: column-vector
    B: column-vector
  Constraints -
    M: multiple of 4
    N: multiple of 4*SVLs
*/

struct loop_220_data {
  uint64_t m;
  uint64_t n;
  float32_t *restrict a;
  float32_t *restrict x;
  float32_t *restrict b;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_220(struct loop_220_data *restrict data) {
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

static void inner_loop_220(struct loop_220_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  float32_t *restrict a = data->a;
  float32_t *restrict x = data->x;
  float32_t *restrict b = data->b;
  float32_t d;
  for (uint64_t i = 0; i < m; i++) {
    d = 0;
    for (uint64_t j = 0; j < n; j++) d += a[(i * n) + j] * x[j];
    b[i] = d;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_220(struct loop_220_data *data)
LOOP_ATTR
{
  uint64_t A_rows = data->m;
  uint64_t A_cols = data->n;

  uint64_t svlw = svcntw();
  uint64_t A_rows_left_vgx4 = A_rows;
  uint64_t ZA_rows_vgx4 = svlw;
  uint64_t ZA_quarter_offset;
  float32_t *A_src = data->a;
  float32_t *X_src = data->x;
  float32_t *B_dst = data->b;

  float32_t *A_ptr;

  svbool_t p_all = svptrue_b32();
  svcount_t c_all = svptrue_c32();

  svfloat32x4_t ldx;
  svfloat32x4_t lda_0, lda_1, lda_2, lda_3;

  svfloat32x2_t ldbh_0, ldbh_1, ldbh_2, ldbh_3;
  svfloat32_t ldb_00, ldb_10, ldb_20, ldb_30,
              ldb_01, ldb_11, ldb_21, ldb_31;

  svfloat64x2_t conv_0, conv_1, conv_2, conv_3;
#define CASTX2(conv, i) svreinterpret_f32(svget2((conv), (i)))
#define TO_F32X2(conv) svcreate2(CASTX2((conv), 0), CASTX2((conv), 1))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (uint64_t A_row_idx = 0; A_row_idx < A_rows; A_row_idx += ZA_rows_vgx4) {
#if !defined(__ARM_FEATURE_SME2p1)
    svzero_za();
#endif
    ZA_rows_vgx4 = A_rows_left_vgx4 < ZA_rows_vgx4 ? A_rows_left_vgx4 : ZA_rows_vgx4;
    for (uint64_t A_col_idx = 0; A_col_idx < A_cols; A_col_idx += 4 * svlw) {

      ldx = svld1_x4(c_all, &X_src[A_col_idx]);

      A_ptr = &A_src[A_row_idx * A_cols + A_col_idx];
      for (uint64_t ZA_group_idx = 0; ZA_group_idx < ZA_rows_vgx4; ZA_group_idx += 4) {
        lda_0 = svld1_x4(c_all, A_ptr);
        lda_1 = svld1_x4(c_all, A_ptr + 1 * A_cols);
        lda_2 = svld1_x4(c_all, A_ptr + 2 * A_cols);
        lda_3 = svld1_x4(c_all, A_ptr + 3 * A_cols);
        svmla_za32_vg1x4(ZA_group_idx + 0, lda_0, ldx);
        svmla_za32_vg1x4(ZA_group_idx + 1, lda_1, ldx);
        svmla_za32_vg1x4(ZA_group_idx + 2, lda_2, ldx);
        svmla_za32_vg1x4(ZA_group_idx + 3, lda_3, ldx);

        A_ptr += A_cols * 4;
      }
    }
    A_rows_left_vgx4 -= ZA_rows_vgx4;

    for (uint64_t ZA_group_idx = 0; ZA_group_idx < ZA_rows_vgx4; ZA_group_idx += 4) {
      ZA_quarter_offset = svlw + ZA_group_idx;
#if defined(__ARM_FEATURE_SME2p1)
      conv_0 = svreadz_za64_f64_vg1x2(ZA_quarter_offset + 0);
      ldbh_0 = TO_F32X2(conv_0);
      conv_1 = svreadz_za64_f64_vg1x2(ZA_quarter_offset + 1);
      ldbh_1 = TO_F32X2(conv_1);
      conv_2 = svreadz_za64_f64_vg1x2(ZA_quarter_offset + 2);
      ldbh_2 = TO_F32X2(conv_2);
      conv_3 = svreadz_za64_f64_vg1x2(ZA_quarter_offset + 3);
      ldbh_3 = TO_F32X2(conv_3);
#else
      conv_0 = svread_za64_f64_vg1x2(ZA_quarter_offset + 0);
      ldbh_0 = TO_F32X2(conv_0);
      conv_1 = svread_za64_f64_vg1x2(ZA_quarter_offset + 1);
      ldbh_1 = TO_F32X2(conv_1);
      conv_2 = svread_za64_f64_vg1x2(ZA_quarter_offset + 2);
      ldbh_2 = TO_F32X2(conv_2);
      conv_3 = svread_za64_f64_vg1x2(ZA_quarter_offset + 3);
      ldbh_3 = TO_F32X2(conv_3);
#endif
      svadd_za32_vg1x2(ZA_group_idx + 0, ldbh_0);
      svadd_za32_vg1x2(ZA_group_idx + 1, ldbh_1);
      svadd_za32_vg1x2(ZA_group_idx + 2, ldbh_2);
      svadd_za32_vg1x2(ZA_group_idx + 3, ldbh_3);

#if defined(__ARM_FEATURE_SME2p1)
      conv_0 = svreadz_za64_f64_vg1x2(ZA_group_idx + 0);
      ldbh_0 = TO_F32X2(conv_0);
      conv_1 = svreadz_za64_f64_vg1x2(ZA_group_idx + 1);
      ldbh_1 = TO_F32X2(conv_1);
      conv_2 = svreadz_za64_f64_vg1x2(ZA_group_idx + 2);
      ldbh_2 = TO_F32X2(conv_2);
      conv_3 = svreadz_za64_f64_vg1x2(ZA_group_idx + 3);
      ldbh_3 = TO_F32X2(conv_3);
#else
      conv_0 = svread_za64_f64_vg1x2(ZA_group_idx + 0);
      ldbh_0 = TO_F32X2(conv_0);
      conv_1 = svread_za64_f64_vg1x2(ZA_group_idx + 1);
      ldbh_1 = TO_F32X2(conv_1);
      conv_2 = svread_za64_f64_vg1x2(ZA_group_idx + 2);
      ldbh_2 = TO_F32X2(conv_2);
      conv_3 = svread_za64_f64_vg1x2(ZA_group_idx + 3);
      ldbh_3 = TO_F32X2(conv_3);
#endif
      ldb_00 = svget2(ldbh_0, 0);
      ldb_01 = svget2(ldbh_0, 1);
      ldb_10 = svget2(ldbh_1, 0);
      ldb_11 = svget2(ldbh_1, 1);
      ldb_20 = svget2(ldbh_2, 0);
      ldb_21 = svget2(ldbh_2, 1);
      ldb_30 = svget2(ldbh_3, 0);
      ldb_31 = svget2(ldbh_3, 1);

      ldb_00 = svadd_x(p_all, ldb_00, ldb_01);
      ldb_10 = svadd_x(p_all, ldb_10, ldb_11);
      ldb_20 = svadd_x(p_all, ldb_20, ldb_21);
      ldb_30 = svadd_x(p_all, ldb_30, ldb_31);

      B_dst[0] = svaddv(p_all, ldb_00);
      B_dst[1] = svaddv(p_all, ldb_10);
      B_dst[2] = svaddv(p_all, ldb_20);
      B_dst[3] = svaddv(p_all, ldb_30);

      B_dst += 4;
    }
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_220(struct loop_220_data *data)
LOOP_ATTR
{
  uint64_t A_rows = data->m;
  uint64_t A_cols = data->n;

  float32_t *A_src = data->a;
  float32_t *X_src = data->x;
  float32_t *B_dst = data->b;

  float32_t *A_ptr;

  uint64_t A_row_idx, A_col_idx;
  svbool_t p_all = svptrue_b32();

  svfloat32_t acc_0, acc_1, acc_2, acc_3;
  svfloat32x4_t lda_0, lda_1, lda_2, lda_3;
  svfloat32x4_t ldx;

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c32();
#define LOADA_QUAD(q) lda_##q = svld1_x4(c_all, &A_ptr[q * A_cols])
#define LOADX_QUAD    ldx     = svld1_x4(c_all, &X_src[A_col_idx])
#else
#define LOADA(q, p)   svld1_vnum(p_all, &A_ptr[q * A_cols], p)
#define LOADX(p)      svld1_vnum(p_all, &X_src[A_col_idx], p)
#define LOADA_QUAD(q) lda_##q = \
  svcreate4(LOADA(q, 0), LOADA(q, 1), LOADA(q, 2), LOADA(q, 3))
#define LOADX_QUAD    ldx = \
  svcreate4(LOADX(0), LOADX(1), LOADX(2), LOADX(3))
#endif

#define FMLA(q, p) \
  acc_##q = svmla_x(p_all, acc_##q, svget4(lda_##q, p), svget4(ldx, p));
#define FMLA_QUAD(q) FMLA(q, 0) FMLA(q, 1) FMLA(q, 2) FMLA(q, 3)

  for (A_row_idx = 0; A_row_idx < A_rows; A_row_idx += 4) {
    acc_0 = svdup_f32(0);
    acc_1 = svdup_f32(0);
    acc_2 = svdup_f32(0);
    acc_3 = svdup_f32(0);

    A_ptr = &A_src[A_row_idx * A_cols];
    for (A_col_idx = 0; A_col_idx < A_cols; A_col_idx += svcntw() * 4) {
      LOADX_QUAD;

      LOADA_QUAD(0);
      LOADA_QUAD(1);
      LOADA_QUAD(2);
      LOADA_QUAD(3);

      FMLA_QUAD(0);
      FMLA_QUAD(1);
      FMLA_QUAD(2);
      FMLA_QUAD(3);

      A_ptr += svcntw() * 4;
    }

    B_dst[A_row_idx + 0] = svaddv(p_all, acc_0);
    B_dst[A_row_idx + 1] = svaddv(p_all, acc_1);
    B_dst[A_row_idx + 2] = svaddv(p_all, acc_2);
    B_dst[A_row_idx + 3] = svaddv(p_all, acc_3);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_220(struct loop_220_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;

  register uint64_t A_col_ptr;
  register uint64_t X_end = (uint64_t) &data->x[A_cols];
  register uint64_t A_row_end = (uint64_t) &data->a[A_rows * A_cols];
  register uint64_t A_ptr_base ;
  register uint64_t X_ptr;
  register uint64_t svlw;
  register uint64_t A_ptr;
  register uint64_t A_rows_left_vgx4;
  register uint64_t ZA_rows_vgx4;
  register uint64_t A_cols_x2 = 2 * A_cols;
  register uint64_t A_cols_x3 = 3 * A_cols;
  register uint64_t A_row_size = A_cols * sizeof(float32_t);
  // x9: slice index register for fmla and mov/movaz
  // x10: second slice index register for Za quarter offset for mov/movaz

  asm volatile("cntw %[svlw]\n"
               :[svlw] "=r"(svlw)::);

  A_rows_left_vgx4 = A_rows;
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
      // Assumption N is multiple of 4*svlw
      "   ld1w    {z0.s-z3.s}, pn8/z, [%[X_ptr]]                      \n"

      "   mov     x9, #0                                              \n"  // ZA group loop counter
      "   mov     %[A_ptr], %[A_col_ptr]                              \n"  // A pointer

      // Loop fill ZA rows, store after
      "3:                                                             \n"
      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], %[A_cols], lsl #2] \n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], %[A_cols_x2], lsl #2] \n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], %[A_cols_x3], lsl #2] \n"
      "   add     %[A_ptr], %[A_ptr], %[A_cols], lsl #4               \n"

      "   fmla    za.s[w9, #0], {z16.s-z19.s}, {z0.s-z3.s}            \n"
      "   fmla    za.s[w9, #1], {z20.s-z23.s}, {z0.s-z3.s}            \n"
      "   fmla    za.s[w9, #2], {z24.s-z27.s}, {z0.s-z3.s}            \n"
      "   fmla    za.s[w9, #3], {z28.s-z31.s}, {z0.s-z3.s}            \n"

      // Loop ZA groups increment
      "   add     x9, x9, #4                                          \n"
      "   cmp     x9, %[ZA_rows_vgx4]                                 \n"
      "   b.lt    3b                                                  \n"

      // loop columns increments
      "   addvl   %[X_ptr], %[X_ptr], #4                              \n"
      "   addvl   %[A_col_ptr], %[A_col_ptr], #4                      \n"
      "   cmp     %[X_ptr], %[X_end]                                  \n"
      "   b.lt    2b                                                  \n"

      // End loop columns, store results
      "   mov     x9, #0                                              \n"

      "4:                                                             \n"
      "   add     x10, %[svlw], x9                                    \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z17.s}, za.s[w10, 0]                         \n"
      "   movaz   {z20.s-z21.s}, za.s[w10, 1]                         \n"
      "   movaz   {z24.s-z25.s}, za.s[w10, 2]                         \n"
      "   movaz   {z28.s-z29.s}, za.s[w10, 3]                         \n"
#else
      "   mov     {z16.s-z17.s}, za.s[w10, 0]                         \n"
      "   mov     {z20.s-z21.s}, za.s[w10, 1]                         \n"
      "   mov     {z24.s-z25.s}, za.s[w10, 2]                         \n"
      "   mov     {z28.s-z29.s}, za.s[w10, 3]                         \n"
#endif
      "   fadd    za.s[w9, 0, vgx2], {z16.s-z17.s}                    \n"
      "   fadd    za.s[w9, 1, vgx2], {z20.s-z21.s}                    \n"
      "   fadd    za.s[w9, 2, vgx2], {z24.s-z25.s}                    \n"
      "   fadd    za.s[w9, 3, vgx2], {z28.s-z29.s}                    \n"

#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z17.s}, za.s[w9, 0]                          \n"
      "   movaz   {z20.s-z21.s}, za.s[w9, 1]                          \n"
      "   movaz   {z24.s-z25.s}, za.s[w9, 2]                          \n"
      "   movaz   {z28.s-z29.s}, za.s[w9, 3]                          \n"
#else
      "   mov     {z16.s-z17.s}, za.s[w9, 0]                          \n"
      "   mov     {z20.s-z21.s}, za.s[w9, 1]                          \n"
      "   mov     {z24.s-z25.s}, za.s[w9, 2]                          \n"
      "   mov     {z28.s-z29.s}, za.s[w9, 3]                          \n"
#endif
      "   fadd    z16.s, z16.s, z17.s                                 \n"
      "   fadd    z20.s, z20.s, z21.s                                 \n"
      "   fadd    z24.s, z24.s, z25.s                                 \n"
      "   fadd    z28.s, z28.s, z29.s                                 \n"
      "   faddv   s0, p0, z16.s                                       \n"
      "   faddv   s1, p0, z20.s                                       \n"
      "   faddv   s2, p0, z24.s                                       \n"
      "   faddv   s3, p0, z28.s                                       \n"

      "   stp     s0, s1, [%[B_dst]], #8                              \n"
      "   stp     s2, s3, [%[B_dst]], #8                              \n"

      "   add     x9, x9, #4                                          \n"
      "   cmp     x9, %[ZA_rows_vgx4]                                 \n"
      "   b.lt    4b                                                  \n"

      // loop rows increments
      "   sub     %[A_rows_left_vgx4], %[A_rows_left_vgx4], %[ZA_rows_vgx4] \n"
      "   madd    %[A_src], %[A_row_size], %[ZA_rows_vgx4], %[A_src]  \n"
      "   cmp     %[A_src], %[A_row_end]                              \n"
      "   b.lt 1b                                                     \n"

      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_col_ptr] "=&r"(A_col_ptr),
        [A_ptr_base] "=&r"(A_ptr_base), [X_ptr] "=&r"(X_ptr), [A_ptr] "=&r"(A_ptr),
        [A_rows_left_vgx4] "+&r"(A_rows_left_vgx4), [ZA_rows_vgx4] "+&r"(ZA_rows_vgx4)
      :
        [A_rows] "r"(A_rows), [A_cols] "r"(A_cols), [X_src] "r"(X_src),
        [A_cols_x2] "r"(A_cols_x2), [A_cols_x3] "r"(A_cols_x3),
        [A_row_size] "r"(A_row_size),
        [A_row_end] "r"(A_row_end), [X_end] "r"(X_end),
         [svlw] "r"(svlw)
      : "z0", "z1", "z2", "z3", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
        "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p8", "x9", "x10",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_220(struct loop_220_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;
  register uint64_t A_ptr = 0;
  register uint64_t X_ptr = 0;
  register uint64_t X_end = (uint64_t)&data->x[data->n];
  register uint64_t A_end = (uint64_t)&data->a[A_rows * A_cols];
  register uint64_t A_col_2 = A_cols * 2;
  register uint64_t A_col_3 = A_cols * 3;

  asm volatile(
      // constants
      "   ptrue   p0.s                                                \n"
      "   ptrue   pn8.s                                               \n"

      // Row loop
      "1:                                                             \n"
      "   mov     z4.s, #0                                            \n"
      "   mov     z5.s, #0                                            \n"
      "   mov     z6.s, #0                                            \n"
      "   mov     z7.s, #0                                            \n"

      // Column loop
      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_ptr], %[A_src]                                  \n"
      "2:                                                             \n"
      "   ld1w    {z0.s-z3.s}, pn8/z, [%[X_ptr]]                      \n"
      "   ld1w    {z16.s-z19.s}, pn8/z, [%[A_ptr]]                    \n"
      "   ld1w    {z20.s-z23.s}, pn8/z, [%[A_ptr], %[A_col_1], lsl #2]\n"
      "   ld1w    {z24.s-z27.s}, pn8/z, [%[A_ptr], %[A_col_2], lsl #2]\n"
      "   ld1w    {z28.s-z31.s}, pn8/z, [%[A_ptr], %[A_col_3], lsl #2]\n"
      "   addvl   %[X_ptr], %[X_ptr], #4                              \n"
      "   addvl   %[A_ptr], %[A_ptr], #4                              \n"
      "   fmla    z4.s, p0/m, z16.s, z0.s                             \n"
      "   fmla    z5.s, p0/m, z20.s, z0.s                             \n"
      "   fmla    z6.s, p0/m, z24.s, z0.s                             \n"
      "   fmla    z7.s, p0/m, z28.s, z0.s                             \n"
      "   fmla    z4.s, p0/m, z17.s, z1.s                             \n"
      "   fmla    z5.s, p0/m, z21.s, z1.s                             \n"
      "   fmla    z6.s, p0/m, z25.s, z1.s                             \n"
      "   fmla    z7.s, p0/m, z29.s, z1.s                             \n"
      "   fmla    z4.s, p0/m, z18.s, z2.s                             \n"
      "   fmla    z5.s, p0/m, z22.s, z2.s                             \n"
      "   fmla    z6.s, p0/m, z26.s, z2.s                             \n"
      "   fmla    z7.s, p0/m, z30.s, z2.s                             \n"
      "   fmla    z4.s, p0/m, z19.s, z3.s                             \n"
      "   fmla    z5.s, p0/m, z23.s, z3.s                             \n"
      "   fmla    z6.s, p0/m, z27.s, z3.s                             \n"
      "   fmla    z7.s, p0/m, z31.s, z3.s                             \n"
      "   cmp     %[X_ptr], %[X_end]                                  \n"
      "   b.lt    2b                                                  \n"

      "   faddv   s4, p0, z4.s                                        \n"
      "   faddv   s5, p0, z5.s                                        \n"
      "   faddv   s6, p0, z6.s                                        \n"
      "   faddv   s7, p0, z7.s                                        \n"

      "   stp     s4, s5, [%[B_dst]], #8                              \n"
      "   stp     s6, s7, [%[B_dst]], #8                              \n"

      "   add     %[A_src], %[A_src], %[A_cols], lsl #4               \n"
      "   cmp     %[A_src], %[A_end]                                  \n"
      "   b.lt 1b                                                     \n"

      : [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_ptr] "=&r"(A_ptr),
        [X_ptr] "=&r"(X_ptr)
      : [X_src] "r"(X_src), [X_end] "r"(X_end), [A_end] "r"(A_end),
        [A_col_1] "r"(A_cols), [A_col_2] "r"(A_col_2), [A_col_3] "r"(A_col_3),
        [A_cols] "r"(A_cols)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
        "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
        "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
        "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_220(struct loop_220_data *data)
LOOP_ATTR
{
  register uint64_t svlw = 0;

  asm volatile("cntw %[svlw]\n"
               :[svlw] "=&r"(svlw)::);

  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;

  register uint64_t A_ptr;
  register uint64_t X_ptr;
  register uint64_t X_end = (uint64_t)&data->x[data->n];
  register uint64_t A_end = (uint64_t)&data->a[A_rows * A_cols];
  register uint64_t A_ptr_base;

  asm volatile(
      // constants
      "   ptrue   p0.s                                                \n"

      // Row loop
      "1:                                                             \n"

      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_ptr_base], %[A_src]                             \n"

      "   mov     z4.s, #0                                            \n"
      "   mov     z5.s, #0                                            \n"
      "   mov     z6.s, #0                                            \n"
      "   mov     z7.s, #0                                            \n"

      // Column loop
      "2:                                                             \n"
      "   mov     %[A_ptr], %[A_ptr_base]                             \n"

      "    ld1w    {z0.s}, p0/z, [%[X_ptr]]                           \n"
      "    ld1w    {z1.s}, p0/z, [%[X_ptr], #1, mul vl]               \n"
      "    ld1w    {z2.s}, p0/z, [%[X_ptr], #2, mul vl]               \n"
      "    ld1w    {z3.s}, p0/z, [%[X_ptr], #3, mul vl]               \n"
      "    addvl   %[X_ptr], %[X_ptr], #4                             \n"

      "    ld1w    {z16.s}, p0/z, [%[A_ptr]]                          \n"
      "    ld1w    {z17.s}, p0/z, [%[A_ptr], #1, mul vl]              \n"
      "    ld1w    {z18.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z19.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_cols], lsl #2              \n"
      "    fmla    z4.s, p0/m, z16.s, z0.s                            \n"
      "    fmla    z4.s, p0/m, z17.s, z1.s                            \n"
      "    fmla    z4.s, p0/m, z18.s, z2.s                            \n"
      "    fmla    z4.s, p0/m, z19.s, z3.s                            \n"

      "    ld1w    {z20.s}, p0/z, [%[A_ptr]]                          \n"
      "    ld1w    {z21.s}, p0/z, [%[A_ptr], #1, mul vl]              \n"
      "    ld1w    {z22.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z23.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_cols], lsl #2              \n"
      "    fmla    z5.s, p0/m, z20.s, z0.s                            \n"
      "    fmla    z5.s, p0/m, z21.s, z1.s                            \n"
      "    fmla    z5.s, p0/m, z22.s, z2.s                            \n"
      "    fmla    z5.s, p0/m, z23.s, z3.s                            \n"

      "    ld1w    {z24.s}, p0/z, [%[A_ptr]]                          \n"
      "    ld1w    {z25.s}, p0/z, [%[A_ptr], #1, mul vl]              \n"
      "    ld1w    {z26.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z27.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    add     %[A_ptr], %[A_ptr], %[A_cols], lsl #2              \n"
      "    fmla    z6.s, p0/m, z24.s, z0.s                            \n"
      "    fmla    z6.s, p0/m, z25.s, z1.s                            \n"
      "    fmla    z6.s, p0/m, z26.s, z2.s                            \n"
      "    fmla    z6.s, p0/m, z27.s, z3.s                            \n"

      "    ld1w    {z28.s}, p0/z, [%[A_ptr]]                          \n"
      "    ld1w    {z29.s}, p0/z, [%[A_ptr], #1, mul vl]              \n"
      "    ld1w    {z30.s}, p0/z, [%[A_ptr], #2, mul vl]              \n"
      "    ld1w    {z31.s}, p0/z, [%[A_ptr], #3, mul vl]              \n"
      "    fmla    z7.s, p0/m, z28.s, z0.s                            \n"
      "    fmla    z7.s, p0/m, z29.s, z1.s                            \n"
      "    fmla    z7.s, p0/m, z30.s, z2.s                            \n"
      "    fmla    z7.s, p0/m, z31.s, z3.s                            \n"

      "    addvl   %[A_ptr_base], %[A_ptr_base], #4                   \n"
      "    cmp     %[X_ptr], %[X_end]                                 \n"
      "    b.lt    2b                                                 \n"

      "    faddv   s4, p0, z4.s                                       \n"
      "    faddv   s5, p0, z5.s                                       \n"
      "    faddv   s6, p0, z6.s                                       \n"
      "    faddv   s7, p0, z7.s                                       \n"

      "    stp     s4, s5, [%[B_dst]], #8                             \n"
      "    stp     s6, s7, [%[B_dst]], #8                             \n"

      "    add     %[A_src], %[A_src], %[A_cols], lsl #4              \n"
      "    cmp     %[A_src], %[A_end]                                 \n"
      "    b.lt 1b                                                    \n"

      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_ptr] "=&r"(A_ptr),
        [X_ptr] "=&r"(X_ptr), [A_ptr_base] "=&r" (A_ptr_base)
      : [A_rows] "r"(A_rows), [X_src] "r"(X_src), [A_cols] "r"(A_cols),
        [X_end] "r"(X_end), [A_end] "r"(A_end)
      : "p0",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
        "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
        "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
        "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_220(struct loop_220_data *data) {
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t B_dst = (uint64_t)data->b;

  register uint64_t A_end = (uint64_t)&data->a[A_rows * A_cols];
  register uint64_t X_end = (uint64_t)&data->x[A_cols];
  register uint64_t A_col_ptr;
  register uint64_t A_ptr;
  register uint64_t X_ptr;
  register uint64_t A_cols_x4 = A_cols * 4;

  asm volatile(
      // Row loop head
      "1:                                                             \n"
      "   movi    v20.4s, #0                                          \n"
      "   movi    v21.4s, #0                                          \n"
      "   movi    v22.4s, #0                                          \n"
      "   movi    v23.4s, #0                                          \n"

      // Column loop
      // N must be multiple of 16
      // M must be multiple of 4
      "   mov     %[X_ptr], %[X_src]                                  \n"
      "   mov     %[A_col_ptr], %[A_src]                              \n"
      "2:                                                             \n"
      "   mov     %[A_ptr], %[A_col_ptr]                              \n"
      "   ld1     {v0.4s,v1.4s,v2.4s,v3.4s}, [%[X_ptr]], #64          \n"

      "   ld1     {v4.4s,v5.4s,v6.4s,v7.4s}, [%[A_ptr]], %[A_cols_x4] \n"
      "   ld1     {v8.4s,v9.4s,v10.4s,v11.4s}, [%[A_ptr]], %[A_cols_x4] \n"
      "   fmla    v20.4s, v4.4s, v0.4s                                \n"
      "   fmla    v20.4s, v5.4s, v1.4s                                \n"
      "   fmla    v20.4s, v6.4s, v2.4s                                \n"
      "   fmla    v20.4s, v7.4s, v3.4s                                \n"
      "   fmla    v21.4s, v8.4s, v0.4s                                \n"
      "   fmla    v21.4s, v9.4s, v1.4s                                \n"
      "   fmla    v21.4s, v10.4s, v2.4s                               \n"
      "   fmla    v21.4s, v11.4s, v3.4s                               \n"

      "   ld1     {v12.4s,v13.4s,v14.4s,v15.4s}, [%[A_ptr]], %[A_cols_x4] \n"
      "   ld1     {v16.4s,v17.4s,v18.4s,v19.4s}, [%[A_ptr]], %[A_cols_x4] \n"
      "   fmla    v22.4s, v12.4s, v0.4s                               \n"
      "   fmla    v22.4s, v13.4s, v1.4s                               \n"
      "   fmla    v22.4s, v14.4s, v2.4s                               \n"
      "   fmla    v22.4s, v15.4s, v3.4s                               \n"
      "   fmla    v23.4s, v16.4s, v0.4s                               \n"
      "   fmla    v23.4s, v17.4s, v1.4s                               \n"
      "   fmla    v23.4s, v18.4s, v2.4s                               \n"
      "   fmla    v23.4s, v19.4s, v3.4s                               \n"

      "   add     %[A_col_ptr], %[A_col_ptr], #64                     \n"
      "   cmp     %[X_ptr], %[X_end]                                  \n"
      "   b.lt    2b                                                  \n"

      // Reduce
      "   faddp   v24.4s, v20.4s, v21.4s                              \n"
      "   faddp   v25.4s, v22.4s, v23.4s                              \n"

      "   faddp   v26.4s, v24.4s, v25.4s                              \n"

      // Store
      "   st1     {v26.4s}, [%[B_dst]], #16                           \n"

      // M loop tail
      "   add     %[A_src], %[A_src], %[A_cols], lsl #4               \n"
      "   cmp     %[A_src], %[A_end]                                  \n"
      "   b.lt    1b                                                  \n"
      :
        [A_src] "+&r"(A_src), [B_dst] "+&r"(B_dst), [A_ptr] "=&r"(A_ptr),
        [A_col_ptr] "=&r"(A_col_ptr), [X_ptr] "=&r"(X_ptr)
      : [A_rows] "r"(A_rows), [A_cols] "r"(A_cols), [X_src] "r"(X_src),
        [A_cols_x4] "r"(A_cols_x4), [X_end] "r"(X_end), [A_end] "r"(A_end)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "cc", "memory");
}

#else

static void inner_loop_220(struct loop_220_data *data) {
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

LOOP_DECL(220, OUTER_LOOP_ATTR)
{
  uint64_t M = 0;  // multiple of 4
  uint64_t N = 0;  // multiple of 4*SVLs
  const uint64_t N_base =
      (uint64_t)(MAX_VL / 8);
  // Try M = N * 4
  if(PROBLEM_SIZE_ACTUAL(N_base * 4, N_base) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    uint64_t N_temp = 0;
    while (true) {
      N_temp += N_base;
      if (PROBLEM_SIZE_ACTUAL(N_temp * 4, N_temp) <= (PROBLEM_SIZE_LIMIT_KIB * 1024)) {
        N = N_temp;
        M = N_temp * 4;
      } else {
        break;
      }
    }
  }
  // Try to get at least 2 row loops and 2 column loops
  else if(PROBLEM_SIZE_ACTUAL(2 * 4, 2 * N_base) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    N = N_base * 2;
    M = ((PROBLEM_SIZE_LIMIT_KIB * 1024) / 4) / (N_base * 2 + 1);
    M -= M % 4;
  }
  // Try to get at least 2 column loops, 1 row loop
  else if(PROBLEM_SIZE_ACTUAL(2 * 4, N_base) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    N = N_base;
    M = ((PROBLEM_SIZE_LIMIT_KIB * 1024) / 4) / (N_base * 2 + 1);
    M -= M % 4;
  }
  // Try to get at least 1 column loops, 1 row loop
  else if(PROBLEM_SIZE_ACTUAL(4, N_base) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    N = N_base;
    M = 4;
  }
  // Error
  else {
    printf("\
ERROR LOOP 220: need at least %" PRIu64 "KiB problem size to \
construct a matrix that satisfies the 4 | M and 4*SVLs | N \
constraints (4 X %" PRIu64 " matrix of 4 byte elements)",\
             4 * N_base * 4 / 1024, N_base);
    return 1;
  }

  struct loop_220_data data = { .m = M, .n = N, };
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

  inner_loops_220(iters, &data);

  int checksum = 0;
#define CHECK(i)                                                            \
  {                                                                         \
    float32_t d = 0;                                                        \
    uint64_t ii = i;                                                        \
    for (uint64_t j = 0; j < N; j++) d += data.a[(ii * N) + j] * data.x[j]; \
    checksum += check_float(d, data.b[i], 0.001f) ? 0 : 1;                  \
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
  FINALISE_LOOP_I(220, passed, "%d", 0, checksum)
#endif

  return passed ? 0 : 1;
}
