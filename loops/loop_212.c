/*----------------------------------------------------------------------------
#
#   Loop 212: 4-bit-FP32 col-major interleaved matrix-vector multiply
#
#   Purpose:
#     Use of 4-bit dequantization (LUT) and DOT instructions.
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
    A: col-major 4-way interleaved
    X: column-vector
    A scaling factors: col-major
    X scaling factors: column-vector
    B: row-major
  Constraints -
    M: multiple of 16 * SVLw (preferably 8 * SVLw^2)
    N: multiple of BLOCK_SIZE
*/

#define BLOCK_SIZE 32
#define BLOCK_SIZE_STR "32"

struct loop_212_data {
  uint64_t m;
  uint64_t n;
  // Dequantized element is fp32
  // size = M * N / 2 bytes
  uint8_t *restrict a;
  int8_t *restrict x;
  float *restrict b;
  // size = M * N / 32 * 4 = M * N / 16 bytes
  float *restrict a_scales;
  // size = N / 32 * 4 = N / 16 bytes
  float *restrict x_scales;
  // 512b Look-up table for dequantization.
  uint8_t lut[64];
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_212(struct loop_212_data *restrict data) {
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
#define LOOP_ATTR SME_ZA_ZT0_ATTR
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

static void inner_loop_212(struct loop_212_data *data) {
  const uint64_t block_size = BLOCK_SIZE;
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *a = data->a;
  int8_t *x = data->x;
  float *b = data->b;
  float *a_scales = data->a_scales;
  float *x_scales = data->x_scales;
  uint64_t col_inc = 2 * m;
  uint8_t first_byte, second_byte;
  int32_t block_dot;
  float row_sum;
  uint8_t *row_ptr;
  uint64_t a_idx, x_idx;
  for (uint64_t row = 0; row < m; row++) {
    row_ptr = &a[2 * row];
    row_sum = 0;
    for (uint64_t col = 0; col < n; col += block_size) {
      block_dot = 0;
      a_idx = col / 4 * col_inc;
      x_idx = col;
      for (uint64_t block = 0; block < block_size / 4; block++) {
        first_byte = row_ptr[a_idx];
        second_byte = row_ptr[a_idx + 1];
        block_dot += ((int8_t)(first_byte & 0x0F) - 8) * x[x_idx];
        block_dot += ((int8_t)(first_byte >> 4) - 8) * x[x_idx + 1];
        block_dot += ((int8_t)(second_byte & 0x0F) - 8) * x[x_idx + 2];
        block_dot += ((int8_t)(second_byte >> 4) - 8) * x[x_idx + 3];
        a_idx += col_inc;
        x_idx += 4;
      }
      row_sum += block_dot * a_scales[(col / block_size) * m + row] * x_scales[col / block_size];
    }
    b[row] = row_sum;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_212(struct loop_212_data *data)
LOOP_ATTR
{
  const uint64_t A_rows = data->m;
  const uint64_t A_cols = data->n;

  uint8_t *A_src = data->a;
  float *A_sf_src = data->a_scales;
  int8_t *X_src = data->x;
  float *X_sf_src = data->x_scales;
  float *B_dst = data->b;
  const uint8_t *lut = data->lut;

  const uint64_t A_col_inc = A_rows * 2;
  const uint64_t A_row_inc = svcntb() * 4;

  svbool_t p_all = svptrue_b8();
  svcount_t c_all = svptrue_c8();

  svfloat32_t row_acc_0, row_acc_1, row_acc_2, row_acc_3,
    row_acc_4, row_acc_5, row_acc_6, row_acc_7;

  svint8_t block_x_0, block_x_1;
  svuint8x4_t lda_0;
  svint8x2_t luti_0, luti_1, luti_2, luti_3;
  svfloat32x4_t block_acc_0, block_acc_1;
  svfloat32x4_t scaling_factors_a_0, scaling_factors_a_1;
  svfloat32_t sf_0, sf_1, sf_2, sf_3, sf_4, sf_5, sf_6, sf_7;
  svfloat32_t scaling_factors_x;
  svfloat32x4_t stb_0, stb_1;

  svldr_zt(0, lut);

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  // One row is an interleaved row: 4 4-bit elements of a column represent one element in the interleaved row -> one 16-bit element
  for(uint64_t A_row_idx = 0; A_row_idx < A_rows * 2; A_row_idx += A_row_inc) {
    row_acc_0 = svdup_f32(0);
    row_acc_1 = svdup_f32(0);
    row_acc_2 = svdup_f32(0);
    row_acc_3 = svdup_f32(0);
    row_acc_4 = svdup_f32(0);
    row_acc_5 = svdup_f32(0);
    row_acc_6 = svdup_f32(0);
    row_acc_7 = svdup_f32(0);
    // Column is interleaved column: 4 times less columns because they are hidden in row.
    for(uint64_t block = 0; block < A_cols / BLOCK_SIZE; block++) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif
      // Load 32 = BLOCK size x values
      block_x_0 = svld1rq(p_all, &X_src[block * BLOCK_SIZE]);
      block_x_1 = svld1rq(p_all, &X_src[block * BLOCK_SIZE + BLOCK_SIZE / 2]);

#define LUTI_BLOCK_INTRINSIC(zix, idx, inc)                   \
      lda_0 = svld1_x4(c_all, &A_src[A_row_idx + (block * BLOCK_SIZE / 4 + (inc)) * A_col_inc]); \
      luti_0 = svluti4_lane_zt_s8_x2(0, svget4(lda_0, 0), 0); \
      luti_1 = svluti4_lane_zt_s8_x2(0, svget4(lda_0, 1), 0); \
      luti_2 = svluti4_lane_zt_s8_x2(0, svget4(lda_0, 2), 0); \
      luti_3 = svluti4_lane_zt_s8_x2(0, svget4(lda_0, 3), 0); \
      svdot_lane_za32_vg1x4(0, svcreate4(svget2(luti_0, 0), svget2(luti_0, 1), svget2(luti_1, 0), svget2(luti_1, 1)), (zix), (idx)); \
      svdot_lane_za32_vg1x4(1, svcreate4(svget2(luti_2, 0), svget2(luti_2, 1), svget2(luti_3, 0), svget2(luti_3, 1)), (zix), (idx));
// End define LUTI_BLOCK_INTRINSIC

      // Repeat for all columns in this block
      LUTI_BLOCK_INTRINSIC(block_x_0, 0, 0)
      LUTI_BLOCK_INTRINSIC(block_x_0, 1, 1)
      LUTI_BLOCK_INTRINSIC(block_x_0, 2, 2)
      LUTI_BLOCK_INTRINSIC(block_x_0, 3, 3)
      LUTI_BLOCK_INTRINSIC(block_x_1, 0, 4)
      LUTI_BLOCK_INTRINSIC(block_x_1, 1, 5)
      LUTI_BLOCK_INTRINSIC(block_x_1, 2, 6)
      LUTI_BLOCK_INTRINSIC(block_x_1, 3, 7)

      // Load Z registers with intermediate values in ZA array
      // Convert all intermediate registers from int32 to float32
#if defined(__ARM_FEATURE_SME2p1)
      block_acc_0 = svcvt_f32(svreadz_za32_s32_vg1x4(0));
      block_acc_1 = svcvt_f32(svreadz_za32_s32_vg1x4(1));
#else
      block_acc_0 = svcvt_f32(svread_za32_s32_vg1x4(0));
      block_acc_1 = svcvt_f32(svread_za32_s32_vg1x4(1));
#endif
      scaling_factors_x = svdup_f32(X_sf_src[block]);
      scaling_factors_a_0 = svld1_x4(c_all, &A_sf_src[A_row_idx / 2 + block * A_rows]);
      scaling_factors_a_1 = svld1_vnum_x4(c_all, &A_sf_src[A_row_idx / 2 + block * A_rows], 4);

      // Combine A and X scaling factors (distributive for Q4_0 and dot product)
      sf_0 = svmul_x(p_all, svget4(scaling_factors_a_0, 0), scaling_factors_x);
      sf_1 = svmul_x(p_all, svget4(scaling_factors_a_0, 1), scaling_factors_x);
      sf_2 = svmul_x(p_all, svget4(scaling_factors_a_0, 2), scaling_factors_x);
      sf_3 = svmul_x(p_all, svget4(scaling_factors_a_0, 3), scaling_factors_x);
      sf_4 = svmul_x(p_all, svget4(scaling_factors_a_1, 0), scaling_factors_x);
      sf_5 = svmul_x(p_all, svget4(scaling_factors_a_1, 1), scaling_factors_x);
      sf_6 = svmul_x(p_all, svget4(scaling_factors_a_1, 2), scaling_factors_x);
      sf_7 = svmul_x(p_all, svget4(scaling_factors_a_1, 3), scaling_factors_x);

      // Multiply intermediates by scaling factors and accumulate for full accumulation
      row_acc_0 = svmla_x(p_all, row_acc_0, svget4(block_acc_0, 0), sf_0);
      row_acc_1 = svmla_x(p_all, row_acc_1, svget4(block_acc_0, 1), sf_1);
      row_acc_2 = svmla_x(p_all, row_acc_2, svget4(block_acc_0, 2), sf_2);
      row_acc_3 = svmla_x(p_all, row_acc_3, svget4(block_acc_0, 3), sf_3);
      row_acc_4 = svmla_x(p_all, row_acc_4, svget4(block_acc_1, 0), sf_4);
      row_acc_5 = svmla_x(p_all, row_acc_5, svget4(block_acc_1, 1), sf_5);
      row_acc_6 = svmla_x(p_all, row_acc_6, svget4(block_acc_1, 2), sf_6);
      row_acc_7 = svmla_x(p_all, row_acc_7, svget4(block_acc_1, 3), sf_7);
    }

    stb_0 = svcreate4(row_acc_0, row_acc_1, row_acc_2, row_acc_3);
    stb_1 = svcreate4(row_acc_4, row_acc_5, row_acc_6, row_acc_7);

    svst1(c_all, &B_dst[A_row_idx / 2], stb_0);
    svst1_vnum(c_all, &B_dst[A_row_idx / 2], 4, stb_1);
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_212(struct loop_212_data *data)
LOOP_ATTR
{
  const uint64_t A_rows = data->m;
  const uint64_t A_cols = data->n;

  uint8_t *A_src = data->a;
  float *A_sf_src = data->a_scales;
  int8_t *X_src = data->x;
  float *X_sf_src = data->x_scales;
  float *B_dst = data->b;
#if defined(__ARM_FEATURE_LUT)
  uint8_t *lut = data->lut;
#endif

  const uint64_t A_col_inc = A_rows * 2;
  const uint64_t A_row_inc = svcntb() * 4;

  svbool_t p_all = svptrue_b8();
#if defined(__ARM_FEATURE_LUT)
  svint8_t lut_table = svreinterpret_s8(svld1(p_all, lut));
#else
  svint8_t dequantize_mask = svdup_s8(0xF);
  svint8_t dequantize_temp;
#endif

  svfloat32_t row_acc_0, row_acc_1, row_acc_2, row_acc_3,
    row_acc_4, row_acc_5, row_acc_6, row_acc_7;
  svint32_t block_acc_0, block_acc_1, block_acc_2, block_acc_3,
    block_acc_4, block_acc_5, block_acc_6, block_acc_7;

  svint8_t block_x_0, block_x_1;
  svfloat32_t scaling_factors_a_0, scaling_factors_a_1,
    scaling_factors_a_2, scaling_factors_a_3,
    scaling_factors_a_4, scaling_factors_a_5,
    scaling_factors_a_6, scaling_factors_a_7;
  svfloat32_t sf_0, sf_1, sf_2, sf_3, sf_4, sf_5, sf_6, sf_7;
  svfloat32_t scaling_factors_x;

  svuint8_t lda_0, lda_1, lda_2, lda_3;
  svint8_t deq_dst_0, deq_dst_1, deq_dst_2, deq_dst_3,
    deq_dst_4, deq_dst_5, deq_dst_6, deq_dst_7;

  uint64_t A_col_idx, A_sf_idx;

  // One row is an interleaved row: 4 4-bit elements of a column represent one element in the interleaved row -> one 16-bit element
  for(uint64_t A_row_idx = 0; A_row_idx < A_rows * 2; A_row_idx += A_row_inc) {
    row_acc_0 = svdup_f32(0);
    row_acc_1 = svdup_f32(0);
    row_acc_2 = svdup_f32(0);
    row_acc_3 = svdup_f32(0);
    row_acc_4 = svdup_f32(0);
    row_acc_5 = svdup_f32(0);
    row_acc_6 = svdup_f32(0);
    row_acc_7 = svdup_f32(0);
    // Column is interleaved column: 4 times less columns because they are hidden in row.
    for(uint64_t block = 0; block < A_cols / BLOCK_SIZE; block++) {
      block_acc_0 = svdup_s32(0);
      block_acc_1 = svdup_s32(0);
      block_acc_2 = svdup_s32(0);
      block_acc_3 = svdup_s32(0);
      block_acc_4 = svdup_s32(0);
      block_acc_5 = svdup_s32(0);
      block_acc_6 = svdup_s32(0);
      block_acc_7 = svdup_s32(0);

      // Load 32 = BLOCK size x values
      block_x_0 = svld1rq(p_all, &X_src[block * BLOCK_SIZE]);
      block_x_1 = svld1rq(p_all, &X_src[block * BLOCK_SIZE + BLOCK_SIZE / 2]);
#if defined(__ARM_FEATURE_LUT)
#define SVEI_DEQUANTIZE(SRC_VEC, DST_VEC_1, DST_VEC_2)                  \
      (DST_VEC_1) = svluti4_lane(lut_table, SRC_VEC, 0);  \
      (DST_VEC_2) = svluti4_lane(lut_table, SRC_VEC, 1);
// end SVEI_DEQUANTIZE
#else
#define SVEI_DEQUANTIZE(SRC_VEC, DST_VEC_1, DST_VEC_2)                  \
      dequantize_temp = svand_x(p_all, svreinterpret_s8((SRC_VEC)), dequantize_mask);     \
      dequantize_temp = svsub_x(p_all, dequantize_temp, 8);             \
      (SRC_VEC) = svlsr_n_u8_x(p_all, (SRC_VEC), (uint16_t) 4); \
      (SRC_VEC) = svsub_x(p_all, (SRC_VEC), 8);                         \
      (DST_VEC_1) = svzip1_s8(dequantize_temp, svreinterpret_s8((SRC_VEC)));              \
      (DST_VEC_2) = svzip2_s8(dequantize_temp, svreinterpret_s8((SRC_VEC)));
// end SVEI_DEQUANTIZE
#endif

#define SVEI_BLOCK(zix, idx, inc)                                       \
      /* load next A col value for every row */                         \
      A_col_idx = A_row_idx + (block * BLOCK_SIZE / 4 + (inc)) * A_col_inc; \
      lda_0 = svreinterpret_u8(svld1(p_all, &A_src[A_col_idx]));        \
      lda_1 = svreinterpret_u8(svld1_vnum(p_all, &A_src[A_col_idx], 1));\
      lda_2 = svreinterpret_u8(svld1_vnum(p_all, &A_src[A_col_idx], 2));\
      lda_3 = svreinterpret_u8(svld1_vnum(p_all, &A_src[A_col_idx], 3));\
      /* 4bit-8bit dequantization */                                    \
      SVEI_DEQUANTIZE(lda_0, deq_dst_0, deq_dst_1)                      \
      SVEI_DEQUANTIZE(lda_1, deq_dst_2, deq_dst_3)                      \
      SVEI_DEQUANTIZE(lda_2, deq_dst_4, deq_dst_5)                      \
      SVEI_DEQUANTIZE(lda_3, deq_dst_6, deq_dst_7)                      \
      /* SDOT between line0 of block x and y in ZA array */             \
      block_acc_0 = svdot_lane(block_acc_0, deq_dst_0, (zix), (idx));   \
      block_acc_1 = svdot_lane(block_acc_1, deq_dst_1, (zix), (idx));   \
      block_acc_2 = svdot_lane(block_acc_2, deq_dst_2, (zix), (idx));   \
      block_acc_3 = svdot_lane(block_acc_3, deq_dst_3, (zix), (idx));   \
      block_acc_4 = svdot_lane(block_acc_4, deq_dst_4, (zix), (idx));   \
      block_acc_5 = svdot_lane(block_acc_5, deq_dst_5, (zix), (idx));   \
      block_acc_6 = svdot_lane(block_acc_6, deq_dst_6, (zix), (idx));   \
      block_acc_7 = svdot_lane(block_acc_7, deq_dst_7, (zix), (idx));
// end define SVE2_BLOCK

      // Repeat for all columns in this block
      SVEI_BLOCK(block_x_0, 0, 0)
      SVEI_BLOCK(block_x_0, 1, 1)
      SVEI_BLOCK(block_x_0, 2, 2)
      SVEI_BLOCK(block_x_0, 3, 3)
      SVEI_BLOCK(block_x_1, 0, 4)
      SVEI_BLOCK(block_x_1, 1, 5)
      SVEI_BLOCK(block_x_1, 2, 6)
      SVEI_BLOCK(block_x_1, 3, 7)

      scaling_factors_x = svdup_f32(X_sf_src[block]);

      A_sf_idx = A_row_idx / 2 + block * A_rows;
      scaling_factors_a_0 = svld1(p_all, &A_sf_src[A_sf_idx]);
      scaling_factors_a_1 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 1);
      scaling_factors_a_2 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 2);
      scaling_factors_a_3 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 3);
      scaling_factors_a_4 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 4);
      scaling_factors_a_5 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 5);
      scaling_factors_a_6 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 6);
      scaling_factors_a_7 = svld1_vnum(p_all, &A_sf_src[A_sf_idx], 7);

      // Combine A and X scaling factors (distributive for Q4_0 and dot product)
      sf_0 = svmul_x(p_all, scaling_factors_a_0, scaling_factors_x);
      sf_1 = svmul_x(p_all, scaling_factors_a_1, scaling_factors_x);
      sf_2 = svmul_x(p_all, scaling_factors_a_2, scaling_factors_x);
      sf_3 = svmul_x(p_all, scaling_factors_a_3, scaling_factors_x);
      sf_4 = svmul_x(p_all, scaling_factors_a_4, scaling_factors_x);
      sf_5 = svmul_x(p_all, scaling_factors_a_5, scaling_factors_x);
      sf_6 = svmul_x(p_all, scaling_factors_a_6, scaling_factors_x);
      sf_7 = svmul_x(p_all, scaling_factors_a_7, scaling_factors_x);

      // Multiply intermediates by scaling factors and accumulate for full accumulation
      row_acc_0 = svmla_x(p_all, row_acc_0, svcvt_f32_x(p_all, block_acc_0), sf_0);
      row_acc_1 = svmla_x(p_all, row_acc_1, svcvt_f32_x(p_all, block_acc_1), sf_1);
      row_acc_2 = svmla_x(p_all, row_acc_2, svcvt_f32_x(p_all, block_acc_2), sf_2);
      row_acc_3 = svmla_x(p_all, row_acc_3, svcvt_f32_x(p_all, block_acc_3), sf_3);
      row_acc_4 = svmla_x(p_all, row_acc_4, svcvt_f32_x(p_all, block_acc_4), sf_4);
      row_acc_5 = svmla_x(p_all, row_acc_5, svcvt_f32_x(p_all, block_acc_5), sf_5);
      row_acc_6 = svmla_x(p_all, row_acc_6, svcvt_f32_x(p_all, block_acc_6), sf_6);
      row_acc_7 = svmla_x(p_all, row_acc_7, svcvt_f32_x(p_all, block_acc_7), sf_7);
    }

    svst1_f32(p_all, &B_dst[A_row_idx / 2], row_acc_0);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 1, row_acc_1);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 2, row_acc_2);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 3, row_acc_3);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 4, row_acc_4);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 5, row_acc_5);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 6, row_acc_6);
    svst1_vnum_f32(p_all, &B_dst[A_row_idx / 2], 7, row_acc_7);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_212(struct loop_212_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t A_sf_src = (uint64_t)data->a_scales;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t X_sf_src = (uint64_t)data->x_scales;
  register uint64_t B_dst = (uint64_t)data->b;
  register uint64_t lut = (uint64_t)&data->lut;
  register uint64_t A_rows_end = (uint64_t) &data->a[A_rows * 2];
  register uint64_t A_cols_ptr = (uint64_t) data->a;
  register uint64_t A_cols_end = (uint64_t) &data->a[A_cols * A_rows / 2];
  register uint64_t A_col_inc = A_rows * 2;
  register uint64_t A_sf_col_inc = A_rows * 4;
  register uint64_t A_sf_row_inc;
  register uint64_t A_sf_col_ptr = 0;
  register uint64_t A_row_inc;
  register uint64_t X_sf_ptr;
  register uint64_t X_ptr;
  // x9: slice index register for sdot and mova

  // Scaling factors are 4 byte values, and we need 4 * cnth values.
  asm volatile("cnth %[A_sf_row_inc], all, mul #16 \n"
               :[A_sf_row_inc] "=&r"(A_sf_row_inc)::);

  // Loading 4 Z registers from A in every loop
  asm volatile("cntb %[A_row_inc], all, mul #4 \n"
               :[A_row_inc] "=&r"(A_row_inc)::);

  asm volatile(
  " ptrue p2.b, all                                                   \n"
  " ptrue pn8.b                                                       \n"
#if defined(__ARM_FEATURE_SME2p1)
  " zero {za}                                                         \n"
#endif
  " ldr zt0, [%[lut]]                                                 \n"
  " mov x9, #0                                                        \n"

  // row loop
  "1:                                                                 \n"
    " mov %[A_cols_ptr], %[A_src]                                     \n"
    " mov %[A_sf_col_ptr], %[A_sf_src]                                \n"
    " mov %[X_sf_ptr], %[X_sf_src]                                    \n"
    " mov %[X_ptr], %[X_src]                                          \n"

    // initialize outer accumutator to zero
    " dup z24.s, #0                                                   \n"
    " dup z25.s, #0                                                   \n"
    " dup z26.s, #0                                                   \n"
    " dup z27.s, #0                                                   \n"
    " dup z28.s, #0                                                   \n"
    " dup z29.s, #0                                                   \n"
    " dup z30.s, #0                                                   \n"
    " dup z31.s, #0                                                   \n"

    // Column loop
    // Handles 1 block at a time = 32 columns = 16 bytes of columns
    "2:                                                               \n"
#if !defined(__ARM_FEATURE_SME2p1)
      " zero {za}                                                     \n"
#endif
      // Process columns blocks

      // load next 32 x values
      " ld1rqb  {z0.b}, p2/z, [%[X_ptr], #0]                          \n"
      " ld1rqb  {z1.b}, p2/z, [%[X_ptr], #16]                         \n"
      " add %[X_ptr], %[X_ptr], #32                                   \n"

#define LUTI_BLOCK(zix)                                                 \
      /* load next A col value for every row */                         \
      " ld1b { z12.b - z15.b }, pn8/z, [%[A_cols_ptr]]              \n" \
      /* 4bit-8bit dequantization */                                    \
      " luti4 { z4.b - z5.b }, zt0, z12[0]                          \n" \
      " luti4 { z6.b - z7.b }, zt0, z13[0]                          \n" \
      " luti4 { z8.b - z9.b }, zt0, z14[0]                          \n" \
      " luti4 { z10.b - z11.b }, zt0, z15[0]                        \n" \
      /* 4way SDOT between line0 of block x and y in ZA array */        \
      " sdot za.s[w9, 0, vgx4], {z4.b-z7.b}, " zix "                \n" \
      " sdot za.s[w9, 1, vgx4], {z8.b-z11.b}, " zix "               \n" \
                                                                        \
      /* Increment the base for "memory layout" columns */              \
      " add %[A_cols_ptr], %[A_cols_ptr], %[A_col_inc]              \n"
// end define LUTI_BLOCK

      // Repeat for all columns in this block
      LUTI_BLOCK("z0.b[0]")
      LUTI_BLOCK("z0.b[1]")
      LUTI_BLOCK("z0.b[2]")
      LUTI_BLOCK("z0.b[3]")
      LUTI_BLOCK("z1.b[0]")
      LUTI_BLOCK("z1.b[1]")
      LUTI_BLOCK("z1.b[2]")
      LUTI_BLOCK("z1.b[3]")

      // Load Z registers with intermediate values in ZA array
#if defined(__ARM_FEATURE_SME2p1)
      " movaz   {z16.s-z19.s}, za.s[w9, 0, vgx4]                      \n"
      " movaz   {z20.s-z23.s}, za.s[w9, 1, vgx4]                      \n"
#else
      " mova    {z16.s-z19.s}, za.s[w9, 0, vgx4]                      \n"
      " mova    {z20.s-z23.s}, za.s[w9, 1, vgx4]                      \n"
#endif
      // Convert all intermediate registers from int32 to float32
      " scvtf {z16.s-z19.s}, {z16.s-z19.s}                            \n"
      " scvtf {z20.s-z23.s}, {z20.s-z23.s}                            \n"

      /* Combine all the SF of x and y */
      " ld1rw z3.s, p2/z, [%[X_sf_ptr]]                               \n"
      " add %[X_sf_ptr], %[X_sf_ptr], #4                              \n"
      // load block scaling factors of A
      " ld1w { z4.s - z7.s }, pn8/z, [%[A_sf_col_ptr]]                \n"
      " ld1w { z8.s - z11.s }, pn8/z, [%[A_sf_col_ptr], #4, mul vl]   \n"
      " add %[A_sf_col_ptr], %[A_sf_col_ptr], %[A_sf_col_inc]         \n"
      // Combine A and X scaling factors (distributive for Q4_0 and dot product)
      " fmul z4.s, z4.s, z3.s                                         \n"
      " fmul z5.s, z5.s, z3.s                                         \n"
      " fmul z6.s, z6.s, z3.s                                         \n"
      " fmul z7.s, z7.s, z3.s                                         \n"
      " fmul z8.s, z8.s, z3.s                                         \n"
      " fmul z9.s, z9.s, z3.s                                         \n"
      " fmul z10.s, z10.s, z3.s                                       \n"
      " fmul z11.s, z11.s, z3.s                                       \n"

      // Multiply intermediates by scaling factors and accumulate for full accumulation
      " fmla z24.s, p2/m, z16.s, z4.s                                 \n"
      " fmla z25.s, p2/m, z17.s, z5.s                                 \n"
      " fmla z26.s, p2/m, z18.s, z6.s                                 \n"
      " fmla z27.s, p2/m, z19.s, z7.s                                 \n"
      " fmla z28.s, p2/m, z20.s, z8.s                                 \n"
      " fmla z29.s, p2/m, z21.s, z9.s                                 \n"
      " fmla z30.s, p2/m, z22.s, z10.s                                \n"
      " fmla z31.s, p2/m, z23.s, z11.s                                \n"

    " cmp %[A_cols_ptr], %[A_cols_end]                                \n"
    " b.lt 2b                                                         \n"

    // Store all vector results to memory
    " st1w { z24.s-z27.s }, pn8, [%[B_dst]]                           \n"
    " st1w { z28.s-z31.s }, pn8, [%[B_dst], #0x4, mul vl]             \n"
    " addvl %[B_dst], %[B_dst], #8                                    \n"

  " add %[A_src], %[A_src], %[A_row_inc]                              \n"
  " add %[A_sf_src], %[A_sf_src], %[A_sf_row_inc]                     \n"
  " cmp %[A_src], %[A_rows_end]                                       \n"
  " b.lt 1b                                                           \n"

    :
      [A_src] "+&r" (A_src),
      [A_sf_src] "+&r" (A_sf_src),
      [B_dst] "+&r" (B_dst),
      [A_sf_col_ptr] "=&r" (A_sf_col_ptr),
      [X_sf_ptr] "=&r" (X_sf_ptr),
      [X_ptr] "=&r" (X_ptr),
      [A_cols_ptr] "=&r" (A_cols_ptr)
    :
      [lut] "r" (lut),
      [A_rows_end] "r" (A_rows_end),
      [A_cols_end] "r" (A_cols_end),
      [A_col_inc] "r" (A_col_inc),
      [A_sf_col_inc] "r" (A_sf_col_inc),
      [A_sf_row_inc] "r" (A_sf_row_inc),
      [X_sf_src] "r" (X_sf_src),
      [X_src] "r" (X_src),
      [A_row_inc] "r" (A_row_inc)
    : "p2", "x9",
      "z0","z1","z2", "z3", "z4", "z5", "z6", "z7",
      "z8","z9","z10","z11","z12","z13","z14","z15",
      "z16","z17","z18","z19","z20","z21","z22","z23",
      "z24","z25","z26","z27","z28","z29","z30","z31",
#ifdef __ARM_STATE_ZA
        "za",
#endif
#ifdef __ARM_STATE_ZT0
        "zt0",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_212(struct loop_212_data *data)
LOOP_ATTR
{
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src = (uint64_t)data->a;
  register uint64_t A_sf_src = (uint64_t)data->a_scales;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t X_sf_src = (uint64_t)data->x_scales;
  register uint64_t B_dst = (uint64_t)data->b;
  register uint64_t A_rows_end = (uint64_t) &data->a[A_rows * 2];
  register uint64_t A_cols_ptr = (uint64_t) data->a;
  register uint64_t A_cols_end = (uint64_t) &data->a[A_cols * A_rows / 2];
  register uint64_t X_ptr = (uint64_t) data->x;
  register uint64_t X_sf_ptr = (uint64_t) data->x_scales;
  register uint64_t lut = (uint64_t)&data->lut;
  register uint64_t A_col_inc = A_rows * 2;
  register uint64_t A_sf_col_inc = A_rows * 4;
  register uint64_t A_sf_row_inc;
  register uint64_t A_sf_col_ptr = 0;
  register uint64_t A_row_inc;

  // Scaling factors are 4 byte values, and we need 4 * cnth values.
  asm volatile("cnth %[A_sf_row_inc], all, mul #16 \n"
               :[A_sf_row_inc] "=&r"(A_sf_row_inc)::);

  // Loading 4 Z registers from A in every loop
  asm volatile("cntb %[A_row_inc], all, mul #4 \n"
               :[A_row_inc] "=&r"(A_row_inc)::);


  asm volatile(
  " ptrue p2.b, all                                                   \n"
#if defined(__ARM_FEATURE_SVE2p1)
  " ptrue pn8.b                                                       \n"
  " ptrue pn9.s                                                       \n"
#endif
#if defined(__ARM_FEATURE_LUT)
  " ldr z3, [%[lut]]                                                  \n"
#else
  // Dequantize mask vector
  " dup z3.b, #15                                                     \n"
#endif

  // row loop
  "1:                                                                 \n"
    " mov %[A_cols_ptr], %[A_src]                                     \n"
    " mov %[A_sf_col_ptr], %[A_sf_src]                                \n"
    " mov %[X_ptr], %[X_src]                                          \n"
    " mov %[X_sf_ptr], %[X_sf_src]                                    \n"

    // initialize outer accumulator to zero
    " dup z24.s, #0                                                   \n"
    " dup z25.s, #0                                                   \n"
    " dup z26.s, #0                                                   \n"
    " dup z27.s, #0                                                   \n"
    " dup z28.s, #0                                                   \n"
    " dup z29.s, #0                                                   \n"
    " dup z30.s, #0                                                   \n"
    " dup z31.s, #0                                                   \n"

    // Column loop
    // Handles 1 block at a time = 32 columns = 16 bytes of columns
    "2:                                                               \n"

      // initialize inner accumulator to zero
      " dup z16.s, #0                                                 \n"
      " dup z17.s, #0                                                 \n"
      " dup z18.s, #0                                                 \n"
      " dup z19.s, #0                                                 \n"
      " dup z20.s, #0                                                 \n"
      " dup z21.s, #0                                                 \n"
      " dup z22.s, #0                                                 \n"
      " dup z23.s, #0                                                 \n"

      // load next 32 x values
      " ld1rqw  {z0.s}, p2/z, [%[X_ptr], #0]                          \n"
      " ld1rqw  {z1.s}, p2/z, [%[X_ptr], #16]                         \n"
      " add %[X_ptr], %[X_ptr], #32                                   \n"

// Dequantize (widen to 8 bit and -8)
#if defined(__ARM_FEATURE_LUT)
#define SVE_DEQUANTIZE(SRC_VEC, DST_VEC_1, DST_VEC_2)                   \
      " luti4 " DST_VEC_1 ".b, {z3.b}, " SRC_VEC "[0]               \n" \
      " luti4 " DST_VEC_2 ".b, {z3.b}, " SRC_VEC "[1]               \n"
// end SVE_DEQUANTIZE
#else
#define SVE_DEQUANTIZE(SRC_VEC, DST_VEC_1, DST_VEC_2)                   \
      " and z2.d, " SRC_VEC ".d, z3.d                               \n" \
      " sub z2.b, z2.b, #8                                          \n" \
      " lsr " SRC_VEC ".b, " SRC_VEC ".b, #4                        \n" \
      " sub " SRC_VEC ".b, " SRC_VEC ".b, #8                        \n" \
      " zip1 " DST_VEC_1 ".b, z2.b, " SRC_VEC ".b                   \n" \
      " zip2 " DST_VEC_2 ".b, z2.b, " SRC_VEC ".b                   \n"
// end SVE_DEQUANTIZE
#endif

#if defined(__ARM_FEATURE_SVE2p1)
#define SVE2_BLOCK(zix)                                                 \
      /* load next A col value for every row */                         \
      " ld1b { z12.b-Z15.b}, pn8/z, [%[A_cols_ptr]]                 \n" \
      /* 4bit-8bit dequantization */                                    \
      SVE_DEQUANTIZE("z12", "z4", "z5")                                 \
      SVE_DEQUANTIZE("z13", "z6", "z7")                                 \
      SVE_DEQUANTIZE("z14", "z8", "z9")                                 \
      SVE_DEQUANTIZE("z15", "z10", "z11")                               \
      /* SDOT between line0 of block x and y in ZA array */             \
      " sdot z16.s, z4.b, " zix "                                   \n" \
      " sdot z17.s, z5.b, " zix "                                   \n" \
      " sdot z18.s, z6.b, " zix "                                   \n" \
      " sdot z19.s, z7.b, " zix "                                   \n" \
      " sdot z20.s, z8.b, " zix "                                   \n" \
      " sdot z21.s, z9.b, " zix "                                   \n" \
      " sdot z22.s, z10.b, " zix "                                  \n" \
      " sdot z23.s, z11.b, " zix "                                  \n" \
                                                                        \
      /* Increment the base for "memory layout" columns */              \
      " add %[A_cols_ptr], %[A_cols_ptr], %[A_col_inc]              \n"
#else
#define SVE2_BLOCK(zix)                                                 \
      /* load next A col value for every row */                         \
      " ld1b z12.b, p2/z, [%[A_cols_ptr]]                           \n" \
      " ld1b z13.b, p2/z, [%[A_cols_ptr], #1, mul vl]               \n" \
      " ld1b z14.b, p2/z, [%[A_cols_ptr], #2, mul vl]               \n" \
      " ld1b z15.b, p2/z, [%[A_cols_ptr], #3, mul vl]               \n" \
      /* 4bit-8bit dequantization */                                    \
      SVE_DEQUANTIZE("z12", "z4", "z5")                                 \
      SVE_DEQUANTIZE("z13", "z6", "z7")                                 \
      SVE_DEQUANTIZE("z14", "z8", "z9")                                 \
      SVE_DEQUANTIZE("z15", "z10", "z11")                               \
      /* SDOT between line0 of block x and y in ZA array */             \
      " sdot z16.s, z4.b, " zix "                                   \n" \
      " sdot z17.s, z5.b, " zix "                                   \n" \
      " sdot z18.s, z6.b, " zix "                                   \n" \
      " sdot z19.s, z7.b, " zix "                                   \n" \
      " sdot z20.s, z8.b, " zix "                                   \n" \
      " sdot z21.s, z9.b, " zix "                                   \n" \
      " sdot z22.s, z10.b, " zix "                                  \n" \
      " sdot z23.s, z11.b, " zix "                                  \n" \
                                                                        \
      /* Increment the base for "memory layout" columns */              \
      " add %[A_cols_ptr], %[A_cols_ptr], %[A_col_inc]              \n"
#endif

// end define SVE2_BLOCK

      // Repeat for all columns in this block
      SVE2_BLOCK("z0.b[0]")
      SVE2_BLOCK("z0.b[1]")
      SVE2_BLOCK("z0.b[2]")
      SVE2_BLOCK("z0.b[3]")
      SVE2_BLOCK("z1.b[0]")
      SVE2_BLOCK("z1.b[1]")
      SVE2_BLOCK("z1.b[2]")
      SVE2_BLOCK("z1.b[3]")

      // Convert all intermediate registers from int32 to float32
      " scvtf z16.s, p2/m, z16.s                                      \n"
      " scvtf z17.s, p2/m, z17.s                                      \n"
      " scvtf z18.s, p2/m, z18.s                                      \n"
      " scvtf z19.s, p2/m, z19.s                                      \n"
      " scvtf z20.s, p2/m, z20.s                                      \n"
      " scvtf z21.s, p2/m, z21.s                                      \n"
      " scvtf z22.s, p2/m, z22.s                                      \n"
      " scvtf z23.s, p2/m, z23.s                                      \n"

      /* Combine all the SF of x and y */
      " ld1rw z2.s, p2/z, [%[X_sf_ptr]]                               \n"
      " add %[X_sf_ptr], %[X_sf_ptr], #4                              \n"
      // load block scaling factors of A
#if defined(__ARM_FEATURE_SVE2p1)
      " ld1w { z4.s-z7.s }, pn9/z, [%[A_sf_col_ptr]]                  \n"
      " ld1w { z8.s-z11.s }, pn9/z, [%[A_sf_col_ptr], #4, mul vl]     \n"
#else
      " ld1w {z4.s}, p2/z, [%[A_sf_col_ptr]]                          \n"
      " ld1w {z5.s}, p2/z, [%[A_sf_col_ptr], #1, mul vl]              \n"
      " ld1w {z6.s}, p2/z, [%[A_sf_col_ptr], #2, mul vl]              \n"
      " ld1w {z7.s}, p2/z, [%[A_sf_col_ptr], #3, mul vl]              \n"
      " ld1w {z8.s}, p2/z, [%[A_sf_col_ptr], #4, mul vl]              \n"
      " ld1w {z9.s}, p2/z, [%[A_sf_col_ptr], #5, mul vl]              \n"
      " ld1w {z10.s}, p2/z, [%[A_sf_col_ptr], #6, mul vl]             \n"
      " ld1w {z11.s}, p2/z, [%[A_sf_col_ptr], #7, mul vl]             \n"
#endif
      " add %[A_sf_col_ptr], %[A_sf_col_ptr], %[A_sf_col_inc]         \n"
      // Combine A and X scaling factors (distributive for Q4_0 and dot product)
      " fmul z4.s, z4.s, z2.s                                         \n"
      " fmul z5.s, z5.s, z2.s                                         \n"
      " fmul z6.s, z6.s, z2.s                                         \n"
      " fmul z7.s, z7.s, z2.s                                         \n"
      " fmul z8.s, z8.s, z2.s                                         \n"
      " fmul z9.s, z9.s, z2.s                                         \n"
      " fmul z10.s, z10.s, z2.s                                       \n"
      " fmul z11.s, z11.s, z2.s                                       \n"

      // Multiply intermediates by scaling factors and accumulate for full accumulation
      " fmla z24.s, p2/m, z16.s, z4.s                                 \n"
      " fmla z25.s, p2/m, z17.s, z5.s                                 \n"
      " fmla z26.s, p2/m, z18.s, z6.s                                 \n"
      " fmla z27.s, p2/m, z19.s, z7.s                                 \n"
      " fmla z28.s, p2/m, z20.s, z8.s                                 \n"
      " fmla z29.s, p2/m, z21.s, z9.s                                 \n"
      " fmla z30.s, p2/m, z22.s, z10.s                                \n"
      " fmla z31.s, p2/m, z23.s, z11.s                                \n"

    " cmp %[A_cols_ptr], %[A_cols_end]                                \n"
    " b.lt 2b                                                         \n"

    // Store all vector results to memory
#if defined(__ARM_FEATURE_SVE2p1)
    " st1w { z24.s-z27.s }, pn9, [%[B_dst]]                           \n"
    " st1w { z28.s-z31.s }, pn9, [%[B_dst], #4, mul vl]               \n"
#else
    " st1w {z24.s}, p2, [%[B_dst]]                                    \n"
    " st1w {z25.s}, p2, [%[B_dst], #1, mul vl]                        \n"
    " st1w {z26.s}, p2, [%[B_dst], #2, mul vl]                        \n"
    " st1w {z27.s}, p2, [%[B_dst], #3, mul vl]                        \n"
    " st1w {z28.s}, p2, [%[B_dst], #4, mul vl]                        \n"
    " st1w {z29.s}, p2, [%[B_dst], #5, mul vl]                        \n"
    " st1w {z30.s}, p2, [%[B_dst], #6, mul vl]                        \n"
    " st1w {z31.s}, p2, [%[B_dst], #7, mul vl]                        \n"
#endif
    " addvl %[B_dst], %[B_dst], #8                                    \n"

  " add %[A_src], %[A_src], %[A_row_inc]                              \n"
  " add %[A_sf_src], %[A_sf_src], %[A_sf_row_inc]                     \n"
  " cmp %[A_src], %[A_rows_end]                                       \n"
  " b.lt 1b                                                           \n"

    :
      [A_src] "+&r" (A_src),
      [A_sf_src] "+&r" (A_sf_src),
      [B_dst] "+&r" (B_dst),
      [A_cols_ptr] "=&r" (A_cols_ptr),
      [X_ptr] "=&r" (X_ptr),
      [X_sf_ptr] "=&r" (X_sf_ptr),
      [A_sf_col_ptr] "=&r" (A_sf_col_ptr)
    :
      [lut] "r" (lut),
      [X_src] "r" (X_src),
      [X_sf_src] "r" (X_sf_src),
      [A_rows_end] "r" (A_rows_end),
      [A_cols_end] "r" (A_cols_end),
      [A_col_inc] "r" (A_col_inc),
      [A_sf_col_inc] "r" (A_sf_col_inc),
      [A_sf_row_inc] "r" (A_sf_row_inc),
      [A_row_inc] "r" (A_row_inc)
    : "p2",
#if defined(__ARM_FEATURE_SVE2p1)
      "p8", "p9",
#endif
      "z0","z1","z2", "z3", "z4", "z5", "z6", "z7",
      "z8","z9","z10","z11","z12","z13","z14","z15",
      "z16","z17","z18","z19","z20","z21","z22","z23",
      "z24","z25","z26","z27","z28","z29","z30","z31",
      "memory","cc");
}

#elif (defined(__ARM_NEON) && defined (__ARM_FEATURE_DOTPROD))

static void inner_loop_212(struct loop_212_data *data) {
  register uint64_t A_rows = data->m;
  register uint64_t A_cols = data->n;
  register uint64_t A_src= (uint64_t)data->a;
  register uint64_t A_sf_src = (uint64_t)data->a_scales;
  register uint64_t X_src = (uint64_t)data->x;
  register uint64_t X_sf_src = (uint64_t)data->x_scales;
  register uint64_t B_dst = (uint64_t)data->b;
  register uint64_t A_rows_end = (uint64_t) &data->a[A_rows * 2];
  register uint64_t A_cols_ptr = (uint64_t) data->a;
  register uint64_t A_cols_end = (uint64_t) &data->a[A_cols * A_rows / 2];
  register uint64_t X_ptr = (uint64_t) data->x;
  register uint64_t X_sf_ptr = (uint64_t) data->x_scales;
  register uint64_t A_col_inc = A_rows * 2;
  register uint64_t A_sf_col_inc = A_rows * 4 - 64;
  const register uint64_t A_sf_row_inc = 128;
  register uint64_t A_sf_col_ptr = 0;
  const register uint64_t A_row_inc = 64;

  asm volatile(
  " movi v0.16b, #0xf                                                 \n"
  " movi v1.16b, #8                                                   \n"

  // row loop
  "1:                                                                 \n"
    " mov %[A_cols_ptr], %[A_src]                                     \n"
    " mov %[A_sf_col_ptr], %[A_sf_src]                                \n"
    " mov %[X_ptr], %[X_src]                                          \n"
    " mov %[X_sf_ptr], %[X_sf_src]                                    \n"

    // initialize outer accumulator to zero
    " movi v24.4s, #0                                                 \n"
    " movi v25.4s, #0                                                 \n"
    " movi v26.4s, #0                                                 \n"
    " movi v27.4s, #0                                                 \n"
    " movi v28.4s, #0                                                 \n"
    " movi v29.4s, #0                                                 \n"
    " movi v30.4s, #0                                                 \n"
    " movi v31.4s, #0                                                 \n"

    // Column loop
    // Handles 1 block at a time = 32 columns = 16 bytes of columns
    "2:                                                               \n"

      // initialize inner accumulator to zero
      " movi v16.4s, #0                                               \n"
      " movi v17.4s, #0                                               \n"
      " movi v18.4s, #0                                               \n"
      " movi v19.4s, #0                                               \n"
      " movi v20.4s, #0                                               \n"
      " movi v21.4s, #0                                               \n"
      " movi v22.4s, #0                                               \n"
      " movi v23.4s, #0                                               \n"

// Dequantize (widen to 8 bit and -8)
#define NEON_DEQUANTIZE(SRC_VEC, DST_VEC_1, DST_VEC_2)                  \
      " and v2.16b, " SRC_VEC ".16b, v0.16b                         \n" \
      " sub v2.16b, v2.16b, v1.16b                                  \n" \
      " ushr " SRC_VEC ".16b, " SRC_VEC ".16b, #4                   \n" \
      " sub " SRC_VEC ".16b, " SRC_VEC ".16b, v1.16b                \n" \
      " zip1 " DST_VEC_1 ".16b, v2.16b, " SRC_VEC ".16b             \n" \
      " zip2 " DST_VEC_2 ".16b, v2.16b, " SRC_VEC ".16b             \n"
// end NEON_DEQUANTIZE

#define NEON_BLOCK                                                    \
      " ld1r {v3.4s}, [%[X_ptr]], #4                              \n" \
      " ld1 {v12.16b,v13.16b,v14.16b,v15.16b}, [%[A_cols_ptr]], %[A_col_inc] \n" \
      NEON_DEQUANTIZE("v12", "v4", "v5")                              \
      NEON_DEQUANTIZE("v13", "v6", "v7")                              \
      NEON_DEQUANTIZE("v14", "v8", "v9")                              \
      NEON_DEQUANTIZE("v15", "v10", "v11")                            \
      " sdot v16.4s, v4.16b, v3.16b                               \n" \
      " sdot v17.4s, v5.16b, v3.16b                               \n" \
      " sdot v18.4s, v6.16b, v3.16b                               \n" \
      " sdot v19.4s, v7.16b, v3.16b                               \n" \
      " sdot v20.4s, v8.16b, v3.16b                               \n" \
      " sdot v21.4s, v9.16b, v3.16b                               \n" \
      " sdot v22.4s, v10.16b, v3.16b                              \n" \
      " sdot v23.4s, v11.16b, v3.16b                              \n"
// end NEON_BLOCK

      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK
      NEON_BLOCK

      " scvtf v16.4s, v16.4s                                          \n"
      " scvtf v17.4s, v17.4s                                          \n"
      " scvtf v18.4s, v18.4s                                          \n"
      " scvtf v19.4s, v19.4s                                          \n"
      " scvtf v20.4s, v20.4s                                          \n"
      " scvtf v21.4s, v21.4s                                          \n"
      " scvtf v22.4s, v22.4s                                          \n"
      " scvtf v23.4s, v23.4s                                          \n"

      /* Combine all the SF of x and y */
      " ld1r {v3.4s}, [%[X_sf_ptr]], #4                               \n"
      // load block scaling factors of A
      " ld1 {v4.4s,v5.4s,v6.4s,v7.4s}, [%[A_sf_col_ptr]], #64         \n"
      " ld1 {v8.4s,v9.4s,v10.4s,v11.4s}, [%[A_sf_col_ptr]], %[A_sf_col_inc] \n"
      // Combine A and X scaling factors (distributive for Q4_0 and dot product)
      " fmul v4.4s, v4.4s, v3.4s                                      \n"
      " fmul v5.4s, v5.4s, v3.4s                                      \n"
      " fmul v6.4s, v6.4s, v3.4s                                      \n"
      " fmul v7.4s, v7.4s, v3.4s                                      \n"
      " fmul v8.4s, v8.4s, v3.4s                                      \n"
      " fmul v9.4s, v9.4s, v3.4s                                      \n"
      " fmul v10.4s, v10.4s, v3.4s                                    \n"
      " fmul v11.4s, v11.4s, v3.4s                                    \n"

      // Multiply intermediates by scaling factors and accumulate for full accumulation
      " fmla v24.4s, v16.4s, v4.4s                                    \n"
      " fmla v25.4s, v17.4s, v5.4s                                    \n"
      " fmla v26.4s, v18.4s, v6.4s                                    \n"
      " fmla v27.4s, v19.4s, v7.4s                                    \n"
      " fmla v28.4s, v20.4s, v8.4s                                    \n"
      " fmla v29.4s, v21.4s, v9.4s                                    \n"
      " fmla v30.4s, v22.4s, v10.4s                                   \n"
      " fmla v31.4s, v23.4s, v11.4s                                   \n"
    " cmp %[A_cols_ptr], %[A_cols_end]                                \n"
    " b.lt 2b                                                         \n"

    // Store all vector results to memory
    " st1 {v24.4s,v25.4s,v26.4s,v27.4s}, [%[B_dst]], #64              \n"
    " st1 {v28.4s,v29.4s,v30.4s,v31.4s}, [%[B_dst]], #64              \n"

  " add %[A_src], %[A_src], %[A_row_inc]                              \n"
  " add %[A_sf_src], %[A_sf_src], %[A_sf_row_inc]                     \n"
  " cmp %[A_src], %[A_rows_end]                                       \n"
  " b.lt 1b                                                           \n"

    :
      [A_src] "+&r" (A_src),
      [A_sf_src] "+&r" (A_sf_src),
      [B_dst] "+&r" (B_dst),
      [A_cols_ptr] "=&r" (A_cols_ptr),
      [X_ptr] "=&r" (X_ptr),
      [X_sf_ptr] "=&r" (X_sf_ptr),
      [A_sf_col_ptr] "=&r" (A_sf_col_ptr)
    :
      [X_src] "r" (X_src),
      [X_sf_src] "r" (X_sf_src),
      [A_rows_end] "r" (A_rows_end),
      [A_cols_end] "r" (A_cols_end),
      [A_col_inc] "r" (A_col_inc),
      [A_sf_col_inc] "r" (A_sf_col_inc),
      [A_sf_row_inc] "r" (A_sf_row_inc),
      [A_row_inc] "r" (A_row_inc)
    : "v0","v1","v2", "v3", "v4", "v5", "v6", "v7",
      "v8","v9","v10","v11","v12","v13","v14","v15",
      "v16","v17","v18","v19","v20","v21","v22","v23",
      "v24","v25","v26","v27","v28","v29","v30","v31",
      "memory","cc");
}

#else

static void inner_loop_212(struct loop_212_data *data) {
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
// A = M * N / 2 uint8_t
// X = N int8_t
// B = M float
// Scaling factors A = M * N / BLOCK_SIZE float
// Scaling factors X = N / BLOCK_SIZE float
// LUT = 64 bytes
#define PROBLEM_SIZE_ACTUAL(m, n, block_size)  \
(((m) * (n))*sizeof(uint8_t) / 2 +             \
(n)*sizeof(uint8_t) +                          \
(m)*sizeof(float) +                            \
((m) * (n) / (block_size))*sizeof(float) +     \
((n) / (block_size))*sizeof(float) +           \
+ 64)

LOOP_DECL(212, OUTER_LOOP_ATTR)
{
  //  M: multiple of 16 * SVLw (preferably 8 * SVLw^2)
  uint64_t M = 0;
  //  N: multiple of BLOCK_SIZE
  uint64_t N = 0;
  const uint64_t block_size = BLOCK_SIZE;
  const uint64_t N_base = block_size;
  // 4 registers, 4 4-bit elements next to each other
  const uint64_t M_base = 4 * 4 * (MAX_VL / 16); // min = 16 * 4 > block_size (32)

  // 1 N_base = 1 columns loop, 1 M_base = 1 row loop
  while (PROBLEM_SIZE_ACTUAL(M + M_base, N + N_base,  block_size) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    M += M_base;
    N += N_base;
  }

  while (PROBLEM_SIZE_ACTUAL(M, N + N_base,  block_size) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    N += N_base;
  }

  while (PROBLEM_SIZE_ACTUAL(M + M_base, N,  block_size) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
    M += M_base;
  }

  if (M == 0) {
    printf("ERROR LOOP 212: need at least %" PRIu64 "KiB to \
            construct the minimal %" PRIu64 " X %" PRIu64 " problem size\n",\
           PROBLEM_SIZE_ACTUAL(M, N + N_base,  block_size) / 1024 + 1,
           M_base , N_base);
    return 1;
  }

  struct loop_212_data data = { .m = M, .n = N, };
  ALLOC_64B(data.a, M * N / 2, "A matrix"); // 4 bit elements
  ALLOC_64B(data.x, N, "x vector");
  ALLOC_64B(data.b, M + 1, "B vector");
  // Padding
  data.b[M] = 1234.0;
  ALLOC_64B(data.a_scales, M * N * 4 / block_size, "A matrix scaling factors");
  ALLOC_64B(data.x_scales, N * 4 / block_size, "x vector scaling factors");

  #if (defined(__ARM_FEATURE_SME2) || defined(HAVE_SME_INTRINSICS))
    for (uint8_t i = 0; i < 64; i++) {
      if (i % 4 == 0)
        data.lut[i] = i / 4 - 8;
      else
        data.lut[i] = 0;
    }
  #else
    uint8_t j=0;
    for (uint8_t i = 0; i < 64; i++) {
      if (i % 4 == 0){
        data.lut[j] = i / 4 - 8;
        j+=1;
      }
    }
  #endif

  fill_uint8(data.a, M * N / 2);
  // no fill_int8
  fill_uint8((uint8_t *) data.x, N);
  fill_float_range(data.a_scales, M * N / block_size, 0.01, 2.0);
  fill_float_range(data.x_scales, N / block_size, 0.01, 2.0);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("Block size = %" PRIu64 "\n", block_size);
  printf("A scaling factor matrix dimension : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N / block_size);
  printf("x scaling factor vector dimension : %" PRIu64 "\n", N / block_size);
  printf("Sizes: A %.1fKiB, A scaling factors %.1fKiB, x %.1fKiB, x scaling factors %.1fKiB, B %.1fKiB\n",
          (M*N/2)/1024.0f, (M*N/block_size)*sizeof(float)/1024.0f,
          N/1024.0f, (N/block_size)*sizeof(float)/1024.0f,
          M*sizeof(float)/1024.0f
         );
  printf("\tTotal space used for inputs and outputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N, block_size)/1024.0f);
#endif

  inner_loops_212(iters, &data);

  int checksum = 0;
  float64_t max_error;
#define CHECK(i)                                                      \
  {                                                                   \
    uint64_t row = i;                                                 \
    int old_checksum = checksum;                                      \
    uint64_t col_inc = 2 * data.m;                                    \
    uint8_t *row_ptr = &data.a[2 * row];                              \
    uint8_t first_byte, second_byte;                                  \
    int32_t block_dot;                                                \
    float64_t row_sum = 0;                                            \
    for (uint64_t col = 0; col < data.n; col += block_size) {         \
      block_dot = 0;                                                  \
      for (uint64_t block = 0; block < block_size / 4; block++) {     \
        first_byte = row_ptr[(col / 4 + block) * col_inc];            \
        second_byte = row_ptr[(col / 4 + block) * col_inc + 1];       \
        block_dot += ((int8_t)(first_byte & 0x0F) - 8) * data.x[col + block * 4];       \
        block_dot += ((int8_t)(first_byte >> 4) - 8) * data.x[col + block * 4 + 1];     \
        block_dot += ((int8_t)(second_byte & 0x0F) - 8) * data.x[col + block * 4 + 2];  \
        block_dot += ((int8_t)(second_byte >> 4) - 8) * data.x[col + block * 4 + 3];    \
      }                                                               \
      row_sum += (float64_t) block_dot * (float64_t) data.a_scales[(col / block_size) * data.m + row] * (float64_t) data.x_scales[col / block_size];  \
    }                                                                 \
    max_error = row_sum * 0.000001;                                   \
    if (row_sum < 0) max_error = -max_error;                          \
    if (max_error < 0.001f) max_error = 0.001f;                       \
    checksum += check_double((double) data.b[i], row_sum, max_error) ? 0 : 1;          \
    if (checksum != old_checksum) printf("ERROR: Checksum %" PRIu64 " wrong (got %10f, expected %10f, max error %10f).\n", row, data.b[i], row_sum, max_error); \
  }

#define CHECK_DEBUG(i)                                                \
  {                                                                   \
    uint64_t row = i;                                                 \
    uint64_t col_inc = 2 * data.m;                                    \
    uint8_t *row_ptr = &data.a[2 * row];                              \
    uint8_t first_byte, second_byte;                                  \
    int32_t block_dot;                                                \
    float64_t row_sum = 0;                                            \
    float32_t print_float_1 = 0;                                      \
    float32_t print_float_2 = 0;                                      \
    printf("\n\nROW %" PRIu64 "\n\n", row);                           \
    for (uint64_t col = 0; col < data.n; col += block_size) {         \
      block_dot = 0;                                                  \
    printf("\n******************************* BLOCK %" PRIu64 " of %" PRIu64 " ****************************\n\n", col / block_size, data.n / block_size);                                        \
      for (uint64_t block = 0; block < block_size / 4; block++) {     \
        first_byte = row_ptr[(col / 4 + block) * col_inc];            \
        second_byte = row_ptr[(col / 4 + block) * col_inc + 1];       \
        printf("first byte %02X , second byte %02X, in Z %02X%02X\n", \
               first_byte,                                            \
               second_byte,                                           \
               second_byte,                                           \
               first_byte);                                           \
        printf("luti input      %8u, %8u, %8u, %8u\n",                \
               (first_byte & 0x0F),                                   \
               (first_byte >> 4),                                     \
               (second_byte & 0x0F),                                  \
               (second_byte >> 4)                                     \
         );                                                           \
        printf("luti conversion %8d, %8d, %8d, %8d\n",                \
               (int8_t)(first_byte & 0x0F) - 8,                       \
               (int8_t)(first_byte >> 4) - 8,                         \
               (int8_t)(second_byte & 0x0F) - 8,                      \
               (int8_t)(second_byte >> 4) - 8                         \
         );                                                           \
        printf("luti conversion %08X, %08X, %08X, %08X\n",            \
               (int8_t)(first_byte & 0x0F) - 8,                       \
               (int8_t)(first_byte >> 4) - 8,                         \
               (int8_t)(second_byte & 0x0F) - 8,                      \
               (int8_t)(second_byte >> 4) - 8                         \
         );                                                           \
        printf("Block index %" PRIu64 " to %" PRIu64 " of %" PRIu64 ", %d + \n", block * 4, block * 4 + 4, block_size, block_dot);         \
        block_dot += ((int8_t)(first_byte & 0x0F) - 8) * data.x[col + block * 4];       \
        block_dot += ((int8_t)(first_byte >> 4) - 8) * data.x[col + block * 4 + 1];     \
        block_dot += ((int8_t)(second_byte & 0x0F) - 8) * data.x[col + block * 4 + 2];  \
        block_dot += ((int8_t)(second_byte >> 4) - 8) * data.x[col + block * 4 + 3];    \
        printf("%8d * %8d + %8d * %8d + %8d * %8d + %8d * %8d = %8d\n",                 \
               ((int8_t)(first_byte & 0x0F) - 8), data.x[col + block * 4],              \
               ((int8_t)(first_byte >> 4) - 8), data.x[col + block * 4 + 1],            \
               ((int8_t)(second_byte & 0x0F) - 8), data.x[col + block * 4 + 2],         \
               ((int8_t)(second_byte >> 4) - 8), data.x[col + block * 4 + 3],           \
               block_dot                                              \
          );                                                          \
        printf("%08X * %08X + %08X * %08X + %08X * %08X + %08X * %08X = %08X\n",        \
               ((int8_t)(first_byte & 0x0F) - 8), data.x[col + block * 4],              \
               ((int8_t)(first_byte >> 4) - 8), data.x[col + block * 4 + 1],            \
               ((int8_t)(second_byte & 0x0F) - 8), data.x[col + block * 4 + 2],         \
               ((int8_t)(second_byte >> 4) - 8), data.x[col + block * 4 + 3],           \
               block_dot                                              \
          );                                                          \
        printf("%8d + %8d + %8d + %8d = %8d\n",                       \
               ((int8_t)(first_byte & 0x0F) - 8) * data.x[col + block * 4],             \
               ((int8_t)(first_byte >> 4) - 8) * data.x[col + block * 4 + 1],           \
               ((int8_t)(second_byte & 0x0F) - 8) * data.x[col + block * 4 + 2],        \
               ((int8_t)(second_byte >> 4) - 8) * data.x[col + block * 4 + 3],          \
               block_dot                                              \
          );                                                          \
        printf("%08X + %08X + %08X + %08X = %08X\n",                  \
               ((int8_t)(first_byte & 0x0F) - 8) * data.x[col + block * 4],             \
               ((int8_t)(first_byte >> 4) - 8) * data.x[col + block * 4 + 1],           \
               ((int8_t)(second_byte & 0x0F) - 8) * data.x[col + block * 4 + 2],        \
               ((int8_t)(second_byte >> 4) - 8) * data.x[col + block * 4 + 3],          \
               block_dot                                              \
          );                                                          \
      }                                                               \
      printf("\nBlock %" PRIu64 " BLOCK SCALING AND ACCUMULATION\n", col / block_size);         \
      printf("A scale address %p\n", &data.a_scales[(col / block_size) * data.m + row]);\
      print_float_1 = row_sum;                                        \
      printf("%08X += %08X * %08X * %08X\n", *(uint32_t *)&print_float_1, block_dot,     \
          *(uint32_t *)&data.a_scales[(col / block_size) * data.m + row] , *(uint32_t *)&data.x_scales[col / block_size]); \
      printf("%f += %d * %f * %f\n", print_float_1, block_dot,        \
          data.a_scales[(col / block_size) * data.m + row] , data.x_scales[col / block_size]); \
      printf("%08X += ", *(uint32_t *)&print_float_1);                \
      print_float_1 = (float64_t) block_dot;                          \
      print_float_2 = (float64_t) data.a_scales[(col / block_size) * data.m + row] * (float64_t) data.x_scales[col / block_size];  \
      printf("%08X * %08X\n", *(uint32_t *)&print_float_1,            \
          *(uint32_t *)&print_float_2);                               \
      printf("%f += %f * %f\n", row_sum, print_float_1,               \
          print_float_2);                                             \
      row_sum += (float64_t) block_dot * (float64_t) data.a_scales[(col / block_size) * data.m + row] * (float64_t) data.x_scales[col / block_size];  \
      print_float_1 = row_sum;                                        \
      printf("row_sum = %f, %08X\n", print_float_1, *(uint32_t *)&print_float_1); \
    }                                                                 \
    max_error = row_sum * 0.000001;                                   \
    if (row_sum < 0) max_error = -max_error;                          \
    if (max_error < 0.001f) max_error = 0.001f;                       \
    checksum += check_double((double) data.b[i], row_sum, max_error) ? 0 : 1; \
    printf("Row check used error %f.\n", row_sum * 0.000001f);        \
    if (checksum != 0) printf("ERROR: Checksum %" PRIu64 " wrong (got %10f, expected %10f).\n", row, data.b[i], row_sum); \
    else printf("Row %" PRIu64 " CORRECT (got %10f, expected %10f).\n", row, data.b[i], row_sum); \
  }

#ifdef FULL_CHECK
  for (int m = 0; m < M; m++) CHECK(m);
  if (data.b[data.m] != 1234.0) {
    printf("ERROR: B end canary overwritten!\n");
    checksum += 1;
  }
#else
  CHECK(0);
  CHECK(M - 1);
  CHECK(M / 2);
  if (data.b[data.m] != 1234.0) {
    printf("ERROR: B end canary overwritten!\n");
    checksum += 1;
  }
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(212, passed, "%d", 0, checksum)
#endif

  return passed ? 0 : 1;
}
