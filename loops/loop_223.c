/*----------------------------------------------------------------------------
#
#   Loop 223: Matrix transposition
#
#   Purpose:
#     Use of simd instructions (LD & ST ZA, LD & ZIP) in transposition.
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
  Constraints - 2x2 tailing
    M: M multiple of 16
    N: Any
*/

struct loop_223_data {
  uint64_t m;
  uint64_t n;
  uint32_t *restrict a;
  uint32_t *restrict at;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_223(struct loop_223_data *restrict data) {
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

static void inner_loop_223(struct loop_223_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint32_t *a = data->a;
  uint32_t *at = data->at;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      at[j * m + i] = a[i * n + j];
    }
  }
}
#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_223(struct loop_223_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint32_t *a = data->a;
  uint32_t *at = data->at;

  uint32_t *a_ptr, *at_ptr;
  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t blk_1 = svl_s * n;
  uint64_t blk_2 = svl_s * m;
  svbool_t p_all = svptrue_b32();
  svcount_t pn_rows, pn_cols;
  svbool_t p_rows_t, p_rows_b;
  svbool_t p_cols_l, p_cols_h;

  svuint32x2_t vec_a00, vec_a01, vec_a02, vec_a03;
  svuint32x2_t vec_a10, vec_a11, vec_a12, vec_a13;
  svuint32x4_t vec_at0, vec_at1, vec_at2, vec_at3;

#define EXTR2(x, y, i) svget2(vec_a##x##y, i)
#define LOAD(i, j, pn, p, idx, adr) \
  { vec_a##i##j = svld1_x2(svpsel_lane_c32(pn, p, idx + j), &a_ptr[adr]); }
#define LOAD_GROUP(i, pn, p, idx, adr)   \
  {                                      \
    LOAD(i, 0, pn, p, idx, adr);         \
    LOAD(i, 1, pn, p, idx, adr + n);     \
    LOAD(i, 2, pn, p, idx, adr + 2 * n); \
    LOAD(i, 3, pn, p, idx, adr + 3 * n); \
  };
#define WRITE_ZA(i, j, l, idx)                                   \
  svwrite_hor_za32_vg4(i, (uint32_t)idx + 0,                     \
                       svcreate4(EXTR2(j, 0, l), EXTR2(j, 1, l), \
                                 EXTR2(j, 2, l), EXTR2(j, 3, l)));

#define STORE(i, j, l, pn, p, idx, adr)                           \
  {                                                               \
    svst1(svpsel_lane_c32(pn, p, idx + l), &at_ptr[adr],          \
          svcreate2(svget4(vec_at##i, l), svget4(vec_at##j, l))); \
  }

  for (m_idx = 0;; m_idx += svl_s * 2) {
    pn_rows = svwhilelt_c32(m_idx, m, 2);
    p_rows_b = svpext_lane_c32(pn_rows, 0);
    p_rows_t = svpext_lane_c32(pn_rows, 1);
    // Break condition
    if (!svptest_first(p_all, p_rows_b)) break;

    for (n_idx = 0;; n_idx += svl_s * 2) {
      pn_cols = svwhilelt_c32(n_idx, n, 2);
      p_cols_l = svpext_lane_c32(pn_cols, 0);
      p_cols_h = svpext_lane_c32(pn_cols, 0);
      // Break condition
      if (!svptest_first(p_all, p_cols_l)) break;

      // ZA0:ZA3 load loop
      a_ptr = &a[m_idx * n + n_idx];
      for (l_idx = 0; l_idx < svl_s; l_idx += 4) {
        // Load input vectors
        LOAD_GROUP(0, pn_cols, p_rows_b, l_idx, 0);
        LOAD_GROUP(1, pn_cols, p_rows_t, l_idx, blk_1);

        // MOV horizontal to ZA
        WRITE_ZA(0, 0, 0, l_idx);
        WRITE_ZA(1, 0, 1, l_idx);
        WRITE_ZA(2, 1, 0, l_idx);
        WRITE_ZA(3, 1, 1, l_idx);

        a_ptr += n * 4;
      }

      // ZA0:ZA3 store loop
      at_ptr = &at[n_idx * m + m_idx];
      for (l_idx = 0; l_idx < svl_s; l_idx += 4) {
        // MOV vertical from ZA
        vec_at0 = svread_ver_za32_u32_vg4(0, (uint32_t)l_idx + 0);
        vec_at1 = svread_ver_za32_u32_vg4(2, (uint32_t)l_idx + 0);
        vec_at2 = svread_ver_za32_u32_vg4(1, (uint32_t)l_idx + 0);
        vec_at3 = svread_ver_za32_u32_vg4(3, (uint32_t)l_idx + 0);

        // Store output vectors
        STORE(0, 1, 0, pn_rows, p_cols_l, l_idx, 0 + 0);
        STORE(0, 1, 1, pn_rows, p_cols_l, l_idx, 0 + m);
        STORE(0, 1, 2, pn_rows, p_cols_l, l_idx, 0 + 2 * m);
        STORE(0, 1, 3, pn_rows, p_cols_l, l_idx, 0 + 3 * m);
        STORE(2, 3, 0, pn_rows, p_cols_h, l_idx, blk_2 + 0);
        STORE(2, 3, 1, pn_rows, p_cols_h, l_idx, blk_2 + m);
        STORE(2, 3, 2, pn_rows, p_cols_h, l_idx, blk_2 + 2 * m);
        STORE(2, 3, 3, pn_rows, p_cols_h, l_idx, blk_2 + 3 * m);

        at_ptr += m * 4;
      }
    }
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void transpose_sve_vl128(uint64_t rows, uint64_t cols, uint32_t *pIn,
                                uint32_t *pOut)
LOOP_ATTR
{
  uint64_t m = rows;
  uint64_t n = cols;
  uint32_t *a = pIn;
  uint32_t *at = pOut;

  uint32_t *ptr_a;
  uint32_t *ptr_at;
  uint64_t vl_s = svcntw();
  uint64_t n_vecs;

  svbool_t p_ld;
  svbool_t p_all = svptrue_b32();
  svuint32_t vec_a0, vec_a1, vec_a2, vec_a3;
  svuint32_t vec_at0, vec_at1, vec_at2, vec_at3;
  svuint32_t vec_t8, vec_t9, vec_t10, vec_t11;

#define LOAD(i, o) vec_a##i = svld1(p_ld, &ptr_a[o])

  for (int32_t x = 0; x < m; x += vl_s) {
    ptr_a = &a[x * n];
    ptr_at = &at[x];

    FOR_LOOP_32(uint64_t, y, 0, n, p_ld) {
      LOAD(0, 0 * n);
      LOAD(1, 1 * n);
      LOAD(2, 2 * n);
      LOAD(3, 3 * n);

      n_vecs = svcntp_b32(p_all, p_ld);

      vec_t8 = svzip1(vec_a0, vec_a2);
      vec_t9 = svzip2(vec_a0, vec_a2);
      vec_t10 = svzip1(vec_a1, vec_a3);
      vec_t11 = svzip2(vec_a1, vec_a3);
      vec_at0 = svzip1(vec_t8, vec_t10);
      vec_at1 = svzip2(vec_t8, vec_t10);
      vec_at2 = svzip1(vec_t9, vec_t11);
      vec_at3 = svzip2(vec_t9, vec_t11);

      switch (n_vecs) {
        default:
          svst1(p_all, &ptr_at[3 * m], vec_at3);
        case 3:
          svst1(p_all, &ptr_at[2 * m], vec_at2);
        case 2:
          svst1(p_all, &ptr_at[1 * m], vec_at1);
        case 1:
          svst1(p_all, &ptr_at[0 * m], vec_at0);
      }

      ptr_a += vl_s;
      ptr_at += 4 * m;
    }
  }
}

static void transpose_sve_vl256(uint64_t rows, uint64_t cols, uint32_t *pIn,
                                uint32_t *pOut)
LOOP_ATTR
{
  uint64_t m = rows;
  uint64_t n = cols;
  uint32_t *a = pIn;
  uint32_t *at = pOut;

  uint32_t *ptr_a;
  uint32_t *ptr_at;
  uint64_t vl_s = svcntw();
  uint64_t n_vecs;

  svbool_t p_ld;
  svbool_t p_all = svptrue_b32();
  svuint32_t vec_a0, vec_a1, vec_a2, vec_a3;
  svuint32_t vec_a4, vec_a5, vec_a6, vec_a7;
  svuint32_t vec_at0, vec_at1, vec_at2, vec_at3;
  svuint32_t vec_at4, vec_at5, vec_at6, vec_at7;
  svuint32_t vec_t8, vec_t9, vec_t10, vec_t11;
  svuint32_t vec_t12, vec_t13, vec_t14, vec_t15;
  svuint32_t vec_t16, vec_t17, vec_t18, vec_t19;
  svuint32_t vec_t20, vec_t21, vec_t22, vec_t23;

#define LOAD(i, o) vec_a##i = svld1(p_ld, &ptr_a[o])

  for (int32_t x = 0; x < m; x += vl_s) {
    ptr_a = &a[x * n];
    ptr_at = &at[x];

    FOR_LOOP_32(uint64_t, y, 0, n, p_ld) {
      LOAD(0, 0 * n);
      LOAD(1, 1 * n);
      LOAD(2, 2 * n);
      LOAD(3, 3 * n);
      LOAD(4, 4 * n);
      LOAD(5, 5 * n);
      LOAD(6, 6 * n);
      LOAD(7, 7 * n);

      n_vecs = svcntp_b32(p_all, p_ld);

      vec_t8 = svzip1(vec_a0, vec_a4);
      vec_t9 = svzip2(vec_a0, vec_a4);
      vec_t10 = svzip1(vec_a1, vec_a5);
      vec_t11 = svzip2(vec_a1, vec_a5);
      vec_t12 = svzip1(vec_a2, vec_a6);
      vec_t13 = svzip2(vec_a2, vec_a6);
      vec_t14 = svzip1(vec_a3, vec_a7);
      vec_t15 = svzip2(vec_a3, vec_a7);

      vec_t16 = svzip1(vec_t8, vec_t12);
      vec_t17 = svzip2(vec_t8, vec_t12);
      vec_t18 = svzip1(vec_t9, vec_t13);
      vec_t19 = svzip2(vec_t9, vec_t13);
      vec_t20 = svzip1(vec_t10, vec_t14);
      vec_t21 = svzip2(vec_t10, vec_t14);
      vec_t22 = svzip1(vec_t11, vec_t15);
      vec_t23 = svzip2(vec_t11, vec_t15);

      vec_at0 = svzip1(vec_t16, vec_t20);
      vec_at1 = svzip2(vec_t16, vec_t20);
      vec_at2 = svzip1(vec_t17, vec_t21);
      vec_at3 = svzip2(vec_t17, vec_t21);
      vec_at4 = svzip1(vec_t18, vec_t22);
      vec_at5 = svzip2(vec_t18, vec_t22);
      vec_at6 = svzip1(vec_t19, vec_t23);
      vec_at7 = svzip2(vec_t19, vec_t23);

      switch (n_vecs) {
        default:
          svst1(p_all, &ptr_at[7 * m], vec_at7);
        case 7:
          svst1(p_all, &ptr_at[6 * m], vec_at6);
        case 6:
          svst1(p_all, &ptr_at[5 * m], vec_at5);
        case 5:
          svst1(p_all, &ptr_at[4 * m], vec_at4);
        case 4:
          svst1(p_all, &ptr_at[3 * m], vec_at3);
        case 3:
          svst1(p_all, &ptr_at[2 * m], vec_at2);
        case 2:
          svst1(p_all, &ptr_at[1 * m], vec_at1);
        case 1:
          svst1(p_all, &ptr_at[0 * m], vec_at0);
      }

      ptr_a += vl_s;
      ptr_at += 8 * m;
    }
  }
}

static void transpose_sve_vl512(uint64_t rows, uint64_t cols, uint32_t *pIn,
                                uint32_t *pOut)
LOOP_ATTR
{
  uint64_t m = rows;
  uint64_t n = cols;
  uint32_t *a = pIn;
  uint32_t *at = pOut;

  uint32_t *ptr_a;
  uint32_t *ptr_at;
  uint64_t vl_s = svcntw();
  uint64_t n_vecs;

  svbool_t p_ld;
  svbool_t p_all = svptrue_b32();
  svuint32_t vec_a0, vec_a1, vec_a2, vec_a3;
  svuint32_t vec_a4, vec_a5, vec_a6, vec_a7;
  svuint32_t vec_a8, vec_a9, vec_a10, vec_a11;
  svuint32_t vec_a12, vec_a13, vec_a14, vec_a15;
  svuint32_t vec_at0, vec_at1, vec_at2, vec_at3;
  svuint32_t vec_at4, vec_at5, vec_at6, vec_at7;
  svuint32_t vec_at8, vec_at9, vec_at10, vec_at11;
  svuint32_t vec_at12, vec_at13, vec_at14, vec_at15;
  svuint32_t vec_t0, vec_t1, vec_t2, vec_t3;
  svuint32_t vec_t4, vec_t5, vec_t6, vec_t7;
  svuint32_t vec_t8, vec_t9, vec_t10, vec_t11;
  svuint32_t vec_t12, vec_t13, vec_t14, vec_t15;
  svuint32_t vec_t16, vec_t17, vec_t18, vec_t19;
  svuint32_t vec_t20, vec_t21, vec_t22, vec_t23;
  svuint32_t vec_t24, vec_t25, vec_t26, vec_t27;
  svuint32_t vec_t28, vec_t29, vec_t30, vec_t31;

#define LOAD(i, o) vec_a##i = svld1(p_ld, &ptr_a[o])

  for (int32_t x = 0; x < m; x += vl_s) {
    ptr_a = &a[x * n];
    ptr_at = &at[x];

    FOR_LOOP_32(uint64_t, y, 0, n, p_ld) {
      LOAD(0, 0 * n);
      LOAD(1, 1 * n);
      LOAD(2, 2 * n);
      LOAD(3, 3 * n);
      LOAD(4, 4 * n);
      LOAD(5, 5 * n);
      LOAD(6, 6 * n);
      LOAD(7, 7 * n);
      LOAD(8, 8 * n);
      LOAD(9, 9 * n);
      LOAD(10, 10 * n);
      LOAD(11, 11 * n);
      LOAD(12, 12 * n);
      LOAD(13, 13 * n);
      LOAD(14, 14 * n);
      LOAD(15, 15 * n);

      n_vecs = svcntp_b32(p_all, p_ld);

      vec_t0 = svzip1(vec_a0, vec_a8);
      vec_t1 = svzip2(vec_a0, vec_a8);
      vec_t2 = svzip1(vec_a1, vec_a9);
      vec_t3 = svzip2(vec_a1, vec_a9);
      vec_t4 = svzip1(vec_a2, vec_a10);
      vec_t5 = svzip2(vec_a2, vec_a10);
      vec_t6 = svzip1(vec_a3, vec_a11);
      vec_t7 = svzip2(vec_a3, vec_a11);
      vec_t8 = svzip1(vec_a4, vec_a12);
      vec_t9 = svzip2(vec_a4, vec_a12);
      vec_t10 = svzip1(vec_a5, vec_a13);
      vec_t11 = svzip2(vec_a5, vec_a13);
      vec_t12 = svzip1(vec_a6, vec_a14);
      vec_t13 = svzip2(vec_a6, vec_a14);
      vec_t14 = svzip1(vec_a7, vec_a15);
      vec_t15 = svzip2(vec_a7, vec_a15);

      vec_t16 = svzip1(vec_t0, vec_t8);
      vec_t17 = svzip2(vec_t0, vec_t8);
      vec_t18 = svzip1(vec_t1, vec_t9);
      vec_t19 = svzip2(vec_t1, vec_t9);
      vec_t20 = svzip1(vec_t2, vec_t10);
      vec_t21 = svzip2(vec_t2, vec_t10);
      vec_t22 = svzip1(vec_t3, vec_t11);
      vec_t23 = svzip2(vec_t3, vec_t11);
      vec_t24 = svzip1(vec_t4, vec_t12);
      vec_t25 = svzip2(vec_t4, vec_t12);
      vec_t26 = svzip1(vec_t5, vec_t13);
      vec_t27 = svzip2(vec_t5, vec_t13);
      vec_t28 = svzip1(vec_t6, vec_t14);
      vec_t29 = svzip2(vec_t6, vec_t14);
      vec_t30 = svzip1(vec_t7, vec_t15);
      vec_t31 = svzip2(vec_t7, vec_t15);

      vec_t0 = svzip1(vec_t16, vec_t24);
      vec_t1 = svzip2(vec_t16, vec_t24);
      vec_t2 = svzip1(vec_t17, vec_t25);
      vec_t3 = svzip2(vec_t17, vec_t25);
      vec_t4 = svzip1(vec_t18, vec_t26);
      vec_t5 = svzip2(vec_t18, vec_t26);
      vec_t6 = svzip1(vec_t19, vec_t27);
      vec_t7 = svzip2(vec_t19, vec_t27);
      vec_t8 = svzip1(vec_t20, vec_t28);
      vec_t9 = svzip2(vec_t20, vec_t28);
      vec_t10 = svzip1(vec_t21, vec_t29);
      vec_t11 = svzip2(vec_t21, vec_t29);
      vec_t12 = svzip1(vec_t22, vec_t30);
      vec_t13 = svzip2(vec_t22, vec_t30);
      vec_t14 = svzip1(vec_t23, vec_t31);
      vec_t15 = svzip2(vec_t23, vec_t31);

      vec_at0 = svzip1(vec_t0, vec_t8);
      vec_at1 = svzip2(vec_t0, vec_t8);
      vec_at2 = svzip1(vec_t1, vec_t9);
      vec_at3 = svzip2(vec_t1, vec_t9);
      vec_at4 = svzip1(vec_t2, vec_t10);
      vec_at5 = svzip2(vec_t2, vec_t10);
      vec_at6 = svzip1(vec_t3, vec_t11);
      vec_at7 = svzip2(vec_t3, vec_t11);
      vec_at8 = svzip1(vec_t4, vec_t12);
      vec_at9 = svzip2(vec_t4, vec_t12);
      vec_at10 = svzip1(vec_t5, vec_t13);
      vec_at11 = svzip2(vec_t5, vec_t13);
      vec_at12 = svzip1(vec_t6, vec_t14);
      vec_at13 = svzip2(vec_t6, vec_t14);
      vec_at14 = svzip1(vec_t7, vec_t15);
      vec_at15 = svzip2(vec_t7, vec_t15);

      switch (n_vecs) {
        default:
          svst1(p_all, &ptr_at[15 * m], vec_at15);
        case 15:
          svst1(p_all, &ptr_at[14 * m], vec_at14);
        case 14:
          svst1(p_all, &ptr_at[13 * m], vec_at13);
        case 13:
          svst1(p_all, &ptr_at[12 * m], vec_at12);
        case 12:
          svst1(p_all, &ptr_at[11 * m], vec_at11);
        case 11:
          svst1(p_all, &ptr_at[10 * m], vec_at10);
        case 10:
          svst1(p_all, &ptr_at[9 * m], vec_at9);
        case 9:
          svst1(p_all, &ptr_at[8 * m], vec_at8);
        case 8:
          svst1(p_all, &ptr_at[7 * m], vec_at7);
        case 7:
          svst1(p_all, &ptr_at[6 * m], vec_at6);
        case 6:
          svst1(p_all, &ptr_at[5 * m], vec_at5);
        case 5:
          svst1(p_all, &ptr_at[4 * m], vec_at4);
        case 4:
          svst1(p_all, &ptr_at[3 * m], vec_at3);
        case 3:
          svst1(p_all, &ptr_at[2 * m], vec_at2);
        case 2:
          svst1(p_all, &ptr_at[1 * m], vec_at1);
        case 1:
          svst1(p_all, &ptr_at[0 * m], vec_at0);
      }

      ptr_a += vl_s;
      ptr_at += 16 * m;
    }
  }
}

static void transpose_sve_vla(uint64_t rows, uint64_t cols, uint32_t *pIn,
                              uint32_t *pOut) {

  uint64_t m = rows;
  uint64_t n = cols;
  uint32_t *a = pIn;
  uint32_t *at = pOut;

  svbool_t p_ld;
  svuint32_t idx = svindex_u32(0, m);
  for (uint64_t x = 0; x < m; x += 1) {
    FOR_LOOP_32(uint64_t, y, 0, n, p_ld) {
      svst1_scatter_index(p_ld, &at[x + y * m], idx,
                          svld1(p_ld, &a[x * n + y]));
    }
  }
}

static void inner_loop_223(struct loop_223_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint32_t *a = data->a;
  uint32_t *at = data->at;
  uint32_t vl = get_sve_vl();

  switch (vl) {
    case 128:
      transpose_sve_vl128(m, n, a, at);
      break;
    case 256:
      transpose_sve_vl256(m, n, a, at);
      break;
    case 512:
      transpose_sve_vl512(m, n, a, at);
      break;
    default:
      transpose_sve_vla(m, n, a, at);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_223(struct loop_223_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t at = (uint64_t)data->at;
  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t m_inc = svl_s * n;
  register uint64_t ld_off2 = 2 * n;
  register uint64_t ld_off3 = 3 * n;
  register uint64_t st_off2 = 2 * m;
  register uint64_t st_off3 = 3 * m;
  register uint64_t nblk = m * svl_s;
  register uint64_t a_ptr;
  register uint64_t at_ptr;
  register uint64_t a_base;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  // x12: slice index register for psel and mova

  asm volatile(
      "   add     %[m_cnd], %[at_dst], %[m], lsl #2                       \n"
      "   whilelt pn8.b, %[at_dst], %[m_cnd], vlx2                        \n"

      // M loop
      "1:                                                                 \n"
      "   mov     %[a_base], %[a_src]                                     \n"
      "   add     %[n_cnd], %[a_src], %[n], lsl #2                        \n"
      "   mov     %[at_ptr], %[at_dst]                                    \n"
      "   whilelt pn9.b, %[a_src], %[n_cnd], vlx2                         \n"
      "   pext    { p0.s, p1.s }, pn8[0]                                  \n"

      // N loop
      "2:                                                                 \n"
      "   mov     %[a_ptr], %[a_base]                                     \n"
      "   pext    { p2.s, p3.s }, pn9[0]                                  \n"

      // ZA0_ZA1 load loop
      "   mov     x12, #0                                                 \n"
      "3:                                                                 \n"
      "   psel    pn10, pn9, p0.s[w12, 0]                                 \n"
      "   psel    pn11, pn9, p0.s[w12, 1]                                 \n"
      "   psel    pn12, pn9, p0.s[w12, 2]                                 \n"
      "   psel    pn13, pn9, p0.s[w12, 3]                                 \n"
      "   ld1w    {z0.s, z8.s},   pn10/z, [%[a_ptr]]                      \n"
      "   ld1w    {z1.s, z9.s},   pn11/z, [%[a_ptr], %[ld_off1], lsl #2]  \n"
      "   ld1w    {z2.s, z10.s},  pn12/z, [%[a_ptr], %[ld_off2], lsl #2]  \n"
      "   ld1w    {z3.s, z11.s},  pn13/z, [%[a_ptr], %[ld_off3], lsl #2]  \n"
      "   mova    za0h.s[w12, 0:3], {z0.s-z3.s}                           \n"
      "   mova    za1h.s[w12, 0:3], {z8.s-z11.s}                          \n"
      "   add     %[a_ptr], %[a_ptr], %[n], lsl #4                        \n"
      "   add     x12, x12, #4                                            \n"
      "   cmp     x12, %[l_cnd]                                           \n"
      "   b.mi    3b                                                      \n"

      // ZA2_ZA3 load loop
      "   mov     x12, #0                                                 \n"
      "4:                                                                 \n"
      "   psel    pn10, pn9, p1.s[w12, 0]                                 \n"
      "   psel    pn11, pn9, p1.s[w12, 1]                                 \n"
      "   psel    pn12, pn9, p1.s[w12, 2]                                 \n"
      "   psel    pn13, pn9, p1.s[w12, 3]                                 \n"
      "   ld1w    {z0.s, z8.s},   pn10/z, [%[a_ptr]]                      \n"
      "   ld1w    {z1.s, z9.s},   pn11/z, [%[a_ptr], %[ld_off1], lsl #2]  \n"
      "   ld1w    {z2.s, z10.s},  pn12/z, [%[a_ptr], %[ld_off2], lsl #2]  \n"
      "   ld1w    {z3.s, z11.s},  pn13/z, [%[a_ptr], %[ld_off3], lsl #2]  \n"
      "   mova    za2h.s[w12, 0:3], {z0.s-z3.s}                           \n"
      "   mova    za3h.s[w12, 0:3], {z8.s-z11.s}                          \n"
      "   add     %[a_ptr], %[a_ptr], %[n], lsl #4                        \n"
      "   add     x12, x12, #4                                            \n"
      "   cmp     x12, %[l_cnd]                                           \n"
      "   b.mi    4b                                                      \n"

      // ZA0_ZA2 store loop
      "   mov     x12, #0                                                 \n"
      "5:                                                                 \n"
      "   psel    pn10, pn8, p2.s[w12, 0]                                 \n"
      "   psel    pn11, pn8, p2.s[w12, 1]                                 \n"
      "   psel    pn12, pn8, p2.s[w12, 2]                                 \n"
      "   psel    pn13, pn8, p2.s[w12, 3]                                 \n"
      "   mova    {z0.s-z3.s},  za0v.s[w12, 0:3]                          \n"
      "   mova    {z8.s-z11.s}, za2v.s[w12, 0:3]                          \n"
      "   st1w    {z0.s, z8.s},    pn10, [%[at_ptr]]                      \n"
      "   st1w    {z1.s, z9.s},    pn11, [%[at_ptr], %[st_off1], lsl #2]  \n"
      "   st1w    {z2.s, z10.s},   pn12, [%[at_ptr], %[st_off2], lsl #2]  \n"
      "   st1w    {z3.s, z11.s},   pn13, [%[at_ptr], %[st_off3], lsl #2]  \n"
      "   add     %[at_ptr], %[at_ptr], %[m], lsl #4                      \n"
      "   add     x12, x12, #4                                            \n"
      "   cmp     x12, %[l_cnd]                                           \n"
      "   b.mi    5b                                                      \n"

      // ZA1_ZA3 store loop
      "   mov     x12, #0                                                 \n"
      "6:                                                                 \n"
      "   psel    pn10, pn8, p3.s[w12, 0]                                 \n"
      "   psel    pn11, pn8, p3.s[w12, 1]                                 \n"
      "   psel    pn12, pn8, p3.s[w12, 2]                                 \n"
      "   psel    pn13, pn8, p3.s[w12, 3]                                 \n"
      "   mova    {z0.s-z3.s},  za1v.s[w12, 0:3]                          \n"
      "   mova    {z8.s-z11.s}, za3v.s[w12, 0:3]                          \n"
      "   st1w    {z0.s, z8.s},    pn10, [%[at_ptr]]                      \n"
      "   st1w    {z1.s, z9.s},    pn11, [%[at_ptr], %[st_off1], lsl #2]  \n"
      "   st1w    {z2.s, z10.s},   pn12, [%[at_ptr], %[st_off2], lsl #2]  \n"
      "   st1w    {z3.s, z11.s},   pn13, [%[at_ptr], %[st_off3], lsl #2]  \n"
      "   add     %[at_ptr], %[at_ptr], %[m], lsl #4                      \n"
      "   add     x12, x12, #4                                            \n"
      "   cmp     x12, %[l_cnd]                                           \n"
      "   b.mi    6b                                                      \n"

      // N loop tail
      "   addvl   %[a_base], %[a_base], #2                                \n"
      "   whilelt pn9.b, %[a_base], %[n_cnd], vlx2                        \n"
      "   b.mi 2b                                                         \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[m_inc], lsl #3                    \n"
      "   addvl   %[at_dst], %[at_dst], #2                                \n"
      "   whilelt pn8.b, %[at_dst], %[m_cnd], vlx2                        \n"
      "   b.mi 1b                                                         \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [a_base] "=&r"(a_base), [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd),
        [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [m_inc] "r"(m_inc), [nblk] "r"(nblk),
        [ld_off1] "r"(n), [ld_off2] "r"(ld_off2), [ld_off3] "r"(ld_off3),
        [st_off1] "r"(m), [st_off2] "r"(st_off2), [st_off3] "r"(st_off3),
        [l_cnd] "r"(svl_s)
      : "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "p0", "p1", "p2",
        "p3", "p8", "p10", "p11", "p12", "p13", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void transpose_sve_vl128(uint64_t m, uint64_t n, uint32_t *restrict a,
                                uint32_t *restrict at)
LOOP_ATTR
{
  register uint64_t ld_off2 = 2 * n;
  register uint64_t ld_off3 = 3 * n;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  register uint64_t st_off2 = 2 * m;
  register uint64_t st_off3 = 3 * m;
  register uint64_t a_ptr;
  register uint64_t at_ptr;
  register uint64_t n_vecs;
  // x12: vector select register for psel

  asm volatile(
      "   ptrue p0.s                                                \n"
      "   add   %[m_cnd], %[at_dst], %[m], lsl #2                   \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   mov   x12, #0                                             \n"
#endif

      // M loop
      "1:                                                           \n"
      "   mov   %[a_ptr], %[a_src]                                  \n"
      "   mov   %[at_ptr], %[at_dst]                                \n"
      "   add   %[n_cnd], %[a_src], %[n], lsl #2                    \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"

      // N loop
      "2:                                                           \n"
      "   ld1w    { z0.s },  p1/z, [%[a_ptr]]                       \n"
      "   ld1w    { z1.s },  p1/z, [%[a_ptr], %[ld_off1], lsl #2]   \n"
      "   ld1w    { z2.s },  p1/z, [%[a_ptr], %[ld_off2], lsl #2]   \n"
      "   ld1w    { z3.s },  p1/z, [%[a_ptr], %[ld_off3], lsl #2]   \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   psel    p2, p0, p1.s[w12, 1]                              \n"
      "   psel    p3, p0, p1.s[w12, 2]                              \n"
      "   psel    p4, p0, p1.s[w12, 3]                              \n"
#else
      "   cntp    %[n_vecs], p0, p1.s                               \n"
#endif

      "   zip1    z16.s, z0.s, z2.s                                 \n"
      "   zip2    z17.s, z0.s, z2.s                                 \n"
      "   zip1    z18.s, z1.s, z3.s                                 \n"
      "   zip2    z19.s, z1.s, z3.s                                 \n"

      "   zip1    z0.s, z16.s, z18.s                                \n"
      "   zip2    z1.s, z16.s, z18.s                                \n"
      "   zip1    z2.s, z17.s, z19.s                                \n"
      "   zip2    z3.s, z17.s, z19.s                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   st1w    { z1.s }, p2, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   st1w    { z2.s }, p3, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   st1w    { z3.s }, p4, [%[at_ptr], %[st_off3], lsl #2]     \n"
#else
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   cmp     %[n_vecs], #1                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z1.s }, p0, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   cmp     %[n_vecs], #2                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z2.s }, p0, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   cmp     %[n_vecs], #3                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z3.s }, p0, [%[at_ptr], %[st_off3], lsl #2]     \n"
#endif
      "3:                                                           \n"
      // N loop tail
      "   addvl   %[a_ptr], %[a_ptr], #1                            \n"
      "   add     %[at_ptr], %[at_ptr], %[m], lsl #4                \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"
      "   b.first 2b                                                \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[n], lsl #4                  \n"
      "   addvl   %[at_dst], %[at_dst], #1                          \n"
      "   cmp     %[at_dst], %[m_cnd]                               \n"
      "   b.lt 1b                                                   \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd), [n_vecs] "=&r"(n_vecs),
        [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [ld_off1] "r"(n), [ld_off2] "r"(ld_off2),
        [ld_off3] "r"(ld_off3), [st_off1] "r"(m), [st_off2] "r"(st_off2),
        [st_off3] "r"(st_off3)
      : "z0", "z1", "z2", "z3", "z16", "z17", "z18", "z19", "p0", "p1",
#if defined(__ARM_FEATURE_SVE2p1)
        "p2", "p3", "p4", "x12",
#endif
        "cc", "memory");
}

static void transpose_sve_vl256(uint64_t m, uint64_t n, uint32_t *a,
                                uint32_t *at)
LOOP_ATTR
{
  register uint64_t ld_off2 = 2 * n;
  register uint64_t ld_off3 = 3 * n;
  register uint64_t ld_off4 = 4 * n;
  register uint64_t ld_off5 = 5 * n;
  register uint64_t ld_off6 = 6 * n;
  register uint64_t ld_off7 = 7 * n;
  register uint64_t st_off2 = 2 * m;
  register uint64_t st_off3 = 3 * m;
  register uint64_t st_off4 = 4 * m;
  register uint64_t st_off5 = 5 * m;
  register uint64_t st_off6 = 6 * m;
  register uint64_t st_off7 = 7 * m;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  register uint64_t a_ptr;
  register uint64_t at_ptr;
  register uint64_t n_vecs;
  // x12: vector select register for psel
  // x13: vector select register for psel

  // 8x8 32b block-based transposition
  asm volatile(
      "   ptrue p0.s                                                \n"
      "   add   %[m_cnd], %[at_dst], %[m], lsl #2                   \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   mov   x12, #0                                             \n"
      "   mov   x13, #4                                             \n"
#endif

      // M loop
      "1:                                                           \n"
      "   mov   %[a_ptr], %[a_src]                                  \n"
      "   mov   %[at_ptr], %[at_dst]                                \n"
      "   add   %[n_cnd], %[a_src], %[n], lsl #2                    \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"

      // N loop
      "2:                                                           \n"
      "   ld1w    { z0.s },  p1/z, [%[a_ptr]]                       \n"
      "   ld1w    { z1.s },  p1/z, [%[a_ptr], %[ld_off1], lsl #2]   \n"
      "   ld1w    { z2.s },  p1/z, [%[a_ptr], %[ld_off2], lsl #2]   \n"
      "   ld1w    { z3.s },  p1/z, [%[a_ptr], %[ld_off3], lsl #2]   \n"
      "   ld1w    { z4.s },  p1/z, [%[a_ptr], %[ld_off4], lsl #2]   \n"
      "   ld1w    { z5.s },  p1/z, [%[a_ptr], %[ld_off5], lsl #2]   \n"
      "   ld1w    { z6.s },  p1/z, [%[a_ptr], %[ld_off6], lsl #2]   \n"
      "   ld1w    { z7.s },  p1/z, [%[a_ptr], %[ld_off7], lsl #2]   \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   psel    p2, p0, p1.s[w12, 1]                              \n"
      "   psel    p3, p0, p1.s[w12, 2]                              \n"
      "   psel    p4, p0, p1.s[w12, 3]                              \n"
#else
      "   cntp    %[n_vecs], p0, p1.s                               \n"
#endif

      "   zip1    z16.s, z0.s, z4.s                                 \n"
      "   zip2    z17.s, z0.s, z4.s                                 \n"
      "   zip1    z18.s, z1.s, z5.s                                 \n"
      "   zip2    z19.s, z1.s, z5.s                                 \n"
      "   zip1    z20.s, z2.s, z6.s                                 \n"
      "   zip2    z21.s, z2.s, z6.s                                 \n"
      "   zip1    z22.s, z3.s, z7.s                                 \n"
      "   zip2    z23.s, z3.s, z7.s                                 \n"

      "   zip1    z24.s, z16.s, z20.s                               \n"
      "   zip2    z25.s, z16.s, z20.s                               \n"
      "   zip1    z26.s, z17.s, z21.s                               \n"
      "   zip2    z27.s, z17.s, z21.s                               \n"
      "   zip1    z28.s, z18.s, z22.s                               \n"
      "   zip2    z29.s, z18.s, z22.s                               \n"
      "   zip1    z30.s, z19.s, z23.s                               \n"
      "   zip2    z31.s, z19.s, z23.s                               \n"

      "   zip1    z0.s, z24.s, z28.s                                \n"
      "   zip2    z1.s, z24.s, z28.s                                \n"
      "   zip1    z2.s, z25.s, z29.s                                \n"
      "   zip2    z3.s, z25.s, z29.s                                \n"
      "   zip1    z4.s, z26.s, z30.s                                \n"
      "   zip2    z5.s, z26.s, z30.s                                \n"
      "   zip1    z6.s, z27.s, z31.s                                \n"
      "   zip2    z7.s, z27.s, z31.s                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   st1w    { z1.s }, p2, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   st1w    { z2.s }, p3, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   st1w    { z3.s }, p4, [%[at_ptr], %[st_off3], lsl #2]     \n"
      "   psel    p2, p0, p1.s[w13, 0]                              \n"
      "   psel    p3, p0, p1.s[w13, 1]                              \n"
      "   psel    p4, p0, p1.s[w13, 2]                              \n"
      "   psel    p5, p0, p1.s[w13, 3]                              \n"
      "   st1w    { z4.s }, p2, [%[at_ptr], %[st_off4], lsl #2]     \n"
      "   st1w    { z5.s }, p3, [%[at_ptr], %[st_off5], lsl #2]     \n"
      "   st1w    { z6.s }, p4, [%[at_ptr], %[st_off6], lsl #2]     \n"
      "   st1w    { z7.s }, p5, [%[at_ptr], %[st_off7], lsl #2]     \n"
#else
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   cmp     %[n_vecs], #1                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z1.s }, p0, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   cmp     %[n_vecs], #2                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z2.s }, p0, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   cmp     %[n_vecs], #3                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z3.s }, p0, [%[at_ptr], %[st_off3], lsl #2]     \n"
      "   cmp     %[n_vecs], #4                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z4.s }, p0, [%[at_ptr], %[st_off4], lsl #2]     \n"
      "   cmp     %[n_vecs], #5                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z5.s }, p0, [%[at_ptr], %[st_off5], lsl #2]     \n"
      "   cmp     %[n_vecs], #6                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z6.s }, p0, [%[at_ptr], %[st_off6], lsl #2]     \n"
      "   cmp     %[n_vecs], #7                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z7.s }, p0, [%[at_ptr], %[st_off7], lsl #2]     \n"
#endif
      "3:                                                           \n"
      // N loop tail
      "   addvl   %[a_ptr], %[a_ptr], #1                            \n"
      "   add     %[at_ptr], %[at_ptr], %[m], lsl #5                \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"
      "   b.first 2b                                                \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[n], lsl #5                  \n"
      "   addvl   %[at_dst], %[at_dst], #1                          \n"
      "   cmp     %[at_dst], %[m_cnd]                               \n"
      "   b.lt 1b                                                   \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd), [n_vecs] "=&r"(n_vecs),
        [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [ld_off1] "r"(n), [ld_off2] "r"(ld_off2),
        [ld_off3] "r"(ld_off3), [ld_off4] "r"(ld_off4), [ld_off5] "r"(ld_off5),
        [ld_off6] "r"(ld_off6), [ld_off7] "r"(ld_off7), [st_off1] "r"(m),
        [st_off2] "r"(st_off2), [st_off3] "r"(st_off3), [st_off4] "r"(st_off4),
        [st_off5] "r"(st_off5), [st_off6] "r"(st_off6), [st_off7] "r"(st_off7)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18",
        "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
        "z29", "z30", "z31", "p0", "p1",
#if defined(__ARM_FEATURE_SVE2p1)
        "p2", "p3", "p4", "p5", "x12", "x13",
#endif
        "cc", "memory");
}

static void transpose_sve_vl512(uint64_t m, uint64_t n, uint32_t *a,
                                uint32_t *at)
LOOP_ATTR
{
  register uint64_t ld_off2 = 2 * n;
  register uint64_t ld_off3 = 3 * n;
  register uint64_t ld_off4 = 4 * n;
  register uint64_t ld_off5 = 5 * n;
  register uint64_t ld_off6 = 6 * n;
  register uint64_t ld_off7 = 7 * n;
  register uint64_t st_off2 = 2 * m;
  register uint64_t st_off3 = 3 * m;
  register uint64_t st_off4 = 4 * m;
  register uint64_t st_off5 = 5 * m;
  register uint64_t st_off6 = 6 * m;
  register uint64_t st_off7 = 7 * m;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  register uint64_t a_ptr;
  register uint64_t at_ptr;
  register uint64_t a_ptr2;
  register uint64_t at_ptr2;
  register uint64_t n_vecs;
  // x12: vector select register for psel
  // x13: vector select register for psel
  // x14: vector select register for psel
  // x15: vector select register for psel

  // 16x16 32b block-based transposition
  asm volatile(
      "   ptrue p0.s                                                \n"
      "   add   %[m_cnd], %[at_dst], %[m], lsl #2                   \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   mov   x12, #0                                             \n"
      "   mov   x13, #4                                             \n"
      "   mov   x14, #8                                             \n"
      "   mov   x15, #12                                            \n"
#endif

      // M loop
      "1:                                                           \n"
      "   mov   %[a_ptr], %[a_src]                                  \n"
      "   mov   %[at_ptr], %[at_dst]                                \n"
      "   add   %[n_cnd], %[a_src], %[n], lsl #2                    \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"

      // N loop
      "2:                                                           \n"
      "   add     %[a_ptr2], %[a_ptr], %[n], lsl #5                 \n"
      "   add     %[at_ptr2], %[at_ptr], %[m], lsl #5               \n"
      "   ld1w    { z0.s },  p1/z, [%[a_ptr]]                       \n"
      "   ld1w    { z1.s },  p1/z, [%[a_ptr], %[ld_off1], lsl #2]   \n"
      "   ld1w    { z2.s },  p1/z, [%[a_ptr], %[ld_off2], lsl #2]   \n"
      "   ld1w    { z3.s },  p1/z, [%[a_ptr], %[ld_off3], lsl #2]   \n"
      "   ld1w    { z4.s },  p1/z, [%[a_ptr], %[ld_off4], lsl #2]   \n"
      "   ld1w    { z5.s },  p1/z, [%[a_ptr], %[ld_off5], lsl #2]   \n"
      "   ld1w    { z6.s },  p1/z, [%[a_ptr], %[ld_off6], lsl #2]   \n"
      "   ld1w    { z7.s },  p1/z, [%[a_ptr], %[ld_off7], lsl #2]   \n"
      "   ld1w    { z8.s },  p1/z, [%[a_ptr2]]                      \n"
      "   ld1w    { z9.s },  p1/z, [%[a_ptr2], %[ld_off1], lsl #2]  \n"
      "   ld1w    { z10.s }, p1/z, [%[a_ptr2], %[ld_off2], lsl #2]  \n"
      "   ld1w    { z11.s }, p1/z, [%[a_ptr2], %[ld_off3], lsl #2]  \n"
      "   ld1w    { z12.s }, p1/z, [%[a_ptr2], %[ld_off4], lsl #2]  \n"
      "   ld1w    { z13.s }, p1/z, [%[a_ptr2], %[ld_off5], lsl #2]  \n"
      "   ld1w    { z14.s }, p1/z, [%[a_ptr2], %[ld_off6], lsl #2]  \n"
      "   ld1w    { z15.s }, p1/z, [%[a_ptr2], %[ld_off7], lsl #2]  \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   psel    p2, p0, p1.s[w12, 1]                              \n"
      "   psel    p3, p0, p1.s[w12, 2]                              \n"
      "   psel    p4, p0, p1.s[w12, 3]                              \n"
#else
      "   cntp    %[n_vecs], p0, p1.s                               \n"
#endif
      "   zip1    z16.s, z0.s, z8.s                                 \n"
      "   zip2    z17.s, z0.s, z8.s                                 \n"
      "   zip1    z18.s, z1.s, z9.s                                 \n"
      "   zip2    z19.s, z1.s, z9.s                                 \n"
      "   zip1    z20.s, z2.s, z10.s                                \n"
      "   zip2    z21.s, z2.s, z10.s                                \n"
      "   zip1    z22.s, z3.s, z11.s                                \n"
      "   zip2    z23.s, z3.s, z11.s                                \n"
      "   zip1    z24.s, z4.s, z12.s                                \n"
      "   zip2    z25.s, z4.s, z12.s                                \n"
      "   zip1    z26.s, z5.s, z13.s                                \n"
      "   zip2    z27.s, z5.s, z13.s                                \n"
      "   zip1    z28.s, z6.s, z14.s                                \n"
      "   zip2    z29.s, z6.s, z14.s                                \n"
      "   zip1    z30.s, z7.s, z15.s                                \n"
      "   zip2    z31.s, z7.s, z15.s                                \n"

      "   zip1    z0.s, z16.s, z24.s                                \n"
      "   zip2    z1.s, z16.s, z24.s                                \n"
      "   zip1    z2.s, z17.s, z25.s                                \n"
      "   zip2    z3.s, z17.s, z25.s                                \n"
      "   zip1    z4.s, z18.s, z26.s                                \n"
      "   zip2    z5.s, z18.s, z26.s                                \n"
      "   zip1    z6.s, z19.s, z27.s                                \n"
      "   zip2    z7.s, z19.s, z27.s                                \n"
      "   zip1    z8.s, z20.s, z28.s                                \n"
      "   zip2    z9.s, z20.s, z28.s                                \n"
      "   zip1    z10.s, z21.s, z29.s                               \n"
      "   zip2    z11.s, z21.s, z29.s                               \n"
      "   zip1    z12.s, z22.s, z30.s                               \n"
      "   zip2    z13.s, z22.s, z30.s                               \n"
      "   zip1    z14.s, z23.s, z31.s                               \n"
      "   zip2    z15.s, z23.s, z31.s                               \n"

      "   zip1    z16.s, z0.s, z8.s                                 \n"
      "   zip2    z17.s, z0.s, z8.s                                 \n"
      "   zip1    z18.s, z1.s, z9.s                                 \n"
      "   zip2    z19.s, z1.s, z9.s                                 \n"
      "   zip1    z20.s, z2.s, z10.s                                \n"
      "   zip2    z21.s, z2.s, z10.s                                \n"
      "   zip1    z22.s, z3.s, z11.s                                \n"
      "   zip2    z23.s, z3.s, z11.s                                \n"
      "   zip1    z24.s, z4.s, z12.s                                \n"
      "   zip2    z25.s, z4.s, z12.s                                \n"
      "   zip1    z26.s, z5.s, z13.s                                \n"
      "   zip2    z27.s, z5.s, z13.s                                \n"
      "   zip1    z28.s, z6.s, z14.s                                \n"
      "   zip2    z29.s, z6.s, z14.s                                \n"
      "   zip1    z30.s, z7.s, z15.s                                \n"
      "   zip2    z31.s, z7.s, z15.s                                \n"

      "   zip1    z0.s, z16.s, z24.s                                \n"
      "   zip2    z1.s, z16.s, z24.s                                \n"
      "   zip1    z2.s, z17.s, z25.s                                \n"
      "   zip2    z3.s, z17.s, z25.s                                \n"
      "   zip1    z4.s, z18.s, z26.s                                \n"
      "   zip2    z5.s, z18.s, z26.s                                \n"
      "   zip1    z6.s, z19.s, z27.s                                \n"
      "   zip2    z7.s, z19.s, z27.s                                \n"
      "   zip1    z8.s, z20.s, z28.s                                \n"
      "   zip2    z9.s, z20.s, z28.s                                \n"
      "   zip1    z10.s, z21.s, z29.s                               \n"
      "   zip2    z11.s, z21.s, z29.s                               \n"
      "   zip1    z12.s, z22.s, z30.s                               \n"
      "   zip2    z13.s, z22.s, z30.s                               \n"
      "   zip1    z14.s, z23.s, z31.s                               \n"
      "   zip2    z15.s, z23.s, z31.s                               \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   st1w    { z1.s }, p2, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   st1w    { z2.s }, p3, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   st1w    { z3.s }, p4, [%[at_ptr], %[st_off3], lsl #2]     \n"
      "   psel    p2, p0, p1.s[w13, 0]                              \n"
      "   psel    p3, p0, p1.s[w13, 1]                              \n"
      "   psel    p4, p0, p1.s[w13, 2]                              \n"
      "   psel    p5, p0, p1.s[w13, 3]                              \n"
      "   st1w    { z4.s }, p2, [%[at_ptr], %[st_off4], lsl #2]     \n"
      "   st1w    { z5.s }, p3, [%[at_ptr], %[st_off5], lsl #2]     \n"
      "   st1w    { z6.s }, p4, [%[at_ptr], %[st_off6], lsl #2]     \n"
      "   st1w    { z7.s }, p5, [%[at_ptr], %[st_off7], lsl #2]     \n"
      "   psel    p2, p0, p1.s[w14, 0]                              \n"
      "   psel    p3, p0, p1.s[w14, 1]                              \n"
      "   psel    p4, p0, p1.s[w14, 2]                              \n"
      "   psel    p5, p0, p1.s[w14, 3]                              \n"
      "   st1w    { z8.s }, p2, [%[at_ptr2]]                        \n"
      "   st1w    { z9.s }, p3, [%[at_ptr2], %[st_off1], lsl #2]    \n"
      "   st1w    { z10.s }, p4, [%[at_ptr2], %[st_off2], lsl #2]   \n"
      "   st1w    { z11.s }, p5, [%[at_ptr2], %[st_off3], lsl #2]   \n"
      "   psel    p2, p0, p1.s[w15, 0]                              \n"
      "   psel    p3, p0, p1.s[w15, 1]                              \n"
      "   psel    p4, p0, p1.s[w15, 2]                              \n"
      "   psel    p5, p0, p1.s[w15, 3]                              \n"
      "   st1w    { z12.s }, p2, [%[at_ptr2], %[st_off4], lsl #2]   \n"
      "   st1w    { z13.s }, p3, [%[at_ptr2], %[st_off5], lsl #2]   \n"
      "   st1w    { z14.s }, p4, [%[at_ptr2], %[st_off6], lsl #2]   \n"
      "   st1w    { z15.s }, p5, [%[at_ptr2], %[st_off7], lsl #2]   \n"
#else
      "   st1w    { z0.s }, p0, [%[at_ptr]]                         \n"
      "   cmp     %[n_vecs], #1                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z1.s }, p0, [%[at_ptr], %[st_off1], lsl #2]     \n"
      "   cmp     %[n_vecs], #2                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z2.s }, p0, [%[at_ptr], %[st_off2], lsl #2]     \n"
      "   cmp     %[n_vecs], #3                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z3.s }, p0, [%[at_ptr], %[st_off3], lsl #2]     \n"
      "   cmp     %[n_vecs], #4                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z4.s }, p0, [%[at_ptr], %[st_off4], lsl #2]     \n"
      "   cmp     %[n_vecs], #5                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z5.s }, p0, [%[at_ptr], %[st_off5], lsl #2]     \n"
      "   cmp     %[n_vecs], #6                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z6.s }, p0, [%[at_ptr], %[st_off6], lsl #2]     \n"
      "   cmp     %[n_vecs], #7                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z7.s }, p0, [%[at_ptr], %[st_off7], lsl #2]     \n"
      "   cmp     %[n_vecs], #8                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z8.s }, p0, [%[at_ptr2]]                        \n"
      "   cmp     %[n_vecs], #9                                     \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z9.s }, p0, [%[at_ptr2], %[st_off1], lsl #2]    \n"
      "   cmp     %[n_vecs], #10                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z10.s }, p0, [%[at_ptr2], %[st_off2], lsl #2]   \n"
      "   cmp     %[n_vecs], #11                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z11.s }, p0, [%[at_ptr2], %[st_off3], lsl #2]   \n"
      "   cmp     %[n_vecs], #12                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z12.s }, p0, [%[at_ptr2], %[st_off4], lsl #2]   \n"
      "   cmp     %[n_vecs], #13                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z13.s }, p0, [%[at_ptr2], %[st_off5], lsl #2]   \n"
      "   cmp     %[n_vecs], #14                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z14.s }, p0, [%[at_ptr2], %[st_off6], lsl #2]   \n"
      "   cmp     %[n_vecs], #15                                    \n"
      "   b.lt    3f                                                \n"

      "   st1w    { z15.s }, p0, [%[at_ptr2], %[st_off7], lsl #2]   \n"
#endif
      "3:                                                           \n"
      // N loop tail
      "   addvl   %[a_ptr], %[a_ptr], #1                            \n"
      "   add     %[at_ptr], %[at_ptr], %[m], lsl #6                \n"
      "   whilelt  p1.b,  %[a_ptr], %[n_cnd]                        \n"
      "   b.first 2b                                                \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[n], lsl #6                  \n"
      "   addvl   %[at_dst], %[at_dst], #1                          \n"
      "   cmp     %[at_dst], %[m_cnd]                               \n"
      "   b.lt 1b                                                   \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd), [a_ptr2] "=&r"(a_ptr2),
        [at_ptr2] "=&r"(at_ptr2), [n_vecs] "=&r"(n_vecs), [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [ld_off1] "r"(n), [ld_off2] "r"(ld_off2),
        [ld_off3] "r"(ld_off3), [ld_off4] "r"(ld_off4), [ld_off5] "r"(ld_off5),
        [ld_off6] "r"(ld_off6), [ld_off7] "r"(ld_off7), [st_off1] "r"(m),
        [st_off2] "r"(st_off2), [st_off3] "r"(st_off3), [st_off4] "r"(st_off4),
        [st_off5] "r"(st_off5), [st_off6] "r"(st_off6), [st_off7] "r"(st_off7)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
        "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20",
        "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30",
        "z31", "p0", "p1",
#if defined(__ARM_FEATURE_SVE2p1)
        "p2", "p3", "p4", "p5", "x12", "x13", "x14", "x15",
#endif
        "cc", "memory");
}

static void transpose_sve_vla(uint64_t rows, uint64_t cols, uint32_t *pIn,
                              uint32_t *pOut){
  register uint64_t m = rows;
  register uint64_t n = cols;
  register uint64_t a = (uint64_t)pIn;
  register uint64_t at = (uint64_t)pOut;

  register uint64_t vl_s;
  asm volatile("cntw %[v]" : [v] "=r"(vl_s)::);
  register uint64_t m_inc = vl_s * m;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  register uint64_t a_ptr;
  register uint64_t at_ptr;

  asm volatile(
      "   add   %[m_cnd], %[at_dst], %[m], lsl #2           \n"
      "   index z0.s, #0, %w[m]                             \n"

      // M loop
      "1:                                                   \n"
      "   mov   %[at_ptr], %[at_dst]                        \n"
      "   mov   %[a_ptr], %[a_src]                          \n"
      "   add   %[n_cnd], %[a_src], %[n], lsl #2            \n"
      "   whilelt p0.b, %[a_src], %[n_cnd]                  \n"

      // N loop
      "2:                                                   \n"
      "   ld1w    { z1.s },  p0/z, [%[a_ptr]]               \n"
      "   st1w    { z1.s },  p0, [%[at_ptr], z0.s, uxtw #2] \n"
      "   add     %[a_ptr], %[a_ptr], %[vl_s], lsl #2       \n"
      "   add     %[at_ptr], %[at_ptr], %[m_inc], lsl #2    \n"
      "   whilelt p0.b, %[a_ptr], %[n_cnd]                  \n"
      "   b.first 2b                                        \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[n], lsl #2          \n"
      "   add     %[at_dst], %[at_dst], #4                  \n"
      "   cmp     %[at_dst], %[m_cnd]                       \n"
      "   b.lt 1b                                           \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd), [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [vl_s] "r"(vl_s), [m_inc] "r"(m_inc)
      : "z0", "z1", "p0", "cc", "memory");
}

static void inner_loop_223(struct loop_223_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint32_t *a = data->a;
  uint32_t *at = data->at;
  uint64_t vl;
  asm volatile("cntb %[v]" : [v] "=&r"(vl)::);

  switch (vl) {
    case 16:
      transpose_sve_vl128(m, n, a, at);
      break;
    case 32:
      transpose_sve_vl256(m, n, a, at);
      break;
    case 64:
      transpose_sve_vl512(m, n, a, at);
      break;
    default:
      transpose_sve_vla(m, n, a, at);
  }
}

#elif defined(__ARM_NEON)

static void inner_loop_223(struct loop_223_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t at = (uint64_t)data->at;

  register uint64_t row_inc = n * 4;
  register uint64_t col_inc = m * 4;
  register uint64_t m_last;
  register uint64_t n_last;
  register uint64_t m_cnd;
  register uint64_t n_cnd;
  register uint64_t a_ptr;
  register uint64_t at_ptr;
  register uint64_t a_base;
  register uint64_t rnd_4 = 0xfffffffc;

  asm volatile(
      "   and   %[m_last], %[m], %[rnd_4]              \n"
      "   and   %[n_last], %[n], %[rnd_4]              \n"
      "   add   %[m_cnd], %[at_dst], %[m_last], lsl #2 \n"

      // M loop
      "1:                                              \n"
      "   mov   %[a_base], %[a_src]                    \n"
      "   mov   %[at_ptr], %[at_dst]                   \n"
      "   add   %[n_cnd], %[a_src], %[n_last], lsl #2  \n"

      // N loop
      "2:                                              \n"
      "   mov   %[a_ptr], %[a_base]                    \n"
      "   ld1   { v0.4s }, [%[a_ptr]], %[row_inc]      \n"
      "   ld1   { v1.4s }, [%[a_ptr]], %[row_inc]      \n"
      "   ld1   { v2.4s }, [%[a_ptr]], %[row_inc]      \n"
      "   ld1   { v3.4s }, [%[a_ptr]], %[row_inc]      \n"

      "   zip1  v16.4s, v0.4s, v2.4s                   \n"
      "   zip2  v17.4s, v0.4s, v2.4s                   \n"
      "   zip1  v18.4s, v1.4s, v3.4s                   \n"
      "   zip2  v19.4s, v1.4s, v3.4s                   \n"

      "   zip1  v0.4s, v16.4s, v18.4s                  \n"
      "   zip2  v1.4s, v16.4s, v18.4s                  \n"
      "   zip1  v2.4s, v17.4s, v19.4s                  \n"
      "   zip2  v3.4s, v17.4s, v19.4s                  \n"

      "   st1   { v0.4s }, [%[at_ptr]], %[col_inc]     \n"
      "   st1   { v1.4s }, [%[at_ptr]], %[col_inc]     \n"
      "   st1   { v2.4s }, [%[at_ptr]], %[col_inc]     \n"
      "   st1   { v3.4s }, [%[at_ptr]], %[col_inc]     \n"

      // N loop tail
      "   add   %[a_base], %[a_base], #16              \n"
      "   cmp   %[a_base], %[n_cnd]                    \n"
      "   b.lt 2b                                      \n"

      // M loop tail
      "   add   %[a_src], %[a_src], %[n], lsl #4       \n"
      "   add   %[at_dst], %[at_dst], #16              \n"
      "   cmp   %[at_dst], %[m_cnd]                    \n"
      "   b.lt 1b                                      \n"

      : [a_ptr] "=&r"(a_ptr), [at_ptr] "=&r"(at_ptr), [at_dst] "+&r"(at),
        [m_cnd] "=&r"(m_cnd), [n_cnd] "=&r"(n_cnd), [m_last] "=&r"(m_last),
        [n_last] "=&r"(n_last), [a_base] "=&r"(a_base), [a_src] "+&r"(a)
      : [m] "r"(m), [n] "r"(n), [row_inc] "r"(row_inc), [col_inc] "r"(col_inc),
        [rnd_4] "r"(rnd_4)
      : "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "cc", "memory");

  // n dimension tails loop in scalar
  for (int x = 0; x < m; x++) {
    for (int y = n_last; y < n; y++) {
      data->at[y * m + x] = data->a[x * n + y];
    }
  }
}

#else

static void inner_loop_223(struct loop_223_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}

#endif

// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 256KiB equates to original problem size (M=256, N=231)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 256
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n) ((m) * (n) * sizeof(uint32_t))

LOOP_DECL(223, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 16
  uint64_t N = 0;  // any

  // For this loop, N should equal to M-1
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m - 1;
    if (PROBLEM_SIZE_ACTUAL(m, n) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
    } else {
      break;
    }
  }

  struct loop_223_data data = {
      .m = M,
      .n = N,
  };
  ALLOC_64B(data.a, M * N, "A matrix");
  ALLOC_64B(data.at, N * M, "At matrix");

  fill_uint32(data.a, M * N);

  inner_loops_223(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y) \
  { checksum += (data.a[(x) * N + (y)] - data.at[(y) * M + (x)] != 0); }

#if defined(FULL_CHECK)
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(223, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
