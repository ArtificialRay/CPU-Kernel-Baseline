/*----------------------------------------------------------------------------
#
#   Loop 211: INT16-INT32 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of i16 to i32 MOPA (or DOT) instructions.
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
    B: row-major
    C: row-major
  Constraints -
    M: multiple of SVLs
    N: multiple of SVLh
    K: even & greater than 3
*/

typedef struct {
  int16_t re, im;
} cint16_t;
typedef struct {
  int32_t re, im;
} cint32_t;

// Scalar multiply-accumulate for verification
static __attribute__((unused)) cint32_t cmla(cint32_t c, cint16_t a, cint16_t b) {
#define RE(v) ((uint32_t) v.re)
#define IM(v) ((uint32_t) v.im)
  c.re += RE(a) * RE(b) - IM(a) * IM(b);
  c.im += RE(a) * IM(b) + IM(a) * RE(b);
  return c;
}
#define CMLA(c,a,b) c = cmla(c,a,b)

struct loop_211_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  cint16_t *restrict a;
  cint16_t *restrict b;
  cint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_211(struct loop_211_data *restrict data) {
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

static void inner_loop_211(struct loop_211_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  cint16_t *restrict a = data->a;
  cint16_t *restrict b = data->b;
  cint32_t *restrict c = data->c;
  cint32_t zero = {0.0f, 0.0f};
  for (uint64_t y = 0; y < m; y++) {
    for (uint64_t x = 0; x < n; x++) {
      c[y * n + x] = zero;
    }
  }

  for (uint64_t z = 0; z < k; z++) {
    for (uint64_t y = 0; y < m; y++) {
      for (uint64_t x = 0; x < n; x++) {
        CMLA(c[y * n + x], a[z * m + y], b[z * n + x]);
      }
    }
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  cint16_t *a = data->a;
  cint16_t *b = data->b;
  cint32_t *c = data->c;

  cint32_t *ptr_c;
  cint16_t *ptr_a, *ptr_b;
  cint16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t l_cnd = svl_s * 4;
  uint64_t c_blk = svl_s * n;

  svcount_t c_all = svptrue_c16();
  svbool_t p_all = svptrue_b16();
  svbool_t p_neg = svtrn1_b16(svpfalse_b(), p_all);

  svint16_t vec_a0, vec_a1;
  svint16x2_t vec_b0, vec_b1;
  svuint8x4_t vec_c0, vec_c1;
  svint16_t vec_r0, vec_r1, vec_i0, vec_i1;
  svint32x2_t vec_o0, vec_o1, vec_o2, vec_o3;

#define MOPA_TILE(t, x, j, i) \
  svmopa_za32_m(t, p_all, p_all, vec_##j##x, svget2(vec_b##x, i))

#define EXTR(x, i) svreinterpret_s32(svget4(vec_c##x, i))
#define CAST(y, i) svget2(vec_o##y, i)
#define QUAD(i, j) svcreate4(CAST(i, 0), CAST(i, 1), CAST(j, 0), CAST(j, 1))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_s) {
    for (n_idx = 0; n_idx < n; n_idx += svl_s * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1(p_all, (int16_t *)&ptr_a[0]);
        vec_a1 = svld1(p_all, (int16_t *)&ptr_a[m]);
        vec_b0 = svld1_x2(c_all, (int16_t *)&ptr_b[0]);
        vec_b1 = svld1_x2(c_all, (int16_t *)&ptr_b[n]);

        vec_i0 = svreinterpret_s16(svrevh_x(p_all, svreinterpret_u32(vec_a0)));
        vec_i1 = svreinterpret_s16(svrevh_x(p_all, svreinterpret_u32(vec_a1)));
        vec_r0 = svneg_m(vec_a0, p_neg, vec_a0);
        vec_r1 = svneg_m(vec_a1, p_neg, vec_a1);

        MOPA_TILE(0, 0, r, 0);
        MOPA_TILE(1, 0, i, 0);
        MOPA_TILE(2, 0, r, 1);
        MOPA_TILE(3, 0, i, 1);
        MOPA_TILE(0, 1, r, 0);
        MOPA_TILE(1, 1, i, 0);
        MOPA_TILE(2, 1, r, 1);
        MOPA_TILE(3, 1, i, 1);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      for (l_idx = 0; l_idx < l_cnd; l_idx += 8) {
#if defined(__ARM_FEATURE_SME2p1)
        vec_c0 = svreadz_hor_za8_u8_vg4(0, l_idx + 0);
        vec_c1 = svreadz_hor_za8_u8_vg4(0, l_idx + 4);
#else
        vec_c0 = svread_hor_za8_u8_vg4(0, l_idx + 0);
        vec_c1 = svread_hor_za8_u8_vg4(0, l_idx + 4);
#endif
        vec_o0 = svzip(svcreate2(EXTR(0, 0), EXTR(0, 1)));
        vec_o1 = svzip(svcreate2(EXTR(0, 2), EXTR(0, 3)));
        vec_o2 = svzip(svcreate2(EXTR(1, 0), EXTR(1, 1)));
        vec_o3 = svzip(svcreate2(EXTR(1, 2), EXTR(1, 3)));

        svst1(c_all, (int32_t *)&ptr_c[0], QUAD(0, 1));
        svst1(c_all, (int32_t *)&ptr_c[n], QUAD(2, 3));

        ptr_c += n * 2;
      }
    }
    c += c_blk;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

#if defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  cint16_t *a = data->a;
  cint16_t *b = data->b;
  cint32_t *c = data->c;

  cint32_t *ptr_c;
  cint16_t *ptr_a, *ptr_b;
  cint16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  uint64_t svl_s = svcntw();
  svbool_t p_all = svptrue_b16();

  // acc_[lane][component][bottom/top]
  svint32_t acc_00r, acc_00i, acc_01r, acc_01i;
  svint32_t acc_10r, acc_10i, acc_11r, acc_11i;
  svint32_t acc_20r, acc_20i, acc_21r, acc_21i;
  svint32_t acc_30r, acc_30i, acc_31r, acc_31i;
  svint16x2_t ldb_0, ldb_1;
  svint16_t lda_0, lda_1;
  svcount_t c_all = svptrue_c16();
  svbool_t p_neg = svtrn1_b16(svpfalse_b(), svptrue_b16());

#define ZERO(l, i) acc_##l##i = svdup_s32(0);
#define ZERO_LANE(l) ZERO(l, 0r) ZERO(l, 0i) ZERO(l, 1r) ZERO(l, 1i)
#define GETC(lane, col, comp) acc_##lane##col##comp
#define GETB(unroll, col) svget2(ldb_##unroll, col)
#define LOADB_PAIR(q) svld1_x2(c_all, (int16_t *)&ptr_b[n * q])
#define STORE_PAIR(q, x) \
  svst2(p_all, (int32_t *)&ptr_c[n * q + x * svl_s], svcreate2(GETC(q, x, r), GETC(q, x, i)));
#define STORE_QUAD(q) \
  STORE_PAIR(q, 0) STORE_PAIR(q, 1)
#define REV(z) \
  svreinterpret_s16(svrevh_x(p_all, svreinterpret_u32(lda_##z)))
#define MAC_GROUP(p, x, z) \
  DOT_##p(x, 0, z) DOT_##p(x, 1, z) DOT_##p(x, 2, z) DOT_##p(x, 3, z)
#define DOT_R(x, y, z) \
  GETC(y, x, r) = svdot_lane(GETC(y, x, r), GETB(z, x), svneg_m(lda_##z, p_neg, lda_##z), y);
#define DOT_I(x, y, z) \
  GETC(y, x, i) = svdot_lane(GETC(y, x, i), GETB(z, x), REV(z), y);

  for (m_idx = 0; m_idx < m; m_idx += 4) {
    for (n_idx = 0; n_idx < n; n_idx += svcnth()) {
      ZERO_LANE(0);
      ZERO_LANE(1);
      ZERO_LANE(2);
      ZERO_LANE(3);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        lda_0 = svld1rq(p_all, (int16_t *)&ptr_a[0]);
        lda_1 = svld1rq(p_all, (int16_t *)&ptr_a[m]);
        ldb_0 = LOADB_PAIR(0);
        ldb_1 = LOADB_PAIR(1);

        MAC_GROUP(R, 0, 0);
        MAC_GROUP(I, 0, 0);
        MAC_GROUP(R, 1, 0);
        MAC_GROUP(I, 1, 0);

        MAC_GROUP(R, 0, 1);
        MAC_GROUP(I, 0, 1);
        MAC_GROUP(R, 1, 1);
        MAC_GROUP(I, 1, 1);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      STORE_QUAD(0);
      STORE_QUAD(1);
      STORE_QUAD(2);
      STORE_QUAD(3);
    }
    c += n * 4;
  }
}

#else

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  cint16_t *a = data->a;
  cint16_t *b = data->b;
  cint32_t *c = data->c;

  cint32_t *ptr_c;
  cint16_t *ptr_a, *ptr_b;
  cint16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  uint64_t svl_s = svcntw();
  svbool_t p_all = svptrue_b16();

  // acc_[lane][component][bottom/top]
  svint32_t acc_00r, acc_00i, acc_01r, acc_01i;
  svint32_t acc_10r, acc_10i, acc_11r, acc_11i;
  svint32_t acc_20r, acc_20i, acc_21r, acc_21i;
  svint32_t acc_30r, acc_30i, acc_31r, acc_31i;
  svint16x2_t ldb_0, ldb_1;
  svint16_t lda_0, lda_1;

#define ZERO(l, i) acc_##l##i = svdup_s32(0);
#define ZERO_LANE(l) ZERO(l, 0r) ZERO(l, 0i) ZERO(l, 1r) ZERO(l, 1i)
#define GETC(lane, col, comp) acc_##lane##col##comp
#define GETB(unroll, col) svget2(ldb_##unroll, col)
#define LOADB(q, p) svld1_vnum(p_all, (int16_t *)&ptr_b[n * q], p)
#define LOADB_PAIR(q) svcreate2(LOADB(q, 0), LOADB(q, 1))
#define STORE_PAIR(q, x) \
  svst2(p_all, (int32_t *)&ptr_c[n * q + x * svl_s], svcreate2(GETC(q, x, r), GETC(q, x, i)));
#define STORE_QUAD(q) \
  STORE_PAIR(q, 0) STORE_PAIR(q, 1)
#define MLA(o, w, x, y, z, l) \
  GETC(y, x, w) = svml##o##_lane(GETC(y, x, w), GETB(z, x), lda_##z, l);
#define MAC_GROUP(p, x, z) \
  MLA_##p(x, 0, z) MLA_##p(x, 1, z) MLA_##p(x, 2, z) MLA_##p(x, 3, z)
#define MLA_R1(x, y, z) \
  MLA(alb, r, x, y, z, 2 * y + 0)  // C[y].r += Az[y].r * Bz.r
#define MLA_R2(x, y, z) \
  MLA(slt, r, x, y, z, 2 * y + 1)  // C[y].r -= Az[y].i * Bz.i
#define MLA_I1(x, y, z) \
  MLA(alb, i, x, y, z, 2 * y + 1)  // C[y].i += Az[y].r * Bz.i
#define MLA_I2(x, y, z) \
  MLA(alt, i, x, y, z, 2 * y + 0)  // C[y].i += Az[y].i * Bz.r

  for (m_idx = 0; m_idx < m; m_idx += 4) {
    for (n_idx = 0; n_idx < n; n_idx += svcnth()) {
      ZERO_LANE(0);
      ZERO_LANE(1);
      ZERO_LANE(2);
      ZERO_LANE(3);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        lda_0 = svld1rq(p_all, (int16_t *)&ptr_a[0]);
        lda_1 = svld1rq(p_all, (int16_t *)&ptr_a[m]);
        ldb_0 = LOADB_PAIR(0);
        ldb_1 = LOADB_PAIR(1);

        MAC_GROUP(R1, 0, 0);
        MAC_GROUP(I1, 0, 0);
        MAC_GROUP(R2, 0, 0);
        MAC_GROUP(I2, 0, 0);
        MAC_GROUP(R1, 1, 0);
        MAC_GROUP(I1, 1, 0);
        MAC_GROUP(R2, 1, 0);
        MAC_GROUP(I2, 1, 0);

        MAC_GROUP(R1, 0, 1);
        MAC_GROUP(I1, 0, 1);
        MAC_GROUP(R2, 0, 1);
        MAC_GROUP(I2, 0, 1);
        MAC_GROUP(R1, 1, 1);
        MAC_GROUP(I1, 1, 1);
        MAC_GROUP(R2, 1, 1);
        MAC_GROUP(I2, 1, 1);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      STORE_QUAD(0);
      STORE_QUAD(1);
      STORE_QUAD(2);
      STORE_QUAD(3);
    }
    c += n * 4;
  }
}
#endif

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t c_blk = svl_s * n;
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t a_cnd = a + 4 * (m * k);
  register uint64_t m2off = m * 2;
  register uint64_t n2off = n * 2;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t m_idx;
  register uint64_t n_idx;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      // LS + component predicates
      "   ptrue   pn8.h                                               \n"
      "   ptrue   p0.h                                                \n"
      "   pfalse  p1.b                                                \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      "   trn1    p1.h, p1.h, p0.h                                    \n"

      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2                \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3                \n"

      // K loop
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      "   ld1h    {z0.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr]]                      \n"
      "   revh    z1.s, p0/m, z0.s                                    \n"
      "   smopa   za1.s, p0/m, p0/m, z1.h, z2.h                       \n"
      "   neg     z0.h, p1/m, z0.h                                    \n"
      "   smopa   za3.s, p0/m, p0/m, z1.h, z3.h                       \n"
      "   ld1h    {z4.h}, p0/z, [%[a_ptr], %[m2off], lsl #1]          \n"
      "   ld1h    {z6.h-z7.h}, pn8/z, [%[b_ptr], %[n2off], lsl #1]    \n"
      "   smopa   za0.s, p0/m, p0/m, z0.h, z2.h                       \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                    \n"
      "   smopa   za2.s, p0/m, p0/m, z0.h, z3.h                       \n"
      "3:                                                             \n"
      "   revh    z5.s, p0/m, z4.s                                    \n"
      "   smopa   za1.s, p0/m, p0/m, z5.h, z6.h                       \n"
      "   neg     z4.h, p1/m, z4.h                                    \n"
      "   smopa   za3.s, p0/m, p0/m, z5.h, z7.h                       \n"
      "   ld1h    {z0.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr]]                      \n"
      "   smopa   za0.s, p0/m, p0/m, z4.h, z6.h                       \n"
      "   smopa   za2.s, p0/m, p0/m, z4.h, z7.h                       \n"
      "   revh    z1.s, p0/m, z0.s                                    \n"
      "   smopa   za1.s, p0/m, p0/m, z1.h, z2.h                       \n"
      "   neg     z0.h, p1/m, z0.h                                    \n"
      "   smopa   za3.s, p0/m, p0/m, z1.h, z3.h                       \n"
      "   ld1h    {z4.h}, p0/z, [%[a_ptr], %[m2off], lsl #1]          \n"
      "   ld1h    {z6.h-z7.h}, pn8/z, [%[b_ptr], %[n2off], lsl #1]    \n"
      "   smopa   za0.s, p0/m, p0/m, z0.h, z2.h                       \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                    \n"
      "   smopa   za2.s, p0/m, p0/m, z0.h, z3.h                       \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"
      "   revh    z5.s, p0/m, z4.s                                    \n"
      "   smopa   za1.s, p0/m, p0/m, z5.h, z6.h                       \n"
      "   neg     z4.h, p1/m, z4.h                                    \n"
      "   smopa   za3.s, p0/m, p0/m, z5.h, z7.h                       \n"
      "   smopa   za0.s, p0/m, p0/m, z4.h, z6.h                       \n"
      "   smopa   za2.s, p0/m, p0/m, z4.h, z7.h                       \n"

      // Store loop
      "   mov     x12, #0                                             \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   mova    {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
#endif
      "   zip     {z0.s-z1.s}, z0.s, z1.s                             \n"
      "   zip     {z2.s-z3.s}, z2.s, z3.s                             \n"
      "   st1w    {z0.s-z3.s}, pn8, [%[c_ptr]]                        \n"
      "4:                                                             \n"
      "   add     x12, x12, #8                                        \n"
      "   zip     {z4.s-z5.s}, z4.s, z5.s                             \n"
      "   zip     {z6.s-z7.s}, z6.s, z7.s                             \n"
      "   st1w    {z4.s-z7.s}, pn8, [%[c_ptr], %[n2off], lsl #2]      \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                    \n"
      "   cmp     x12, %[l_cnd]                                  \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   mova    {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
#endif
      "   zip     {z0.s-z1.s}, z0.s, z1.s                             \n"
      "   zip     {z2.s-z3.s}, z2.s, z3.s                             \n"
      "   st1w    {z0.s-z3.s}, pn8, [%[c_ptr]]                        \n"
      "   b.mi    4b                                                  \n"
      "   zip     {z4.s-z5.s}, z4.s, z5.s                             \n"
      "   zip     {z6.s-z7.s}, z6.s, z7.s                             \n"
      "   st1w    {z4.s-z7.s}, pn8, [%[c_ptr], %[n2off], lsl #2]      \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                               \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_blk], lsl #3                \n"
      "   incw    %[m_idx]                                            \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_src] "r"(a), [b_src] "r"(b),
        [l_cnd] "r"(l_cnd), [m2off] "r"(m2off), [n2off] "r"(n2off),
        [a_cnd] "r"(a_cnd), [c_blk] "r"(c_blk)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p1", "p8", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t off_l = m * 2;
  register uint64_t off_1 = n * 2;
  register uint64_t off_2 = n * 4;
  register uint64_t off_3 = n * 6;

  register uint64_t a_cnd = a + 4 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.h                                                \n"
      "   ptrue   pn8.h                                               \n"

      "   pfalse  p1.b                                                \n"
      "   trn1    p1.h, p1.h, p0.h                                    \n"
      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      // Accumulators
      "   mov     z10.s, #0                                           \n"
      "   mov     z11.s, #0                                           \n"
      "   mov     z12.s, #0                                           \n"
      "   mov     z13.s, #0                                           \n"
      "   mov     z14.s, #0                                           \n"
      "   mov     z15.s, #0                                           \n"
      "   mov     z16.s, #0                                           \n"
      "   mov     z17.s, #0                                           \n"
      "   mov     z20.s, #0                                           \n"
      "   mov     z21.s, #0                                           \n"
      "   mov     z22.s, #0                                           \n"
      "   mov     z23.s, #0                                           \n"
      "   mov     z24.s, #0                                           \n"
      "   mov     z25.s, #0                                           \n"
      "   mov     z26.s, #0                                           \n"
      "   mov     z27.s, #0                                           \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2                \n"
      "3:                                                             \n"
      "   ld1rqh  {z0.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr]]                      \n"
      "   revh    z1.s, p0/m, z0.s                                    \n"
      "   neg    z0.h, p1/m, z0.h                                     \n"

      "   sdot    z10.s, z2.h, z0.h[0]                                \n"
      "   sdot    z12.s, z2.h, z0.h[1]                                \n"
      "   sdot    z14.s, z2.h, z0.h[2]                                \n"
      "   sdot    z16.s, z2.h, z0.h[3]                                \n"

      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr], %[off_l], lsl #1]          \n"
      "   sdot    z11.s, z2.h, z1.h[0]                                \n"
      "   sdot    z13.s, z2.h, z1.h[1]                                \n"
      "   sdot    z15.s, z2.h, z1.h[2]                                \n"
      "   sdot    z17.s, z2.h, z1.h[3]                                \n"

      "   ld1h    {z6.h-z7.h}, pn8/z, [%[b_ptr], %[off_1], lsl #1]    \n"
      "   revh    z5.s, p0/m, z4.s                                    \n"
      "   neg    z4.h, p1/m, z4.h                                     \n"

      "   sdot    z20.s, z3.h, z0.h[0]                                \n"
      "   sdot    z22.s, z3.h, z0.h[1]                                \n"
      "   sdot    z24.s, z3.h, z0.h[2]                                \n"
      "   sdot    z26.s, z3.h, z0.h[3]                                \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                    \n"
      "   sdot    z21.s, z3.h, z1.h[0]                                \n"
      "   sdot    z23.s, z3.h, z1.h[1]                                \n"
      "   sdot    z25.s, z3.h, z1.h[2]                                \n"
      "   sdot    z27.s, z3.h, z1.h[3]                                \n"

      "   sdot    z10.s, z6.h, z4.h[0]                                \n"
      "   sdot    z12.s, z6.h, z4.h[1]                                \n"
      "   sdot    z14.s, z6.h, z4.h[2]                                \n"
      "   sdot    z16.s, z6.h, z4.h[3]                                \n"

      "   sdot    z11.s, z6.h, z5.h[0]                                \n"
      "   sdot    z13.s, z6.h, z5.h[1]                                \n"
      "   sdot    z15.s, z6.h, z5.h[2]                                \n"
      "   sdot    z17.s, z6.h, z5.h[3]                                \n"

      "   sdot    z20.s, z7.h, z4.h[0]                                \n"
      "   sdot    z22.s, z7.h, z4.h[1]                                \n"
      "   sdot    z24.s, z7.h, z4.h[2]                                \n"
      "   sdot    z26.s, z7.h, z4.h[3]                                \n"

      "   sdot    z21.s, z7.h, z5.h[0]                                \n"
      "   sdot    z23.s, z7.h, z5.h[1]                                \n"
      "   sdot    z25.s, z7.h, z5.h[2]                                \n"
      "   sdot    z27.s, z7.h, z5.h[3]                                \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #3                \n"
      "   addvl   %[c1ptr], %[c0ptr], #2                              \n"
      "   st2w    {z10.s,z11.s}, p0, [%[c0ptr]]                       \n"
      "   st2w    {z20.s,z21.s}, p0, [%[c1ptr]]                       \n"
      "   st2w    {z12.s,z13.s}, p0, [%[c0ptr], %[off_1], lsl #2]     \n"
      "   st2w    {z22.s,z23.s}, p0, [%[c1ptr], %[off_1], lsl #2]     \n"
      "   st2w    {z14.s,z15.s}, p0, [%[c0ptr], %[off_2], lsl #2]     \n"
      "   st2w    {z24.s,z25.s}, p0, [%[c1ptr], %[off_2], lsl #2]     \n"
      "   st2w    {z16.s,z17.s}, p0, [%[c0ptr], %[off_3], lsl #2]     \n"
      "   st2w    {z26.s,z27.s}, p0, [%[c1ptr], %[off_3], lsl #2]     \n"

      // N loop tail
      "   inch    %[n_idx]                                            \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                    \n"
      "   add     %[m_idx], %[m_idx], #4                              \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c0ptr] "=&r"(c0ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c),
        [c1ptr] "=&r"(c1ptr)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_1] "r"(off_1), [off_2] "r"(off_2), [off_3] "r"(off_3),
        [off_l] "r"(off_l), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "p1", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_211(struct loop_211_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t off_l = m * 2;
  register uint64_t off_1 = n * 2;
  register uint64_t off_2 = n * 4;
  register uint64_t off_3 = n * 6;

  register uint64_t a_cnd = a + 4 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.h                                                \n"

      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      // Accumulators
      "   mov     z10.s, #0                                           \n"
      "   mov     z11.s, #0                                           \n"
      "   mov     z12.s, #0                                           \n"
      "   mov     z13.s, #0                                           \n"
      "   mov     z14.s, #0                                           \n"
      "   mov     z15.s, #0                                           \n"
      "   mov     z16.s, #0                                           \n"
      "   mov     z17.s, #0                                           \n"
      "   mov     z20.s, #0                                           \n"
      "   mov     z21.s, #0                                           \n"
      "   mov     z22.s, #0                                           \n"
      "   mov     z23.s, #0                                           \n"
      "   mov     z24.s, #0                                           \n"
      "   mov     z25.s, #0                                           \n"
      "   mov     z26.s, #0                                           \n"
      "   mov     z27.s, #0                                           \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2                \n"
      "3:                                                             \n"
      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], %[off_l], lsl #1]          \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                    \n"
      "   ld1h    {z0.h}, p0/z, [%[b_ptr]]                            \n"
      "   ld1h    {z1.h}, p0/z, [%[b_ptr], #1, mul vl]                \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                    \n"
      "   ld1h    {z2.h}, p0/z, [%[b_ptr]]                            \n"
      "   ld1h    {z3.h}, p0/z, [%[b_ptr], #1, mul vl]                \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                    \n"
      "   smlalb  z10.s, z0.h, z4.h[0]                                \n"
      "   smlalb  z14.s, z0.h, z4.h[2]                                \n"
      "   smlalb  z20.s, z0.h, z4.h[4]                                \n"
      "   smlalb  z24.s, z0.h, z4.h[6]                                \n"
      "   smlalb  z11.s, z0.h, z4.h[1]                                \n"
      "   smlalb  z15.s, z0.h, z4.h[3]                                \n"
      "   smlalb  z21.s, z0.h, z4.h[5]                                \n"
      "   smlalb  z25.s, z0.h, z4.h[7]                                \n"
      "   smlslt  z10.s, z0.h, z4.h[1]                                \n"
      "   smlslt  z14.s, z0.h, z4.h[3]                                \n"
      "   smlslt  z20.s, z0.h, z4.h[5]                                \n"
      "   smlslt  z24.s, z0.h, z4.h[7]                                \n"
      "   smlalt  z11.s, z0.h, z4.h[0]                                \n"
      "   smlalt  z15.s, z0.h, z4.h[2]                                \n"
      "   smlalt  z21.s, z0.h, z4.h[4]                                \n"
      "   smlalt  z25.s, z0.h, z4.h[6]                                \n"
      "   smlalb  z12.s, z1.h, z4.h[0]                                \n"
      "   smlalb  z16.s, z1.h, z4.h[2]                                \n"
      "   smlalb  z22.s, z1.h, z4.h[4]                                \n"
      "   smlalb  z26.s, z1.h, z4.h[6]                                \n"
      "   smlalb  z13.s, z1.h, z4.h[1]                                \n"
      "   smlalb  z17.s, z1.h, z4.h[3]                                \n"
      "   smlalb  z23.s, z1.h, z4.h[5]                                \n"
      "   smlalb  z27.s, z1.h, z4.h[7]                                \n"
      "   smlslt  z12.s, z1.h, z4.h[1]                                \n"
      "   smlslt  z16.s, z1.h, z4.h[3]                                \n"
      "   smlslt  z22.s, z1.h, z4.h[5]                                \n"
      "   smlslt  z26.s, z1.h, z4.h[7]                                \n"
      "   smlalt  z13.s, z1.h, z4.h[0]                                \n"
      "   smlalt  z17.s, z1.h, z4.h[2]                                \n"
      "   smlalt  z23.s, z1.h, z4.h[4]                                \n"
      "   smlalt  z27.s, z1.h, z4.h[6]                                \n"
      "   smlalb  z10.s, z2.h, z5.h[0]                                \n"
      "   smlalb  z14.s, z2.h, z5.h[2]                                \n"
      "   smlalb  z20.s, z2.h, z5.h[4]                                \n"
      "   smlalb  z24.s, z2.h, z5.h[6]                                \n"
      "   smlalb  z11.s, z2.h, z5.h[1]                                \n"
      "   smlalb  z15.s, z2.h, z5.h[3]                                \n"
      "   smlalb  z21.s, z2.h, z5.h[5]                                \n"
      "   smlalb  z25.s, z2.h, z5.h[7]                                \n"
      "   smlslt  z10.s, z2.h, z5.h[1]                                \n"
      "   smlslt  z14.s, z2.h, z5.h[3]                                \n"
      "   smlslt  z20.s, z2.h, z5.h[5]                                \n"
      "   smlslt  z24.s, z2.h, z5.h[7]                                \n"
      "   smlalt  z11.s, z2.h, z5.h[0]                                \n"
      "   smlalt  z15.s, z2.h, z5.h[2]                                \n"
      "   smlalt  z21.s, z2.h, z5.h[4]                                \n"
      "   smlalt  z25.s, z2.h, z5.h[6]                                \n"
      "   smlalb  z12.s, z3.h, z5.h[0]                                \n"
      "   smlalb  z16.s, z3.h, z5.h[2]                                \n"
      "   smlalb  z22.s, z3.h, z5.h[4]                                \n"
      "   smlalb  z26.s, z3.h, z5.h[6]                                \n"
      "   smlalb  z13.s, z3.h, z5.h[1]                                \n"
      "   smlalb  z17.s, z3.h, z5.h[3]                                \n"
      "   smlalb  z23.s, z3.h, z5.h[5]                                \n"
      "   smlalb  z27.s, z3.h, z5.h[7]                                \n"
      "   smlslt  z12.s, z3.h, z5.h[1]                                \n"
      "   smlslt  z16.s, z3.h, z5.h[3]                                \n"
      "   smlslt  z22.s, z3.h, z5.h[5]                                \n"
      "   smlslt  z26.s, z3.h, z5.h[7]                                \n"
      "   smlalt  z13.s, z3.h, z5.h[0]                                \n"
      "   smlalt  z17.s, z3.h, z5.h[2]                                \n"
      "   smlalt  z23.s, z3.h, z5.h[4]                                \n"
      "   smlalt  z27.s, z3.h, z5.h[6]                                \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #3                \n"
      "   addvl   %[c1ptr], %[c0ptr], #2                              \n"
      "   st2w    {z10.s,z11.s}, p0, [%[c0ptr]]                       \n"
      "   st2w    {z12.s,z13.s}, p0, [%[c1ptr]]                       \n"
      "   st2w    {z14.s,z15.s}, p0, [%[c0ptr], %[off_1], lsl #2]     \n"
      "   st2w    {z16.s,z17.s}, p0, [%[c1ptr], %[off_1], lsl #2]     \n"
      "   st2w    {z20.s,z21.s}, p0, [%[c0ptr], %[off_2], lsl #2]     \n"
      "   st2w    {z22.s,z23.s}, p0, [%[c1ptr], %[off_2], lsl #2]     \n"
      "   st2w    {z24.s,z25.s}, p0, [%[c0ptr], %[off_3], lsl #2]     \n"
      "   st2w    {z26.s,z27.s}, p0, [%[c1ptr], %[off_3], lsl #2]     \n"

      // N loop tail
      "   inch    %[n_idx]                                            \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                    \n"
      "   add     %[m_idx], %[m_idx], #4                              \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_1] "r"(off_1), [off_2] "r"(off_2), [off_3] "r"(off_3),
        [off_l] "r"(off_l), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z10", "z11", "z12", "z13", "z14",
        "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "p0", "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_211(struct loop_211_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_inc = m * 4;
  register uint64_t n_inc = n * 4;
  register uint64_t o_inc = n * 8 - 32;

  register uint64_t a_cnd = a + (m_inc * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                \n"
      "1:                                                     \n"

      // N loop head
      "   mov     %[n_idx], #0                                \n"
      "2:                                                     \n"

      // Accumulators
      "   movi    v10.4s, #0                                  \n"
      "   movi    v11.4s, #0                                  \n"
      "   movi    v12.4s, #0                                  \n"
      "   movi    v13.4s, #0                                  \n"
      "   movi    v14.4s, #0                                  \n"
      "   movi    v15.4s, #0                                  \n"
      "   movi    v16.4s, #0                                  \n"
      "   movi    v17.4s, #0                                  \n"
      "   movi    v20.4s, #0                                  \n"
      "   movi    v21.4s, #0                                  \n"
      "   movi    v22.4s, #0                                  \n"
      "   movi    v23.4s, #0                                  \n"
      "   movi    v24.4s, #0                                  \n"
      "   movi    v25.4s, #0                                  \n"
      "   movi    v26.4s, #0                                  \n"
      "   movi    v27.4s, #0                                  \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2        \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2        \n"
      "3:                                                     \n"
      "   ld1     {v4.8h}, [%[a_ptr]], %[m_inc]               \n"
      "   ld1     {v5.8h}, [%[a_ptr]], %[m_inc]               \n"
      "   ld2     {v0.8h,v1.8h}, [%[b_ptr]], %[n_inc]         \n"
      "   ld2     {v2.8h,v3.8h}, [%[b_ptr]], %[n_inc]         \n"
      "   smlal   v10.4s, v0.4h, v4.h[0]                      \n"
      "   smlal   v14.4s, v0.4h, v4.h[2]                      \n"
      "   smlal   v20.4s, v0.4h, v4.h[4]                      \n"
      "   smlal   v24.4s, v0.4h, v4.h[6]                      \n"
      "   smlal   v11.4s, v1.4h, v4.h[0]                      \n"
      "   smlal   v15.4s, v1.4h, v4.h[2]                      \n"
      "   smlal   v21.4s, v1.4h, v4.h[4]                      \n"
      "   smlal   v25.4s, v1.4h, v4.h[6]                      \n"
      "   smlsl   v10.4s, v1.4h, v4.h[1]                      \n"
      "   smlsl   v14.4s, v1.4h, v4.h[3]                      \n"
      "   smlsl   v20.4s, v1.4h, v4.h[5]                      \n"
      "   smlsl   v24.4s, v1.4h, v4.h[7]                      \n"
      "   smlal   v11.4s, v0.4h, v4.h[1]                      \n"
      "   smlal   v15.4s, v0.4h, v4.h[3]                      \n"
      "   smlal   v21.4s, v0.4h, v4.h[5]                      \n"
      "   smlal   v25.4s, v0.4h, v4.h[7]                      \n"
      "   smlal2  v12.4s, v0.8h, v4.h[0]                      \n"
      "   smlal2  v16.4s, v0.8h, v4.h[2]                      \n"
      "   smlal2  v22.4s, v0.8h, v4.h[4]                      \n"
      "   smlal2  v26.4s, v0.8h, v4.h[6]                      \n"
      "   smlal2  v13.4s, v1.8h, v4.h[0]                      \n"
      "   smlal2  v17.4s, v1.8h, v4.h[2]                      \n"
      "   smlal2  v23.4s, v1.8h, v4.h[4]                      \n"
      "   smlal2  v27.4s, v1.8h, v4.h[6]                      \n"
      "   smlsl2  v12.4s, v1.8h, v4.h[1]                      \n"
      "   smlsl2  v16.4s, v1.8h, v4.h[3]                      \n"
      "   smlsl2  v22.4s, v1.8h, v4.h[5]                      \n"
      "   smlsl2  v26.4s, v1.8h, v4.h[7]                      \n"
      "   smlal2  v13.4s, v0.8h, v4.h[1]                      \n"
      "   smlal2  v17.4s, v0.8h, v4.h[3]                      \n"
      "   smlal2  v23.4s, v0.8h, v4.h[5]                      \n"
      "   smlal2  v27.4s, v0.8h, v4.h[7]                      \n"
      "   smlal   v10.4s, v2.4h, v5.h[0]                      \n"
      "   smlal   v14.4s, v2.4h, v5.h[2]                      \n"
      "   smlal   v20.4s, v2.4h, v5.h[4]                      \n"
      "   smlal   v24.4s, v2.4h, v5.h[6]                      \n"
      "   smlal   v11.4s, v3.4h, v5.h[0]                      \n"
      "   smlal   v15.4s, v3.4h, v5.h[2]                      \n"
      "   smlal   v21.4s, v3.4h, v5.h[4]                      \n"
      "   smlal   v25.4s, v3.4h, v5.h[6]                      \n"
      "   smlsl   v10.4s, v3.4h, v5.h[1]                      \n"
      "   smlsl   v14.4s, v3.4h, v5.h[3]                      \n"
      "   smlsl   v20.4s, v3.4h, v5.h[5]                      \n"
      "   smlsl   v24.4s, v3.4h, v5.h[7]                      \n"
      "   smlal   v11.4s, v2.4h, v5.h[1]                      \n"
      "   smlal   v15.4s, v2.4h, v5.h[3]                      \n"
      "   smlal   v21.4s, v2.4h, v5.h[5]                      \n"
      "   smlal   v25.4s, v2.4h, v5.h[7]                      \n"
      "   smlal2  v12.4s, v2.8h, v5.h[0]                      \n"
      "   smlal2  v16.4s, v2.8h, v5.h[2]                      \n"
      "   smlal2  v22.4s, v2.8h, v5.h[4]                      \n"
      "   smlal2  v26.4s, v2.8h, v5.h[6]                      \n"
      "   smlal2  v13.4s, v3.8h, v5.h[0]                      \n"
      "   smlal2  v17.4s, v3.8h, v5.h[2]                      \n"
      "   smlal2  v23.4s, v3.8h, v5.h[4]                      \n"
      "   smlal2  v27.4s, v3.8h, v5.h[6]                      \n"
      "   smlsl2  v12.4s, v3.8h, v5.h[1]                      \n"
      "   smlsl2  v16.4s, v3.8h, v5.h[3]                      \n"
      "   smlsl2  v22.4s, v3.8h, v5.h[5]                      \n"
      "   smlsl2  v26.4s, v3.8h, v5.h[7]                      \n"
      "   smlal2  v13.4s, v2.8h, v5.h[1]                      \n"
      "   smlal2  v17.4s, v2.8h, v5.h[3]                      \n"
      "   smlal2  v23.4s, v2.8h, v5.h[5]                      \n"
      "   smlal2  v27.4s, v2.8h, v5.h[7]                      \n"
      "   cmp     %[a_ptr], %[a_cnd]                          \n"
      "   b.mi    3b                                          \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3        \n"
      "   st2     {v10.4s,v11.4s}, [%[c_ptr]], #32            \n"
      "   st2     {v12.4s,v13.4s}, [%[c_ptr]], %[o_inc]       \n"
      "   st2     {v14.4s,v15.4s}, [%[c_ptr]], #32            \n"
      "   st2     {v16.4s,v17.4s}, [%[c_ptr]], %[o_inc]       \n"
      "   st2     {v20.4s,v21.4s}, [%[c_ptr]], #32            \n"
      "   st2     {v22.4s,v23.4s}, [%[c_ptr]], %[o_inc]       \n"
      "   st2     {v24.4s,v25.4s}, [%[c_ptr]], #32            \n"
      "   st2     {v26.4s,v27.4s}, [%[c_ptr]], %[o_inc]       \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                      \n"
      "   cmp     %[n_idx], %[n]                              \n"
      "   b.mi    2b                                          \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5            \n"
      "   add     %[m_idx], %[m_idx], #4                      \n"
      "   cmp     %[m_idx], %[m]                              \n"
      "   b.mi    1b                                          \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [m_inc] "r"(m_inc), [n_inc] "r"(n_inc), [o_inc] "r"(o_inc),
        [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "cc", "memory");
}

#else

static void inner_loop_211(struct loop_211_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif


// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 96KiB equates to original problem size (M=64, K=128, N=128)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 96
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(cint16_t))

LOOP_DECL(211, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLs
  uint64_t N = 0;  // multiple of SVLh
  uint64_t K = 0;  // even

  // For this loop, M should be equal to N/2, K should be equal to N
  const uint64_t M_base = MAX_VL / 32;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m * 2;   // Automatically a multiple of SVLh
    uint64_t k = m * 2;   // Automatically even
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_211_data data = {
      .m = M,
      .n = N,
      .k = K,
  };

  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_int16((int16_t *)data.a, 2 * M * K);
  fill_int16((int16_t *)data.b, 2 * K * N);

  inner_loops_211(iters, &data);

  int checksum = 0;
#define CHECK(x, y)                                      \
  {                                                      \
    cint32_t v = data.c[(x)*N + (y)];                    \
    cint32_t d = {0.0f, 0.0f};                           \
    for (int k = 0; k < K; k++)                          \
      CMLA(d, data.a[k * M + (x)], data.b[k * N + (y)]); \
    checksum += (int)((d.re != v.re) || (d.im != v.im)); \
  }
#ifdef FULL_CHECK
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, 0);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(211, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
