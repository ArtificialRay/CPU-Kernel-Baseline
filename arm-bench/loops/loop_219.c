/*----------------------------------------------------------------------------
#
#   Loop 219: INT8-INT32 col-major matrix-vector multiply
#
#   Purpose:
#     Use of i8 to i32 VDOT instruction.
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
    B: column-vector
    C: column-vector
  Constraints -
    M: multiple of 4*SVLb
    N: multiple of 16
*/

struct loop_219_data {
  uint64_t m;
  uint64_t n;
  uint8_t *restrict a;
  uint8_t *restrict b;
  uint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_219(struct loop_219_data *restrict data) {
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



// Scalar widening multiply-add
#define MLA(w, u, v) (w) += (uint32_t)(u) * (uint32_t)(v)

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_219(struct loop_219_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  uint32_t *restrict c = data->c;
  for (uint64_t y = 0; y < m; y++) {
    uint32_t d = 0;
    for (uint64_t x = 0; x < n; x++) MLA(d, a[x*m+y], b[x]);
    c[y] = d;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_219(struct loop_219_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint64_t svl_b = svcntb();
  uint64_t m_blk = (svl_b * svl_b) / 4;
  uint64_t m_idx, n_idx, l_idx;
  uint64_t o_idx, o_cnd;

  uint8_t *a_bar, *a_ptr;
  uint32_t *c_cnd, *c_ptr;

  svcount_t c_all = svptrue_c8();
  svbool_t p_all = svptrue_b8();

  svuint8_t ldb;
  svuint8x4_t lda_0, lda_1, lda_2, lda_3;
  svuint32x4_t stc_0, stc_1, stc_2, stc_3;

#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define LOAD(q) lda_##q = svld1_x4(c_all, &a_ptr[m * q]);
#define LOAD_GROUP LOAD(0) LOAD(1) LOAD(2) LOAD(3)
#define EXTR(q, p) svget4(lda_##q, p)

#define VDOT_QUAD(q) svcreate4(EXTR(0, q), EXTR(1, q), EXTR(2, q), EXTR(3, q))
#define VDOT(l, q) svvdot_lane_za32_vg1x4(l_idx + q, VDOT_QUAD(q), ldb, l);
#define VDOT_GROUP(l) VDOT(l, 0) VDOT(l, 1) VDOT(l, 2) VDOT(l, 3)

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += m_blk) {
    o_cnd = MIN(m_idx + m_blk, m);
#if !defined(__ARM_FEATURE_SME2p1)
    svzero_za();
#endif
    n_idx = 0;
    a_bar = a;
    while (n_idx < n) {
      ldb = svld1rq(p_all, &b[n_idx]);

      l_idx = 0;
      o_idx = m_idx;
      while (o_idx < o_cnd) {
        a_ptr = &a_bar[o_idx];
        LOAD_GROUP;
        VDOT_GROUP(0);
        a_ptr += 4 * m;
        LOAD_GROUP;
        VDOT_GROUP(1);
        a_ptr += 4 * m;
        LOAD_GROUP;
        VDOT_GROUP(2);
        a_ptr += 4 * m;
        LOAD_GROUP;
        VDOT_GROUP(3);
        l_idx += 4;
        o_idx += 4 * svl_b;
      }

      n_idx += 16;
      a_bar += 16 * m;
    }

    l_idx = 0;
    c_ptr = &c[m_idx];
    c_cnd = &c[o_cnd];
    while (c_ptr < c_cnd) {
#if defined(__ARM_FEATURE_SME2p1)
      stc_0 = svreadz_za32_u32_vg1x4(l_idx + 0);
      stc_1 = svreadz_za32_u32_vg1x4(l_idx + 1);
      stc_2 = svreadz_za32_u32_vg1x4(l_idx + 2);
      stc_3 = svreadz_za32_u32_vg1x4(l_idx + 3);
#else
      stc_0 = svread_za32_u32_vg1x4(l_idx + 0);
      stc_1 = svread_za32_u32_vg1x4(l_idx + 1);
      stc_2 = svread_za32_u32_vg1x4(l_idx + 2);
      stc_3 = svread_za32_u32_vg1x4(l_idx + 3);
#endif
      svst4_vnum(p_all, c_ptr, 0x0, stc_0);
      svst4_vnum(p_all, c_ptr, 0x4, stc_1);
      svst4_vnum(p_all, c_ptr, 0x8, stc_2);
      svst4_vnum(p_all, c_ptr, 0xc, stc_3);
      l_idx += 4;
      c_ptr += 4 * svl_b;
    }
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static svuint8x4_t svzip4(svuint8_t a, svuint8_t b, svuint8_t c, svuint8_t d)
LOOP_ATTR
{
  svuint8_t e = svzip1(a, c);
  svuint8_t f = svzip2(a, c);
  svuint8_t g = svzip1(b, d);
  svuint8_t h = svzip2(b, d);
  return svcreate4(svzip1(e, g), svzip2(e, g), svzip1(f, h), svzip2(f, h));
}

static void inner_loop_219(struct loop_219_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint8_t *a_ptr;
  uint32_t *c_ptr;

  uint64_t m_idx, n_idx;
  svbool_t p_ldx = svptrue_pat_b8(SV_VL8);

  svuint32x4_t acc_0, acc_1;
  svuint8x4_t zip_0, zip_1;
  svuint8x2_t lda_0, lda_1, lda_2, lda_3;
  svuint8_t ldx;

#define ZERO svdup_u32(0)
#define ZERO_QUAD(q) acc_##q = svcreate4(ZERO, ZERO, ZERO, ZERO)

#define GETA(q, p) svget2(lda_##q, p)
#define GETB(q, p) svget4(acc_##q, p)

#define ZIP4_QUAD(i) zip_##i = \
  svzip4(GETA(0, i), GETA(1, i), GETA(2, i), GETA(3, i))

#define UDOT(l, q, p) svdot_lane(GETB(q, p), svget4(zip_##q, p), ldx, l)
#define UDOT_LANE(l, q) acc_##q = \
  svcreate4(UDOT(l, q, 0), UDOT(l, q, 1), UDOT(l, q, 2), UDOT(l, q, 3))

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c8();
#define LOAD_PAIR(i) lda_##i = svld1_x2(c_all, &a_ptr[m * i])
#define STORE_QUAD(q) svst1_vnum(c_all, c_ptr, q * 4, acc_##q)
#else
  svbool_t p_all = svptrue_b8();
#define LOAD(i, p) svld1_vnum(p_all, &a_ptr[m * i], p)
#define LOAD_PAIR(i) lda_##i = svcreate2(LOAD(i, 0), LOAD(i, 1))
#define STORE(q, p) svst1_vnum(p_all, c_ptr, q * 4 + p, GETB(q, p));
#define STORE_QUAD(q) STORE(q, 0) STORE(q, 1) STORE(q, 2) STORE(q, 3)
#endif

  for (m_idx = 0; m_idx < m; m_idx += svcntb() * 2) {
    ZERO_QUAD(0);
    ZERO_QUAD(1);

    a_ptr = &a[m_idx];
    for (n_idx = 0; n_idx < n; n_idx += 8) {
      ldx = svld1rq(p_ldx, &b[n_idx]);
      LOAD_PAIR(0);
      LOAD_PAIR(1);
      LOAD_PAIR(2);
      LOAD_PAIR(3);
      ZIP4_QUAD(0);
      ZIP4_QUAD(1);
      UDOT_LANE(0, 0);
      UDOT_LANE(0, 1);
      a_ptr += 4 * m;
      LOAD_PAIR(0);
      LOAD_PAIR(1);
      LOAD_PAIR(2);
      LOAD_PAIR(3);
      ZIP4_QUAD(0);
      ZIP4_QUAD(1);
      UDOT_LANE(1, 0);
      UDOT_LANE(1, 1);
      a_ptr += 4 * m;
    }

    c_ptr = &c[m_idx];
    STORE_QUAD(0);
    STORE_QUAD(1);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_219(struct loop_219_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_h;
  asm volatile("cnth %[v]" : [v] "=&r"(svl_h)::);
  register uint64_t m_blk = svl_h * svl_h;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t o_idx;
  register uint64_t o_cnd;
  register uint64_t a_bar;
  register uint64_t a_ptr;
  register uint64_t c_ptr;
  register uint64_t c_cnd;

  register uint64_t a2off = m * 2;
  register uint64_t a3off = m * 3;
  // x9: slice index register for uvdot and mova

  asm volatile(
      "   ptrue   pn8.b                                                   \n"
      "   ptrue   p0.b                                                    \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                    \n"
#endif

      // M loop head
      "   mov     %[m_idx], #0                                            \n"
      "1:                                                                 \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                    \n"
#endif
      "   add     %[o_cnd], %[m_idx], %[m_blk]                            \n"
      "   cmp     %[o_cnd], %[m]                                          \n"
      "   csel    %[o_cnd], %[o_cnd], %[m], lt                            \n"

      // N loop head
      "   mov     %[n_idx], #0                                            \n"
      "   mov     %[a_bar], %[a_src]                                      \n"
      "2:                                                                 \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]                      \n"

      // M-block dot-product loop
      "   mov     x9, #0                                                  \n"
      "   mov     %[o_idx], %[m_idx]                                      \n"
      "3:                                                                 \n"
      "   add     %[a_ptr], %[a_bar], %[o_idx]                            \n"
      "   ld1b    {z16.b,z20.b,z24.b,z28.b}, pn8/z, [%[a_ptr]]            \n"
      "   ld1b    {z17.b,z21.b,z25.b,z29.b}, pn8/z, [%[a_ptr], %[a1off]]  \n"
      "   ld1b    {z18.b,z22.b,z26.b,z30.b}, pn8/z, [%[a_ptr], %[a2off]]  \n"
      "   ld1b    {z19.b,z23.b,z27.b,z31.b}, pn8/z, [%[a_ptr], %[a3off]]  \n"
      "   uvdot   za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[0]               \n"
      "   uvdot   za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[0]               \n"
      "   uvdot   za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[0]               \n"
      "   uvdot   za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[0]               \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                        \n"
      "   ld1b    {z16.b,z20.b,z24.b,z28.b}, pn8/z, [%[a_ptr]]            \n"
      "   ld1b    {z17.b,z21.b,z25.b,z29.b}, pn8/z, [%[a_ptr], %[a1off]]  \n"
      "   ld1b    {z18.b,z22.b,z26.b,z30.b}, pn8/z, [%[a_ptr], %[a2off]]  \n"
      "   ld1b    {z19.b,z23.b,z27.b,z31.b}, pn8/z, [%[a_ptr], %[a3off]]  \n"
      "   uvdot   za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[1]               \n"
      "   uvdot   za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[1]               \n"
      "   uvdot   za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[1]               \n"
      "   uvdot   za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[1]               \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                        \n"
      "   ld1b    {z16.b,z20.b,z24.b,z28.b}, pn8/z, [%[a_ptr]]            \n"
      "   ld1b    {z17.b,z21.b,z25.b,z29.b}, pn8/z, [%[a_ptr], %[a1off]]  \n"
      "   ld1b    {z18.b,z22.b,z26.b,z30.b}, pn8/z, [%[a_ptr], %[a2off]]  \n"
      "   ld1b    {z19.b,z23.b,z27.b,z31.b}, pn8/z, [%[a_ptr], %[a3off]]  \n"
      "   uvdot   za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[2]               \n"
      "   uvdot   za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[2]               \n"
      "   uvdot   za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[2]               \n"
      "   uvdot   za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[2]               \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                        \n"
      "   ld1b    {z16.b,z20.b,z24.b,z28.b}, pn8/z, [%[a_ptr]]            \n"
      "   ld1b    {z17.b,z21.b,z25.b,z29.b}, pn8/z, [%[a_ptr], %[a1off]]  \n"
      "   ld1b    {z18.b,z22.b,z26.b,z30.b}, pn8/z, [%[a_ptr], %[a2off]]  \n"
      "   ld1b    {z19.b,z23.b,z27.b,z31.b}, pn8/z, [%[a_ptr], %[a3off]]  \n"
      "   uvdot   za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[3]               \n"
      "   uvdot   za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[3]               \n"
      "   uvdot   za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[3]               \n"
      "   uvdot   za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[3]               \n"
      "   add     x9, x9, #4                                              \n"
      "   incb    %[o_idx], all, mul #4                                   \n"
      "   cmp     %[o_idx], %[o_cnd]                                      \n"
      "   b.mi    3b                                                      \n"

      // N loop tail
      "   add     %[a_bar], %[a_bar], %[m], lsl #4                        \n"
      "   add     %[n_idx], %[n_idx], #16                                 \n"
      "   cmp     %[n_idx], %[n]                                          \n"
      "   b.mi    2b                                                      \n"

      // Store loop
      "   mov     x9, #0                                                  \n"
      "   add     %[c_ptr], %[c_dst], %[m_idx], lsl #2                    \n"
      "   add     %[c_cnd], %[c_dst], %[o_cnd], lsl #2                    \n"
      "4:                                                                 \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z19.s}, za.s[w9, 0, vgx4]                        \n"
      "   movaz   {z20.s-z23.s}, za.s[w9, 1, vgx4]                        \n"
      "   movaz   {z24.s-z27.s}, za.s[w9, 2, vgx4]                        \n"
      "   movaz   {z28.s-z31.s}, za.s[w9, 3, vgx4]                        \n"
#else
      "   mova    {z16.s-z19.s}, za.s[w9, 0, vgx4]                        \n"
      "   mova    {z20.s-z23.s}, za.s[w9, 1, vgx4]                        \n"
      "   mova    {z24.s-z27.s}, za.s[w9, 2, vgx4]                        \n"
      "   mova    {z28.s-z31.s}, za.s[w9, 3, vgx4]                        \n"
#endif
      "   add     x9, x9, #4                                              \n"
      "   st4w    {z16.s-z19.s}, p0, [%[c_ptr]]                           \n"
      "   st4w    {z20.s-z23.s}, p0, [%[c_ptr], #0x4, mul vl]             \n"
      "   st4w    {z24.s-z27.s}, p0, [%[c_ptr], #0x8, mul vl]             \n"
      "   st4w    {z28.s-z31.s}, p0, [%[c_ptr], #0xc, mul vl]             \n"
      "   addvl   %[c_ptr], %[c_ptr], #16                                 \n"
      "   cmp     %[c_ptr], %[c_cnd]                                      \n"
      "   b.mi    4b                                                      \n"

      // M loop tail
      "   add     %[m_idx], %[m_idx], %[m_blk]                            \n"
      "   cmp     %[m_idx], %[m]                                          \n"
      "   b.mi    1b                                                      \n"

      : [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [a_bar] "=&r"(a_bar),
        [a_ptr] "=&r"(a_ptr), [c_ptr] "=&r"(c_ptr), [o_idx] "=&r"(o_idx),
        [o_cnd] "=&r"(o_cnd), [c_cnd] "=&r"(c_cnd), [c_dst] "+&r"(c)
      : [a2off] "r"(a2off), [a3off] "r"(a3off), [m_blk] "r"(m_blk),
        [a1off] "r"(m), [a_src] "r"(a), [b_src] "r"(b), [m] "r"(m), [n] "r"(n)
      : "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
        "z26", "z27", "z28", "z29", "z30", "z31", "z0", "p0", "p8", "x9",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_219(struct loop_219_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t a2off = m * 2;
  register uint64_t a3off = m * 3;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;

  asm volatile(
      "   ptrue   p0.b, VL8                                 \n"
      "   ptrue   pn8.b                                     \n"

      // M loop head
      "   mov     %[m_idx], #0                              \n"
      "1:                                                   \n"
      "   mov     z4.s , #0                                 \n"
      "   mov     z5.s , #0                                 \n"
      "   mov     z6.s , #0                                 \n"
      "   mov     z7.s , #0                                 \n"
      "   mov     z12.s, #0                                 \n"
      "   mov     z13.s, #0                                 \n"
      "   mov     z14.s, #0                                 \n"
      "   mov     z15.s, #0                                 \n"

      // N loop
      "   mov     %[n_idx], #0                              \n"
      "   add     %[a_ptr], %[a_src], %[m_idx]              \n"
      "2:                                                   \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]        \n"
      "   ld1b    {z16.b-z17.b}, pn8/z, [%[a_ptr]]          \n"
      "   ld1b    {z18.b-z19.b}, pn8/z, [%[a_ptr], %[a1off]]\n"
      "   ld1b    {z20.b-z21.b}, pn8/z, [%[a_ptr], %[a2off]]\n"
      "   ld1b    {z22.b-z23.b}, pn8/z, [%[a_ptr], %[a3off]]\n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   zip1    z24.b, z16.b, z20.b                       \n"
      "   zip2    z25.b, z16.b, z20.b                       \n"
      "   zip1    z26.b, z18.b, z22.b                       \n"
      "   zip2    z27.b, z18.b, z22.b                       \n"
      "   zip1    z28.b, z17.b, z21.b                       \n"
      "   zip2    z29.b, z17.b, z21.b                       \n"
      "   zip1    z30.b, z19.b, z23.b                       \n"
      "   zip2    z31.b, z19.b, z23.b                       \n"
      "   zip1    z16.b, z24.b, z26.b                       \n"
      "   zip2    z17.b, z24.b, z26.b                       \n"
      "   zip1    z18.b, z25.b, z27.b                       \n"
      "   zip2    z19.b, z25.b, z27.b                       \n"
      "   zip1    z20.b, z28.b, z30.b                       \n"
      "   zip2    z21.b, z28.b, z30.b                       \n"
      "   zip1    z22.b, z29.b, z31.b                       \n"
      "   zip2    z23.b, z29.b, z31.b                       \n"
      "   udot    z4.s , z16.b, z0.b[0]                     \n"
      "   udot    z5.s , z17.b, z0.b[0]                     \n"
      "   udot    z6.s , z18.b, z0.b[0]                     \n"
      "   udot    z7.s , z19.b, z0.b[0]                     \n"
      "   udot    z12.s, z20.b, z0.b[0]                     \n"
      "   udot    z13.s, z21.b, z0.b[0]                     \n"
      "   udot    z14.s, z22.b, z0.b[0]                     \n"
      "   udot    z15.s, z23.b, z0.b[0]                     \n"
      "   ld1b    {z16.b-z17.b}, pn8/z, [%[a_ptr]]          \n"
      "   ld1b    {z18.b-z19.b}, pn8/z, [%[a_ptr], %[a1off]]\n"
      "   ld1b    {z20.b-z21.b}, pn8/z, [%[a_ptr], %[a2off]]\n"
      "   ld1b    {z22.b-z23.b}, pn8/z, [%[a_ptr], %[a3off]]\n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   zip1    z24.b, z16.b, z20.b                       \n"
      "   zip2    z25.b, z16.b, z20.b                       \n"
      "   zip1    z26.b, z18.b, z22.b                       \n"
      "   zip2    z27.b, z18.b, z22.b                       \n"
      "   zip1    z28.b, z17.b, z21.b                       \n"
      "   zip2    z29.b, z17.b, z21.b                       \n"
      "   zip1    z30.b, z19.b, z23.b                       \n"
      "   zip2    z31.b, z19.b, z23.b                       \n"
      "   zip1    z16.b, z24.b, z26.b                       \n"
      "   zip2    z17.b, z24.b, z26.b                       \n"
      "   zip1    z18.b, z25.b, z27.b                       \n"
      "   zip2    z19.b, z25.b, z27.b                       \n"
      "   zip1    z20.b, z28.b, z30.b                       \n"
      "   zip2    z21.b, z28.b, z30.b                       \n"
      "   zip1    z22.b, z29.b, z31.b                       \n"
      "   zip2    z23.b, z29.b, z31.b                       \n"
      "   udot    z4.s , z16.b, z0.b[1]                     \n"
      "   udot    z5.s , z17.b, z0.b[1]                     \n"
      "   udot    z6.s , z18.b, z0.b[1]                     \n"
      "   udot    z7.s , z19.b, z0.b[1]                     \n"
      "   udot    z12.s, z20.b, z0.b[1]                     \n"
      "   udot    z13.s, z21.b, z0.b[1]                     \n"
      "   udot    z14.s, z22.b, z0.b[1]                     \n"
      "   udot    z15.s, z23.b, z0.b[1]                     \n"
      "   add     %[n_idx], %[n_idx], #8                    \n"
      "   cmp     %[n_idx], %[n]                            \n"
      "   b.mi    2b                                        \n"

      // Store
      "   st1w    {z4.s-z7.s}  , pn8, [%[c_dst], #0, mul vl]\n"
      "   st1w    {z12.s-z15.s}, pn8, [%[c_dst], #4, mul vl]\n"
      "   addvl   %[c_dst], %[c_dst], #8                    \n"

      // M loop tail
      "   incb    %[m_idx], all, mul #2                     \n"
      "   cmp     %[m_idx], %[m]                            \n"
      "   b.mi    1b                                        \n"

      : [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [a_ptr] "=&r"(a_ptr),
        [c_dst] "+&r"(c)
      : [a1off] "r"(m), [a2off] "r"(a2off), [a3off] "r"(a3off),
        [a_src] "r"(a), [b_src] "r"(b), [m] "r"(m), [n] "r"(n)
      : "z0", "z4", "z5", "z6", "z7", "z12", "z13", "z14", "z15", "z16",
        "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "z28", "z29", "z30", "z31", "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_219(struct loop_219_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t a2off = m * 2;
  register uint64_t a3off = m * 3;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a0ptr;
  register uint64_t a1ptr;
  register uint64_t a1src;

  asm volatile(
      "   ptrue   p0.b, VL8                                 \n"
      "   ptrue   p1.b                                      \n"
      "   addvl   %[a1src], %[a0src], #1                    \n"

      // M loop head
      "   mov     %[m_idx], #0                              \n"
      "1:                                                   \n"
      "   mov     z1.s, #0                                  \n"
      "   mov     z2.s, #0                                  \n"
      "   mov     z3.s, #0                                  \n"
      "   mov     z4.s, #0                                  \n"
      "   mov     z5.s, #0                                  \n"
      "   mov     z6.s, #0                                  \n"
      "   mov     z7.s, #0                                  \n"
      "   mov     z8.s, #0                                  \n"

      // N loop
      "   mov     %[n_idx], #0                              \n"
      "   add     %[a0ptr], %[a0src], %[m_idx]              \n"
      "   add     %[a1ptr], %[a1src], %[m_idx]              \n"
      "2:                                                   \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]        \n"
      "   ld1b    {z16.b}, p1/z, [%[a0ptr]]                 \n"
      "   ld1b    {z17.b}, p1/z, [%[a1ptr]]                 \n"
      "   ld1b    {z18.b}, p1/z, [%[a0ptr], %[a1off]]       \n"
      "   ld1b    {z19.b}, p1/z, [%[a1ptr], %[a1off]]       \n"
      "   ld1b    {z20.b}, p1/z, [%[a0ptr], %[a2off]]       \n"
      "   ld1b    {z21.b}, p1/z, [%[a1ptr], %[a2off]]       \n"
      "   ld1b    {z22.b}, p1/z, [%[a0ptr], %[a3off]]       \n"
      "   ld1b    {z23.b}, p1/z, [%[a1ptr], %[a3off]]       \n"
      "   add     %[a0ptr], %[a0ptr], %[m], lsl #2          \n"
      "   add     %[a1ptr], %[a1ptr], %[m], lsl #2          \n"
      "   zip1    z24.b, z16.b, z20.b                       \n"
      "   zip2    z25.b, z16.b, z20.b                       \n"
      "   zip1    z26.b, z18.b, z22.b                       \n"
      "   zip2    z27.b, z18.b, z22.b                       \n"
      "   zip1    z28.b, z17.b, z21.b                       \n"
      "   zip2    z29.b, z17.b, z21.b                       \n"
      "   zip1    z30.b, z19.b, z23.b                       \n"
      "   zip2    z31.b, z19.b, z23.b                       \n"
      "   zip1    z16.b, z24.b, z26.b                       \n"
      "   zip2    z17.b, z24.b, z26.b                       \n"
      "   zip1    z18.b, z25.b, z27.b                       \n"
      "   zip2    z19.b, z25.b, z27.b                       \n"
      "   zip1    z20.b, z28.b, z30.b                       \n"
      "   zip2    z21.b, z28.b, z30.b                       \n"
      "   zip1    z22.b, z29.b, z31.b                       \n"
      "   zip2    z23.b, z29.b, z31.b                       \n"
      "   udot    z1.s, z16.b, z0.b[0]                      \n"
      "   udot    z2.s, z17.b, z0.b[0]                      \n"
      "   udot    z3.s, z18.b, z0.b[0]                      \n"
      "   udot    z4.s, z19.b, z0.b[0]                      \n"
      "   udot    z5.s, z20.b, z0.b[0]                      \n"
      "   udot    z6.s, z21.b, z0.b[0]                      \n"
      "   udot    z7.s, z22.b, z0.b[0]                      \n"
      "   udot    z8.s, z23.b, z0.b[0]                      \n"
      "   ld1b    {z16.b}, p1/z, [%[a0ptr]]                 \n"
      "   ld1b    {z17.b}, p1/z, [%[a1ptr]]                 \n"
      "   ld1b    {z18.b}, p1/z, [%[a0ptr], %[a1off]]       \n"
      "   ld1b    {z19.b}, p1/z, [%[a1ptr], %[a1off]]       \n"
      "   ld1b    {z20.b}, p1/z, [%[a0ptr], %[a2off]]       \n"
      "   ld1b    {z21.b}, p1/z, [%[a1ptr], %[a2off]]       \n"
      "   ld1b    {z22.b}, p1/z, [%[a0ptr], %[a3off]]       \n"
      "   ld1b    {z23.b}, p1/z, [%[a1ptr], %[a3off]]       \n"
      "   add     %[a0ptr], %[a0ptr], %[m], lsl #2          \n"
      "   add     %[a1ptr], %[a1ptr], %[m], lsl #2          \n"
      "   zip1    z24.b, z16.b, z20.b                       \n"
      "   zip2    z25.b, z16.b, z20.b                       \n"
      "   zip1    z26.b, z18.b, z22.b                       \n"
      "   zip2    z27.b, z18.b, z22.b                       \n"
      "   zip1    z28.b, z17.b, z21.b                       \n"
      "   zip2    z29.b, z17.b, z21.b                       \n"
      "   zip1    z30.b, z19.b, z23.b                       \n"
      "   zip2    z31.b, z19.b, z23.b                       \n"
      "   zip1    z16.b, z24.b, z26.b                       \n"
      "   zip2    z17.b, z24.b, z26.b                       \n"
      "   zip1    z18.b, z25.b, z27.b                       \n"
      "   zip2    z19.b, z25.b, z27.b                       \n"
      "   zip1    z20.b, z28.b, z30.b                       \n"
      "   zip2    z21.b, z28.b, z30.b                       \n"
      "   zip1    z22.b, z29.b, z31.b                       \n"
      "   zip2    z23.b, z29.b, z31.b                       \n"
      "   udot    z1.s, z16.b, z0.b[1]                      \n"
      "   udot    z2.s, z17.b, z0.b[1]                      \n"
      "   udot    z3.s, z18.b, z0.b[1]                      \n"
      "   udot    z4.s, z19.b, z0.b[1]                      \n"
      "   udot    z5.s, z20.b, z0.b[1]                      \n"
      "   udot    z6.s, z21.b, z0.b[1]                      \n"
      "   udot    z7.s, z22.b, z0.b[1]                      \n"
      "   udot    z8.s, z23.b, z0.b[1]                      \n"
      "   add     %[n_idx], %[n_idx], #8                    \n"
      "   cmp     %[n_idx], %[n]                            \n"
      "   b.mi    2b                                        \n"

      // Store
      "   st1w    {z1.s}, p1, [%[c_dst], #0, mul vl]        \n"
      "   st1w    {z2.s}, p1, [%[c_dst], #1, mul vl]        \n"
      "   st1w    {z3.s}, p1, [%[c_dst], #2, mul vl]        \n"
      "   st1w    {z4.s}, p1, [%[c_dst], #3, mul vl]        \n"
      "   st1w    {z5.s}, p1, [%[c_dst], #4, mul vl]        \n"
      "   st1w    {z6.s}, p1, [%[c_dst], #5, mul vl]        \n"
      "   st1w    {z7.s}, p1, [%[c_dst], #6, mul vl]        \n"
      "   st1w    {z8.s}, p1, [%[c_dst], #7, mul vl]        \n"
      "   addvl   %[c_dst], %[c_dst], #8                    \n"

      // M loop tail
      "   incb    %[m_idx], all, mul #2                     \n"
      "   cmp     %[m_idx], %[m]                            \n"
      "   b.mi    1b                                        \n"

      : [a0ptr] "=&r"(a0ptr), [a1ptr] "=&r"(a1ptr), [a1src] "+&r"(a1src),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [a1off] "r"(m), [a2off] "r"(a2off), [a3off] "r"(a3off),
        [a0src] "r"(a), [b_src] "r"(b), [m] "r"(m), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z16", "z17",
        "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
        "z28", "z29", "z30", "z31", "p0", "p1", "cc", "memory");
}

#elif (defined(__ARM_NEON) && defined (__ARM_FEATURE_DOTPROD))

static void inner_loop_219(struct loop_219_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_cnd = (uint64_t)&(data->c[m]);
  register uint64_t n_cnd = (uint64_t)&(data->b[n]);

  register uint64_t a_ptr;
  register uint64_t b_ptr;

  asm volatile(
      // M loop head
      "1:                                                         \n"
      "   movi    v1.4s, #0                                       \n"
      "   movi    v2.4s, #0                                       \n"
      "   movi    v3.4s, #0                                       \n"
      "   movi    v4.4s, #0                                       \n"
      "   movi    v5.4s, #0                                       \n"
      "   movi    v6.4s, #0                                       \n"
      "   movi    v7.4s, #0                                       \n"
      "   movi    v8.4s, #0                                       \n"

      // N loop
      "   mov     %[a_ptr], %[a_src]                              \n"
      "   mov     %[b_ptr], %[b_src]                              \n"
      "2:                                                         \n"
      "   ld1     {v0.8b}, [%[b_ptr]], #8                         \n"
      "   ld1     {v16.16b,v17.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v18.16b,v19.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v20.16b,v21.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v22.16b,v23.16b}, [%[a_ptr]], %[m]             \n"
      "   zip1    v24.16b, v16.16b, v20.16b                       \n"
      "   zip2    v25.16b, v16.16b, v20.16b                       \n"
      "   zip1    v26.16b, v18.16b, v22.16b                       \n"
      "   zip2    v27.16b, v18.16b, v22.16b                       \n"
      "   zip1    v28.16b, v17.16b, v21.16b                       \n"
      "   zip2    v29.16b, v17.16b, v21.16b                       \n"
      "   zip1    v30.16b, v19.16b, v23.16b                       \n"
      "   zip2    v31.16b, v19.16b, v23.16b                       \n"
      "   zip1    v16.16b, v24.16b, v26.16b                       \n"
      "   zip2    v17.16b, v24.16b, v26.16b                       \n"
      "   zip1    v18.16b, v25.16b, v27.16b                       \n"
      "   zip2    v19.16b, v25.16b, v27.16b                       \n"
      "   zip1    v20.16b, v28.16b, v30.16b                       \n"
      "   zip2    v21.16b, v28.16b, v30.16b                       \n"
      "   zip1    v22.16b, v29.16b, v31.16b                       \n"
      "   zip2    v23.16b, v29.16b, v31.16b                       \n"
      "   udot    v1.4s, v16.16b, v0.4b[0]                        \n"
      "   udot    v2.4s, v17.16b, v0.4b[0]                        \n"
      "   udot    v3.4s, v18.16b, v0.4b[0]                        \n"
      "   udot    v4.4s, v19.16b, v0.4b[0]                        \n"
      "   udot    v5.4s, v20.16b, v0.4b[0]                        \n"
      "   udot    v6.4s, v21.16b, v0.4b[0]                        \n"
      "   udot    v7.4s, v22.16b, v0.4b[0]                        \n"
      "   udot    v8.4s, v23.16b, v0.4b[0]                        \n"
      "   ld1     {v16.16b,v17.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v18.16b,v19.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v20.16b,v21.16b}, [%[a_ptr]], %[m]             \n"
      "   ld1     {v22.16b,v23.16b}, [%[a_ptr]], %[m]             \n"
      "   zip1    v24.16b, v16.16b, v20.16b                       \n"
      "   zip2    v25.16b, v16.16b, v20.16b                       \n"
      "   zip1    v26.16b, v18.16b, v22.16b                       \n"
      "   zip2    v27.16b, v18.16b, v22.16b                       \n"
      "   zip1    v28.16b, v17.16b, v21.16b                       \n"
      "   zip2    v29.16b, v17.16b, v21.16b                       \n"
      "   zip1    v30.16b, v19.16b, v23.16b                       \n"
      "   zip2    v31.16b, v19.16b, v23.16b                       \n"
      "   zip1    v16.16b, v24.16b, v26.16b                       \n"
      "   zip2    v17.16b, v24.16b, v26.16b                       \n"
      "   zip1    v18.16b, v25.16b, v27.16b                       \n"
      "   zip2    v19.16b, v25.16b, v27.16b                       \n"
      "   zip1    v20.16b, v28.16b, v30.16b                       \n"
      "   zip2    v21.16b, v28.16b, v30.16b                       \n"
      "   zip1    v22.16b, v29.16b, v31.16b                       \n"
      "   zip2    v23.16b, v29.16b, v31.16b                       \n"
      "   udot    v1.4s, v16.16b, v0.4b[1]                        \n"
      "   udot    v2.4s, v17.16b, v0.4b[1]                        \n"
      "   udot    v3.4s, v18.16b, v0.4b[1]                        \n"
      "   udot    v4.4s, v19.16b, v0.4b[1]                        \n"
      "   udot    v5.4s, v20.16b, v0.4b[1]                        \n"
      "   udot    v6.4s, v21.16b, v0.4b[1]                        \n"
      "   udot    v7.4s, v22.16b, v0.4b[1]                        \n"
      "   udot    v8.4s, v23.16b, v0.4b[1]                        \n"
      "   cmp     %[b_ptr], %[n_cnd]                              \n"
      "   b.mi    2b                                              \n"

      // Store
      "   st1     {v1.4s,v2.4s,v3.4s,v4.4s}, [%[c_dst]], #64      \n"
      "   st1     {v5.4s,v6.4s,v7.4s,v8.4s}, [%[c_dst]], #64      \n"

      // M loop tail
      "   add     %[a_src], %[a_src], #32                         \n"
      "   cmp     %[c_dst], %[m_cnd]                              \n"
      "   b.mi    1b                                              \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr),
        [a_src] "+&r"(a), [c_dst] "+&r"(c)
      : [m_cnd] "r"(m_cnd), [n_cnd] "r"(n_cnd), [m] "r"(m), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31", "cc", "memory");
}

#else

static void inner_loop_219(struct loop_219_data *data) {
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
// Default of 257KiB equates to original problem size (M=1024, N=256)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n) ((n)*((m)+1)*sizeof(uint8_t))

LOOP_DECL(219, OUTER_LOOP_ATTR)
{
  // Work out values for M and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 4*SVLb
  uint64_t N = 0;  // multiple of 16

  const uint64_t M_base = MAX_VL / 2;
  while (true) {
    // N must a multiple of 16 (which it will implicitly will be as M is
    // guaranteed to be) and should be 4x smaller than M for this
    // loop's M-to-N ratio.
    uint64_t m = M + M_base;
    uint64_t n = m / 4;
    if (PROBLEM_SIZE_ACTUAL(m,n) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
    } else {
      break;
    }
  }

  struct loop_219_data data = { .m = M, .n = N, };
  ALLOC_64B(data.a, M * N, "A matrix");
  ALLOC_64B(data.b, N * 1, "x vector");
  ALLOC_64B(data.c, M * 1, "b vector");

  fill_uint8(data.a, M * N);
  fill_uint8(data.b, N * 1);

  inner_loops_219(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x 1\n", M, N, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(j)                                                    \
  {                                                                 \
    uint32_t d = 0;                                                 \
    for (int n = 0; n < N; n++) MLA(d, data.a[n*M+j], data.b[n]);   \
    checksum += (int)(d != data.c[j]);                              \
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
  FINALISE_LOOP_I(219, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
