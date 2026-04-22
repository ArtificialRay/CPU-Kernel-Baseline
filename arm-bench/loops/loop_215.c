/*----------------------------------------------------------------------------
#
#   Loop 215: UINT8-UINT32 col-major interleaved matrix-vector multiply
#
#   Purpose:
#     Use of u8 to u32 DOT instruction.
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
    A: column-major, 4-way interleaved columns
    B: column-vector
    C: column-vector
  Constraints -
    M: multiple of 6*SVLb
    N: multiple of 16
*/

struct loop_215_data {
  uint64_t m;
  uint64_t n;
  uint8_t *restrict a;
  uint8_t *restrict b;
  uint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_215(struct loop_215_data *restrict data) {
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
// Interleaved index into A matrix
#define ZIP(m, y, x) ((m) * ((y) - ((y) % 4)) + ((x) * 4) + ((y) % 4))

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_215(struct loop_215_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  uint32_t *restrict c = data->c;
  for (uint64_t y = 0; y < m; y++) {
    uint32_t d = 0;
    for (uint64_t x = 0; x < n; x += 4) {
      MLA(d, a[x * m + y * 4 + 0], b[x + 0]);
      MLA(d, a[x * m + y * 4 + 1], b[x + 1]);
      MLA(d, a[x * m + y * 4 + 2], b[x + 2]);
      MLA(d, a[x * m + y * 4 + 3], b[x + 3]);
    }
    c[y] = d;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_215(struct loop_215_data *data) LOOP_ATTR {
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
  svuint64x4_t stc_0, stc_1, stc_2, stc_3;

#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define LOAD(q) lda_##q = svld1_vnum_x4(c_all, a_ptr, 4 * q);
#define LOAD_GROUP LOAD(0) LOAD(1) LOAD(2) LOAD(3)

#define UDOT(l, q) svdot_lane_za32_vg1x4(l_idx + q, lda_##q, ldb, l);
#define UDOT_GROUP(l) UDOT(l, 0) UDOT(l, 1) UDOT(l, 2) UDOT(l, 3)

#define CAST(i, j) svreinterpret_u32(svget4(stc_##i, j))
#define QUAD(i) svcreate4(CAST(i, 0), CAST(i, 1), CAST(i, 2), CAST(i, 3))

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
        a_ptr = &a_bar[o_idx * 4];
        LOAD_GROUP;
        UDOT_GROUP(0);
        a_ptr += 4 * m;
        LOAD_GROUP;
        UDOT_GROUP(1);
        a_ptr += 4 * m;
        LOAD_GROUP;
        UDOT_GROUP(2);
        a_ptr += 4 * m;
        LOAD_GROUP;
        UDOT_GROUP(3);
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
      stc_0 = svreadz_za64_u64_vg1x4(l_idx + 0);
      stc_1 = svreadz_za64_u64_vg1x4(l_idx + 1);
      stc_2 = svreadz_za64_u64_vg1x4(l_idx + 2);
      stc_3 = svreadz_za64_u64_vg1x4(l_idx + 3);
#else
      stc_0 = svread_za64_u64_vg1x4(l_idx + 0);
      stc_1 = svread_za64_u64_vg1x4(l_idx + 1);
      stc_2 = svread_za64_u64_vg1x4(l_idx + 2);
      stc_3 = svread_za64_u64_vg1x4(l_idx + 3);
#endif
      svst1_vnum(c_all, c_ptr, 0x0, QUAD(0));
      svst1_vnum(c_all, c_ptr, 0x4, QUAD(1));
      svst1_vnum(c_all, c_ptr, 0x8, QUAD(2));
      svst1_vnum(c_all, c_ptr, 0xc, QUAD(3));
      l_idx += 4;
      c_ptr += 4 * svl_b;
    }
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_215(struct loop_215_data *data) LOOP_ATTR {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint8_t *a_ptr;
  uint32_t *c_ptr;

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b8();

  svuint32x4_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5;
  svuint8x4_t lda_0, lda_1, lda_2, lda_3, lda_4, lda_5;
  svuint8_t ldx;

#define ZERO svdup_u32(0)
#define ZERO_QUAD(q) acc_##q = svcreate4(ZERO, ZERO, ZERO, ZERO)

#define GETA(q, p) svget4(lda_##q, p)
#define GETB(q, p) svget4(acc_##q, p)

#define UDOT(l, q, p) svdot_lane(GETB(q, p), GETA(q, p), ldx, l)
#define UDOT_LANE(l, q) \
  acc_##q =             \
      svcreate4(UDOT(l, q, 0), UDOT(l, q, 1), UDOT(l, q, 2), UDOT(l, q, 3))

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c8();
#define LOAD_QUAD(q) lda_##q = svld1_vnum_x4(c_all, a_ptr, q * 4)
#define STORE_QUAD(q) svst1_vnum(c_all, c_ptr, q * 4, acc_##q);
#else
#define LOAD(q, p) svld1_vnum(p_all, a_ptr, q * 4 + p)
#define LOAD_QUAD(q) \
  lda_##q = svcreate4(LOAD(q, 0), LOAD(q, 1), LOAD(q, 2), LOAD(q, 3))
#define STORE(q, p) svst1_vnum(p_all, c_ptr, q * 4 + p, GETB(q, p));
#define STORE_QUAD(q) STORE(q, 0) STORE(q, 1) STORE(q, 2) STORE(q, 3)
#endif

  for (m_idx = 0; m_idx < m; m_idx += svcntb() * 6) {
    ZERO_QUAD(0);
    ZERO_QUAD(1);
    ZERO_QUAD(2);
    ZERO_QUAD(3);
    ZERO_QUAD(4);
    ZERO_QUAD(5);

    a_ptr = &a[m_idx * 4];
    for (n_idx = 0; n_idx < n; n_idx += 16) {
      ldx = svld1rq(p_all, &b[n_idx]);
      LOAD_QUAD(0);
      LOAD_QUAD(1);
      LOAD_QUAD(2);
      LOAD_QUAD(3);
      LOAD_QUAD(4);
      LOAD_QUAD(5);
      UDOT_LANE(0, 0);
      UDOT_LANE(0, 1);
      UDOT_LANE(0, 2);
      UDOT_LANE(0, 3);
      UDOT_LANE(0, 4);
      UDOT_LANE(0, 5);
      a_ptr += 4 * m;
      LOAD_QUAD(0);
      LOAD_QUAD(1);
      LOAD_QUAD(2);
      LOAD_QUAD(3);
      LOAD_QUAD(4);
      LOAD_QUAD(5);
      UDOT_LANE(1, 0);
      UDOT_LANE(1, 1);
      UDOT_LANE(1, 2);
      UDOT_LANE(1, 3);
      UDOT_LANE(1, 4);
      UDOT_LANE(1, 5);
      a_ptr += 4 * m;
      LOAD_QUAD(0);
      LOAD_QUAD(1);
      LOAD_QUAD(2);
      LOAD_QUAD(3);
      LOAD_QUAD(4);
      LOAD_QUAD(5);
      UDOT_LANE(2, 0);
      UDOT_LANE(2, 1);
      UDOT_LANE(2, 2);
      UDOT_LANE(2, 3);
      UDOT_LANE(2, 4);
      UDOT_LANE(2, 5);
      a_ptr += 4 * m;
      LOAD_QUAD(0);
      LOAD_QUAD(1);
      LOAD_QUAD(2);
      LOAD_QUAD(3);
      LOAD_QUAD(4);
      LOAD_QUAD(5);
      UDOT_LANE(3, 0);
      UDOT_LANE(3, 1);
      UDOT_LANE(3, 2);
      UDOT_LANE(3, 3);
      UDOT_LANE(3, 4);
      UDOT_LANE(3, 5);
      a_ptr += 4 * m;
    }

    c_ptr = &c[m_idx];
    STORE_QUAD(0);
    STORE_QUAD(1);
    STORE_QUAD(2);
    STORE_QUAD(3);
    STORE_QUAD(4);
    STORE_QUAD(5);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_215(struct loop_215_data *data) LOOP_ATTR {
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
  // x9: slice index register for udot and mova

  asm volatile(
      "   ptrue   pn8.b                                               \n"
      "   ptrue   p0.b                                                \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      "   add     %[o_cnd], %[m_idx], %[m_blk]                        \n"
      "   cmp     %[o_cnd], %[m]                                      \n"
      "   csel    %[o_cnd], %[o_cnd], %[m], lt                        \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "   mov     %[a_bar], %[a_src]                                  \n"
      "2:                                                             \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]                  \n"

      // M-block dot-product loop
      "   mov     x9, #0                                              \n"
      "   mov     %[o_idx], %[m_idx]                                  \n"
      "3:                                                             \n"
      "   add     %[a_ptr], %[a_bar], %[o_idx], lsl #2                \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                    \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], #0x4, mul vl]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], #0x8, mul vl]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], #0xc, mul vl]      \n"
      "   udot    za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[0]           \n"
      "   udot    za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[0]           \n"
      "   udot    za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[0]           \n"
      "   udot    za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[0]           \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                    \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], #0x4, mul vl]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], #0x8, mul vl]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], #0xc, mul vl]      \n"
      "   udot    za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[1]           \n"
      "   udot    za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[1]           \n"
      "   udot    za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[1]           \n"
      "   udot    za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[1]           \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                    \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], #0x4, mul vl]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], #0x8, mul vl]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], #0xc, mul vl]      \n"
      "   udot    za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[2]           \n"
      "   udot    za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[2]           \n"
      "   udot    za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[2]           \n"
      "   udot    za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[2]           \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                    \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], #0x4, mul vl]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], #0x8, mul vl]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], #0xc, mul vl]      \n"
      "   udot    za.s[w9, 0, vgx4], {z16.b-z19.b}, z0.b[3]           \n"
      "   udot    za.s[w9, 1, vgx4], {z20.b-z23.b}, z0.b[3]           \n"
      "   udot    za.s[w9, 2, vgx4], {z24.b-z27.b}, z0.b[3]           \n"
      "   udot    za.s[w9, 3, vgx4], {z28.b-z31.b}, z0.b[3]           \n"
      "   add     x9, x9, #4                                          \n"
      "   incb    %[o_idx], all, mul #4                               \n"
      "   cmp     %[o_idx], %[o_cnd]                                  \n"
      "   b.mi    3b                                                  \n"

      // N loop tail
      "   add     %[a_bar], %[a_bar], %[m], lsl #4                    \n"
      "   add     %[n_idx], %[n_idx], #16                             \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // Store loop
      "   mov     x9, #0                                              \n"
      "   add     %[c_ptr], %[c_dst], %[m_idx], lsl #2                \n"
      "   add     %[c_cnd], %[c_dst], %[o_cnd], lsl #2                \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   movaz   {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   movaz   {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   movaz   {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#else
      "   mova    {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   mova    {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   mova    {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   mova    {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#endif
      "   addvl   %[c_cnd], %[c_cnd], #-16                            \n"
      "   cmp     %[c_ptr], %[c_cnd]                                  \n"
      "   b.pl    5f                                                  \n"
      "4:                                                             \n"
      "   add     x9, x9, #4                                          \n"
      "   st1w    {z16.s-z19.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z20.s-z23.s}, pn8, [%[c_ptr], #0x4, mul vl]        \n"
      "   st1w    {z24.s-z27.s}, pn8, [%[c_ptr], #0x8, mul vl]        \n"
      "   st1w    {z28.s-z31.s}, pn8, [%[c_ptr], #0xc, mul vl]        \n"
      "   addvl   %[c_ptr], %[c_ptr], #16                             \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   movaz   {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   movaz   {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   movaz   {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#else
      "   mova    {z16.s-z19.s}, za.s[w9, 0, vgx4]                    \n"
      "   mova    {z20.s-z23.s}, za.s[w9, 1, vgx4]                    \n"
      "   mova    {z24.s-z27.s}, za.s[w9, 2, vgx4]                    \n"
      "   mova    {z28.s-z31.s}, za.s[w9, 3, vgx4]                    \n"
#endif
      "   cmp     %[c_ptr], %[c_cnd]                                  \n"
      "   b.mi    4b                                                  \n"
      "5:                                                             \n"
      "   st1w    {z16.s-z19.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z20.s-z23.s}, pn8, [%[c_ptr], #0x4, mul vl]        \n"
      "   st1w    {z24.s-z27.s}, pn8, [%[c_ptr], #0x8, mul vl]        \n"
      "   st1w    {z28.s-z31.s}, pn8, [%[c_ptr], #0xc, mul vl]        \n"

      // M loop tail
      "   add     %[m_idx], %[m_idx], %[m_blk]                        \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [a_bar] "=&r"(a_bar),
        [a_ptr] "=&r"(a_ptr), [c_ptr] "=&r"(c_ptr), [o_idx] "=&r"(o_idx),
        [o_cnd] "=&r"(o_cnd), [c_cnd] "=&r"(c_cnd), [c_dst] "+&r"(c)
      :
      [m] "r"(m), [n] "r"(n), [m_blk] "r"(m_blk), [a_src] "r"(a), [b_src] "r"(b)
      : "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
        "z26", "z27", "z28", "z29", "z30", "z31", "z0", "p0", "p8", "x9",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_215(struct loop_215_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t a1ptr;

  asm volatile(
      "   ptrue   p0.b                                            \n"
      "   ptrue   pn8.b                                           \n"

      // M loop head
      "   mov     %[m_idx], #0                                    \n"
      "1:                                                         \n"
      "   mov     z8.s, #0                                        \n"
      "   mov     z9.s, #0                                        \n"
      "   mov     z10.s, #0                                       \n"
      "   mov     z11.s, #0                                       \n"
      "   mov     z12.s, #0                                       \n"
      "   mov     z13.s, #0                                       \n"
      "   mov     z14.s, #0                                       \n"
      "   mov     z15.s, #0                                       \n"
      "   mov     z16.s, #0                                       \n"
      "   mov     z17.s, #0                                       \n"
      "   mov     z18.s, #0                                       \n"
      "   mov     z19.s, #0                                       \n"
      "   mov     z20.s, #0                                       \n"
      "   mov     z21.s, #0                                       \n"
      "   mov     z22.s, #0                                       \n"
      "   mov     z23.s, #0                                       \n"
      "   mov     z24.s, #0                                       \n"
      "   mov     z25.s, #0                                       \n"
      "   mov     z26.s, #0                                       \n"
      "   mov     z27.s, #0                                       \n"
      "   mov     z28.s, #0                                       \n"
      "   mov     z29.s, #0                                       \n"
      "   mov     z30.s, #0                                       \n"
      "   mov     z31.s, #0                                       \n"

      // N loop
      "   mov     %[n_idx], #0                                    \n"
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2            \n"
      "2:                                                         \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]              \n"
      "   addvl   %[a1ptr], %[a_ptr], #16                         \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x0, mul vl]  \n"
      "   udot    z8.s, z4.b , z0.b[0]                            \n"
      "   udot    z9.s, z5.b , z0.b[0]                            \n"
      "   udot    z10.s, z6.b , z0.b[0]                           \n"
      "   udot    z11.s, z7.b , z0.b[0]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0x4, mul vl]    \n"
      "   udot    z12.s, z4.b, z0.b[0]                            \n"
      "   udot    z13.s, z5.b, z0.b[0]                            \n"
      "   udot    z14.s, z6.b, z0.b[0]                            \n"
      "   udot    z15.s, z7.b, z0.b[0]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x8, mul vl]  \n"
      "   udot    z16.s, z4.b , z0.b[0]                           \n"
      "   udot    z17.s, z5.b , z0.b[0]                           \n"
      "   udot    z18.s, z6.b , z0.b[0]                           \n"
      "   udot    z19.s, z7.b , z0.b[0]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0xc, mul vl]    \n"
      "   udot    z20.s, z4.b, z0.b[0]                            \n"
      "   udot    z21.s, z5.b, z0.b[0]                            \n"
      "   udot    z22.s, z6.b, z0.b[0]                            \n"
      "   udot    z23.s, z7.b, z0.b[0]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a1ptr], #0x0, mul vl]  \n"
      "   udot    z24.s, z4.b , z0.b[0]                           \n"
      "   udot    z25.s, z5.b , z0.b[0]                           \n"
      "   udot    z26.s, z6.b , z0.b[0]                           \n"
      "   udot    z27.s, z7.b , z0.b[0]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a1ptr], #0x4, mul vl]    \n"
      "   udot    z28.s, z4.b, z0.b[0]                            \n"
      "   udot    z29.s, z5.b, z0.b[0]                            \n"
      "   udot    z30.s, z6.b, z0.b[0]                            \n"
      "   udot    z31.s, z7.b, z0.b[0]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                \n"
      "   addvl   %[a1ptr], %[a_ptr], #16                         \n"
      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x0, mul vl]  \n"
      "   udot    z8.s, z4.b , z0.b[1]                            \n"
      "   udot    z9.s, z5.b , z0.b[1]                            \n"
      "   udot    z10.s, z6.b , z0.b[1]                           \n"
      "   udot    z11.s, z7.b , z0.b[1]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0x4, mul vl]    \n"
      "   udot    z12.s, z4.b, z0.b[1]                            \n"
      "   udot    z13.s, z5.b, z0.b[1]                            \n"
      "   udot    z14.s, z6.b, z0.b[1]                            \n"
      "   udot    z15.s, z7.b, z0.b[1]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x8, mul vl]  \n"
      "   udot    z16.s, z4.b , z0.b[1]                           \n"
      "   udot    z17.s, z5.b , z0.b[1]                           \n"
      "   udot    z18.s, z6.b , z0.b[1]                           \n"
      "   udot    z19.s, z7.b , z0.b[1]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0xc, mul vl]    \n"
      "   udot    z20.s, z4.b, z0.b[1]                            \n"
      "   udot    z21.s, z5.b, z0.b[1]                            \n"
      "   udot    z22.s, z6.b, z0.b[1]                            \n"
      "   udot    z23.s, z7.b, z0.b[1]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a1ptr], #0x0, mul vl]  \n"
      "   udot    z24.s, z4.b , z0.b[1]                           \n"
      "   udot    z25.s, z5.b , z0.b[1]                           \n"
      "   udot    z26.s, z6.b , z0.b[1]                           \n"
      "   udot    z27.s, z7.b , z0.b[1]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a1ptr], #0x4, mul vl]    \n"
      "   udot    z28.s, z4.b, z0.b[1]                            \n"
      "   udot    z29.s, z5.b, z0.b[1]                            \n"
      "   udot    z30.s, z6.b, z0.b[1]                            \n"
      "   udot    z31.s, z7.b, z0.b[1]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                \n"
      "   addvl   %[a1ptr], %[a_ptr], #16                         \n"
      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x0, mul vl]  \n"
      "   udot    z8.s, z4.b , z0.b[2]                            \n"
      "   udot    z9.s, z5.b , z0.b[2]                            \n"
      "   udot    z10.s, z6.b , z0.b[2]                           \n"
      "   udot    z11.s, z7.b , z0.b[2]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0x4, mul vl]    \n"
      "   udot    z12.s, z4.b, z0.b[2]                            \n"
      "   udot    z13.s, z5.b, z0.b[2]                            \n"
      "   udot    z14.s, z6.b, z0.b[2]                            \n"
      "   udot    z15.s, z7.b, z0.b[2]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x8, mul vl]  \n"
      "   udot    z16.s, z4.b , z0.b[2]                           \n"
      "   udot    z17.s, z5.b , z0.b[2]                           \n"
      "   udot    z18.s, z6.b , z0.b[2]                           \n"
      "   udot    z19.s, z7.b , z0.b[2]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0xc, mul vl]    \n"
      "   udot    z20.s, z4.b, z0.b[2]                            \n"
      "   udot    z21.s, z5.b, z0.b[2]                            \n"
      "   udot    z22.s, z6.b, z0.b[2]                            \n"
      "   udot    z23.s, z7.b, z0.b[2]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a1ptr], #0x0, mul vl]  \n"
      "   udot    z24.s, z4.b , z0.b[2]                           \n"
      "   udot    z25.s, z5.b , z0.b[2]                           \n"
      "   udot    z26.s, z6.b , z0.b[2]                           \n"
      "   udot    z27.s, z7.b , z0.b[2]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a1ptr], #0x4, mul vl]    \n"
      "   udot    z28.s, z4.b, z0.b[2]                            \n"
      "   udot    z29.s, z5.b, z0.b[2]                            \n"
      "   udot    z30.s, z6.b, z0.b[2]                            \n"
      "   udot    z31.s, z7.b, z0.b[2]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                \n"
      "   addvl   %[a1ptr], %[a_ptr], #16                         \n"
      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x0, mul vl]  \n"
      "   udot    z8.s, z4.b , z0.b[3]                            \n"
      "   udot    z9.s, z5.b , z0.b[3]                            \n"
      "   udot    z10.s, z6.b , z0.b[3]                           \n"
      "   udot    z11.s, z7.b , z0.b[3]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0x4, mul vl]    \n"
      "   udot    z12.s, z4.b, z0.b[3]                            \n"
      "   udot    z13.s, z5.b, z0.b[3]                            \n"
      "   udot    z14.s, z6.b, z0.b[3]                            \n"
      "   udot    z15.s, z7.b, z0.b[3]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a_ptr], #0x8, mul vl]  \n"
      "   udot    z16.s, z4.b , z0.b[3]                           \n"
      "   udot    z17.s, z5.b , z0.b[3]                           \n"
      "   udot    z18.s, z6.b , z0.b[3]                           \n"
      "   udot    z19.s, z7.b , z0.b[3]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a_ptr], #0xc, mul vl]    \n"
      "   udot    z20.s, z4.b, z0.b[3]                            \n"
      "   udot    z21.s, z5.b, z0.b[3]                            \n"
      "   udot    z22.s, z6.b, z0.b[3]                            \n"
      "   udot    z23.s, z7.b, z0.b[3]                            \n"

      "   ld1b    {z4.b-z7.b},   pn8/z, [%[a1ptr], #0x0, mul vl]  \n"
      "   udot    z24.s, z4.b , z0.b[3]                           \n"
      "   udot    z25.s, z5.b , z0.b[3]                           \n"
      "   udot    z26.s, z6.b , z0.b[3]                           \n"
      "   udot    z27.s, z7.b , z0.b[3]                           \n"
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[a1ptr], #0x4, mul vl]    \n"
      "   udot    z28.s, z4.b, z0.b[3]                            \n"
      "   udot    z29.s, z5.b, z0.b[3]                            \n"
      "   udot    z30.s, z6.b, z0.b[3]                            \n"
      "   udot    z31.s, z7.b, z0.b[3]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                \n"
      "   add     %[n_idx], %[n_idx], #16                         \n"
      "   cmp     %[n_idx], %[n]                                  \n"
      "   b.mi    2b                                              \n"

      // Store
      "   st1w    {z8.s-z11.s}, pn8, [%[c_dst], #0x0, mul vl]     \n"
      "   st1w    {z12.s-z15.s}, pn8, [%[c_dst], #0x4, mul vl]    \n"
      "   st1w    {z16.s-z19.s}, pn8, [%[c_dst], #0x8, mul vl]    \n"
      "   st1w    {z20.s-z23.s}, pn8, [%[c_dst], #0xc, mul vl]    \n"
      "   addvl   %[c_dst], %[c_dst], #16                         \n"
      "   st1w    {z24.s-z27.s}, pn8, [%[c_dst], #0x0, mul vl]    \n"
      "   st1w    {z28.s-z31.s}, pn8, [%[c_dst], #0x4, mul vl]    \n"
      "   addvl   %[c_dst], %[c_dst], #8                          \n"

      // M loop tail
      "   incb    %[m_idx], all, mul #6                           \n"
      "   cmp     %[m_idx], %[m]                                  \n"
      "   b.mi    1b                                              \n"

      : [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [a_ptr] "=&r"(a_ptr),
        [a1ptr] "=&r"(a1ptr), [a_src] "+&r"(a), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [b_src] "r"(b)
      : "z0", "z4", "z5", "z6", "z7", "z8", "z12", "z13", "z14", "z15", "z16",
        "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "z28", "z29", "z30", "z31", "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_215(struct loop_215_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t a1ptr;

  asm volatile(
      // Offset pointers by 8*VL to make use of -ve indices
      "   addvl   %[a_src], %[a_src], #8                    \n"
      "   addvl   %[c_dst], %[c_dst], #8                    \n"
      "   ptrue   p0.b                                      \n"

      // M loop head
      "   mov     %[m_idx], #0                              \n"
      "1:                                                   \n"
      "   mov     z8.s, #0                                  \n"
      "   mov     z9.s, #0                                  \n"
      "   mov     z10.s, #0                                 \n"
      "   mov     z11.s, #0                                 \n"
      "   mov     z12.s, #0                                 \n"
      "   mov     z13.s, #0                                 \n"
      "   mov     z14.s, #0                                 \n"
      "   mov     z15.s, #0                                 \n"
      "   mov     z16.s, #0                                 \n"
      "   mov     z17.s, #0                                 \n"
      "   mov     z18.s, #0                                 \n"
      "   mov     z19.s, #0                                 \n"
      "   mov     z20.s, #0                                 \n"
      "   mov     z21.s, #0                                 \n"
      "   mov     z22.s, #0                                 \n"
      "   mov     z23.s, #0                                 \n"
      "   mov     z24.s, #0                                 \n"
      "   mov     z25.s, #0                                 \n"
      "   mov     z26.s, #0                                 \n"
      "   mov     z27.s, #0                                 \n"
      "   mov     z28.s, #0                                 \n"
      "   mov     z29.s, #0                                 \n"
      "   mov     z30.s, #0                                 \n"
      "   mov     z31.s, #0                                 \n"

      // N loop
      "   mov     %[n_idx], #0                              \n"
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2      \n"
      "2:                                                   \n"
      "   ld1rqb  {z0.b}, p0/z, [%[b_src], %[n_idx]]        \n"
      "   addvl   %[a1ptr], %[a_ptr], #8                    \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-8, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-7, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr], #-6, mul vl]     \n"
      "   udot    z8.s, z1.b, z0.b[0]                       \n"
      "   udot    z9.s, z2.b, z0.b[0]                       \n"
      "   udot    z10.s, z3.b, z0.b[0]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr], #-5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr], #-4, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr], #-3, mul vl]     \n"
      "   udot    z11.s, z4.b, z0.b[0]                      \n"
      "   udot    z12.s, z5.b, z0.b[0]                      \n"
      "   udot    z13.s, z6.b, z0.b[0]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-1, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #0, mul vl]     \n"
      "   udot    z14.s, z1.b, z0.b[0]                      \n"
      "   udot    z15.s, z2.b, z0.b[0]                      \n"
      "   udot    z16.s, z3.b, z0.b[0]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr],  #1, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr],  #2, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr],  #3, mul vl]     \n"
      "   udot    z17.s, z4.b, z0.b[0]                      \n"
      "   udot    z18.s, z5.b, z0.b[0]                      \n"
      "   udot    z19.s, z6.b, z0.b[0]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr],  #4, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr],  #5, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #6, mul vl]     \n"
      "   udot    z20.s, z1.b, z0.b[0]                      \n"
      "   udot    z21.s, z2.b, z0.b[0]                      \n"
      "   udot    z22.s, z3.b, z0.b[0]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #-1, mul vl]    \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #0, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #1, mul vl]     \n"
      "   udot    z23.s, z4.b, z0.b[0]                      \n"
      "   udot    z24.s, z5.b, z0.b[0]                      \n"
      "   udot    z25.s, z6.b, z0.b[0]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a1ptr],  #2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a1ptr],  #3, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a1ptr],  #4, mul vl]     \n"
      "   udot    z26.s, z1.b, z0.b[0]                      \n"
      "   udot    z27.s, z2.b, z0.b[0]                      \n"
      "   udot    z28.s, z3.b, z0.b[0]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #6, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #7, mul vl]     \n"
      "   udot    z29.s, z4.b, z0.b[0]                      \n"
      "   udot    z30.s, z5.b, z0.b[0]                      \n"
      "   udot    z31.s, z6.b, z0.b[0]                      \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   addvl   %[a1ptr], %[a_ptr], #8                    \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-8, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-7, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr], #-6, mul vl]     \n"
      "   udot    z8.s, z1.b, z0.b[1]                       \n"
      "   udot    z9.s, z2.b, z0.b[1]                       \n"
      "   udot    z10.s, z3.b, z0.b[1]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr], #-5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr], #-4, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr], #-3, mul vl]     \n"
      "   udot    z11.s, z4.b, z0.b[1]                      \n"
      "   udot    z12.s, z5.b, z0.b[1]                      \n"
      "   udot    z13.s, z6.b, z0.b[1]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-1, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #0, mul vl]     \n"
      "   udot    z14.s, z1.b, z0.b[1]                      \n"
      "   udot    z15.s, z2.b, z0.b[1]                      \n"
      "   udot    z16.s, z3.b, z0.b[1]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr],  #1, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr],  #2, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr],  #3, mul vl]     \n"
      "   udot    z17.s, z4.b, z0.b[1]                      \n"
      "   udot    z18.s, z5.b, z0.b[1]                      \n"
      "   udot    z19.s, z6.b, z0.b[1]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr],  #4, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr],  #5, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #6, mul vl]     \n"
      "   udot    z20.s, z1.b, z0.b[1]                      \n"
      "   udot    z21.s, z2.b, z0.b[1]                      \n"
      "   udot    z22.s, z3.b, z0.b[1]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #-1, mul vl]    \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #0, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #1, mul vl]     \n"
      "   udot    z23.s, z4.b, z0.b[1]                      \n"
      "   udot    z24.s, z5.b, z0.b[1]                      \n"
      "   udot    z25.s, z6.b, z0.b[1]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a1ptr],  #2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a1ptr],  #3, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a1ptr],  #4, mul vl]     \n"
      "   udot    z26.s, z1.b, z0.b[1]                      \n"
      "   udot    z27.s, z2.b, z0.b[1]                      \n"
      "   udot    z28.s, z3.b, z0.b[1]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #6, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #7, mul vl]     \n"
      "   udot    z29.s, z4.b, z0.b[1]                      \n"
      "   udot    z30.s, z5.b, z0.b[1]                      \n"
      "   udot    z31.s, z6.b, z0.b[1]                      \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   addvl   %[a1ptr], %[a_ptr], #8                    \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-8, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-7, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr], #-6, mul vl]     \n"
      "   udot    z8.s, z1.b, z0.b[2]                       \n"
      "   udot    z9.s, z2.b, z0.b[2]                       \n"
      "   udot    z10.s, z3.b, z0.b[2]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr], #-5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr], #-4, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr], #-3, mul vl]     \n"
      "   udot    z11.s, z4.b, z0.b[2]                      \n"
      "   udot    z12.s, z5.b, z0.b[2]                      \n"
      "   udot    z13.s, z6.b, z0.b[2]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-1, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #0, mul vl]     \n"
      "   udot    z14.s, z1.b, z0.b[2]                      \n"
      "   udot    z15.s, z2.b, z0.b[2]                      \n"
      "   udot    z16.s, z3.b, z0.b[2]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr],  #1, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr],  #2, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr],  #3, mul vl]     \n"
      "   udot    z17.s, z4.b, z0.b[2]                      \n"
      "   udot    z18.s, z5.b, z0.b[2]                      \n"
      "   udot    z19.s, z6.b, z0.b[2]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr],  #4, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr],  #5, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #6, mul vl]     \n"
      "   udot    z20.s, z1.b, z0.b[2]                      \n"
      "   udot    z21.s, z2.b, z0.b[2]                      \n"
      "   udot    z22.s, z3.b, z0.b[2]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #-1, mul vl]    \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #0, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #1, mul vl]     \n"
      "   udot    z23.s, z4.b, z0.b[2]                      \n"
      "   udot    z24.s, z5.b, z0.b[2]                      \n"
      "   udot    z25.s, z6.b, z0.b[2]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a1ptr],  #2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a1ptr],  #3, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a1ptr],  #4, mul vl]     \n"
      "   udot    z26.s, z1.b, z0.b[2]                      \n"
      "   udot    z27.s, z2.b, z0.b[2]                      \n"
      "   udot    z28.s, z3.b, z0.b[2]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #6, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #7, mul vl]     \n"
      "   udot    z29.s, z4.b, z0.b[2]                      \n"
      "   udot    z30.s, z5.b, z0.b[2]                      \n"
      "   udot    z31.s, z6.b, z0.b[2]                      \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   addvl   %[a1ptr], %[a_ptr], #8                    \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-8, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-7, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr], #-6, mul vl]     \n"
      "   udot    z8.s, z1.b, z0.b[3]                       \n"
      "   udot    z9.s, z2.b, z0.b[3]                       \n"
      "   udot    z10.s, z3.b, z0.b[3]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr], #-5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr], #-4, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr], #-3, mul vl]     \n"
      "   udot    z11.s, z4.b, z0.b[3]                      \n"
      "   udot    z12.s, z5.b, z0.b[3]                      \n"
      "   udot    z13.s, z6.b, z0.b[3]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr], #-2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr], #-1, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #0, mul vl]     \n"
      "   udot    z14.s, z1.b, z0.b[3]                      \n"
      "   udot    z15.s, z2.b, z0.b[3]                      \n"
      "   udot    z16.s, z3.b, z0.b[3]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a_ptr],  #1, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a_ptr],  #2, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a_ptr],  #3, mul vl]     \n"
      "   udot    z17.s, z4.b, z0.b[3]                      \n"
      "   udot    z18.s, z5.b, z0.b[3]                      \n"
      "   udot    z19.s, z6.b, z0.b[3]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a_ptr],  #4, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a_ptr],  #5, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a_ptr],  #6, mul vl]     \n"
      "   udot    z20.s, z1.b, z0.b[3]                      \n"
      "   udot    z21.s, z2.b, z0.b[3]                      \n"
      "   udot    z22.s, z3.b, z0.b[3]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #-1, mul vl]    \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #0, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #1, mul vl]     \n"
      "   udot    z23.s, z4.b, z0.b[3]                      \n"
      "   udot    z24.s, z5.b, z0.b[3]                      \n"
      "   udot    z25.s, z6.b, z0.b[3]                      \n"

      "   ld1b    {z1.b}, p0/z, [%[a1ptr],  #2, mul vl]     \n"
      "   ld1b    {z2.b}, p0/z, [%[a1ptr],  #3, mul vl]     \n"
      "   ld1b    {z3.b}, p0/z, [%[a1ptr],  #4, mul vl]     \n"
      "   udot    z26.s, z1.b, z0.b[3]                      \n"
      "   udot    z27.s, z2.b, z0.b[3]                      \n"
      "   udot    z28.s, z3.b, z0.b[3]                      \n"

      "   ld1b    {z4.b}, p0/z, [%[a1ptr],  #5, mul vl]     \n"
      "   ld1b    {z5.b}, p0/z, [%[a1ptr],  #6, mul vl]     \n"
      "   ld1b    {z6.b}, p0/z, [%[a1ptr],  #7, mul vl]     \n"
      "   udot    z29.s, z4.b, z0.b[3]                      \n"
      "   udot    z30.s, z5.b, z0.b[3]                      \n"
      "   udot    z31.s, z6.b, z0.b[3]                      \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2          \n"
      "   add     %[n_idx], %[n_idx], #16                   \n"
      "   cmp     %[n_idx], %[n]                            \n"
      "   b.mi    2b                                        \n"

      // Store
      "   st1w    {z8.s}, p0, [%[c_dst], #-8, mul vl]       \n"
      "   st1w    {z9.s}, p0, [%[c_dst], #-7, mul vl]       \n"
      "   st1w    {z10.s}, p0, [%[c_dst], #-6, mul vl]      \n"
      "   st1w    {z11.s}, p0, [%[c_dst], #-5, mul vl]      \n"
      "   st1w    {z12.s}, p0, [%[c_dst], #-4, mul vl]      \n"
      "   st1w    {z13.s}, p0, [%[c_dst], #-3, mul vl]      \n"
      "   st1w    {z14.s}, p0, [%[c_dst], #-2, mul vl]      \n"
      "   st1w    {z15.s}, p0, [%[c_dst], #-1, mul vl]      \n"
      "   addvl   %[c_dst], %[c_dst], #8                    \n"

      "   st1w    {z16.s}, p0, [%[c_dst], #-8, mul vl]      \n"
      "   st1w    {z17.s}, p0, [%[c_dst], #-7, mul vl]      \n"
      "   st1w    {z18.s}, p0, [%[c_dst], #-6, mul vl]      \n"
      "   st1w    {z19.s}, p0, [%[c_dst], #-5, mul vl]      \n"
      "   st1w    {z20.s}, p0, [%[c_dst], #-4, mul vl]      \n"
      "   st1w    {z21.s}, p0, [%[c_dst], #-3, mul vl]      \n"
      "   st1w    {z22.s}, p0, [%[c_dst], #-2, mul vl]      \n"
      "   st1w    {z23.s}, p0, [%[c_dst], #-1, mul vl]      \n"
      "   st1w    {z24.s}, p0, [%[c_dst],  #0, mul vl]      \n"
      "   st1w    {z25.s}, p0, [%[c_dst],  #1, mul vl]      \n"
      "   st1w    {z26.s}, p0, [%[c_dst],  #2, mul vl]      \n"
      "   st1w    {z27.s}, p0, [%[c_dst],  #3, mul vl]      \n"
      "   st1w    {z28.s}, p0, [%[c_dst],  #4, mul vl]      \n"
      "   st1w    {z29.s}, p0, [%[c_dst],  #5, mul vl]      \n"
      "   st1w    {z30.s}, p0, [%[c_dst],  #6, mul vl]      \n"
      "   st1w    {z31.s}, p0, [%[c_dst],  #7, mul vl]      \n"
      "   addvl   %[c_dst], %[c_dst], #16                   \n"

      // M loop tail
      "   incb    %[m_idx], all, mul #6                     \n"
      "   cmp     %[m_idx], %[m]                            \n"
      "   b.mi    1b                                        \n"

      : [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [a_ptr] "=&r"(a_ptr),
        [a1ptr] "=&r"(a1ptr), [a_src] "+&r"(a), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z16", "z17",
        "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
        "z28", "z29", "z30", "z31", "p0", "cc", "memory");
}

#elif (defined(__ARM_NEON) && defined (__ARM_FEATURE_DOTPROD))

static void inner_loop_215(struct loop_215_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_cnd = (uint64_t)&(data->c[m]);
  register uint64_t n_cnd = (uint64_t)&(data->b[n]);

  register uint64_t a_ptr;
  register uint64_t a1ptr;
  register uint64_t b_ptr;

  asm volatile(
      // M loop head
      "1:                                                             \n"
      "   movi    v8.4s, #0                                           \n"
      "   movi    v9.4s, #0                                           \n"
      "   movi    v10.4s, #0                                          \n"
      "   movi    v11.4s, #0                                          \n"
      "   movi    v12.4s, #0                                          \n"
      "   movi    v13.4s, #0                                          \n"
      "   movi    v14.4s, #0                                          \n"
      "   movi    v15.4s, #0                                          \n"
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

      // N loop
      "   mov     %[a_ptr], %[a_src]                                  \n"
      "   mov     %[b_ptr], %[b_src]                                  \n"
      "2:                                                             \n"
      "   ldr     q0, [%[b_ptr]], #16                                 \n"
      "   mov     %[a1ptr], %[a_ptr]                                  \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v8.4s, v1.16b, v0.4b[0]                             \n"
      "   udot    v9.4s, v2.16b, v0.4b[0]                             \n"
      "   udot    v10.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v11.4s, v4.16b, v0.4b[0]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v12.4s, v1.16b, v0.4b[0]                            \n"
      "   udot    v13.4s, v2.16b, v0.4b[0]                            \n"
      "   udot    v14.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v15.4s, v4.16b, v0.4b[0]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v16.4s, v1.16b, v0.4b[0]                            \n"
      "   udot    v17.4s, v2.16b, v0.4b[0]                            \n"
      "   udot    v18.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v19.4s, v4.16b, v0.4b[0]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v20.4s, v1.16b, v0.4b[0]                            \n"
      "   udot    v21.4s, v2.16b, v0.4b[0]                            \n"
      "   udot    v22.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v23.4s, v4.16b, v0.4b[0]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v24.4s, v1.16b, v0.4b[0]                            \n"
      "   udot    v25.4s, v2.16b, v0.4b[0]                            \n"
      "   udot    v26.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v27.4s, v4.16b, v0.4b[0]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v28.4s, v1.16b, v0.4b[0]                            \n"
      "   udot    v29.4s, v2.16b, v0.4b[0]                            \n"
      "   udot    v30.4s, v3.16b, v0.4b[0]                            \n"
      "   udot    v31.4s, v4.16b, v0.4b[0]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   mov     %[a1ptr], %[a_ptr]                                  \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v8.4s, v1.16b, v0.4b[1]                             \n"
      "   udot    v9.4s, v2.16b, v0.4b[1]                             \n"
      "   udot    v10.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v11.4s, v4.16b, v0.4b[1]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v12.4s, v1.16b, v0.4b[1]                            \n"
      "   udot    v13.4s, v2.16b, v0.4b[1]                            \n"
      "   udot    v14.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v15.4s, v4.16b, v0.4b[1]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v16.4s, v1.16b, v0.4b[1]                            \n"
      "   udot    v17.4s, v2.16b, v0.4b[1]                            \n"
      "   udot    v18.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v19.4s, v4.16b, v0.4b[1]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v20.4s, v1.16b, v0.4b[1]                            \n"
      "   udot    v21.4s, v2.16b, v0.4b[1]                            \n"
      "   udot    v22.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v23.4s, v4.16b, v0.4b[1]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v24.4s, v1.16b, v0.4b[1]                            \n"
      "   udot    v25.4s, v2.16b, v0.4b[1]                            \n"
      "   udot    v26.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v27.4s, v4.16b, v0.4b[1]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v28.4s, v1.16b, v0.4b[1]                            \n"
      "   udot    v29.4s, v2.16b, v0.4b[1]                            \n"
      "   udot    v30.4s, v3.16b, v0.4b[1]                            \n"
      "   udot    v31.4s, v4.16b, v0.4b[1]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   mov     %[a1ptr], %[a_ptr]                                  \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v8.4s, v1.16b, v0.4b[2]                             \n"
      "   udot    v9.4s, v2.16b, v0.4b[2]                             \n"
      "   udot    v10.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v11.4s, v4.16b, v0.4b[2]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v12.4s, v1.16b, v0.4b[2]                            \n"
      "   udot    v13.4s, v2.16b, v0.4b[2]                            \n"
      "   udot    v14.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v15.4s, v4.16b, v0.4b[2]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v16.4s, v1.16b, v0.4b[2]                            \n"
      "   udot    v17.4s, v2.16b, v0.4b[2]                            \n"
      "   udot    v18.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v19.4s, v4.16b, v0.4b[2]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v20.4s, v1.16b, v0.4b[2]                            \n"
      "   udot    v21.4s, v2.16b, v0.4b[2]                            \n"
      "   udot    v22.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v23.4s, v4.16b, v0.4b[2]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v24.4s, v1.16b, v0.4b[2]                            \n"
      "   udot    v25.4s, v2.16b, v0.4b[2]                            \n"
      "   udot    v26.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v27.4s, v4.16b, v0.4b[2]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v28.4s, v1.16b, v0.4b[2]                            \n"
      "   udot    v29.4s, v2.16b, v0.4b[2]                            \n"
      "   udot    v30.4s, v3.16b, v0.4b[2]                            \n"
      "   udot    v31.4s, v4.16b, v0.4b[2]                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   mov     %[a1ptr], %[a_ptr]                                  \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v8.4s, v1.16b, v0.4b[3]                             \n"
      "   udot    v9.4s, v2.16b, v0.4b[3]                             \n"
      "   udot    v10.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v11.4s, v4.16b, v0.4b[3]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v12.4s, v1.16b, v0.4b[3]                            \n"
      "   udot    v13.4s, v2.16b, v0.4b[3]                            \n"
      "   udot    v14.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v15.4s, v4.16b, v0.4b[3]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v16.4s, v1.16b, v0.4b[3]                            \n"
      "   udot    v17.4s, v2.16b, v0.4b[3]                            \n"
      "   udot    v18.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v19.4s, v4.16b, v0.4b[3]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v20.4s, v1.16b, v0.4b[3]                            \n"
      "   udot    v21.4s, v2.16b, v0.4b[3]                            \n"
      "   udot    v22.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v23.4s, v4.16b, v0.4b[3]                            \n"

      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v24.4s, v1.16b, v0.4b[3]                            \n"
      "   udot    v25.4s, v2.16b, v0.4b[3]                            \n"
      "   udot    v26.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v27.4s, v4.16b, v0.4b[3]                            \n"
      "   ld1     {v1.16b,v2.16b,v3.16b,v4.16b}, [%[a1ptr]], #64      \n"
      "   udot    v28.4s, v1.16b, v0.4b[3]                            \n"
      "   udot    v29.4s, v2.16b, v0.4b[3]                            \n"
      "   udot    v30.4s, v3.16b, v0.4b[3]                            \n"
      "   udot    v31.4s, v4.16b, v0.4b[3]                            \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   cmp     %[b_ptr], %[n_cnd]                                  \n"
      "   b.mi    2b                                                  \n"

      // Store
      "   st1     {v8.4s,v9.4s,v10.4s,v11.4s}, [%[c_dst]], #64        \n"
      "   st1     {v12.4s,v13.4s,v14.4s,v15.4s}, [%[c_dst]], #64      \n"
      "   st1     {v16.4s,v17.4s,v18.4s,v19.4s}, [%[c_dst]], #64      \n"
      "   st1     {v20.4s,v21.4s,v22.4s,v23.4s}, [%[c_dst]], #64      \n"
      "   st1     {v24.4s,v25.4s,v26.4s,v27.4s}, [%[c_dst]], #64      \n"
      "   st1     {v28.4s,v29.4s,v30.4s,v31.4s}, [%[c_dst]], #64      \n"

      // M loop tail
      "   add     %[a_src], %[a_src], #384                            \n"
      "   cmp     %[c_dst], %[m_cnd]                                  \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [a1ptr] "=&r"(a1ptr), [a_src] "+&r"(a),
        [b_ptr] "=&r"(b_ptr), [c_dst] "+&r"(c)
      : [m_cnd] "r"(m_cnd), [n_cnd] "r"(n_cnd), [m] "r"(m), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31", "cc", "memory");
}

#else

static void inner_loop_215(struct loop_215_data *data) {
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
// Default of 257KiB equates to original problem size (M=1024, N=256)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n) ((n) * ((m) + 1) * sizeof(uint8_t))

LOOP_DECL(215, OUTER_LOOP_ATTR) {
  // Work out values for M and N to fit within problem size limit
  uint64_t M = MAX_VL / 2;  // multiple of 16*SVLs
  uint64_t N = 0;           // multiple of 16

  const uint64_t N_base = 16;
#if !defined(__ARM_FEATURE_SME2)
  // Scale M dimension for sve or neon versions only
  uint64_t m_inc = 24 * (get_vl() / 32);
  M = m_inc + (((uint64_t) M / m_inc) * m_inc);
#endif

  while (true) {
    uint64_t n = N + N_base;
    if (PROBLEM_SIZE_ACTUAL(M, n) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      N = n;
    } else {
      break;
    }
  }

  // increasing loop iterations
  iters *= 10;

  struct loop_215_data data = {
      .m = M,
      .n = N,
  };
  ALLOC_64B(data.a, M * N, "A matrix");
  ALLOC_64B(data.b, N * 1, "x vector");
  ALLOC_64B(data.c, M * 1, "b vector");

  fill_uint8(data.a, M * N);
  fill_uint8(data.b, N * 1);

  inner_loops_215(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x 1\n", M, N, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(j)                                                         \
  {                                                                      \
    uint32_t d = 0;                                                      \
    for (int n = 0; n < N; n++) MLA(d, data.a[ZIP(M, n, j)], data.b[n]); \
    checksum += (int)(d != data.c[j]);                                   \
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
  FINALISE_LOOP_I(215, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
