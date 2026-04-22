/*----------------------------------------------------------------------------
#
#   Loop 217: INT8-INT32 row-major matrix-vector multiply
#
#   Purpose:
#     Use of i8 to i32 DOT and ADDV instructions.
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
    B: row-vector
    C: row-vector
  Constraints -
    M: multiple of 8
    N: multiple of 4*SVLb
*/

struct loop_217_data {
  uint64_t m;
  uint64_t n;
  uint8_t *restrict a;
  uint8_t *restrict b;
  uint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_217(struct loop_217_data *restrict data) {
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

static void inner_loop_217(struct loop_217_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  uint32_t *restrict c = data->c;
  for (uint64_t y = 0; y < m; y++) {
    uint32_t d = 0;
    for (uint64_t x = 0; x < n; x++) MLA(d, a[y * n + x], b[x]);
    c[y] = d;
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_217(struct loop_217_data *data) LOOP_ATTR {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint64_t svl_s = svcntw();

  uint8_t *a_ptr;
  uint32_t *c_ptr = c;
  uint64_t l_cnd, l_idx, o_idx, n_idx;

  svcount_t c_all = svptrue_c8();
  svbool_t p_all = svptrue_b8();

  svuint8x4_t ldb;
  svuint8x4_t lda_0, lda_1, lda_2, lda_3;
  svuint64x2_t stc_0, stc_1, stc_2, stc_3;
  svuint32_t res_0, res_1, res_2, res_3;

#define CAST(i, j) svreinterpret_u32(svget2(stc_##i, j))
#define PAIR(i) svcreate2(CAST(i, 0), CAST(i, 1))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif
  for (int64_t m_rem = m; m_rem > 0; m_rem -= l_cnd) {
#if !defined(__ARM_FEATURE_SME2p1)
    svzero_za();
#endif
    l_cnd = (m_rem < svl_s) ? m_rem : svl_s;

    for (n_idx = 0; n_idx < n; n_idx += 4 * svcntb()) {
      ldb = svld1_x4(c_all, &b[n_idx]);
      a_ptr = &a[n_idx];
      for (l_idx = 0; l_idx < l_cnd; l_idx += 4, a_ptr += n * 4) {
        lda_0 = svld1_x4(c_all, &a_ptr[n * 0]);
        lda_1 = svld1_x4(c_all, &a_ptr[n * 1]);
        lda_2 = svld1_x4(c_all, &a_ptr[n * 2]);
        lda_3 = svld1_x4(c_all, &a_ptr[n * 3]);
        svdot_za32_vg1x4(l_idx + 0, lda_0, ldb);
        svdot_za32_vg1x4(l_idx + 1, lda_1, ldb);
        svdot_za32_vg1x4(l_idx + 2, lda_2, ldb);
        svdot_za32_vg1x4(l_idx + 3, lda_3, ldb);
      }
    }

    for (l_idx = 0, o_idx = svcntw(); l_idx < l_cnd; l_idx += 4, o_idx += 4) {
#if defined(__ARM_FEATURE_SME2p1)
      stc_0 = svreadz_za64_u64_vg1x2(o_idx + 0);
      stc_1 = svreadz_za64_u64_vg1x2(o_idx + 1);
      stc_2 = svreadz_za64_u64_vg1x2(o_idx + 2);
      stc_3 = svreadz_za64_u64_vg1x2(o_idx + 3);
#else
      stc_0 = svread_za64_u64_vg1x2(o_idx + 0);
      stc_1 = svread_za64_u64_vg1x2(o_idx + 1);
      stc_2 = svread_za64_u64_vg1x2(o_idx + 2);
      stc_3 = svread_za64_u64_vg1x2(o_idx + 3);
#endif
      svadd_za32_vg1x2(l_idx + 0, PAIR(0));
      svadd_za32_vg1x2(l_idx + 1, PAIR(1));
      svadd_za32_vg1x2(l_idx + 2, PAIR(2));
      svadd_za32_vg1x2(l_idx + 3, PAIR(3));
#if defined(__ARM_FEATURE_SME2p1)
      stc_0 = svreadz_za64_u64_vg1x2(l_idx + 0);
      stc_1 = svreadz_za64_u64_vg1x2(l_idx + 1);
      stc_2 = svreadz_za64_u64_vg1x2(l_idx + 2);
      stc_3 = svreadz_za64_u64_vg1x2(l_idx + 3);
#else
      stc_0 = svread_za64_u64_vg1x2(l_idx + 0);
      stc_1 = svread_za64_u64_vg1x2(l_idx + 1);
      stc_2 = svread_za64_u64_vg1x2(l_idx + 2);
      stc_3 = svread_za64_u64_vg1x2(l_idx + 3);
#endif
      res_0 = svadd_m(p_all, CAST(0, 0), CAST(0, 1));
      res_1 = svadd_m(p_all, CAST(1, 0), CAST(1, 1));
      res_2 = svadd_m(p_all, CAST(2, 0), CAST(2, 1));
      res_3 = svadd_m(p_all, CAST(3, 0), CAST(3, 1));
      c_ptr[l_idx + 0] = svaddv(p_all, res_0);
      c_ptr[l_idx + 1] = svaddv(p_all, res_1);
      c_ptr[l_idx + 2] = svaddv(p_all, res_2);
      c_ptr[l_idx + 3] = svaddv(p_all, res_3);
    }
    a += n * svl_s;
    c_ptr += svl_s;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_217(struct loop_217_data *data) LOOP_ATTR {
  uint64_t m = data->m;
  uint64_t n = data->n;

  uint8_t *a = data->a, *a_ptr;
  uint8_t *b = data->b, *b_ptr;
  uint32_t *c = data->c, *c_ptr;

  svbool_t p_all = svptrue_b8();

  svuint32_t acc_00, acc_01, acc_02, acc_03;
  svuint32_t acc_10, acc_11, acc_12, acc_13;
  svuint8x4_t lda_0, lda_1, lda_2, lda_3;
  svuint8x4_t ldx;

#define GETA(q, p) svget4(lda_##q, p)
#define GETX(p) svget4(ldx, p)

#define UDOT(s, q, p) acc_##s##q = svdot(acc_##s##q, GETA(q, p), GETX(p));
#define UDOT_QUAD(s, q) UDOT(s, q, 0) UDOT(s, q, 1) UDOT(s, q, 2) UDOT(s, q, 3)

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c8();
#define LOAD_QUAD(s, q) lda_##q = svld1_x4(c_all, &a_ptr[(4 * s + q) * n])
#define LOADX_QUAD ldx = svld1_x4(c_all, b_ptr)
#else
#define LOAD(s, q, p) svld1_vnum(p_all, &a_ptr[(4 * s + q) * n], p)
#define LOAD_QUAD(s, q) \
  lda_##q =             \
      svcreate4(LOAD(s, q, 0), LOAD(s, q, 1), LOAD(s, q, 2), LOAD(s, q, 3))
#define LOADX(p) svld1_vnum(p_all, b_ptr, p)
#define LOADX_QUAD ldx = svcreate4(LOADX(0), LOADX(1), LOADX(2), LOADX(3))
#endif

  for (c_ptr = c; c_ptr < &c[m]; c_ptr += 8, a += 8 * n) {
    acc_00 = svdup_u32(0);
    acc_01 = svdup_u32(0);
    acc_02 = svdup_u32(0);
    acc_03 = svdup_u32(0);
    acc_10 = svdup_u32(0);
    acc_11 = svdup_u32(0);
    acc_12 = svdup_u32(0);
    acc_13 = svdup_u32(0);

    a_ptr = a;
    b_ptr = b;
    while (b_ptr < &b[n]) {
      LOADX_QUAD;

      LOAD_QUAD(0, 0);
      LOAD_QUAD(0, 1);
      LOAD_QUAD(0, 2);
      LOAD_QUAD(0, 3);
      UDOT_QUAD(0, 0);
      UDOT_QUAD(0, 1);
      UDOT_QUAD(0, 2);
      UDOT_QUAD(0, 3);

      LOAD_QUAD(1, 0);
      LOAD_QUAD(1, 1);
      LOAD_QUAD(1, 2);
      LOAD_QUAD(1, 3);
      UDOT_QUAD(1, 0);
      UDOT_QUAD(1, 1);
      UDOT_QUAD(1, 2);
      UDOT_QUAD(1, 3);

      a_ptr += 4 * svcntb();
      b_ptr += 4 * svcntb();
    }

    c_ptr[0] = svaddv(p_all, acc_00);
    c_ptr[1] = svaddv(p_all, acc_01);
    c_ptr[2] = svaddv(p_all, acc_02);
    c_ptr[3] = svaddv(p_all, acc_03);
    c_ptr[4] = svaddv(p_all, acc_10);
    c_ptr[5] = svaddv(p_all, acc_11);
    c_ptr[6] = svaddv(p_all, acc_12);
    c_ptr[7] = svaddv(p_all, acc_13);
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_217(struct loop_217_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t a2off = n * 2;
  register uint64_t a3off = n * 3;
  register uint64_t a_blk = n * svl_s;

  register uint64_t l_cnd;
  register uint64_t a_ptr;
  register uint64_t n_idx;
  register uint64_t m_rem = m;
  // x9: slice index register for udot and mova
  // x10: second slice index register for mova

  asm volatile(
      "   ptrue   pn8.b                                               \n"
      "   ptrue   p0.s                                                \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      // M loop head
      "1:                                                             \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                \n"
#endif
      "   cmp     %[m_rem], %[svl_s]                                  \n"
      "   csel    %[l_cnd], %[m_rem], %[svl_s], lt                    \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"
      "   ld1b    {z0.b-z3.b}, pn8/z, [%[b_src], %[n_idx]]            \n"

      // M-block dot-product loop
      "   add     %[a_ptr], %[a_src], %[n_idx]                        \n"
      "   mov     x9, #0                                              \n"
      "3:                                                             \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                    \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], %[a1off]]          \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], %[a2off]]          \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], %[a3off]]          \n"
      "   add     %[a_ptr], %[a_ptr], %[n], lsl #2                    \n"
      "   udot    za.s[w9, 0, vgx4], {z16.b-z19.b}, {z0.b-z3.b}\n"
      "   udot    za.s[w9, 1, vgx4], {z20.b-z23.b}, {z0.b-z3.b}\n"
      "   udot    za.s[w9, 2, vgx4], {z24.b-z27.b}, {z0.b-z3.b}\n"
      "   udot    za.s[w9, 3, vgx4], {z28.b-z31.b}, {z0.b-z3.b}\n"
      "   add     x9, x9, #4                                          \n"
      "   cmp     x9, %[l_cnd]                                        \n"
      "   b.mi    3b                                                  \n"

      // N loop tail
      "   addvl   %[n_idx], %[n_idx], #4                              \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // Store loop
      "   cntw    x10                                                 \n"
      "   mov     x9, #0                                              \n"
      "4:                                                             \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.s-z1.s}, za.s[w10, 0, vgx2]                     \n"
      "   movaz   {z2.s-z3.s}, za.s[w10, 1, vgx2]                     \n"
      "   movaz   {z4.s-z5.s}, za.s[w10, 2, vgx2]                     \n"
      "   movaz   {z6.s-z7.s}, za.s[w10, 3, vgx2]                     \n"
#else
      "   mova    {z0.s-z1.s}, za.s[w10, 0, vgx2]                     \n"
      "   mova    {z2.s-z3.s}, za.s[w10, 1, vgx2]                     \n"
      "   mova    {z4.s-z5.s}, za.s[w10, 2, vgx2]                     \n"
      "   mova    {z6.s-z7.s}, za.s[w10, 3, vgx2]                     \n"
#endif
      "   add     za.s[w9, 0, vgx2], {z0.s-z1.s}                      \n"
      "   add     za.s[w9, 1, vgx2], {z2.s-z3.s}                      \n"
      "   add     za.s[w9, 2, vgx2], {z4.s-z5.s}                      \n"
      "   add     za.s[w9, 3, vgx2], {z6.s-z7.s}                      \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.s-z1.s}, za.s[w9, 0, vgx2]                      \n"
      "   movaz   {z2.s-z3.s}, za.s[w9, 1, vgx2]                      \n"
      "   movaz   {z4.s-z5.s}, za.s[w9, 2, vgx2]                      \n"
      "   movaz   {z6.s-z7.s}, za.s[w9, 3, vgx2]                      \n"
#else
      "   mova    {z0.s-z1.s}, za.s[w9, 0, vgx2]                      \n"
      "   mova    {z2.s-z3.s}, za.s[w9, 1, vgx2]                      \n"
      "   mova    {z4.s-z5.s}, za.s[w9, 2, vgx2]                      \n"
      "   mova    {z6.s-z7.s}, za.s[w9, 3, vgx2]                      \n"
#endif
      "   add     z20.s, z0.s, z1.s                                   \n"
      "   add     z21.s, z2.s, z3.s                                   \n"
      "   add     z22.s, z4.s, z5.s                                   \n"
      "   add     z23.s, z6.s, z7.s                                   \n"
      "   uaddv   d0, p0, z20.s                                       \n"
      "   uaddv   d1, p0, z21.s                                       \n"
      "   uaddv   d2, p0, z22.s                                       \n"
      "   uaddv   d3, p0, z23.s                                       \n"
      "   stp     s0, s1, [%[c_dst]], #8                              \n"
      "   stp     s2, s3, [%[c_dst]], #8                              \n"
      "   add     x9, x9, #4                                          \n"
      "   add     x10, x10, #4                                        \n"
      "   cmp     x9, %[l_cnd]                                        \n"
      "   b.mi    4b                                                  \n"

      // M loop tail
      "   sub     %[m_rem], %[m_rem], %[l_cnd]                        \n"
      "   add     %[a_src], %[a_src], %[a_blk]                        \n"
      "   cmp     %[m_rem], #0                                        \n"
      "   b.gt    1b                                                  \n"

      : [n_idx] "=&r"(n_idx), [l_cnd] "=&r"(l_cnd), [a_ptr] "=&r"(a_ptr),
        [a_src] "+&r"(a), [c_dst] "+&r"(c), [m_rem] "+&r"(m_rem)
      : [a2off] "r"(a2off), [a3off] "r"(a3off), [a1off] "r"(n),
        [a_blk] "r"(a_blk), [svl_s] "r"(svl_s), [b_src] "r"(b), [n] "r"(n)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18",
        "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
        "z29", "z30", "z31", "p0", "p8", "x9", "x10",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_217(struct loop_217_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t n_cnd = (uint64_t)(&data->b[n]);
  register uint64_t a2off = n * 2;
  register uint64_t a3off = n * 3;
  register uint64_t a_inc;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.b                                            \n"
      "   ptrue   pn8.b                                           \n"
      "   addvl   %[a_inc], %[n], #-1                             \n"

      // M loop head
      "   mov     %[m_idx], #0                                    \n"
      "1:                                                         \n"
      "   mov     z0.s, #0                                        \n"
      "   mov     z1.s, #0                                        \n"
      "   mov     z2.s, #0                                        \n"
      "   mov     z3.s, #0                                        \n"
      "   mov     z4.s, #0                                        \n"
      "   mov     z5.s, #0                                        \n"
      "   mov     z6.s, #0                                        \n"
      "   mov     z7.s, #0                                        \n"

      // N loop
      "   mov     %[b_ptr], %[b_src]                              \n"
      "   mov     %[a_ptr], %[a_src]                              \n"
      "2:                                                         \n"
      "   ld1b    {z12.b-z15.b}, pn8/z, [%[b_ptr]]                \n"
      "   addvl   %[b_ptr], %[b_ptr], #4                          \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], %[a1off]]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], %[a2off]]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], %[a3off]]      \n"
      "   add     %[a_ptr], %[a_ptr], %[n], lsl #2                \n"
      "   udot    z0.s, z16.b, z12.b                              \n"
      "   udot    z1.s, z20.b, z12.b                              \n"
      "   udot    z2.s, z24.b, z12.b                              \n"
      "   udot    z3.s, z28.b, z12.b                              \n"
      "   udot    z0.s, z17.b, z13.b                              \n"
      "   udot    z1.s, z21.b, z13.b                              \n"
      "   udot    z2.s, z25.b, z13.b                              \n"
      "   udot    z3.s, z29.b, z13.b                              \n"
      "   udot    z0.s, z18.b, z14.b                              \n"
      "   udot    z1.s, z22.b, z14.b                              \n"
      "   udot    z2.s, z26.b, z14.b                              \n"
      "   udot    z3.s, z30.b, z14.b                              \n"
      "   udot    z0.s, z19.b, z15.b                              \n"
      "   udot    z1.s, z23.b, z15.b                              \n"
      "   udot    z2.s, z27.b, z15.b                              \n"
      "   udot    z3.s, z31.b, z15.b                              \n"
      "   ld1b    {z16.b-z19.b}, pn8/z, [%[a_ptr]]                \n"
      "   ld1b    {z20.b-z23.b}, pn8/z, [%[a_ptr], %[a1off]]      \n"
      "   ld1b    {z24.b-z27.b}, pn8/z, [%[a_ptr], %[a2off]]      \n"
      "   ld1b    {z28.b-z31.b}, pn8/z, [%[a_ptr], %[a3off]]      \n"
      "   sub     %[a_ptr], %[a_ptr], %[a_inc], lsl #2            \n"
      "   udot    z4.s, z16.b, z12.b                              \n"
      "   udot    z5.s, z20.b, z12.b                              \n"
      "   udot    z6.s, z24.b, z12.b                              \n"
      "   udot    z7.s, z28.b, z12.b                              \n"
      "   udot    z4.s, z17.b, z13.b                              \n"
      "   udot    z5.s, z21.b, z13.b                              \n"
      "   udot    z6.s, z25.b, z13.b                              \n"
      "   udot    z7.s, z29.b, z13.b                              \n"
      "   udot    z4.s, z18.b, z14.b                              \n"
      "   udot    z5.s, z22.b, z14.b                              \n"
      "   udot    z6.s, z26.b, z14.b                              \n"
      "   udot    z7.s, z30.b, z14.b                              \n"
      "   udot    z4.s, z19.b, z15.b                              \n"
      "   udot    z5.s, z23.b, z15.b                              \n"
      "   udot    z6.s, z27.b, z15.b                              \n"
      "   udot    z7.s, z31.b, z15.b                              \n"
      "   cmp     %[b_ptr], %[n_cnd]                              \n"
      "   b.mi    2b                                              \n"

      // Reduce and store
      "   uaddv   d0, p0, z0.s                                    \n"
      "   uaddv   d1, p0, z1.s                                    \n"
      "   uaddv   d2, p0, z2.s                                    \n"
      "   uaddv   d3, p0, z3.s                                    \n"
      "   uaddv   d4, p0, z4.s                                    \n"
      "   uaddv   d5, p0, z5.s                                    \n"
      "   uaddv   d6, p0, z6.s                                    \n"
      "   uaddv   d7, p0, z7.s                                    \n"
      "   stp     s0, s1, [%[c0dst]], #8                          \n"
      "   stp     s2, s3, [%[c0dst]], #8                          \n"
      "   stp     s4, s5, [%[c0dst]], #8                          \n"
      "   stp     s6, s7, [%[c0dst]], #8                          \n"

      // M loop tail
      "   add     %[m_idx], %[m_idx], #8                          \n"
      "   add     %[a_src], %[a_src], %[n], lsl #3                \n"
      "   cmp     %[m_idx], %[m]                                  \n"
      "   b.mi    1b                                              \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [a_inc] "=&r"(a_inc), [a_src] "+&r"(a), [c0dst] "+&r"(c)
      : [a2off] "r"(a2off), [a3off] "r"(a3off), [a1off] "r"(n), [b_src] "r"(b),
        [m] "r"(m), [n] "r"(n), [n_cnd] "r"(n_cnd)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z12", "z13", "z14",
        "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p8", "cc",
        "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_217(struct loop_217_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t n_cnd = (uint64_t)(&data->b[n]);
  register uint64_t a3off = n * 3;
  register uint64_t a_inc;

  register uint64_t a0ptr;
  register uint64_t a1ptr;
  register uint64_t a2ptr;
  register uint64_t a3ptr;
  register uint64_t b_ptr;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.b                                            \n"
      "   addvl   %[a_inc], %[n], #-1                             \n"

      // M loop head
      "   mov     %[m_idx], #0                                    \n"
      "1:                                                         \n"
      "   mov     z0.s, #0                                        \n"
      "   mov     z1.s, #0                                        \n"
      "   mov     z2.s, #0                                        \n"
      "   mov     z3.s, #0                                        \n"
      "   mov     z4.s, #0                                        \n"
      "   mov     z5.s, #0                                        \n"
      "   mov     z6.s, #0                                        \n"
      "   mov     z7.s, #0                                        \n"

      // N loop
      "   mov     %[b_ptr], %[b_src]                              \n"
      "   mov     %[a0ptr], %[a_src]                              \n"
      "   add     %[a1ptr], %[a_src], %[n]                        \n"
      "   add     %[a2ptr], %[a_src], %[n], lsl #1                \n"
      "   add     %[a3ptr], %[a_src], %[a3off]                    \n"
      "2:                                                         \n"
      "   ld1b    {z12.b}, p0/z, [%[b_ptr], #0, mul vl]           \n"
      "   ld1b    {z13.b}, p0/z, [%[b_ptr], #1, mul vl]           \n"
      "   ld1b    {z14.b}, p0/z, [%[b_ptr], #2, mul vl]           \n"
      "   ld1b    {z15.b}, p0/z, [%[b_ptr], #3, mul vl]           \n"
      "   addvl   %[b_ptr], %[b_ptr], #4                          \n"
      "   ld1b    {z16.b}, p0/z, [%[a0ptr], #0, mul vl]           \n"
      "   ld1b    {z17.b}, p0/z, [%[a0ptr], #1, mul vl]           \n"
      "   ld1b    {z18.b}, p0/z, [%[a0ptr], #2, mul vl]           \n"
      "   ld1b    {z19.b}, p0/z, [%[a0ptr], #3, mul vl]           \n"
      "   ld1b    {z20.b}, p0/z, [%[a1ptr], #0, mul vl]           \n"
      "   ld1b    {z21.b}, p0/z, [%[a1ptr], #1, mul vl]           \n"
      "   ld1b    {z22.b}, p0/z, [%[a1ptr], #2, mul vl]           \n"
      "   ld1b    {z23.b}, p0/z, [%[a1ptr], #3, mul vl]           \n"
      "   ld1b    {z24.b}, p0/z, [%[a2ptr], #0, mul vl]           \n"
      "   ld1b    {z25.b}, p0/z, [%[a2ptr], #1, mul vl]           \n"
      "   ld1b    {z26.b}, p0/z, [%[a2ptr], #2, mul vl]           \n"
      "   ld1b    {z27.b}, p0/z, [%[a2ptr], #3, mul vl]           \n"
      "   ld1b    {z28.b}, p0/z, [%[a3ptr], #0, mul vl]           \n"
      "   ld1b    {z29.b}, p0/z, [%[a3ptr], #1, mul vl]           \n"
      "   ld1b    {z30.b}, p0/z, [%[a3ptr], #2, mul vl]           \n"
      "   ld1b    {z31.b}, p0/z, [%[a3ptr], #3, mul vl]           \n"
      "   add     %[a0ptr], %[a0ptr], %[n], lsl #2                \n"
      "   add     %[a1ptr], %[a1ptr], %[n], lsl #2                \n"
      "   add     %[a2ptr], %[a2ptr], %[n], lsl #2                \n"
      "   add     %[a3ptr], %[a3ptr], %[n], lsl #2                \n"
      "   udot    z0.s, z16.b, z12.b                              \n"
      "   udot    z1.s, z20.b, z12.b                              \n"
      "   udot    z2.s, z24.b, z12.b                              \n"
      "   udot    z3.s, z28.b, z12.b                              \n"
      "   udot    z0.s, z17.b, z13.b                              \n"
      "   udot    z1.s, z21.b, z13.b                              \n"
      "   udot    z2.s, z25.b, z13.b                              \n"
      "   udot    z3.s, z29.b, z13.b                              \n"
      "   udot    z0.s, z18.b, z14.b                              \n"
      "   udot    z1.s, z22.b, z14.b                              \n"
      "   udot    z2.s, z26.b, z14.b                              \n"
      "   udot    z3.s, z30.b, z14.b                              \n"
      "   udot    z0.s, z19.b, z15.b                              \n"
      "   udot    z1.s, z23.b, z15.b                              \n"
      "   udot    z2.s, z27.b, z15.b                              \n"
      "   udot    z3.s, z31.b, z15.b                              \n"
      "   ld1b    {z16.b}, p0/z, [%[a0ptr], #0, mul vl]           \n"
      "   ld1b    {z17.b}, p0/z, [%[a0ptr], #1, mul vl]           \n"
      "   ld1b    {z18.b}, p0/z, [%[a0ptr], #2, mul vl]           \n"
      "   ld1b    {z19.b}, p0/z, [%[a0ptr], #3, mul vl]           \n"
      "   ld1b    {z20.b}, p0/z, [%[a1ptr], #0, mul vl]           \n"
      "   ld1b    {z21.b}, p0/z, [%[a1ptr], #1, mul vl]           \n"
      "   ld1b    {z22.b}, p0/z, [%[a1ptr], #2, mul vl]           \n"
      "   ld1b    {z23.b}, p0/z, [%[a1ptr], #3, mul vl]           \n"
      "   ld1b    {z24.b}, p0/z, [%[a2ptr], #0, mul vl]           \n"
      "   ld1b    {z25.b}, p0/z, [%[a2ptr], #1, mul vl]           \n"
      "   ld1b    {z26.b}, p0/z, [%[a2ptr], #2, mul vl]           \n"
      "   ld1b    {z27.b}, p0/z, [%[a2ptr], #3, mul vl]           \n"
      "   ld1b    {z28.b}, p0/z, [%[a3ptr], #0, mul vl]           \n"
      "   ld1b    {z29.b}, p0/z, [%[a3ptr], #1, mul vl]           \n"
      "   ld1b    {z30.b}, p0/z, [%[a3ptr], #2, mul vl]           \n"
      "   ld1b    {z31.b}, p0/z, [%[a3ptr], #3, mul vl]           \n"
      "   sub     %[a0ptr], %[a0ptr], %[a_inc], lsl #2            \n"
      "   sub     %[a1ptr], %[a1ptr], %[a_inc], lsl #2            \n"
      "   sub     %[a2ptr], %[a2ptr], %[a_inc], lsl #2            \n"
      "   sub     %[a3ptr], %[a3ptr], %[a_inc], lsl #2            \n"
      "   udot    z4.s, z16.b, z12.b                              \n"
      "   udot    z5.s, z20.b, z12.b                              \n"
      "   udot    z6.s, z24.b, z12.b                              \n"
      "   udot    z7.s, z28.b, z12.b                              \n"
      "   udot    z4.s, z17.b, z13.b                              \n"
      "   udot    z5.s, z21.b, z13.b                              \n"
      "   udot    z6.s, z25.b, z13.b                              \n"
      "   udot    z7.s, z29.b, z13.b                              \n"
      "   udot    z4.s, z18.b, z14.b                              \n"
      "   udot    z5.s, z22.b, z14.b                              \n"
      "   udot    z6.s, z26.b, z14.b                              \n"
      "   udot    z7.s, z30.b, z14.b                              \n"
      "   udot    z4.s, z19.b, z15.b                              \n"
      "   udot    z5.s, z23.b, z15.b                              \n"
      "   udot    z6.s, z27.b, z15.b                              \n"
      "   udot    z7.s, z31.b, z15.b                              \n"
      "   cmp     %[b_ptr], %[n_cnd]                              \n"
      "   b.mi    2b                                              \n"

      // Reduce and store
      "   uaddv   d0, p0, z0.s                                    \n"
      "   uaddv   d1, p0, z1.s                                    \n"
      "   uaddv   d2, p0, z2.s                                    \n"
      "   uaddv   d3, p0, z3.s                                    \n"
      "   uaddv   d4, p0, z4.s                                    \n"
      "   uaddv   d5, p0, z5.s                                    \n"
      "   uaddv   d6, p0, z6.s                                    \n"
      "   uaddv   d7, p0, z7.s                                    \n"
      "   stp     s0, s1, [%[c0dst]], #8                          \n"
      "   stp     s2, s3, [%[c0dst]], #8                          \n"
      "   stp     s4, s5, [%[c0dst]], #8                          \n"
      "   stp     s6, s7, [%[c0dst]], #8                          \n"

      // M loop tail
      "   add     %[m_idx], %[m_idx], #8                          \n"
      "   add     %[a_src], %[a_src], %[n], lsl #3                \n"
      "   cmp     %[m_idx], %[m]                                  \n"
      "   b.mi    1b                                              \n"

      : [a0ptr] "=&r"(a0ptr), [a1ptr] "=&r"(a1ptr), [m_idx] "=&r"(m_idx),
        [a2ptr] "=&r"(a2ptr), [a3ptr] "=&r"(a3ptr), [b_ptr] "=&r"(b_ptr),
        [a_inc] "=&r"(a_inc), [a_src] "+&r"(a), [c0dst] "+&r"(c)
      : [b_src] "r"(b), [m] "r"(m), [n] "r"(n),
        [n_cnd] "r"(n_cnd), [a3off] "r"(a3off)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z12", "z13", "z14",
        "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "cc", "memory");
}

#elif (defined(__ARM_NEON) && defined (__ARM_FEATURE_DOTPROD))

static void inner_loop_217(struct loop_217_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_cnd = (uint64_t)&(data->c[m]);
  register uint64_t n_cnd = (uint64_t)&(data->b[n]);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t a_bar;

  asm volatile(
      // M loop head
      "1:                                                             \n"
      "   movi    v0.4s, #0                                           \n"
      "   movi    v1.4s, #0                                           \n"
      "   movi    v2.4s, #0                                           \n"
      "   movi    v3.4s, #0                                           \n"
      "   movi    v4.4s, #0                                           \n"
      "   movi    v5.4s, #0                                           \n"
      "   movi    v6.4s, #0                                           \n"
      "   movi    v7.4s, #0                                           \n"

      // N loop
      "   mov     %[a_bar], %[a_src]                                  \n"
      "   mov     %[b_ptr], %[b_src]                                  \n"
      "2:                                                             \n"
      "   mov     %[a_ptr], %[a_bar]                                  \n"
      "   ld1     {v12.16b,v13.16b,v14.16b,v15.16b}, [%[b_ptr]], #64  \n"
      "   add     %[a_bar], %[a_bar], #64                             \n"
      "   ld1     {v16.16b,v17.16b,v18.16b,v19.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v20.16b,v21.16b,v22.16b,v23.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v24.16b,v25.16b,v26.16b,v27.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v28.16b,v29.16b,v30.16b,v31.16b}, [%[a_ptr]], %[n] \n"
      "   udot    v0.4s, v16.16b, v12.16b                             \n"
      "   udot    v1.4s, v20.16b, v12.16b                             \n"
      "   udot    v2.4s, v24.16b, v12.16b                             \n"
      "   udot    v3.4s, v28.16b, v12.16b                             \n"
      "   udot    v0.4s, v17.16b, v13.16b                             \n"
      "   udot    v1.4s, v21.16b, v13.16b                             \n"
      "   udot    v2.4s, v25.16b, v13.16b                             \n"
      "   udot    v3.4s, v29.16b, v13.16b                             \n"
      "   udot    v0.4s, v18.16b, v14.16b                             \n"
      "   udot    v1.4s, v22.16b, v14.16b                             \n"
      "   udot    v2.4s, v26.16b, v14.16b                             \n"
      "   udot    v3.4s, v30.16b, v14.16b                             \n"
      "   udot    v0.4s, v19.16b, v15.16b                             \n"
      "   udot    v1.4s, v23.16b, v15.16b                             \n"
      "   udot    v2.4s, v27.16b, v15.16b                             \n"
      "   udot    v3.4s, v31.16b, v15.16b                             \n"
      "   ld1     {v16.16b,v17.16b,v18.16b,v19.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v20.16b,v21.16b,v22.16b,v23.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v24.16b,v25.16b,v26.16b,v27.16b}, [%[a_ptr]], %[n] \n"
      "   ld1     {v28.16b,v29.16b,v30.16b,v31.16b}, [%[a_ptr]], %[n] \n"
      "   udot    v4.4s, v16.16b, v12.16b                             \n"
      "   udot    v5.4s, v20.16b, v12.16b                             \n"
      "   udot    v6.4s, v24.16b, v12.16b                             \n"
      "   udot    v7.4s, v28.16b, v12.16b                             \n"
      "   udot    v4.4s, v17.16b, v13.16b                             \n"
      "   udot    v5.4s, v21.16b, v13.16b                             \n"
      "   udot    v6.4s, v25.16b, v13.16b                             \n"
      "   udot    v7.4s, v29.16b, v13.16b                             \n"
      "   udot    v4.4s, v18.16b, v14.16b                             \n"
      "   udot    v5.4s, v22.16b, v14.16b                             \n"
      "   udot    v6.4s, v26.16b, v14.16b                             \n"
      "   udot    v7.4s, v30.16b, v14.16b                             \n"
      "   udot    v4.4s, v19.16b, v15.16b                             \n"
      "   udot    v5.4s, v23.16b, v15.16b                             \n"
      "   udot    v6.4s, v27.16b, v15.16b                             \n"
      "   udot    v7.4s, v31.16b, v15.16b                             \n"
      "   cmp     %[b_ptr], %[n_cnd]                                  \n"
      "   b.mi    2b                                                  \n"

      // Reduce and store
      "   addv    s0, v0.4s                                           \n"
      "   addv    s1, v1.4s                                           \n"
      "   addv    s2, v2.4s                                           \n"
      "   addv    s3, v3.4s                                           \n"
      "   addv    s4, v4.4s                                           \n"
      "   addv    s5, v5.4s                                           \n"
      "   addv    s6, v6.4s                                           \n"
      "   addv    s7, v7.4s                                           \n"
      "   stp     s0, s1, [%[c_dst]], #8                              \n"
      "   stp     s2, s3, [%[c_dst]], #8                              \n"
      "   stp     s4, s5, [%[c_dst]], #8                              \n"
      "   stp     s6, s7, [%[c_dst]], #8                              \n"

      // M loop tail
      "   add     %[a_src], %[a_src], %[n], lsl #3                    \n"
      "   cmp     %[c_dst], %[m_cnd]                                  \n"
      "   b.mi    1b                                                  \n"

      : [a_bar] "=&r"(a_bar), [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr),
        [a_src] "+&r"(a), [c_dst] "+&r"(c)
      : [m_cnd] "r"(m_cnd), [n_cnd] "r"(n_cnd), [b_src] "r"(b), [n] "r"(n)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31", "cc", "memory");
}

#else

static void inner_loop_217(struct loop_217_data *data) {
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
// Default of 257KiB equates to original problem size (M=256, N=1024)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n) ((n) * ((m) + 1) * sizeof(uint8_t))

LOOP_DECL(217, OUTER_LOOP_ATTR) {
  // Work out values for M and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 8
  uint64_t N = 0;  // multiple of 4*SVLb

  const uint64_t N_base = MAX_VL / 2;
  while (true) {
    // M must a multiple of 8 (which it will implicitly will be as N is
    // guaranteed to be) and should be 4x smaller than N for this
    // loop's M-to-N ratio.
    uint64_t n = N + N_base;
    uint64_t m = n / 4;
    if (PROBLEM_SIZE_ACTUAL(m, n) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
    } else {
      break;
    }
  }

  // increasing loop iterations
  iters *= 10;

  struct loop_217_data data = {
      .m = M,
      .n = N,
  };
  ALLOC_64B(data.a, M * N, "A matrix");
  ALLOC_64B(data.b, N * 1, "x vector");
  ALLOC_64B(data.c, M * 1, "b vector");

  fill_uint8(data.a, M * N);
  fill_uint8(data.b, N * 1);

  inner_loops_217(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 "\n", M, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x 1\n", M, N, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(j)                                                        \
  {                                                                     \
    uint32_t d = 0;                                                     \
    for (int n = 0; n < N; n++) MLA(d, data.a[(j) * N + n], data.b[n]); \
    checksum += (int)(d != data.c[j]);                                  \
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
  FINALISE_LOOP_I(217, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
