/*----------------------------------------------------------------------------
#
#   Loop 207: INT1-INT32 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of 1-bit MOPA instructions.
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
    M: multiple of SVLh
    N: multiple of SVLh
    K: even and greater than 4

  Note: A and B matrices are considered to be re-arranged,
        as required by the INT1 -> INT32 matrix multiplication.
*/

struct loop_207_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint32_t *restrict a;
  uint32_t *restrict b;
  uint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_207(struct loop_207_data *restrict data) {
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



// LOOP 207 helpers
uint32_t dot_int1(uint32_t A, uint32_t B, int width) {
  int32_t sum = 0;
  uint32_t r = (A ^ B);
  for (int i = 0; i < width; ++i) {
    sum += ((!((r >> i) & 1)) ? 1 : 0);
  }
  return sum;
}

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)


static void inner_loop_207(struct loop_207_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint32_t *restrict a = data->a;
  uint32_t *restrict b = data->b;
  uint32_t *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      c[x * n + y] = 0;
    }
  }

  for (uint64_t z = 0; z < k; z++) {
    for (uint64_t x = 0; x < m; x++) {
      for (uint64_t y = 0; y < n; y++) {
        c[x * n + y] += dot_int1(a[z * m + x], b[z * n + y], 32);
      }
    }
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_207(struct loop_207_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;

  uint32_t *ptr_a, *ptr_b, *ptr_c;
  uint32_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t l_cnd = svl_s * 4;
  uint64_t c_blk = svl_s * n;
  uint64_t c_off = c_blk + n;

  svcount_t c_all = svptrue_c32();
  svbool_t p_all = svptrue_b32();

  svuint8x4_t vec_c0, vec_c1;
  svuint32x2_t vec_a0, vec_a1, vec_b0, vec_b1;

#define MOPA_TILE(t, x, i, j) \
  svbmopa_za32_m(t, p_all, p_all, svget2(vec_a##x, i), svget2(vec_b##x, j))

#define EXTR(x, i) svreinterpret_u32(svget4(vec_c##x, i))
#define STORE_PAIR(x, i, j, o) \
  svst1(c_all, &ptr_c[o], svcreate2(EXTR(x, i), EXTR(x, j)))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_s * 2) {
    for (n_idx = 0; n_idx < n; n_idx += svl_s * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif
      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x2(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x2(c_all, &ptr_a[m]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n]);

        MOPA_TILE(0, 0, 0, 0);
        MOPA_TILE(1, 0, 0, 1);
        MOPA_TILE(2, 0, 1, 0);
        MOPA_TILE(3, 0, 1, 1);
        MOPA_TILE(0, 1, 0, 0);
        MOPA_TILE(1, 1, 0, 1);
        MOPA_TILE(2, 1, 1, 0);
        MOPA_TILE(3, 1, 1, 1);

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
        STORE_PAIR(0, 0, 1, 0);
        STORE_PAIR(1, 0, 1, n);
        STORE_PAIR(0, 2, 3, c_blk);
        STORE_PAIR(1, 2, 3, c_off);

        ptr_c += n * 2;
      }
    }
    c += c_blk * 2;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_207(struct loop_207_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;

  uint32_t *ptr_a, *ptr_b, *ptr_c;
  uint32_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b32();

  svuint32_t lda_0, lda_1;
  svuint32x2_t ldb;
  svuint32_t acc_00, acc_01, acc_10, acc_11;

#define ZERO(i, j) acc_##i##j = svdup_u32(0)
#define BCNT(i, j) svcnt_x(p_all, sveor3(lda_##i, svget2(ldb, j), 0xFFFFFFFF))
#define BMLA(i, j) acc_##i##j = svadd_x(p_all, acc_##i##j, BCNT(i, j))

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c32();
#define LOADB_PAIR svld1_x2(c_all, ptr_b)
#define STORE_PAIR(q) \
  svst1(c_all, &ptr_c[n * q], svcreate2(acc_##q##0, acc_##q##1))
#else
#define LOADB(p) svld1_vnum(p_all, ptr_b, p)
#define LOADB_PAIR svcreate2(LOADB(0), LOADB(1))
#define STORE(q, p) svst1_vnum(p_all, &ptr_c[n * q], p, acc_##q##p);
#define STORE_PAIR(q) STORE(q, 0) STORE(q, 1)
#endif

  for (m_idx = 0; m_idx < m; m_idx += 2) {
    for (n_idx = 0; n_idx < n; n_idx += svcntw() * 2) {
      ZERO(0, 0);
      ZERO(0, 1);
      ZERO(1, 0);
      ZERO(1, 1);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        ldb   = LOADB_PAIR;
        lda_0 = svdup_u32(ptr_a[0]);
        lda_1 = svdup_u32(ptr_a[1]);

        BMLA(0, 0);
        BMLA(0, 1);
        BMLA(1, 0);
        BMLA(1, 1);

        ptr_a += m;
        ptr_b += n;
      }

      ptr_c = &c[n_idx];
      STORE_PAIR(0);
      STORE_PAIR(1);
    }
    c += n * 2;
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_207(struct loop_207_data *data)
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

  register uint64_t a_cnd = (uint64_t)&data->a[m * k];
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t c_blk = svl_s * n;
  register uint64_t c_off = c_blk + n;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      "   ptrue   p0.s                                                      \n"
      "   ptrue   pn8.s                                                     \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif

      // M loop head
      "   mov     %[m_idx], #0                                              \n"
      "1:                                                                   \n"

      // N loop head
      "   mov     %[n_idx], #0                                              \n"
      "2:                                                                   \n"

      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2                      \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2                      \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2                      \n"

      // K loop
      "   ld1w    { z2.s-z3.s }, pn8/z, [%[a_ptr]]                          \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1w    { z0.s-z1.s }, pn8/z, [%[b_ptr]]                          \n"
      "   bmopa   za0.s, p0/m, p0/m, z2.s, z0.s                             \n"
      "   bmopa   za1.s, p0/m, p0/m, z2.s, z1.s                             \n"
      "   ld1w    { z6.s-z7.s }, pn8/z, [%[a_ptr], %[m], lsl #2]            \n"
      "   ld1w    { z4.s-z5.s }, pn8/z, [%[b_ptr], %[n], lsl #2]            \n"
      "   bmopa   za2.s, p0/m, p0/m, z3.s, z0.s                             \n"
      "   bmopa   za3.s, p0/m, p0/m, z3.s, z1.s                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "3:                                                                   \n"
      "   bmopa   za0.s, p0/m, p0/m, z6.s, z4.s                             \n"
      "   bmopa   za1.s, p0/m, p0/m, z6.s, z5.s                             \n"
      "   ld1w    { z2.s-z3.s }, pn8/z, [%[a_ptr]]                          \n"
      "   ld1w    { z0.s-z1.s }, pn8/z, [%[b_ptr]]                          \n"
      "   bmopa   za2.s, p0/m, p0/m, z7.s, z4.s                             \n"
      "   bmopa   za3.s, p0/m, p0/m, z7.s, z5.s                             \n"
      "   ld1w    { z6.s-z7.s }, pn8/z, [%[a_ptr], %[m], lsl #2]            \n"
      "   ld1w    { z4.s-z5.s }, pn8/z, [%[b_ptr], %[n], lsl #2]            \n"
      "   bmopa   za0.s, p0/m, p0/m, z2.s, z0.s                             \n"
      "   bmopa   za1.s, p0/m, p0/m, z2.s, z1.s                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   bmopa   za2.s, p0/m, p0/m, z3.s, z0.s                             \n"
      "   bmopa   za3.s, p0/m, p0/m, z3.s, z1.s                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.mi    3b                                                        \n"
      "   bmopa   za0.s, p0/m, p0/m, z6.s, z4.s                             \n"
      "   bmopa   za1.s, p0/m, p0/m, z6.s, z5.s                             \n"
      "   bmopa   za2.s, p0/m, p0/m, z7.s, z4.s                             \n"
      "   bmopa   za3.s, p0/m, p0/m, z7.s, z5.s                             \n"

      // Store loop
      "   mov     x12, #0                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr]]                            \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_blk], lsl #2]          \n"
      "4:                                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr], %[n], lsl #2]              \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_off], lsl #2]          \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                          \n"
      "   add     x12, x12, #8                                              \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr]]                            \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_blk], lsl #2]          \n"
      "   cmp     x12, %[l_cnd]                                             \n"
      "   b.mi    4b                                                        \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr], %[n], lsl #2]              \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_off], lsl #2]          \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                                     \n"
      "   cmp     %[n_idx], %[n]                                            \n"
      "   b.mi    2b                                                        \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_blk], lsl #3                      \n"
      "   incw    %[m_idx], all, mul #2                                     \n"
      "   cmp     %[m_idx], %[m]                                            \n"
      "   b.mi    1b                                                        \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [c_blk] "r"(c_blk),
        [c_off] "r"(c_off), [l_cnd] "r"(l_cnd), [a_cnd] "r"(a_cnd),
        [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "p0", "p8", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_207(struct loop_207_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t a_cnd = (uint64_t)&data->a[m * k];
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.s                                              \n"
      "   ptrue   pn8.s                                             \n"
      "   mov     z6.b, #0xFF                                       \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"
      "3:                                                           \n"
      "   ld1rw   {z0.s}, p0/z, [%[a_ptr], #0]                      \n"
      "   ld1rw   {z1.s}, p0/z, [%[a_ptr], #4]                      \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                  \n"
      "   ld1w    {z4.s-z5.s}, pn8/z, [%[b_ptr]]                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
      "   mov     z2.d, z0.d                                        \n"
      "   mov     z3.d, z1.d                                        \n"
      "   eor3    z0.d, z0.d, z4.d, z6.d                            \n"
      "   eor3    z1.d, z1.d, z4.d, z6.d                            \n"
      "   eor3    z2.d, z2.d, z5.d, z6.d                            \n"
      "   eor3    z3.d, z3.d, z5.d, z6.d                            \n"
      "   cnt     z10.s, p0/m, z0.s                                 \n"
      "   cnt     z11.s, p0/m, z1.s                                 \n"
      "   cnt     z12.s, p0/m, z2.s                                 \n"
      "   cnt     z13.s, p0/m, z3.s                                 \n"
      "   add     z20.s, z20.s, z10.s                               \n"
      "   add     z21.s, z21.s, z12.s                               \n"
      "   add     z22.s, z22.s, z11.s                               \n"
      "   add     z23.s, z23.s, z13.s                               \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[n], lsl #2]      \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #3                  \n"
      "   add     %[m_idx], %[m_idx], #2                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c_ptr] "=&r"(c_ptr), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_207(struct loop_207_data *data)
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
  register uint64_t a_cnd = (uint64_t)&data->a[m * k];
  register uint64_t off_2 = n + svl_s;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.s                                              \n"
      "   mov     z6.b, #0xFF                                       \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"
      "3:                                                           \n"
      "   ld1rw  {z0.s}, p0/z, [%[a_ptr], #0]                       \n"
      "   ld1rw  {z1.s}, p0/z, [%[a_ptr], #4]                       \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                  \n"
      "   ld1w    {z4.s}, p0/z, [%[b_ptr]]                          \n"
      "   ld1w    {z5.s}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"

      "   mov     z2.d, z0.d                                        \n"
      "   mov     z3.d, z1.d                                        \n"

      "   eor3     z0.d, z0.d, z4.d, z6.d                           \n"
      "   eor3     z1.d, z1.d, z4.d, z6.d                           \n"
      "   eor3     z2.d, z2.d, z5.d, z6.d                           \n"
      "   eor3     z3.d, z3.d, z5.d, z6.d                           \n"

      "   cnt     z10.s, p0/M, z0.s                                 \n"
      "   cnt     z11.s, p0/M, z1.s                                 \n"
      "   cnt     z12.s, p0/M, z2.s                                 \n"
      "   cnt     z13.s, p0/M, z3.s                                 \n"

      "   add     z20.s, z20.s, z10.s                               \n"
      "   add     z21.s, z21.s, z12.s                               \n"
      "   add     z22.s, z22.s, z11.s                               \n"
      "   add     z23.s, z23.s, z13.s                               \n"

      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   st1w    {z20.s}, p0, [%[c_ptr]]                           \n"
      "   st1w    {z21.s}, p0, [%[c_ptr], #1, mul vl]               \n"
      "   st1w    {z22.s}, p0, [%[c_ptr], %[off_1], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c_ptr], %[off_2], lsl #2]         \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #3                  \n"
      "   add     %[m_idx], %[m_idx], #2                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c_ptr] "=&r"(c_ptr), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_2] "r"(off_2), [off_1] "r"(n), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "cc", "memory");
}

#else

static void inner_loop_207(struct loop_207_data *data) {
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
// Default of 256KiB equates to original problem size (M=128, K=256, N=128)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 256
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(float))

LOOP_DECL(207, OUTER_LOOP_ATTR)
{
  uint64_t M = 0; // multiple of SVLh
  uint64_t N = 0; // multiple of SVLh
  uint64_t K = 0; // even and greater than 4

  // For this loop, K should remain as 2*M, M and N must be equal
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m;
    uint64_t k = m * 2;
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_207_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_uint32(data.a, M * K);
  fill_uint32(data.b, K * N);

  inner_loops_207(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         (float)(((M * K) + (K * N)) / 256));
#endif

  int checksum = 0;
#define CHECK(x, y)                                              \
  {                                                              \
    uint32_t v = data.c[(x)*N + (y)];                            \
    uint32_t d = 0;                                              \
    uint32_t *ptr_a = (uint32_t *)data.a;                        \
    uint32_t *ptr_b = (uint32_t *)data.b;                        \
    for (int k = 0; k < K; k++) {                                \
      d += dot_int1(ptr_a[k * M + (x)], ptr_b[k * N + (y)], 32); \
    }                                                            \
    int e = (int)(d != v);                                       \
    checksum += e;                                               \
  }
#ifdef FULL_CHECK
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, 0);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(207, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
