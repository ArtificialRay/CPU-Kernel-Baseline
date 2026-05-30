/*----------------------------------------------------------------------------
#
#   Loop 208: BF16-BF16 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of bf16 to bf16 MOPA (or MLA) instructions.
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
    N: multiple of SVLb
    K: even
*/

struct loop_208_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bfloat16_t *restrict a;
  bfloat16_t *restrict b;
  bfloat16_t *restrict c;
};

static inline __attribute__((unused)) bfloat16_t bf16_mla(bfloat16_t c, bfloat16_t a, bfloat16_t b) {
  return f32_to_bf16((bf16_to_f32(a) * bf16_to_f32(b)) + bf16_to_f32(c));
}

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_208(struct loop_208_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(__ARM_FEATURE_SME2p1)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(__ARM_NEON)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_208(struct loop_208_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *restrict a = data->a;
  bfloat16_t *restrict b = data->b;
  bfloat16_t *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      bfloat16_t d = 0;
      for (uint64_t z = 0; z < k; z++)
        d = bf16_mla(d, a[z*m+x], b[z*n+y]);
      c[x*n+y] = d;
    }
  }
}

#elif (defined(HAVE_SME_INTRINSICS) && defined(__ARM_FEATURE_SME2p1) && defined(__ARM_FEATURE_SME_B16B16))

static void inner_loop_208(struct loop_208_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *a = (bfloat16_t *) data->a;
  bfloat16_t *b = (bfloat16_t *) data->b;
  bfloat16_t *c = (bfloat16_t *) data->c;

  bfloat16_t *ptr_a, *ptr_b, *ptr_c;
  bfloat16_t *cnd_k = &a[m*k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_h = svcnth();
  uint64_t l_cnd = svl_h * 2;
  uint64_t c_blk = svl_h * n;

  svcount_t c_all = svptrue_c16();
  svbool_t  p_all = svptrue_b16();

  svbfloat16_t   vec_a0, vec_a1;
  svbfloat16x2_t vec_b0, vec_b1;
  svuint8x4_t   vec_c0, vec_c1;

  #define MOPA_TILE(t,x,i) \
    svmopa_za16_m(t, p_all, p_all, vec_a##x, svget2(vec_b##x,i))

  #define EXTR(x,i) svreinterpret_bf16(svget4(vec_c##x,i))
  #define STORE_PAIR(x,i,j,y) \
    svst1(c_all, &ptr_c[n*(y)], svcreate2( EXTR(x,i), EXTR(x,j) ))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_h) {
    for (n_idx = 0; n_idx < n; n_idx += svl_h * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1(p_all, &ptr_a[0]);
        vec_a1 = svld1(p_all, &ptr_a[m]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n]);

        MOPA_TILE(0,0,0);
        MOPA_TILE(1,0,1);
        MOPA_TILE(0,1,0);
        MOPA_TILE(1,1,1);

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
        STORE_PAIR(0,0,1,0);
        STORE_PAIR(0,2,3,1);
        STORE_PAIR(1,0,1,2);
        STORE_PAIR(1,2,3,3);

        ptr_c += n * 4;
      }
    }
    c += c_blk;
  }
}

#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE2p1) && defined(__ARM_FEATURE_SVE_B16B16))

static void inner_loop_208(struct loop_208_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *a = (bfloat16_t *) data->a;
  bfloat16_t *b = (bfloat16_t *) data->b;
  bfloat16_t *c = (bfloat16_t *) data->c;
  bfloat16_t zero = vcvth_bf16_f32(0.0f);

  bfloat16_t *ptr_a, *ptr_b, *ptr_c;
  bfloat16_t *cnd_k = &a[m*k];

  uint64_t m_idx, n_idx;
  uint64_t svl_h = svcnth();
  svbool_t p_all = svptrue_b16();
  svcount_t c_all = svptrue_c16();

  svbfloat16_t acc_00, acc_01, acc_02, acc_03;
  svbfloat16_t acc_04, acc_05, acc_06, acc_07;
  svbfloat16_t acc_10, acc_11, acc_12, acc_13;
  svbfloat16_t acc_14, acc_15, acc_16, acc_17;
  svbfloat16_t vec_a0, vec_a1;
  svbfloat16x2_t vec_b0, vec_b1;

  #define ZERO(i,l) acc_##i##l = svdup_bf16(zero)
  #define ZERO_LANE(l) { ZERO(0,l); ZERO(1,l); }

  #define MLA_PAIR(x,y,l) \
    {acc_0##l = svmla_lane(acc_0##l, svget2(vec_b##y,0), vec_a##x, l);\
     acc_1##l = svmla_lane(acc_1##l, svget2(vec_b##y,1), vec_a##x, l);}

  #define MLA_LANE(l) \
    { MLA_PAIR(0,0,l); MLA_PAIR(1,1,l); }

  #define STORE(l) svst1(c_all, &ptr_c[n*l],  svcreate2(acc_0##l, acc_1##l))

  for (m_idx = 0; m_idx < m; m_idx += 8) {
    for (n_idx = 0; n_idx < n; n_idx += 2*svl_h) {
      ZERO_LANE(0); ZERO_LANE(1);
      ZERO_LANE(2); ZERO_LANE(3);
      ZERO_LANE(4); ZERO_LANE(5);
      ZERO_LANE(6); ZERO_LANE(7);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1rq  (p_all, &ptr_a[0]);
        vec_a1 = svld1rq  (p_all, &ptr_a[m]);
        vec_b0 = svld1_x2 (c_all, &ptr_b[0]);
        vec_b1 = svld1_x2 (c_all, &ptr_b[n]);

        MLA_LANE(0); MLA_LANE(1);
        MLA_LANE(2); MLA_LANE(3);
        MLA_LANE(4); MLA_LANE(5);
        MLA_LANE(6); MLA_LANE(7);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      STORE(0); STORE(1);
      STORE(2); STORE(3);
      STORE(4); STORE(5);
      STORE(6); STORE(7);
    }
    c += n * 8;
  }
}

#elif (defined(__ARM_FEATURE_SME2p1) && defined(__ARM_FEATURE_SME_B16B16))

static void inner_loop_208(struct loop_208_data *data)
LOOP_ATTR
{

  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t) data->a;
  register uint64_t b = (uint64_t) data->b;
  register uint64_t c = (uint64_t) data->c;

  register uint64_t svl_h;
  asm volatile( "cnth %[v]" : [v] "=&r" (svl_h) :: );

  register uint64_t c_blk = svl_h * n;
  register uint64_t l_cnd = svl_h * 2 - 8;
  register uint64_t a_cnd = a + 2 * (m*k);
  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova/movaz

  asm volatile(
    "   ptrue   p0.h                                                      \n"
    "   ptrue   pn8.h                                                     \n"
    "   zero    {za}                                                      \n"
    // M loop head
    "   mov     %[m_idx], #0                                              \n"
    "1:                                                                   \n"

    // N loop head
    "   mov     %[n_idx], #0                                              \n"
    "2:                                                                   \n"

    "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                      \n"
    "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                      \n"
    "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1                      \n"

    // K loop
    "   ld1h    {z4.h}, p0/z, [%[a_ptr]]                                  \n"
    "   ld1h    {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]                    \n"
    "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                            \n"
    "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr], %[n], lsl #1]              \n"
    "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                          \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                          \n"
    "   bfmopa  za0.h, p0/m, p0/m, z4.h, z0.h                             \n"
    "   bfmopa  za1.h, p0/m, p0/m, z4.h, z1.h                             \n"
    "3:                                                                   \n"
    "   ld1h    {z4.h}, p0/z, [%[a_ptr]]                                  \n"
    "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                            \n"
    "   bfmopa  za0.h, p0/m, p0/m, z5.h, z2.h                             \n"
    "   bfmopa  za1.h, p0/m, p0/m, z5.h, z3.h                             \n"
    "   ld1h    {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]                    \n"
    "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr], %[n], lsl #1]              \n"
    "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                          \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                          \n"
    "   bfmopa  za0.h, p0/m, p0/m, z4.h, z0.h                             \n"
    "   bfmopa  za1.h, p0/m, p0/m, z4.h, z1.h                             \n"
    "   cmp     %[a_ptr], %[a_cnd]                                        \n"
    "   b.mi    3b                                                        \n"
    "   bfmopa  za0.h, p0/m, p0/m, z5.h, z2.h                             \n"
    "   bfmopa  za1.h, p0/m, p0/m, z5.h, z3.h                             \n"

    // Store loop
    "   mov     x12, #0                                                   \n"
    "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
    "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
    "4:                                                                   \n"
    "   st1h    {z0.h-z1.h}, pn8, [%[c_ptr]]                              \n"
    "   st1h    {z2.h-z3.h}, pn8, [%[c_ptr], %[off_1], lsl #1]            \n"
    "   st1h    {z4.h-z5.h}, pn8, [%[c_ptr], %[off_2], lsl #1]            \n"
    "   st1h    {z6.h-z7.h}, pn8, [%[c_ptr], %[off_3], lsl #1]            \n"
    "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                          \n"
    "   add     x12, x12, #8                                              \n"
    "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
    "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
    "   cmp     x12, %[l_cnd]                                             \n"
    "   b.mi    4b                                                        \n"
    "   st1h    {z0.h-z1.h}, pn8, [%[c_ptr]]                              \n"
    "   st1h    {z2.h-z3.h}, pn8, [%[c_ptr], %[off_1], lsl #1]            \n"
    "   st1h    {z4.h-z5.h}, pn8, [%[c_ptr], %[off_2], lsl #1]            \n"
    "   st1h    {z6.h-z7.h}, pn8, [%[c_ptr], %[off_3], lsl #1]            \n"

    // N loop tail
    "   inch    %[n_idx], all, mul #2                                     \n"
    "   cmp     %[n_idx], %[n]                                            \n"
    "   b.mi    2b                                                        \n"

    // M loop tail
    "   add     %[c_dst], %[c_dst], %[c_blk], lsl #1                      \n"
    "   inch    %[m_idx]                                                  \n"
    "   cmp     %[m_idx], %[m]                                            \n"
    "   b.mi    1b                                                        \n"

    : [a_ptr] "=&r" (a_ptr), [b_ptr] "=&r" (b_ptr), [c_ptr] "=&r" (c_ptr),
      [m_idx] "=&r" (m_idx), [n_idx] "=&r" (n_idx), [c_dst] "+&r" (c)
    : [m] "r" (m), [n] "r" (n), [k] "r" (k),
      [l_cnd] "r" (l_cnd), [a_cnd] "r" (a_cnd), [c_blk] "r" (c_blk),
      [off_2] "r" (off_2), [off_3] "r" (off_3), [off_1] "r" (n),
      [a_src] "r" (a), [b_src] "r" (b)
    : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
      "p0", "p8",  "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif (defined(__ARM_FEATURE_SVE2p1) && defined(__ARM_FEATURE_SVE_B16B16))

static void inner_loop_208(struct loop_208_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t) data->a;
  register uint64_t b = (uint64_t) data->b;
  register uint64_t c = (uint64_t) data->c;

  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t a_cnd = a + 2 * (m*k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
    "   ptrue   p0.h                                                \n"
    "   ptrue   pn8.h                                               \n"

    // M loop head
    "   mov     %[m_idx], #0                                        \n"
    "1:                                                             \n"

    // N loop head
    "   mov     %[n_idx], #0                                        \n"
    "2:                                                             \n"

    // Accumulators
    "   mov     z10.h, #0                                           \n"
    "   mov     z11.h, #0                                           \n"
    "   mov     z12.h, #0                                           \n"
    "   mov     z13.h, #0                                           \n"
    "   mov     z14.h, #0                                           \n"
    "   mov     z15.h, #0                                           \n"
    "   mov     z16.h, #0                                           \n"
    "   mov     z17.h, #0                                           \n"
    "   mov     z20.h, #0                                           \n"
    "   mov     z21.h, #0                                           \n"
    "   mov     z22.h, #0                                           \n"
    "   mov     z23.h, #0                                           \n"
    "   mov     z24.h, #0                                           \n"
    "   mov     z25.h, #0                                           \n"
    "   mov     z26.h, #0                                           \n"
    "   mov     z27.h, #0                                           \n"

    // K loop
    "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                \n"
    "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                \n"
    "3:                                                             \n"
    "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                            \n"
    "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]              \n"
    "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
    "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                      \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #1                    \n"
    "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr]]                      \n"

    "   add     %[b_ptr], %[b_ptr], %[n], lsl #1                    \n"
    "   bfmla    z10.h, z0.h, z4.h[0]                               \n"
    "   bfmla    z12.h, z0.h, z4.h[1]                               \n"
    "   bfmla    z14.h, z0.h, z4.h[2]                               \n"
    "   bfmla    z16.h, z0.h, z4.h[3]                               \n"
    "   bfmla    z20.h, z0.h, z4.h[4]                               \n"
    "   bfmla    z22.h, z0.h, z4.h[5]                               \n"
    "   bfmla    z24.h, z0.h, z4.h[6]                               \n"
    "   bfmla    z26.h, z0.h, z4.h[7]                               \n"
    "   bfmla    z11.h, z1.h, z4.h[0]                               \n"
    "   bfmla    z13.h, z1.h, z4.h[1]                               \n"
    "   bfmla    z15.h, z1.h, z4.h[2]                               \n"
    "   bfmla    z17.h, z1.h, z4.h[3]                               \n"
    "   bfmla    z21.h, z1.h, z4.h[4]                               \n"
    "   bfmla    z23.h, z1.h, z4.h[5]                               \n"
    "   bfmla    z25.h, z1.h, z4.h[6]                               \n"
    "   bfmla    z27.h, z1.h, z4.h[7]                               \n"
    "   bfmla    z10.h, z2.h, z5.h[0]                               \n"
    "   bfmla    z12.h, z2.h, z5.h[1]                               \n"
    "   bfmla    z14.h, z2.h, z5.h[2]                               \n"
    "   bfmla    z16.h, z2.h, z5.h[3]                               \n"
    "   bfmla    z20.h, z2.h, z5.h[4]                               \n"
    "   bfmla    z22.h, z2.h, z5.h[5]                               \n"
    "   bfmla    z24.h, z2.h, z5.h[6]                               \n"
    "   bfmla    z26.h, z2.h, z5.h[7]                               \n"
    "   bfmla    z11.h, z3.h, z5.h[0]                               \n"
    "   bfmla    z13.h, z3.h, z5.h[1]                               \n"
    "   bfmla    z15.h, z3.h, z5.h[2]                               \n"
    "   bfmla    z17.h, z3.h, z5.h[3]                               \n"
    "   bfmla    z21.h, z3.h, z5.h[4]                               \n"
    "   bfmla    z23.h, z3.h, z5.h[5]                               \n"
    "   bfmla    z25.h, z3.h, z5.h[6]                               \n"
    "   bfmla    z27.h, z3.h, z5.h[7]                               \n"
    "   cmp     %[a_ptr], %[a_cnd]                                  \n"
    "   b.mi    3b                                                  \n"

    // Store
    "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1                \n"

    "   st1h    {z10.h-z11.h}, pn8, [%[c_ptr]]                      \n"
    "   st1h    {z12.h-z13.h}, pn8, [%[c_ptr], %[off_1], lsl #1]    \n"
    "   st1h    {z14.h-z15.h}, pn8, [%[c_ptr], %[off_2], lsl #1]    \n"
    "   st1h    {z16.h-z17.h}, pn8, [%[c_ptr], %[off_3], lsl #1]    \n"
    "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                    \n"
    "   st1h    {z20.h-z21.h}, pn8, [%[c_ptr]]                      \n"
    "   st1h    {z22.h-z23.h}, pn8, [%[c_ptr], %[off_1], lsl #1]    \n"
    "   st1h    {z24.h-z25.h}, pn8, [%[c_ptr], %[off_2], lsl #1]    \n"
    "   st1h    {z26.h-z27.h}, pn8, [%[c_ptr], %[off_3], lsl #1]    \n"

    // N loop tail
    "   inch    %[n_idx], all, mul #2                               \n"
    "   cmp     %[n_idx], %[n]                                      \n"
    "   b.mi    2b                                                  \n"

    // M loop tail
    "   add     %[c_dst], %[c_dst], %[n], lsl #4                    \n"
    "   add     %[m_idx], %[m_idx], #8                              \n"
    "   cmp     %[m_idx], %[m]                                      \n"
    "   b.mi    1b                                                  \n"

    : [a_ptr] "=&r" (a_ptr), [b_ptr] "=&r" (b_ptr), [m_idx] "=&r" (m_idx),
      [c_ptr] "=&r" (c_ptr), [n_idx] "=&r" (n_idx), [c_dst] "+&r" (c)
    : [m] "r" (m), [n] "r" (n), [k] "r" (k), [a_cnd] "r" (a_cnd),
      [off_2] "r" (off_2), [off_3] "r" (off_3), [off_1] "r" (n),
      [a_src] "r" (a), [b_src] "r" (b)
    : "z0", "z1", "z2", "z3", "z4", "z5",
      "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",
      "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
      "p0", "cc", "memory"
  );
}

#else

static void inner_loop_208(struct loop_208_data *data) {
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
// Default of 128KiB equates to original problem size (M=128, K=128, N=256)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 128
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(bfloat16_t))

LOOP_DECL(208, OUTER_LOOP_ATTR)
{

  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLh
  uint64_t K = 0;  // even

  // For this loop, N should be equal to M, K should be 2*M
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m ;      // Automatically a multiple of SVLh
    uint64_t k = m * 2;   // Automatically a multiple of 2
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  // increasing loop iterations
  iters *= 10;

  struct loop_208_data data = {.m = M, .n = N, .k = K,};
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C vector");

  fill_bf16(data.a, M*K);
  fill_bf16(data.b, K*N);

  inner_loops_208(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x,y)                                                                       \
{                                                                                        \
  bfloat16_t d = 0;                                                                      \
  for (uint64_t k = 0; k < K; k++)                                                       \
    d = bf16_mla(d, data.a[k*M+(x)], data.b[k*N+(y)]);                                   \
  checksum += (int)!check_float(bf16_to_f32(d), bf16_to_f32(data.c[(x)*N+(y)]), 4e-3f);  \
}

#ifdef FULL_CHECK
  for (uint64_t m = 0; m < M; m++)
    for (uint64_t n = 0; n < N; n++) CHECK(m,n);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(208, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
