/*----------------------------------------------------------------------------
#
#   Loop 130: FP32 matrix-matrix multiply using
#
#   Purpose:
#     Use of fp32 MMLA instructions.
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
    A: column-major, 2-way interleaved
    B: row-major, 2-way interleaved
    C: row-major
  Constraints -
    M: multiple of 12
    N: multiple of 2*SVLs
    K: even
*/

struct loop_130_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float32_t *restrict a;
  float32_t *restrict b;
  float32_t *restrict c;
};

static inline __attribute__((unused)) float32_t fp32_dot2(uint64_t x, uint64_t y, uint64_t z,
                                  struct loop_130_data *data) {
  float32_t const *a = &data->a[z * data->m + y * 2];
  float32_t const *b = &data->b[z * data->n + x * 2];
  return a[0] * b[0] + a[1] * b[1];
}

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_130(struct loop_130_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_130(struct loop_130_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float32_t *restrict c = data->c;

  for (uint64_t y = 0; y < m; y++)
    for (uint64_t x = 0; x < n; x++) c[y * n + x] = 0.0f;

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z += 2)
    for (uint64_t y = 0; y < m; y++)
      for (uint64_t x = 0; x < n; x++) c[y * n + x] += fp32_dot2(x, y, z, data);
}

#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE_MATMUL_FP32))

static void inner_loop_130(struct loop_130_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float32_t *a = data->a;
  float32_t *b = data->b;
  float32_t *c = data->c;

  float32_t *ptr_a, *ptr_b, *ptr_c;
  float32_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b32();

  svfloat32x4_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5;
  svfloat32_t lda_0, lda_1, lda_2, lda_3, lda_4, lda_5;
  svfloat32x4_t ldb;

#define ZERO svdup_f32(0.0f)
#define ZERO_QUAD(x) acc_##x = svcreate4(ZERO, ZERO, ZERO, ZERO)

#define LOADA(y) svld1rq(p_all, &ptr_a[4 * y])

#define GETC(x, y) svreinterpret_u64(svget4(acc_##y, x))
#define UNZIP(x, y, z) \
  svreinterpret_f32(svuzp##z(GETC(2 * x + 0, y), GETC(2 * x + 1, y)))

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c32();
#define LOADB_QUAD ldb = svld1_x4(c_all, ptr_b)
#define UNZIP_PAIR(y, z) svcreate2(UNZIP(0, y, z), UNZIP(1, y, z))
#define STORE_PAIR(y, z) \
  svst1_vnum(c_all, &ptr_c[n * (2 * y + z - 1)], 0, UNZIP_PAIR(y, z))
#else
#define LOADB(x) svld1_vnum(p_all, ptr_b, x)
#define LOADB_QUAD ldb = svcreate4(LOADB(0), LOADB(1), LOADB(2), LOADB(3))
#define STORE(x, y, z) \
  svst1_vnum(p_all, &ptr_c[n * (2 * y + z - 1)], x, UNZIP(x, y, z));
#define STORE_PAIR(y, z) STORE(0, y, z) STORE(1, y, z)
#endif

  for (m_idx = 0; m_idx < m; m_idx += 12) {
    for (n_idx = 0; n_idx < n; n_idx += svcntw() * 2) {
      ZERO_QUAD(0);
      ZERO_QUAD(1);
      ZERO_QUAD(2);
      ZERO_QUAD(3);
      ZERO_QUAD(4);
      ZERO_QUAD(5);

      ptr_a = &a[m_idx * 2];
      ptr_b = &b[n_idx * 2];

      while (ptr_a < cnd_k) {
        lda_0 = LOADA(0);
        lda_1 = LOADA(1);
        lda_2 = LOADA(2);
        lda_3 = LOADA(3);
        lda_4 = LOADA(4);
        lda_5 = LOADA(5);

        LOADB_QUAD;

        acc_0 = svcreate4(svmmla(svget4(acc_0, 0), lda_0, svget4(ldb, 0)),
                          svmmla(svget4(acc_0, 1), lda_0, svget4(ldb, 1)),
                          svmmla(svget4(acc_0, 2), lda_0, svget4(ldb, 2)),
                          svmmla(svget4(acc_0, 3), lda_0, svget4(ldb, 3)));
        acc_1 = svcreate4(svmmla(svget4(acc_1, 0), lda_1, svget4(ldb, 0)),
                          svmmla(svget4(acc_1, 1), lda_1, svget4(ldb, 1)),
                          svmmla(svget4(acc_1, 2), lda_1, svget4(ldb, 2)),
                          svmmla(svget4(acc_1, 3), lda_1, svget4(ldb, 3)));
        acc_2 = svcreate4(svmmla(svget4(acc_2, 0), lda_2, svget4(ldb, 0)),
                          svmmla(svget4(acc_2, 1), lda_2, svget4(ldb, 1)),
                          svmmla(svget4(acc_2, 2), lda_2, svget4(ldb, 2)),
                          svmmla(svget4(acc_2, 3), lda_2, svget4(ldb, 3)));
        acc_3 = svcreate4(svmmla(svget4(acc_3, 0), lda_3, svget4(ldb, 0)),
                          svmmla(svget4(acc_3, 1), lda_3, svget4(ldb, 1)),
                          svmmla(svget4(acc_3, 2), lda_3, svget4(ldb, 2)),
                          svmmla(svget4(acc_3, 3), lda_3, svget4(ldb, 3)));
        acc_4 = svcreate4(svmmla(svget4(acc_4, 0), lda_4, svget4(ldb, 0)),
                          svmmla(svget4(acc_4, 1), lda_4, svget4(ldb, 1)),
                          svmmla(svget4(acc_4, 2), lda_4, svget4(ldb, 2)),
                          svmmla(svget4(acc_4, 3), lda_4, svget4(ldb, 3)));
        acc_5 = svcreate4(svmmla(svget4(acc_5, 0), lda_5, svget4(ldb, 0)),
                          svmmla(svget4(acc_5, 1), lda_5, svget4(ldb, 1)),
                          svmmla(svget4(acc_5, 2), lda_5, svget4(ldb, 2)),
                          svmmla(svget4(acc_5, 3), lda_5, svget4(ldb, 3)));

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      STORE_PAIR(0, 1);
      STORE_PAIR(0, 2);
      STORE_PAIR(1, 1);
      STORE_PAIR(1, 2);
      STORE_PAIR(2, 1);
      STORE_PAIR(2, 2);
      STORE_PAIR(3, 1);
      STORE_PAIR(3, 2);
      STORE_PAIR(4, 1);
      STORE_PAIR(4, 2);
      STORE_PAIR(5, 1);
      STORE_PAIR(5, 2);
    }
    c += n * 12;
  }
}

#elif (defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE_MATMUL_FP32))

static void inner_loop_130(struct loop_130_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + 4 * (m * k);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
#if defined(__ARM_FEATURE_SVE2p1)
  register uint64_t c_ptr;
#else
  register uint64_t c0ptr;
  register uint64_t c1ptr;
#endif

  asm volatile(
#if defined(__ARM_FEATURE_SVE2p1)
      "   ptrue   pn8.s                                             \n"
#endif
      "   ptrue   p0.s                                              \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z8.s,  #0                                         \n"
      "   mov     z9.s,  #0                                         \n"
      "   mov     z10.s, #0                                         \n"
      "   mov     z11.s, #0                                         \n"
      "   mov     z12.s, #0                                         \n"
      "   mov     z13.s, #0                                         \n"
      "   mov     z14.s, #0                                         \n"
      "   mov     z15.s, #0                                         \n"
      "   mov     z16.s, #0                                         \n"
      "   mov     z17.s, #0                                         \n"
      "   mov     z18.s, #0                                         \n"
      "   mov     z19.s, #0                                         \n"
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"
      "   mov     z24.s, #0                                         \n"
      "   mov     z25.s, #0                                         \n"
      "   mov     z26.s, #0                                         \n"
      "   mov     z27.s, #0                                         \n"
      "   mov     z28.s, #0                                         \n"
      "   mov     z29.s, #0                                         \n"
      "   mov     z30.s, #0                                         \n"
      "   mov     z31.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1rqw  {z0.s}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqw  {z1.s}, p0/z, [%[a_ptr], #16]                     \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   ld1w    {z4.s-z7.s}, pn8/z, [%[b_ptr]]                    \n"
#else
      "   ld1w    {z4.s}, p0/z, [%[b_ptr], #0, mul vl]              \n"
      "   ld1w    {z5.s}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   ld1w    {z6.s}, p0/z, [%[b_ptr], #2, mul vl]              \n"
      "   ld1w    {z7.s}, p0/z, [%[b_ptr], #3, mul vl]              \n"
#endif
      "   fmmla   z16.s, z0.s, z4.s                                 \n"
      "   fmmla   z20.s, z0.s, z5.s                                 \n"
      "   fmmla   z24.s, z0.s, z6.s                                 \n"
      "   fmmla   z28.s, z0.s, z7.s                                 \n"

      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   ld1rqw  {z2.s}, p0/z, [%[a_ptr], #32]                     \n"
      "   fmmla   z17.s, z1.s, z4.s                                 \n"
      "   fmmla   z21.s, z1.s, z5.s                                 \n"
      "   fmmla   z25.s, z1.s, z6.s                                 \n"
      "   fmmla   z29.s, z1.s, z7.s                                 \n"

      "   ld1rqw  {z3.s}, p0/z, [%[a_ptr], #48]                     \n"
      "   fmmla   z18.s, z2.s, z4.s                                 \n"
      "   fmmla   z22.s, z2.s, z5.s                                 \n"
      "   fmmla   z26.s, z2.s, z6.s                                 \n"
      "   fmmla   z30.s, z2.s, z7.s                                 \n"

      "   ld1rqw  {z0.s}, p0/z, [%[a_ptr], #64]                     \n"
      "   fmmla   z19.s, z3.s, z4.s                                 \n"
      "   fmmla   z23.s, z3.s, z5.s                                 \n"
      "   fmmla   z27.s, z3.s, z6.s                                 \n"
      "   fmmla   z31.s, z3.s, z7.s                                 \n"

      "   ld1rqw  {z1.s}, p0/z, [%[a_ptr], #80]                     \n"
      "   fmmla   z8.s,  z0.s, z4.s                                 \n"
      "   fmmla   z10.s, z0.s, z5.s                                 \n"
      "   fmmla   z12.s, z0.s, z6.s                                 \n"
      "   fmmla   z14.s, z0.s, z7.s                                 \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   fmmla   z9.s,  z1.s, z4.s                                 \n"
      "   fmmla   z11.s, z1.s, z5.s                                 \n"
      "   fmmla   z13.s, z1.s, z6.s                                 \n"
      "   fmmla   z15.s, z1.s, z7.s                                 \n"

      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

  // Store
#if defined(__ARM_FEATURE_SVE2p1)
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
#else
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
#endif
      "   uzp1    z0.d, z16.d, z20.d                                \n"
      "   uzp1    z1.d, z24.d, z28.d                                \n"
      "   uzp2    z2.d, z16.d, z20.d                                \n"
      "   uzp2    z3.d, z24.d, z28.d                                \n"
      "   uzp1    z4.d, z17.d, z21.d                                \n"
      "   uzp1    z5.d, z25.d, z29.d                                \n"
      "   uzp2    z6.d, z17.d, z21.d                                \n"
      "   uzp2    z7.d, z25.d, z29.d                                \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c1off], lsl #2]    \n"
      "   st1w    {z4.s-z5.s}, pn8, [%[c_ptr], %[c2off], lsl #2]    \n"
      "   st1w    {z6.s-z7.s}, pn8, [%[c_ptr], %[c3off], lsl #2]    \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z0.s}, p0, [%[c0ptr]]                            \n"
      "   st1w    {z1.s}, p0, [%[c1ptr]]                            \n"
      "   st1w    {z2.s}, p0, [%[c0ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z3.s}, p0, [%[c1ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z4.s}, p0, [%[c0ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z5.s}, p0, [%[c1ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z6.s}, p0, [%[c0ptr], %[c3off], lsl #2]          \n"
      "   st1w    {z7.s}, p0, [%[c1ptr], %[c3off], lsl #2]          \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif
      "   uzp1    z0.d, z18.d, z22.d                                \n"
      "   uzp1    z1.d, z26.d, z30.d                                \n"
      "   uzp2    z2.d, z18.d, z22.d                                \n"
      "   uzp2    z3.d, z26.d, z30.d                                \n"
      "   uzp1    z4.d, z19.d, z23.d                                \n"
      "   uzp1    z5.d, z27.d, z31.d                                \n"
      "   uzp2    z6.d, z19.d, z23.d                                \n"
      "   uzp2    z7.d, z27.d, z31.d                                \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c1off], lsl #2]    \n"
      "   st1w    {z4.s-z5.s}, pn8, [%[c_ptr], %[c2off], lsl #2]    \n"
      "   st1w    {z6.s-z7.s}, pn8, [%[c_ptr], %[c3off], lsl #2]    \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z0.s}, p0, [%[c0ptr]]                            \n"
      "   st1w    {z1.s}, p0, [%[c1ptr]]                            \n"
      "   st1w    {z2.s}, p0, [%[c0ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z3.s}, p0, [%[c1ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z4.s}, p0, [%[c0ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z5.s}, p0, [%[c1ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z6.s}, p0, [%[c0ptr], %[c3off], lsl #2]          \n"
      "   st1w    {z7.s}, p0, [%[c1ptr], %[c3off], lsl #2]          \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif
      "   uzp1    z16.d, z8.d,  z10.d                               \n"
      "   uzp1    z17.d, z12.d, z14.d                               \n"
      "   uzp2    z20.d, z8.d,  z10.d                               \n"
      "   uzp2    z21.d, z12.d, z14.d                               \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z16.s-z17.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                  \n"
#else
      "   st1w    {z16.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z17.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z20.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z21.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #3                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #3                  \n"
#endif
      "   uzp1    z18.d, z9.d,  z11.d                               \n"
      "   uzp1    z19.d, z13.d, z15.d                               \n"
      "   uzp2    z22.d, z9.d,  z11.d                               \n"
      "   uzp2    z23.d, z13.d, z15.d                               \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z18.s-z19.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
#else
      "   st1w    {z18.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z19.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z22.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
#endif

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
#if defined(__ARM_FEATURE_SVE2p1)
        [c_ptr] "=&r"(c_ptr), [n_idx] "=&r"(n_idx),
#else
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
#endif
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
        "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20",
        "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30",
        "z31",
#if defined(__ARM_FEATURE_SVE2p1)
        "p8", "p0",
#else
        "p0",
#endif
        "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_130(struct loop_130_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t a_inc = m * 4 * 2 - 32;
  register uint64_t b_inc = n * 4 * 2 - 32;
  register uint64_t c_inc = n * 4;
  register uint64_t a_cnd = (uint64_t)&data->a[m * k];

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t a1ptr;
  register uint64_t b_ptr;
  register uint64_t b1ptr;
  register uint64_t c_ptr;

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                \n"
      "1:                                                     \n"

      // N loop head
      "   mov     %[n_idx], #0                                \n"
      "2:                                                     \n"

      // Accumulators
      "   movi    v8.4s,  #0                                  \n"
      "   movi    v9.4s,  #0                                  \n"
      "   movi    v10.4s, #0                                  \n"
      "   movi    v11.4s, #0                                  \n"
      "   movi    v12.4s, #0                                  \n"
      "   movi    v13.4s, #0                                  \n"
      "   movi    v14.4s, #0                                  \n"
      "   movi    v15.4s, #0                                  \n"
      "   movi    v16.4s, #0                                  \n"
      "   movi    v17.4s, #0                                  \n"
      "   movi    v18.4s, #0                                  \n"
      "   movi    v19.4s, #0                                  \n"
      "   movi    v20.4s, #0                                  \n"
      "   movi    v21.4s, #0                                  \n"
      "   movi    v22.4s, #0                                  \n"
      "   movi    v23.4s, #0                                  \n"
      "   movi    v24.4s, #0                                  \n"
      "   movi    v25.4s, #0                                  \n"
      "   movi    v26.4s, #0                                  \n"
      "   movi    v27.4s, #0                                  \n"
      "   movi    v28.4s, #0                                  \n"
      "   movi    v29.4s, #0                                  \n"
      "   movi    v30.4s, #0                                  \n"
      "   movi    v31.4s, #0                                  \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3        \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3        \n"
      "3:                                                     \n"
      "   add     %[a1ptr], %[a_ptr], %[m], lsl #2            \n"
      "   add     %[b1ptr], %[b_ptr], %[n], lsl #2            \n"
      "   ld2     {v0.4s,v1.4s}, [%[a_ptr]], #32              \n"
      "   ld2     {v4.4s,v5.4s}, [%[b_ptr]], #32              \n"
      "   ld2     {v2.4s,v3.4s}, [%[a_ptr]], %[a_inc]         \n"
      "   ld2     {v6.4s,v7.4s}, [%[b_ptr]], %[b_inc]         \n"

      "   fmla    v10.4s, v4.4s, v0.s[0]                      \n"
      "   fmla    v12.4s, v4.4s, v0.s[1]                      \n"
      "   fmla    v14.4s, v4.4s, v0.s[2]                      \n"
      "   fmla    v16.4s, v4.4s, v0.s[3]                      \n"

      "   fmla    v10.4s, v5.4s, v1.s[0]                      \n"
      "   fmla    v12.4s, v5.4s, v1.s[1]                      \n"
      "   fmla    v14.4s, v5.4s, v1.s[2]                      \n"
      "   fmla    v16.4s, v5.4s, v1.s[3]                      \n"

      "   fmla    v20.4s, v4.4s, v2.s[0]                      \n"
      "   fmla    v22.4s, v4.4s, v2.s[1]                      \n"
      "   fmla    v24.4s, v4.4s, v2.s[2]                      \n"
      "   fmla    v26.4s, v4.4s, v2.s[3]                      \n"

      "   fmla    v20.4s, v5.4s, v3.s[0]                      \n"
      "   fmla    v22.4s, v5.4s, v3.s[1]                      \n"
      "   fmla    v24.4s, v5.4s, v3.s[2]                      \n"
      "   fmla    v26.4s, v5.4s, v3.s[3]                      \n"

      "   fmla    v11.4s, v6.4s, v0.s[0]                      \n"
      "   fmla    v13.4s, v6.4s, v0.s[1]                      \n"
      "   fmla    v15.4s, v6.4s, v0.s[2]                      \n"
      "   fmla    v17.4s, v6.4s, v0.s[3]                      \n"

      "   fmla    v11.4s, v7.4s, v1.s[0]                      \n"
      "   fmla    v13.4s, v7.4s, v1.s[1]                      \n"
      "   fmla    v15.4s, v7.4s, v1.s[2]                      \n"
      "   fmla    v17.4s, v7.4s, v1.s[3]                      \n"

      "   ld2     {v0.4s,v1.4s}, [%[a1ptr]], #32              \n"
      "   fmla    v21.4s, v6.4s, v2.s[0]                      \n"
      "   fmla    v23.4s, v6.4s, v2.s[1]                      \n"
      "   fmla    v25.4s, v6.4s, v2.s[2]                      \n"
      "   fmla    v27.4s, v6.4s, v2.s[3]                      \n"

      "   ld2     {v4.4s,v5.4s}, [%[b1ptr]], #32              \n"
      "   fmla    v21.4s, v7.4s, v3.s[0]                      \n"
      "   fmla    v23.4s, v7.4s, v3.s[1]                      \n"
      "   fmla    v25.4s, v7.4s, v3.s[2]                      \n"
      "   fmla    v27.4s, v7.4s, v3.s[3]                      \n"

      "   fmla    v8.4s,  v4.4s, v0.s[0]                      \n"
      "   fmla    v18.4s, v4.4s, v0.s[1]                      \n"
      "   fmla    v28.4s, v4.4s, v0.s[2]                      \n"
      "   fmla    v30.4s, v4.4s, v0.s[3]                      \n"

      "   ld2     {v2.4s,v3.4s}, [%[a1ptr]], %[a_inc]         \n"
      "   fmla    v8.4s,  v5.4s, v1.s[0]                      \n"
      "   fmla    v18.4s, v5.4s, v1.s[1]                      \n"
      "   fmla    v28.4s, v5.4s, v1.s[2]                      \n"
      "   fmla    v30.4s, v5.4s, v1.s[3]                      \n"

      "   fmla    v9.4s,  v4.4s, v2.s[0]                      \n"
      "   fmla    v19.4s, v4.4s, v2.s[1]                      \n"
      "   fmla    v29.4s, v4.4s, v2.s[2]                      \n"
      "   fmla    v31.4s, v4.4s, v2.s[3]                      \n"

      "   fmla    v9.4s,  v5.4s, v3.s[0]                      \n"
      "   fmla    v19.4s, v5.4s, v3.s[1]                      \n"
      "   fmla    v29.4s, v5.4s, v3.s[2]                      \n"
      "   fmla    v31.4s, v5.4s, v3.s[3]                      \n"

      "   cmp     %[a_ptr], %[a_cnd]                          \n"
      "   b.mi    3b                                          \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2        \n"
      "   st1     {v10.4s,v11.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v12.4s,v13.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v14.4s,v15.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v24.4s,v25.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v26.4s,v27.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v8.4s,v9.4s},   [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v18.4s,v19.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v28.4s,v29.4s}, [%[c_ptr]], %[c_inc]       \n"
      "   st1     {v30.4s,v31.4s}, [%[c_ptr]], %[c_inc]       \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                      \n"
      "   cmp     %[n_idx], %[n]                              \n"
      "   b.mi    2b                                          \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5            \n"
      "   add     %[m_idx], %[m_idx], #8                      \n"
      "   cmp     %[m_idx], %[m]                              \n"
      "   b.mi    1b                                          \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c),
        [a1ptr] "=&r"(a1ptr), [b1ptr] "=&r"(b1ptr)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [a_inc] "r"(a_inc), [b_inc] "r"(b_inc), [c_inc] "r"(c_inc),
        [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "cc", "memory");
}

#else

static void inner_loop_130(struct loop_130_data *data) {
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
// Default of 192KiB equates to original problem size (M=128, K=128, N=64)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 192
#endif

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n, k) ((k) * ((m) + (n)) * sizeof(float))

LOOP_DECL(130, NS_SVE_LOOP_ATTR) {
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 12
  uint64_t N = 0;  // multiple of 2*SVLs
  uint64_t K = 0;  // even

  // For this loop, K should equal to N/4, M must be equal to N/2
  const uint64_t K_base = MAX_VL / 8;
  uint64_t m = 7 * 12;

  while (true) {
    uint64_t k = K + K_base;
    uint64_t n = k / 2;
    if (PROBLEM_SIZE_ACTUAL(m, n, k) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_130_data data = {
      .m = M,
      .n = N,
      .k = K,
  };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_float(data.a, M * K);
  fill_float(data.b, K * N);

  inner_loops_130(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N, K) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(y, x)                                                 \
  {                                                                 \
    float32_t d = 0.0f;                                             \
    for (int k = 0; k < K; k += 2) d += fp32_dot2(x, y, k, &data);  \
    checksum += (int)!check_float(d, data.c[(y) * N + (x)], 1e-3f); \
  }
#ifdef FULL_CHECK
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
  FINALISE_LOOP_I(130, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
