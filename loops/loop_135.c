/*----------------------------------------------------------------------------
#
#   Loop 135: INT8-INT32 matrix-matrix multiply using MMLA
#
#   Purpose:
#     Use of i8 to i32 MMLA instructions.
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
    A: column-major, 8 way interleaved
    B: row-major, 8 way interleaved
    C: row-major
  Constraints -
    M: multiple of 8
    N: multiple of 2*SVLs
    K: multiple of 8

  Note: A and B matrices are considered to be re-arranged,
        as required by the SMMLA, INT8 -> INT32 matrix multiplication.
*/

struct loop_135_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  int8_t *restrict a;
  int8_t *restrict b;
  int32_t *restrict c;
};

static inline __attribute__((unused)) int32_t int32_dot8(uint64_t i, uint64_t j, uint64_t k,
                                 struct loop_135_data *data) {
  int8_t const *a = &data->a[k * data->m + 8 * i];
  int8_t const *b = &data->b[k * data->n + 8 * j];
  return (int32_t)a[0] * (int32_t)b[0] + (int32_t)a[1] * (int32_t)b[1] +
         (int32_t)a[2] * (int32_t)b[2] + (int32_t)a[3] * (int32_t)b[3] +
         (int32_t)a[4] * (int32_t)b[4] + (int32_t)a[5] * (int32_t)b[5] +
         (int32_t)a[6] * (int32_t)b[6] + (int32_t)a[7] * (int32_t)b[7];
}

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_135(struct loop_135_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_135(struct loop_135_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  int32_t *restrict c = data->c;

  for (uint64_t x = 0; x < m; x++)
    for (uint64_t y = 0; y < n; y++) c[x * n + y] = 0;

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z += 8)
    for (uint64_t x = 0; x < m; x++)
      for (uint64_t y = 0; y < n; y++)
        c[x * n + y] += int32_dot8(x, y, z, data);
}
#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE_MATMUL_INT8))

static void inner_loop_135(struct loop_135_data *data) {
  const uint64_t m = data->m;
  const uint64_t n = data->n;
  const uint64_t k = data->k;
  const int8_t* const a = data->a;
  const int8_t* const b = data->b;
  int32_t* const c = data->c;
  const svbool_t pg = svptrue_b8();

  // M loop
  for(uint64_t m_idx = 0; m_idx < m; m_idx += 12) {

    // N loop
    for(uint64_t n_idx = 0; n_idx < n; n_idx += 2 * svcntw()) {
      // 24 Accumulators
      svint32_t acc0_0 = svdup_s32(0);
      svint32_t acc0_1 = svdup_s32(0);
      svint32_t acc0_2 = svdup_s32(0);
      svint32_t acc0_3 = svdup_s32(0);
      svint32_t acc1_0 = svdup_s32(0);
      svint32_t acc1_1 = svdup_s32(0);
      svint32_t acc1_2 = svdup_s32(0);
      svint32_t acc1_3 = svdup_s32(0);
      svint32_t acc2_0 = svdup_s32(0);
      svint32_t acc2_1 = svdup_s32(0);
      svint32_t acc2_2 = svdup_s32(0);
      svint32_t acc2_3 = svdup_s32(0);
      svint32_t acc3_0 = svdup_s32(0);
      svint32_t acc3_1 = svdup_s32(0);
      svint32_t acc3_2 = svdup_s32(0);
      svint32_t acc3_3 = svdup_s32(0);
      svint32_t acc4_0 = svdup_s32(0);
      svint32_t acc4_1 = svdup_s32(0);
      svint32_t acc4_2 = svdup_s32(0);
      svint32_t acc4_3 = svdup_s32(0);
      svint32_t acc5_0 = svdup_s32(0);
      svint32_t acc5_1 = svdup_s32(0);
      svint32_t acc5_2 = svdup_s32(0);
      svint32_t acc5_3 = svdup_s32(0);

      // K loop
      for(uint64_t k_idx = 0; k_idx < k; k_idx += 8) {
        const int8_t* const a_ptr = a + k_idx * m + 8 * m_idx;
        const int8_t* const b_ptr = b + k_idx * n + 8 * n_idx;

	svint8_t vb_0 = svld1_vnum_s8(pg, b_ptr, 0);
	svint8_t vb_1 = svld1_vnum_s8(pg, b_ptr, 1);
	svint8_t vb_2 = svld1_vnum_s8(pg, b_ptr, 2);
	svint8_t vb_3 = svld1_vnum_s8(pg, b_ptr, 3);

        svint8_t va_0 = svld1rq_s8(pg, a_ptr);
	acc0_0 = svmmla_s32(acc0_0, va_0, vb_0);
	acc0_1 = svmmla_s32(acc0_1, va_0, vb_1);
	acc0_2 = svmmla_s32(acc0_2, va_0, vb_2);
	acc0_3 = svmmla_s32(acc0_3, va_0, vb_3);

        svint8_t va_1 = svld1rq_s8(pg, a_ptr + 16);
	acc1_0 = svmmla_s32(acc1_0, va_1, vb_0);
	acc1_1 = svmmla_s32(acc1_1, va_1, vb_1);
	acc1_2 = svmmla_s32(acc1_2, va_1, vb_2);
	acc1_3 = svmmla_s32(acc1_3, va_1, vb_3);

        svint8_t va_2 = svld1rq_s8(pg, a_ptr + 32);
	acc2_0 = svmmla_s32(acc2_0, va_2, vb_0);
	acc2_1 = svmmla_s32(acc2_1, va_2, vb_1);
	acc2_2 = svmmla_s32(acc2_2, va_2, vb_2);
	acc2_3 = svmmla_s32(acc2_3, va_2, vb_3);

        svint8_t va_3 = svld1rq_s8(pg, a_ptr + 48);
	acc3_0 = svmmla_s32(acc3_0, va_3, vb_0);
	acc3_1 = svmmla_s32(acc3_1, va_3, vb_1);
	acc3_2 = svmmla_s32(acc3_2, va_3, vb_2);
	acc3_3 = svmmla_s32(acc3_3, va_3, vb_3);

        svint8_t va_4 = svld1rq_s8(pg, a_ptr + 64);
	acc4_0 = svmmla_s32(acc4_0, va_4, vb_0);
	acc4_1 = svmmla_s32(acc4_1, va_4, vb_1);
	acc4_2 = svmmla_s32(acc4_2, va_4, vb_2);
	acc4_3 = svmmla_s32(acc4_3, va_4, vb_3);

        svint8_t va_5 = svld1rq_s8(pg, a_ptr + 80);
	acc5_0 = svmmla_s32(acc5_0, va_5, vb_0);
	acc5_1 = svmmla_s32(acc5_1, va_5, vb_1);
	acc5_2 = svmmla_s32(acc5_2, va_5, vb_2);
	acc5_3 = svmmla_s32(acc5_3, va_5, vb_3);
      }

      // Store
      int32_t* const c0ptr = c + m_idx * n + n_idx;
      int32_t* const c1ptr = c0ptr + svcntw();

      svint64_t res_0l = svuzp1_s64(svreinterpret_s64_s32(acc0_0), svreinterpret_s64_s32(acc0_1));
      svint64_t res_0h = svuzp1_s64(svreinterpret_s64_s32(acc0_2), svreinterpret_s64_s32(acc0_3));
      svint64_t res_1l = svuzp2_s64(svreinterpret_s64_s32(acc0_0), svreinterpret_s64_s32(acc0_1));
      svint64_t res_1h = svuzp2_s64(svreinterpret_s64_s32(acc0_2), svreinterpret_s64_s32(acc0_3));
      svint64_t res_2l = svuzp1_s64(svreinterpret_s64_s32(acc1_0), svreinterpret_s64_s32(acc1_1));
      svint64_t res_2h = svuzp1_s64(svreinterpret_s64_s32(acc1_2), svreinterpret_s64_s32(acc1_3));
      svint64_t res_3l = svuzp2_s64(svreinterpret_s64_s32(acc1_0), svreinterpret_s64_s32(acc1_1));
      svint64_t res_3h = svuzp2_s64(svreinterpret_s64_s32(acc1_2), svreinterpret_s64_s32(acc1_3));

      svst1_s32(pg, c0ptr, svreinterpret_s32_s64(res_0l));
      svst1_s32(pg, c1ptr, svreinterpret_s32_s64(res_0h));
      svst1_s32(pg, c0ptr + n, svreinterpret_s32_s64(res_1l));
      svst1_s32(pg, c1ptr + n, svreinterpret_s32_s64(res_1h));
      svst1_s32(pg, c0ptr + 2 * n, svreinterpret_s32_s64(res_2l));
      svst1_s32(pg, c1ptr + 2 * n, svreinterpret_s32_s64(res_2h));
      svst1_s32(pg, c0ptr + 3 * n, svreinterpret_s32_s64(res_3l));
      svst1_s32(pg, c1ptr + 3 * n, svreinterpret_s32_s64(res_3h));

      svint64_t res_4l = svuzp1_s64(svreinterpret_s64_s32(acc2_0), svreinterpret_s64_s32(acc2_1));
      svint64_t res_4h = svuzp1_s64(svreinterpret_s64_s32(acc2_2), svreinterpret_s64_s32(acc2_3));
      svint64_t res_5l = svuzp2_s64(svreinterpret_s64_s32(acc2_0), svreinterpret_s64_s32(acc2_1));
      svint64_t res_5h = svuzp2_s64(svreinterpret_s64_s32(acc2_2), svreinterpret_s64_s32(acc2_3));
      svint64_t res_6l = svuzp1_s64(svreinterpret_s64_s32(acc3_0), svreinterpret_s64_s32(acc3_1));
      svint64_t res_6h = svuzp1_s64(svreinterpret_s64_s32(acc3_2), svreinterpret_s64_s32(acc3_3));
      svint64_t res_7l = svuzp2_s64(svreinterpret_s64_s32(acc3_0), svreinterpret_s64_s32(acc3_1));
      svint64_t res_7h = svuzp2_s64(svreinterpret_s64_s32(acc3_2), svreinterpret_s64_s32(acc3_3));

      svst1_s32(pg, c0ptr + 4 * n, svreinterpret_s32_s64(res_4l));
      svst1_s32(pg, c1ptr + 4 * n, svreinterpret_s32_s64(res_4h));
      svst1_s32(pg, c0ptr + 5 * n, svreinterpret_s32_s64(res_5l));
      svst1_s32(pg, c1ptr + 5 * n, svreinterpret_s32_s64(res_5h));
      svst1_s32(pg, c0ptr + 6 * n, svreinterpret_s32_s64(res_6l));
      svst1_s32(pg, c1ptr + 6 * n, svreinterpret_s32_s64(res_6h));
      svst1_s32(pg, c0ptr + 7 * n, svreinterpret_s32_s64(res_7l));
      svst1_s32(pg, c1ptr + 7 * n, svreinterpret_s32_s64(res_7h));

      svint64_t res_8l = svuzp1_s64(svreinterpret_s64_s32(acc4_0), svreinterpret_s64_s32(acc4_1));
      svint64_t res_8h = svuzp1_s64(svreinterpret_s64_s32(acc4_2), svreinterpret_s64_s32(acc4_3));
      svint64_t res_9l = svuzp2_s64(svreinterpret_s64_s32(acc4_0), svreinterpret_s64_s32(acc4_1));
      svint64_t res_9h = svuzp2_s64(svreinterpret_s64_s32(acc4_2), svreinterpret_s64_s32(acc4_3));
      svint64_t res_10l = svuzp1_s64(svreinterpret_s64_s32(acc5_0), svreinterpret_s64_s32(acc5_1));
      svint64_t res_10h = svuzp1_s64(svreinterpret_s64_s32(acc5_2), svreinterpret_s64_s32(acc5_3));
      svint64_t res_11l = svuzp2_s64(svreinterpret_s64_s32(acc5_0), svreinterpret_s64_s32(acc5_1));
      svint64_t res_11h = svuzp2_s64(svreinterpret_s64_s32(acc5_2), svreinterpret_s64_s32(acc5_3));

      svst1_s32(pg, c0ptr + 8 * n, svreinterpret_s32_s64(res_8l));
      svst1_s32(pg, c1ptr + 8 * n, svreinterpret_s32_s64(res_8h));
      svst1_s32(pg, c0ptr + 9 * n, svreinterpret_s32_s64(res_9l));
      svst1_s32(pg, c1ptr + 9 * n, svreinterpret_s32_s64(res_9h));
      svst1_s32(pg, c0ptr + 10 * n, svreinterpret_s32_s64(res_10l));
      svst1_s32(pg, c1ptr + 10 * n, svreinterpret_s32_s64(res_10h));
      svst1_s32(pg, c0ptr + 11 * n, svreinterpret_s32_s64(res_11l));
      svst1_s32(pg, c1ptr + 11 * n, svreinterpret_s32_s64(res_11h));
    }
  }

}


#elif (defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE_MATMUL_INT8))

static void inner_loop_135(struct loop_135_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + (m * k);

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
      "   ptrue   pn8.b                                             \n"
      "   ptrue   pn9.s                                             \n"
#endif
      "   ptrue   p0.b                                              \n"

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
      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], #16]                     \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[b_ptr]]                    \n"
#else
      "   ld1b    {z4.b}, p0/z, [%[b_ptr], #0, mul vl]              \n"
      "   ld1b    {z5.b}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   ld1b    {z6.b}, p0/z, [%[b_ptr], #2, mul vl]              \n"
      "   ld1b    {z7.b}, p0/z, [%[b_ptr], #3, mul vl]              \n"
#endif
      "   smmla   z16.s, z0.b, z4.b                                 \n"
      "   smmla   z20.s, z0.b, z5.b                                 \n"
      "   smmla   z24.s, z0.b, z6.b                                 \n"
      "   smmla   z28.s, z0.b, z7.b                                 \n"

      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   ld1rqb  {z2.b}, p0/z, [%[a_ptr], #32]                     \n"
      "   smmla   z17.s, z1.b, z4.b                                 \n"
      "   smmla   z21.s, z1.b, z5.b                                 \n"
      "   smmla   z25.s, z1.b, z6.b                                 \n"
      "   smmla   z29.s, z1.b, z7.b                                 \n"

      "   ld1rqb  {z3.b}, p0/z, [%[a_ptr], #48]                     \n"
      "   smmla   z18.s, z2.b, z4.b                                 \n"
      "   smmla   z22.s, z2.b, z5.b                                 \n"
      "   smmla   z26.s, z2.b, z6.b                                 \n"
      "   smmla   z30.s, z2.b, z7.b                                 \n"

      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr], #64]                     \n"
      "   smmla   z19.s, z3.b, z4.b                                 \n"
      "   smmla   z23.s, z3.b, z5.b                                 \n"
      "   smmla   z27.s, z3.b, z6.b                                 \n"
      "   smmla   z31.s, z3.b, z7.b                                 \n"

      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], #80]                     \n"
      "   smmla   z8.s,  z0.b, z4.b                                 \n"
      "   smmla   z10.s, z0.b, z5.b                                 \n"
      "   smmla   z12.s, z0.b, z6.b                                 \n"
      "   smmla   z14.s, z0.b, z7.b                                 \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   smmla   z9.s,  z1.b, z4.b                                 \n"
      "   smmla   z11.s, z1.b, z5.b                                 \n"
      "   smmla   z13.s, z1.b, z6.b                                 \n"
      "   smmla   z15.s, z1.b, z7.b                                 \n"
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
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18",
        "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
        "z29", "z30", "z31", "p0", "cc", "memory");
}

#elif (defined(__ARM_NEON) && defined(__ARM_FEATURE_SVE_MATMUL_INT8))

static void inner_loop_135(struct loop_135_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t c_inc = n * 4;
  register uint64_t a_cnd = (uint64_t)&data->a[m * k];

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t a1ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   movi    v8.4s, #0                                         \n"
      "   movi    v9.4s, #0                                         \n"
      "   movi    v10.4s, #0                                        \n"
      "   movi    v11.4s, #0                                        \n"
      "   movi    v12.4s, #0                                        \n"
      "   movi    v13.4s, #0                                        \n"
      "   movi    v14.4s, #0                                        \n"
      "   movi    v15.4s, #0                                        \n"
      "   movi    v16.4s, #0                                        \n"
      "   movi    v17.4s, #0                                        \n"
      "   movi    v18.4s, #0                                        \n"
      "   movi    v19.4s, #0                                        \n"
      "   movi    v20.4s, #0                                        \n"
      "   movi    v21.4s, #0                                        \n"
      "   movi    v22.4s, #0                                        \n"
      "   movi    v23.4s, #0                                        \n"
      "   movi    v24.4s, #0                                        \n"
      "   movi    v25.4s, #0                                        \n"
      "   movi    v26.4s, #0                                        \n"
      "   movi    v27.4s, #0                                        \n"
      "   movi    v28.4s, #0                                        \n"
      "   movi    v29.4s, #0                                        \n"
      "   movi    v30.4s, #0                                        \n"
      "   movi    v31.4s, #0                                        \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1     {v0.16b,v1.16b,v2.16b,v3.16b}, [%[a_ptr]]         \n"
      "   ld1     {v4.16b,v5.16b,v6.16b,v7.16b}, [%[b_ptr]]         \n"
      "   smmla   v16.4s, v0.16b, v4.16b                            \n"
      "   smmla   v20.4s, v0.16b, v5.16b                            \n"
      "   smmla   v24.4s, v0.16b, v6.16b                            \n"
      "   smmla   v28.4s, v0.16b, v7.16b                            \n"

      "   add     %[a1ptr], %[a_ptr], #64                           \n"
      "   smmla   v17.4s, v1.16b, v4.16b                            \n"
      "   smmla   v21.4s, v1.16b, v5.16b                            \n"
      "   smmla   v25.4s, v1.16b, v6.16b                            \n"
      "   smmla   v29.4s, v1.16b, v7.16b                            \n"

      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   smmla   v18.4s, v2.16b, v4.16b                            \n"
      "   smmla   v22.4s, v2.16b, v5.16b                            \n"
      "   smmla   v26.4s, v2.16b, v6.16b                            \n"
      "   smmla   v30.4s, v2.16b, v7.16b                            \n"

      "   ld1     {v0.16b,v1.16b}, [%[a1ptr]]                       \n"
      "   smmla   v19.4s, v3.16b, v4.16b                            \n"
      "   smmla   v23.4s, v3.16b, v5.16b                            \n"
      "   smmla   v27.4s, v3.16b, v6.16b                            \n"
      "   smmla   v31.4s, v3.16b, v7.16b                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   smmla   v8.4s,  v0.16b, v4.16b                            \n"
      "   smmla   v10.4s, v0.16b, v5.16b                            \n"
      "   smmla   v12.4s, v0.16b, v6.16b                            \n"
      "   smmla   v14.4s, v0.16b, v7.16b                            \n"

      "   smmla   v9.4s,  v1.16b, v4.16b                            \n"
      "   smmla   v11.4s, v1.16b, v5.16b                            \n"
      "   smmla   v13.4s, v1.16b, v6.16b                            \n"
      "   smmla   v15.4s, v1.16b, v7.16b                            \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   uzp1    v0.2d, v16.2d, v20.2d                             \n"
      "   uzp1    v1.2d, v24.2d, v28.2d                             \n"
      "   uzp2    v2.2d, v16.2d, v20.2d                             \n"
      "   uzp2    v3.2d, v24.2d, v28.2d                             \n"
      "   uzp1    v4.2d, v17.2d, v21.2d                             \n"
      "   uzp1    v5.2d, v25.2d, v29.2d                             \n"
      "   uzp2    v6.2d, v17.2d, v21.2d                             \n"
      "   uzp2    v7.2d, v25.2d, v29.2d                             \n"
      "   st1     {v0.4s,v1.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v2.4s,v3.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v4.4s,v5.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v6.4s,v7.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   uzp1    v0.2d, v18.2d, v22.2d                             \n"
      "   uzp1    v1.2d, v26.2d, v30.2d                             \n"
      "   uzp2    v2.2d, v18.2d, v22.2d                             \n"
      "   uzp2    v3.2d, v26.2d, v30.2d                             \n"
      "   uzp1    v4.2d, v19.2d, v23.2d                             \n"
      "   uzp1    v5.2d, v27.2d, v31.2d                             \n"
      "   uzp2    v6.2d, v19.2d, v23.2d                             \n"
      "   uzp2    v7.2d, v27.2d, v31.2d                             \n"
      "   st1     {v0.4s,v1.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v2.4s,v3.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v4.4s,v5.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v6.4s,v7.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   uzp1    v16.2d, v8.2d,  v10.2d                            \n"
      "   uzp1    v17.2d, v12.2d, v14.2d                            \n"
      "   uzp2    v20.2d, v8.2d,  v10.2d                            \n"
      "   uzp2    v21.2d, v12.2d, v14.2d                            \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   uzp1    v18.2d, v9.2d,  v11.2d                            \n"
      "   uzp1    v19.2d, v13.2d, v15.2d                            \n"
      "   uzp2    v22.2d, v9.2d,  v11.2d                            \n"
      "   uzp2    v23.2d, v13.2d, v15.2d                            \n"
      "   st1     {v18.4s,v19.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[c_inc]             \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                            \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [a1ptr] "=&r"(a1ptr), [b_ptr] "=&r"(b_ptr),
        [c_ptr] "=&r"(c_ptr), [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [c_inc] "r"(c_inc), [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "cc", "memory");
}

#else

static void inner_loop_135(struct loop_135_data *data) {
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
// Default of 192KiB equates to original problem size (M=256, K=256, N=512)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 197
#endif

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n, k) ((k) * ((m) + (n)) * sizeof(int8_t))

LOOP_DECL(135, NS_SVE_LOOP_ATTR) {
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 24
  uint64_t N = 0;  // multiple of 4*SVLs
  uint64_t K = 0;  // multiple of 8

  // For this loop, K should equal to N * 6
  const uint64_t K_base = MAX_VL / 8;
  uint64_t m = 5 * 24;

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

  struct loop_135_data data = {
      .m = M,
      .n = N,
      .k = K,
  };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_int8(data.a, M * K);
  fill_int8(data.b, K * N);

  inner_loops_135(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N, K) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                \
  {                                                \
    int32_t d = 0;                                 \
    for (int k = 0; k < K; k += 8) {               \
      d += int32_dot8(x, y, k, &data);             \
    }                                              \
    checksum += (int)(d != data.c[(x) * N + (y)]); \
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
  FINALISE_LOOP_I(135, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
