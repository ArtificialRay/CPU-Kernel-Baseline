/*----------------------------------------------------------------------------
#
#   Loop 222: FP16 convolution
#
#   Purpose:
#     Use of mutli-vector LD and f16 MLA instructions.
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
    Input image: row-major, horizontally zero-padded
    Output image: row-major
    Kernel: vector
  Constraints -
    M: multiple of SVLs
    N: multiple of SVLh*4
*/

struct loop_222_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float16_t *restrict kernel;
  float16_t *restrict values;
  float16_t *restrict buffer;
  float16_t *restrict result;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_222(struct loop_222_data *restrict data) {
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
#elif defined(__ARM_NEON)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif




// Each row is zero-padded on both ends in order to simplify boundary logic
// Each pair of consecutive rows share padding so as to reduce memory usage
#define IMAGE_BORDER(k)     ((k) / 2)
#define IMAGE_STRIDE(k,n)   ((n) + (k) - IMAGE_BORDER(k))

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define CONVOLVE_INC_ROWS   4
#define CONVOLVE_INC_COLS   1

static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 4)
{
  FLOAT16_t acc_0 = 0.0f;
  FLOAT16_t acc_1 = 0.0f;
  FLOAT16_t acc_2 = 0.0f;
  FLOAT16_t acc_3 = 0.0f;

  for (; k_ptr < k_cnd; k_ptr++, a_ptr += a_inc) {
    FLOAT16_t k = fp16_to_native(k_ptr[0]);
    acc_0  += k * fp16_to_native(a_ptr[a_str * 0]);
    acc_1  += k * fp16_to_native(a_ptr[a_str * 1]);
    acc_2  += k * fp16_to_native(a_ptr[a_str * 2]);
    acc_3  += k * fp16_to_native(a_ptr[a_str * 3]);
  }

  b_ptr[b_inc * 0] = native_to_fp16(acc_0);
  b_ptr[b_inc * 1] = native_to_fp16(acc_1);
  b_ptr[b_inc * 2] = native_to_fp16(acc_2);
  b_ptr[b_inc * 3] = native_to_fp16(acc_3);
}

#elif (defined(HAVE_SME_INTRINSICS) && defined(__ARM_FEATURE_SME2p1))

#define CONVOLVE_INC_ROWS   (svl_h / 2)
#define CONVOLVE_INC_COLS   (svl_h * 4)

static inline void convolve(
  float16_t *a_src,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert((n_row <= SVL_s) && ((n_row % 4) == 0))
LOOP_ATTR
{
  svcount_t c_all = svptrue_c16();

  svfloat16_t   kernel;
  svuint64x4_t  str_0, str_1, str_2, str_3;
  uint64_t l_idx;
  float16_t *a_ptr;

#define CAST(q,p) svreinterpret_f16(svget4(str_##q, p))
#define QUAD(q)   svcreate4(CAST(q, 0), CAST(q, 1), CAST(q, 2), CAST(q, 3))

  svzero_za();
  for (; k_ptr < k_cnd; k_ptr++, a_src += a_inc) {
    kernel = svdup_f16(k_ptr[0]);
    a_ptr = a_src;
    for (l_idx = 0; l_idx < n_row; l_idx++, a_ptr += a_str) {
      svfloat16x4_t ldr = svld1_x4(c_all, a_ptr);
      svmla_lane_za16_vg1x4(l_idx, ldr, kernel, 0);
    }
  }
  for (l_idx = 0; l_idx < n_row; l_idx += 4, b_ptr += b_inc * 4) {
    str_0 = svread_za64_u64_vg1x4(l_idx + 0);
    str_1 = svread_za64_u64_vg1x4(l_idx + 1);
    str_2 = svread_za64_u64_vg1x4(l_idx + 2);
    str_3 = svread_za64_u64_vg1x4(l_idx + 3);
    svst1(c_all, &b_ptr[b_inc * 0], QUAD(0));
    svst1(c_all, &b_ptr[b_inc * 1], QUAD(1));
    svst1(c_all, &b_ptr[b_inc * 2], QUAD(2));
    svst1(c_all, &b_ptr[b_inc * 3], QUAD(3));
  }
}

#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE2p1))

#define CONVOLVE_INC_ROWS   1
#define CONVOLVE_INC_COLS   (svl_h * 4)

static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 1)
LOOP_ATTR
{
  svcount_t c_all = svptrue_c16();

  svfloat16_t acc_0 = svdup_f16(0.0f);
  svfloat16_t acc_1 = svdup_f16(0.0f);
  svfloat16_t acc_2 = svdup_f16(0.0f);
  svfloat16_t acc_3 = svdup_f16(0.0f);

  for (; k_ptr < k_cnd; k_ptr++, a_ptr += a_inc) {
    svfloat16_t   lda = svdup_f16(k_ptr[0]);
    svfloat16x4_t ldb = svld1_x4(c_all, a_ptr);
    acc_0 = svmla_lane(acc_0, svget4(ldb, 0), lda, 0);
    acc_1 = svmla_lane(acc_1, svget4(ldb, 1), lda, 0);
    acc_2 = svmla_lane(acc_2, svget4(ldb, 2), lda, 0);
    acc_3 = svmla_lane(acc_3, svget4(ldb, 3), lda, 0);
  }

  svst1(c_all, b_ptr, svcreate4(acc_0, acc_1, acc_2, acc_3));
}

#elif defined(__ARM_FEATURE_SME2p1)

#define CONVOLVE_INC_ROWS   (svl_h / 2)
#define CONVOLVE_INC_COLS   (svl_h * 4)

static inline void convolve(
  float16_t *a_src,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert((n_row <= SVL_s) && ((n_row % 4) == 0))
LOOP_ATTR
{
  register uint64_t a_ptr;
  register uint64_t w2off = b_inc * 2;
  register uint64_t w3off = b_inc * 3;
  // x8: slice index register for fmla and mova

  asm volatile(
    "   ptrue   p0.h                                                    \n"
    "   ptrue   pn8.h                                                   \n"
    "   zero    {za}                                                    \n"

    "1:                                                                 \n"
    "   ld1rh   {z0.h}, p0/z, [%[k_ptr]]                                \n"
    "   mov     %[a_ptr], %[a_src]                                      \n"
    "   mov     x8, #0                                                  \n"
    "2:                                                                 \n"
    "   ld1h    {z16.h-z19.h}, pn8/z, [%[a_ptr]]                        \n"
    "   fmla    za.h[w8, 0, vgx4], {z16.h-z19.h}, z0.h[0]               \n"
    "   add     x8, x8, #1                                              \n"
    "   add     %[a_ptr], %[a_ptr], %[a_str], lsl #1                    \n"
    "   cmp     x8, %[n_row]                                            \n"
    "   b.mi    2b                                                      \n"
    "   add     %[k_ptr], %[k_ptr], #2                                  \n"
    "   add     %[a_src], %[a_src], %[a_inc], lsl #1                    \n"
    "   cmp     %[k_ptr], %[k_cnd]                                      \n"
    "   b.mi    1b                                                      \n"

    "   mov     x8, #0                                                  \n"
    "3:                                                                 \n"
    "   mova    {z16.d-z19.d}, za.d[w8, 0, vgx4]                        \n"
    "   mova    {z20.d-z23.d}, za.d[w8, 1, vgx4]                        \n"
    "   mova    {z24.d-z27.d}, za.d[w8, 2, vgx4]                        \n"
    "   mova    {z28.d-z31.d}, za.d[w8, 3, vgx4]                        \n"
    "   st1h    {z16.h-z19.h}, pn8, [%[b_ptr]]                          \n"
    "   st1h    {z20.h-z23.h}, pn8, [%[b_ptr], %[w1off], lsl #1]        \n"
    "   st1h    {z24.h-z27.h}, pn8, [%[b_ptr], %[w2off], lsl #1]        \n"
    "   st1h    {z28.h-z31.h}, pn8, [%[b_ptr], %[w3off], lsl #1]        \n"
    "   add     x8, x8, #4                                              \n"
    "   add     %[b_ptr], %[b_ptr], %[b_inc], lsl #3                    \n"
    "   cmp     x8, %[n_row]                                            \n"
    "   b.mi    3b                                                      \n"

    : [a_ptr] "=&r"(a_ptr), [b_ptr] "+&r"(b_ptr), [k_ptr] "+&r"(k_ptr),
      [a_src] "+&r"(a_src)
    : [a_str] "r"(a_str), [a_inc] "r"(a_inc), [b_inc] "r"(b_inc),
      [w1off] "r"(b_inc), [w2off] "r"(w2off), [w3off] "r"(w3off),
      [k_cnd] "r"(k_cnd), [n_row] "r"(n_row)
    : "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
      "z26", "z27", "z28", "z29", "z30", "z31", "z0",
      "p0", "p8", "x8",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

#define CONVOLVE_INC_ROWS   1
#define CONVOLVE_INC_COLS   (svl_h * 4)

static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 1)
LOOP_ATTR
{
  asm volatile(
    "   ptrue   p0.h                                                    \n"
    "   ptrue   pn8.h                                                   \n"
    "   mov     z4.h, #0                                                \n"
    "   mov     z5.h, #0                                                \n"
    "   mov     z6.h, #0                                                \n"
    "   mov     z7.h, #0                                                \n"
    "1:                                                                 \n"
    "   ld1rh   {z0.h}, p0/z, [%[k_ptr]]                                \n"
    "   ld1h    {z12.h-z15.h}, pn8/z, [%[a_ptr]]                        \n"
    "   add     %[k_ptr], %[k_ptr], #2                                  \n"
    "   add     %[a_ptr], %[a_ptr], %[a_inc], lsl #1                    \n"
    "   fmla    z4.h, z12.h, z0.h[0]                                    \n"
    "   fmla    z5.h, z13.h, z0.h[0]                                    \n"
    "   fmla    z6.h, z14.h, z0.h[0]                                    \n"
    "   fmla    z7.h, z15.h, z0.h[0]                                    \n"
    "   cmp     %[k_ptr], %[k_cnd]                                      \n"
    "   b.mi    1b                                                      \n"
    "   st1h    {z4.h-z7.h}, pn8, [%[b_ptr]]                            \n"
    : [a_ptr] "+r"(a_ptr), [k_ptr] "+r"(k_ptr)
    : [a_inc] "r"(a_inc), [k_cnd] "r"(k_cnd), [b_ptr] "r"(b_ptr)
    : "z0", "z4", "z5", "z6", "z7", "z12", "z13", "z14", "z15",
      "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

#define CONVOLVE_INC_ROWS   1
#define CONVOLVE_INC_COLS   (svl_h * 4)

static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 1)
LOOP_ATTR
{
  asm volatile(
    "   ptrue   p0.h                                                    \n"
    "   mov     z4.h, #0                                                \n"
    "   mov     z5.h, #0                                                \n"
    "   mov     z6.h, #0                                                \n"
    "   mov     z7.h, #0                                                \n"
    "1:                                                                 \n"
    "   ld1rh   {z0.h}, p0/z, [%[k_ptr]]                                \n"
    "   add     %[k_ptr], %[k_ptr], #2                                  \n"
    "   ld1h    {z12.h}, p0/z, [%[a_ptr], #0, mul vl]                   \n"
    "   ld1h    {z13.h}, p0/z, [%[a_ptr], #1, mul vl]                   \n"
    "   ld1h    {z14.h}, p0/z, [%[a_ptr], #2, mul vl]                   \n"
    "   ld1h    {z15.h}, p0/z, [%[a_ptr], #3, mul vl]                   \n"
    "   add     %[a_ptr], %[a_ptr], %[a_inc], lsl #1                    \n"
    "   fmla    z4.h, z12.h, z0.h[0]                                    \n"
    "   fmla    z5.h, z13.h, z0.h[0]                                    \n"
    "   fmla    z6.h, z14.h, z0.h[0]                                    \n"
    "   fmla    z7.h, z15.h, z0.h[0]                                    \n"
    "   cmp     %[k_ptr], %[k_cnd]                                      \n"
    "   b.mi    1b                                                      \n"
    "   st1h    {z4.h}, p0, [%[b_ptr], #0, mul vl]                      \n"
    "   st1h    {z5.h}, p0, [%[b_ptr], #1, mul vl]                      \n"
    "   st1h    {z6.h}, p0, [%[b_ptr], #2, mul vl]                      \n"
    "   st1h    {z7.h}, p0, [%[b_ptr], #3, mul vl]                      \n"
    : [a_ptr] "+r"(a_ptr), [k_ptr] "+r"(k_ptr)
    : [a_inc] "r"(a_inc), [k_cnd] "r"(k_cnd), [b_ptr] "r"(b_ptr)
    : "z0", "z4", "z5", "z6", "z7", "z12", "z13", "z14", "z15",
      "p0", "cc", "memory");
}

#elif defined(__ARM_NEON)

#define CONVOLVE_INC_ROWS   1
#define CONVOLVE_INC_COLS   32

static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 1)
{
  a_inc = a_inc * 2;
  asm volatile(
    "   movi    v4.16b, #0                                               \n"
    "   movi    v5.16b, #0                                               \n"
    "   movi    v6.16b, #0                                               \n"
    "   movi    v7.16b, #0                                               \n"
    "1:                                                                 \n"
    "   ld1     {v0.h}[0], [%[k_ptr]], #2                               \n"
    "   ld1     {v12.8h,v13.8h,v14.8h,v15.8h}, [%[a_ptr]], %[a_inc]     \n"
    "   fmla    v4.8h, v12.8h, v0.h[0]                                  \n"
    "   fmla    v5.8h, v13.8h, v0.h[0]                                  \n"
    "   fmla    v6.8h, v14.8h, v0.h[0]                                  \n"
    "   fmla    v7.8h, v15.8h, v0.h[0]                                  \n"
    "   cmp     %[k_ptr], %[k_cnd]                                      \n"
    "   b.mi    1b                                                      \n"
    "   st1     {v4.8h,v5.8h,v6.8h,v7.8h}, [%[b_ptr]]                   \n"
    : [a_ptr] "+r"(a_ptr), [k_ptr] "+r"(k_ptr)
    : [a_inc] "r"(a_inc), [k_cnd] "r"(k_cnd), [b_ptr] "r"(b_ptr)
    : "v0", "v4", "v5", "v6", "v7", "v12", "v13", "v14", "v15",
      "cc", "memory");
}

#else

#define CONVOLVE_INC_ROWS   4
#define CONVOLVE_INC_COLS   1
static inline void convolve(
  float16_t *a_ptr,
  float16_t *b_ptr,
  float16_t *k_ptr,
  float16_t *k_cnd,
  uint64_t a_str,
  uint64_t a_inc,
  uint64_t b_inc,
  uint64_t n_row)       // assert(n_row == 1)
{
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}

#endif

static void inner_loop_222(struct loop_222_data *data)
LOOP_ATTR
{
  uint64_t image_height = data->m;
  uint64_t image_width  = data->n;
  uint64_t kernel_width = data->k;
  uint64_t border_width = IMAGE_BORDER(kernel_width);
  uint64_t image_stride = IMAGE_STRIDE(kernel_width, image_width);

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME2)
  uint64_t svl_h;
  asm volatile("cnth %[v]" : [v] "=r"(svl_h)::);
#endif

  uint64_t row_inc = CONVOLVE_INC_ROWS;
  uint64_t col_inc = CONVOLVE_INC_COLS;

  float16_t *kernel_ptr = data->kernel;
  float16_t *kernel_end = &kernel_ptr[kernel_width];

  float16_t *src, *dst;
  uint64_t row, col;

  // Horizontal convolution
  src = data->values;
  dst = data->buffer + (border_width * image_width);  // top border region
  for (row = 0; row < image_height; row += row_inc) {
    for (col = 0; col < image_width; col += col_inc) {
      convolve(&src[col], &dst[col], kernel_ptr, kernel_end,
               image_stride, 1, image_width, row_inc);
    }
    src += row_inc * image_stride;
    dst += row_inc * image_width;
  }

  // Vertical convolution
  src = data->buffer;
  dst = data->result;
  for (row = 0; row < image_height; row += row_inc) {
    for (col = 0; col < image_width; col += col_inc) {
      convolve(&src[col], &dst[col], kernel_ptr, kernel_end,
               image_width, image_width, image_width, row_inc);
    }
    src += row_inc * image_width;
    dst += row_inc * image_width;
  }
}
#endif /* !HAVE_CANDIDATE */

// Reference implementation (scalar, per-pixel)
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
static inline int check_pixel(struct loop_222_data *data, int x, int y) {
  int image_height = data->m;
  int image_width  = data->n;
  int kernel_width = data->k;
  int border_width = IMAGE_BORDER(kernel_width);
  int image_stride = IMAGE_STRIDE(kernel_width, image_width);

  int off = border_width - MIN(border_width, y);
  int lim = MIN(kernel_width, image_height + border_width - y);
  int row, col;

  float16_t *kernel = data->kernel;
  float16_t *bar, *ptr;
  FLOAT16_t acx, acy;

  bar = (void *)&(data->values)[image_stride * (off + y - border_width) + x];
  for (row = off, acy = 0.0f; row < lim; row++) {
    ptr = bar;
    for (col = 0, acx = 0.0f; col < kernel_width; col++)
      acx += fp16_to_native(kernel[col]) * fp16_to_native(ptr[col]);
    acy += fp16_to_native(kernel[row]) * acx;
    bar += image_stride;
  }

  FLOAT16_t val = fp16_to_native(data->result[y * image_width + x]);
  return !check_float(acy, val, 0.01f);
}

// Ensure the maxSVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// (indicates default problem size requested)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 257
#endif

// Element count of input data buffer
#define SIZEOF_DATA(m,n,k)  ((m) * IMAGE_STRIDE(k,n) + IMAGE_BORDER(k))

// Element count of intermediate buffer
#define SIZEOF_TEMP(m,n,k)  ((n) * ((m) + (k)))

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) \
  ( SIZEOF_DATA(m,n,k) + SIZEOF_TEMP(m,n,k) + (k) ) * sizeof(float16_t)

LOOP_DECL(222, OUTER_LOOP_ATTR)
{
  uint64_t M = 0;   // multiple of SVLs
  uint64_t N = 0;   // multiple of SVLh*4
  uint64_t K = 40;  // typical kernel size

  for (int N_base = 4 * (MAX_VL / 16), n = N_base; ; n += N_base) {
    int m = n / 8;
    if (PROBLEM_SIZE_ACTUAL(m,n,K) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
    } else {
      break;
    }
  }

  struct loop_222_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.kernel, K                  , "convolution kernel"  );
  ALLOC_64B(data.values, SIZEOF_DATA(M,N,K) , "input data"          );
  ALLOC_64B(data.buffer, SIZEOF_TEMP(M,N,K) , "intermediate buffer" );
  ALLOC_64B(data.result, M * N              , "result image"        );

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", N = %" PRIu64 ", K = %" PRIu64 "\n", M, N, K);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  fill_fp16(data.kernel, K);
  {
    // Ensure kernel values are non-negative and normalised
    float sum = 0.0f;
    for (int i = 0; i < K; i++)
      sum += fabsf(data.kernel[i] = fp16_to_native(data.kernel[i]));
    for (int i = 0; i < K; i++)
      data.kernel[i] = native_to_fp16(fp16_to_native(data.kernel[i]) / sum);

    // Skip zero-padding regions between rows
    int stride = IMAGE_STRIDE(K, N);
    for (int i = 0, j = IMAGE_BORDER(K); i < M; i++, j += stride)
      fill_fp16(&data.values[j], N);
  }

  inner_loops_222(iters, &data);

  int checksum = 0;
#define CHECK(y,x) checksum += check_pixel(&data, (x), (y))
#ifdef FULL_CHECK
  for (int y = 0; y < M; y++)
    for (int x = 0; x < N; x++)
      CHECK(y, x);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(222, passed, "%d", 0, checksum)
#endif

  return passed ? 0 : 1;
}
