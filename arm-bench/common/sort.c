/*----------------------------------------------------------------------------
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "sort.h"

// Used by vector quicksort as tidying step
#if defined(HAVE_SVE_INTRINSICS)  // OET
void com_sort_oet(uint32_t n, int32_t *restrict data, uint32_t partition) {
  uint32_t k = n - 1;
  for (uint32_t i = 0; i < partition; i++) {
    for (uint32_t j = i % 2; j < k; j += 2 * svcntw()) {
      svbool_t p_loop = svwhilelt_b32(0u, (n - j) / 2);

      svint32x2_t a01 = svld2(p_loop, data + j);
      svint32_t a0 = svget2(a01, 0);
      svint32_t a1 = svget2(a01, 1);

      svbool_t p_comp = svcmplt(p_loop, a0, a1);
      svint32_t b0 = svsel(p_comp, a0, a1);
      svint32_t b1 = svsel(p_comp, a1, a0);

      svst2(p_loop, data + j, svcreate2(b0, b1));
    }
  }
}
#elif defined(__ARM_FEATURE_SVE)  // OET
void com_sort_oet(uint32_t n, int32_t *restrict data, uint32_t partition) {
  uint32_t mvl2 = 0;  // 2*MVL in int32_t elements
  asm volatile(
      "addvl  %x0, %x0, #1    \n"
      "asr    %x0, %x0, #1    \n"
      : "+r"(mvl2));

  for (uint32_t i = 0; i < partition; i++) {
    uint32_t j = i % 2;
    uint32_t remaining = n - j;
    while (remaining >= 2) {
      uint32_t vals_per_iteration = min2_32u(remaining, mvl2);
      uint32_t vals_per_register = vals_per_iteration / 2;
      asm volatile(
          "whilelt    p1.s, wzr, %w[vpr]                              \n"
          "ld2w       {z0.s, z1.s}, p1/z, [%x[data], %x[j], lsl #2]   \n"
          "cmplt      p2.s, p1/z, z0.s, z1.s                          \n"
          "sel        z2.s, p2, z0.s, z1.s                            \n"
          "sel        z3.s, p2, z1.s, z0.s                            \n"
          "st2w       {z2.s, z3.s}, p1  , [%x[data], %x[j], lsl #2]   \n"
          :
          : [data] "r"(data), [j] "r"(j), [vpr] "r"(vals_per_register)
          : "v0", "v1", "v2", "v3", "p1", "p2", "cc", "memory");
      j += vals_per_iteration;
      remaining -= vals_per_iteration;
    }
  }
}
#endif                   // OET

// Used by scalar quicksort as tidying step
void com_sort_insertion(uint32_t n, int32_t *restrict data) {
  uint32_t i, j;
  for (i = 1; i < n; i++) {
    int32_t tmp = data[i];
    for (j = i; j > 0 && data[j - 1] > tmp; j--) {
      data[j] = data[j - 1];
    }
    data[j] = tmp;
  }
}

void com_sort_radix(uint32_t n, int32_t *restrict data,
                    int32_t *restrict temp) {
  int32_t min = data[0];
  int32_t max = data[0];
  for (uint32_t i = 1; i < n; i++) {
    min = (min < data[i]) ? min : data[i];
    max = (max > data[i]) ? max : data[i];
  }

  // Shift all values by the minimum to avoid dealing with -ve numbers
  for (uint32_t i = 0; i < n; i++) data[i] -= min;

  // Determine the number of passes from the (shifted) maximum
  uint32_t lim = max - min;
#define BIT_STRIDE 8  // 256 bins
#define BIN_INDEX(i, o) (((uint32_t)data[i] >> o) % (1 << BIT_STRIDE))

  for (int off = 0; lim != 0; off += BIT_STRIDE, lim >>= BIT_STRIDE) {
    // Initialise all bins
    uint32_t count[1 << BIT_STRIDE] = {0};

    // Count occurrences of each digit
    for (uint32_t i = 0; i < n; i++) count[BIN_INDEX(i, off)]++;

    // Cumulatively sum the bins so that each tracks the output index
    uint32_t count_tmp = count[0];
    for (int j = 1; j < (1 << BIT_STRIDE); j++) {
      count[j] += count_tmp;
      count_tmp = count[j];
    }

    // Construct the output array in the scratch buffer
    for (uint32_t i = n; i > 0; i--)
      temp[--count[BIN_INDEX(i - 1, off)]] = data[i - 1];

    // Copy across
    memcpy(data, temp, sizeof(data[0]) * n);
  }

  // Add back the offset to obtain the original values
  for (uint32_t i = 0; i < n; i++) data[i] += min;
}

#if defined(HAVE_SVE_INTRINSICS)  // Radix sort helpers
radix_t find_radix_sub_offset(uint32_t n, int32_t *restrict data) {
  svint32_t acc_min, acc_max;
  svint32_t vec_off, vec_val;

  svbool_t q = svwhilelt_b32(0u, n);
  acc_min = svld1(q, data);
  acc_max = acc_min;

  uint32_t i;
  svbool_t p;

  FOR_LOOP_32(, i, 0, n, p) {
    vec_val = svld1(p, data + i);
    acc_min = svmin_m(p, acc_min, vec_val);
    acc_max = svmax_m(p, acc_max, vec_val);
  }

  int32_t min = svminv(q, acc_min);
  int32_t max = svmaxv(q, acc_max);
  radix_t result = {
      .bits = sizeof(int32_t) * 8 - __builtin_clz(max - min),
      .offset = min,
  };
  vec_off = svdup_s32(result.offset);

  FOR_LOOP_32(, i, 0, n, p) {
    vec_val = svld1(p, data + i);
    vec_val = svsub_m(p, vec_val, vec_off);
    svst1(p, data + i, vec_val);
  }

  return result;
}

void post_add_offset(uint32_t n, int32_t *restrict data, int32_t off) {
  svint32_t vec_val, vec_off = svdup_s32(off);

  svbool_t p;
  FOR_LOOP_32(uint32_t, i, 0, n, p) {
    vec_val = svld1(p, data + i);
    vec_val = svadd_m(p, vec_val, vec_off);
    svst1(p, data + i, vec_val);
  }
}
#elif defined(__ARM_FEATURE_SVE)  // Radix sort helpers
radix_t find_radix_sub_offset(uint32_t n, int32_t *restrict data) {
  uint32_t min, max, rdx, i;

  i = 0;
  asm volatile(
      "   whilelt p0.s, %w[i], %w[n]                      \n"
      "   ld1w    z1.s, p0/z, [%[data], %x[i], lsl #2]    \n"
      "   mov     z2.d, z1.d                              \n"
      "   incw    %x[i]                                    \n"
      "   whilelt p1.s, %w[i], %w[n]                      \n"
      "   b.none  2f                                      \n"
      "1: ld1w    z0.s, p1/z, [%[data], %x[i], lsl #2]    \n"
      "   smax    z1.s, p1/m, z1.s, z0.s                  \n"
      "   smin    z2.s, p1/m, z2.s, z0.s                  \n"
      "   incw    %x[i]                                   \n"
      "   whilelt p1.s, %w[i], %w[n]                      \n"
      "   b.first 1b                                      \n"
      "2: smaxv   s1, p0, z1.s                            \n"
      "   sminv   s2, p0, z2.s                            \n"
      "   fmov    %w[max], s1                             \n"
      "   fmov    %w[min], s2                             \n"
      "   sub     %w[rdx], %w[max], %w[min]               \n"
      "   clz     %w[rdx], %w[rdx]                        \n"
      : [i] "+r"(i), [min] "=&r"(min), [max] "=&r"(max), [rdx] "=&r"(rdx)
      : [n] "r"(n), [data] "r"(data)
      : "z0", "z1", "z2", "p0", "p1", "cc", "memory");

  radix_t result = {
      .bits = sizeof(int32_t) * 8 - rdx,
      .offset = min,
  };

  i = 0;
  asm volatile(
      "   dup     z1.s, %w[off]                       \n"
      "   whilelt p0.s, %w[i], %w[n]                  \n"
      "   b.none  4f                                  \n"
      "3:                                             \n"
      "   ld1w    z0.s, p0/z, [%[data], %x[i], lsl #2]\n"
      "   sub     z0.s, z0.s, z1.s                    \n"
      "   st1w    z0.s, p0  , [%[data], %x[i], lsl #2]\n"
      "   incw    %x[i]                               \n"
      "   whilelt p0.s, %w[i], %w[n]                  \n"
      "   b.first 3b                                  \n"
      "4:                                             \n"
      : [i] "+r"(i)
      : [n] "r"(n), [data] "r"(data), [off] "r"(result.offset)
      : "z0", "z1", "p0", "cc", "memory");

  return result;
}

void post_add_offset(uint32_t n, int32_t *restrict data, int32_t off) {
  uint32_t i = 0;
  asm volatile(
      "   dup     z1.s, %w[off]                       \n"
      "   whilelt p0.s, %w[i], %w[n]                  \n"
      "   b.none  2f                                  \n"
      "1:                                             \n"
      "   ld1w    z0.s, p0/z, [%[data], %x[i], lsl #2]\n"
      "   add     z0.s, z0.s, z1.s                    \n"
      "   st1w    z0.s, p0  , [%[data], %x[i], lsl #2]\n"
      "   incw    %x[i]                               \n"
      "   whilelt p0.s, %w[i], %w[n]                  \n"
      "   b.first 1b                                  \n"
      "2:                                             \n"
      : [i] "+r"(i)
      : [data] "r"(data), [off] "r"(off), [n] "r"(n)
      : "z0", "z1", "p0", "cc", "memory");
}
#endif                   // Radix sort helpers
