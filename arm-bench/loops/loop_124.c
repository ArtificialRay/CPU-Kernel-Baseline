/*----------------------------------------------------------------------------
#
#   Loop 124: Radix sort
#
#   Purpose:
#     Use of simd instructions in radix sort.
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

struct loop_124_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
  uint32_t *restrict hist;
  uint32_t *restrict prfx;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_124(struct loop_124_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
// Implementation
static void NOINLINE do_sort(struct loop_124_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;
  int32_t *temp = input->temp;

  com_sort_radix(n, data, temp);
}
#elif defined(__ARM_FEATURE_SVE) || defined(HAVE_SVE_INTRINSICS)
#define MAX_BIN 16  // known to be a good "all-purpose" value

static inline SWAP_DECL(swap_u32p, uint32_t *);

static void NOINLINE do_sort(struct loop_124_data *restrict input) {
  uint32_t n = input->n;
  uint32_t mvl;
  asm volatile("cntw %x0" : "=r"(mvl));

  uint32_t num_bins = MAX_BIN;
  if (IS_NOT_POWER_OF_2(num_bins)) {
    printf(" - Error: number of bins must be a power-of-2\n");
    return;
  }
  if (n < mvl) {
    printf(" - Error: buffer size must be greater than VL\n");
    return;
  }

  if (input->hist == NULL)
    ALLOC_64B(input->hist, mvl * num_bins, "histogram");
  if (input->prfx == NULL)
    ALLOC_64B(input->prfx, mvl, "prefix sum table");

  int32_t *data = input->data;
  int32_t *temp = input->temp;
  uint32_t *histogram = input->hist;
  uint32_t *prefixsum = input->prfx;

  radix_t radix = find_radix_sub_offset(n, data);

  uint32_t num_bins_log2 = mylog2_32(num_bins);
  uint32_t num_passes = (radix.bits + num_bins_log2 - 1) / num_bins_log2;
  uint32_t stride = (n + mvl - 1) / mvl;
  uint32_t std_vl = (n + stride - 1) / stride;
  stride = (n + std_vl - 1) / std_vl;
  uint32_t short_iterations = (std_vl - (n % std_vl)) % std_vl;
  uint32_t switching_i = stride - short_iterations;

  uint32_t *data_inp = (uint32_t *)data;
  uint32_t *data_out = (uint32_t *)temp;
  uint32_t *data_ptr, *hist_ptr, i;

#ifdef HAVE_SVE_INTRINSICS  // ACLE/SVE

  svbool_t all = svptrue_b32();
  svuint32_t zeros = svdup_u32(0);
  svuint32_t mvl_span = svdup_u32(mvl);
  svuint32_t and_mask = svdup_u32(num_bins - 1);
  svuint32_t idx_bins = svindex_u32(0, num_bins);
  svuint32_t idx_iota = svindex_u32(0, 1);
  svuint32_t off_strd = svindex_u32(0, stride * sizeof(int32_t));

  svbool_t p;
  svuint32_t val, his;

  uint32_t radix_shift = 0;
  for (uint32_t pass = 0; pass < num_passes; pass++) {
    // duplicate SHIFT RIGHT amount to vector
    svuint32_t radix_vec = svdup_u32(radix_shift);

    // reset private histograms to zero
    for (i = 0; i < num_bins; i++) svst1_vnum(all, histogram, i, zeros);

    // build private histograms
    for (i = 0, data_ptr = data_inp; i < stride; i++, data_ptr++) {
      // fix for situations when n is not multiple of mvl
      uint32_t m = (i < switching_i) - 1;
      uint32_t this_vl = ((std_vl - 1) & m) | (std_vl & ~m);
      p = svwhilelt_b32(0u, this_vl);

      // load data and extract bin numbers
      val = svldnt1_gather_offset(p, data_ptr, off_strd);
      val = svlsr_x(p, val, radix_vec);
      val = svand_x(p, val, and_mask);
      val = svmul_x(p, val, mvl_span);
      val = svadd_x(p, val, idx_iota);

      // use bin numbers to increment entries in histogram
      his = svld1_gather_index(p, histogram, val);
      his = svadd_x(p, his, 1);
      svst1_scatter_index(p, histogram, val, his);
    }

    // perform prefix sum over global histogram structure
    // (synchronisation between lanes)
    val = svdup_u32(0);
    for (i = 0, hist_ptr = histogram; i < num_bins; i++, hist_ptr++) {
      his = svld1_gather_index(all, hist_ptr, idx_bins);
      val = svadd_x(all, val, his);
    }
    svst1(all, prefixsum, val);
    for (uint32_t temp_sum = 0, i = 0; i < mvl; i++) {
      uint32_t temp_elm = prefixsum[i];
      prefixsum[i] = temp_sum;
      temp_sum += temp_elm;
    }
    val = svld1(all, prefixsum);
    for (i = 0, hist_ptr = histogram; i < num_bins; i++, hist_ptr++) {
      his = svld1_gather_index(all, hist_ptr, idx_bins);
      svst1_scatter_index(all, hist_ptr, idx_bins, val);
      val = svadd_x(all, val, his);
    }

    // rank & permute
    for (i = 0, data_ptr = data_inp; i < stride; i++, data_ptr++) {
      // fix for situations when n is not multiple of mvl
      uint32_t m = (i < switching_i) - 1;
      uint32_t this_vl = ((std_vl - 1) & m) | (std_vl & ~m);
      p = svwhilelt_b32(0u, this_vl);

      // load data and extract bin numbers
      svuint32_t cpy = svldnt1_gather_offset(p, data_ptr, off_strd);
      val = cpy;
      val = svlsr_x(p, val, radix_vec);
      val = svand_x(p, val, and_mask);
      val = svmul_x(p, val, mvl_span);
      val = svadd_x(p, val, idx_iota);

      // increment offset and store in histogram
      his = svld1_gather_index(p, histogram, val);
      svst1_scatter_index(p, histogram, val, svadd_x(p, his, 1));

      // store data to auxiliary array
      his = svlsl_x(all, his, 2);
      svstnt1_scatter_offset(p, data_out, his, cpy);
    }

    // exchange input/output data arrays
    swap_u32p(&data_inp, &data_out);

    // update shift amount
    radix_shift += num_bins_log2;
  }

  // exchange data arrays if necessary
  if ((void *)data_inp != (void *)data) {
    FOR_LOOP_32(, i, 0, n, p)
    svst1(p, data_out + i, svld1(p, data_inp + i));
  }

#else  // ACLE/SVE

  asm volatile(
      "ptrue  p0.s                    \n"
      "dup    z0.s, #0                \n"
      "dup    z1.s, %w[and_mask]      \n"  // AND mask
      "dup    z2.s, %w[mvl]           \n"  // MVL
      "index  z3.s, #0, #1            \n"  // IOTA
      "index  z4.s, #0, %w[stride]    \n"  // STRIDE
#ifdef __ARM_FEATURE_SVE2
      // use non-temporal gathers, need to scale
      "lsl    z4.s, z4.s, #2          \n"
#endif
      ::[stride] "r"(stride),
      [mvl] "r"(mvl), [and_mask] "r"(num_bins - 1)
      : "z0", "z1", "z2", "z3", "z4", "p0");

  uint32_t radix_shift = 0;
  for (uint32_t pass = 0; pass < num_passes; pass++) {
    // reset private histograms to zero
    for (i = 0, hist_ptr = histogram; i < num_bins; i++) {
      asm volatile(
          "st1w   z0.s, p0, [%[hist]]     \n"
          "addvl  %[hist], %[hist], #1    \n"
          : [hist] "+&r"(hist_ptr)
          :
          : "z0", "p0", "memory");
    }

    // duplicate SHIFT RIGHT amount to vector
    asm volatile("dup z5.s, %w[rs]" : : [rs] "r"(radix_shift));

    // build private histograms
    for (i = 0, data_ptr = data_inp; i < stride; i++, data_ptr++) {
      // fix for situations when n is not multiple of mvl
      uint32_t m = (i < switching_i) - 1;
      uint32_t this_vl = ((std_vl - 1) & m) | (std_vl & ~m);
      asm volatile(
          "whilelt    p1.s, wzr, %w[this_vl]                  \n"
      // load data and extract bin numbers
#ifdef __ARM_FEATURE_SVE2
          "ldnt1w     z6.s, p1/z, [z4.s, %[data]]             \n"
#else
          "ld1w       z6.s, p1/z, [%[data], z4.s, uxtw #2]    \n"
#endif
          "lsr        z6.s, p1/m, z6.s, z5.s                  \n"
          "and        z6.s, p1/m, z6.s, z1.s                  \n"
          "mul        z6.s, p1/m, z6.s, z2.s                  \n"
          "add        z6.s, p1/m, z6.s, z3.s                  \n"
          // use bin numbers to increment entries in histogram
          "ld1w       z7.s, p1/z, [%[hist], z6.s, uxtw #2]    \n"
          "add        z7.s, z7.s, #1                          \n"
          "st1w       z7.s, p1  , [%[hist], z6.s, uxtw #2]    \n" ::[data] "r"(
              data_ptr),
          [hist] "r"(histogram), [this_vl] "r"(this_vl)
          : "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p1", "cc", "memory");
    }

    // perform prefix sum over global histogram structure
    // (synchronisation between lanes)
    asm volatile(
        "index  z6.s, #0, %w[num_bins]  \n"
        "dup    z7.s, #0                \n" ::[num_bins] "r"(num_bins)
        : "z6", "z7");
    for (i = 0, hist_ptr = histogram; i < num_bins; i++, hist_ptr++) {
      asm volatile(
          "ld1w   z8.s, p0/z, [%[hist], z6.s, uxtw #2]    \n"
          "add    z7.s, p0/m, z7.s, z8.s                  \n" ::[hist] "r"(
              hist_ptr)
          : "z6", "z7", "z8", "p0", "memory");
    }
    asm volatile("st1w z7.s, p0, [%[pfs]]" ::[pfs] "r"(prefixsum)
                 : "z7", "p0", "memory");
    for (uint32_t temp_sum = 0, i = 0; i < mvl; i++) {
      uint32_t temp_elm = prefixsum[i];
      prefixsum[i] = temp_sum;
      temp_sum += temp_elm;
    }
    asm volatile("ld1w z7.s, p0/z, [%[pfs]]" ::[pfs] "r"(prefixsum)
                 : "z7", "p0", "memory");
    for (i = 0, hist_ptr = histogram; i < num_bins; i++, hist_ptr++) {
      asm volatile(
          "ld1w   z8.s, p0/z, [%[hist], z6.s, uxtw #2]    \n"
          "st1w   z7.s, p0  , [%[hist], z6.s, uxtw #2]    \n"
          "add    z7.s, p0/m, z7.s, z8.s                  \n" ::[hist] "r"(
              hist_ptr)
          : "z6", "z7", "z8", "p0", "memory");
    }

    // rank & permute
    for (i = 0, data_ptr = data_inp; i < stride; i++, data_ptr++) {
      // fix for situations when n is not multiple of mvl
      uint32_t m = (i < switching_i) - 1;
      uint32_t this_vl = ((std_vl - 1) & m) | (std_vl & ~m);
      asm volatile(
          "whilelt    p1.s, wzr, %w[this_vl]                  \n"
      // load data and extract bin numbers
#ifdef __ARM_FEATURE_SVE2
          "ldnt1w     z6.s, p1/z, [z4.s, %[data]]             \n"
#else
          "ld1w       z6.s, p1/z, [%[data], z4.s, uxtw #2]    \n"
#endif
          "mov        z8.d, z6.d                              \n"
          "lsr        z6.s, p1/m, z6.s, z5.s                  \n"
          "and        z6.s, p1/m, z6.s, z1.s                  \n"
          "mul        z6.s, p1/m, z6.s, z2.s                  \n"
          "add        z6.s, p1/m, z6.s, z3.s                  \n"
          // store data to auxiliary array
          "ld1w       z7.s, p1/z, [%[hist], z6.s, uxtw #2]    \n"
#ifdef __ARM_FEATURE_SVE2
          // use non-temporal scatter, need to scale
          "lsl        z9.s, z7.s, #2                          \n"
          "stnt1w     z8.s, p1  , [z9.s, %[dest]]             \n"
#else
          "st1w       z8.s, p1  , [%[dest], z7.s, uxtw #2]    \n"
#endif
          // increment offset and store in histogram
          "add        z7.s, z7.s, #1                          \n"
          "st1w       z7.s, p1  , [%[hist], z6.s, uxtw #2]    \n" ::[data] "r"(
              data_ptr),
          [dest] "r"(data_out), [hist] "r"(histogram), [this_vl] "r"(this_vl)
          : "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p1", "cc", "memory");
    }

    // exchange input/output data arrays
    swap_u32p(&data_inp, &data_out);

    // update shift amount
    radix_shift += num_bins_log2;
  }

  // exchange data arrays if necessary
  if ((void *)data_inp != (void *)data) {
    asm volatile(
        "   mov     %w[i], wzr                          \n"
        "   whilelt p1.s, %w[i], %w[n]                  \n"
        "   b.none  4f                                  \n"
        "3:                                             \n"
        "   ld1w    z2.s, p1/z, [%[src], %x[i], lsl #2] \n"
        "   st1w    z2.s, p1  , [%[dst], %x[i], lsl #2] \n"
        "   incw    %x[i]                               \n"
        "   whilelt p1.s, %w[i], %w[n]                  \n"
        "   b.first 3b                                  \n"
        "4:                                             \n"
        : [i] "+&r"(i)
        : [src] "r"(data_inp), [dst] "r"(data_out), [n] "r"(n)
        : "z2", "p1", "cc", "memory");
  }

#endif  // ACLE/SVE

  post_add_offset(n, data, radix.offset);
}

#else
static void NOINLINE do_sort(struct loop_124_data *restrict input) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif  // Implementation

#if !defined(HAVE_CANDIDATE)

static void inner_loop_124(struct loop_124_data *restrict input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256
#endif

LOOP_DECL(124, NS_SVE_LOOP_ATTR)
{
  struct loop_124_data data = { .n = SIZE, .hist = NULL, .prfx = NULL, };

  ALLOC_64B(data.data, SIZE, "data array");
  ALLOC_64B(data.temp, SIZE, "intermediate buffer");

  inner_loops_124(iters, &data);

  int res = check_sorted(SIZE, data.data);
  bool passed = (res == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(124, passed, "%d", 0, res)
#endif
  return passed ? 0 : 1;
}
