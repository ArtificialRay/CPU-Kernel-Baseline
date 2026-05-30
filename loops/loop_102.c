/*----------------------------------------------------------------------------
#
#   Loop 102: General histogram
#
#   Purpose:
#     Use of HISTCNT instruction.
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


struct loop_102_data {
  uint32_t *restrict histogram;
  uint64_t histogram_size;
  uint32_t *restrict records;
  int64_t num_records;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_102(struct loop_102_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void NOINLINE update(uint32_t *histogram, uint32_t *records,
                            int64_t num_records) {
  for (int i = 0; i < num_records; i++) {
    uint32_t entry = records[i];
    histogram[entry] += 1;
  }
}
#elif defined(HAVE_SVE_INTRINSICS)
static void NOINLINE update(uint32_t *histogram, uint32_t *records,
                            int64_t num_records) {
  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, num_records, p) {
    svuint32_t recs = svld1(p, records + i);
    svuint32_t hist = svld1_gather_index(p, histogram, recs);
    svuint32_t cnts = svhistcnt_z(p, recs, recs);
    hist = svadd_x(p, hist, cnts);
    svst1_scatter_index(p, histogram, recs, hist);
  }
}
#elif defined(__ARM_FEATURE_SVE2)
static void NOINLINE update(uint32_t *histogram, uint32_t *records,
                            int64_t num_records) {
  int64_t i = 0;

  asm volatile(
      "       whilelo p0.s, %[i], %[num_records]                    \n"
      "1:     ld1w    {z1.s}, p0/z, [%[records], %[i], lsl #2]      \n"
      "       ld1w    {z2.s}, p0/z, [%[histogram], z1.s, uxtw #2]   \n"
      "       incw    %[i]                                          \n"
      "       histcnt z0.s, p0/z, z1.s, z1.s                        \n"
      "       add     z2.s, p0/m, z2.s, z0.s                        \n"
      "       st1w    {z2.s}, p0, [%[histogram], z1.s, uxtw #2]     \n"
      "       whilelo p0.s, %[i], %[num_records]                    \n"
      "       b.any   1b                                            \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [num_records] "r"(num_records), [histogram] "r"(histogram),
        [records] "r"(records)
      : "v0", "v1", "v2", "v3", "p0", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
// No SVE1/Neon/Scalar versions
static void NOINLINE update(uint32_t *histogram, uint32_t *records,
                            int64_t num_records) {
  uint64_t entry;
  uint64_t count;
  uint32_t *lmt = records + num_records;

  asm volatile(
      "1:   ldr   %w[entry], [%[records]], #4                   \n"
      "     ldr   %w[count], [%[histogram], %[entry], lsl #2]   \n"
      "     add   %w[count], %w[count], #0x1                    \n"
      "     str   %w[count], [%[histogram], %[entry], lsl #2]   \n"
      "     cmp   %[records], %[lmt]                            \n"
      "     b.ne  1b                                            \n"
      // output operands, source operands, and clobber list
      : [entry] "=&r"(entry), [records] "+&r"(records), [count] "=&r"(count)
      : [lmt] "r"(lmt), [histogram] "r"(histogram)
      : "memory", "cc");
}
#else
static void inner_loop_102(struct loop_102_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_102(struct loop_102_data *restrict input) {
  uint32_t *histogram = input->histogram;
  uint64_t histogram_size = input->histogram_size;
  uint32_t *records = input->records;
  int64_t num_records = input->num_records;

  memset(histogram, 0, histogram_size);
  update(histogram, records, num_records);
}
#endif /* !HAVE_CANDIDATE */

#define MAX_VAL 100
#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(102, NS_SVE_LOOP_ATTR)
{
  struct loop_102_data data = { .num_records = SIZE };
  data.histogram_size = sizeof(data.histogram[0]) * MAX_VAL;

  ALLOC_64B(data.histogram, MAX_VAL, "histogram");
  ALLOC_64B(data.records, SIZE, "record buffer");

  fill_uint32(data.records, SIZE);
  for (int i = 0; i < SIZE; i++) {
    data.records[i] %= MAX_VAL;
  }

  inner_loops_102(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < MAX_VAL; i++) {
    checksum += data.histogram[i] * i;
  }

  bool passed = checksum == 0x00078c31;
#ifndef STANDALONE
  FINALISE_LOOP_I(102, passed, "0x%08"PRIx32, 0x00078c31, checksum)
#endif
  return passed ? 0 : 1;
}
