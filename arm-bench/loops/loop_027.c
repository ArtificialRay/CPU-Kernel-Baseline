/*----------------------------------------------------------------------------
#
#   Loop 027: FP32 square root
#
#   Purpose:
#     Use of FSQRT instruction.
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


struct loop_027_data {
  float *restrict input;
  float *restrict output;
  int64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_027(struct loop_027_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_027(struct loop_027_data *restrict data) {
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = __builtin_sqrtf(input[i]);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_027(struct loop_027_data *restrict data)
LOOP_ATTR
{
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, size, p) {
    svfloat32_t inp = svld1(p, input + i);
    svfloat32_t out = svsqrt_x(p, inp);
    svst1(p, output + i, out);
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_027(struct loop_027_data *restrict data)
LOOP_ATTR
{
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  int64_t offset = 0;
  asm volatile(
      "   whilelt pn8.s, %[i], %[n], vlx2                   \n"
      "   b.none  2f                                        \n"
      "1:                                                   \n"
      "   ld1w    {z0.s-z1.s}, pn8/z, [%[a], %[i], lsl #2]  \n"
      "   pext    {p0.s,p1.s}, pn8[0]                       \n"
      "   fsqrt   z0.s, p0/m, z0.s                          \n"
      "   fsqrt   z1.s, p1/m, z1.s                          \n"
      "   st1w    {z0.s-z1.s}, pn8, [%[b], %[i], lsl #2]    \n"
      "   incw    %[i], all, mul #2                         \n"
      "   whilelt pn8.s, %[i], %[n], vlx2                   \n"
      "   b.first 1b                                        \n"
      "2:                                                   \n"
      : [i] "+&r"(offset)
      : [a] "r"(input), [b] "r"(output), [n] "r"(size)
      : "z0", "z1", "p0", "p1", "p8", "cc", "memory");
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_027(struct loop_027_data *restrict data)
LOOP_ATTR
{
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  int64_t offset = 0, input_1st = 0, output_1st = 0;
  asm volatile(
      "       ptrue       p0.s                                            \n"
      "       cntw        %[offset]                                       \n"

      "       whilelt     p1.s, %[offset], %[size]                        \n"
      "       b.nfrst     2f                                              \n"

      "       addvl       %[input_1st], %[input], #-1                     \n"
      "       addvl       %[output_1st], %[output], #-1                   \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset), [input_1st] "+&r"(input_1st),
        [output_1st] "+&r"(output_1st)
      : [input] "r"(input), [output] "r"(output), [size] "r"(size)
      : "p0", "cc");
  asm volatile(
      "1:     ld1w        z0.s, p0/z, [%[input_1st], %[offset], lsl #2]   \n"
      "       ld1w        z1.s, p1/z, [%[input], %[offset], lsl #2]       \n"

      "       fsqrt       z0.s, p0/m, z0.s                                \n"
      "       fsqrt       z1.s, p1/m, z1.s                                \n"

      "       st1w        z0.s, p0, [%[output_1st], %[offset], lsl #2]    \n"
      "       st1w        z1.s, p1, [%[output], %[offset], lsl #2]        \n"

      "       incw        %[offset], all, mul #2                          \n"
      "       whilelt     p1.s, %[offset], %[size]                        \n"
      "       b.first     1b                                              \n"

      "2:     decw        %[offset]                                       \n"
      "       whilelt     p1.s, %[offset], %[size]                        \n"
      "       b.nfrst     3f                                              \n"

      "       ld1w        z0.s, p1/z, [%[input], %[offset], lsl #2]       \n"
      "       fsqrt       z0.s, p1/m, z0.s                                \n"
      "       st1w        z0.s, p1, [%[output], %[offset], lsl #2]        \n"

      "3:                                                                 \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset)
      : [input] "r"(input), [output] "r"(output), [size] "r"(size),
        [input_1st] "r"(input_1st), [output_1st] "r"(output_1st)
      : "v0", "v1", "p0", "p1", "cc", "memory");
}
#elif defined(__ARM_NEON)
static void inner_loop_027(struct loop_027_data *restrict data) {
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input + ((size >> 3) << 5);
  size = (int64_t)input + (size << 2);
  asm volatile(
      "       add     %[input], %[input], #32             \n"
      "       add     %[output], %[output], #32           \n"
      "       cmp     %[input], %[lmt]                    \n"
      "       b.gt    2f                                  \n"

      "1:     ldp     q0, q1, [%[input], #-32]            \n"
      "       fsqrt   v2.4s, v0.4s                        \n"
      "       fsqrt   v3.4s, v1.4s                        \n"
      "       stp     q2, q3, [%[output], #-32]           \n"

      "       add     %[input], %[input], #32             \n"
      "       add     %[output], %[output], #32           \n"
      "       cmp     %[input], %[lmt]                    \n"
      "       b.le    1b                                  \n"

      "2:     sub     %[input], %[input], #28             \n"
      "       sub     %[output], %[output], #28           \n"
      "       cmp     %[input], %[size]                   \n"
      "       b.gt    4f                                  \n"

      "3:     ldr     s0, [%[input], #-4]                 \n"
      "       fsqrt   s2, s0                              \n"
      "       str     s2, [%[output], #-4]                \n"

      "       add     %[input], %[input], #4              \n"
      "       add     %[output], %[output], #4            \n"
      "       cmp     %[input], %[size]                   \n"
      "       b.le    3b                                  \n"

      "4:                                                 \n"
      // output operands, source operands, and clobber list
      : [input] "+&r"(input), [output] "+&r"(output)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "v0", "v1", "v2", "v3", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_027(struct loop_027_data *restrict data) {
  float *input = data->input;
  float *output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input + ((size >> 1) << 3);
  size = (int64_t)input + (size << 2);
  asm volatile(
      "       add     %[input], %[input], #4          \n"
      "       add     %[output], %[output], #4        \n"
      "       cmp     %[input], %[lmt]                \n"
      "       b.gt    2f                              \n"

      "1:     ldp     s0, s1, [%[input], #-4]         \n"
      "       fsqrt   s2, s0                          \n"
      "       fsqrt   s3, s1                          \n"
      "       stp     s2, s3, [%[output], #-4]        \n"

      "       add     %[input], %[input], #8          \n"
      "       add     %[output], %[output], #8        \n"
      "       cmp     %[input], %[lmt]                \n"
      "       b.lt    1b                              \n"

      "2:     cmp     %[input], %[size]               \n"
      "       b.gt    3f                              \n"

      "       ldr     s0, [%[input], #-4]             \n"
      "       fsqrt   s2, s0                          \n"
      "       str     s2, [%[output], #-4]            \n"

      "3:                                             \n"
      // output operands, source operands, and clobber list
      : [input] "+&r"(input), [output] "+&r"(output)
      : [size] "r"(size), [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "cc", "memory");
}
#else
static void inner_loop_027(struct loop_027_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(027, SC_SVE_LOOP_ATTR)
{
  struct loop_027_data data = { .size = SIZE };

  ALLOC_64B(data.input, SIZE, "input data");
  ALLOC_64B(data.output, SIZE, "output buffer");

  fill_float(data.input, SIZE);
  fill_float(data.output, SIZE);

  inner_loops_027(iters, &data);

  float res = 0.0f;
  for (int64_t i = 0; i < SIZE; i++) {
    res += i * data.output[i];
  }

  bool passed = check_float(res, 33218676.0f, 1.0f);
#ifndef STANDALONE
  FINALISE_LOOP_F(27, passed, "%9.6f", 33218676.0f, 1.0f, res)
#endif
  return passed ? 0 : 1;
}
