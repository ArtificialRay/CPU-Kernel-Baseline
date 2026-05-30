/*----------------------------------------------------------------------------
#
#   Loop 028: FP64 fast division
#
#   Purpose:
#     Use of FRECPE and FRECPS instructions.
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


struct loop_028_data {
  double *restrict input1;
  double *restrict input2;
  double *restrict output;
  int64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_028(struct loop_028_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_028(struct loop_028_data *restrict data) {
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = input1[i] / input2[i];
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_028(struct loop_028_data *restrict data)
LOOP_ATTR
{
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  svbool_t p;
  FOR_LOOP_64(int64_t, i, 0, size, p) {
    svfloat64_t inp1 = svld1(p, input1 + i);
    svfloat64_t inp2 = svld1(p, input2 + i);

    svfloat64_t inv2 = svrecpe(inp2);
    svfloat64_t stp2;
    for (int j = 0; j < 3; j++) {
      stp2 = svrecps(inp2, inv2);
      inv2 = svmul_m(p, inv2, stp2);
    }

    svfloat64_t out = svmul_m(p, inv2, inp1);
    svst1(p, output + i, out);
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_028(struct loop_028_data *restrict data)
LOOP_ATTR
{
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  int64_t offset = 0;
  asm volatile(
      "   ptrue   p0.d                                      \n"
      "   whilelt pn8.d, %[i], %[n], vlx2                   \n"
      "   b.none  2f                                        \n"
      "1:                                                   \n"
      "   ld1d    {z0.d-z1.d}, pn8/z, [%[a], %[i], lsl #3]  \n"
      "   ld1d    {z2.d-z3.d}, pn8/z, [%[b], %[i], lsl #3]  \n"
      "   frecpe  z4.d, z2.d                                \n"
      "   frecpe  z5.d, z3.d                                \n"
      "   frecps  z6.d, z2.d, z4.d                          \n"
      "   frecps  z7.d, z3.d, z5.d                          \n"
      "   fmul    z4.d, p0/m, z4.d, z6.d                    \n"
      "   fmul    z5.d, p0/m, z5.d, z7.d                    \n"
      "   frecps  z6.d, z2.d, z4.d                          \n"
      "   frecps  z7.d, z3.d, z5.d                          \n"
      "   fmul    z4.d, p0/m, z4.d, z6.d                    \n"
      "   fmul    z5.d, p0/m, z5.d, z7.d                    \n"
      "   frecps  z6.d, z2.d, z4.d                          \n"
      "   frecps  z7.d, z3.d, z5.d                          \n"
      "   fmul    z4.d, p0/m, z4.d, z6.d                    \n"
      "   fmul    z5.d, p0/m, z5.d, z7.d                    \n"
      "   fmul    z4.d, p0/m, z4.d, z0.d                    \n"
      "   fmul    z5.d, p0/m, z5.d, z1.d                    \n"
      "   st1d    {z4.d-z5.d}, pn8, [%[c], %[i], lsl #3]    \n"
      "   incd    %[i], all, mul #2                         \n"
      "   whilelt pn8.d, %[i], %[n], vlx2                   \n"
      "   b.first 1b                                        \n"
      "2:                                                   \n"
      : [i] "+&r"(offset)
      : [a] "r"(input1), [b] "r"(input2), [c] "r"(output), [n] "r"(size)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p8",
        "cc",  "memory");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_028(struct loop_028_data *restrict data)
LOOP_ATTR
{
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  int64_t offset = 0, input1_1st = 0, input2_1st = 0, output_1st = 0;
  asm volatile(
      "       ptrue       p0.d                                            \n"
      "       cntd        %[offset]                                       \n"

      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     2f                                              \n"

      "       addvl       %[input1_1st], %[input1], #-1                   \n"
      "       addvl       %[input2_1st], %[input2], #-1                   \n"
      "       addvl       %[output_1st], %[output], #-1                   \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset), [input1_1st] "+&r"(input1_1st),
        [input2_1st] "+&r"(input2_1st), [output_1st] "+&r"(output_1st)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [size] "r"(size)
      : "p0", "p1", "cc");
  asm volatile(
      "1:     ld1d        z2.d, p0/z, [%[input2_1st], %[offset], lsl #3]  \n"
      "       ld1d        z3.d, p1/z, [%[input2], %[offset], lsl #3]      \n"
      "       ld1d        z0.d, p0/z, [%[input1_1st], %[offset], lsl #3]  \n"
      "       ld1d        z1.d, p1/z, [%[input1], %[offset], lsl #3]      \n"

      "       frecpe      z4.d, z2.d                                      \n"
      "       frecpe      z5.d, z3.d                                      \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       frecps      z7.d, z3.d, z5.d                                \n"
      "       fmul        z4.d, p0/m, z4.d, z6.d                          \n"
      "       fmul        z5.d, p1/m, z5.d, z7.d                          \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       frecps      z7.d, z3.d, z5.d                                \n"
      "       fmul        z4.d, p0/m, z4.d, z6.d                          \n"
      "       fmul        z5.d, p1/m, z5.d, z7.d                          \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       frecps      z7.d, z3.d, z5.d                                \n"
      "       fmul        z4.d, p0/m, z4.d, z6.d                          \n"
      "       fmul        z5.d, p1/m, z5.d, z7.d                          \n"

      "       fmul        z4.d, p0/m, z4.d, z0.d                          \n"
      "       fmul        z5.d, p1/m, z5.d, z1.d                          \n"

      "       st1d        z4.d, p0, [%[output_1st], %[offset], lsl #3]    \n"
      "       st1d        z5.d, p1, [%[output], %[offset], lsl #3]        \n"

      "       incd        %[offset], all, mul #2                          \n"
      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.first     1b                                              \n"

      "2:     decd        %[offset]                                       \n"
      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     3f                                              \n"

      "       ld1d        z0.d, p1/z, [%[input1], %[offset], lsl #3]      \n"
      "       ld1d        z2.d, p1/z, [%[input2], %[offset], lsl #3]      \n"

      "       frecpe      z4.d, z2.d                                      \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       fmul        z4.d, p1/m, z4.d, z6.d                          \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       fmul        z4.d, p1/m, z4.d, z6.d                          \n"

      "       frecps      z6.d, z2.d, z4.d                                \n"
      "       fmul        z4.d, p1/m, z4.d, z6.d                          \n"

      "       fmul        z4.d, p1/m, z4.d, z0.d                          \n"

      "       st1d        z4.d, p1, [%[output], %[offset], lsl #3]        \n"

      "3:                                                                 \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [size] "r"(size), [input1_1st] "r"(input1_1st),
        [input2_1st] "r"(input2_1st), [output_1st] "r"(output_1st)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "p0", "p1",
        "cc", "memory");
}
#elif defined(__ARM_NEON)

/*
  The FDIV instruction was found to be faster than the FRECPE/FRECPS
  implementation for scalar code only
*/
static void inner_loop_028(struct loop_028_data *restrict data) {
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input1 + ((size >> 2) << 5);
  size = (int64_t)input1 + (size << 3);
  asm volatile(
      "       add         %[input1], %[input1], #32       \n"
      "       add         %[input2], %[input2], #32       \n"
      "       add         %[output], %[output], #32       \n"
      "       cmp         %[input1], %[lmt]               \n"
      "       b.gt        2f                              \n"

      "1:     ldp         q0, q1, [%[input1], #-32]       \n"
      "       ldp         q2, q3, [%[input2], #-32]       \n"

      "       frecpe      v4.2d, v2.2d                    \n"
      "       frecpe      v5.2d, v3.2d                    \n"

      "       frecps      v6.2d, v2.2d, v4.2d             \n"
      "       frecps      v7.2d, v3.2d, v5.2d             \n"
      "       fmul        v4.2d, v4.2d, v6.2d             \n"
      "       fmul        v5.2d, v5.2d, v7.2d             \n"

      "       frecps      v6.2d, v2.2d, v4.2d             \n"
      "       frecps      v7.2d, v3.2d, v5.2d             \n"
      "       fmul        v4.2d, v4.2d, v6.2d             \n"
      "       fmul        v5.2d, v5.2d, v7.2d             \n"

      "       frecps      v6.2d, v2.2d, v4.2d             \n"
      "       frecps      v7.2d, v3.2d, v5.2d             \n"
      "       fmul        v4.2d, v4.2d, v6.2d             \n"
      "       fmul        v5.2d, v5.2d, v7.2d             \n"

      "       fmul        v4.2d, v4.2d, v0.2d             \n"
      "       fmul        v5.2d, v5.2d, v1.2d             \n"

      "       stp         q4, q5, [%[output], #-32]       \n"

      "       add         %[input1], %[input1], #32       \n"
      "       add         %[input2], %[input2], #32       \n"
      "       add         %[output], %[output], #32       \n"
      "       cmp         %[input1], %[lmt]               \n"
      "       b.le        1b                              \n"

      "2:     sub         %[input1], %[input1], #24       \n"
      "       sub         %[input2], %[input2], #24       \n"
      "       sub         %[output], %[output], #24       \n"
      "       cmp         %[input1], %[size]              \n"
      "       b.gt        4f                              \n"

      "3:     ldr         d0, [%[input1], #-8]            \n"
      "       ldr         d2, [%[input2], #-8]            \n"
      "       fdiv        d4, d0, d2                      \n"
      "       str         d4, [%[output], #-8]            \n"

      "       add         %[input1], %[input1], #8        \n"
      "       add         %[input2], %[input2], #8        \n"
      "       add         %[output], %[output], #8        \n"
      "       cmp         %[input1], %[size]              \n"
      "       b.le        3b                              \n"

      "4:                                                 \n"
      // output operands, source operands, and clobber list
      : [input1] "+&r"(input1), [input2] "+&r"(input2), [output] "+&r"(output)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

/*
  The FDIV instruction was found to be faster than the FRECPE/FRECPS
  implementation for scalar code only
*/
static void inner_loop_028(struct loop_028_data *restrict data) {
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input1 + ((size >> 1) << 4);
  size = (int64_t)input1 + (size << 3);
  asm volatile(
      "       add         %[input1], %[input1], #8        \n"
      "       add         %[input2], %[input2], #8        \n"
      "       add         %[output], %[output], #8        \n"
      "       cmp         %[input1], %[lmt]               \n"
      "       b.gt        2f                              \n"

      "1:     ldp         d0, d1, [%[input1], #-8]        \n"
      "       ldp         d2, d3, [%[input2], #-8]        \n"
      "       fdiv        d4, d0, d2                      \n"
      "       fdiv        d5, d1, d3                      \n"
      "       stp         d4, d5, [%[output], #-8]        \n"

      "       add         %[input1], %[input1], #16       \n"
      "       add         %[input2], %[input2], #16       \n"
      "       add         %[output], %[output], #16       \n"
      "       cmp         %[input1], %[lmt]               \n"
      "       b.lt        1b                              \n"

      "2:     cmp         %[input1], %[size]              \n"
      "       b.gt        3f                              \n"

      "       ldr         d0, [%[input1], #-8]            \n"
      "       ldr         d2, [%[input2], #-8]            \n"
      "       fdiv        d4, d0, d2                      \n"
      "       str         d4, [%[output], #-8]            \n"

      "3:                                                 \n"
      // output operands, source operands, and clobber list
      : [input1] "+&r"(input1), [input2] "+&r"(input2), [output] "+&r"(output)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}
#else
static void inner_loop_028(struct loop_028_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(028, SC_SVE_LOOP_ATTR)
{
  struct loop_028_data data = { .size = SIZE };

  ALLOC_64B(data.input1, SIZE, "1st data source");
  ALLOC_64B(data.input2, SIZE, "2nd data source");
  ALLOC_64B(data.output, SIZE, "output buffer");

  fill_double(data.input1, SIZE);
  fill_double(data.input2, SIZE);
  fill_double(data.output, SIZE);

  inner_loops_028(iters, &data);

  double res = 0.0;
  for (int64_t i = 0; i < SIZE; i++) {
    res += i * data.output[i];
  }

  bool passed = check_double(res, 197677574.738559, 1.0);
#ifndef STANDALONE
  FINALISE_LOOP_F(28, passed, "%9.6f", 197677574.738559, 1.0, res)
#endif
  return passed ? 0 : 1;
}
