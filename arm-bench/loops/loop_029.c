/*----------------------------------------------------------------------------
#
#   Loop 029: FP64 multiply by power of 2
#
#   Purpose:
#     Use of FSCALE instruction.
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


typedef union {
  double dbl;
  uint64_t integer;
} b64;

struct loop_029_data {
  double *restrict input;
  int64_t *restrict scale;
  double *restrict output;
  int64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_029(struct loop_029_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_029(struct loop_029_data *restrict data) {
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = __builtin_scalbn(input[i], (int)scale[i]);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_029(struct loop_029_data *restrict data)
LOOP_ATTR
{
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  svbool_t p;
  FOR_LOOP_64(int64_t, i, 0, size, p) {
    svfloat64_t inp = svld1(p, input + i);
    svint64_t pwr = svld1(p, scale + i);
    svfloat64_t out = svscale_x(p, inp, pwr);
    svst1(p, output + i, out);
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_029(struct loop_029_data *restrict data)
LOOP_ATTR
{
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  int64_t offset = 0;
  asm volatile(
      "   whilelt pn8.d, %[i], %[n], vlx2                           \n"
      "   b.none  2f                                                \n"
      "1:                                                           \n"
      "   ld1d    {z0.d-z1.d}, pn8/z, [%[a], %[i], lsl #3]          \n"
      "   ld1d    {z2.d-z3.d}, pn8/z, [%[b], %[i], lsl #3]          \n"
      "   pext    {p0.d,p1.d}, pn8[0]                               \n"
      "   fscale  z0.d, p0/m, z0.d, z2.d                            \n"
      "   fscale  z1.d, p1/m, z1.d, z3.d                            \n"
      "   st1d    {z0.d-z1.d}, pn8, [%[c], %[i], lsl #3]            \n"
      "   incd    %[i], all, mul #2                                 \n"
      "   whilelt pn8.d, %[i], %[n], vlx2                           \n"
      "   b.first 1b                                                \n"
      "2:                                                           \n"
      : [i] "+&r"(offset)
      : [a] "r"(input), [b] "r"(scale), [c] "r"(output), [n] "r"(size)
      : "z0", "z1", "z2", "z3", "p0", "p1", "p8", "cc",  "memory");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_029(struct loop_029_data *restrict data)
LOOP_ATTR
{
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  int64_t offset = 0, input_1st = 0, scale_1st = 0, output_1st = 0;
  asm volatile(
      "       cntd        %[offset]                                       \n"

      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     2f                                              \n"

      "       ptrue       p0.d                                            \n"
      "       addvl       %[input_1st], %[input], #-1                     \n"
      "       addvl       %[scale_1st], %[scale], #-1                     \n"
      "       addvl       %[output_1st], %[output], #-1                   \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset), [input_1st] "+&r"(input_1st),
        [scale_1st] "+&r"(scale_1st), [output_1st] "+&r"(output_1st)
      : [input] "r"(input), [scale] "r"(scale), [output] "r"(output),
        [size] "r"(size)
      : "p0", "p1", "cc");
  asm volatile(
      "1:     ld1d        z0.d, p0/z, [%[input_1st], %[offset], lsl #3]   \n"
      "       ld1d        z1.d, p1/z, [%[input], %[offset], lsl #3]       \n"
      "       ld1d        z2.d, p0/z, [%[scale_1st], %[offset], lsl #3]   \n"
      "       ld1d        z3.d, p1/z, [%[scale], %[offset], lsl #3]       \n"

      "       fscale      z0.d, p0/m, z0.d, z2.d                          \n"
      "       fscale      z1.d, p1/m, z1.d, z3.d                          \n"

      "       st1d        z0.d, p0, [%[output_1st], %[offset], lsl #3]    \n"
      "       st1d        z1.d, p1, [%[output], %[offset], lsl #3]        \n"

      "       incd        %[offset], all, mul #2                          \n"
      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.first     1b                                              \n"

      "2:     decd        %[offset]                                       \n"
      "       whilelt     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     3f                                              \n"

      "       ld1d        z0.d, p1/z, [%[input], %[offset], lsl #3]       \n"
      "       ld1d        z2.d, p1/z, [%[scale], %[offset], lsl #3]       \n"
      "       fscale      z0.d, p1/m, z0.d, z2.d                          \n"
      "       st1d        z0.d, p1, [%[output], %[offset], lsl #3]        \n"

      "3:                                                                 \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset)
      : [input] "r"(input), [scale] "r"(scale), [output] "r"(output),
        [size] "r"(size), [input_1st] "r"(input_1st),
        [scale_1st] "r"(scale_1st), [output_1st] "r"(output_1st)
      : "p0", "p1", "v0", "v1", "v2", "v3", "p0", "p1", "cc", "memory");
}
#elif defined(__ARM_NEON)

static void inner_loop_029(struct loop_029_data *restrict data) {
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input + ((size >> 2) << 5) - 32;
  size = (int64_t)input + (size << 3) - 8;
  asm volatile(
      "       mov         x3, #2047                       \n"
      "       lsl         x3, x3, #52                     \n"
      "       mvn         x3, x3                          \n"
      "       dup         v6.2d, x3                       \n"

      "       cmp         %[input], %[lmt]                \n"
      "       b.gt        2f                              \n"

      "1:     ldp         q0, q1, [%[input]], #32         \n"
      "       ldp         q2, q3, [%[scale]], #32         \n"

      "       shl         v2.2d, v2.2d, #52               \n"
      "       shl         v3.2d, v3.2d, #52               \n"
      "       add         v4.2d, v0.2d, v2.2d             \n"
      "       add         v5.2d, v1.2d, v3.2d             \n"
      "       and         v0.16b, v0.16b, v6.16b          \n"
      "       and         v1.16b, v1.16b, v6.16b          \n"
      "       orr         v4.16b, v0.16b, v4.16b          \n"
      "       orr         v5.16b, v1.16b, v5.16b          \n"

      "       stp         q4, q5, [%[output]], #32        \n"

      "       cmp         %[input], %[lmt]                \n"
      "       b.le        1b                              \n"

      "2:     cmp         %[input], %[size]               \n"
      "       b.gt        4f                              \n"

      "3:     ldr         x0, [%[input]], #8              \n"
      "       ldr         x1, [%[scale]], #8              \n"
      "       add         x2, x0, x1, lsl #52             \n"
      "       and         x0, x0, x3                      \n"
      "       orr         x2, x0, x2                      \n"
      "       str         x2, [%[output]], #8             \n"

      "       cmp         %[input], %[size]               \n"
      "       b.le        3b                              \n"

      "4:                                                 \n"
      // output operands, source operands, and clobber list
      : [input] "+&r"(input), [scale] "+&r"(scale), [output] "+&r"(output)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc",
        "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_029(struct loop_029_data *restrict data) {
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  int64_t lmt = (int64_t)input + ((size >> 1) << 4) - 16;
  size = (int64_t)input + (size << 3) - 8;
  asm volatile(
      "       mov     x6, #2047                       \n"
      "       lsl     x6, x6, #52                     \n"
      "       mvn     x6, x6                          \n"

      "       cmp     %[input], %[lmt]                \n"
      "       b.gt    2f                              \n"

      "1:     ldp     x0, x1, [%[input]], #16         \n"
      "       ldp     x2, x3, [%[scale]], #16         \n"

      "       add     x4, x0, x2, lsl #52             \n"
      "       add     x5, x1, x3, lsl #52             \n"
      "       and     x0, x0, x6                      \n"
      "       and     x1, x1, x6                      \n"
      "       orr     x4, x0, x4                      \n"
      "       orr     x5, x1, x5                      \n"

      "       stp     x4, x5, [%[output]], #16        \n"

      "       cmp     %[input], %[lmt]                \n"
      "       b.le    1b                              \n"

      "2:     cmp     %[input], %[size]               \n"
      "       b.gt    3f                              \n"

      "       ldr     x0, [%[input]]                  \n"
      "       ldr     x2, [%[scale]]                  \n"
      "       add     x4, x0, x2, lsl #52             \n"
      "       and     x0, x0, x6                      \n"
      "       orr     x4, x0, x4                      \n"
      "       str     x4, [%[output]]                 \n"

      "3:                                             \n"
      // output operands, source operands, and clobber list
      : [input] "+&r"(input), [scale] "+&r"(scale), [output] "+&r"(output)
      : [size] "r"(size), [lmt] "r"(lmt)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "cc", "memory");
}
#else
static void inner_loop_029(struct loop_029_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(029, SC_SVE_LOOP_ATTR)
{
  struct loop_029_data data = { .size = SIZE };

  ALLOC_64B(data.input, SIZE, "input data");
  ALLOC_64B(data.scale, SIZE, "scaling parameters");
  ALLOC_64B(data.output, SIZE, "output buffer");

  fill_double(data.input, SIZE);
  fill_int64_range(data.scale, SIZE, -512, 511);
  fill_double(data.output, SIZE);

  inner_loops_029(iters, &data);

  b64 res;
  res.integer = 0;
  for (int i = 1; i < SIZE; i++) {
    b64 temp;
    temp.dbl = data.output[i];
    res.integer ^= temp.integer;
  }

  bool passed = check_exact_double(res.dbl, 0x39fdfe4f027bfce2);
#ifndef STANDALONE
  FINALISE_LOOP_I(29, passed, "%"PRIx64, (uint64_t) 0x39fdfe4f027bfce2, res.integer)
#endif
  return passed ? 0 : 1;
}
