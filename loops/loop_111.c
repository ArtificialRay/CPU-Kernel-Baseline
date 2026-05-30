/*----------------------------------------------------------------------------
#
#   Loop 111: FP64 overflow handling
#
#   Purpose:
#     Use of FLOGB and FSCALE instructions.
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

struct loop_111_data {
  double *restrict input1;
  double *restrict input2;
  double *restrict output;
  int64_t *restrict exponent;
  int64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_111(struct loop_111_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_111(struct loop_111_data *restrict input) {
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = __builtin_frexp(input1[i], (int *)&exponent[i]);
    output[i] = __builtin_ldexp(output[i], 1);
    output[i] *= input2[i];
    --exponent[i];
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_111(struct loop_111_data *restrict input)
LOOP_ATTR
{
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  svbool_t p;
  FOR_LOOP_64(int64_t, i, 0, size, p) {
    svfloat64_t inp1 = svld1_f64(p, input1 + i);
    svfloat64_t inp2 = svld1_f64(p, input2 + i);
    svint64_t exp1 = svlogb_z(p, inp1);
    svfloat64_t man1 = svscale_x(p, inp1, svneg_z(p, exp1));
    svfloat64_t man2 = svmul_x(p, man1, inp2);
    svst1(p, output + i, man2);
    svst1(p, exponent + i, exp1);
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_111(struct loop_111_data *restrict input)
LOOP_ATTR
{
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  uint64_t size = input->size;
  uint64_t offset = 0;
  asm volatile(
      "       whilelt     pn8.d, %[i], %[n], vlx2                         \n"
      "       b.none      2f                                              \n"
      "1:                                                                 \n"
      "       ld1d        {z0.d-z1.d}, pn8/z, [%[input1], %[i], lsl #3]   \n"
      "       ld1d        {z2.d-z3.d}, pn8/z, [%[input2], %[i], lsl #3]   \n"
      "       pext        {p0.d, p1.d}, pn8[0]                            \n"
      "       movprfx     z4, z0                                          \n"
      "       flogb       z4.d, p0/m, z0.d                                \n"
      "       movprfx     z5, z1                                          \n"
      "       flogb       z5.d, p1/m, z1.d                                \n"
      "       movprfx     z6, z4                                          \n"
      "       neg         z6.d, p0/m, z4.d                                \n"
      "       movprfx     z7, z5                                          \n"
      "       neg         z7.d, p1/m, z5.d                                \n"
      "       fscale      z0.d, p0/m, z0.d, z6.d                          \n"
      "       fscale      z1.d, p1/m, z1.d, z7.d                          \n"
      "       fmul        z0.d, p0/m, z0.d, z2.d                          \n"
      "       fmul        z1.d, p1/m, z1.d, z3.d                          \n"
      "       st1d        {z0.d-z1.d}, pn8, [%[output], %[i], lsl #3]     \n"
      "       st1d        {z4.d-z5.d}, pn8, [%[exponent], %[i], lsl #3]   \n"
      "       incd        %[i], all, mul #2                               \n"
      "       whilelt     pn8.d, %[i], %[n], vlx2                         \n"
      "       b.first     1b                                              \n"
      "2:                                                                 \n"
      : [i] "+&r"(offset)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [exponent] "r"(exponent), [n] "r"(size)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
        "p0", "p1", "p8", "cc", "memory");
}
#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_111(struct loop_111_data *restrict input)
LOOP_ATTR
{
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  int64_t offset = 0, input1_1st = 0, input2_1st = 0, output_1st = 0,
          exponent_1st = 0;
  asm volatile(
      "       cntd        %[offset]                                       \n"

      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     2f                                              \n"

      "       ptrue       p0.d                                            \n"
      "       addvl       %[input1_1st], %[input1], #-1                   \n"
      "       addvl       %[input2_1st], %[input2], #-1                   \n"
      "       addvl       %[output_1st], %[output], #-1                   \n"
      "       addvl       %[exponent_1st], %[exponent], #-1               \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset), [input1_1st] "+&r"(input1_1st),
        [input2_1st] "+&r"(input2_1st), [output_1st] "+&r"(output_1st),
        [exponent_1st] "+&r"(exponent_1st)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [exponent] "r"(exponent), [size] "r"(size)
      : "p0", "p1", "cc");
  asm volatile(
      "1:     ld1d        z0.d, p0/z, [%[input1_1st], %[offset], lsl #3]  \n"
      "       ld1d        z1.d, p1/z, [%[input1], %[offset], lsl #3]      \n"
      "       ld1d        z2.d, p0/z, [%[input2_1st], %[offset], lsl #3]  \n"
      "       ld1d        z3.d, p1/z, [%[input2], %[offset], lsl #3]      \n"

      "       movprfx     z4, z0                                          \n"
      "       flogb       z4.d, p0/m, z0.d                                \n"
      "       movprfx     z5, z1                                          \n"
      "       flogb       z5.d, p1/m, z1.d                                \n"

      "       movprfx     z6, z4                                          \n"
      "       neg         z6.d, p0/m, z4.d                                \n"
      "       movprfx     z7, z5                                          \n"
      "       neg         z7.d, p1/m, z5.d                                \n"

      "       fscale      z0.d, p0/m, z0.d, z6.d                          \n"
      "       fscale      z1.d, p1/m, z1.d, z7.d                          \n"

      "       fmul        z0.d, p0/m, z0.d, z2.d                          \n"
      "       fmul        z1.d, p1/m, z1.d, z3.d                          \n"

      "       st1d        z0.d, p0, [%[output_1st], %[offset], lsl #3]    \n"
      "       st1d        z1.d, p1, [%[output], %[offset], lsl #3]        \n"
      "       st1d        z4.d, p0, [%[exponent_1st], %[offset], lsl #3]  \n"
      "       st1d        z5.d, p1, [%[exponent], %[offset], lsl #3]      \n"

      "       incd        %[offset], all, mul #2                          \n"
      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.first     1b                                              \n"

      "2:     decd        %[offset]                                       \n"
      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     3f                                              \n"

      "       ld1d        z0.d, p1/z, [%[input1], %[offset], lsl #3]      \n"
      "       ld1d        z2.d, p1/z, [%[input2], %[offset], lsl #3]      \n"

      "       flogb       z4.d, p1/m, z0.d                                \n"
      "       neg         z6.d, p1/m, z4.d                                \n"
      "       fscale      z0.d, p1/m, z0.d, z6.d                          \n"
      "       fmul        z0.d, p1/m, z0.d, z2.d                          \n"

      "       st1d        z0.d, p1, [%[output], %[offset], lsl #3]        \n"
      "       st1d        z4.d, p1, [%[exponent], %[offset], lsl #3]      \n"

      "3:                                                                 \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [exponent] "r"(exponent), [size] "r"(size),
        [input1_1st] "r"(input1_1st), [input2_1st] "r"(input2_1st),
        [output_1st] "r"(output_1st), [exponent_1st] "r"(exponent_1st)
      : "p0", "p1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc",
        "memory");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_111(struct loop_111_data *restrict input)
LOOP_ATTR
{
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  int64_t offset = 0, input1_1st = 0, input2_1st = 0, output_1st = 0,
          exponent_1st = 0, mask = 1023;
  asm volatile(
      "       dup         z8.d, %[mask]                                   \n"

      "       cntd        %[offset]                                       \n"

      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     2f                                              \n"

      "       ptrue       p0.d                                            \n"
      "       addvl       %[input1_1st], %[input1], #-1                   \n"
      "       addvl       %[input2_1st], %[input2], #-1                   \n"
      "       addvl       %[output_1st], %[output], #-1                   \n"
      "       addvl       %[exponent_1st], %[exponent], #-1               \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset), [input1_1st] "+&r"(input1_1st),
        [input2_1st] "+&r"(input2_1st), [output_1st] "+&r"(output_1st),
        [exponent_1st] "+&r"(exponent_1st)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [exponent] "r"(exponent), [size] "r"(size), [mask] "r"(mask)
      : "p0", "p1", "cc");
  asm volatile(
      "1:     ld1d        z0.d, p0/z, [%[input1_1st], %[offset], lsl #3]  \n"
      "       ld1d        z1.d, p1/z, [%[input1], %[offset], lsl #3]      \n"
      "       ld1d        z2.d, p0/z, [%[input2_1st], %[offset], lsl #3]  \n"
      "       ld1d        z3.d, p1/z, [%[input2], %[offset], lsl #3]      \n"

      "       lsl         z4.d, z0.d, #1                                  \n"
      "       lsl         z5.d, z1.d, #1                                  \n"
      "       lsr         z4.d, p0/m, z4.d, #53                           \n"
      "       lsr         z5.d, p1/m, z5.d, #53                           \n"
      "       sub         z6.d, z8.d, z4.d                                \n"
      "       sub         z7.d, z8.d, z5.d                                \n"
      "       sub         z4.d, z4.d, z8.d                                \n"
      "       sub         z5.d, z5.d, z8.d                                \n"

      "       fscale      z0.d, p0/m, z0.d, z6.d                          \n"
      "       fscale      z1.d, p1/m, z1.d, z7.d                          \n"

      "       fmul        z0.d, p0/m, z0.d, z2.d                          \n"
      "       fmul        z1.d, p1/m, z1.d, z3.d                          \n"

      "       st1d        z0.d, p0, [%[output_1st], %[offset], lsl #3]    \n"
      "       st1d        z1.d, p1, [%[output], %[offset], lsl #3]        \n"
      "       st1d        z4.d, p0, [%[exponent_1st], %[offset], lsl #3]  \n"
      "       st1d        z5.d, p1, [%[exponent], %[offset], lsl #3]      \n"

      "       incd        %[offset], all, mul #2                          \n"
      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.first     1b                                              \n"

      "2:     decd        %[offset]                                       \n"
      "       whilelo     p1.d, %[offset], %[size]                        \n"
      "       b.nfrst     3f                                              \n"

      "       ld1d        z0.d, p1/z, [%[input1], %[offset], lsl #3]      \n"
      "       ld1d        z2.d, p1/z, [%[input2], %[offset], lsl #3]      \n"

      "       lsl         z4.d, z0.d, #1                                  \n"
      "       lsr         z4.d, p1/m, z4.d, #53                           \n"
      "       sub         z6.d, z8.d, z4.d                                \n"
      "       sub         z4.d, z4.d, z8.d                                \n"
      "       fscale      z0.d, p1/m, z0.d, z6.d                          \n"
      "       fmul        z0.d, p1/m, z0.d, z2.d                          \n"

      "       st1d        z0.d, p1, [%[output], %[offset], lsl #3]        \n"
      "       st1d        z4.d, p1, [%[exponent], %[offset], lsl #3]      \n"

      "3:                                                                 \n"
      // output operands, source operands, and clobber list
      : [offset] "+&r"(offset)
      : [input1] "r"(input1), [input2] "r"(input2), [output] "r"(output),
        [exponent] "r"(exponent), [size] "r"(size),
        [input1_1st] "r"(input1_1st), [input2_1st] "r"(input2_1st),
        [output_1st] "r"(output_1st), [exponent_1st] "r"(exponent_1st)
      : "p0", "p1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}
#elif defined(__ARM_NEON)

static void inner_loop_111(struct loop_111_data *restrict input) {
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  int64_t lmt = (int64_t)input1 + ((size >> 2) << 5) - 32;
  size = (int64_t)input1 + (size << 3) - 32;
  asm volatile(
      "       mov         x4, #1023                           \n"
      "       dup         v8.2d, x4                           \n"
      "       mov         x5, #2047                           \n"
      "       lsl         x5, x5, #52                         \n"
      "       mvn         x5, x5                              \n"
      "       dup         v9.2d, x5                           \n"

      "       cmp         %[input1], %[lmt]                   \n"
      "       b.hi        2f                                  \n"

      "1:     ld1         {v0.2d, v1.2d}, [%[input1]], #32    \n"
      "       ld1         {v2.2d, v3.2d}, [%[input2]], #32    \n"

      "       shl         v4.2d, v0.2d, #1                    \n"
      "       shl         v5.2d, v1.2d, #1                    \n"
      "       ushr        v4.2d, v4.2d, #53                   \n"
      "       ushr        v5.2d, v5.2d, #53                   \n"
      "       sub         v4.2d, v4.2d, v8.2d                 \n"
      "       sub         v5.2d, v5.2d, v8.2d                 \n"

      "       shl         v6.2d, v4.2d, #52                   \n"
      "       shl         v7.2d, v5.2d, #52                   \n"
      "       sub         v6.2d, v0.2d, v6.2d                 \n"
      "       sub         v7.2d, v1.2d, v7.2d                 \n"
      "       bif         v0.16b, v6.16b, v9.16b              \n"
      "       bif         v1.16b, v7.16b, v9.16b              \n"

      "       fmul        v0.2d, v0.2d, v2.2d                 \n"
      "       fmul        v1.2d, v1.2d, v3.2d                 \n"

      "       st1         {v0.2d, v1.2d}, [%[output]], #32    \n"
      "       st1         {v4.2d, v5.2d}, [%[exponent]], #32  \n"

      "       cmp         %[input1], %[lmt]                   \n"
      "       b.ls        1b                                  \n"

      "2:     cmp         %[input1], %[size]                  \n"
      "       b.hi        4f                                  \n"

      "3:     ldr         x0, [%[input1]], #8                 \n"
      "       ldr         x1, [%[input2]], #8                 \n"

      "       lsl         x2, x0, #1                          \n"
      "       lsr         x2, x2, #53                         \n"
      "       sub         x2, x2, x4                          \n"

      "       sub         x3, x0, x2, lsl #52                 \n"
      "       and         x0, x0, x5                          \n"
      "       orr         x0, x0, x3                          \n"

      "       fmov        d0, x0                              \n"
      "       fmov        d1, x1                              \n"
      "       fmul        d0, d0, d1                          \n"

      "       str         d0, [%[output]], #8                 \n"
      "       str         x2, [%[exponent]], #8               \n"

      "       cmp         %[input1], %[size]                  \n"
      "       b.ls        3b                                  \n"

      "4:                                                     \n"
      // output operands, source operands, and clobber list
      : [input1] "+&r"(input1), [input2] "+&r"(input2), [output] "+&r"(output),
        [exponent] "+&r"(exponent)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_111(struct loop_111_data *restrict input) {
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  int64_t lmt = (int64_t)input1 + ((size >> 1) << 4) - 16;
  size = (int64_t)input1 + (size << 3) - 8;
  asm volatile(
      "       mov     x8, #1023                       \n"
      "       mov     x9, #2047                       \n"
      "       lsl     x9, x9, #52                     \n"
      "       mvn     x9, x9                          \n"

      "       cmp     %[input1], %[lmt]               \n"
      "       b.gt    2f                              \n"

      "1:     ldp     x0, x1, [%[input1]], #16        \n"
      "       ldp     d2, d3, [%[input2]], #16        \n"

      "       ubfx    x4, x0, #52, #11                \n"
      "       ubfx    x5, x1, #52, #11                \n"
      "       sub     x4, x4, x8                      \n"
      "       sub     x5, x5, x8                      \n"

      "       sub     x6, x0, x4, lsl #52             \n"
      "       sub     x7, x1, x5, lsl #52             \n"
      "       and     x0, x0, x9                      \n"
      "       and     x1, x1, x9                      \n"
      "       orr     x0, x0, x6                      \n"
      "       orr     x1, x1, x7                      \n"

      "       fmov    d0, x0                          \n"
      "       fmov    d1, x1                          \n"
      "       fmul    d0, d0, d2                      \n"
      "       fmul    d1, d1, d3                      \n"

      "       stp     d0, d1, [%[output]], #16        \n"
      "       stp     x4, x5, [%[exponent]], #16      \n"

      "       cmp     %[input1], %[lmt]               \n"
      "       b.le    1b                              \n"

      "2:     cmp     %[input1], %[size]              \n"
      "       b.gt    3f                              \n"

      "       ldr     x0, [%[input1]]                 \n"
      "       ldr     d2, [%[input2]]                 \n"

      "       ubfx    x4, x0, #52, #11                \n"
      "       sub     x4, x4, x8                      \n"

      "       sub     x6, x0, x4, lsl #52             \n"
      "       and     x0, x0, x9                      \n"
      "       orr     x0, x0, x6                      \n"

      "       fmov    d0, x0                          \n"
      "       fmul    d0, d0, d2                      \n"

      "       str     d0, [%[output]]                 \n"
      "       str     x4, [%[exponent]]               \n"

      "3:                                             \n"
      // output operands, source operands, and clobber list
      : [input1] "+&r"(input1), [input2] "+&r"(input2), [output] "+&r"(output),
        [exponent] "+&r"(exponent)
      : [lmt] "r"(lmt), [size] "r"(size)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "cc",
        "memory");
}
#else

static void inner_loop_111(struct loop_111_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(111, SC_SVE_LOOP_ATTR)
{
  struct loop_111_data data = { .size = SIZE };

  ALLOC_64B(data.input1, SIZE, "1st operand buffer");
  ALLOC_64B(data.input2, SIZE, "2nd operand buffer");
  ALLOC_64B(data.output, SIZE, "result mantissa buffer");
  ALLOC_64B(data.exponent, SIZE, "result exponent buffer");

  fill_double_range(data.input1, SIZE, __builtin_ldexp(DBL_MAX, -20), DBL_MAX);
  fill_double_range(data.input2, SIZE, 0, DBL_MAX / 4);
  fill_double(data.output, SIZE);
  fill_int64(data.exponent, SIZE);

  inner_loops_111(iters, &data);

  b64 res_out, res_exp;
  res_out.integer = 0;
  res_exp.integer = 0;
  for (int i = 1; i < SIZE; i++) {
    b64 temp;
    temp.dbl = data.output[i];
    res_out.integer ^= temp.integer;
    temp.integer = data.exponent[i];
    res_exp.integer ^= temp.integer;
  }

  int64_t mask_exp = 0xffffffff;
  bool correct_out = (res_out.integer == 0x7feef5a535c84856);
  bool correct_exp = ((res_exp.integer & mask_exp) == 0x3fd);
  bool passed = correct_out && correct_exp;
#ifndef STANDALONE
  FINALISE_LOOP_FP64(111, passed, correct_out, correct_exp, "%"PRIx64, (uint64_t)0x7feef5a535c84856, (uint64_t)0x3fd, res_out.integer, (res_exp.integer & mask_exp))
#endif
  return passed ? 0 : 1;
}
