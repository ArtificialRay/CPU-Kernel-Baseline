/*----------------------------------------------------------------------------
#
#   Loop 037: FP32 complex vector product
#
#   Purpose:
#     Use of fp32 FCMLA instruction.
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


typedef struct cfloat32_t {
  float re, im;
} cfloat32_t;

struct loop_037_data {
  cfloat32_t *restrict a0;
  cfloat32_t *restrict b0;
  cfloat32_t *restrict c0;
  uint64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_037(struct loop_037_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_037(struct loop_037_data *restrict input) {
  cfloat32_t *a = input->a0;
  cfloat32_t *b = input->b0;
  cfloat32_t *c = input->c0;
  uint64_t size = input->size;

  uint64_t i;
  for (i = 0; i < size; i++) {
    c[i].re = (a[i].re * b[i].re) - (a[i].im * b[i].im);
    c[i].im = (a[i].re * b[i].im) + (a[i].im * b[i].re);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_037(struct loop_037_data *restrict input)
LOOP_ATTR
{
  cfloat32_t *a0 = input->a0;
  cfloat32_t *b0 = input->b0;
  cfloat32_t *c0 = input->c0;
  uint64_t size = input->size;

  svbool_t all = svptrue_b32();
  svbool_t p;
  FOR_LOOP_64(uint64_t, i, 0, size, p) {
    svfloat32_t a1 = svreinterpret_f32(svld1_u64(p, (uint64_t *)(a0 + i)));
    svfloat32_t b1 = svreinterpret_f32(svld1_u64(p, (uint64_t *)(b0 + i)));
    svfloat32_t c1 = svdup_f32(0);
    c1 = svcmla_x(all, c1, a1, b1, 0);
    c1 = svcmla_x(all, c1, a1, b1, 90);
    svst1(p, (uint64_t *)(c0 + i), svreinterpret_u64(c1));
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_037(struct loop_037_data *restrict input)
LOOP_ATTR
{
  cfloat32_t *a0 = input->a0;
  cfloat32_t *b0 = input->b0;
  cfloat32_t *c0 = input->c0;
  uint64_t size = input->size;
  uint64_t count = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       ptrue   p0.s                                        \n"
      "       whilelt pn8.d, %[i], %[n], vlx2                     \n"
      "       b.none  2f                                          \n"

      // Loop begins here:
      // This loop is unrolled such that multiply operation is performed on
      // 2*(VL/64) complex elements in a single iteration
      "1:                                                         \n"
      "       dup     z0.s, #0                                    \n"
      "       dup     z1.s, #0                                    \n"

      // Load complex elements from a and b arrays
      "       ld1d    {z10.d-z11.d}, pn8/z, [%[a0], %[i], lsl #3] \n"
      "       ld1d    {z12.d-z13.d}, pn8/z, [%[b0], %[i], lsl #3] \n"

      // Integer Complex multiplication
      "       fcmla   z0.s, p0/m, z10.s, z12.s, #0                \n"
      "       fcmla   z0.s, p0/m, z10.s, z12.s, #90               \n"
      "       fcmla   z1.s, p0/m, z11.s, z13.s, #0                \n"
      "       fcmla   z1.s, p0/m, z11.s, z13.s, #90               \n"

      // Store result to c array
      "       st1d    {z0.d-z1.d}, pn8, [%[c0], %[i], lsl #3]     \n"

      // Increment processed elements by 2*(VL/64) to account for unrolled loop
      "       incd    %[i], all, mul #2                           \n"
      "       whilelt pn8.d, %[i], %[n], vlx2                     \n"
      "       b.first 1b                                          \n"

      // End of operation
      "2:                                                         \n"

      : [i] "+&r"(count)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [n] "r"(size)
      : "z0", "z1", "z10", "z11", "z12", "z13", "p0", "p8", "memory", "cc");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_037(struct loop_037_data *restrict input)
LOOP_ATTR
{
  cfloat32_t *a0 = input->a0;
  cfloat32_t *b0 = input->b0;
  cfloat32_t *c0 = input->c0;
  uint64_t size = input->size;

  uint64_t count = 0, count_1 = 0;
  uint64_t a1 = 0, b1 = 0, c1 = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       ptrue p2.s                                  \n"
      "       whilelt p0.d, %[count], %[size]             \n"
      "       b.none  3f                                  \n"
      "       cntd    %[count_1]                          \n"
      "       whilelt p1.d, %[count_1], %[size]           \n"
      "       b.none  2f                                  \n"

      // Create a, b, c array pointers at offset of VL in order to unroll the
      // loop
      "       addvl   %[a1], %[a0], #1                    \n"
      "       addvl   %[b1], %[b0], #1                    \n"
      "       addvl   %[c1], %[c0], #1                    \n"

      // Loop begins here:
      // This loop is unrolled such that multiply operation is performed on
      // 2*(VL/64) complex elements in a single iteration

      "1:                                                 \n"
      "       dup     z0.s, #0                            \n"
      "       dup     z1.s, #0                            \n"

      // Load complex elements from a and b arrays
      "       ld1d    {z10.d}, p0/z, [%[a0], %[count], lsl #3] \n"
      "       ld1d    {z12.d}, p0/z, [%[b0], %[count], lsl #3] \n"
      "       ld1d    {z11.d}, p1/z, [%[a1], %[count], lsl #3] \n"
      "       ld1d    {z13.d}, p1/z, [%[b1], %[count], lsl #3] \n"

      // Integer Complex multiplication
      "       fcmla    z0.s, p2/m, z10.s, z12.s, #0              \n"
      "       fcmla    z0.s, p2/m, z10.s, z12.s, #90             \n"
      "       fcmla    z1.s, p2/m, z11.s, z13.s, #0              \n"
      "       fcmla    z1.s, p2/m, z11.s, z13.s, #90             \n"

      // Store result to c array
      "       st1d    {z0.d}, p0, [%[c0], %[count], lsl #3]  \n"
      "       st1d    {z1.d}, p1, [%[c1], %[count], lsl #3]  \n"

      // Increment processed elements by 2*(VL/64) to account for unrolled loop
      "       incd    %[count], all, mul #2               \n"
      "       whilelt p1.d, %[count], %[size]             \n"
      "       b.first 1b                                  \n"
      // Loop ends here:
      // Since loop was unrolled, check if <=VL/64 elements are available to
      // process
      "       decd    %[count]                            \n"
      "       whilelt p0.d, %[count], %[size]             \n"
      "       b.none  3f                                  \n"

      // Process last <=(VL/64) complex elements
      "2:                                                 \n"
      "       dup     z0.s, #0                            \n"
      "       ld1d    {z10.d}, p0/z, [%[a0], %[count], lsl #3] \n"
      "       ld1d    {z12.d}, p0/z, [%[b0], %[count], lsl #3] \n"
      "       fcmla    z0.s, p2/m, z10.s, z12.s, #0              \n"
      "       fcmla    z0.s, p2/m, z10.s, z12.s, #90             \n"
      "       st1d    {z0.d}, p0, [%[c0], %[count], lsl #3]    \n"

      // End of operation
      "3:                                                 \n"
      : [a1] "+&r"(a1), [b1] "+&r"(b1), [c1] "+&r"(c1), [count] "+&r"(count),
        [count_1] "+&r"(count_1)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [size] "r"(size)
      : "z0", "z1", "z10", "z11", "z12", "z13", "p0", "p1", "memory", "cc");
}
#elif defined(__ARM_NEON)

static void inner_loop_037(struct loop_037_data *restrict input) {
  cfloat32_t *a0 = input->a0;
  cfloat32_t *b0 = input->b0;
  cfloat32_t *c0 = input->c0;
  uint64_t size = input->size;

  float reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0;

  asm volatile(
      // Check whether there are >=4 elements to process before starting loop
      // If not, proceed to loop tail
      "       cbz     %[size], 3f                         \n"
      "       cmp     %[size], #4                         \n"
      "       blt     2f                                  \n"

      // Loop begins here:
      "1:                                                 \n"
      "       movi    v0.4s, #0x0                         \n"
      "       movi    v1.4s, #0x0                         \n"

      // Load elements from a & b arrays such that real and imaginary parts are
      // de-interleaved
      "       ld2     {v10.4s,v11.4s}, [%[a0]], #32       \n"
      "       ld2     {v20.4s,v21.4s}, [%[b0]], #32       \n"

      // Perform complex multiplication
      "       fmla     v0.4s, v10.4s, v20.4s               \n"
      "       fmls     v0.4s, v11.4s, v21.4s               \n"
      "       fmla     v1.4s, v10.4s, v21.4s               \n"
      "       fmla     v1.4s, v11.4s, v20.4s               \n"

      // Store the result to c array
      "       st2     {v0.4s,v1.4s}, [%[c0]], #32         \n"

      // Compare whether are >=4 elements left to continue with loop iterations
      "       sub     %[size], %[size], #4                \n"
      "       cmp     %[size], #4                         \n"
      "       bge     1b                                  \n"

      // Loop ends here:
      // Process loop tail if any
      "2:                                                 \n"
      "       cbz     %[size], 3f                         \n"
      "       ldp     %s[reg0], %s[reg1], [%[a0]], #8     \n"
      "       ldp     %s[reg2], %s[reg3], [%[b0]], #8     \n"
      "       fmul    %s[reg4], %s[reg0], %s[reg2]        \n"
      "       fmsub   %s[reg4], %s[reg1], %s[reg3], %s[reg4]    \n"
      "       fmul    %s[reg5], %s[reg0], %s[reg3]       \n"
      "       fmadd   %s[reg5], %s[reg1], %s[reg2], %s[reg5]    \n"
      "       stp     %s[reg4], %s[reg5], [%[c0]], #8     \n"
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 2b                         \n"
      "3:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&w"(reg0),
        [reg1] "+&w"(reg1), [reg2] "+&w"(reg2), [reg3] "+&w"(reg3),
        [reg4] "+&w"(reg4), [reg5] "+&w"(reg5), [size] "+&r"(size)
      :
      : "v0", "v1", "v10", "v11", "v20", "v21", "v0", "v1", "memory", "cc");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_037(struct loop_037_data *restrict input) {
  cfloat32_t *a0 = input->a0;
  cfloat32_t *b0 = input->b0;
  cfloat32_t *c0 = input->c0;
  uint64_t size = input->size;

  float reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0;

  asm volatile(
      "       cbz     %[size], 2f                         \n"
      "1:                                                 \n"
      "       ldp     %s[reg0], %s[reg1], [%[a0]], #8     \n"
      "       ldp     %s[reg2], %s[reg3], [%[b0]], #8     \n"
      "       fmul    %s[reg4], %s[reg0], %s[reg2]        \n"
      "       fmsub   %s[reg4], %s[reg1], %s[reg3], %s[reg4]    \n"
      "       fmul    %s[reg5], %s[reg0], %s[reg3]        \n"
      "       fmadd   %s[reg5], %s[reg1], %s[reg2], %s[reg5]    \n"
      "       stp     %s[reg4], %s[reg5], [%[c0]], #8     \n"
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 1b                         \n"
      "2:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&w"(reg0),
        [reg1] "+&w"(reg1), [reg2] "+&w"(reg2), [reg3] "+&w"(reg3),
        [reg4] "+&w"(reg4), [reg5] "+&w"(reg5), [size] "+&r"(size)
      :
      : "memory", "cc");
}
#else
static void inner_loop_037(struct loop_037_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(037, SC_SVE_LOOP_ATTR)
{
  struct loop_037_data data = { .size = SIZE };

  ALLOC_64B(data.a0, SIZE, "1st operand buffer");
  ALLOC_64B(data.b0, SIZE, "2nd operand buffer");
  ALLOC_64B(data.c0, SIZE, "result buffer");

  cfloat32_t *a = data.a0, *b = data.b0, *c = data.c0;

  fill_float((float *)a, 2 * SIZE);
  fill_float((float *)b, 2 * SIZE);
  fill_float((float *)c, 2 * SIZE);

  inner_loops_037(iters, &data);

  uint64_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    float exp_re = (a[i].re * b[i].re) - (a[i].im * b[i].im);
    float exp_im = (a[i].re * b[i].im) + (a[i].im * b[i].re);
    float delta = fabsf(c[i].re - exp_re) + fabsf(c[i].im - exp_im);
    if (delta > 0.001) {
      checksum += 1;
    }
  }

  bool passed = checksum == 0;
#ifndef STANDALONE
  FINALISE_LOOP_I(37, passed, "%"PRId64, (uint64_t) 0, checksum)
#endif
  return passed ? 0 : 1;
}
