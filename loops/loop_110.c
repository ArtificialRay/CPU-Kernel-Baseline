/*----------------------------------------------------------------------------
#
#   Loop 110: UINT32 complex dot
#
#   Purpose:
#     Use of u32 CDOT instruction.
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

typedef struct cint8_t {
  int8_t re, im;
} cint8_t;

typedef struct cint32_t {
  int32_t re, im;
} cint32_t;

struct loop_110_data {
  cint8_t *restrict a0;
  cint8_t *restrict b0;
  cint32_t *restrict c0;
  uint64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_110(struct loop_110_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_110(struct loop_110_data *restrict input) {
  cint8_t *a = input->a0;
  cint8_t *b = input->b0;
  cint32_t *c = input->c0;
  uint64_t size = input->size;

  uint64_t i;

  for (i = 0; i < size; i++) {
    c[i].re =
        (int32_t)(((a[2 * i].re * b[2 * i].re) - (a[2 * i].im * b[2 * i].im)) +
                  ((a[2 * i + 1].re * b[2 * i + 1].re) -
                   (a[2 * i + 1].im * b[2 * i + 1].im)));
    c[i].im =
        (int32_t)(((a[2 * i].im * b[2 * i].re) + (a[2 * i].re * b[2 * i].im)) +
                  ((a[2 * i + 1].im * b[2 * i + 1].re) +
                   (a[2 * i + 1].re * b[2 * i + 1].im)));
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_110(struct loop_110_data *restrict input)
LOOP_ATTR
{
  cint8_t *a0 = input->a0;
  cint8_t *b0 = input->b0;
  cint32_t *c0 = input->c0;
  uint64_t size = input->size;

  svbool_t p;
  FOR_LOOP_32(uint64_t, i, 0, size, p) {
    svint8_t a1 = svreinterpret_s8(svld1_s32(p, (int32_t *)(a0 + 2 * i)));
    svint8_t b1 = svreinterpret_s8(svld1_s32(p, (int32_t *)(b0 + 2 * i)));
    svint32_t cr = svcdot(svdup_s32(0), a1, b1, 0);
    svint32_t ci = svcdot(svdup_s32(0), a1, b1, 90);
    svint32x2_t c1 = svcreate2(cr, ci);
    svst2(p, (int32_t *)(c0 + i), c1);
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_110(struct loop_110_data *restrict input)
LOOP_ATTR
{
  cint8_t *a = input->a0;
  cint8_t *b = input->b0;
  cint32_t *c = input->c0;
  uint64_t size = input->size;
  uint64_t size_1 = input->size * 2;
  uint64_t count = 0, count_1 = 0;
  uint64_t c1 = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       whilelt pn8.s, %[i], %[n], vlx2                     \n"
      "       b.none  2f                                          \n"
      "       ptrue   p0.b                                        \n"
      "       addvl   %[c1], %[c], #2                    \n"

      // Loop begins here:
      // This loop is unrolled such that cdot operation is performed on upto
      // (VL/32) pairs of complex elements in a single iteration
      "1:                                                         \n"
      "       mov     z0.s, #0                                    \n"
      "       mov     z1.s, #0                                    \n"
      "       mov     z2.s, #0                                    \n"
      "       mov     z3.s, #0                                    \n"

      // Load complex elements from a and b arrays
      "       ld1w    {z4.s-z5.s}, pn8/z, [%[a], %[i], lsl #2]    \n"
      "       ld1w    {z6.s-z7.s}, pn8/z, [%[b], %[i], lsl #2]    \n"

      // Integer Complex dot product with rotate
      "       cdot    z0.s, z4.b, z6.b, #0                        \n"  // Real part
      "       cdot    z1.s, z4.b, z6.b, #90                       \n"  // Imaginary part
      "       cdot    z2.s, z5.b, z7.b, #0                        \n"  // Real part
      "       cdot    z3.s, z5.b, z7.b, #90                       \n"  // Imaginary part

      // Store results to c array such that real & imaginary parts are
      // interleaved
      "       st2w    {z0.s,z1.s}, p0, [%[c], %[j], lsl #2]      \n"
      "       st2w    {z2.s,z3.s}, p0, [%[c1], %[j], lsl #2]      \n"

      // Increment processed elements by 2*(VL/32) to account for unrolled loop
      "       incw    %[i], all, mul #2                           \n"
      "       incw    %[j], all, mul #4                           \n"
      "       whilelt pn8.s, %[i], %[n], vlx2                     \n"
      "       b.first 1b                                          \n"

      // Loop ends here:
      "2:                                                         \n"

      : [i] "+&r"(count), [j] "+&r"(count_1), [c1] "+&r"(c1)
      : [a] "r"(a), [b] "r"(b), [c] "r"(c), [n] "r"(size), [m] "r"(size_1)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p8",
        "memory", "cc");
}
#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_110(struct loop_110_data *restrict input)
LOOP_ATTR
{
  cint8_t *a0 = input->a0;
  cint8_t *b0 = input->b0;
  cint32_t *c0 = input->c0;
  uint64_t size = input->size;

  uint64_t count = 0, count_1 = 0;
  uint64_t a1 = 0, b1 = 0, c1 = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       whilelt p0.s, %[count], %[size]             \n"
      "       b.none  3f                                  \n"
      "       cntw    %[count_1]                          \n"
      "       whilelt p1.s, %[count_1], %[size]           \n"
      "       mov     %[count_1], %[count]                \n"
      "       b.none  2f                                  \n"

      // Create additional a, b, c array pointers in order to unroll the loop
      "       addvl   %[a1], %[a0], #1                    \n"
      "       addvl   %[b1], %[b0], #1                    \n"
      "       addvl   %[c1], %[c0], #2                    \n"

      // Loop begins here:
      // This loop is unrolled such that cdot operation is performed on (VL/32)
      // pairs of complex elements in a single iteration

      "1:                                                 \n"
      "       mov     z0.s, #0                            \n"
      "       mov     z1.s, #0                            \n"
      "       mov     z2.s, #0                            \n"
      "       mov     z3.s, #0                            \n"

      // Load complex elements from a and b arrays
      "       ld1w    {z10.s}, p0/z, [%[a0], %[count], lsl #2]    \n"
      "       ld1w    {z20.s}, p0/z, [%[b0], %[count], lsl #2]    \n"
      "       ld1w    {z11.s}, p1/z, [%[a1], %[count], lsl #2]    \n"
      "       ld1w    {z21.s}, p1/z, [%[b1], %[count], lsl #2]    \n"

      // Integer Complex dot product with rotate
      "       cdot    z0.s, z10.b, z20.b, #0             \n"  // Real part
      "       cdot    z1.s, z10.b, z20.b, #90            \n"  // Imaginary part
      "       cdot    z2.s, z11.b, z21.b, #0             \n"  // Real part
      "       cdot    z3.s, z11.b, z21.b, #90            \n"  // Imaginary part

      // Store result to c array such that real & imaginary parts are
      // interleaved
      "       st2w    {z0.s,z1.s}, p0, [%[c0], %[count_1], lsl #2]  \n"
      "       st2w    {z2.s,z3.s}, p1, [%[c1], %[count_1], lsl #2]  \n"

      // Increment processed elements by 2*(VL/32) to account for unrolled loop
      "       incw    %[count], all, mul #3               \n"
      "       incw    %[count_1], all, mul #4             \n"
      "       whilelt p1.s, %[count], %[size]             \n"
      "       decw    %[count]                            \n"
      "       b.first 1b                                  \n"

      // Loop ends here:
      // Since loop was unrolled, check if <=VL/32 elements are available to
      // process
      "       whilelt p0.s, %[count], %[size]             \n"
      "       b.none  3f                                  \n"

      // Process last <=(VL/32) complex elements
      "2:                                                 \n"
      "       mov     z0.s, #0                            \n"
      "       mov     z1.s, #0                            \n"
      "       ld1w    {z10.s}, p0/z, [%[a0], %[count], lsl #2] \n"
      "       ld1w    {z20.s}, p0/z, [%[b0], %[count], lsl #2] \n"
      "       cdot    z0.s, z10.b, z20.b, #0                   \n"
      "       cdot    z1.s, z10.b, z20.b, #90                  \n"
      "       st2w    {z0.s,z1.s}, p0, [%[c0], %[count_1], lsl #2]  \n"

      // End of operation
      "3:                                                 \n"
      : [a1] "+&r"(a1), [b1] "+&r"(b1), [c1] "+&r"(c1), [count] "+&r"(count),
        [count_1] "+&r"(count_1)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [size] "r"(size)
      : "z0", "z1", "z2", "z3", "z10", "z11", "z20", "z21", "p0", "p1",
        "memory", "cc");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_110(struct loop_110_data *restrict input)
LOOP_ATTR
{
  cint8_t *a0 = input->a0;
  cint8_t *b0 = input->b0;
  cint32_t *c0 = input->c0;
  uint64_t size = input->size;

  uint64_t count = 0, count_1 = 0;
  uint64_t a1 = 0, b1 = 0, c1 = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       whilelt p0.s, %[count], %[size]             \n"
      "       b.none  3f                                  \n"
      "       cntw    %[count_1]                          \n"
      "       whilelt p1.s, %[count_1], %[size]           \n"
      "       mov     %[count_1], %[count]                \n"
      "       b.none  2f                                  \n"

      // Create additional a, b, c array pointers in order to unroll the loop
      "       addvl   %[a1], %[a0], #1                    \n"
      "       addvl   %[b1], %[b0], #1                    \n"
      "       addvl   %[c1], %[c0], #2                    \n"

      // Setup predicates for computing swapped and negated real or
      // imaginary components of input
      "       ptrue   p4.h                                \n"
      "       ptrue   p3.s                                \n"
      "       not     p2.b, p4/z, p3.b                    \n"

      // Loop begins here:
      "1:                                                 \n"
      "       mov     z0.s, #0                            \n"
      "       mov     z1.s, #0                            \n"
      "       mov     z2.s, #0                            \n"
      "       mov     z3.s, #0                            \n"
      "       mov     z4.s, #0                            \n"
      "       mov     z5.s, #0                            \n"
      "       mov     z6.s, #0                            \n"
      "       mov     z7.s, #0                            \n"

      // Load complex elements from a and b arrays
      "       ld1w    {z20.s}, p0/z, [%[b0], %[count], lsl #2]    \n"
      "       ld1w    {z21.s}, p1/z, [%[b1], %[count], lsl #2]    \n"
      "       ld1w    {z10.s}, p0/z, [%[a0], %[count], lsl #2]    \n"
      "       ld1w    {z11.s}, p1/z, [%[a1], %[count], lsl #2]    \n"

      // Unpack & sign extend loaded byte elements to half-word size
      // This is to avoid overflow during subtraction operation
      "       sunpklo z22.h, z20.b                       \n"
      "       sunpkhi z23.h, z20.b                       \n"
      "       sunpklo z24.h, z21.b                       \n"
      "       sunpkhi z25.h, z21.b                       \n"

      "       sunpklo z12.h, z10.b                       \n"
      "       sunpkhi z13.h, z10.b                       \n"
      "       sunpklo z14.h, z11.b                       \n"
      "       sunpkhi z15.h, z11.b                       \n"

      // To calculate imaginary portion of complex dot product, reverse the
      // bytes in each hafword of input
      "       revh    z26.s, p3/m, z22.s                 \n"
      "       revh    z27.s, p3/m, z23.s                 \n"
      "       revh    z28.s, p3/m, z24.s                 \n"
      "       revh    z29.s, p3/m, z25.s                 \n"

      // To calculate real portion of complex dot product, negate the imaginary
      // components of loaded data
      "       neg     z22.h, p2/m, z22.h                 \n"
      "       neg     z23.h, p2/m, z23.h                 \n"
      "       neg     z24.h, p2/m, z24.h                 \n"
      "       neg     z25.h, p2/m, z25.h                 \n"

      // Compute Integer Complex dot product with rotate using sdot instruction
      "       sdot    z1.d, z12.h, z26.h                 \n"
      "       sdot    z3.d, z13.h, z27.h                 \n"
      "       sdot    z0.d, z12.h, z22.h                 \n"
      "       sdot    z2.d, z13.h, z23.h                 \n"
      "       sdot    z5.d, z14.h, z28.h                 \n"
      "       sdot    z7.d, z15.h, z29.h                 \n"
      "       sdot    z4.d, z14.h, z24.h                 \n"
      "       sdot    z6.d, z15.h, z25.h                 \n"

      // Pack output to 32-bits of real and imaginary parts
      "       uzp1    z1.s, z1.s, z3.s                   \n"
      "       uzp1    z0.s, z0.s, z2.s                   \n"
      "       uzp1    z5.s, z5.s, z7.s                   \n"
      "       uzp1    z4.s, z4.s, z6.s                   \n"

      // Store result to c array
      "       st2w    {z0.s,z1.s}, p0, [%[c0], %[count_1], lsl #2]  \n"
      "       st2w    {z4.s,z5.s}, p1, [%[c1], %[count_1], lsl #2]  \n"

      // Increment processed elements
      "       incw    %[count], all, mul #3               \n"
      "       incw    %[count_1], all, mul #4             \n"
      "       whilelt p1.s, %[count], %[size]             \n"
      "       decw    %[count]                            \n"
      "       b.first 1b                                  \n"

      // Loop ends here:
      // Since loop was unrolled, check if <=VL/32 elements are available to
      // process
      "       whilelt p0.s, %[count], %[size]             \n"
      "       b.none  3f                                  \n"

      // Process last <=(VL/32) complex elements
      "2:                                                 \n"
      "       mov     z0.s, #0                            \n"
      "       mov     z1.s, #0                            \n"
      "       mov     z2.s, #0                            \n"
      "       mov     z3.s, #0                            \n"
      "       ld1w    {z20.s}, p0/z, [%[b0], %[count], lsl #2] \n"
      "       ld1w    {z10.s}, p0/z, [%[a0], %[count], lsl #2] \n"
      "       sunpklo z22.h, z20.b                        \n"
      "       sunpkhi z23.h, z20.b                        \n"
      "       sunpklo z12.h, z10.b                        \n"
      "       sunpkhi z13.h, z10.b                        \n"
      "       revh    z26.s, p3/m, z22.s                  \n"
      "       revh    z27.s, p3/m, z23.s                  \n"
      "       neg     z22.h, p2/m, z22.h                  \n"
      "       neg     z23.h, p2/m, z23.h                  \n"
      "       sdot    z1.d, z12.h, z26.h                  \n"
      "       sdot    z3.d, z13.h, z27.h                  \n"
      "       sdot    z0.d, z12.h, z22.h                  \n"
      "       sdot    z2.d, z13.h, z23.h                  \n"
      "       uzp1    z1.s, z1.s, z3.s                    \n"
      "       uzp1    z0.s, z0.s, z2.s                    \n"
      "       st2w    {z0.s,z1.s}, p0, [%[c0], %[count_1], lsl #2]  \n"

      // End of operation
      "3:                                                 \n"
      : [a1] "+&r"(a1), [b1] "+&r"(b1), [c1] "+&r"(c1), [count] "+&r"(count),
        [count_1] "+&r"(count_1)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [size] "r"(size)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z28", "z29", "p0", "p1", "p2", "p3", "p4", "memory", "cc");
}
#elif defined(__ARM_NEON)

static void inner_loop_110(struct loop_110_data *restrict input) {
  cint8_t *a0 = input->a0;
  cint8_t *b0 = input->b0;
  cint32_t *c0 = input->c0;
  uint64_t size = input->size;

  uint32_t reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0,
           reg7 = 0, reg8 = 0, reg9 = 0;

  asm volatile(
      // Check whether there are >=16 elements to process before starting loop
      // If not, proceed to loop tail
      "       cbz     %[size], 3f                         \n"
      "       cmp     %[size], #16                        \n"
      "       blt     2f                                  \n"

      // Loop begins here:
      // This loop is unrolled such that 16 outputs are produced in one
      // iteration

      "1:                                                 \n"
      "       movi    v0.4s, #0x0                         \n"
      "       movi    v1.4s, #0x0                         \n"

      // Load elements from a & b arrays in de-interleaved fashion
      "       ld4     {v10.16b,v11.16b,v12.16b,v13.16b}, [%[a0]], #64     \n"
      "       ld4     {v20.16b,v21.16b,v22.16b,v23.16b}, [%[b0]], #64     \n"

      // Inputs have to be expanded to word size to prevent overflow/underflow
      // Expand inputs to half-word size
      "       sxtl    v14.8h, v10.8b                      \n"
      "       sxtl2   v15.8h, v10.16b                     \n"
      "       sxtl    v16.8h, v11.8b                      \n"
      "       sxtl2   v17.8h, v11.16b                     \n"
      "       sxtl    v18.8h, v12.8b                      \n"
      "       sxtl2   v19.8h, v12.16b                     \n"
      "       sxtl    v10.8h, v13.8b                      \n"
      "       sxtl2   v11.8h, v13.16b                     \n"
      "       sxtl    v24.8h, v20.8b                      \n"
      "       sxtl2   v25.8h, v20.16b                     \n"
      "       sxtl    v26.8h, v21.8b                      \n"
      "       sxtl2   v27.8h, v21.16b                     \n"
      "       sxtl    v28.8h, v22.8b                      \n"
      "       sxtl2   v29.8h, v22.16b                     \n"
      "       sxtl    v20.8h, v23.8b                      \n"
      "       sxtl2   v21.8h, v23.16b                     \n"

      // Expand inputs to word size
      "       sxtl    v12.4s, v14.4h                      \n"
      "       sxtl    v22.4s, v24.4h                      \n"
      "       sxtl    v8.4s, v18.4h                       \n"
      "       sxtl    v9.4s, v28.4h                       \n"
      "       sxtl    v13.4s, v16.4h                      \n"
      "       sxtl    v23.4s, v26.4h                      \n"
      "       sxtl    v2.4s, v10.4h                       \n"
      "       sxtl    v3.4s, v20.4h                       \n"

      // Perform complex dot product operation with rotation #0
      "       mla     v0.4s, v12.4s, v22.4s               \n"
      "       mls     v0.4s, v13.4s, v23.4s               \n"
      "       mla     v0.4s, v8.4s, v9.4s                 \n"
      "       mls     v0.4s, v2.4s, v3.4s                 \n"

      // Perform complex dot product operation with rotation #90
      "       mla     v1.4s, v12.4s, v23.4s               \n"
      "       mla     v1.4s, v13.4s, v22.4s               \n"
      "       mla     v1.4s, v8.4s, v3.4s                 \n"
      "       mla     v1.4s, v2.4s, v9.4s                 \n"

      // Store the result to c array
      "       st2     {v0.4s,v1.4s}, [%[c0]], #32         \n"

      "       movi    v0.4s, #0x0                         \n"
      "       movi    v1.4s, #0x0                         \n"

      // Expand inputs to word size
      "       sxtl2   v12.4s, v14.8h                      \n"
      "       sxtl2   v22.4s, v24.8h                      \n"
      "       sxtl2   v8.4s, v18.8h                       \n"
      "       sxtl2   v9.4s, v28.8h                       \n"
      "       sxtl2   v13.4s, v16.8h                      \n"
      "       sxtl2   v23.4s, v26.8h                      \n"
      "       sxtl2   v2.4s, v10.8h                       \n"
      "       sxtl2   v3.4s, v20.8h                       \n"

      // Perform complex dot product operation with rotation #0
      "       mla     v0.4s, v12.4s, v22.4s               \n"
      "       mls     v0.4s, v13.4s, v23.4s               \n"
      "       mla     v0.4s, v8.4s, v9.4s                 \n"
      "       mls     v0.4s, v2.4s, v3.4s                 \n"

      // Perform complex dot product operation with rotation #90
      "       mla     v1.4s, v12.4s, v23.4s               \n"
      "       mla     v1.4s, v13.4s, v22.4s               \n"
      "       mla     v1.4s, v8.4s, v3.4s                 \n"
      "       mla     v1.4s, v2.4s, v9.4s                 \n"

      // Store the result to c array
      "       st2     {v0.4s,v1.4s}, [%[c0]], #32         \n"

      "       movi    v0.4s, #0x0                         \n"
      "       movi    v1.4s, #0x0                         \n"

      // Expand inputs to word size
      "       sxtl    v12.4s, v15.4h                      \n"
      "       sxtl    v22.4s, v25.4h                      \n"
      "       sxtl    v8.4s, v19.4h                       \n"
      "       sxtl    v9.4s, v29.4h                       \n"
      "       sxtl    v13.4s, v17.4h                      \n"
      "       sxtl    v23.4s, v27.4h                      \n"
      "       sxtl    v2.4s, v11.4h                       \n"
      "       sxtl    v3.4s, v21.4h                       \n"

      // Perform complex dot product operation with rotation #0
      "       mla     v0.4s, v12.4s, v22.4s               \n"
      "       mls     v0.4s, v13.4s, v23.4s               \n"
      "       mla     v0.4s, v8.4s, v9.4s                 \n"
      "       mls     v0.4s, v2.4s, v3.4s                 \n"

      // Perform complex dot product operation with rotation #90
      "       mla     v1.4s, v12.4s, v23.4s               \n"
      "       mla     v1.4s, v13.4s, v22.4s               \n"
      "       mla     v1.4s, v8.4s, v3.4s                 \n"
      "       mla     v1.4s, v2.4s, v9.4s                 \n"

      // Store the result to c array
      "       st2     {v0.4s,v1.4s}, [%[c0]], #32         \n"

      "       movi    v0.4s, #0x0                         \n"
      "       movi    v1.4s, #0x0                         \n"

      // Expand inputs to word size
      "       sxtl2   v12.4s, v15.8h                      \n"
      "       sxtl2   v22.4s, v25.8h                      \n"
      "       sxtl2   v8.4s, v19.8h                       \n"
      "       sxtl2   v9.4s, v29.8h                       \n"
      "       sxtl2   v13.4s, v17.8h                      \n"
      "       sxtl2   v23.4s, v27.8h                      \n"
      "       sxtl2   v2.4s, v11.8h                       \n"
      "       sxtl2   v3.4s, v21.8h                       \n"

      // Perform complex dot product operation with rotation #0
      "       mla     v0.4s, v12.4s, v22.4s               \n"
      "       mls     v0.4s, v13.4s, v23.4s               \n"
      "       mla     v0.4s, v8.4s, v9.4s                 \n"
      "       mls     v0.4s, v2.4s, v3.4s                 \n"

      // Perform complex dot product operation with rotation #90
      "       mla     v1.4s, v12.4s, v23.4s               \n"
      "       mla     v1.4s, v13.4s, v22.4s               \n"
      "       mla     v1.4s, v8.4s, v3.4s                 \n"
      "       mla     v1.4s, v2.4s, v9.4s                 \n"

      // Store the result to c array
      "       st2     {v0.4s,v1.4s}, [%[c0]], #32         \n"
      "       sub     %[size], %[size], #16                 \n"

      // Compare whether are >=16 elements left to continue with loop iterations
      "       cmp     %[size], #16                        \n"
      "       bge     1b                                  \n"

      // Loop ends here:
      // Process loop tail if any
      "2:                                                 \n"
      "       cbz     %[size], 3f                         \n"
      "       ldrsb   %w[reg0], [%[a0]], #1               \n"
      "       ldrsb   %w[reg1], [%[a0]], #1               \n"
      "       ldrsb   %w[reg2], [%[a0]], #1               \n"
      "       ldrsb   %w[reg3], [%[a0]], #1               \n"
      "       ldrsb   %w[reg4], [%[b0]], #1               \n"
      "       ldrsb   %w[reg5], [%[b0]], #1               \n"
      "       ldrsb   %w[reg6], [%[b0]], #1               \n"
      "       ldrsb   %w[reg7], [%[b0]], #1               \n"
      "       madd    %w[reg8], %w[reg0], %w[reg4], wzr   \n"
      "       msub    %w[reg8], %w[reg1], %w[reg5], %w[reg8] \n"
      "       madd    %w[reg8], %w[reg2], %w[reg6], %w[reg8] \n"
      "       msub    %w[reg8], %w[reg3], %w[reg7], %w[reg8] \n"
      "       madd    %w[reg9], %w[reg0], %w[reg5], wzr   \n"
      "       madd    %w[reg9], %w[reg1], %w[reg4], %w[reg9] \n"
      "       madd    %w[reg9], %w[reg2], %w[reg7], %w[reg9] \n"
      "       madd    %w[reg9], %w[reg3], %w[reg6], %w[reg9] \n"
      "       stp     %w[reg8], %w[reg9], [%[c0]], #8     \n"
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 2b                         \n"

      "3:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&r"(reg0),
        [reg1] "+&r"(reg1), [reg2] "+&r"(reg2), [reg3] "+&r"(reg3),
        [reg4] "+&r"(reg4), [reg5] "+&r"(reg5), [reg6] "+&r"(reg6),
        [reg7] "+&r"(reg7), [reg8] "+&r"(reg8), [reg9] "+&r"(reg9),
        [size] "+&r"(size)
      :
      : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "v28", "v29", "memory", "cc");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_110(struct loop_110_data *restrict input) {
  cint8_t *a0 = input->a0;
  cint8_t *b0 = input->b0;
  cint32_t *c0 = input->c0;
  uint64_t size = input->size;

  uint32_t reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0, reg6 = 0,
           reg7 = 0, reg8 = 0, reg9 = 0;

  asm volatile(
      "       cbz     %[size], 2f                         \n"
      "1:                                                 \n"

      // Load & sign-extend each byte of input to 32-bits
      "       ldrsb   %w[reg0], [%[a0]], #1               \n"
      "       ldrsb   %w[reg1], [%[a0]], #1               \n"
      "       ldrsb   %w[reg2], [%[a0]], #1               \n"
      "       ldrsb   %w[reg3], [%[a0]], #1               \n"
      "       ldrsb   %w[reg4], [%[b0]], #1               \n"
      "       ldrsb   %w[reg5], [%[b0]], #1               \n"
      "       ldrsb   %w[reg6], [%[b0]], #1               \n"
      "       ldrsb   %w[reg7], [%[b0]], #1               \n"

      // Perform complex dot product operation with rotation #0
      "       madd    %w[reg8], %w[reg0], %w[reg4], wzr   \n"
      "       msub    %w[reg8], %w[reg1], %w[reg5], %w[reg8] \n"
      "       madd    %w[reg8], %w[reg2], %w[reg6], %w[reg8] \n"
      "       msub    %w[reg8], %w[reg3], %w[reg7], %w[reg8] \n"

      // Perform complex dot product operation with rotation #90
      "       madd    %w[reg9], %w[reg0], %w[reg5], wzr   \n"
      "       madd    %w[reg9], %w[reg1], %w[reg4], %w[reg9] \n"
      "       madd    %w[reg9], %w[reg2], %w[reg7], %w[reg9] \n"
      "       madd    %w[reg9], %w[reg3], %w[reg6], %w[reg9] \n"

      // Save result to c array
      "       stp     %w[reg8], %w[reg9], [%[c0]], #8     \n"

      // Check loop exit condition
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 1b                         \n"
      "2:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&r"(reg0),
        [reg1] "+&r"(reg1), [reg2] "+&r"(reg2), [reg3] "+&r"(reg3),
        [reg4] "+&r"(reg4), [reg5] "+&r"(reg5), [reg6] "+&r"(reg6),
        [reg7] "+&r"(reg7), [reg8] "+&r"(reg8), [reg9] "+&r"(reg9),
        [size] "+&r"(size)
      :
      : "memory", "cc");
}
#else
static void inner_loop_110(struct loop_110_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(110, SC_SVE_LOOP_ATTR)
{
  struct loop_110_data data = { .size = SIZE };

  ALLOC_64B(data.a0, SIZE, "1st operand buffer");
  ALLOC_64B(data.b0, SIZE, "2nd operand buffer");
  ALLOC_64B(data.c0, SIZE, "result buffer");

  cint8_t *a = data.a0, *b = data.b0;
  cint32_t *c = data.c0;

  fill_uint8((uint8_t *)a, 4 * SIZE);
  fill_uint8((uint8_t *)b, 4 * SIZE);
  fill_uint32((uint32_t *)c, 2 * SIZE);

  inner_loops_110(iters, &data);

  uint64_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    if ((c[i].re !=
         (int32_t)(((a[2 * i].re * b[2 * i].re) - (a[2 * i].im * b[2 * i].im)) +
                   ((a[2 * i + 1].re * b[2 * i + 1].re) -
                    (a[2 * i + 1].im * b[2 * i + 1].im)))) ||
        (c[i].im !=
         (int32_t)(((a[2 * i].im * b[2 * i].re) + (a[2 * i].re * b[2 * i].im)) +
                   ((a[2 * i + 1].im * b[2 * i + 1].re) +
                    (a[2 * i + 1].re * b[2 * i + 1].im))))) {
      checksum += 1;
    }
  }

  bool passed = checksum == 0;
#ifndef STANDALONE
  FINALISE_LOOP_I(110, passed, "%"PRId64, (uint64_t) 0, checksum)
#endif
  return passed ? 0 : 1;
}
