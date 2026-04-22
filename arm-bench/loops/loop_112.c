/*----------------------------------------------------------------------------
#
#   Loop 112: UINT32 complex MAC
#
#   Purpose:
#     Use of CMLA instruction.
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


typedef struct cuint32_t {
  uint32_t re, im;
} cuint32_t;

struct loop_112_data {
  cuint32_t *restrict a0;
  cuint32_t *restrict b0;
  cuint32_t *restrict c0;
  uint64_t size;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_112(struct loop_112_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_112(struct loop_112_data *restrict input) {
  cuint32_t *restrict a = input->a0;
  cuint32_t *restrict b = input->b0;
  cuint32_t *restrict c = input->c0;
  uint64_t size = input->size;

  uint64_t i;
  for (i = 0; i < size; i++) {
    c[i].re = (a[i].re * b[i].re) - (a[i].im * b[i].im);
    c[i].im = (a[i].re * b[i].im) + (a[i].im * b[i].re);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_112(struct loop_112_data *restrict input)
LOOP_ATTR
{
  cuint32_t *restrict a0 = input->a0;
  cuint32_t *restrict b0 = input->b0;
  cuint32_t *restrict c0 = input->c0;
  uint64_t size = input->size;

  svbool_t p;
  FOR_LOOP_64(uint64_t, i, 0, size, p) {
    svuint32_t a1 = svreinterpret_u32(svld1_u64(p, (uint64_t *)(a0 + i)));
    svuint32_t b1 = svreinterpret_u32(svld1_u64(p, (uint64_t *)(b0 + i)));
    svuint32_t c1 = svdup_u32(0);
    c1 = svcmla(c1, a1, b1, 0);
    c1 = svcmla(c1, a1, b1, 90);
    svst1(p, (uint64_t *)(c0 + i), svreinterpret_u64(c1));
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))

static void inner_loop_112(struct loop_112_data *restrict input)
LOOP_ATTR
{
  cuint32_t *restrict a = input->a0;
  cuint32_t *restrict b = input->b0;
  cuint32_t *restrict c = input->c0;
  uint64_t size = input->size * 2;
  uint64_t count = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       whilelt pn8.s, %[i], %[n], vlx2                     \n"
      "       b.none  2f                                          \n"

      // Loop begins here:
      // This loop is unrolled such that add operation is performed on upto
      // 2*(VL/64) complex elements in a single iteration
      "1:                                                         \n"
      "       dup     z0.s, #0                                    \n"
      "       dup     z1.s, #0                                    \n"

      // Load complex elements from a and b arrays
      "       ld1w    {z10.s-z11.s}, pn8/z, [%[a], %[i], lsl #2]  \n"
      "       ld1w    {z20.s-z21.s}, pn8/z, [%[b], %[i], lsl #2]  \n"

      // Integer Complex multiplication
      "       cmla    z0.s, z10.s, z20.s, #0                      \n"
      "       cmla    z0.s, z10.s, z20.s, #90                     \n"
      "       cmla    z1.s, z11.s, z21.s, #0                      \n"
      "       cmla    z1.s, z11.s, z21.s, #90                     \n"

      // Store result to c array
      "       st1w    {z0.s-z1.s}, pn8, [%[c], %[i], lsl #2]      \n"

      // Increment processed elements by 2*(VL/64) to account for unrolled loop
      "       incw    %[i], all, mul #2                           \n"
      "       whilelt pn8.s, %[i], %[n], vlx2                     \n"
      "       b.first 1b                                          \n"

      // End of operation
      "2:                                                         \n"

      : [i] "+&r"(count)
      : [a] "r"(a), [b] "r"(b), [c] "r"(c), [n] "r"(size)
      : "z0", "z1", "z10", "z11", "z20", "z21", "p8", "cc", "memory");
}
#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_112(struct loop_112_data *restrict input)
LOOP_ATTR
{
  cuint32_t *restrict a0 = input->a0;
  cuint32_t *restrict b0 = input->b0;
  cuint32_t *restrict c0 = input->c0;
  uint64_t size = input->size;

  uint64_t count = 0, count_1 = 0;
  uint64_t a1 = 0, b1 = 0, c1 = 0;

  asm volatile(
      // Check if there are any elements to process before start of loop
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
      "       ld1d    {z20.d}, p0/z, [%[b0], %[count], lsl #3] \n"
      "       ld1d    {z11.d}, p1/z, [%[a1], %[count], lsl #3] \n"
      "       ld1d    {z21.d}, p1/z, [%[b1], %[count], lsl #3] \n"

      // Integer Complex multiplication
      "       cmla    z0.s, z10.s, z20.s, #0              \n"
      "       cmla    z0.s, z10.s, z20.s, #90             \n"
      "       cmla    z1.s, z11.s, z21.s, #0              \n"
      "       cmla    z1.s, z11.s, z21.s, #90             \n"

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
      "       ld1d    {z20.d}, p0/z, [%[b0], %[count], lsl #3] \n"
      "       cmla    z0.s, z10.s, z20.s, #0              \n"
      "       cmla    z0.s, z10.s, z20.s, #90             \n"
      "       st1d    {z0.d}, p0, [%[c0], %[count], lsl #3]    \n"

      // End of operation
      "3:                                                 \n"
      : [a1] "+&r"(a1), [b1] "+&r"(b1), [c1] "+&r"(c1), [count] "+&r"(count),
        [count_1] "+&r"(count_1)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [size] "r"(size)
      : "z0", "z1", "z10", "z11", "z20", "z21", "p0", "p1", "memory", "cc");
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_112(struct loop_112_data *restrict input)
LOOP_ATTR
{
  cuint32_t *restrict a0 = input->a0;
  cuint32_t *restrict b0 = input->b0;
  cuint32_t *restrict c0 = input->c0;
  uint64_t size = input->size;

  uint64_t count = 0, count_1 = 0;
  uint64_t size_32 = (size << 1);

  asm volatile(
      // Check if there are any elements to process before start of loop
      "       whilelt p0.s, %[count], %[size_32]          \n"
      "       b.none  3f                                  \n"
      "       cntw    %[count_1]                          \n"
      "       whilelt p1.s, %[count_1], %[size_32]        \n"
      "       b.none  2f                                  \n"

      // Loop begins here:
      // This loop is unrolled such that multiply operation is performed on
      // 2*(VL/64) complex elements in a single iteration

      "1:                                                 \n"
      "       uzp1    p2.s, p0.s, p1.s                    \n"

      "       dup     z0.s, #0                            \n"
      "       dup     z1.s, #0                            \n"

      // Load complex elements from a and b arrays
      "       ld2w    {z10.s,z11.s}, p2/z, [%[a0], %[count], lsl #2] \n"
      "       ld2w    {z20.s,z21.s}, p2/z, [%[b0], %[count], lsl #2] \n"

      // Integer Complex multiplication
      "       mla    z0.s, p2/m, z10.s, z20.s                  \n"
      "       mls    z0.s, p2/m, z11.s, z21.s                  \n"
      "       mla    z1.s, p2/m, z10.s, z21.s                  \n"
      "       mla    z1.s, p2/m, z11.s, z20.s                  \n"

      // Store result to c array
      "       st2w    {z0.s,z1.s}, p2, [%[c0], %[count], lsl #2] \n"

      // Increment processed elements by 2*(VL/64) to account for unrolled loop
      "       incw    %[count], all, mul #2               \n"
      "       whilelt p1.s, %[count], %[size_32]          \n"
      "       b.first 1b                                  \n"

      // Loop ends here:
      // Since loop was unrolled, check if <=VL/64 elements are available to
      // process
      "       decw    %[count]                            \n"
      "       whilelt p0.s, %[count], %[size_32]          \n"
      "       b.none  3f                                  \n"

      // Process last <=(VL/64) complex elements
      "2:                                                 \n"
      "       pfalse  p1.b                                \n"
      "       uzp1    p2.s, p0.s, p1.s                    \n"
      "       dup     z0.s, #0                            \n"
      "       dup     z1.s, #0                            \n"
      "       ld2w    {z10.s,z11.s}, p2/z, [%[a0], %[count], lsl #2] \n"
      "       ld2w    {z20.s,z21.s}, p2/z, [%[b0], %[count], lsl #2] \n"
      "       mla     z0.s, p2/m, z10.s, z20.s                  \n"
      "       mls     z0.s, p2/m, z11.s, z21.s                  \n"
      "       mla     z1.s, p2/m, z10.s, z21.s                  \n"
      "       mla     z1.s, p2/m, z11.s, z20.s                  \n"
      "       st2w    {z0.s,z1.s}, p2, [%[c0], %[count], lsl #2]  \n"

      // End of operation
      "3:                                                 \n"
      : [count] "+&r"(count), [count_1] "+&r"(count_1)
      : [a0] "r"(a0), [b0] "r"(b0), [c0] "r"(c0), [size_32] "r"(size_32)
      : "z0", "z1", "z10", "z11", "z20", "z21", "p0", "p1", "p2", "memory",
        "cc");
}
#elif defined(__ARM_NEON)

static void inner_loop_112(struct loop_112_data *restrict input) {
  cuint32_t *restrict a0 = input->a0;
  cuint32_t *restrict b0 = input->b0;
  cuint32_t *restrict c0 = input->c0;
  uint64_t size = input->size;

  uint32_t reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0;

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
      "       mla     v0.4s, v10.4s, v20.4s               \n"
      "       mls     v0.4s, v11.4s, v21.4s               \n"
      "       mla     v1.4s, v10.4s, v21.4s               \n"
      "       mla     v1.4s, v11.4s, v20.4s               \n"

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
      "       ldp     %w[reg0], %w[reg1], [%[a0]], #8     \n"
      "       ldp     %w[reg2], %w[reg3], [%[b0]], #8     \n"
      "       madd    %w[reg4], %w[reg0], %w[reg2], wzr   \n"
      "       msub    %w[reg4], %w[reg1], %w[reg3], %w[reg4]    \n"
      "       madd    %w[reg5], %w[reg0], %w[reg3], wzr   \n"
      "       madd    %w[reg5], %w[reg1], %w[reg2], %w[reg5]    \n"
      "       stp     %w[reg4], %w[reg5], [%[c0]], #8     \n"
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 2b                         \n"
      "3:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&r"(reg0),
        [reg1] "+&r"(reg1), [reg2] "+&r"(reg2), [reg3] "+&r"(reg3),
        [reg4] "+&r"(reg4), [reg5] "+&r"(reg5), [size] "+&r"(size)
      :
      : "v0", "v1", "v10", "v11", "v20", "v21", "v0", "v1", "memory", "cc");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static void inner_loop_112(struct loop_112_data *restrict input) {
  cuint32_t *restrict a0 = input->a0;
  cuint32_t *restrict b0 = input->b0;
  cuint32_t *restrict c0 = input->c0;
  uint64_t size = input->size;

  uint32_t reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0, reg4 = 0, reg5 = 0;

  asm volatile(
      "       cbz     %[size], 2f                         \n"
      "1:                                                 \n"
      "       ldp     %w[reg0], %w[reg1], [%[a0]], #8     \n"
      "       ldp     %w[reg2], %w[reg3], [%[b0]], #8     \n"
      "       madd    %w[reg4], %w[reg0], %w[reg2], wzr   \n"
      "       msub    %w[reg4], %w[reg1], %w[reg3], %w[reg4]    \n"
      "       madd    %w[reg5], %w[reg0], %w[reg3], wzr   \n"
      "       madd    %w[reg5], %w[reg1], %w[reg2], %w[reg5]    \n"
      "       stp     %w[reg4], %w[reg5], [%[c0]], #8     \n"
      "       sub     %[size], %[size], #1                \n"
      "       cbnz    %[size], 1b                         \n"
      "2:                                                 \n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [c0] "+&r"(c0), [reg0] "+&r"(reg0),
        [reg1] "+&r"(reg1), [reg2] "+&r"(reg2), [reg3] "+&r"(reg3),
        [reg4] "+&r"(reg4), [reg5] "+&r"(reg5), [size] "+&r"(size)
      :
      : "memory", "cc");
}
#else

static void inner_loop_112(struct loop_112_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(112, SC_SVE_LOOP_ATTR)
{
  struct loop_112_data data = { .size = SIZE };

  ALLOC_64B(data.a0, SIZE, "1st operand buffer");
  ALLOC_64B(data.b0, SIZE, "2nd operand buffer");
  ALLOC_64B(data.c0, SIZE, "result buffer");

  cuint32_t *a = data.a0, *b = data.b0, *c = data.c0;

  fill_uint32((uint32_t *)a, 2 * SIZE);
  fill_uint32((uint32_t *)b, 2 * SIZE);
  fill_uint32((uint32_t *)c, 2 * SIZE);

  inner_loops_112(iters, &data);

  uint64_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    if ((c[i].re != ((a[i].re * b[i].re) - (a[i].im * b[i].im))) ||
        (c[i].im != ((a[i].re * b[i].im) + (a[i].im * b[i].re)))) {
      checksum += 1;
    }
  }

  bool passed = checksum == 0;
#ifndef STANDALONE
  FINALISE_LOOP_I(112, passed, "%"PRId64, (uint64_t) 0, checksum)
#endif
  return passed ? 0 : 1;
}
