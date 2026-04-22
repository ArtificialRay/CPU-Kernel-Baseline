/*----------------------------------------------------------------------------
#
#   Loop 107: UINT128 multiply
#
#   Purpose:
#     Use of ADCL[B/T] instructions.
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


#ifndef __SIZEOF_INT128__  // GNU C
#error "Need int128"
#endif

#define uint128_t unsigned __int128

typedef struct uint256_t {
  uint128_t low, high;
} uint256_t;

struct loop_107_data {
  uint128_t *restrict a;
  uint128_t *restrict b;
  uint256_t *restrict c;
  int64_t n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_107(struct loop_107_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static uint256_t mult256(uint128_t x, uint128_t y) {
  uint64_t a = x >> 64, b = x;
  uint64_t c = y >> 64, d = y;
  // (a*2^64 + b) * (c*2^64 + d) =
  // (a*c) * 2^128 + (a*d + b*c)*2^64 + (b*d)
  uint128_t ac = (uint128_t)a * c;
  uint128_t ad = (uint128_t)a * d;
  uint128_t bc = (uint128_t)b * c;
  uint128_t bd = (uint128_t)b * d;
  uint128_t carry =
      (uint128_t)(uint64_t)ad + (uint128_t)(uint64_t)bc + (bd >> 64u);
  uint128_t high = ac + (ad >> 64u) + (bc >> 64u) + (carry >> 64u);
  uint128_t low = (ad << 64u) + (bc << 64u) + bd;
  return (uint256_t){low, high};
}
static void inner_loop_107(struct loop_107_data *restrict input) {
  uint128_t *restrict a = input->a;
  uint128_t *restrict b = input->b;
  uint256_t *restrict c = input->c;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    c[i] = mult256(a[i], b[i]);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_107(struct loop_107_data *restrict input)
LOOP_ATTR
{
  uint128_t *restrict a = input->a;
  uint128_t *restrict b = input->b;
  uint256_t *restrict c = input->c;
  int64_t n = input->n;

  svuint64_t zeros = svdup_u64(0);

  // This could eventually make use of the SVE2p1 REVD or EXTQ instructions.
  svuint64_t idx = svreinterpret_u64(svindex_u32(0, 1));
  idx = svrevw_z(svptrue_b8(), idx);
  idx = svunpklo(svreinterpret_u32(idx));

  svbool_t p;
  FOR_LOOP_64(int64_t, i, 0, n * 2, p) {
    svuint64_t a_vec = svld1(p, ((uint64_t *)a) + i);
    svuint64_t b_vec = svld1(p, ((uint64_t *)b) + i);
    svuint64_t a_sel = svtbl(a_vec, idx);

    svuint64_t term1 = svmul_x(p, a_vec, b_vec);
    svuint64_t term2 = svmul_x(p, a_sel, b_vec);

    svuint64_t term3 = svmulh_x(p, a_vec, b_vec);
    svuint64_t term4 = svmulh_x(p, a_sel, b_vec);

    svuint64_t term5 = svtrn2(term1, zeros);
    svuint64_t term6 = svtrn2(term3, zeros);

    term3 = svadclb(term3, term2, zeros);
    term5 = svadclb(term5, term4, term3);
    term6 = svadclb(term6, zeros, term5);
    term3 = svadclt(term3, term2, zeros);
    term5 = svadclt(term5, term4, term3);
    term6 = svadclb(term6, zeros, term5);

    svuint64_t c_lo = svtrn1(term1, term5);
    svuint64_t c_hi = svtrn1(term3, term6);
    svst2(p, ((uint64_t *)c) + (i * 2), svcreate2(c_lo, c_hi));
  }
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))
static void inner_loop_107(struct loop_107_data *restrict input)
LOOP_ATTR
{
  uint128_t *restrict a = input->a;
  uint128_t *restrict b = input->b;
  uint256_t *restrict c = input->c;
  int64_t n = input->n;

  int64_t i = 0;
  int64_t ii = 0;
  int64_t nn = n * 2;

  /*
  RES[0] = MUL(X[0], Y[0])
  RES[1] = MULH(X[0], Y[0]) + MUL(X[1], Y[0]) + MUL(X[0], Y[1])
  RES[2] = MUL(X[1], Y[1]) + MULH(X[1], Y[0]) + MULH(X[0], Y[1]) + carry(RES[1])
  RES[3] = MULH(X[1], Y[1]) + carry(RES[2])
  */

  asm volatile(
      "   mov     z10.d, #0                                 \n"
      "   ptrue   p0.d                                      \n"
      "   index   z3.s, #0, #1                              \n"
      "   revw    z3.d, p0/m, z3.d                          \n"
      "   uunpklo z3.d, z3.s                                \n"

      "1: ld1d    {z0.d}, p0/z, [%[a], %[i], lsl #3]        \n"
      "   ld1d    {z2.d}, p0/z, [%[b], %[i], lsl #3]        \n"
      "   tbl     z1.d, z0.d, z3.d                          \n"

      "   mul     z20.d, z0.d, z2.d                         \n"
      "   umulh   z24.d, z0.d, z2.d                         \n"
      "   mul     z22.d, z1.d, z2.d                         \n"
      "   umulh   z26.d, z1.d, z2.d                         \n"

      "   trn2    z23.d, z20.d, z10.d                       \n"
      "   trn2    z27.d, z24.d, z10.d                       \n"
      "   incd    %[i]                                      \n"

      "   adclb   z24.d, z22.d, z10.d                       \n"
      "   adclb   z23.d, z26.d, z24.d                       \n"
      "   adclb   z27.d, z10.d, z23.d                       \n"
      "   adclt   z24.d, z22.d, z10.d                       \n"
      "   adclt   z23.d, z26.d, z24.d                       \n"
      "   adclb   z27.d, z10.d, z23.d                       \n"

      "   trn1    z0.d, z20.d, z23.d                        \n"
      "   trn1    z1.d, z24.d, z27.d                        \n"

      "   st2d    {z0.d, z1.d}, p0, [%[c], %[ii], lsl #3]   \n"

      "   incd    %[ii], all, mul #2                        \n"
      "   whilelo p0.d, %[i], %[nn]                         \n"
      "   b.any   1b                                        \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [ii] "+&r"(ii)
      : [a] "r"(a), [b] "r"(b), [c] "r"(c), [nn] "r"(nn)
      : "v0", "v1", "v2", "v3", "v10", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "p0", "cc", "memory");
}
#else
static void inner_loop_107(struct loop_107_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef STANDALONE
static void print_hex128(uint128_t n) {
  printf("%016" PRIx64 " %016" PRIx64, (uint64_t)(n >> 64), (uint64_t)n);
}

static void print_hex256(uint256_t n) {
  print_hex128(n.high);
  printf(" ");
  print_hex128(n.low);
}
#endif

static uint256_t build_uint256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  uint256_t res = {0};
  res.high = (((uint128_t)a) << 64) + b;
  res.low = (((uint128_t)c) << 64) + d;
  return res;
}

#ifndef SIZE
#define SIZE 1000
#endif

LOOP_DECL(107, SC_SVE_LOOP_ATTR)
{
  struct loop_107_data data = { .n = SIZE };

  ALLOC_64B(data.a, SIZE, "1st operand array");
  ALLOC_64B(data.b, SIZE, "2nd operand array");
  ALLOC_64B(data.c, SIZE, "result array");

  fill_uint64((void *)data.a, 2 * SIZE);
  fill_uint64((void *)data.b, 2 * SIZE);
  fill_uint64((void *)data.c, 4 * SIZE);

  inner_loops_107(iters, &data);

  uint256_t checksum = {0};
  for (int i = 0; i < SIZE; i++) {
    checksum.high ^= data.c[i].high;
    checksum.low ^= data.c[i].low;
  }

  uint256_t correct = build_uint256(0xd58362e6a5975321L, 0xe7ebcf4b56db310dL,
                                    0x607e5e66f9b4e2c2L, 0x2dc5b356d3eda60aL);
  bool passed = checksum.high == correct.high && checksum.low == correct.low;
#ifndef STANDALONE
  FINALISE_LOOP_U256(107, passed, correct, checksum)
#endif
  return passed ? 0 : 1;
}
