/*----------------------------------------------------------------------------
#
#   Loop 026: Convert UTF-16 to chars
#
#   Purpose:
#     Use of gathers load instruction.
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


struct loop_026_data {
  uint16_t *p;
  uint8_t *d;
  uint16_t *lmt;
};

#define MASK_TABLE1 0x3f
#define MASK_TABLE2 0x1ff
#define MASK_TABLE3 0x1ff
#define SIZE_TABLE1 MASK_TABLE1 + 1
#define SIZE_TABLE2 MASK_TABLE2 + 1
#define SIZE_TABLE3 MASK_TABLE3 + 1
#define BAD_VALUE 0x100

static uint16_t table1[SIZE_TABLE1] __attribute__((unused));
static uint16_t table2[SIZE_TABLE2] __attribute__((unused));
static uint16_t table3[SIZE_TABLE3] __attribute__((unused));

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_026(struct loop_026_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void NOINLINE convert_utf16_to_bytes(uint16_t *restrict a,
                                            uint8_t *restrict b, int64_t n) {
  for (int i = 0; i < n; i++) {
    uint32_t raw = a[i];
    uint32_t first = table1[raw >> 10];
    first += (raw >> 4) & 0x3f;
    uint32_t second = table2[first];
    second += raw & 0xf;
    uint32_t result = table3[second];
    if (result >= BAD_VALUE) {  // very unlikely
      break;
    }
    b[i] = result;
  }
}
#elif defined(HAVE_SVE_INTRINSICS)
static void NOINLINE convert_utf16_to_bytes(uint16_t *restrict a,
                                            uint8_t *restrict b, int64_t n) {
  svuint32_t bad = svdup_u32(0x100);

  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, n, p) {
    svuint32_t raw = svld1uh_u32(p, a + i);
    svuint32_t raw_1, raw_2, raw_3;

    raw_1 = svlsr_x(p, raw, 10);
    svuint32_t tbl_1 = svld1uh_gather_index_u32(p, table1, raw_1);

    raw_2 = svlsr_x(p, raw, 4);
    raw_2 = svand_x(p, raw_2, 0x3f);
    raw_2 = svadd_x(p, raw_2, tbl_1);
    svuint32_t tbl_2 = svld1uh_gather_index_u32(p, table2, raw_2);

    raw_3 = svand_x(p, raw, 0xf);
    raw_3 = svadd_x(p, raw_3, tbl_2);
    svuint32_t tbl_3 = svld1uh_gather_index_u32(p, table3, raw_3);

    svbool_t c = svcmpge_u32(p, tbl_3, bad);
    if (svptest_any(p, c)) {
      c = svbrkb_z(p, c);
      svst1b(c, b + i, tbl_3);
      break;
    } else {
      svst1b(p, b + i, tbl_3);
    }
  }
}
#elif defined(__ARM_FEATURE_SVE)
static void NOINLINE convert_utf16_to_bytes(uint16_t *restrict a,
                                            uint8_t *restrict b, int64_t n) {
  int64_t i = 0;

  asm volatile(
      "   whilelo  p0.s, %[i], %[n]                         \n"
      "   mov      z10.s, 0x100                             \n"  // BAD_VALUE
      "1: ld1h     {z1.s}, p0/z, [%[a], %[i], lsl #1]       \n"  // raw
      "   lsr      z2.s, z1.s, #10                          \n"  // raw >> 10
      "   ld1h     {z3.s}, p0/z, [%[table1], z2.s, uxtw #1] \n"  // first
      "   lsr      z4.s, z1.s, #4                           \n"  // raw >> 4
      "   and      z4.s, z4.s, #0x3f                        \n"  // & 0x3f
      "   add      z4.s, z4.s, z3.s                         \n"  // += first
      "   ld1h     {z5.s}, p0/z, [%[table2], z4.s, uxtw #1] \n"  // second
      "   movprfx  z6, z1                                   \n"
      "   and      z6.s, z6.s, #0xf                         \n"  // raw & 0xf
      "   add      z6.s, z6.s, z5.s                         \n"  // += second
      "   ld1h     {z7.s}, p0/z, [%[table3], z6.s, uxtw #1] \n"  // result
      "   cmpge    p1.s, p0/z, z7.s, z10.s                  \n"  // >=BAD_VALUE
      "   b.any    2f                                       \n"
      "   st1b     {z7.s}, p0, [%[b], %[i]]                 \n"  // b[i] = rslt
      "   incw     %[i]                                     \n"
      "   whilelo  p0.s, %[i], %[n]                         \n"
      "   b.any    1b                                       \n"
      "   b        3f                                       \n"
      "2: brkb     p1.b, p0/z, p1.b                         \n"
      "   st1b     {z7.s}, p1, [%[b], %[i]]                 \n"  // b[i] = rslt
      "3:                                                   \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [n] "r"(n), [a] "r"(a), [b] "r"(b), [table1] "r"(table1),
        [table2] "r"(table2), [table3] "r"(table3)
      : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "p0", "p1",
        "cc", "memory");
}
// No Neon version
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
// Original code has similar unrolled optimization
static void NOINLINE convert_utf16_to_bytes(uint16_t *restrict a,
                                            uint8_t *restrict b, int64_t n) {
  uint16_t c, v;
  uint16_t check;
  uint16_t *src = a;
  uint8_t *dst = b;
  int i = 0;

  for (; i < n - 3; i += 4) {
    c = *src++;
    check = v = table3[table2[table1[c >> 10] + ((c >> 4) & 0x3f)] + (c & 0xf)];
    *dst++ = (uint8_t)v;
    c = *src++;
    check |= v =
        table3[table2[table1[c >> 10] + ((c >> 4) & 0x3f)] + (c & 0xf)];
    *dst++ = (uint8_t)v;
    c = *src++;
    check |= v =
        table3[table2[table1[c >> 10] + ((c >> 4) & 0x3f)] + (c & 0xf)];
    *dst++ = (uint8_t)v;
    c = *src++;
    check |= v =
        table3[table2[table1[c >> 10] + ((c >> 4) & 0x3f)] + (c & 0xf)];
    *dst++ = (uint8_t)v;

    if (check >= BAD_VALUE) {
      src -= 4;
      dst -= 4;
      break;
    }
  }

  for (; i < n; i++) {
    c = *src++;
    v = table3[table2[table1[c >> 10] + ((c >> 4) & 0x3f)] + (c & 0xf)];
    if (v >= BAD_VALUE) {  // very unlikely
      break;
    }
    *dst++ = (uint8_t)v;
  }
}
#else
static void inner_loop_026(struct loop_026_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_026(struct loop_026_data *restrict data) {
  uint16_t *p = data->p;
  uint8_t *d = data->d;
  uint16_t *lmt = data->lmt;

  while (p < lmt) {
    uint16_t length = p[0];
    convert_utf16_to_bytes(p, d, length);
    p += length;
    d += length;
  }
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 2000
#endif

LOOP_DECL(026, NS_SVE_LOOP_ATTR)
{
  struct loop_026_data data;

  ALLOC_64B(data.p, SIZE, "input data");
  ALLOC_64B(data.d, SIZE, "output array");

  fill_uint16(data.p, SIZE);
  fill_uint8 (data.d, SIZE);
  fill_uint16(table1, SIZE_TABLE1);
  fill_uint16(table2, SIZE_TABLE2);
  fill_uint16(table3, SIZE_TABLE3);

  for (int i = 0; i < SIZE_TABLE1; i++) {
    table1[i] = table1[i] & (MASK_TABLE2 - 0x3f);
  }

  for (int i = 0; i < SIZE_TABLE2; i++) {
    table2[i] = table2[i] & (MASK_TABLE3 - 0xf);
  }

  for (int i = 0; i < SIZE_TABLE3; i++) {
    table3[i] = table3[i] & 0xff;
  }

  int min_length = 55;
  int max_length = 255;
  int range = max_length - min_length;
  int start = 0;
  int end = (rand_uint32() % range) + min_length;
  int last = 0;
  while (end < SIZE) {
    data.p[start] = end - start;
    last = end;
    start = end;
    end = start + (rand_uint32() % range) + min_length;
  }
  data.lmt = data.p + last;

  inner_loops_026(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    checksum ^= data.d[i] << (8 * (i % 4));
  }

  uint32_t correct = 0xbe444cd1;
  bool passed = checksum == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(26, passed, "0x%08"PRIx32, correct, checksum)
#endif
  return passed ? 0 : 1;
}
