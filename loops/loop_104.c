/*----------------------------------------------------------------------------
#
#   Loop 104: Byte historgram
#
#   Purpose:
#     Use of HISTSEG instruction.
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


struct loop_104_data {
  uint32_t *histogram;
  uint64_t histogram_size;
  uint8_t *data;
  int n;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_104(struct loop_104_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
void NOINLINE update(uint32_t *restrict histogram, uint8_t *data, int n) {
  for (int i = 0; i < n; i++) histogram[data[i]] += 1;
}
#elif defined(HAVE_SVE_INTRINSICS)
#define IDX(i, o) svuint8_t idx_##i = svadd_x(all, idx_0, o)
#define ACC(n) svuint8_t acc_##n = svdup_u8(0)

static void NOINLINE histogram_1(uint32_t *hist, uint8_t *data, uint64_t n) {
  svbool_t all = svptrue_b8();
  svuint8_t cap = svdup_u8(191);
  svuint8_t top = svdup_u8(0x3f);

  svuint8_t idx_0 = svindex_u8(0, 1);
  IDX(1, 16);
  IDX(2, 32);
  IDX(3, 48);
  IDX(4, 64);
  IDX(5, 80);
  IDX(6, 96);
  IDX(7, 112);

  for (uint64_t i = 0; i < n;) {
    ACC(0);
    ACC(1);
    ACC(2);
    ACC(3);
    ACC(4);
    ACC(5);
    ACC(6);
    ACC(7);
    ACC(8);
    ACC(9);
    ACC(a);
    ACC(b);
    ACC(c);
    ACC(d);
    ACC(e);
    ACC(f);

    while (1) {
      svuint8_t val_0 = svld1rq(all, data + i);
      svuint8_t val_1 = svand_x(all, val_0, top);
      svbool_t cmp = svcmpgt(all, val_0, 127);
      val_1 = svsel(cmp, val_1, cap);

#define STEP(i, j, k) \
  acc_##i = svadd_x(all, acc_##i, svhistseg(idx_##j, val_##k))
      STEP(0, 0, 0);
      STEP(8, 0, 1);
      STEP(1, 1, 0);
      STEP(9, 1, 1);
      STEP(2, 2, 0);
      STEP(a, 2, 1);
      STEP(3, 3, 0);
      STEP(b, 3, 1);
      STEP(4, 4, 0);
      STEP(c, 4, 1);
      STEP(5, 5, 0);
      STEP(d, 5, 1);
      STEP(6, 6, 0);
      STEP(e, 6, 1);
      STEP(7, 7, 0);
      STEP(f, 7, 1);
#undef STEP

      i += 16;
      if (i >= n) {
        break;
      } else if (i % 64 == 0) {
        val_0 = acc_0;
        val_1 = acc_1;

#define UMAX(i, j) val_##i = svmax_x(all, val_##i, acc_##j)
        UMAX(0, 2);
        UMAX(1, 3);
        UMAX(0, 4);
        UMAX(1, 5);
        UMAX(0, 6);
        UMAX(1, 7);
        UMAX(0, 8);
        UMAX(1, 9);
        UMAX(0, a);
        UMAX(1, b);
        UMAX(0, c);
        UMAX(1, d);
        UMAX(0, e);
        UMAX(1, f);
#undef UMAX

        cmp = svcmplt(all, cap, svmax_x(all, val_0, val_1));
        if (svptest_any(cmp, cmp)) break;
      }
    }

    svuint16_t lo, hi;
    svuint32x2_t v1, v2;
#define WB_128(n, o)                              \
  v1 = svld2_vnum(all, ptr, 0x##o);               \
  v2 = svld2_vnum(all, ptr, 0x##o + 2);           \
  lo = svunpklo(acc_##n);                         \
  hi = svunpkhi(acc_##n);                         \
  v1 = svset2(v1, 0, svaddwb(svget2(v1, 0), lo)); \
  v1 = svset2(v1, 1, svaddwt(svget2(v1, 1), lo)); \
  v2 = svset2(v2, 0, svaddwb(svget2(v2, 0), hi)); \
  v2 = svset2(v2, 1, svaddwt(svget2(v2, 1), hi)); \
  svst2_vnum(all, ptr, 0x##o, v1);                \
  svst2_vnum(all, ptr, 0x##o + 2, v2);

    uint32_t *ptr = hist;
    WB_128(0, 0);
    WB_128(1, 4);
    WB_128(2, 8);
    WB_128(3, c);
    ptr += 4 * svcntb();
    WB_128(4, 0);
    WB_128(5, 4);
    WB_128(6, 8);
    WB_128(7, c);
    ptr += 4 * svcntb();
    WB_128(8, 0);
    WB_128(9, 4);
    WB_128(a, 8);
    WB_128(b, c);
    ptr += 4 * svcntb();
    WB_128(c, 0);
    WB_128(d, 4);
    WB_128(e, 8);
    WB_128(f, c);
  }
}

static void NOINLINE histogram_2(uint32_t *hist, uint8_t *data, uint64_t n) {
  svbool_t all = svptrue_b8();
  svuint8_t cap = svdup_u8(191);

  svuint8_t idx_0 = svindex_u8(0, 1);
  IDX(1, 32);
  IDX(2, 64);
  IDX(3, 96);
  IDX(4, 128);
  IDX(5, 160);
  IDX(6, 192);
  IDX(7, 224);

  for (uint64_t i = 0; i < n;) {
    ACC(0);
    ACC(1);
    ACC(2);
    ACC(3);
    ACC(4);
    ACC(5);
    ACC(6);
    ACC(7);

    while (1) {
      svuint8_t val = svld1rq(all, data + i);

#define STEP(i) acc_##i = svadd_x(all, acc_##i, svhistseg(idx_##i, val))
      STEP(0);
      STEP(1);
      STEP(2);
      STEP(3);
      STEP(4);
      STEP(5);
      STEP(6);
      STEP(7);
#undef STEP

      i += 16;
      if (i >= n) {
        break;
      } else if (i % 64 == 0) {
        svuint8_t val_0 = acc_0;
        svuint8_t val_1 = acc_1;

#define UMAX(i, j) val_##i = svmax_x(all, val_##i, acc_##i)
        UMAX(0, 2);
        UMAX(1, 3);
        UMAX(0, 4);
        UMAX(1, 5);
        UMAX(0, 6);
        UMAX(1, 7);
#undef UMAX

        svbool_t cmp = svcmplt(all, cap, svmax_x(all, val_0, val_1));
        if (svptest_any(cmp, cmp)) break;
      }
    }

#if LD2W
    svuint16_t lo, hi;
    svuint32x2_t v11, v21, v12, v22;
#define WB_256(m, n, o)                                    \
  v1##m = svld2_vnum(all, ptr, 0x##o);                     \
  v2##m = svld2_vnum(all, ptr, 0x##o + 2);                 \
  lo = svunpklo(acc_##n);                                  \
  hi = svunpkhi(acc_##n);                                  \
  v1##m = svset2(v1##m, 0, svaddwb(svget2(v1##m, 0), lo)); \
  v1##m = svset2(v1##m, 1, svaddwt(svget2(v1##m, 1), lo)); \
  v2##m = svset2(v2##m, 0, svaddwb(svget2(v2##m, 0), hi)); \
  v2##m = svset2(v2##m, 1, svaddwt(svget2(v2##m, 1), hi)); \
  svst2_vnum(all, ptr, 0x##o, v1##m);                      \
  svst2_vnum(all, ptr, 0x##o + 2, v2##m);

    uint32_t *ptr = hist;
    WB_256(1, 0, 0);
    WB_256(2, 1, 4);
    WB_256(1, 2, 8);
    WB_256(2, 3, c);
    ptr += 4 * svcntb();
    WB_256(1, 4, 0);
    WB_256(2, 5, 4);
    WB_256(1, 6, 8);
    WB_256(2, 7, c);
#else
    svuint16_t lo1, lo2, hi1, hi2;
    svuint32x4_t v1, v2;
#define WB_256(m, n, o)                                    \
  v##m = svld4_vnum(all, hist, o);                         \
  lo##m = svextb_x(all, svreinterpret_u16(acc_##n));       \
  hi##m = svlsr_x(all, svreinterpret_u16(acc_##n), 8);     \
  v##m = svset4(v##m, 0, svaddwb(svget4(v##m, 0), lo##m)); \
  v##m = svset4(v##m, 2, svaddwt(svget4(v##m, 2), lo##m)); \
  v##m = svset4(v##m, 1, svaddwb(svget4(v##m, 1), hi##m)); \
  v##m = svset4(v##m, 3, svaddwt(svget4(v##m, 3), hi##m)); \
  svst4_vnum(all, hist, o, v##m);

    WB_256(1, 0, 0);
    WB_256(2, 1, 4);
    WB_256(1, 2, 8);
    WB_256(2, 3, 12);
    WB_256(1, 4, 16);
    WB_256(2, 5, 20);
    WB_256(1, 6, 24);
    WB_256(2, 7, 28);
#endif
  }
}

static void NOINLINE histogram_4(uint32_t *hist, uint8_t *data, uint64_t n) {
  svbool_t all = svptrue_b8();
  svuint8_t cap = svdup_u8(191);

  svuint8_t idx_0 = svindex_u8(0, 1);
  IDX(1, 64);
  IDX(2, 128);
  IDX(3, 192);

  for (uint8_t *end = data + n; data < end;) {
    ACC(0);
    ACC(1);
    ACC(2);
    ACC(3);

    while (1) {
      svuint8_t val_0 = svld1rq(all, data + 0);
      svuint8_t val_1 = svld1rq(all, data + 16);
      svuint8_t val_2 = svld1rq(all, data + 32);
      svuint8_t val_3 = svld1rq(all, data + 48);

#define STEP(i, j) acc_##i = svadd_x(all, acc_##i, svhistseg(idx_##i, val_##j))
      STEP(0, 0);
      STEP(0, 1);
      STEP(0, 2);
      STEP(0, 3);
      STEP(1, 0);
      STEP(1, 1);
      STEP(1, 2);
      STEP(1, 3);
      STEP(2, 0);
      STEP(2, 1);
      STEP(2, 2);
      STEP(2, 3);
      STEP(3, 0);
      STEP(3, 1);
      STEP(3, 2);
      STEP(3, 3);
#undef STEP

      data += 64;
      if (data >= end) {
        break;
      } else {
        svbool_t cmp_0 = svcmpge(all, cap, acc_0);
        svbool_t cmp_1 = svcmpge(all, cap, acc_1);
        svbool_t cmp_2 = svcmpge(all, cap, acc_2);
        svbool_t cmp_3 = svcmpge(all, cap, acc_3);

        cmp_0 = svand_z(cmp_0, cmp_1, cmp_2);
        cmp_3 = svnand_z(all, cmp_0, cmp_3);
        if (svptest_any(cmp_3, cmp_3)) break;
      }
    }

// TODO: simplify and let compiler re-order loads/stores?
#if LD2W
    svuint16_t lo, hi;
    svuint32x2_t v11, v21, v12, v22;

#define LOAD_512(m, o)                      \
  v1##m = svld2_vnum(all, hist, 0x##o + 0); \
  v2##m = svld2_vnum(all, hist, 0x##o + 2);

#define ADDW_512(m)                                        \
  v1##m = svset2(v1##m, 0, svaddwb(svget2(v1##m, 0), lo)); \
  v1##m = svset2(v1##m, 1, svaddwt(svget2(v1##m, 1), lo)); \
  v2##m = svset2(v2##m, 0, svaddwb(svget2(v2##m, 0), hi)); \
  v2##m = svset2(v2##m, 1, svaddwt(svget2(v2##m, 1), hi));

#define STOR_512(m, o)                     \
  svst2_vnum(all, hist, 0x##o + 0, v1##m); \
  svst2_vnum(all, hist, 0x##o + 2, v2##m);

#define UNPK_512(m, n)    \
  lo = svunpklo(acc_##n); \
  hi = svunpkhi(acc_##n)
#else
    svuint16_t lo1, lo2, hi1, hi2;
    svuint32x4_t v1, v2;

#define LOAD_512(m, o) v##m = svld4_vnum(all, hist, 0x##o)

#define ADDW_512(m)                                        \
  v##m = svset4(v##m, 0, svaddwb(svget4(v##m, 0), lo##m)); \
  v##m = svset4(v##m, 2, svaddwt(svget4(v##m, 2), lo##m)); \
  v##m = svset4(v##m, 1, svaddwb(svget4(v##m, 1), hi##m)); \
  v##m = svset4(v##m, 3, svaddwt(svget4(v##m, 3), hi##m));

#define UNPK_512(m, n)                               \
  lo##m = svextb_x(all, svreinterpret_u16(acc_##n)); \
  hi##m = svlsr_x(all, svreinterpret_u16(acc_##n), 8);

#define STOR_512(m, o) svst4_vnum(all, hist, 0x##o, v##m)
#endif

    LOAD_512(1, 0);
    LOAD_512(2, 4);

    UNPK_512(1, 0);
    ADDW_512(1);
    STOR_512(1, 0);
    LOAD_512(1, 8);

    UNPK_512(2, 1);
    ADDW_512(2);
    STOR_512(2, 4);
    LOAD_512(2, c);

    UNPK_512(1, 2);
    ADDW_512(1);
    UNPK_512(2, 3);
    ADDW_512(2);

    STOR_512(1, 8);
    STOR_512(2, a);
  }
}

static void NOINLINE histogram_8(uint32_t *hist, uint8_t *data, uint64_t n) {
  svbool_t all = svptrue_b8();
  svuint8_t cap = svdup_u8(191);

  svuint8_t idx_0 = svindex_u8(0, 1);
  IDX(1, 128);

  for (uint8_t *end = data + n; data < end;) {
    ACC(0);
    ACC(1);

    while (1) {
      svuint8_t val_0 = svld1rq(all, data + 0);
      svuint8_t val_1 = svld1rq(all, data + 16);
      svuint8_t val_2 = svld1rq(all, data + 32);
      svuint8_t val_3 = svld1rq(all, data + 48);

#define STEP(i, j) acc_##i = svadd_x(all, acc_##i, svhistseg(idx_##i, val_##j))
      STEP(0, 0);
      STEP(0, 1);
      STEP(0, 2);
      STEP(0, 3);
      STEP(1, 0);
      STEP(1, 1);
      STEP(1, 2);
      STEP(1, 3);
#undef STEP

      data += 64;
      if (data >= end) {
        break;
      } else {
        svbool_t cmp_0 = svcmpge(all, cap, acc_0);
        svbool_t cmp_1 = svcmpge(all, cap, acc_1);

        cmp_0 = svnand_z(all, cmp_0, cmp_1);
        if (svptest_any(cmp_0, cmp_0)) break;
      }
    }

// TODO: same as for 512-bit
#if LD2W
    svuint16_t lo, hi;
    svuint32x2_t v11, v21, v12, v22;

#define LOAD_1024(m, o)                     \
  v1##m = svld2_vnum(all, hist, 0x##o + 0); \
  v2##m = svld2_vnum(all, hist, 0x##o + 2);

#define ADDW_1024(m)                                       \
  v1##m = svset2(v1##m, 0, svaddwb(svget2(v1##m, 0), lo)); \
  v1##m = svset2(v1##m, 1, svaddwt(svget2(v1##m, 1), lo)); \
  v2##m = svset2(v2##m, 0, svaddwb(svget2(v2##m, 0), hi)); \
  v2##m = svset2(v2##m, 1, svaddwt(svget2(v2##m, 1), hi));

#define STOR_1024(m, o)                    \
  svst2_vnum(all, hist, 0x##o + 0, v1##m); \
  svst2_vnum(all, hist, 0x##o + 2, v2##m);

#define UNPK_1024(m, n)   \
  lo = svunpklo(acc_##n); \
  hi = svunpkhi(acc_##n)
#else
    svuint16_t lo1, lo2, hi1, hi2;
    svuint32x4_t v1, v2;

#define LOAD_1024(m, o) v##m = svld4_vnum(all, hist, 0x##o)

#define ADDW_1024(m)                                       \
  v##m = svset4(v##m, 0, svaddwb(svget4(v##m, 0), lo##m)); \
  v##m = svset4(v##m, 2, svaddwt(svget4(v##m, 2), lo##m)); \
  v##m = svset4(v##m, 1, svaddwb(svget4(v##m, 1), hi##m)); \
  v##m = svset4(v##m, 3, svaddwt(svget4(v##m, 3), hi##m));

#define UNPK_1024(m, n)                              \
  lo##m = svextb_x(all, svreinterpret_u16(acc_##n)); \
  hi##m = svlsr_x(all, svreinterpret_u16(acc_##n), 8);

#define STOR_1024(m, o) svst4_vnum(all, hist, 0x##o, v##m)
#endif

    LOAD_1024(1, 0);
    LOAD_1024(2, 4);
    UNPK_1024(1, 0);
    ADDW_1024(1);
    UNPK_1024(2, 1);
    ADDW_1024(2);
    STOR_1024(1, 0);
    STOR_1024(2, 4);
  }
}

#undef IDX
#undef ACC

static void NOINLINE update(uint32_t *histogram, uint8_t *data, int n) {
  switch (8 * svcntb()) {
    case 128:
      histogram_1(histogram, data, n);
      break;
    case 256:
      histogram_2(histogram, data, n);
      break;
    case 512:
      histogram_4(histogram, data, n);
      break;
    case 1024:
      histogram_8(histogram, data, n);
      break;

    default: {
      svbool_t p;
      FOR_LOOP_32(int, i, 0, n, p) {
        svuint32_t idx = svld1ub_u32(p, data + i);
        svuint32_t val = svld1_gather_index(p, histogram, idx);
        svuint32_t cnt = svhistcnt_z(p, idx, idx);
        val = svadd_x(p, val, cnt);
        svst1_scatter_index(p, histogram, idx, val);
      }
      break;
    }
  }
}
#elif defined(__ARM_FEATURE_SVE2) && !defined(HAVE_AUTOVEC)
void histogram_vls(uint32_t *histogram, uint8_t *data, int n);
#define update histogram_vls
#else
void NOINLINE update(uint32_t *restrict histogram, uint8_t *data, int n) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_104(struct loop_104_data *restrict input) {
  uint32_t *histogram = input->histogram;
  uint64_t histogram_size = input->histogram_size;
  uint8_t *data = input->data;
  int n = input->n;

  memset(histogram, 0, histogram_size);
  update(histogram, data, n);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256
#endif

LOOP_DECL(104, NS_SVE_LOOP_ATTR)
{
  struct loop_104_data data = { .data = sample_json };
  size_t json_len = (size_t)sample_json_size;
  data.n = json_len - (json_len % 16);
  data.histogram_size = sizeof(data.histogram[0]) * SIZE;

  ALLOC_64B(data.histogram, SIZE, "histogram");

  inner_loops_104(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < 256; i++) {
    checksum += data.histogram[i] * i;
  }

  bool passed = checksum == 0x000ed612;
#ifndef STANDALONE
  FINALISE_LOOP_I(104, passed, "0x%08"PRIx32, 0x000ed612, checksum)
#endif
  return passed ? 0 : 1;
}
