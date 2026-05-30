/*----------------------------------------------------------------------------
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

// LOOP (NUMBER, NAME, PURPOSE, STREAMING ATTR)
//     STREAMING ATTR = STREAMING_COMPATIBLE, for SVE streaming compatible
//                        or STREAMING for SME2 loops

LOOP(001, "FP32 inner product",                "Use of fp32 MLA instruction", STREAMING_COMPATIBLE)
LOOP(002, "UINT32 inner product",              "Use of u32 MLA instruction", STREAMING_COMPATIBLE)
LOOP(003, "FP64 inner product",                "Use of fp64 MLA instruction", STREAMING_COMPATIBLE)
LOOP(004, "UINT64 inner product",              "Use of u64 MLA instruction", STREAMING_COMPATIBLE)
LOOP(005, "strlen short strings",              "Use of FF and NF loads instructions")
LOOP(006, "strlen long strings",               "Use of FF and NF loads instructions")
LOOP(008, "Precise fp64 add reduction",        "Use of FADDA instructions")
LOOP(009, "Pointer chasing",                   "Use of CTERM and BRK instructions")
LOOP(010, "Conditional reduction (fp)",        "Use of CLAST (SIMD&FP scalar) instructions", STREAMING_COMPATIBLE)
LOOP(012, "Particle motion",                   "Use of FP compare instructions", STREAMING_COMPATIBLE)
LOOP(019, "Mark objects",                      "Use of scatters store instruction")
LOOP(022, "TCP checksum",                      "Use of simd instructions for misaligned accesses", STREAMING_COMPATIBLE)
LOOP(023, "Conjugate Gradient",                "Use of gathers load instruction")
LOOP(024, "Sum of abs diffs",                  "Use of DOT instruction", STREAMING_COMPATIBLE)
LOOP(025, "FP32 small matrix-matrix multiply", "Use of fp32 indexed MLA instruction")
LOOP(026, "Convert UTF-16 to chars",           "Use of gathers load instruction")
LOOP(027, "FP32 square root",                  "Use of FSQRT instruction", STREAMING_COMPATIBLE)
LOOP(028, "FP64 fast division",                "Use of FRECPE and FRECPS instructions", STREAMING_COMPATIBLE)
LOOP(029, "FP64 multiply by power of 2",       "Use of FSCALE instruction", STREAMING_COMPATIBLE)
LOOP(031, "small-lengths inline memcpy test",  "Use of simd-based memcpy for small lengths and at varied alignments", STREAMING_COMPATIBLE)
LOOP(032, "FP64 banded linear equations",      "Use of strided gather and INC instructions")
LOOP(033, "FP64 Inner product",                "Use of a tight loop with INC and WHILE instructions", STREAMING_COMPATIBLE)
LOOP(034, "Short string compares",             "Use of FFR for strcmp")
LOOP(035, "Array addition",                    "Use of WHILE for loop control", STREAMING_COMPATIBLE)
LOOP(036, "Sparse matrix Gauss Step",          "Use of gather-based matrix processing")
LOOP(037, "FP32 complex vector product",       "Use of fp32 FCMLA instruction", STREAMING_COMPATIBLE)
LOOP(038, "Fp16 convolution",                  "Use of fp16 multiply-add instructions", STREAMING_COMPATIBLE)
LOOP(040, "Clamp operation",                   "Use of simd clamp (or min, max and shift) instructions", STREAMING_COMPATIBLE)

// SVE2 features
LOOP(101, "Upscale filter",                                  "Use of top/bottom instructions", STREAMING_COMPATIBLE)
LOOP(102, "General histogram",                               "Use of HISTCNT instruction")
LOOP(103, "whitespace",                                      "Use of MATCH and NMATCH instructions")
LOOP(104, "Byte historgram",                                 "Use of HISTSEG instruction")
LOOP(105, "Cascade summation",                               "Use of pairwise FP add instruction", STREAMING_COMPATIBLE)
LOOP(106, "Sheep and goats",                                 "Use of BGRP instruction")
LOOP(107, "UINT128 multiply",                                "Use of ADCL[B/T] instructions", STREAMING_COMPATIBLE)
LOOP(108, "Pixel manipulation",                              "Use of LD4 with shift-accumulate instructions", STREAMING_COMPATIBLE)
LOOP(109, "UINT32 complex addition",                         "Use of u32 CADD instruction", STREAMING_COMPATIBLE)
LOOP(110, "UINT32 complex dot",                              "Use of u32 CDOT instruction", STREAMING_COMPATIBLE)
LOOP(111, "FP64 overflow handling",                          "Use of FLOGB and FSCALE instructions", STREAMING_COMPATIBLE)
LOOP(112, "UINT32 complex MAC",                              "Use of CMLA instruction", STREAMING_COMPATIBLE)
LOOP(113, "UINT32 Pairs addition",                           "Use of u32 ADDP", STREAMING_COMPATIBLE)
LOOP(114, "Auto-correlation",                                "Use of shifts, widening mult and load-replicate instructions", STREAMING_COMPATIBLE)
LOOP(120, "Insertion sort",                                  "Use of CMPLT instruction")
LOOP(121, "Quicksort",                                       "Use of CMPLT with COMPACT and CNTP instructions")
LOOP(122, "Odd-Even transposition sort",                     "Use of CMPLT with SEL instructions")
LOOP(123, "Bitonic mergesort",                               "Use of CMPGT with SEL instructions")
LOOP(124, "Radix sort",                                      "Use of simd instructions in radix sort")
LOOP(126, "conditional update",                              "Use of simd loop with conditional update", STREAMING_COMPATIBLE)
LOOP(127, "loop early exit",                                 "Use of simd loop with early exit", STREAMING_COMPATIBLE)
LOOP(128, "alias in contiguous access",                      "Use of simd loop with possible alias in contiguous mem access", STREAMING_COMPATIBLE)
LOOP(130, "FP32 matrix-matrix multiply using",               "Use of fp32 MMLA instructions")
LOOP(135, "INT8-INT32 matrix-matrix multiply using MMLA",    "Use of i8 to i32 MMLA instructions")
LOOP(136, "INT4-INT32 matrix-matrix multiply using MMLA",    "Use of 4-bit dequantization (LUT) with i8 to i32 MMLA instructions")
LOOP(137, "BF16-FP32 matrix-matrix multiply using MMLA",     "Use of bf16 to fp32 MMLA instructions")

// SME2 features
LOOP(201, "FP64 matrix-matrix multiply using MOPA / DOT",                "Use of fp64 MOPA (or MLA) instructions", STREAMING)
LOOP(202, "FP32 matrix-matrix multiply using MOPA / DOT",                "Use of fp32 MOPA (or MLA) instructions", STREAMING)
LOOP(204, "FP16 matrix-matrix multiply using MOPA / DOT",                "Use of fp16 to fp16 MOPA (or MLA) instructions", STREAMING)
LOOP(205, "INT8-INT32 matrix-matrix multiply using MOPA / DOT",          "Use of i8 to i32 MOPA (or DOT) instructions", STREAMING)
LOOP(206, "INT16-INT64 matrix-matrix multiply using MOPA / DOT",         "Use of i16 to i64 MOPA (or DOT) instructions", STREAMING)
LOOP(207, "INT1-INT32 matrix-matrix multiply using MOPA / DOT",          "Use of 1-bit MOPA instructions", STREAMING)
LOOP(208, "BF16-BF16 matrix-matrix multiply using MOPA / DOT",           "Use of bf16 to bf16 MOPA (or MLA) instructions", STREAMING)
LOOP(210, "BF16-FP32 matrix-matrix multiply using MOPA / DOT",           "Use of bf16 to fp32 MOPA (or DOT) instructions", STREAMING)
LOOP(211, "INT16-INT32 matrix-matrix multiply using MOPA / DOT",         "Use of i16 to i32 MOPA (or DOT) instructions", STREAMING)
LOOP(212, "4-bit-FP32 col-major interleaved matrix-vector multiply",     "Use of 4-bit dequantization (LUT) and DOT instructions", STREAMING)
LOOP(215, "UINT8-UINT32 col-major interleaved matrix-vector multiply",   "Use of u8 to u32 DOT instruction", STREAMING)
LOOP(216, "FP32 col-major matrix-vector multiply",                       "Use of fp32 MLA instruction", STREAMING)
LOOP(217, "INT8-INT32 row-major matrix-vector multiply",                 "Use of i8 to i32 DOT and ADDV instructions", STREAMING)
LOOP(218, "FP64 col-major matrix-vector multiply",                       "Use of fp64 MLA instruction", STREAMING)
LOOP(219, "INT8-INT32 col-major matrix-vector multiply",                 "Use of i8 to i32 VDOT instruction", STREAMING)
LOOP(220, "FP32 row-major matrix-vector multiply",                       "Use of fp32 MLA and ADDV instructions", STREAMING)
LOOP(221, "FP64 row-major matrix-vector multiply",                       "Use of fp64 MLA and ADDV instructions", STREAMING)
LOOP(222, "FP16 convolution",                                            "Use of mutli-vector LD and f16 MLA instructions", STREAMING)
LOOP(223, "Matrix transposition",                                        "Use of simd instructions (LD & ST ZA, LD & ZIP) in transposition", STREAMING)
LOOP(231, "BF16-FP32 col-major interleaved matrix-vector multiply",      "Use of bf16 to fp32 DOT instruction")
LOOP(245, "INT8-INT32 matrix-matrix multiply with rearrangement",        "Use of MOPA, DOT & MMLA instructions with matrix rearrangements", STREAMING)
