<!--
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
-->

# List of Loops

| Number | Name | Purpose |
|:------:|:-----|:--------|
| 001 | FP32 inner product | Use of fp32 MLA instruction |
| 002 | UINT32 inner product | Use of u32 MLA instruction |
| 003 | FP64 inner product | Use of fp64 MLA instruction |
| 004 | UINT64 inner product | Use of u64 MLA instruction |
| 005 | strlen short strings | Use of FF and NF loads instructions |
| 006 | strlen long strings | Use of FF and NF loads instructions |
| 008 | Precise fp64 add reduction | Use of FADDA instructions |
| 009 | Pointer chasing | Use of CTERM and BRK instructions |
| 010 | Conditional reduction (fp) | Use of CLAST (SIMD&FP scalar) instructions |
| 012 | Particle motion | Use of FP compare instructions |
| 019 | Mark objects | Use of scatters store instruction |
| 022 | TCP checksum | Use of simd instructions for misaligned accesses |
| 023 | Conjugate Gradient | Use of gathers load instruction |
| 024 | Sum of abs diffs | Use of DOT instruction |
| 025 | FP32 small matrix-matrix multiply | Use of fp32 indexed MLA instruction |
| 026 | Convert UTF-16 to chars | Use of gathers load instruction |
| 027 | FP32 square root | Use of FSQRT instruction |
| 028 | FP64 fast division | Use of FRECPE and FRECPS instructions |
| 029 | FP64 multiply by power of 2 | Use of FSCALE instruction |
| 031 | small-lengths inline memcpy test | Use of simd-based memcpy for small lengths and at varied alignments |
| 032 | FP64 banded linear equations | Use of strided gather and INC instructions |
| 033 | FP64 Inner product | Use of a tight loop with INC and WHILE instructions |
| 034 | Short string compares | Use of FFR for strcmp |
| 035 | Array addition | Use of WHILE for loop control |
| 036 | Sparse matrix Gauss Step | Use of gather-based matrix processing |
| 037 | FP32 complex vector product | Use of fp32 FCMLA instruction |
| 038 | Fp16 convolution | Use of fp16 multiply-add instructions |
| 040 | Clamp operation | Use of simd clamp (or min, max and shift) instructions |
| 101 | Upscale filter | Use of top/bottom instructions |
| 102 | General histogram | Use of HISTCNT instruction |
| 103 | whitespace | Use of MATCH and NMATCH instructions |
| 104 | Byte historgram | Use of HISTSEG instruction |
| 105 | Cascade summation | Use of pairwise FP add instruction |
| 106 | Sheep and goats | Use of BGRP instruction |
| 107 | UINT128 multiply | Use of ADCL[B/T] instructions |
| 108 | Pixel manipulation | Use of LD4 with shift-accumulate instructions |
| 109 | UINT32 complex addition | Use of u32 CADD instruction |
| 110 | UINT32 complex dot | Use of u32 CDOT instruction |
| 111 | FP64 overflow handling | Use of FLOGB and FSCALE instructions |
| 112 | UINT32 complex MAC | Use of CMLA instruction |
| 113 | UINT32 Pairs addition | Use of u32 ADDP |
| 114 | Auto-correlation | Use of shifts, widening mult and load-replicate instructions |
| 120 | Insertion sort | Use of CMPLT instruction |
| 121 | Quicksort | Use of CMPLT with COMPACT and CNTP instructions |
| 122 | Odd-Even transposition sort | Use of CMPLT with SEL instructions |
| 123 | Bitonic mergesort | Use of CMPGT with SEL instructions |
| 124 | Radix sort | Use of simd instructions in radix sort |
| 126 | conditional update | Use of simd loop with conditional update |
| 127 | loop early exit | Use of simd loop with early exit |
| 128 | alias in contiguous access | Use of simd loop with possible alias in contiguous mem access |
| 130 | FP32 matrix-matrix multiply using | Use of fp32 MMLA instructions |
| 135 | INT8-INT32 matrix-matrix multiply using MMLA | Use of i8 to i32 MMLA instructions |
| 136 | INT4-INT32 matrix-matrix multiply using MMLA | Use of 4-bit dequantization (LUT) with i8 to i32 MMLA instructions |
| 137 | BF16-FP32 matrix-matrix multiply using MMLA | Use of bf16 to fp32 MMLA instructions |
| 201 | FP64 matrix-matrix multiply using MOPA / DOT | Use of fp64 MOPA (or MLA) instructions |
| 202 | FP32 matrix-matrix multiply using MOPA / DOT | Use of fp32 MOPA (or MLA) instructions |
| 204 | FP16 matrix-matrix multiply using MOPA / DOT | Use of fp16 to fp16 MOPA (or MLA) instructions |
| 205 | INT8-INT32 matrix-matrix multiply using MOPA / DOT | Use of i8 to i32 MOPA (or DOT) instructions |
| 206 | INT16-INT64 matrix-matrix multiply using MOPA / DOT | Use of i16 to i64 MOPA (or DOT) instructions |
| 207 | INT1-INT32 matrix-matrix multiply using MOPA / DOT | Use of 1-bit MOPA instructions |
| 208 | BF16-BF16 matrix-matrix multiply using MOPA / DOT | Use of bf16 to bf16 MOPA (or MLA) instructions |
| 210 | BF16-FP32 matrix-matrix multiply using MOPA / DOT | Use of bf16 to fp32 MOPA (or DOT) instructions |
| 211 | INT16-INT32 matrix-matrix multiply using MOPA / DOT | Use of i16 to i32 MOPA (or DOT) instructions |
| 212 | 4-bit-FP32 col-major interleaved matrix-vector multiply | Use of 4-bit dequantization (LUT) and DOT instructions |
| 215 | UINT8-UINT32 col-major interleaved matrix-vector multiply | Use of u8 to u32 DOT instruction |
| 216 | FP32 col-major matrix-vector multiply | Use of fp32 MLA instruction |
| 217 | INT8-INT32 row-major matrix-vector multiply | Use of i8 to i32 DOT and ADDV instructions |
| 218 | FP64 col-major matrix-vector multiply | Use of fp64 MLA instruction |
| 219 | INT8-INT32 col-major matrix-vector multiply | Use of i8 to i32 VDOT instruction |
| 220 | FP32 row-major matrix-vector multiply | Use of fp32 MLA and ADDV instructions |
| 221 | FP64 row-major matrix-vector multiply | Use of fp64 MLA and ADDV instructions |
| 222 | FP16 convolution | Use of mutli-vector LD and f16 MLA instructions |
| 223 | Matrix transposition | Use of simd instructions (LD & ST ZA, LD & ZIP) in transposition |
| 231 | BF16-FP32 col-major interleaved matrix-vector multiply | Use of bf16 to fp32 DOT instruction |
| 245 | INT8-INT32 matrix-matrix multiply with rearrangement | Use of MOPA, DOT & MMLA instructions with matrix rearrangements |
