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

#ifdef __ARM_FEATURE_SVE2
#define LD2W 1

#ifdef __APPLE__
// On Darwin public symbols must be prefixed with '_'.
#define SYMBOL(X) _ ## X
#else // __APPLE__
#define SYMBOL(X) X
#endif // __APPLE__

    .text
    .global SYMBOL(histogram_vls)
#ifndef __APPLE__
    // Darwin has not `.type` directive.
    .type SYMBOL(histogram_vls), %function
#endif // __APPLE__

    SYMBOL(histogram_vls):
    // SVE2 implementation for 256-entries histogram
    // VLS implementation (with HISTSEG) for VL128, VL256, VL512, VL1024
    // VLA implementation (with HISTCNT) for other VLs
    // x0: histogram buffer address, histogram entries are 32-bit. There are 256 entries since records are 8-bit
    // x1: records buffer address, records are 8-bit
    // w2: number of records (multiple of 16) = image size

    mov     w2, w2

    rdvl    x7, #1

    cmp     x7, #16
    b.eq    .L_vls128

    cmp     x7, #32
    b.eq    .L_vls256

    cmp     x7, #64
    b.eq    .L_vls512

    cmp     x7, #128
    b.eq    .L_vls1024

.L_vla:
    mov     x4, #0              // loop counter on num_records

    whilelo p1.s, x4, x2
    b.nfrst .L_return_vla

.L_loop_vla:
    ld1b    z1.s, p1/z, [x1, x4]
    ld1w    z2.s, p1/z, [x0, z1.s, UXTW #2]
    histcnt z0.s, p1/z, z1.s, z1.s
    add     z2.s, p1/m, z2.s, z0.s
    st1w    z2.s, p1, [x0, z1.s, UXTW #2]
    incw    x4
    whilelo p1.s, x4, x2
    b.first .L_loop_vla

.L_return_vla:
    ret


.L_vls128:
    ptrue   p0.b
    dup     z7.b, #191     // 255-4*16
    dup     z6.b, #0x7f

    cbz     w2, .L_return_vls128

    // Initialize z8.b-z15.b with all existing histogram entries: 0..127
    index   z8.b, #0, #1                    // 0..15
    movprfx z9, z8
    add     z9.b, z9.b, #16                 // 16..31
    movprfx z10, z8
    add     z10.b, z10.b, #32               // 32..47
    movprfx z11, z8
    add     z11.b, z11.b, #48               // 48..63
    movprfx z12, z8
    add     z12.b, z12.b, #64               // 64..79
    movprfx z13, z8
    add     z13.b, z13.b, #80               // 80..95
    movprfx z14, z8
    add     z14.b, z14.b, #96               // 96..111
    movprfx z15, z8
    add     z15.b, z15.b, #112              // 112..127

    mov     x5, #0                          // loop counter on num_records

.L_loop_outer_vls128:
    // Initialize z16.b-z31.b counters to 0
    mov     z16.b, #0
    mov     z17.b, #0
    mov     z18.b, #0
    mov     z19.b, #0
    mov     z20.b, #0
    mov     z21.b, #0
    mov     z22.b, #0
    mov     z23.b, #0
    mov     z24.b, #0
    mov     z25.b, #0
    mov     z26.b, #0
    mov     z27.b, #0
    mov     z28.b, #0
    mov     z29.b, #0
    mov     z30.b, #0
    mov     z31.b, #0

.L_loop_inner_vls128:
    ld1rqb  z0.b, p0/z, [x1, x5]

    // in order to reuse z8-z15, put in z1 values greater than 127, then
    // clear the top bit and fill the gaps with some value grater than 127
    cmphi   p1.b, p0/z, z0.b, #127
    and     z1.d, z0.d, z6.d
    sel     z1.b, p1, z1.b, z7.b

    add     x5, x5, #16

    histseg z2.b, z8.b, z0.b
    histseg z3.b, z8.b, z1.b
    add     z16.b, z16.b, z2.b
    add     z24.b, z24.b, z3.b
    histseg z4.b, z9.b, z0.b
    histseg z5.b, z9.b, z1.b
    add     z17.b, z17.b, z4.b
    add     z25.b, z25.b, z5.b

    histseg z2.b, z10.b, z0.b
    histseg z3.b, z10.b, z1.b
    add     z18.b, z18.b, z2.b
    add     z26.b, z26.b, z3.b
    histseg z4.b, z11.b, z0.b
    histseg z5.b, z11.b, z1.b
    add     z19.b, z19.b, z4.b
    add     z27.b, z27.b, z5.b

    histseg z2.b, z12.b, z0.b
    histseg z3.b, z12.b, z1.b
    add     z20.b, z20.b, z2.b
    add     z28.b, z28.b, z3.b
    histseg z4.b, z13.b, z0.b
    histseg z5.b, z13.b, z1.b
    add     z21.b, z21.b, z4.b
    add     z29.b, z29.b, z5.b

    histseg z2.b, z14.b, z0.b
    histseg z3.b, z14.b, z1.b
    add     z22.b, z22.b, z2.b
    add     z30.b, z30.b, z3.b
    histseg z4.b, z15.b, z0.b
    histseg z5.b, z15.b, z1.b
    add     z23.b, z23.b, z4.b
    add     z31.b, z31.b, z5.b

    cmp     x5, x2
    b.hs    .L_loop_inner_end_vls128

    and     x6, x5, #63
    cbnz    x6, .L_loop_inner_vls128

    movprfx z0.b, p0/m, z16.b
    umax    z0.b, p0/m, z0.b, z17.b
    movprfx z1.b, p0/m, z18.b
    umax    z1.b, p0/m, z1.b, z19.b
    umax    z0.b, p0/m, z0.b, z20.b
    umax    z1.b, p0/m, z1.b, z21.b
    umax    z0.b, p0/m, z0.b, z22.b
    umax    z1.b, p0/m, z1.b, z23.b
    umax    z1.b, p0/m, z1.b, z24.b
    umax    z1.b, p0/m, z1.b, z25.b
    umax    z1.b, p0/m, z1.b, z26.b
    umax    z1.b, p0/m, z1.b, z27.b
    umax    z1.b, p0/m, z1.b, z28.b
    umax    z1.b, p0/m, z1.b, z29.b
    umax    z1.b, p0/m, z1.b, z30.b
    umax    z1.b, p0/m, z1.b, z31.b
    umax    z0.b, p0/m, z0.b, z1.b
    cmplo   p1.b, p0/z, z7.b, z0.b
    b.none  .L_loop_inner_vls128

.L_loop_inner_end_vls128:
    mov x9, x0


#define WRITE_BACK_128(n, hash, i1, i2)                   \
    ld2w    {z2.s, z3.s}, p0/z, [x9, hash##i1, MUL VL] ;  \
    ld2w    {z4.s, z5.s}, p0/z, [x9, hash##i2, MUL VL] ;  \
    uunpklo z0.h, z##n.b                               ;  \
    uunpkhi z1.h, z##n.b                               ;  \
    uaddwb  z2.s, z2.s, z0.h                           ;  \
    uaddwt  z3.s, z3.s, z0.h                           ;  \
    uaddwb  z4.s, z4.s, z1.h                           ;  \
    uaddwt  z5.s, z5.s, z1.h                           ;  \
    st2w    {z2.s, z3.s}, p0, [x9, hash##i1, MUL VL]   ;  \
    st2w    {z4.s, z5.s}, p0, [x9, hash##i2, MUL VL]

WRITE_BACK_128(16, #, 0, 2)
WRITE_BACK_128(17, #, 4, 6)
WRITE_BACK_128(18, #, 8, 10)
WRITE_BACK_128(19, #, 12, 14)

    incb    x9, ALL, MUL #16
WRITE_BACK_128(20, #, 0, 2)
WRITE_BACK_128(21, #, 4, 6)
WRITE_BACK_128(22, #, 8, 10)
WRITE_BACK_128(23, #, 12, 14)

    incb    x9, ALL, MUL #16
WRITE_BACK_128(24, #, 0, 2)
WRITE_BACK_128(25, #, 4, 6)
WRITE_BACK_128(26, #, 8, 10)
WRITE_BACK_128(27, #, 12, 14)

    incb    x9, ALL, MUL #16
WRITE_BACK_128(28, #, 0, 2)
WRITE_BACK_128(29, #, 4, 6)
WRITE_BACK_128(30, #, 8, 10)
WRITE_BACK_128(31, #, 12, 14)
#undef WRITE_BACK_128

    cmp     x5, x2
    b.lo    .L_loop_outer_vls128

.L_return_vls128:
    ret


.L_vls256:
    ptrue   p0.b
    dup     z29.b, #191     // 255-4*16

    cbz     w2, .L_return_vls256

    // Initialize z11.b-z18.b with all existing histogram entries: 0..255
    index   z11.b, #0, #1                   // z11: 0..31
    movprfx z12, z11
    add     z12.b, z12.b, #32               // z12: 32..63
    movprfx z13, z11
    add     z13.b, z13.b, #64               // z13: 64..95
    movprfx z14, z11
    add     z14.b, z14.b, #96               // z14: 96..127
    movprfx z15, z11
    add     z15.b, z15.b, #128              // z15: 128..159
    movprfx z16, z11
    add     z16.b, z16.b, #160              // z16: 160  191
    movprfx z17, z11
    add     z17.b, z17.b, #192              // z17: 192..223
    movprfx z18, z11
    add     z18.b, z18.b, #224              // z18: 224..255

    mov     x5, #0                          // loop counter on num_records

.L_loop_outer_vls256:
    // Initialize z21.b-z28.b counters to 0
    mov     z21.b, #0       // will contain counters fot histogram entries 0..31
    mov     z22.b, #0
    mov     z23.b, #0
    mov     z24.b, #0
    mov     z25.b, #0
    mov     z26.b, #0
    mov     z27.b, #0
    mov     z28.b, #0

.L_loop_inner_vls256:
    ld1rqb  z0.b, p0/z, [x1, x5]
    add     x5, x5, #16
    histseg z1.b, z11.b, z0.b
    histseg z2.b, z12.b, z0.b
    histseg z3.b, z13.b, z0.b
    histseg z4.b, z14.b, z0.b
    add     z21.b, z21.b, z1.b
    add     z22.b, z22.b, z2.b
    histseg z5.b, z15.b, z0.b
    histseg z6.b, z16.b, z0.b
    histseg z7.b, z17.b, z0.b
    histseg z8.b, z18.b, z0.b
    add     z23.b, z23.b, z3.b
    add     z24.b, z24.b, z4.b
    add     z25.b, z25.b, z5.b
    add     z26.b, z26.b, z6.b
    cmp     x5, x2
    add     z27.b, z27.b, z7.b
    add     z28.b, z28.b, z8.b
    b.hs    .L_loop_inner_end_vls256

    and     x6, x5, #63
    cbnz    x6, .L_loop_inner_vls256

    movprfx z0.b, p0/m, z21.b
    umax    z0.b, p0/m, z0.b, z22.b
    movprfx z1.b, p0/m, z23.b
    umax    z1.b, p0/m, z1.b, z24.b
    umax    z0.b, p0/m, z0.b, z25.b
    umax    z1.b, p0/m, z1.b, z26.b
    umax    z0.b, p0/m, z0.b, z27.b
    umax    z1.b, p0/m, z1.b, z28.b
    umax    z0.b, p0/m, z0.b, z1.b
    cmplo   p1.b, p0/z, z29.b, z0.b
    b.none  .L_loop_inner_vls256

.L_loop_inner_end_vls256:
#if LD2W
    mov x9, x0
    ld2w    {z2.s, z3.s}, p0/z, [x9, #0, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x9, #2, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x9, #4, MUL VL]
    uunpklo z0.h, z21.b
    uunpkhi z1.h, z21.b
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    ld2w    {z8.s, z9.s}, p0/z, [x9, #6, MUL VL]
    uunpklo z0.h, z22.b
    uunpkhi z1.h, z22.b
    st2w    {z2.s, z3.s}, p0, [x9, #0, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x9, #2, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    ld2w    {z2.s, z3.s}, p0/z, [x9, #8, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x9, #10, MUL VL]
    uunpklo z0.h, z23.b
    uunpkhi z1.h, z23.b
    st2w    {z6.s, z7.s}, p0, [x9, #4, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x9, #6, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x9, #12, MUL VL]
    ld2w    {z8.s, z9.s}, p0/z, [x9, #14, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uunpklo z0.h, z24.b
    uunpkhi z1.h, z24.b
    st2w    {z2.s, z3.s}, p0, [x9, #8, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x9, #10, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    st2w    {z6.s, z7.s}, p0, [x9, #12, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x9, #14, MUL VL]

    incb    x9, ALL, MUL #16
    ld2w    {z2.s, z3.s}, p0/z, [x9, #0, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x9, #2, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x9, #4, MUL VL]
    uunpklo z0.h, z25.b
    uunpkhi z1.h, z25.b
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    ld2w    {z8.s, z9.s}, p0/z, [x9, #6, MUL VL]
    uunpklo z0.h, z26.b
    uunpkhi z1.h, z26.b
    st2w    {z2.s, z3.s}, p0, [x9, #0, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x9, #2, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    ld2w    {z2.s, z3.s}, p0/z, [x9, #8, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x9, #10, MUL VL]
    uunpklo z0.h, z27.b
    uunpkhi z1.h, z27.b
    st2w    {z6.s, z7.s}, p0, [x9, #4, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x9, #6, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x9, #12, MUL VL]
    ld2w    {z8.s, z9.s}, p0/z, [x9, #14, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uunpklo z0.h, z28.b
    uunpkhi z1.h, z28.b
    st2w    {z2.s, z3.s}, p0, [x9, #8, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x9, #10, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    st2w    {z6.s, z7.s}, p0, [x9, #12, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x9, #14, MUL VL]
#else
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #0, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #4, MUL VL]
    uxtb    z0.h, p0/m, z21.h
    lsr     z1.h, z21.h, #8
    uxtb    z10.h, p0/m, z22.h
    lsr     z20.h, z22.h, #8
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z4.s, z4.s, z0.h
    uaddwb  z3.s, z3.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uaddwb  z6.s, z6.s, z10.h
    uaddwt  z8.s, z8.s, z10.h
    uaddwb  z7.s, z7.s, z20.h
    uaddwt  z9.s, z9.s, z20.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #0, MUL VL]
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #8, MUL VL]
    uxtb    z0.h, p0/m, z23.h
    lsr     z1.h, z23.h, #8
    uxtb    z10.h, p0/m, z24.h
    lsr     z20.h, z24.h, #8
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #4, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #12, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z4.s, z4.s, z0.h
    uaddwb  z3.s, z3.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uaddwb  z6.s, z6.s, z10.h
    uaddwt  z8.s, z8.s, z10.h
    uaddwb  z7.s, z7.s, z20.h
    uaddwt  z9.s, z9.s, z20.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #8, MUL VL]
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #16, MUL VL]
    uxtb    z0.h, p0/m, z25.h
    lsr     z1.h, z25.h, #8
    uxtb    z10.h, p0/m, z26.h
    lsr     z20.h, z26.h, #8
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #12, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #20, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z4.s, z4.s, z0.h
    uaddwb  z3.s, z3.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uaddwb  z6.s, z6.s, z10.h
    uaddwt  z8.s, z8.s, z10.h
    uaddwb  z7.s, z7.s, z20.h
    uaddwt  z9.s, z9.s, z20.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #16, MUL VL]
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #24, MUL VL]
    uxtb    z0.h, p0/m, z27.h
    lsr     z1.h, z27.h, #8
    uxtb    z10.h, p0/m, z28.h
    lsr     z20.h, z28.h, #8
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #20, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #28, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z4.s, z4.s, z0.h
    uaddwb  z3.s, z3.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uaddwb  z6.s, z6.s, z10.h
    uaddwt  z8.s, z8.s, z10.h
    uaddwb  z7.s, z7.s, z20.h
    uaddwt  z9.s, z9.s, z20.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #24, MUL VL]
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #28, MUL VL]
#endif
    cmp     x5, x2
    b.lo    .L_loop_outer_vls256

.L_return_vls256:
    ret


.L_vls512:
    ptrue   p0.b
    dup     z29.b, #191     // 255-4*16

    cbz     w2, .L_return_vls512

    // Initialize z10.b-z13.b with all existing histogram entries: 0..255
    index   z10.b, #0, #1                   // z10: 0..63
    movprfx z11, z10
    add     z11.b, z11.b, #64               // z11: 64..127
    movprfx z12, z10
    add     z12.b, z12.b, #128              // z12: 128..191
    movprfx z13, z10
    add     z13.b, z13.b, #192              // z13: 192..255

    add     x6, x1, x2

.L_loop_outer_vls512:
    // Initialize z20.b-z23.b counters to 0
    mov     z20.b, #0       // will contain counters fot histogram entries 0..63
    mov     z21.b, #0
    mov     z22.b, #0
    mov     z23.b, #0

.L_loop_inner_vls512:
    ld1rqb  z0.b, p0/z, [x1, #0]
    ld1rqb  z1.b, p0/z, [x1, #16]
    ld1rqb  z2.b, p0/z, [x1, #32]
    ld1rqb  z3.b, p0/z, [x1, #48]

    histseg z14.b, z10.b, z0.b
    histseg z15.b, z11.b, z0.b
    histseg z16.b, z12.b, z0.b
    histseg z17.b, z13.b, z0.b

    histseg z18.b, z10.b, z1.b
    histseg z19.b, z11.b, z1.b
    histseg z4.b,  z12.b, z1.b
    histseg z5.b,  z13.b, z1.b

    add     x1, x1, #64
    histseg z6.b,  z10.b, z2.b
    histseg z7.b,  z11.b, z2.b
    histseg z8.b,  z12.b, z2.b
    histseg z9.b,  z13.b, z2.b

    histseg z24.b, z10.b, z3.b
    histseg z25.b, z11.b, z3.b
    histseg z26.b, z12.b, z3.b
    histseg z27.b, z13.b, z3.b

    add     z20.b, z20.b, z14.b
    add     z21.b, z21.b, z15.b
    add     z22.b, z22.b, z16.b
    add     z23.b, z23.b, z17.b

    cmp     x1, x6
    add     z20.b, z20.b, z18.b
    add     z21.b, z21.b, z19.b
    add     z22.b, z22.b, z4.b
    add     z23.b, z23.b, z5.b

    add     z20.b, z20.b, z6.b
    add     z21.b, z21.b, z7.b
    add     z22.b, z22.b, z8.b
    add     z23.b, z23.b, z9.b

    add     z20.b, z20.b, z24.b
    add     z21.b, z21.b, z25.b
    add     z22.b, z22.b, z26.b
    add     z23.b, z23.b, z27.b

    b.hs    .L_loop_inner_end_vls512

    cmphs   p1.b, p0/z, z29.b, z20.b
    cmphs   p2.b, p0/z, z29.b, z21.b
    cmphs   p3.b, p0/z, z29.b, z22.b
    cmphs   p4.b, p0/z, z29.b, z23.b
    and     p1.b, p1/z, p2.b, p3.b  // p1 = p1 & p2 & p3
    nands   p4.b, p0/z, p1.b, p4.b  // p4 = ~(p0 & p1 & p4)
    b.none  .L_loop_inner_vls512

.L_loop_inner_end_vls512:
#if LD2W
    ld2w    {z2.s, z3.s}, p0/z, [x0, #0, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x0, #2, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x0, #4, MUL VL]
    uunpklo z0.h, z20.b
    uunpkhi z1.h, z20.b
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    ld2w    {z8.s, z9.s}, p0/z, [x0, #6, MUL VL]
    uunpklo z0.h, z21.b
    uunpkhi z1.h, z21.b
    st2w    {z2.s, z3.s}, p0, [x0, #0, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x0, #2, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    ld2w    {z2.s, z3.s}, p0/z, [x0, #8, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x0, #10, MUL VL]
    uunpklo z0.h, z22.b
    uunpkhi z1.h, z22.b
    st2w    {z6.s, z7.s}, p0, [x0, #4, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x0, #6, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x0, #12, MUL VL]
    ld2w    {z8.s, z9.s}, p0/z, [x0, #14, MUL VL]
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    uunpklo z0.h, z23.b
    uunpkhi z1.h, z23.b
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    st2w    {z2.s, z3.s}, p0, [x0, #8, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x0, #10, MUL VL]
    st2w    {z6.s, z7.s}, p0, [x0, #12, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x0, #14, MUL VL]
#else
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #0, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #4, MUL VL]
    uxtb    z14.h, p0/m, z20.h
    lsr     z15.h, z20.h, #8
    uxtb    z16.h, p0/m, z21.h
    lsr     z17.h, z21.h, #8
    uaddwb  z2.s, z2.s, z14.h
    uaddwt  z4.s, z4.s, z14.h
    uaddwb  z3.s, z3.s, z15.h
    uaddwt  z5.s, z5.s, z15.h
    uaddwb  z6.s, z6.s, z16.h
    uaddwt  z8.s, z8.s, z16.h
    uaddwb  z7.s, z7.s, z17.h
    uaddwt  z9.s, z9.s, z17.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #0, MUL VL]
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #8, MUL VL]
    uxtb    z14.h, p0/m, z22.h
    lsr     z15.h, z22.h, #8
    uxtb    z16.h, p0/m, z23.h
    lsr     z17.h, z23.h, #8
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #4, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #12, MUL VL]
    uaddwb  z2.s, z2.s, z14.h
    uaddwt  z4.s, z4.s, z14.h
    uaddwb  z3.s, z3.s, z15.h
    uaddwt  z5.s, z5.s, z15.h
    uaddwb  z6.s, z6.s, z16.h
    uaddwt  z8.s, z8.s, z16.h
    uaddwb  z7.s, z7.s, z17.h
    uaddwt  z9.s, z9.s, z17.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #8, MUL VL]
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #12, MUL VL]
#endif
    cmp     x1, x6
    b.lo    .L_loop_outer_vls512

.L_return_vls512:
    ret



.L_vls1024:
    ptrue   p0.b
    dup     z29.b, #191     // 255-4*16

    cbz     w2, .L_return_vls1024

    // Initialize z10.b-z11.b with all existing histogram entries: 0..255
    index   z10.b, #0, #1                   // z10: 0..127
    movprfx z11, z10
    add     z11.b, z11.b, #128               // z11: 128..255

    add     x6, x1, x2

.L_loop_outer_vls1024:
    // Initialize z20.b-z21.b counters to 0
    mov     z20.b, #0       // will contain counters fot histogram entries 0..127
    mov     z21.b, #0

.L_loop_inner_vls1024:
    ld1rqb  z0.b, p0/z, [x1, #0]
    ld1rqb  z1.b, p0/z, [x1, #16]
    ld1rqb  z2.b, p0/z, [x1, #32]
    ld1rqb  z3.b, p0/z, [x1, #48]

    histseg z12.b, z10.b, z0.b
    histseg z13.b, z11.b, z0.b
    histseg z14.b, z10.b, z1.b
    histseg z15.b, z11.b, z1.b
    add     x1, x1, #64
    histseg z16.b, z10.b, z2.b
    histseg z17.b, z11.b, z2.b
    histseg z18.b, z10.b, z3.b
    histseg z19.b, z11.b, z3.b

    cmp     x1, x6
    add     z20.b, z20.b, z12.b
    add     z21.b, z21.b, z13.b
    add     z20.b, z20.b, z14.b
    add     z21.b, z21.b, z15.b
    add     z20.b, z20.b, z16.b
    add     z21.b, z21.b, z17.b
    add     z20.b, z20.b, z18.b
    add     z21.b, z21.b, z19.b
    b.hs    .L_loop_inner_end_vls1024

    cmphs   p1.b, p0/z, z29.b, z20.b
    cmphs   p2.b, p0/z, z29.b, z21.b
    nands   p4.b, p0/z, p1.b, p2.b  // p4 = ~(p0 & p1 & p2)
    b.none  .L_loop_inner_vls1024

.L_loop_inner_end_vls1024:
#if LD2W
    ld2w    {z2.s, z3.s}, p0/z, [x0, #0, MUL VL]
    ld2w    {z4.s, z5.s}, p0/z, [x0, #2, MUL VL]
    ld2w    {z6.s, z7.s}, p0/z, [x0, #4, MUL VL]
    uunpklo z0.h, z20.b
    uunpkhi z1.h, z20.b
    uaddwb  z2.s, z2.s, z0.h
    uaddwt  z3.s, z3.s, z0.h
    uaddwb  z4.s, z4.s, z1.h
    uaddwt  z5.s, z5.s, z1.h
    ld2w    {z8.s, z9.s}, p0/z, [x0, #6, MUL VL]
    uunpklo z0.h, z21.b
    uunpkhi z1.h, z21.b
    st2w    {z2.s, z3.s}, p0, [x0, #0, MUL VL]
    st2w    {z4.s, z5.s}, p0, [x0, #2, MUL VL]
    uaddwb  z6.s, z6.s, z0.h
    uaddwt  z7.s, z7.s, z0.h
    uaddwb  z8.s, z8.s, z1.h
    uaddwt  z9.s, z9.s, z1.h
    st2w    {z6.s, z7.s}, p0, [x0, #4, MUL VL]
    st2w    {z8.s, z9.s}, p0, [x0, #6, MUL VL]
#else
    ld4w    {z2.s, z3.s, z4.s, z5.s}, p0/z, [x0, #0, MUL VL]
    ld4w    {z6.s, z7.s, z8.s, z9.s}, p0/z, [x0, #4, MUL VL]
    uxtb    z12.h, p0/m, z20.h
    lsr     z13.h, z20.h, #8
    uxtb    z14.h, p0/m, z21.h
    lsr     z15.h, z21.h, #8
    uaddwb  z2.s, z2.s, z12.h
    uaddwt  z4.s, z4.s, z12.h
    uaddwb  z3.s, z3.s, z13.h
    uaddwt  z5.s, z5.s, z13.h
    uaddwb  z6.s, z6.s, z14.h
    uaddwt  z8.s, z8.s, z14.h
    uaddwb  z7.s, z7.s, z15.h
    uaddwt  z9.s, z9.s, z15.h
    st4w    {z2.s, z3.s, z4.s, z5.s}, p0, [x0, #0, MUL VL]
    st4w    {z6.s, z7.s, z8.s, z9.s}, p0, [x0, #4, MUL VL]
#endif
    cmp     x1, x6
    b.lo    .L_loop_outer_vls1024

.L_return_vls1024:
    ret

#ifndef __APPLE__
    // Darwin has no `.size` directive.
    .size   SYMBOL(histogram_vls), .-SYMBOL(histogram_vls)
#endif // __APPLE__

#endif  // __ARM_FEATURE_SVE2
