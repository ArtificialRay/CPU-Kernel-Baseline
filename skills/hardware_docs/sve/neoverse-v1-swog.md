# Arm Neoverse V1 (Graviton3) — Software Optimization Guide (Issue 7.0, r1p2)

> ⚠️ This is **Neoverse V1 = Graviton3**, NOT the V2/Graviton4 core the benchmark
> runs on by default. Use this ONLY when the target hardware/ISA is Graviton3.
> For Graviton4 (Neoverse V2, the usual target) use references/neoverse-v2-swog.md instead.
> Per-instruction latency/throughput/pipeline tables live in §3. Source: Arm doc 110659.

Arm® Neoverse™ V1
Revision: r1p2

Software Optimization Guide
All rights reserved.

Arm® Neoverse™ V1
Software Optimization Guide

Release information

Document history
Issue            Date                   Confidentiality          Change
1.0              25 March 2019          Confidential             First release for r0p0
2.0              3 December 2019        Confidential             First release for r1p0
3.0              29 May 2020            Confidential             First release for r1p1
4.0              13 November 2020       Non-Confidential         Second release for r1p1
5.0              14 April 2021          Non-Confidential         Third release for r1p1
6.0              15 July 2022           Non-Confidential         First release for r1p2
7.0              11 July 2025           Non-Confidential         Second release for r1p2

Non-Confidential Proprietary Notice
This document is protected by copyright and other related rights and the practice or implementation of the
information contained in this document may be protected by one or more patents or pending patent
applications. No part of this document may be reproduced in any form by any means without the express prior
written permission of Arm. No license, express or implied, by estoppel or otherwise to any intellectual property
rights is granted by this document unless specifically stated.

Your access to the information in this document is conditional upon your acceptance that you will not use or
permit others to use the information for the purposes of determining whether implementations infringe any
third party patents.

THIS DOCUMENT IS PROVIDED “AS IS”. ARM PROVIDES NO REPRESENTATIONS AND NO WARRANTIES,
EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
OF MERCHANTABILITY, SATISFACTORY QUALITY, NON-INFRINGEMENT OR FITNESS FOR A
PARTICULAR PURPOSE WITH RESPECT TO THE DOCUMENT. For the avoidance of doubt, Arm makes no
representation with respect to, has undertaken no analysis to identify or understand the scope and content of,
patents, copyrights, trade secrets, or other rights.

This document may include technical inaccuracies or typographical errors.

TO THE EXTENT NOT PROHIBITED BY LAW, IN NO EVENT WILL ARM BE LIABLE FOR ANY DAMAGES,
INCLUDING WITHOUT LIMITATION ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, PUNITIVE, OR
CONSEQUENTIAL DAMAGES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY,
ARISING OUT OF ANY USE OF THIS DOCUMENT, EVEN IF ARM HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

This document consists solely of commercial items. You shall be responsible for ensuring that any use,
duplication or disclosure of this document complies fully with any relevant export laws and regulations to assure
that this document or any portion thereof is not exported, directly or indirectly, in violation of such export laws.
Use of the word “partner” in reference to Arm's customers is not intended to create or refer to any partnership
relationship with any other company. Arm may make changes to this document at any time and without notice.

This document may be translated into other languages for convenience, and you agree that if there is any
conflict between the English version of this document and any translation, the terms of the English version of
the Agreement shall prevail.

The Arm corporate logo and words marked with ® or ™ are registered trademarks or trademarks of Arm Limited
(or its affiliates) in the US and/or elsewhere. All rights reserved. Other brands and names mentioned in this
document may be the trademarks of their respective owners. Please follow Arm's trademark usage guidelines at
https://www.arm.com/company/policies/trademarks.

Arm Limited. Company 02557590 registered in England.

110 Fulbourn Road, Cambridge, England CB1 9NJ.

(LES-PRE-20349)

Confidentiality Status
This document is Non-Confidential. The right to use, copy and disclose this document may be subject to license
restrictions in accordance with the terms of the agreement entered into by Arm and the party that Arm
delivered this document to.

Unrestricted Access is an Arm internal classification.

Product Status
The information in this document is Final, that is for a developed product.

Web Address
developer.arm.com

Progressive terminology commitment
Arm values inclusive communities. Arm recognizes that we and our industry have used terms that can be
offensive. Arm strives to lead the industry and create change.

This document includes terms that can be offensive. We will replace these terms in a future issue of this
document. If you find offensive terms in this document, please email terms@arm.com.

Contents

                                                                                                      1 Introduction

1 Introduction
1.1 Product revision status
The rxpy identifier indicates the revision status of the product described in this book, for example,
r1p2, where:
rx
                Identifies the major revision of the product, for example, r1.
py
                Identifies the minor revision or modification status of the product, for example, p2.

1.2 Intended audience
This document is for system designers, system integrators, and programmers who are designing or
programming a System-on-Chip (SoC) that uses an Arm core.

1.3 Scope
This document describes aspects of the Neoverse™ V1 core micro-architecture that influence
software performance. Micro-architectural detail is limited to that which is useful for software
optimization.

Documentation extends only to software visible behavior of the Neoverse V1 core and not to the
hardware rationale behind the behavior.

1.4 Conventions
The following subsections describe conventions used in Arm documents.

1.4.1 Glossary
The Arm Glossary is a list of terms used in Arm documentation, together with definitions for those
terms. The Arm Glossary does not contain terms that are industry standard unless the Arm meaning
differs from the generally accepted meaning.

See the Arm Glossary for more information: https://developer.arm.com/glossary.

                                                                                            1 Introduction

1.4.1.1 Terms and Abbreviations
This document uses the following terms and abbreviations.
Term              Meaning
ALU               Arithmetic and Logical Unit
ASIMD             Advanced SIMD
MOP               Macro-OPeration
µOP               Micro-OPeration
SQRT              Square Root
T32               AArch32 Thumb® instruction set
FP                Floating-point

                                                                                                   1 Introduction

1.4.2 Typographical conventions
Convention       Use
italic           Introduces citations.
bold             Highlights interface elements, such as menu names. Denotes signal names. Also used for
                 terms in descriptive lists, where appropriate.
monospace        Denotes text that you can enter at the keyboard, such as commands, file and program
                 names, and source code.
monospace bold   Denotes language keywords when used outside example code.
monospace        Denotes a permitted abbreviation for a command or option. You can enter the underlined
underline        text instead of the full command or option name.
<and>            Encloses replaceable terms for assembler syntax where they appear in code or code
                 fragments.
                 For example:
                  MRC p15, 0, <Rd>, <CRn>, <CRm>, <Opcode_2>

SMALL CAPITALS   Used in body text for a few terms that have specific technical meanings, that are defined in
                 the Arm® Glossary. For example, IMPLEMENTATION DEFINED, IMPLEMENTATION SPECIFIC,
                 UNKNOWN, and UNPREDICTABLE.

                 This represents a recommendation which, if not followed, might lead to system failure or
                 damage.

                 This represents a requirement for the system that, if not followed, might result in system
                 failure or damage.

                 This represents a requirement for the system that, if not followed, will result in system
                 failure or damage.

                 This represents an important piece of information that needs your attention.

                 This represents a useful tip that might make it easier, better or faster to perform a task.

                 This is a reminder of something important that relates to the information you are reading.

                                                                                                1 Introduction

1.5 Additional reading
This document contains information that is specific to this product. See the following documents for
other relevant information:

Table 1-1 Arm publications
Document name                                        Document ID             Licensee only
Arm® Architecture Reference Manual, Armv8, for Armv8- DDI 0487               No
A architecture profile
Arm® Neoverse™ V1 Core Technical Reference Manual    101427                  No

                                                                                             1 Introduction

1.6 Feedback
Arm welcomes feedback on this product and its documentation.

1.6.1 Feedback on this product
If you have any comments or suggestions about this product, contact your supplier and give:
•   The product name.
•   The product revision or version.
•   An explanation with as much information as you can provide. Include symptoms and diagnostic
    procedures if appropriate.

1.6.2 Feedback on content
If you have comments on content, send an email to errata@arm.com and give:
•   The title Arm® Neoverse™ V1 Software Optimization Guide.
•   If applicable, the page number(s) to which your comments refer.
•   A concise explanation of your comments.
Arm also welcomes general suggestions for additions and improvements.

Arm tests the PDF only in Adobe Acrobat and Acrobat Reader and cannot guarantee the quality of
the represented document when used with any other PDF reader.

                                                                                                2 Overview

2 Overview
The Neoverse V1 core is a high-performance and low-power Arm product that implements the
Armv8-A architecture.

The Neoverse V1 core supports:
•   Full implementation of the Armv8.4-A A64, A32, and T32 instruction sets excluding the following
    features:
    o   Armv8.4-A secure EL2
    o   Armv8.4-A small page table support
    o   Armv8.4-A Outer Shareable TMO/TMO-by-range
•   AArch32 execution state at Exception level EL0 only. AArch64 execution state at all Exception
    levels (EL0 to EL3).
•   The traps for EL1 and EL0 cache controls, PSTATE SSBS (Speculative Store Bypass Safe) bit that
    supports software mitigation for Spectre Variant 4, and the speculation barriers (CSDB, SSBB,
    PSSBB) instructions introduced in the Armv8.5-A extension.
•   Cache clean to Point of Deep Persistence introduced in the Armv8.5-A extension.
•   The random number instructions introduced in the Armv8.5-A extension.
•   The Enhanced Pointer Authentication, excluding the optional FPAC extension introduced in the
    Armv8.6-A extension.
•   The BFloat16 extension introduced in the Armv8.6-A extension.
•   The Int8 matrix multiply instructions introduced in the Armv8.6-A extension.
•   The Data Gathering Hint instruction introduced in the Armv8.6-A extension.
•   Support for Arm TrustZone® technology.
•   Support for Statistical Profiling Extension (SPE).
•   Support for Scalable Vector Extension (SVE) with 256-bit vector length.

This document describes elements of the Neoverse V1 core micro-architecture that influence
software performance so that software and compilers can be optimized accordingly.

                                                                                               2 Overview

2.1 Pipeline overview
The following figure describes the high-level Neoverse V1 instruction processing pipeline.
Instructions are first fetched and then decoded into internal Macro-OPerations (MOPs). From there,
the MOPs proceed through register renaming and dispatch stages. A MOP can be split into two
Micro-OPerations (µOPs) further down the pipeline after the decode stage. Once dispatched, µOPs
wait for their operands and issue out-of-order to one of fifteen issue pipelines. Each issue pipeline
can accept one µOP per cycle.

                                                                                                   2 Overview

Figure 2-1 Neoverse V1 core pipeline

                                                     Branch 0

                                                     Branch 1

                                                     Integer Single-Cycle 0

                     Decode,                         Integer Single-Cycle 1
    Fetch            Rename,
                     Dispatch
                                                     Integer Single /Multi-Cycle 0

                                                     Integer Single /Multi-Cycle 1

                                                     FP/ASIMD 0
                                                                              FP/ASIMD 0
                                           Issue
                                                     FP/ASIMD 1

                                                     FP/ASIMD 2

                                                     FP/ASIMD 3

                                                     Load/Store 0

                                                     Load/Store 1

                                                     Load 2

                                                     Store data 0

                                                     Store data 1

              IN ORDER                                              OUT OF ORDER

                                                                                                     2 Overview

The execution pipelines support different types of operations, as follows:

Table 2-1 Neoverse V1 core operations
Instruction groups             Instructions
Branch 0/1                     Branch µOPs
Integer Single-Cycle 0/1       Integer ALU µOPs
Integer Single/Multi-cycle 0/1 Integer shift-ALU, multiply, divide, CRC and sum-of-absolute-differences µOPs
Load/Store 0/1                 Load, Store address generation and special memory µOPs
Load 2                         Load µOPs
Store data 0/1                 Store data µOPs
FP/ASIMD-0                     ASIMD ALU, ASIMD misc, ASIMD integer multiply, FP convert, FP misc, FP add, FP
                               multiply, FP divide, FP sqrt, crypto µOPs, store data µOPs
FP/ASIMD-1                     ASIMD ALU, ASIMD misc, FP misc, FP add, FP multiply, ASIMD shift µOPs, store data
                               µOPs, crypto µOPs.
FP/ASIMD-2                     ASIMD ALU, ASIMD misc, ASIMD integer multiply, FP convert, FP misc, FP add, FP
                               multiply, FP divide, FP sqrt, crypto µOPs.
FP/ASIMD-3                     ASIMD ALU, ASIMD misc, FP misc, FP add, FP multiply, ASIMD shift µOPs, crypto µOPs

                                                                                        3 Instruction characteristics

3 Instruction characteristics
3.1 Instruction tables
This chapter describes high-level performance characteristics for most Armv8.2-A A32, T32, and A64
instructions. A series of tables summarize the effective execution latency and throughput (instruction
bandwidth per cycle), pipelines utilized, and special behaviours associated with each group of
instructions. Utilized pipelines correspond to the execution pipelines described in chapter 2.

In the tables below, Execute Latency is defined as the minimum latency seen by an operation
dependent on an instruction in the described group.

In the tables below, Execution Throughput is defined as the maximum throughput (in instructions per
cycle) of the specified instruction group that can be achieved in the entirety of the Neoverse V1
microarchitecture. However, scheduler dependencies may limit the maximum achievable execution
throughput.

3.2 Legend for reading the utilized pipelines
Table 3-1 Neoverse V1 core pipeline names and symbols
Pipeline name                                                         Symbol used in tables
Branch 0/1                                                            B
Integer single Cycle 0/1                                              S
Integer single Cycle 0/1 and single/multicycle 0/1                    I
Integer single/multicycle 0/1                                         M
Integer multicycle 0                                                  M0
Load/Store 01                                                         L01
Load/Store 0/1 and Load 2                                             L
Store data 0/1                                                        D
FP/ASIMD 0/1/2/3                                                      V
FP/ASIMD 0/1                                                          V01
FP/ASIMD 0/2                                                          V02
FP/ASIMD 1/3                                                          V13
FP/ASIMD 0                                                            V0
FP/ASIMD 1                                                            V1

                                                                                       3 Instruction characteristics

3.3 Branch instructions
Table 3-2 AArch64 Branch instructions
Instruction Group                   AArch64             Execution       Execution       Utilized      Notes
                                    Instructions        Latency         Throughput      Pipelines
Branch, immed                       B                   1               2               B             -
Branch, register                    BR, RET             1               2               B             -
Branch and link, immed              BL                  1               2               B, S          -
Branch and link, register           BLR                 1               2               B, S          -
Compare and branch                  CBZ, CBNZ, TBZ,     1               2               B             -
                                    TBNZ

Table 3-3 AAarch32 Branch instructions
Instruction Group                   AArch32             Execution       Execution       Utilized      Notes
                                    Instructions        Latency         Throughput      Pipelines

Branch, immed                       B                   1               2               B             -
Branch, register                    BX                  1               2               B             -
Branch and link, immed              BL, BLX             1               2               B, S          -
Branch and link, register           BLX                 1               2               B, S          -
Compare and branch                  CBZ, CBNZ           1               2               B             -

3.4 Arithmetic and logical instructions
Table 3-4 AArch64 Arithmetic and logical instructions
Instruction Group                   AArch64             Execution       Execution       Utilized      Notes
                                    Instructions        Latency         Throughput      Pipelines
ALU, basic                          ADD, ADC, AND,      1               4               I             -
                                    BIC, EON, EOR,
                                    ORN, ORR, SUB,
                                    SBC
ALU, basic, flagset                 ADDS, ADCS,         1               3               I             -
                                    ANDS, BICS, SUBS,
                                    SBCS
ALU, extend and shift               ADD{S}, SUB{S}      2               2               M             -
Arithmetic, LSL shift, shift <= 4   ADD, SUB            1               4               I             -
Arithmetic, flagset, LSL shift,     ADDS, SUBS          1               3               I             -
shift <= 4
Arithmetic, LSR/ASR/ROR shift ADD{S}, SUB{S}            2               2               M             -
or LSL shift > 4
Conditional compare                 CCMN, CCMP          1               4               I             -

                                                                                          3 Instruction characteristics

Instruction Group                    AArch64               Execution       Execution       Utilized      Notes
                                     Instructions          Latency         Throughput      Pipelines
Conditional select                   CSEL, CSINC,          1               4               I             -
                                     CSINV, CSNEG
Logical, shift, no flagset           AND, BIC, EON,        1               4               I             -
                                     EOR, ORN, ORR
Logical, shift, flagset              ANDS, BICS            2               2               M             -
Flag manipulation instructions       SETF8, SETF16,        1               3               I             -
                                     RMIF, CFINV

Table 3-5 AArch32 Arithmetic and logical instructions
Instruction Group                    AArch32               Execution       Execution       Utilized      Notes
                                     Instructions          Latency         Throughput      Pipelines

ALU, basic, unconditional, no        ADD, ADC, ADR,        1               4               I             -
flagset                              AND, BIC, EOR,
                                     ORN, ORR, RSB,
                                     RSC, SUB, SBC
ALU, basic, unconditional,           ADDS, ADCS,           1               3               I             -
flagset                              ANDS, BICS, CMN,
                                     CMP, EORS, ORNS,
                                     ORRS, RSBS, RSCS,
                                     SUBS, SBCS, TEQ,
                                     TST
ALU, basic, conditional              ADD{S}, ADC{S},   1                   1               M0            -
                                     AND{S}, BIC{S},
                                     CMN, CMP, EOR{S|,
                                     ORN{S}, ORR{S},
                                     RSB{S}, RSC{S},
                                     SUB{S}, SBC{S},
                                     TEQ, TST
ALU, basic, shift by register,       (same as ALU basic,   2               1               I, M0         -
conditional                          conditional)
ALU, basic, shift by register,       (same as ALU, basic, 2                1               M0            -
unconditional, flagset               unconditional,
                                     flagset)
Arithmetic, shift by register,       ADD, ADC, RSB,        2               1               M0            -
unconditional, no flagset            RSC, SUB, SBC
Logical, shift by register,          AND, BIC, EOR,        1               1               M0            -
unconditional, no flagset            ORN, ORR
Arithmetic, LSL shift by immed,      ADD, ADC, RSB,        1               4               I             -
shift <= 4, unconditional, no        RSC, SUB, SBC
flagset
Arithmetic, LSL shift by immed, ADDS, ADCS, RSBS, 1                        3               I             -
shift <= 4, unconditional, flagset RSCS, SUBS, SBCS
Arithmetic, LSL shift by immed,      ADD{S}, ADC{S},       1               1               M0            -
shift <= 4, conditional              RSB{S}, RSC{S},
                                     SUB{S}, SBC{S}

                                                                                         3 Instruction characteristics

Instruction Group                   AArch32               Execution       Execution        Utilized         Notes
                                    Instructions          Latency         Throughput       Pipelines

Arithmetic, LSR/ASR/ROR shift ADD{S}, ADC{S},             2               2                M                -
by immed or LSL shift by immed RSB{S}, RSC{S},
> 4, unconditional             SUB{S}, SBC{S}
Arithmetic, LSR/ASR/ROR shift ADD{S}, ADC{S},             2               1                M0               -
by immed or LSL shift by immed RSB{S}, RSC{S},
> 4, conditional               SUB{S}, SBC{S}
Logical, shift by immed, no         AND, BIC, EOR,        1               4                I                -
flagset, unconditional              ORN, ORR
Logical, shift by immed, no         AND, BIC, EOR,        1               1                M0               -
flagset, conditional                ORN, ORR
Logical, shift by immed, flagset,   ANDS, BICS, EORS,     2               2                M                -
unconditional                       ORNS, ORRS
Logical, shift by immed, flagset,   ANDS, BICS, EORS,     2               1                M0               -
conditional                         ORNS, ORRS
Test/Compare, shift by immed        CMN, CMP, TEQ,        2               2                M                -
                                    TST
Branch forms                        -                     +1              2                +B               1
Notes:
    1.   Branch forms are possible when the instruction destination register is the PC. For those cases, an additional
         branch µOP is required. This adds 1 cycle to the latency.

3.5 Move and shift instructions
Table 3-6 AArch32 Move and shift instructions
Instruction Group                   AArch32               Execution       Execution        Utilized         Notes
                                    Instructions          Latency         Throughput       Pipelines
Move, basic                         MOV, MOVW, MVN 1                      4                I                -
Move, basic, flagset                MOVS, MVNS            1               3                I
Move, shift by immed, no flagset ASR, LSL, LSR, ROR,      1               4                I                -
                                 RRX, MVN
Move, shift by immed, flagset       ASRS, LSLS, LSRS, 2                   2                M                -
                                    RORS, RRXS, MVNS
Move, shift by register, no         ASR, LSL, LSR, ROR,   1               4                I                -
flagset, unconditional              RRX, MVN
Move, shift by register, no         ASR, LSL, LSR, ROR,   2               2                I                -
flagset, conditional                RRX, MVN
Move, shift by register, flagset    ASRS, LSLS, LSRS, 2                   1                M0               -
                                    RORS, RRXS, MVNS
Move, top                           MOVT                  1               4                I                -
Move, branch forms                  -                     +1              2                +B               -

                                                                                       3 Instruction characteristics

3.6 Divide and multiply instructions
Table 3-7 AArch64 Divide and multiply instructions
Instruction Group                 AArch64               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines
Divide, W-form                    SDIV, UDIV            5 to 12         1/12 to 1/5     M0            1
Divide, X-form                    SDIV, UDIV            5 to 20         1/20 to 1/5     M0            1
Multiply                          MUL, MNEG             2               2               M             -
Multiply accumulate, W-form       MADD, MSUB            2(1)            1               M0            2
Multiply accumulate, X-form       MADD, MSUB            2(1)            1               M0            2
Multiply accumulate long          SMADDL, SMSUBL, 2(1)                  1               M0            2
                                  UMADDL, UMSUBL
Multiply high                     SMULH, UMULH          3               2               M             2
Multiply long                     SMNEGL, SMULL,        2               2               M             -
                                  UMNEGL, UMULL

Table 3-8 AArch32 Divide and multiply instructions
Instruction Group                 AArch32               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines

Divide                            SDIV, UDIV            5 to 12         1/12 to 1/5     M0            1
Multiply, unconditional           MUL, SMULBB,    2                     2               M             -
                                  SMULBT, SMULTB,
                                  SMULTT, SMULWB,
                                  SMULWT,
                                  SMMUL{R},
                                  SMUAD{X},
                                  SMUSD{X}
Multiply, conditional             MUL, SMULBB,    2                     1               M0            -
                                  SMULBT, SMULTB,
                                  SMULTT, SMULWB,
                                  SMULWT,
                                  SMMUL{R},
                                  SMUAD{X},
                                  SMUSD{X}
Multiply accumulate,              MLA, MLS,             3               1               M0, I         -
conditional                       SMLABB, SMLABT,
                                  SMLATB, SMLATT,
                                  SMLAWB,
                                  SMLAWT,
                                  SMLAD{X},
                                  SMLSD{X},
                                  SMMLA{R},
                                  SMMLS{R}

                                                                                         3 Instruction characteristics

Instruction Group                 AArch32               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines

Multiply accumulate,              MLA, MLS,             2(1)             1                M0               2
unconditional                     SMLABB, SMLABT,
                                  SMLATB, SMLATT,
                                  SMLAWB,
                                  SMLAWT,
                                  SMLAD{X},
                                  SMLSD{X},
                                  SMMLA{R},
                                  SMMLS{R}
Multiply accumulate               UMAAL                 4                1                I, M0            -
accumulate long, conditional
Multiply accumulate               UMAAL                 3                1                I, M0            -
accumulate long, unconditional
Multiply accumulate long, no      SMLAL, SMLALBB, 3                      1                M0, I            -
flagset                           SMLALBT,
                                  SMLALTB,
                                  SMLALTT,
                                  SMLALD{X},
                                  SMLSLD{X}, UMLAL
Multiply accumulate long,         SMLAL, SMLALBB, 4                      1                M0, I            -
flagset                           SMLALBT,
                                  SMLALTB,
                                  SMLALTT,
                                  SMLALD{X},
                                  SMLSLD{X}, UMLAL
Multiply long, unconditional, no SMULL, UMULL           2                2                M                -
flagset
Multiply long, unconditional,     SMULLS, UMULLS        3                1                M, I             -
flagset
Multiply long, conditional        SMULL{S},             3                1                M, I             -
                                  UMULL{S}
Notes:
    1.   Integer divides are performed using an iterative algorithm and block any subsequent divide operations until
         complete. Early termination is possible, depending upon the data values.
    2.   Multiply-accumulate pipelines support late-forwarding of accumulate operands from similar µOPs, allowing a
         typical sequence of multiply-accumulate µOPs to issue one every N cycles (accumulate latency N shown in
         parentheses). Accumulator forwarding is not supported for consumers of 64 bit multiply high operations.

                                                                                          3 Instruction characteristics

3.7 Saturating and parallel arithmetic instructions
Table 3-9 AArch32 Saturating and parallel arithmetic instructions
Instruction Group                    AArch32               Execution       Execution       Utilized      Notes
                                     Instructions          Latency         Throughput      Pipelines

Parallel arith, unconditional        SADD16, SADD8,        2               1               M             -
                                     SSUB16, SSUB8,
                                     UADD16, UADD8,
                                     USUB16, USUB8
Parallel arith, conditional          SADD16, SADD8,        2(4)            1               M0, I         1
                                     SSUB16, SSUB8,
                                     UADD16, UADD8,
                                     USUB16, USUB8
Parallel arith with exchange,        SASX, SSAX, UASX,     3               2               I, M          -
unconditional                        USAX
Parallel arith with exchange,        SASX, SSAX, UASX,     3(5)            1               I, M0         1
conditional                          USAX
Parallel halving arith,              SHADD16,         2                    2               M             -
unconditional                        SHADD8,
                                     SHSUB16, SHSUB8,
                                     UHADD16,
                                     UHADD8,
                                     UHSUB16,
                                     UHSUB8
Parallel halving arith,              SHADD16,         2                    1               M0            -
conditional                          SHADD8,
                                     SHSUB16, SHSUB8,
                                     UHADD16,
                                     UHADD8,
                                     UHSUB16,
                                     UHSUB8
Parallel halving arith with          SHASX, SHSAX,         3               1               I, M0         -
exchange                             UHASX, UHSAX
Parallel saturating arith,           QADD16, QADD8,        2               2               M             -
unconditional                        QSUB16, QSUB8,
                                     UQADD16,
                                     UQADD8,
                                     UQSUB16,
                                     UQSUB8
Parallel saturating arith,           QADD16, QADD8,        2               1               M0            -
conditional                          QSUB16, QSUB8,
                                     UQADD16,
                                     UQADD8,
                                     UQSUB16,
                                     UQSUB8
Parallel saturating arith with       QASX, QSAX,           3               2               I, M          -
exchange, unconditional              UQASX, UQSAX
Parallel saturating arith with       QASX, QSAX,           3               1               I, M0         -
exchange, conditional                UQASX, UQSAX

                                                                                         3 Instruction characteristics

Instruction Group                 AArch32               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines

Saturate, unconditional           SSAT, SSAT16,         2                2                M                -
                                  USAT, USAT16
Saturate, conditional             SSAT, SSAT16,         2                1                M0               -
                                  USAT, USAT16
Saturating arith, unconditional   QADD, QSUB            2                2                M                -
Saturating arith, conditional     QADD, QSUB            2                1                M0               -
Saturating doubling arith,        QDADD, QDSUB          3                1                M, M             -
unconditional
Saturating doubling arith         QDADD, QDSUB          3                1                M, M0            -
conditional
Notes:
    1.   GE-setting instructions require three extra µOPs and two additional cycles to conditionally update the GE field
         (GE latency shown in parentheses).

3.8 Pointer Authentication instructions
Table 3-10 AArch64 pointer authentication instructions
Instruction Group                 AArch64               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines
Authenticate data address         AUTDA, AUTDB,         5                1                M0               -
                                  AUTDZA, AUTDZB
Authenticate instruction          AUTIA, AUTIB,     5                    1                M0               -
address                           AUTIA1716,
                                  AUTIB1716,
                                  AUTIASP, AUTIBSP,
                                  AUTIAZ, AUTIBZ,
                                  AUTIZA, AUTIZB
Branch and link, register, with   BLRAA, BLRAAZ,        6                1                M0, B            -
pointer authentication            BLRAB, BLRABZ
Branch, register, with pointer    BRAA, BRAAZ,          6                1                M0, B            -
authentication                    BRAB, BRABZ
Branch, return, with pointer      RETA, RETB            6                1                M0, B            -
authentication
Compute pointer                   PACDA, PACDB,         5                1                M0               -
authentication code for data      PACDZA, PACDZB
address
Compute pointer                   PACGA                 5                1                M0               -
authentication code, using
generic key

                                                                                         3 Instruction characteristics

Instruction Group                   AArch64               Execution       Execution       Utilized      Notes
                                    Instructions          Latency         Throughput      Pipelines
Compute pointer                     PACIA, PACIB,     5                   1               M0            -
authentication code for             PACIA1716,
instruction address                 PACIB1716,
                                    PACIASP, PACIBSP,
                                    PACIAZ, PACIBZ,
                                    PACIZA, PACIZB
Load register, with pointer         LDRAA, LDRAB          9               1               M0, L         -
authentication
Strip pointer authentication        XPACD, XPACI,         2               1               M0            -
code                                XPACLRI

3.9 Miscellaneous data-processing instructions
Table 3-11 AArch64 Miscellaneous data-processing instructions
Instruction Group                   AArch64               Execution       Execution       Utilized      Notes
                                    Instructions          Latency         Throughput      Pipelines
Address generation                  ADR, ADRP             1               4               I             -
Bitfield extract, one reg           EXTR                  1               4               I             -
Bitfield extract, two regs          EXTR                  3               2               I, M          -
Bitfield move, basic                SBFM, UBFM            1               4               I             -
Bitfield move, insert               BFM                   2               2               M             -
Count leading                       CLS, CLZ              1               4               I             -
Move immed                          MOVN, MOVK,           1               4               I             -
                                    MOVZ
Reverse bits/bytes                  RBIT, REV, REV16,     1               4               I             -
                                    REV32
Variable shift                      ASRV, LSLV, LSRV,     1               4               I             -
                                    RORV

                                                                                       3 Instruction characteristics

Table 3-12 AArch32 Miscellaneous data-processing instructions
Instruction Group                 AArch32               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines

Bit field extract                 SBFX, UBFX            1               4               I             -
Bit field insert/clear,           BFI, BFC              2               2               M             -
unconditional
Bit field insert/clear, conditional BFI, BFC            2               1               M0            -
Count leading zeros               CLZ                   1               4               I             -
Pack halfword, unconditional      PKH                   2               2               M             -
Pack halfword, conditional        PKH                   2               1               M0            -
Reverse bits/bytes                RBIT, REV, REV16,     1               4               I             -
                                  REVSH
Select bytes, unconditional       SEL                   1               4               I             -
Select bytes, conditional         SEL                   2               2               I             -
Sign/zero extend, normal          SXTB, SXTH, UXTB,     1               4               I             -
                                  UXTH
Sign/zero extend, parallel,       SXTB16, UXTB16        2               2               M             -
unconditional
Sign/zero extend, parallel,       SXTB16, UXTB16        2               1               M0            -
conditional
Sign/zero extend and add,         SXTAB, SXTAH,         2               2               M             -
normal, unconditional             UXTAB, UXTAH
Sign/zero extend and add,         SXTAB, SXTAH,         2               1               M0            -
normal, conditional               UXTAB, UXTAH
Sign/zero extend and add,         SXTAB16,              4               1               M             -
parallel, unconditional           UXTAB16
Sign/zero extend and add,         SXTAB16,              4               1               M, M0         -
parallel, conditional             UXTAB16
Sum of absolute differences       USAD8                 2               1               M0            -
Sum of absolute differences       USADA8                2               1               M0            -
accumulate, unconditional
Sum of absolute differences       USADA8                3               1               M0, I         -
accumulate, conditional

                                                                                      3 Instruction characteristics

3.10 Load instructions
The latencies shown assume the memory access hits in the Level 1 Data Cache and represent the
maximum latency to load all the registers written by the instruction.

Table 3-13 AArch64 Load instructions
Instruction Group                 AArch64              Execution       Execution       Utilized      Notes
                                  Instructions         Latency         Throughput      Pipelines
Load register, literal            LDR, LDRSW, PRFM 4                   3               L             -
Load register, unscaled immed     LDUR, LDURB,    4                    3               L             -
                                  LDURH, LDURSB,
                                  LDURSH, LDURSW,
                                  PRFUM
Load register, immed post-        LDR, LDRB, LDRH,     4               3               L, I          -
index                             LDRSB, LDRSH,
                                  LDRSW
Load register, immed pre-index    LDR, LDRB, LDRH,     4               3               L, I          -
                                  LDRSB, LDRSH,
                                  LDRSW
Load register, immed              LDTR, LDTRB,         4               3               L             -
unprivileged                      LDTRH, LDTRSB,
                                  LDTRSH, LDTRSW
Load register, unsigned immed     LDR, LDRB, LDRH,     4               3               L             -
                                  LDRSB, LDRSH,
                                  LDRSW, PRFM
Load register, register offset,   LDR, LDRB, LDRH,     4               3               L             -
basic                             LDRSB, LDRSH,
                                  LDRSW, PRFM
Load register, register offset,   LDR, LDRSW, PRFM 4                   3               L             -
scale by 4/8
Load register, register offset,   LDRH, LDRSH          5               3               I, L          -
scale by 2
Load register, register offset,   LDR, LDRB, LDRH,     4               3               L             -
extend                            LDRSB, LDRSH,
                                  LDRSW, PRFM
Load register, register offset,   LDR, LDRSW, PRFM 4                   3               L             -
extend, scale by 4/8
Load register, register offset,   LDRH, LDRSH          5               3               I, L          -
extend, scale by 2
Load pair, signed immed offset,   LDP, LDNP            4               3               L             -
normal, W-form
Load pair, signed immed offset,   LDP, LDNP            4               1               L             -
normal, X-form
Load pair, signed immed offset,   LDPSW                5               1               I, L          -
signed words

                                                                                      3 Instruction characteristics

Instruction Group                AArch64               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines
Load pair, immed post-index or   LDP                   4               3               L, I          -
immed pre-index, normal, W-
form
Load pair, immed post-index or   LDP                   4               1               L, I          -
immed pre-index, normal, X-
form
Load pair, immed post-index or   LDPSW                 5               1               I, L          -
immed pre-index, signed words

Table 3-14 AArch32 Load instructions
Instruction Group                AArch32               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines

Load, immed offset               LDR{T}, LDRB{T},      4               3               L             1, 2
                                 LDRD, LDRH{T},
                                 LDRSB{T},
                                 LDRSH{T}
Load, register offset, plus      LDR, LDRB, LDRD,      4               3               L             1 ,2
                                 LDRH, LDRSB,
                                 LDRSH
Load, register offset, minus     LDR, LDRB, LDRD,      5               3               I, L          1, 2
                                 LDRH, LDRSB,
                                 LDRSH
Load, scaled register offset,    LDR, LDRB             4               3               L             1, 2
plus, LSL2
Load, scaled register offset,    LDR, LDRB, LDRH,      5               3               I, L          1, 2
other                            LDRSB, LDRSH
Load, immed pre-indexed          LDR, LDRB, LDRD,      4               3               L, I          1, 2
                                 LDRH, LDRSB,
                                 LDRSH
Load, register pre-indexed       LDRH, LDRSB,          5               3               I, L, M0      1, 2, 3
                                 LDRSH
Load, register pre-indexed       LDRD                  4               3               L, M0         1, 2, 3
Load, scaled register pre-       LDR, LDRB             4               3               L, M0         1, 2, 3
indexed, plus, LSL2
Load, scaled register pre-       LDR, LDRB             4               3               L, M0         1, 2, 3
indexed, unshifted
Load, scaled register pre-       LDR, LDRB             5               3               I, L, M0      1, 2, 3
indexed, other
Load, immed post-indexed         LDR{T}, LDRB{T},      4               3               L, I          1, 2
                                 LDRD, LDRH{T},
                                 LDRSB{T},
                                 LDRSH{T}
Load, register post-indexed      LDR{T}, LDRB{T},   5                  3               I, L, M0      1, 2, 3
                                 LDRH{T}, LDRSB{T},
                                 LDRSH{T}

                                                                                         3 Instruction characteristics

Instruction Group                  AArch32                Execution        Execution       Utilized         Notes
                                   Instructions           Latency          Throughput      Pipelines

Load, register post-indexed        LDRD                   4                3               L, M0            1, 2, 3
Preload, immed offset              PLD, PLDW              4                3               L                -
Preload, register offset, plus,    PLD, PLDW              4                3               L                -
LSL2 and unshifted
Preload, register offset, minus    PLD, PLDW              5                3               I, L             -
Load multiple, no writeback,       LDMIA, LDMIB,          N                3/R             L                1, 4, 5
base reg not in list               LDMDA, LDMDB
Load multiple, no writeback,       LDMIA, LDMIB,          1+ N             3/R             I, L             1, 4, 5
base reg in list                   LDMDA, LDMDB
Load multiple, writeback           LDMIA, LDMIB,          1+ N             3/R             L, I             1, 4, 5
                                   LDMDA, LDMDB,
                                   POP
(Load, all branch forms)           -                      +1               -               +B               6
Notes:
    1.   Conditional loads have extra µOP(s) which goes down pipeline 'I' and have 1 cycle extra latency compared to their
         unconditional counterparts.
    2.   Conditional loads go down L01 pipe and have an execution throughput of 2 whereas unconditional versions have a
         throughput of 3.
    3.   The address update op goes down pipeline ‘I’ if the load is unconditional.
    4.   N is floor [ (num_reg+5)/6].
    5.   R is floor [(num_reg +1)/2].
    6.   Branch forms are possible when the instruction destination register is the PC. For those cases, an additional
         branch µOP is required. This adds 1 cycle to the latency.

3.11 Store instructions
The following table describes performance characteristics for standard store instructions. Store
µOPs are split into address and data µOPs. Once executed, stores are buffered and committed in the
background.

Table 3-15 AArch64 Store instructions
Instruction Group                  AArch64                Execution        Execution       Utilized         Notes
                                   Instructions           Latency          Throughput      Pipelines
Store register, unscaled immed     STUR, STURB,           1                2               L01, D           -
                                   STURH
Store register, immed post-        STR, STRB, STRH        1                2               L01, D           -
index
Store register, immed pre-index STR, STRB, STRH           1                2               L01, D           -
Store register, immed              STTR, STTRB,           1                2               L01, D           -
unprivileged                       STTRH
Store register, unsigned immed STR, STRB, STRH            1                2               L01, D           -

                                                                                        3 Instruction characteristics

Instruction Group                  AArch64               Execution       Execution       Utilized        Notes
                                   Instructions          Latency         Throughput      Pipelines
Store register, register offset,   STR, STRB, STRH       1               2               L01, D          -
basic
Store register, register offset,   STR                   1               2               L01, D          -
scaled by 4/8
Store register, register offset,   STRH                  2               2               I, L01, D       -
scaled by 2
Store register, register offset,   STR, STRB, STRH       1               2               L01, D          -
extend
Store register, register offset,   STR                   1               2               L01, D          -
extend, scale by 4/8
Store register, register offset,   STRH                  2               2               I, L01, D       -
extend, scale by 1
Store pair, immed offset           STP, STNP             1               2               L01, D          -
Store pair, immed post-index       STP                   1               2               L01, D, I       -
Store pair, immed pre-index        STP                   1               2               L01, D, I       -

Table 3-16 AArch32 Store instructions
Instruction Group                  AArch32               Execution       Execution       Utilized        Notes
                                   Instructions          Latency         Throughput      Pipelines

Store, immed offset                STR{T}, STRB{T},      1               2               L01, D          -
                                   STRD, STRH{T}
Store, register offset, plus       STR, STRB, STRD,      1               2               L01, D          -
                                   STRH
Store, register offset, minus      STR, STRB, STRD,      1               2               L01, D          -
                                   STRH
Store, scaled register             STR, STRB             1               2               L01, D          -
offset, plus, no shift
Store, scaled register offset,     STR, STRB             1               2               L01, D          -
plus, LSL2
Store, scaled register offset,     STR, STRB             2               2               I, L01, D       -
plus, other
Store, scaled register offset,     STR, STRB             2               2               I, L01, D       -
minus
Store, immed pre-indexed           STR, STRB, STRD,      1               2               L01, D, I       -
                                   STRH
Store, register pre-indexed,       STR, STRB, STRD,      1               2               L01, D, M0      1
plus, no shift                     STRH
Store, register pre-indexed,       STR, STRB, STRD,      2               2               I, L01, D, M0   1
minus                              STRH
Store, scaled register pre-        STR, STRB             1               2               L01, D, M0      1
indexed, plus LSL2
Store, scaled register pre-        STR, STRB             2               2               I, L01, D, M0   1
indexed, other

                                                                                        3 Instruction characteristics

Instruction Group                 AArch32                Execution         Execution     Utilized      Notes
                                  Instructions           Latency           Throughput    Pipelines

Store, immed post-indexed         STR{T}, STRB{T},       1                 2             L01, D, I     -
                                  STRD, STRH{T}
Store, register post-indexed      STRH{T}, STRD          1                 2             L01, D, M0    1
Store, register post-indexed      STR{T}, STRB{T}        1                 2             L01, D, M0    1
Store, scaled register post-      STR{T}, STRB{T}        1                 2             L01, D, M0    2
indexed
Store multiple, no writeback      STMIA, STMIB,          N                 1/N           L01, D        3
                                  STMDA, STMDB
Store multiple, writeback         STMIA, STMIB,          N                 1/N           L01, D        3
                                  STMDA, STMDB,
                                  PUSH

Notes:
    1.   The address update op goes down pipeline ‘I’ if the store is unconditional.
    2.   The address update op goes down pipeline 'M' if the store is unconditional.
    3.   For store multiple instructions, N=floor((num_regs+3)/4).

3.12 FP data processing instructions
Table 3-17 AArch64 FP data processing instructions
Instruction Group                 AArch64                Execution         Execution     Utilized      Notes
                                  Instructions           Latency           Throughput    Pipelines
FP absolute value                 FABS                   2                 4             V             -
FP arithmetic                     FADD, FSUB             2                 4             V             -
FP compare                        FCCMP{E},              2                 1             V0            -
                                  FCMP{E}
FP divide, H-form                 FDIV                   7                 8/7           V02           1
FP divide, S-form                 FDIV                   7 to 10           8/9 to 8/7    V02           1
FP divide, D-form                 FDIV                   7 to 15           2/7 to 4/7    V02           1
FP min/max                        FMIN, FMINNM,          2                 4             V             -
                                  FMAX, FMAXNM
FP multiply                       FMUL, FNMUL            3                 4             V             2
FP multiply accumulate            FMADD, FMSUB,          4 (2)             4             V             3
                                  FNMADD,
                                  FNMSUB
FP negate                         FNEG                   2                 4             V             -
FP round to integral              FRINTA, FRINTI,        3                 2             V02           -
                                  FRINTM, FRINTN,
                                  FRINTP, FRINTX,
                                  FRINTZ
FP select                         FCSEL                  2                 2             V01           -

                                                                                      3 Instruction characteristics

Instruction Group                AArch64               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines
FP square root, H-form           FSQRT                 7               8/7             V02           1
FP square root, S-form           FSQRT                 7 to 9          1 to 8/7        V02           1
FP square root, D-form           FSQRT                 7 to 16         4/15 to 4/7     V02           1

Table 3-18 AArch32 FP data processing instructions
Instruction Group                AArch32               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines

VFP absolute value               VABS                  2               2               V01           -
VFP arith                        VADD, VSUB            2               2               V01           -
VFP compare, unconditional       VCMP, VCMPE           2               1               V0            -
VFP compare, conditional         VCMP, VCMPE           4               1               V01, V0       -
VFP convert                      VCVT{R}, VCVTB,       3               1               V0            -
                                 VCVTT, VCVTA,
                                 VCVTM, VCVTN,
                                 VCVTP
VFP convert to BFloat16          VCVTB, VCVTT          4               1               V0            -
VFP divide, H-form               VDIV                  7               4/7             V0            1
VFP divide, S-form               VDIV                  7 to 10         4/9 to 4/7      V0            1
VFP divide, D-form               VDIV                  7 to 15         1/7 to 2/7      V0            1
VFP max/min                      VMAXNM,               2               2               V01           -
                                 VMINNM
VFP multiply                     VMUL, VNMUL           3               2               V01           2
VFP multiply accumulate          VMLA, VMLS,           5 (2)           2               V01           3
(chained)                        VNMLA, VNMLS
VFP multiply accumulate          VFMA, VFMS,           4 (2)           2               V01           3
(fused)                          VFNMA, VFNMS
VFP negate                       VNEG                  2               2               V01           -
VFP round to integral            VRINTA, VRINTM,       3               1               V0            -
                                 VRINTN, VRINTP,
                                 VRINTR, VRINTX,
                                 VRINTZ
VFP select                       VSELEQ, VSELGE,       2               2               V01           -
                                 VSELGT, VSELVS
VFP square root, H-form          VSQRT                 7               4/7             V0            1
VFP square root, S-form          VSQRT                 7 to 9          1/2 to 4/7      V0            1
VFP square root, D-form          VSQRT                 7 to 16         2/15 to 2/7     V0            1

                                                                                        3 Instruction characteristics

Notes:
   1.    FP divide and square root operations are performed using an iterative algorithm and block subsequent similar
         operations to the same pipeline until complete.
   2.    FP multiply-accumulate pipelines support late-forwarding of the result from FP multiply µOPs to the accumulate
         operands of an FP multiply-accumulate µOP. The latter can potentially be issued 1 cycle after the FP multiply µOP
         has been issued.
   3.    FP multiply-accumulate pipelines support late-forwarding of accumulate operands from similar µOPs, allowing a
         typical sequence of multiply-accumulate µOPs to issue one every N cycles (accumulate latency N shown in
         parentheses).

3.13 FP miscellaneous instructions
Table 3-19 AArch64 FP miscellaneous instructions
Instruction Group                 AArch64               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines
FP convert, from gen to vec reg SCVTF, UCVTF            3                1                M0               -
FP convert, from vec to gen reg   FCVTAS, FCVTAU, 3                      1                V0               -
                                  FCVTMS, FCVTMU,
                                  FCVTNS, FCVTNU,
                                  FCVTPS, FCVTPU,
                                  FCVTZS, FCVTZU
FP convert, Javascript from vec FJCVTZS                 3                1                V0               -
to gen reg
FP convert, from vec to vec reg   FCVT, FCVTXN          3                2                V02              -
FP move, immed                    FMOV                  2                4                V                -
FP move, register                 FMOV                  2                4                V                -
FP transfer, from gen to low      FMOV                  3                1                M0               -
half of vec reg
FP transfer, from gen to high     FMOV                  5                1                M0, V            -
half of vec reg
FP transfer, from vec to gen reg FMOV                   2                1                V1               -

Table 3-20 AArch32 FP miscellaneous instructions
Instruction Group                 AArch32               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines

VFP move, extraction              VMOVX                 2                2                V                -
VFP move, immed                   VMOV                  2                2                V01              -
VFP move, insert                  VINS                  2                2                V                -
VFP move, register                VMOV                  2                2                V01              -
VFP transfer, core to vfp, single VMOV                  5                1                M0, V01          -
reg to S-reg, cond
VFP transfer, core to vfp, single VMOV                  3                1                M0               -
reg to S-reg, uncond

                                                                                     3 Instruction characteristics

Instruction Group                 AArch32             Execution       Execution       Utilized      Notes
                                  Instructions        Latency         Throughput      Pipelines

VFP transfer, core to vfp, single VMOV                5               1               M0, V01       -
reg to upper/lower half of D-reg
VFP transfer, core to vfp, 2 regs VMOV                6               1/2             M0, V01       -
to 2 S-regs, cond
VFP transfer, core to vfp, 2 regs VMOV                4               1/2             M0            -
to 2 S-regs, uncond
VFP transfer, core to vfp, 2 regs VMOV                5               1               M0, V01       -
to D-reg, cond
VFP transfer, core to vfp, 2 regs VMOV                3               1               M0            -
to D-reg, uncond
VFP transfer, vfp S-reg or       VMOV                 3               1               V1, I         -
upper/lower half of vfp D-reg to
core reg, cond
VFP transfer, vfp S-reg or       VMOV                 2               1               V1            -
upper/lower half of vfp D-reg to
core reg, uncond
VFP transfer, vfp 2 S-regs or D- VMOV                 3               1               V1, I         -
reg to 2 core regs, cond
VFP transfer, vfp 2 S-regs or D- VMOV                 2               1               V1            -
reg to 2 core regs, uncond

3.14 FP load instructions
The latencies shown assume the memory access hits in the Level 1 Data Cache and represent the
maximum latency to load all the vector registers written by the instruction. Compared to standard
loads, an extra cycle is required to forward results to FP/ASIMD pipelines.

Table 3-21 AArch64 FP load instructions
Instruction Group                 AArch64             Execution       Execution       Utilized      Notes
                                  Instructions        Latency         Throughput      Pipelines
Load vector reg, literal, S/D/Q   LDR                 6               3               L             -
forms
Load vector reg, unscaled         LDUR                6               3               L             -
immed
Load vector reg, immed post-      LDR                 6               3               L, I          -
index
Load vector reg, immed pre-       LDR                 6               3               L, I          -
index
Load vector reg, unsigned         LDR                 6               3               L             -
immed
Load vector reg, register offset, LDR                 6               3               L             -
basic

                                                                                         3 Instruction characteristics

Instruction Group                  AArch64               Execution       Execution        Utilized          Notes
                                   Instructions          Latency         Throughput       Pipelines
Load vector reg, register offset, LDR                    6               3                L                 -
scale, S/D-form
Load vector reg, register offset, LDR                    7               3                I, L              -
scale, H/Q-form
Load vector reg, register offset, LDR                    6               3                L                 -
extend
Load vector reg, register offset, LDR                    6               3                L                 -
extend, scale, S/D-form
Load vector reg, register offset, LDR                    7               3                I, L              -
extend, scale, H/Q-form
Load vector pair, immed offset,    LDP, LDNP             6               3                L                 -
S/D-form
Load vector pair, immed offset,    LDP, LDNP             6               3/2              L                 -
Q-form
Load vector pair, immed post-      LDP                   6               3                I, L              -
index, S/D-form
Load vector pair, immed post-      LDP                   6               3/2              L, I              -
index, Q-form
Load vector pair, immed pre-       LDP                   6               3                I, L              -
index, S/D-form
Load vector pair, immed pre-       LDP                   6               3/2              L, I              -
index, Q-form

Table 3-22 AArch32 FP load instructions
Instruction Group                  AArch32               Execution       Execution        Utilized          Notes
                                   Instructions          Latency         Throughput       Pipelines

FP load, register                  VLDR                  6               3 (2)            L                 1, 6, 7
FP load multiple, S form           VLDMIA, VLDMDB, N(N*)                 3/R (2/R)        L                 1, 2, 3, 4, 6, 7
                                   VPOP
FP load multiple, D form           VLDMIA, VLDMDB, N(N*)                 3/R (2/R)        L, V              1, 2, 3, 4, 6, 7
                                   VPOP
(FP load, writeback forms)         -                     (1)             -                +I                5, 7

Notes:
    1.   Condition loads have an extra uop which goes down pipeline 'V' and have 2 cycle extra latency compared to their
         unconditional counterparts.
    2.   N is (num_reg)/6 + 5.
    3.   N* is (num_reg)/4 + 5.
    4.   R is num_reg/2.
    5.   Writeback forms of load instructions require an extra µOP to update the base address. This update is typically
         performed in parallel with or prior to the load µOP (update latency shown in parentheses).
    6.   The number is parenthesis represents the latency and throughput of conditional loads.
    7.   Conditional loads go down L01 pipe.

                                                                                     3 Instruction characteristics

3.15 FP store instructions
Stores MOPs are split into store address and store data µOPs at dispatch time. Once executed, stores
are buffered and committed in the background.

Table 3-23 AArch64 FP store instructions
Instruction Group                AArch64              Execution       Execution       Utilized      Notes
                                 Instructions         Latency         Throughput      Pipelines
Store vector reg, unscaled       STUR                 2               2               L01, V01      -
immed, B/H/S/D-form
Store vector reg, unscaled       STUR                 2               2               L01, V01      -
immed, Q-form
Store vector reg, immed post-    STR                  2               2               L01, V01      -
index, B/H/S/D-form
Store vector reg, immed post-    STR                  2               2               L01, V01      -
index, Q-form
Store vector reg, immed pre-     STR                  2               2               L01, V01      -
index, B/H/S/D-form
Store vector reg, immed pre-     STR                  2               2               L01, V01      -
index, Q-form
Store vector reg, unsigned       STR                  2               2               L01, V01      -
immed, B/H/S/D-form
Store vector reg, unsigned       STR                  2               2               L01, V01      -
immed, Q-form
Store vector reg, register offset, STR                2               2               L01, V01      -
basic, B/H/S/D-form
Store vector reg, register offset, STR                2               2               L01, V01      -
basic, Q-form
Store vector reg, register offset, STR                2               2               I, L01, V01   -
scale, H-form
Store vector reg, register offset, STR                2               2               L01, V01      -
scale, S/D-form
Store vector reg, register offset, STR                2               2               I, L01, V01   -
scale, Q-form
Store vector reg, register offset, STR                2               2               L01, V01      -
extend, B/H/S/D-form
Store vector reg, register offset, STR                2               2               L01, V01      -
extend, Q-form
Store vector reg, register offset, STR                2               2               I, L01, V01   -
extend, scale, H-form
Store vector reg, register offset, STR                2               2               L01, V 01     -
extend, scale, S/D-form
Store vector reg, register offset, STR                2               2               I, L01, V01   -
extend, scale, Q-form
Store vector pair, immed offset, STP, STNP            2               2               L01, V01      -
S-form

                                                                                         3 Instruction characteristics

Instruction Group                 AArch64                Execution       Execution        Utilized         Notes
                                  Instructions           Latency         Throughput       Pipelines
Store vector pair, immed offset, STP, STNP               2               2                L01, V01         -
D-form
Store vector pair, immed offset, STP, STNP               2               2                L01, V01         -
Q-form
Store vector pair, immed post-    STP                    2               2                I, L01, V01      -
index, S-form
Store vector pair, immed post-    STP                    2               2                I, L01, V01      -
index, D-form
Store vector pair, immed post-    STP                    2               1                I, L01, V01      -
index, Q-form
Store vector pair, immed pre-     STP                    2               2                I, L01, V01      -
index, S-form
Store vector pair, immed pre-     STP                    2               2                I, L01, V01      -
index, D-form
Store vector pair, immed pre-     STP                    2               1                I, L01, V01      -
index, Q-form

Table 3-24 AArch32 FP store instructions
Instruction Group                 AArch32                Execution       Execution        Utilized         Notes
                                  Instructions           Latency         Throughput       Pipelines

FP store, immed offset            VSTR                   2               2                L01, V01         -
FP store multiple, S-form         VSTMIA, VSTMDB,        N+1             2/R              L01, V01         1, 2
                                  VPUSH
FP store multiple, D-form         VSTMIA, VSTMDB,        N+1             2/R              L01, V01         1, 2
                                  VPUSH
(FP store, writeback forms)       -                      (1)             -                +I               3
Notes:
   1.    For store multiple instructions, N = (num_regs/2).
   2.    R is num_regs.
   3.    Writeback forms of store instructions require an extra µOP to update the base address. This update is typically
         performed in parallel with or prior to the store µOP (update latency shown in parentheses).

                                                                                      3 Instruction characteristics

3.16 ASIMD integer instructions
Table 3-25 AArch64 ASIMD integer instructions
Instruction Group                AArch64               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines
ASIMD absolute diff              SABD, UABD            2               4               V             -
ASIMD absolute diff accum        SABA, UABA            4(1)            2               V13           2
ASIMD absolute diff accum long SABAL(2),               4(1)            2               V13           2
                               UABAL(2)
ASIMD absolute diff long         SABDL(2),             2               4               V             -
                                 UABDL(2)
ASIMD arith, basic               ABS, ADD, NEG,        2               4               V             -
                                 SADDL(2),
                                 SADDW(2),
                                 SHADD, SHSUB,
                                 SSUBL(2),
                                 SSUBW(2), SUB,
                                 UADDL(2),
                                 UADDW(2),
                                 UHADD, UHSUB,
                                 USUBL(2),
                                 USUBW(2)
ASIMD arith, complex             ADDHN(2),      2                      4               V             -
                                 RADDHN(2),
                                 RSUBHN(2),
                                 SQABS, SQADD,
                                 SQNEG, SQSUB,
                                 SRHADD,
                                 SUBHN(2),
                                 SUQADD, UQADD,
                                 UQSUB, URHADD,
                                 USQADD
ASIMD arith, dot product         VSDOT, VUDOT          2               2               V             -
ASIMD arith, pair-wise           ADDP, SADDLP,         2               4               V             -
                                 UADDLP
ASIMD arith, reduce, 4H/4S       ADDV, SADDLV,         3               2               V13           -
                                 UADDLV
ASIMD arith, reduce, 8B/8H       ADDV, SADDLV,         5               2               V13, V        -
                                 UADDLV
ASIMD arith, reduce, 16B         ADDV, SADDLV,         6               1               V13           -
                                 UADDLV
ASIMD compare                    CMEQ, CMGE,           2               4               V             -
                                 CMGT, CMHI,
                                 CMHS, CMLE,
                                 CMLT, CMTST
ASIMD dot product                SDOT, UDOT            3 (1)           4               V             2
ASIMD dot product using          SUDOT, USDOT          3(1)            4               V             2
signed and unsigned integers

                                                                                      3 Instruction characteristics

Instruction Group                AArch64               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines
ASIMD logical                    AND, BIC, EOR,        2               4               V             -
                                 MOV, MVN, NOT,
                                 ORN, ORR
ASIMD matrix multiply-           SMMLA, UMMLA,         3(1)            4               V             2
accumulate                       USMMLA
ASIMD max/min, basic and pair- SMAX, SMAXP,            2               4               V             -
wise                           SMIN, SMINP,
                               UMAX, UMAXP,
                               UMIN, UMINP
ASIMD max/min, reduce, 4H/4S SMAXV, SMINV,             3               2               V13           -
                             UMAXV, UMINV
ASIMD max/min, reduce,           SMAXV, SMINV,         5               2               V13, V        -
8B/8H                            UMAXV, UMINV
ASIMD max/min, reduce, 16B       SMAXV, SMINV,         6               1               V13           -
                                 UMAXV, UMINV
ASIMD multiply                   MUL, SQDMULH,         4               2               V02           -
                                 SQRDMULH
ASIMD multiply accumulate        MLA, MLS              4(1)            2               V02           1
ASIMD multiply accumulate        SQRDMLAH,             4               2               V02           -
high                             SQRDMLSH
ASIMD multiply accumulate        SMLAL(2),             4(1)            2               V02           1
long                             SMLSL(2),
                                 UMLAL(2),
                                 UMLSL(2)
ASIMD multiply accumulate        SQDMLAL(2),           4               2               V02           -
saturating long                  SQDMLSL(2)
ASIMD multiply/multiply long     PMUL, PMULL(2)        3               2               V01           3
(8x8) polynomial, D-form
ASIMD multiply/multiply long     PMUL, PMULL(2)        3               2               V01           3
(8x8) polynomial, Q-form
ASIMD multiply long              SMULL(2),             3               2               V02           -
                                 UMULL(2),
                                 SQDMULL(2)
ASIMD pairwise add and           SADALP, UADALP        4(1)            2               V13           2
accumulate long
ASIMD shift accumulate           SSRA, SRSRA, USRA, 4(1)               2               V13           2
                                 URSRA
ASIMD shift by immed, basic      SHL, SHLL(2),         2               2               V13           -
                                 SHRN(2), SSHLL(2),
                                 SSHR, SXTL(2),
                                 USHLL(2), USHR,
                                 UXTL(2)
ASIMD shift by immed and         SLI, SRI              2               2               V13           -
insert, basic

                                                                                        3 Instruction characteristics

Instruction Group                  AArch64               Execution       Execution       Utilized      Notes
                                   Instructions          Latency         Throughput      Pipelines
ASIMD shift by immed, complex RSHRN(2),                  4               2               V13           -
                              SQRSHRN(2),
                              SQRSHRUN(2),
                              SQSHL{U},
                              SQSHRN(2),
                              SQSHRUN(2),
                              SRSHR,
                              UQRSHRN(2),
                              UQSHL,
                              UQSHRN(2),
                              URSHR
ASIMD shift by register, basic     SSHL, USHL            2               2               V13           -
ASIMD shift by register,           SRSHL, SQRSHL,        4               2               V13           -
complex                            SQSHL, URSHL,
                                   UQRSHL, UQSHL

Table 3-26 AArch32 ASIMD integer instructions
Instruction Group                  AArch32               Execution       Execution       Utilized      Notes
                                   Instructions          Latency         Throughput      Pipelines

ASIMD absolute diff                VABD                  2               2               V01           -
ASIMD absolute diff accum          VABA                  4(1)            1               V1            2
ASIMD absolute diff accum long VABAL                     4(1)            1               V1            2
ASIMD absolute diff long           VABDL                 2               2               V01           -
ASIMD arith, basic                 VADD, VADDL,          2               2               V01           -
                                   VADDW, VNEG,
                                   VSUB, VSUBL,
                                   VSUBW
ASIMD arith, complex               VABS, VADDHN,         2               2               V01           -
                                   VHADD, VHSUB,
                                   VQABS, VQADD,
                                   VQNEG, VQSUB,
                                   VRADDHN,
                                   VRHADD,
                                   VRSUBHN,
                                   VSUBHN
ASIMD arith, pair-wise             VPADD, VPADDL         2               2               V01           -
ASIMD compare                      VCEQ, VCGE,           2               2               V01           -
                                   VCGT, VCLE, VTST
ASIMD dot product                  VSDOT, VUDOT          3(1)            2               V01           2
ASIMD dot product using            VSUDOT, VUSDOT 3(1)                   2               V01           2
signed and unsigned integers
ASIMD logical                      VAND, VBIC,           2               2               V01           -
                                   VMVN, VORR,
                                   VORN, VEOR

                                                                                        3 Instruction characteristics

Instruction Group                  AArch32               Execution       Execution       Utilized        Notes
                                   Instructions          Latency         Throughput      Pipelines

ASIMD matrix multiply-             VSMMLA,               3(1)            2               V01             2
accumulate                         VUMMLA,
                                   VUSMMLA
ASIMD max/min                      VMAX, VMIN,           2               2               V01             -
                                   VPMAX, VPMIN
ASIMD multiply                     VMUL, VQDMULH, 4                      1               V0              -
                                   VQRDMULH
ASIMD multiply accumulate          VMLA, VMLS            4(1)            1               V0              1
ASIMD multiply accumulate          VMLAL, VMLSL          4(1)            1               V0              1
long
ASIMD multiply accumulate          VQDMLAL,              4               1               V0              -
saturating long                    VQDMLSL
ASIMD multiply/multiply long       VMUL (.P8), VMULL 3                   1               V0              -
(8x8) polynomial, D-form           (.P8)
ASIMD multiply (8x8)               VMUL (.P8)            3               1               V0              -
polynomial, Q-form
ASIMD multiply long                VMULL (.S, .I),       3               1               V0              -
                                   VQDMULL
ASIMD pairwise add and             VPADAL                4(1)            1               V1              1
accumulate
ASIMD shift accumulate             VSRA, VRSRA           4(1)            1               V1              1
ASIMD shift by immed, basic        VMOVL, VSHL,          2               1               V1              -
                                   VSHLL, VSHR,
                                   VSHRN
ASIMD shift by immed and           VSLI, VSRI            2               1               V1              -
insert, basic
ASIMD shift by immed, complex VQRSHRN,        4                          1               V1              -
                              VQRSHRUN,
                              VQSHL{U},
                              VQSHRN,
                              VQSHRUN, VRSHR,
                              VRSHRN
ASIMD shift by register, basic     VSHL                  2               1               V1              -
ASIMD shift by register,           VQRSHL, VQSHL,        4               1               V1              -
complex                            VRSHL
Notes:
   1.    Multiply-accumulate pipelines support late-forwarding of accumulate operands from similar µOPs, allowing a
         typical sequence of integer multiply-accumulate µOPs to issue one every cycle or one every other cycle
         (accumulate latency shown in parentheses).
   2.    Other accumulate pipelines also support late-forwarding of accumulate operands from similar µOPs, allowing a
         typical sequence of such µOPs to issue one every cycle (accumulate latency shown in parentheses).
   3.    This category includes instructions of the form “PMULL Vd.8H, Vn.8B, Vm.8B” and “PMULL2 Vd.8H, Vn.16B,
         Vm.16B”.

                                                                                      3 Instruction characteristics

3.17 ASIMD floating-point instructions
Table 3-27 AArch64 ASIMD Floating Point instructions
Instruction Group                AArch64               Execution       Execution       Utilized      Notes
                                 Instructions          Latency         Throughput      Pipelines
ASIMD FP absolute                FABS, FABD            2               4               V             -
value/difference
ASIMD FP arith, normal           FADD, FSUB,           2               4               V             -
                                 FADDP
ASIMD FP compare                 FACGE, FACGT,         2               4               V             -
                                 FCMEQ, FCMGE,
                                 FCMGT, FCMLE,
                                 FCMLT
ASIMD FP complex add             FCADD                 2               4               V             -
ASIMD FP complex multiply        FCMLA                 4(2)            4               V             1
add
ASIMD FP convert, long (F16 to FCVTL(2)                4               1               V02           -
F32)
ASIMD FP convert, long (F32 to FCVTL(2)                3               2               V02           -
F64)
ASIMD FP convert, narrow         FCVTN(2)              4               1               V02           -
(F32 to F16)
ASIMD FP convert, narrow         FCVTN(2),             3               2               V02           -
(F64 to F32)                     FCVTXN(2)
ASIMD FP convert, other, D-      FCVTAS, FCVTAU, 3                     2               V02           -
form F32 and Q-form F64          FCVTMS, FCVTMU,
                                 FCVTNS, FCVTNU,
                                 FCVTPS, FCVTPU,
                                 FCVTZS, FCVTZU,
                                 SCVTF, UCVTF
ASIMD FP convert, other, D-      FCVTAS, FCVTAU, 4                     1               V02           -
form F16 and Q-form F32          FCVTMS, FCVTMU,
                                 FCVTNS, FCVTNU,
                                 FCVTPS, FCVTPU,
                                 FCVTZS, FCVTZU,
                                 SCVTF, UCVTF
ASIMD FP convert, other, Q-      FCVTAS, FCVTAU, 6                     1/2             V02           -
form F16                         FCVTMS, FCVTMU,
                                 FCVTNS, FCVTNU,
                                 FCVTPS, FCVTPU,
                                 FCVTZS, FCVTZU,
                                 SCVTF, UCVTF
ASIMD FP divide, D-form, F16     FDIV                  7               2/7             V02           3
ASIMD FP divide, D-form, F32     FDIV                  7 to 10         4/9 to 4/7      V02           3
ASIMD FP divide, Q-form, F16     FDIV                  10 to 13        2/13 to 1/5     V02           3
ASIMD FP divide, Q-form, F32     FDIV                  7 to 10         2/9 to 2/7      V02           3
ASIMD FP divide, Q-form, F64     FDIV                  7 to 15         1/7 to 2/7      V02           3

                                                                                  3 Instruction characteristics

Instruction Group              AArch64             Execution       Execution       Utilized      Notes
                               Instructions        Latency         Throughput      Pipelines
ASIMD FP max/min, normal       FMAX, FMAXNM,       2               4               V             -
                               FMIN, FMINNM
ASIMD FP max/min, pairwise     FMAXP,          2                   4               V             -
                               FMAXNMP, FMINP,
                               FMINNMP
ASIMD FP max/min, reduce,      FMAXV,          4                   2               V             -
F32 and D-form F16             FMAXNMV, FMINV,
                               FMINNMV
ASIMD FP max/min, reduce, Q- FMAXV,          6                     4/3             V             -
form F16                     FMAXNMV, FMINV,
                             FMINNMV
ASIMD FP multiply              FMUL, FMULX         3               4               V             2
ASIMD FP multiply accumulate   FMLA, FMLS          4(2)            4               V             1
ASIMD FP multiply accumulate   FMLAL(2),           5(2)            4               V             1
long                           FMLSL(2)
ASIMD FP negate                FNEG                2               4               V             -
ASIMD FP round, D-form F32     FRINTA, FRINTI,     3               2               V02           -
and Q-form F64                 FRINTM, FRINTN,
                               FRINTP, FRINTX,
                               FRINTZ
ASIMD FP round, D-form F16     FRINTA, FRINTI,     4               1               V02           -
and Q-form F32                 FRINTM, FRINTN,
                               FRINTP, FRINTX,
                               FRINTZ
ASIMD FP round, Q-form F16     FRINTA, FRINTI,     6               1/2             V02           -
                               FRINTM, FRINTN,
                               FRINTP, FRINTX,
                               FRINTZ
ASIMD FP square root, D-form, FSQRT                7               2/7             V02           3
F16
ASIMD FP square root, D-form, FSQRT                7 to 10         4/9 to 4/7      V02           3
F32
ASIMD FP square root, Q-form, FSQRT                11 to 13        2/13 to 2/11    V02           3
F16
ASIMD FP square root, Q-form, FSQRT                7 to 10         2/9 to 2/7      V02           3
F32
ASIMD FP square root, Q-form, FSQRT                7 to 16         2/15 to 2/7     V02           3
F64

                                                                                    3 Instruction characteristics

Table 3-28 AArch32 ASIMD Floating Point instructions
Instruction Group               AArch32              Execution       Execution       Utilized      Notes
                                Instructions         Latency         Throughput      Pipelines

ASIMD FP absolute value         VABS                 2               2               V01           -
ASIMD FP arith                  VABD, VADD,          2               2               V01           -
                                VPADD, VSUB
ASIMD FP compare                VACGE, VACGT,        2               2               V01           -
                                VACLE, VACLT,
                                VCEQ, VCGE,
                                VCGT, VCLE
ASIMD FP complex add            VCADD                2               2               V01           -
ASIMD FP complex multiply       VCMLA                4(2)            2               V01           2
add
ASIMD FP convert, integer, D-   VCVT, VCVTA,         3               1               V0            -
form                            VCVTM, VCVTN,
                                VCVTP
ASIMD FP convert, integer, Q-   VCVT, VCVTA,         4               1               V0            -
form                            VCVTM, VCVTN,
                                VCVTP
ASIMD FP convert, fixed, D-     VCVT                 3               1               V0            -
form
ASIMD FP convert, fixed, Q-     VCVT                 4               1               V0            -
form
ASIMD FP convert, half-         VCVT                 4               1               V0            -
precision
ASIMD FP max/min                VMAX, VMIN,          2               2               V01           -
                                VPMAX, VPMIN,
                                VMAXNM,
                                VMINNM
ASIMD FP multiply               VMUL, VNMUL          3               2               V01           2
ASIMD FP chained multiply       VMLA, VMLS           5(2)            2               V01           1
accumulate
ASIMD FP fused multiply         VFMA, VFMS           4(2)            2               V01           1
accumulate
ASIMD FP multiply accumulate    VFMAL, VFMSL         5(2)            2               V01           1
long
ASIMD FP negate                 VNEG                 2               2               V01           -
ASIMD FP round to integral, D- VRINTA, VRINTM,       3               1               V0            -
form                           VRINTN, VRINTP,
                               VRINTX, VRINTZ
ASIMD FP round to integral, Q- VRINTA, VRINTM,       4               1               V0            -
form                           VRINTN, VRINTP,
                               VRINTX, VRINTZ

                                                                                        3 Instruction characteristics

Notes:
   1.    ASIMD multiply-accumulate pipelines support late-forwarding of accumulate operands from similar µOPs,
         allowing a typical sequence of floating-point multiply-accumulate µOPs to issue one every N cycles (accumulate
         latency N shown in parentheses).
   2.    ASIMD multiply-accumulate pipelines support late-forwarding of the result from ASIMD FP multiply µOPs to the
         accumulate operands of an ASIMD FP multiply-accumulate µOP. The latter can potentially be issued 1 cycle after
         the ASIMD FP multiply µOP has been issued.
   3.    ASIMD divide and square root operations are performed using an iterative algorithm and block subsequent similar
         operations to the same pipeline until complete.

3.18 ASIMD BFloat16 (BF16) instructions
Table 3-29 AArch64 ASIMD BFloat (BF16) instructions
Instruction Group                AArch64                Execution       Execution        Utilized         Notes
                                 Instructions           Latency         Throughput       Pipelines
ASIMD convert, F32 to BF16       BFCVTN, BFCVTN2 4                      1                V02              -
ASIMD dot product                BFDOT                  4(2)            4                V                1
ASIMD matrix multiply            BFMMLA                 5(3)            4                V                1
accumulate
ASIMD multiply accumulate        BFMLALB,               4(2)            4                V                1
long                             BFMLALT
Scalar convert, F32 to BF16      BFCVT                  3               2                V02              -

Table 3-30 AArch32 ASIMD BFloat (BF16) instructions
Instruction Group                AArch32                Execution       Execution        Utilized         Notes
                                 Instructions           Latency         Throughput       Pipelines

ASIMD convert, F32 to BF16       VCVTB, VCVTT           4               1                V0               -
ASIMD dot product                VDOT                   4(2)            2                V01              1
ASIMD matrix multiply            VMMLA                  5(3)            2                V01              1
accumulate
ASIMD multiply accumulate        VFMAB, VFMAT           4(2)            2                V01              1
long
Scalar convert, F32 to BF16      VCVT                   3               1                V0               -
Notes:
   1.    ASIMD pipelines that execute these instructions support late-forwarding of accumulate operands from similar
         µOPs, allowing a typical sequence of µOPs to issue one every N cycles (accumulate latency N shown in
         parentheses).

                                                                                     3 Instruction characteristics

3.19 ASIMD miscellaneous instructions
Table 3-31 AArch64 ASIMD miscellaneous instructions
Instruction Group               AArch64               Execution       Execution       Utilized      Notes
                                Instructions          Latency         Throughput      Pipelines
ASIMD bit reverse               RBIT                  2               4               V             -
ASIMD bitwise insert            BIF, BIT, BSL         2               4               V             -
ASIMD count                     CLS, CLZ, CNT         2               4               V             -
ASIMD duplicate, gen reg        DUP                   3               1               M0            -
ASIMD duplicate, element        DUP                   2               4               V             -
ASIMD extract                   EXT                   2               4               V             -
ASIMD extract narrow            XTN(2)                2               4               V             -
ASIMD extract narrow,           SQXTN(2),             4               2               V13           -
saturating                      SQXTUN(2),
                                UQXTN(2)
ASIMD insert, element to        INS                   2               4               V             -
element
ASIMD move, FP immed            FMOV                  2               4               V             -
ASIMD move, integer immed       MOVI, MVNI            2               4               V             -
ASIMD reciprocal and square     URECPE, URSQRTE 3                     2               V02           -
root estimate, D-form U32
ASIMD reciprocal and square     URECPE, URSQRTE 4                     1               V02           -
root estimate, Q-form U32
ASIMD reciprocal and square     FRECPE, FRSQRTE       3               2               V02           -
root estimate, D-form F32 and
scalar forms
ASIMD reciprocal and square     FRECPE, FRSQRTE       4               1               V02           -
root estimate, D-form F16 and
Q-form F32
ASIMD reciprocal and square     FRECPE, FRSQRTE       6               1/2             V02           -
root estimate, Q-form F16
ASIMD reciprocal exponent       FRECPX                3               2               V02           -
ASIMD reciprocal step           FRECPS, FRSQRTS       4               4               V             -
ASIMD reverse                   REV16, REV32,         2               4               V             -
                                REV64
ASIMD table lookup, 1 or 2      TBL                   2               2               V01           -
table regs
ASIMD table lookup, 3 table     TBL                   4               1               V01           -
regs
ASIMD table lookup, 4 table     TBL                   4               2/3             V01           -
regs
ASIMD table lookup extension,   TBX                   2               2               V01           -
1 table reg

                                                                                       3 Instruction characteristics

Instruction Group                 AArch64               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines
ASIMD table lookup extension,     TBX                   4               1               V01           -
2 table reg
ASIMD table lookup extension,     TBX                   6               2/3             V01           -
3 table reg
ASIMD table lookup extension,     TBX                   6               2/5             V01           -
4 table reg
ASIMD transfer, element to gen UMOV, SMOV               2               1               V             -
reg
ASIMD transfer, gen reg to        INS                   5               1               M0, V         -
element
ASIMD transpose                   TRN1, TRN2            2               4               V             -
ASIMD unzip/zip                   UZP1, UZP2, ZIP1,     2               4               V             -
                                  ZIP2

Table 3-32 AArch32 ASIMD miscellaneous instructions
Instruction Group                 AArch32               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines

ASIMD bitwise insert              VBIF, VBIT, VBSL      2               2               V01           -
ASIMD count                       VCLS, VCLZ, VCNT      2               2               V01           -
ASIMD duplicate, core reg         VDUP                  3               1               M0            -
ASIMD duplicate, scalar           VDUP                  2               2               V01           -
ASIMD extract                     VEXT                  2               2               V01           -
ASIMD move, immed                 VMOV                  2               2               V01           -
ASIMD move, register              VMOV                  2               2               V01           -
ASIMD move, narrowing             VMOVN                 2               2               V01           -
ASIMD move, saturating            VQMOVN,               4               1               V1            -
                                  VQMOVUN
ASIMD reciprocal estimate, D-     VRECPE, VRSQRTE 3                     1               V0            -
form F32 and F64
ASIMD reciprocal estimate, D-     VRECPE, VRSQRTE 4                     1               V0            -
form F16 and Q-form F32
ASIMD reciprocal estimate, Q-     VRECPE, VRSQRTE 6                     1/4             V0            -
form F16

ASIMD reciprocal step             VRECPS, VRSQRTS       5               2               V01           -
ASIMD reverse                     VREV16, VREV32,       2               2               V01           -
                                  VREV64
ASIMD swap                        VSWP                  4               2/3             V01           -
ASIMD table lookup, 1 or 2        VTBL                  2               2               V01           -
table regs

                                                                                       3 Instruction characteristics

Instruction Group                 AArch32               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines

ASIMD table lookup, 3 table       VTBL                  4               1               V01           -
regs
ASIMD table lookup, 4 table       VTBL                  6               2/3             V01           -
regs
ASIMD table lookup extension,     VTBX                  2               2               V01           -
1 reg
ASIMD table lookup extension,     VTBX                  4               1               V01           -
2 table reg
ASIMD table lookup extension,     VTBX                  6               2/3             V01           -
3 table reg
ASIMD table lookup extension,     VTBX                  6               2/5             V01           -
4 table reg
ASIMD transfer, scalar to core    VMOV                  2               1               V1            -
reg, word
ASIMD transfer, scalar to core    VMOV                  3               1               V1, I         -
reg, byte/hword
ASIMD transfer, core reg to       VMOV                  5               1               M0, V01       -
scalar
ASIMD transpose                   VTRN                  4               2/3             V01           -
ASIMD unzip/zip                   VUZP, VZIP            4               2/3             V01           -

3.20 ASIMD load instructions
The latencies shown assume the memory access hits in the Level 1 Data Cache and represent the
maximum latency to load all the vector registers written by the instruction. Compared to standard
loads, an extra cycle is required to forward results to FP/ASIMD pipelines.

Table 3-33 AArch64 ASIMD load instructions
Instruction Group                 AArch64               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines
ASIMD load, 1 element,            LD1                   6               3               L             -
multiple, 1 reg, D-form
ASIMD load, 1 element,            LD1                   6               3               L             -
multiple, 1 reg, Q-form
ASIMD load, 1 element,            LD1                   6               3/2             L             -
multiple, 2 reg, D-form
ASIMD load, 1 element,            LD1                   6               3/2             L             -
multiple, 2 reg, Q-form
ASIMD load, 1 element,            LD1                   6               1               L             -
multiple, 3 reg, D-form
ASIMD load, 1 element,            LD1                   6               1               L             -
multiple, 3 reg, Q-form

                                                                                       3 Instruction characteristics

Instruction Group                 AArch64               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines
ASIMD load, 1 element,            LD1                   6               3/2             L             -
multiple, 4 reg, D-form
ASIMD load, 1 element,            LD1                   7               3/4             L             -
multiple, 4 reg, Q-form
ASIMD load, 1 element, one        LD1                   8               3               L, V          -
lane, B/H/S
ASIMD load, 1 element, one        LD1                   8               3               L, V          -
lane, D
ASIMD load, 1 element, all        LD1R                  8               3               L, V          -
lanes, D-form, B/H/S
ASIMD load, 1 element, all        LD1R                  8               3               L, V          -
lanes, D-form, D
ASIMD load, 1 element, all        LD1R                  8               3               L, V          -
lanes, Q-form
ASIMD load, 2 element,            LD2                   8               3               L, V          -
multiple, D-form, B/H/S
ASIMD load, 2 element,            LD2                   8               3/2             L, V          -
multiple, Q-form, B/H/S
ASIMD load, 2 element,            LD2                   8               3/2             L, V          -
multiple, Q-form, D
ASIMD load, 2 element, one        LD2                   8               2               L, V          -
lane, B/H
ASIMD load, 2 element, one        LD2                   8               2               L, V          -
lane, S
ASIMD load, 2 element, one        LD2                   8               2               L, V          -
lane, D
ASIMD load, 2 element, all        LD2R                  8               2               L, V          -
lanes, D-form, B/H/S
ASIMD load, 2 element, all        LD2R                  8               2               L, V          -
lanes, D-form, D
ASIMD load, 2 element, all        LD2R                  8               2               L, V          -
lanes, Q-form
ASIMD load, 3 element,            LD3                   8               4/3             L, V          -
multiple, D-form, B/H/S
ASIMD load, 3 element,            LD3                   8               1               L, V          -
multiple, Q-form, B/H/S
ASIMD load, 3 element,            LD3                   8               1               L, V          -
multiple, Q-form, D
ASIMD load, 3 element, one        LD3                   8               4/3             L, V          -
lane, B/H
ASIMD load, 3 element, one        LD3                   8               4/3             L, V          -
lane, S
ASIMD load, 3 element, one        LD3                   8               4/3             L, V          -
lane, D

                                                                                       3 Instruction characteristics

Instruction Group                 AArch64               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines
ASIMD load, 3 element, all        LD3R                  8               4/3             L, V          -
lanes, D-form, B/H/S
ASIMD load, 3 element, all        LD3R                  8               4/3             L, V          -
lanes, D-form, D
ASIMD load, 3 element, all        LD3R                  8               4/3             L, V          -
lanes, Q-form, B/H/S
ASIMD load, 3 element, all        LD3R                  8               4/3             L, V          -
lanes, Q-form, D
ASIMD load, 4 element,            LD4                   8               1               L, V          -
multiple, D-form, B/H/S
ASIMD load, 4 element,            LD4                   9               1/2             L, V          -
multiple, Q-form, B/H/S
ASIMD load, 4 element,            LD4                   9               1/2             L, V          -
multiple, Q-form, D
ASIMD load, 4 element, one        LD4                   8               1               L, V          -
lane, B/H
ASIMD load, 4 element, one        LD4                   8               1               L, V          -
lane, S
ASIMD load, 4 element, one        LD4                   8               1               L, V          -
lane, D
ASIMD load, 4 element, all        LD4R                  8               1               L, V          -
lanes, D-form, B/H/S
ASIMD load, 4 element, all        LD4R                  8               1               L, V          -
lanes, D-form, D
ASIMD load, 4 element, all        LD4R                  8               1               L, V          -
lanes, Q-form, B/H/S
ASIMD load, 4 element, all        LD4R                  8               1               L, V          -
lanes, Q-form, D
(ASIMD load, writeback form)      -                     -               -                   I         1

Table 3-34 AArch32 ASIMD load instructions
Instruction Group                 AArch32               Execution       Execution       Utilized      Notes
                                  Instructions          Latency         Throughput      Pipelines

ASIMD load, 1 element,            VLD1                  6               3(2)            L             2
multiple, 1 reg
ASIMD load, 1 element,            VLD1                  6               3(2)            L             2
multiple, 2 reg
ASIMD load, 1 element,            VLD1                  6               3/2(1)          L             2
multiple, 3 reg
ASIMD load, 1 element,            VLD1                  6               3/2(1)          L             2
multiple, 4 reg
ASIMD load, 1 element, one        VLD1                  8               3(2)            L, V          2
lane

                                                                                         3 Instruction characteristics

Instruction Group                 AArch32               Execution        Execution        Utilized         Notes
                                  Instructions          Latency          Throughput       Pipelines

ASIMD load, 1 element, all        VLD1                  8                3(2)             LV               2
lanes, 1 reg
ASIMD load, 1 element, all        VLD1                  8                2                L, V             2
lanes, 2 reg
ASIMD load, 2 element,            VLD2                  8                2                L, V             2
multiple, 2 reg
ASIMD load, 2 element,            VLD2                  8                1                L, V             2
multiple, 4 reg
ASIMD load, 2 element, one        VLD2                  8                2                L, V             2
lane, size 32
ASIMD load, 2 element, one        VLD2                  8                2                L, V             2
lane, size 8/16
ASIMD load, 2 element, all lanes VLD2                   8                2                L, V             2
ASIMD load, 3 element,            VLD3                  9                4/3 (1)          L, V             2
multiple, 3 reg
ASIMD load, 3 element, one        VLD3                  8                4/3 (1)          L, V             2
lane, size 32
ASIMD load, 3 element, one        VLD3                  8                4/3 (1)          L, V             2
lane, size 8/16
ASIMD load, 3 element, all lanes VLD3                   8                4/3 (1)          L, V             2
ASIMD load, 4 element,            VLD4                  8                1                L, V             2
multiple, 4 reg
ASIMD load, 4 element, one        VLD4                  8                1                L, V             2
lane, size 32
ASIMD load, 4 element, one        VLD4                  8                1                L, V             2
lane, size 8/16
ASIMD load, 4 element, all lanes VLD4                   8                1                L, V             2
(ASIMD load, writeback form)      -                     -                -                I                1
Notes:
   1.    Writeback forms of load instructions require an extra µOP to update the base address. This update is typically
         performed in parallel with the load µOP (update latency shown in parentheses).
   2.    Conditional loads go down L01 pipe and the number in parenthesis represents their throughput when different
         from the unconditional forms.

                                                                                   3 Instruction characteristics

3.21 ASIMD store instructions
Stores MOPs are split into store address and store data µOPs at dispatch time. Once executed, stores
are buffered and committed in the background.

Table 3-35 AArch64 ASIMD store instructions
Instruction Group             AArch64               Execution       Execution       Utilized      Notes
                              Instructions          Latency         Throughput      Pipelines
ASIMD store, 1 element,       ST1                   2               2               L01, V01      -
multiple, 1 reg, D-form
ASIMD store, 1 element,       ST1                   2               2               L01, V01      -
multiple, 1 reg, Q-form
ASIMD store, 1 element,       ST1                   2               2               L01, V01      -
multiple, 2 reg, D-form
ASIMD store, 1 element,       ST1                   2               1               L01, V01      -
multiple, 2 reg, Q-form
ASIMD store, 1 element,       ST1                   2               1               L01, V01      -
multiple, 3 reg, D-form
ASIMD store, 1 element,       ST1                   2               2/3             L01, V01      -
multiple, 3 reg, Q-form
ASIMD store, 1 element,       ST1                   2               1               L01, V01      -
multiple, 4 reg, D-form
ASIMD store, 1 element,       ST1                   2               1/2             L01, V01      -
multiple, 4 reg, Q-form
ASIMD store, 1 element, one   ST1                   4               2               L01, V01      -
lane, B/H/S
ASIMD store, 1 element, one   ST1                   4               2               L01, V01      -
lane, D
ASIMD store, 2 element,       ST2                   4               2               V01, L01      -
multiple, D-form, B/H/S
ASIMD store, 2 element,       ST2                   4               1               V01, L01      -
multiple, Q-form, B/H/S
ASIMD store, 2 element,       ST2                   4               1               V01, L01      -
multiple, Q-form, D
ASIMD store, 2 element, one   ST2                   4               2               V01, L01      -
lane, B/H/S
ASIMD store, 2 element, one   ST2                   4               2               V01, L01      -
lane, D
ASIMD store, 3 element,       ST3                   4               1               V01, L01      -
multiple, D-form, B/H/S
ASIMD store, 3 element,       ST3                   5               2/3             V01, L01      -
multiple, Q-form, B/H/S
ASIMD store, 3 element,       ST3                   5               2/3             V01, L01      -
multiple, Q-form, D
ASIMD store, 3 element, one   ST3                   4               1               V01, L01      -
lane, B/H

                                                                                   3 Instruction characteristics

Instruction Group               AArch64             Execution       Execution       Utilized      Notes
                                Instructions        Latency         Throughput      Pipelines
ASIMD store, 3 element, one     ST3                 4               1               V01, L01      -
lane, S
ASIMD store, 3 element, one     ST3                 4               1               V01, L01      -
lane, D
ASIMD store, 4 element,         ST4                 6               1/3             V01, L01      -
multiple, D-form, B/H/S
ASIMD store, 4 element,         ST4                 7               1/6             V01, L01      -
multiple, Q-form, B/H/S
ASIMD store, 4 element,         ST4                 4               1/4             V01, L01      -
multiple, Q-form, D
ASIMD store, 4 element, one     ST4                 6               4/3             V01, L01      -
lane, B/H
ASIMD store, 4 element, one     ST4                 6               4/3             V01, L01      -
lane, S
ASIMD store, 4 element, one     ST4                 4               1               V01, L01      -
lane, D
(ASIMD store, writeback form)   -                   -               -               I             1

Table 3-36 AArch32 ASIMD store instructions
Instruction Group               AArch32             Execution       Execution       Utilized      Notes
                                Instructions        Latency         Throughput      Pipelines

ASIMD store, 1 element,         VST1                2               2               L01, V01      -
multiple, 1 reg
ASIMD store, 1 element,         VST1                2               2               L01, V01      -
multiple, 2 reg
ASIMD store, 1 element,         VST1                2               1               L01, V01      -
multiple, 3 reg
ASIMD store, 1 element,         VST1                2               1               L01, V01      -
multiple, 4 reg
ASIMD store, 1 element, one     VST1                4               2               V01, L01      -
lane
ASIMD store, 2 element,         VST2                4               4/3             V01, L01      -
multiple, 2 reg
ASIMD store, 2 element,         VST2                4               2/3             V01, L01      -
multiple, 4 reg
ASIMD store, 2 element, one     VST2                4               2               V01, L01      -
lane
ASIMD store, 3 element,         VST3                4               1               V01, L01      -
multiple, 3 reg
ASIMD store, 3 element, one     VST3                4               1               V01, L01      -
lane, size 32
ASIMD store, 3 element, one     VST3                4               1               V01, L01      -
lane, size 8/16

                                                                                          3 Instruction characteristics

Instruction Group                 AArch32               Execution        Execution         Utilized        Notes
                                  Instructions          Latency          Throughput        Pipelines

ASIMD store, 4 element,           VST4                  4                2/3               V01, L01        -
multiple, 4 reg
ASIMD store, 4 element, one       VST4                  4                4/3               V01, L01        -
lane, size 32
ASIMD store, 4 element, one       VST4                  4                4/3               V01, L01        -
lane, size 8/16
(ASIMD store, writeback form)     -                     (1)              -                 +I              1
Notes:
   1.    Writeback forms of store instructions require an extra µOP to update the base address. This update is typically
         performed in parallel with the store µOP (update latency shown in parentheses).

3.22 Cryptography extensions
Table 3-37 AArch64 Cryptography extensions
Instruction Group               AArch64                  Execution           Execution     Utilized         Notes
                                Instructions             Latency             Throughput    Pipelines
Crypto AES ops                  AESD, AESE, AESIMC, 2                        4             V                -
                                AESMC
Crypto polynomial (64x64)       PMULL (2)                2                   4             V                -
multiply long
Crypto SHA1 hash                SHA1H                    2                   1             V0               -
acceleration op
Crypto SHA1 hash                SHA1C, SHA1M,            4                   1             V0               -
acceleration ops                SHA1P
Crypto SHA1 schedule            SHA1SU0, SHA1SU1         2                   1             V0               -
acceleration ops
Crypto SHA256 hash              SHA256H, SHA256H2 4                          1             V0               -
acceleration ops
Crypto SHA256 schedule          SHA256SU0,               2                   1             V0               -
acceleration ops                SHA256SU1
Crypto SHA512 hash              SHA512H,                 2                   1             V0               -
acceleration ops                SHA512H2,
                                SHA512SU0,
                                SHA512SU1
Crypto SHA3 ops                 BCAX, EOR3, RAX1,        2                   1             V0               -
                                XAR
Crypto SM3 ops                  SM3PARTW1,        2                          1             V0               -
                                SM3PARTW2SM3SS1,
                                SM3TT1A, SM3TT1B,
                                SM3TT2A, SM3TT2B
Crypto SM4 ops                  SM4E, SM4EKEY            4                   1             V0               -

                                                                                       3 Instruction characteristics

Table 3-38 AArch32 Cryptography extensions
Instruction Group                AArch32               Execution        Execution       Utilized         Notes
                                 Instructions          Latency          Throughput      Pipelines

Crypto AES ops                   AESD, AESE,           2                2               V01              1
                                 AESIMC, AESMC
Crypto polynomial (64x64)        VMULL.P64             2                2               V01              -
multiply long
Crypto SHA1 hash acceleration SHA1H                    2                1               V0               -
ops
Crypto SHA1 hash acceleration SHA1C, SHA1M,            4                1               V0               -
ops                           SHA1P
Crypto SHA1 schedule             SHA1SU0,              2                1               V0               -
acceleration ops                 SHA1SU1
Crypto SHA256 hash               SHA256H,              4                1               V0               -
acceleration ops                 SHA256H2
Crypto SHA256 schedule           SHA256SU0,            2                1               V0               -
acceleration ops                 SHA256SU1
Notes:
   1.    Adjacent AESE/AESMC instruction pairs and adjacent AESD/AESIMC instruction pairs will exhibit the
         performance characteristics described in Section 4.6.

3.23 CRC
Table 3-39 AArch64 CRC
Instruction Group                AArch64               Execution        Execution       Utilized         Notes
                                 Instructions          Latency          Throughput      Pipelines
CRC checksum ops                 CRC32, CRC32C         2                1               M0               1

Table 3-40 AArch32 CRC
Instruction Group                AArch32               Execution        Execution       Utilized         Notes
                                 Instructions          Latency          Throughput      Pipelines

CRC checksum ops                 CRC32, CRC32C         2                1               M0               1
Notes:
   1.    CRC execution supports late-forwarding of the result from a producer µOP to a consumer µOP. This results in a 1
         cycle reduction in latency as seen by the consumer.

                                                                                       3 Instruction characteristics

3.24 SVE Predicate instructions
Table 3-41 SVE Predicate instructions
Instruction Group                 SVE Instruction        Execution       Execution       Utilized     Notes
                                                         Latency         Throughput      Pipelines
Loop control, based on            BRKA, BRKB, BRKN,      2               1               M0           1
predicate                         BRKPA, BRKPB
Loop control, based on            BRKAS, BRKBS,          3               1/2             M0           1
predicate and flag setting        BRKNS, BRKPAS,
                                  BRKPBS
Loop control, based on GPR        WHILELE, WHILELO,      3               1/2             M0           -
                                  WHILELS, WHILELT
Loop terminate                    CTERMEQ,               1               1               M0           -
                                  CTERMNE
Predicate counting scalar         ADDPL, ADDVL,     2                    1               M0           -
                                  CNTB, CNTH, CNTW,
                                  CNTD, DECB, DECH,
                                  DECW, DECD, INCB,
                                  INCH, INCW, INCD,
                                  RDVL, SQDECB,
                                  SQDECH, SQDECW,
                                  SQDECD, SQINCB,
                                  SQINCH, SQINCW,
                                  SQINCD, UQDECB,
                                  UQDECH, UQDECW,
                                  UQDECD, UQINCB,
                                  UQINCH, UQINCW,
                                  UQINCD
Predicate counting scalar,        CNTP, DECP, INCP,      2               1               M0           -
active predicate                  SQDECP, SQINCP,
                                  UQDECP, UQINCP
Predicate counting vector,        DECP, INCP,            7               1/2             M0, V01      -
active predicate                  SQDECP, SQINCP,
                                  UQDECP, UQINCP
Predicate logical                 AND, BIC, EOR, MOV, 1                  1               M0           1
                                  NAND, NOR, NOT,
                                  ORN, ORR
Predicate logical, flag setting   ANDS, BICS, EORS,      2               1/2             M0           1
                                  MOV, NANDS, NORS,
                                  NOTS, ORNS, ORRS
Predicate reverse                 REV                    2               1               M0           -
Predicate select                  SEL                    1               1               M0           -
Predicate set/initialize/find     PFALSE, PFIRST,        2               1               M0           -
next                              PNEXT, PTEST,
                                  PTRUE
Predicate set/initialize, set     PTRUES                 3               1/2             M0           -
flags
Predicate transpose               TRN1, TRN2             2               1               M0           -

                                                                                         3 Instruction characteristics

Instruction Group                SVE Instruction          Execution        Execution        Utilized       Notes
                                                          Latency          Throughput       Pipelines
Predicate unpack and widen       PUNPKHI, PUNPKLO         2                1                M0             -
Predicate zip/unzip              ZIP1, ZIP2, UZP1,        2                1                M0             -
                                 UZP2
Notes:
    1.   When the governing predicate is the same as destination, the latency is increased by one cycle.

3.25 SVE integer instructions
Table 3-42 SVE integer instructions
Instruction Group                SVE Instruction          Execution        Execution        Utilized       Notes
                                                          Latency          Throughput       Pipelines
Arithmetic, basic                ABS, ADD, ADR,     2                      2                V01            -
                                 CNOT, NEG, SABD,
                                 SMAX, SMIN, SQADD,
                                 SQSUB, SUB, SUBR,
                                 UABD, UMAX, UMIN,
                                 UQADD, UQSUB
Arithmetic, shift                ASR, ASRR, LSL, LSLR,    2                1                V1             -
                                 LSR, LSRR
Arithmetic, shift right for      ASRD                     4                1                V1             -
divide
Count/reverse bits               CLS, CLZ, CNT, RBIT      2                2                V01            -
Broadcast logical bitmask        DUPM, MOV                2                2                V01            -
immediate to vector
Compare and set flags            CMPEQ, CMPGE,            4                1                V0, M0
                                 CMPGT, CMPHI,                                                             1
                                 CMPHS, CMPLE,
                                 CMPLO, CMPLS,
                                 CMPLT, CMPNE
Conditional extract              CLASTA, CLASTB           9                1                M0, V1         -
operations, scalar form
Conditional extract              CLASTA, CLASTB,          3                1                V1             -
operations, SIMD&FP scalar       COMPACT, SPLICE
and vector forms
Convert to floating point, 64b   SCVTF, UCVTF             3                1                V0             -
to float or convert to double
Convert to floating point, 32b   SCVTF, UCVTF             4                1/2              V0             -
to single or half
Convert to floating point, 16b   SCVTF, UCVTF             6                1/4              V0             -
to half
Copy, scalar                     CPY                      5                1                M0, V01        -
Copy, scalar SIMD&FP or imm CPY                           2                2                V01            -

                                                                                        3 Instruction characteristics

Instruction Group                 SVE Instruction         Execution       Execution       Utilized     Notes
                                                          Latency         Throughput      Pipelines
Divides, 32 bit                   SDIV, SDIVR, UDIV,      7 to 12         1/11 to 1/7     V0           3
                                  UDIVR
Divides, 64 bit                   SDIV, SDIVR, UDIV,      7 to 20         1/20 to 1/7     V0           3
                                  UDIVR
Dot product, 8 bit                SDOT, UDOT              3(1)            2               V01          2
Dot product, 8 bit, using         SUDOT, USDOT            3(1)            2               V            2
signed and unsigned integers
Dot product, 16 bit               SDOT, UDOT              4(1)            1               V0           2
Duplicate, immediate and          DUP, MOV                2               2               V01          -
indexed form
Duplicate, scalar form            DUP, MOV                3               1               M0           -
Extend, sign or zero              SXTB, SXTH, SXTW,       2               1               V1           -
                                  UXTB, UXTH, UXTW
Extract                           EXT                     2               2               V01          -
Extract/insert operation,         LASTA, LASTB            3               1               V1           -
SIMD and FP scalar form
Extract/insert operation,         LASTA, LASTB            5               1               V01          -
scalar
Horizontal operations, B, H, S    INDEX                   4               1               V0           -
form, imm, imm
Horizontal operations, B, H, S INDEX                      7               1               M0, V0       -
form, scalar, imm/ scalar/ imm,
scalar
Horizontal operations, D form, INDEX                      5               1/2             V0           -
imm, imm
Horizontal operations, D form, INDEX                      8               1/2             M0, V0       -
scalar, imm/ scalar/ imm, scalar
Insert operation, SIMD and FP INSR                        2               2               V01          -
scalar form
Insert operation, scalar          INSR                    5               1               M0, V01      -
Logical                           AND, BIC, EON, EOR,     2               2               V01          -
                                  MOV, NOT, ORN,
                                  ORR
Move prefix                       MOVPRFX                 2               2               V01          -
Matrix multiply-accumulate        SMMLA, UMMLA,           3(1)            2               V01          2
                                  USMMLA
Multiply, B, H, S element size    MUL, SMULH,             4               1               V0           -
                                  UMULH
Multiply, D element size          MUL, SMULH,             5               1/2             V0           -
                                  UMULH
Multiply accumulate, D            MLA, MLS, MAD, MSB, 5(2)                1/2             V0           2
element size

                                                                                         3 Instruction characteristics

Instruction Group                SVE Instruction          Execution        Execution        Utilized        Notes
                                                          Latency          Throughput       Pipelines
Predicate counting vector        CNT, DECB, DECH,  2                       2                V01             -
                                 DECW, DECD, INCB,
                                 INCH, INCW, INCD,
                                 SQDECB, SQDECH,
                                 SQDECW, SQDECD,
                                 SQINCB, SQINCH,
                                 SQINCW, SQINCD,
                                 UQDECB, UQDECH,
                                 UQDECW, UQDECD,
                                 UQINCB, UQINCH,
                                 UQINCW, UQINCD
Reduction, arithmetic, B form    SADDV, UADDV,            14               1/2              V01, V1, V13,   -
                                 SMAXV, SMINV,                                              V
                                 UMAXV, UMINV
Reduction, arithmetic, H form    SADDV, UADDV,            12               1/2              V01, V1, V      -
                                 SMAXV, SMINV,
                                 UMAXV, UMINV
Reduction, arithmetic, S form    SADDV, UADDV,            10               1/2              V01, V1, V      -
                                 SMAXV, SMINV,
                                 UMAXV, UMINV
Reduction, max/min, D form       SMAXV, SMINV,            8                2                V               -
                                 UMAXV, UMINV
Reduction, logical               ANDV, EORV, ORV          12               1/2              V01             -

Reverse, vector                  REV, REVB, REVH,         2                2                V01             -
                                 REVW
Select, vector form              MOV, SEL                 2                2                V01             -
Table lookup                     TBL                      2                2                V01             -
Transpose, vector form           TRN1, TRN2               2                2                V01             -
Unpack and extend                SUNPKHI, SUNPKLO, 2                       2                V01             -
                                 UUNPKHI, UUNPKLO
Zip/unzip                        UZP1, UZP2, ZIP1,        2                2                V01             -
                                 ZIP2
Notes:
   1.    When the governing predicate is the same as destination, the latency is increased by one cycle.
   2.    SVE accumulate pipelines support late-forwarding of accumulate operands from similar µOPs, allowing a typical
         sequence of such µOPs to issue one every N cycles (accumulate latency N shown in parentheses).
   3.    SVE integer divide operations are performed using an iterative algorithm and block subsequent similar operations
         to the same pipeline until complete.

                                                                                        3 Instruction characteristics

3.26 SVE floating-point instructions
Table 3-43 SVE floating-point instructions
Instruction Group                 SVE Instruction        Execution       Execution       Utilized      Notes
                                                         Latency         Throughput      Pipelines
Floating point absolute           FABD, FABS             2               2               V01           -
value/difference
Floating point arithmetic         FADD, FADDP, FNEG, 2                   2               V01           -
                                  FSUB, FSUBR
Floating point associative add,   FADDA                  19              1/18            V0            -
F16
Floating point associative add,   FADDA                  11              1/10            V0            -
F32
Floating point associative add,   FADDA                  8               2/3             V01           -
F64
Floating point compare            FACGE, FACGT,          2               1               V0            -
                                  FACLE, FACLT,
                                  FCMEQ, FCMGE,
                                  FCMGT, FCMLE,
                                  FCMLT, FCMNE,
                                  FCMUO
Floating point complex add        FCADD                  3               2               V01           -
Floating point complex            FCMLA                  5(2)            2               V01           1
multiply add
Floating point convert, long or   FCVT                   4               1/2             V0            -
narrow (F16 to F32 or F32 to
F16)
Floating point convert, long or   FCVT                   3               1               V0            -
narrow (F16 to F64, F32 to
F64, F64 to F32 or F64 to
F16)
Floating point convert to         FCVTZS, FCVTZU         6               1/4             V0            -
integer, F16
Floating point convert to         FCVTZS, FCVTZU         4               1/2             V0            -
integer, F32
Floating point convert to         FCVTZS, FCVTZU         3               1               V0            -
integer, F64
Floating point copy               FCPY, FDUP, FMOV       2               2               V01           -
Floating point divide, F16        FDIV, FDIVR            10 to 13        1/12 to 1/10    V0            2
Floating point divide, F32        FDIV, FDIVR            7 to 10         1/9 to 1/7      V0            2
Floating point divide, F64        FDIV, FDIVR            7 to 15         1/14 to 1/7     V0            2
Floating point min/max            FMAX, DMIN,            2               2               V01           -
                                  FMAXNM, FMINNM
Floating point multiply           FSCALE, FMUL,          3               2               V01           -
                                  FMULX

                                                                                        3 Instruction characteristics

Instruction Group                SVE Instruction         Execution       Execution       Utilized         Notes
                                                         Latency         Throughput      Pipelines
Floating point multiply          FMLA, FMLS, FMAD,       4(2)            2               V01              1
accumulate                       FMSB, FNMAD,
                                 FNMLA, FNMLS,
                                 FNMSB
Floating point reciprocal        FRECPE, FRSQRTE         6               1               V0               -
estimate, F16
Floating point reciprocal        FRECPE, FRSQRTE         4               1               V0               -
estimate, F32
Floating point reciprocal        FRECPE, FRSQRTE         3               1               V0               -
estimate, F64
Floating point reciprocal        FRECPX                  3               1               V0
exponent
Floating point reciprocal step   FRECPS, FRSQRTS         4               2               V01              -
Floating point reduction, F16    FADDV, FMAXNMV,         13              1/3             V01              -
                                 FMAXV, FMINNMV,
                                 FMINV
Floating point reduction, F32    FADDV, FMAXNMV,         11              2/5             V01, V           -
                                 FMAXV, FMINNMV,
                                 FMINV
Floating point reduction, F64    FADDV, FMAXNMV,         9               1/2             V01, V           -
                                 FMAXV, FMINNMV,
                                 FMINV
Floating point round to          FRINTA, FRINTM,         6               1               V0               -
integral, F16                    FRINTN, FRINTP,
                                 FRINTX, FRINTZ
Floating point round to          FRINTA, FRINTM,         4               1               V0               -
integral, F32                    FRINTN, FRINTP,
                                 FRINTX, FRINTZ
Floating point round to          FRINTA, FRINTM,         3               1               V0               -
integral, F64                    FRINTN, FRINTP,
                                 FRINTX, FRINTZ
Floating point square root,      FSQRT                   10 to 13        1/12 to 1/10    V0               2
F16
Floating point square root,      FSQRT                   7 to 10         1/9 to 1/7      V0               2
F32
Floating point square root F64 FSQRT                     7 to 16         1/14 to 1/7     V0               2
Floating point trigonometric     FEXPA, FTMAD,           3               2               V01              -
                                 FTSMUL, FTSSEL
Notes:
    1.   SVE multiply-accumulate pipelines support late-forwarding of accumulate operands from similar µOPs, allowing a
         typical sequence of floating-point multiply-accumulate µOPs to issue one every N cycles (accumulate latency N
         shown in parentheses).
    2.   SVE divide and square root operations are performed using an iterative algorithm and block subsequent similar
         operations to the same pipeline until complete.

                                                                                       3 Instruction characteristics

3.27 SVE BFloat16 (BF16) instructions
Table 3-44 SVE Bfloat16 (BF16) instructions
Instruction Group               SVE Instruction         Execution        Execution       Utilized        Notes
                                                        Latency          Throughput      Pipelines
Convert, F32 to BF16            BFCVT, BFCVTNT          4                1               V0              -
Dot product                     BFDOT                   4(2)             2               V01             1
Matrix multiply accumulate      BFMMLA                  5(3)             2               V01             1
Multiply accumulate long        BFMLALB, BFMLALT        5(2)             2               V01             1
Notes:
   1.    SVE pipelines that execute these instructions support late-forwarding of accumulate operands from similar µOPs,
         allowing a typical sequence of µOPs to issue one every N cycles (accumulate latency N shown in parentheses).

                                                                                      3 Instruction characteristics

3.28 SVE Load instructions
The latencies shown assume the memory access hits in the Level 1 Data Cache and represent the
maximum latency to load all the vector registers written by the instruction.

Table 3-45 SVE Load instructions
Instruction Group                 SVE Instruction       Execution       Execution       Utilized     Notes
                                                        Latency         Throughput      Pipelines
Load vector                       LDR                   6               2               L01          -
Load predicate                    LDR                   6               2               L, M         -
Contiguous load, scalar + imm     LD1B, LD1D, LD1H,     6               2               L01          -
                                  LD1W, LD1SB,
                                  LD1SH, LD1SW,
Contiguous load, scalar +         LD1H, LD1SH           7               2               L01, S       -
scalar
Contiguous load, scalar +         LD1B, LD1D, LD1W,     6               2               L01          -
scalar                            LD1SB, LD1SW
Contiguous load broadcast,        LD1RB, LD1RH,         6               2               L01          -
scalar + imm                      LD1RD, LD1RW,
                                  LD1RSB, LD1RSH,
                                  LD1RSW, LD1RQB,
                                  LD1RQD, LD1RQH,
                                  LD1RQH,
Contiguous load broadcast,        LD1RQH                7               2               L01, S       -
scalar + scalar
Contiguous load broadcast,        LD1RQB, LD1RQH,       6               2               L01          -
scalar + scalar                   LD1RQW
Non temporal load, scalar +       LDNT1B, LDNT1D,       6               2               L01          -
imm                               LDNT1H, LDNT1W
Non temporal load, scalar +       LDNT1H                7               2               L01, S       -
scalar
Non temporal load, scalar +       LDNT1B, LDNT1D,       6               2               L01, S       -
scalar                            LDNT1W
Contiguous first faulting load,   LDFF1B, LDFF1D,       6               2               L01, S       -
scalar + scalar                   LDFF1W, LDFF1SB,
                                  LDFF1SD, LDFF1SW
Contiguous first faulting load,   LDFF1H, LDFF1SH       7               2               L01, S       -
scalar + scalar
Contiguous non faulting load,     LDNF1B, LDNF1D,       6               2               L01          -
scalar + imm                      LDNF1H, LDNF1W,
                                  LDNF1SB, LDNF1SH,
                                  LDNF1SW
Contiguous Load two               LD2B, LD2D, LD2H,     8               1               V01, L01     -
structures to two vectors,        LD2W
scalar + imm

                                                                                     3 Instruction characteristics

Instruction Group              SVE Instruction         Execution       Execution       Utilized      Notes
                                                       Latency         Throughput      Pipelines
Contiguous Load two            LD2H                    10              1               V01, L01, S   -
structures to two vectors,
scalar + scalar
Contiguous Load two            LD2B, LD2D, LD2W        9               1               V01, L01      -
structures to two vectors,
scalar + scalar
Contiguous Load three          LD3B, LD3D, LD3H,       11              1/3             V01, L01      -
structures to three vectors,   LD3W
scalar + imm
Contiguous Load three          LD3B, LD3D, LD3H,       13              1/3             V01, L01, S   -
structures to three vectors,   LD3W
scalar + scalar
Contiguous Load four           LD4B, LD4D, LD4H        12              1/4             V01, L01      -
structures to four vectors,    LD4W
scalar + imm
Contiguous Load four           LD4B, LD4D, LD4H,       13              1/4             L01, V01, S   -
structures to four vectors,    LD4W
scalar + scalar
Gather load, vector + imm, 32- LD1B, LD1H, LD1W,       11              1/4             L, V          -
bit element size               LD1SB, LD1SH,
                               LD1SW, LDFF1B,
                               LDFF1H, LDFF1W,
                               LDFF1SB, LDFF1SH,
                               LDFF1SW
Gather load, vector + imm, 64- LD1B, LD1D, LD1H,       9               1/2             L, V          -
bit element size               LD1W, LD1SB,
                               LD1SH, LD1SW,
                               LDFF1B, LDFF1D
                               LDFF1H, LDFF1W,
                               LDFF1SB, LDFF1SD,
                               LDFF1SH, LDFF1SW
Gather load, 32-bit scaled     LD1H, LD1SH,            11              1/4             L, V          -
offset                         LDFF1H, LDFF1SH,
                               LD1W, LDFF1W,
                               LDFF1SW
Gather load, 32-bit unpacked   LD1B, LD1SB,            9               1/2             L, V          -
unscaled offset                LDFF1B, LDFF1SB,
                               LD1D, LDFF1D,
                               LD1H, LD1SH,
                               LDFF1H, LDFF1SH,
                               LD1W, LD1SW,
                               LDFF1W, LDFF1SW

                                                                                     3 Instruction characteristics

3.29 SVE Store instructions
Table 3-46 SVE Store instructions
Instruction Group               SVE Instruction        Execution       Execution       Utilized     Notes
                                                       Latency         Throughput      Pipelines
Store from predicate reg        STR                    1               2               L01          -
Store from vector reg           STR                    2               2               L01, V       -
Contiguous store, scalar + imm ST1B, ST1H, ST1D,       2               2               L01, V       -
                               ST1W
Contiguous store, scalar +      ST1H                   2               2               L01, S, V    -
scalar
Contiguous store, scalar +      ST1B, ST1D, ST1W       2               2               L01, V       -
scalar
Contiguous store two            ST2B, ST2H, ST2D,      4               1               L01, V       -
structures from two vectors,    ST2W
scalar + imm
Contiguous store two            ST2H                   4               1               L01, S, V    -
structures from two vectors,
scalar + scalar
Contiguous store two            ST2B, ST2D, ST2W       4               1               L01, V       -
structures from two vectors,
scalar + scalar
Contiguous store three         ST3B, ST3D, ST3H,       7               2/9             L01, V       -
structures from three vectors, ST3W
scalar + imm
Contiguous store three         ST3H                    7               2/9             L01, S, V    -
structures from three vectors,
scalar + scalar
Contiguous store three         ST3B, ST3D, ST3W        7               2/9             L01, S, V    -
structures from three vectors,
scalar + scalar
Contiguous store four           ST2B, ST4D, ST4H,      11              1/9             L01, V       -
structures from four vectors,   ST4W
scalar + imm
Contiguous store four           ST4H                   11              1/9             L01, S, V    -
structures from four vectors,
scalar + scalar
Contiguous store four           ST4B, ST4D, ST4W       11              1/9             L01, S, V    -
structures from four vectors,
scalar + scalar
Non temporal store, scalar +    STNT1B, STNT1D,        2               2               L01, V       -
imm                             STNT1H, STNT1W
Non temporal store, scalar +    STNT1H                 2               2               L01, S, V    -
scalar
Non temporal store, scalar +    STNT1B, STNT1D,        2               2               L01, V       -
scalar                          STNT1W

                                                                                          3 Instruction characteristics

Instruction Group                   SVE Instruction         Execution       Execution       Utilized     Notes
                                                            Latency         Throughput      Pipelines
Scatter store vector + imm 32- ST1B, ST1H, ST1W             10              1/4             L01, V       -
bit element size
Scatter store vector + imm 64- ST1B, ST1D, ST1H,            6               1/2             L01, V       -
bit element size               ST1W
Scatter store, 32-bit scaled        ST1H, ST1W              10              1/4             L01, V       -
offset
Scatter store, 32-bit unpacked ST1B, ST1D, ST1H,            6               1/2             L01, V       -
unscaled offset                ST1W
Scatter store, 32-bit unpacked ST1D, ST1H, ST1W             6               1/2             L01, V       -
scaled offset
Scatter store, 32-bit unscaled      ST1B, ST1H, ST1W        10              1/4             L01, V       -
offset
Scatter store, 64-bit scaled        ST1D, ST1H, ST1W        6               1/2             L01, V       -
offset
Scatter store, 64-bit unscaled      ST1B, ST1D, ST1H,       6               1/2             L01, V       -
offset                              ST1W

3.30 SVE Miscellaneous instructions
Table 3-47 SVE miscellaneous instructions
Instruction Group                   SVE Instruction         Execution       Execution       Utilized     Notes
                                                            Latency         Throughput      Pipelines
Read first fault register,          RDFFR                   2               1               M0           -
unpredicated
Read first fault register,          RDFFR                   3               1/2             M0           -
predicated
Read first fault register and set RDFFRS                    4               1/3             M            -
flags
Set first fault register            SETFFR                  2               1               M0           -
Write to first fault register       WRFFR                   2               1               M0           -

                                                                                      4 Special considerations

4 Special considerations
4.1 Dispatch constraints
Dispatch of µOPs from the in-order portion to the out-of-order portion of the microarchitecture
includes several constraints. It is important to consider these constraints during code generation to
maximize the effective dispatch bandwidth and subsequent execution bandwidth of Neoverse V1.

The dispatch stage can process up to 8 MOPs per cycle and dispatch up to 16 µOPs per cycle, with the
following limitations on the number of µOPs of each type that may be simultaneously dispatched.
•   Up to 4 µOPs utilizing the S or B pipelines
•   Up to 4 µOPs utilizing the M pipelines
•   Up to 2 µOPs utilizing the M0 pipelines
•   Up to 2 µOPs utilizing the V02 pipelines
•   Up to 2 µOPs utilizing the V13 pipelines
•   Up to 6 µOPs utilizing the L pipelines

In the event there are more µOPs available to be dispatched in a given cycle than can be supported by
the constraints above, µOPs will be dispatched in oldest to youngest age-order to the extent allowed
by the above.

4.2 Dispatch stall
In the event of a V-pipeline µOP containing more than 1 quad-word register source, a portion or all of
which was previously written as one or multiple single words, that µOP will stall in dispatch for three
cycles. This stall occurs only on the first such instance, and subsequent consumers of the same
register will not experience this stall.

4.3 Optimizing memory routines
To achieve maximum throughput for memory copy (or similar loops), one should do the following:
•   Unroll the loop to include multiple load and store operations per iteration, minimizing the
    overheads of looping.
•   Align stores on 32B boundary wherever possible.
•   Use non-writeback forms of LDP and STP/STR instructions interleaving them like shown in the
    examples below:
    For forward copies:
    Loop_start:
        SUBS    X2,X2,#96
        LDP     Q3,Q4,[x1,#0]

                                                                                      4 Special considerations
        STP    Q3,Q4,[x0,#0]
        LDP    Q3,Q4,[x1,#32]
        STP    Q3,Q4,[x0,#32]
        LDP    Q3,Q4,[x1,#64]
        STP    Q3,Q4,[x0,#64]
        ADD    X1,X1,#96
        ADD    X0,X0,#96
        BGT    Loop_start

    For backward copies:
    Loop_start:
        SUBS   x2,x2,#96
        LDP    q4,q3,[x1,#-32]
        STR    q3,[x0,#-16]
       STR     q4,[x0,#-32]
        LDP    q4,q3,[x1,#-64]
        STR    q3,[x0,#-48]
        STR    q4,[x0,#-64]
        LDP    q4,q3,[x1,#-96]
        STP    q3,[x0,#-80]
       STR     q4,[x0,#-96]
        SUB    x1,x1,#96
        SUB    x0,x0,#96
        BGT    Loop_start

A recommended copy routine for AArch32 would look like the sequence above but would use
LDRD/STRD instructions. Avoid load-/store-multiple instruction encodings (such as LDM and STM).

It should be noted that the atomicity requirements introduced in the Armv8.4-A extension cause
memory routines that use LDP X to perform worse on Neoverse-V1 when compared to other ARM
v8.2-A cores.

To achieve maximum throughput on memset, it is recommended that one do the following:
•   Unroll the loop to include multiple load and store operations per iteration, minimizing the
    overheads of looping.
    Loop_start:
       STP        q1,q3,[x0,#0]
       STP        q1,q3,[x0,#0x20]
       STP        q1,q3,[x0,#0x40]
       STP        q1,q3,[x0,#0x60]
       ADD        x0,x0,#0x80
       SUBS       x2,x2,#0x80

                                                                                       4 Special considerations
          B.GT      Loop_start

To achieve maximum performance on memset to zero, it is recommended that one use DC ZVA
instead of STP. An optimal routine might look something like the following:
Loop_start:
    SUBS         x2,x2,#0x80
    DC           ZVA,x0
    ADD          x0,x0,#0x40
    DC           ZVA,x0
    ADD          x0,x0,#0x40
    B.GT         Loop_start

4.4 Load/Store alignment
The Armv8-A architecture allows many types of load and store accesses to be arbitrarily aligned. The
Neoverse V1 core handles most unaligned accesses without performance penalties. However, there
are cases which could reduce bandwidth or incur additional latency, as described below:
•   Load operations that cross a cache-line (64-byte) boundary
•   Quad-word load operations that are not 4B aligned
•   Store operations that cross a 32B boundary

4.5 Store to Load Forwarding
The Neoverse V1 core allows data to be forwarded from store instructions to a load instruction with
the restrictions mentioned below:

•   Load start address should align with the start or middle address of the older store. This does not
    apply to LDPs that load 2 32b registers or LDRDs

•   Loads of size greater than 8 bytes can get the data forwarded from a maximum of 2 stores. If
    there are 2 stores, then each store should forward to either first or second half of the load

•   Loads of size less than or equal to 8 bytes can get their data forwarded from only 1 store

                                                                                      4 Special considerations

4.6 AES encryption/decryption
In AArch64 mode, Neoverse V1 can issue four AESE/AESMC/AESD/AESIMC instructions every
cycle (fully pipelined) with an execution latency of two cycles. Note that pairs of dependent
AESE/AESMC and AESD/AESIMC instructions are higher performance when they are adjacent in the
program code and both instructions use the same destination register since they are fused (see
Section 4.13 on Instruction Fusion). This means encryption or decryption for at least eight data
chunks should be interleaved for maximum performance: In AArch32 mode, the issue rate drops to
two per cycle. However, the code below is optimal in either mode.

AESE   data0, key0
AESMC data0, data0
AESE   data1, key0
AESMC data1, data1
AESE   data2, key0
AESMC data2, data2
AESE   data3, key0
AESMC data3, data3
AESE   data4, key0
AESMC data4, data4
AESE   data5, key0
AESMC data5, data5
AESE   data6, key0
AESMC data6, data6
AESE   data7, key0
AESMC data7, data7
AESE   data0, key1
AESMC data0, data0
...

                                                                                         4 Special considerations

4.7 Region based fast forwarding
The forwarding logic in the V pipelines is optimized to provide optimal latency for instructions which
are expected to commonly forward to one another. The effective latency of FP and ASIMD
instructions as described in section 3 is increased by one cycle if the producer and consumer
instructions are not part of the same forwarding region. These optimized forwarding regions are
defined in the following table.

Table 4-1 Optimized forwarding regions
Region         Instruction Types                                                                 Notes
1              ASIMD/SVE integer ALU, ASIMD/SVE integer shift, ASIMD/scalar insert and move,     1
               ASIMD/SVE integer abs/cmp/max/min and the ASIMD miscellaneous instructions in
               tables 3-31 and 3-32.
2              FP/ASIMD/SVE floating-point multiply, FP/ASIMD/SVE floating point multiply-       1,2,3
               accumulate, FP/ASIMD/SVE compare, FP/ASIMD/SVE add/sub and the ASIMD
               miscellaneous instructions in tables 3-31 and 3-32.
3              ASIMD/SVE Crypto and SHA1/SHA256                                                  -
4              ASIMD/SVE AES, ASIMD/SVE polynomial multiply and all the instruction types in     1
               region 1.
5              ASIMD/SVE BFDOT and BFMMLA instructions                                           -

Notes:
    1.   Reciprocal step and estimate instructions are excluded from this region.
    2.   ASIMD extract narrow, saturating instructions are excluded from this region.
    3.   ASIMD miscellaneous instructions can only be consumers of this region.

The following instructions are not a part of any region:
•   FP/ASIMD/SVE floating-point div/sqrt and SVE integer divides
•   FP/ASIMD/SVE convert and rounding instructions that do not write to general purpose registers
•   ASIMD/SVE integer mul/mac
•   ASIMD/SVE integer reduction
In addition to the regions mentioned in the table above, all instructions in regions 1 and 2 can fast
forward to FP/ASIMD/SVE stores, FP/ASIMD vector to integer register transfers and ASIMD
converts that write to general purpose registers.

More special notes about the forwarding region in table 4-1:
•   Fast forwarding will not occur in AArch32 mode if the consuming register’s width is greater than
    that of the producer.
•   Element sources (the non-vector operand in "by element" multiplies) used by ASIMD/SVE
    floating-point multiply and multiply-accumulate operations cannot be consumers.

                                                                                      4 Special considerations

•   Complex shift by immediate/register and shift accumulate instructions cannot be producers (see
    sections 3.16 and 3.25) in region 1.
•   Extract narrow, saturating instructions cannot be producers (see sections 3.19 and 3.25) in
    region 1.
•   Absolute difference accumulate and pairwise add and accumulate instructions cannot be
    producers (see sections 3.16 and 3.25) in region 1.
•   For floating-point producer-consumer pairs, the precision of the instructions should match
    (single, double or half) in region 2.
•   Pair-wise floating-point instructions cannot be producers or consumers in region 2.

It is not advisable to interleave instructions belonging to different regions. Also, certain instructions
can only be producers or consumers in a particular region but not both (see footnote 3 for table 4-1).
For example, the code below interleaves producers and consumers from regions 1 and 2. This will
result in and additional latency of 1 cycle as seen by FMUL.
      FSUB v27.2s, v28.2s, v20.2s – Region 2
      FADD v20.2s, v28.2s, v20.2s – Region 2
      MOV v27.s[1], v20.s[1] - Region 2 producer but not a region 2 consumer
      FMUL v26.2s, v27.2s, v6.2s – Region 2

4.8 Branch instruction alignment
Branch instruction and branch target instruction alignment and density can affect performance.

For best case performance, avoid placing more than four branch instructions within an aligned 32-
byte instruction memory region.

4.9 FPCR self-synchronization
Programmers and compiler writers should note that writes to the FPCR register are self-
synchronizing, i.e. its effect on subsequent instructions can be relied upon without an intervening
context synchronizing operation.

4.10 Special register access
The Neoverse V1 core performs register renaming for general purpose registers to enable
speculative and out-of-order instruction execution. But most special-purpose registers are not
renamed. Instructions that read or write non-renamed registers are subjected to one or more of the
following additional execution constraints:
•   Non-Speculative Execution – Instructions may only execute non-speculatively.

                                                                                                4 Special considerations

•   In-Order Execution – Instructions must execute in-order with respect to other similar
    instructions or in some cases all instructions.
•   Flush Side-Effects – Instructions trigger a flush side-effect after executing for synchronization.

The table below summarizes various special-purpose register read accesses and the associated
execution constraints or side-effects.

Table 4-2 Special-purpose register read accesses
Register Read                Non-Speculative                           In-Order           Flush Side-Effect     Notes
APSR                         Yes                                       Yes                No                    3
CurrentEL                    No                                        Yes                No                    -
DAIF                         No                                        Yes                No                    -
DLR_EL0                      No                                        Yes                No                    -
DSPSR_EL0                    No                                        Yes                No                    -
ELR_*                        No                                        Yes                No                    -
FPCR                         No                                        Yes                No                    -
FPSCR                        Yes                                       Yes                No                    2
FPSR                         Yes                                       Yes                No                    2
NZCV                         No                                        No                 No                    1
SP_*                         No                                        No                 No                    1
SPSel                        No                                        Yes                No                    -
SPSR_*                       No                                        Yes                No                    -
FFR                          No                                        Yes                No                    -
Notes:
    1.   The NZCV and SP registers are fully renamed.
    2.   FPSR/FPSCR reads must wait for all prior instructions that may update the status flags to execute and retire.
    3.   APSR reads must wait for all prior instructions that may set the Q bit to execute and retire.

The table below summarizes various special-purpose register write accesses and the associated
execution constraints or side-effects.

Table 4-3 Special-purpose register write accesses
Register Write               Non-Speculative                           In-Order           Flush Side-Effect     Notes
APSR                         Yes                                       Yes                No                    4
DAIF                         Yes                                       Yes                No                    -
DLR_EL0                      Yes                                       Yes                No                    -
DSPSR_EL0                    Yes                                       Yes                No                    -
ELR_*                        Yes                                       Yes                No                    -
FPCR                         Yes                                       Yes                Maybe                 2
FPSCR                        Yes                                       Yes                Maybe                 2, 3
FPSR                         Yes                                       Yes                No                    3
                                     ©
                         Copyright 2025 Arm Limited (or its affiliates). All rights reserved.

                                                                                               4 Special considerations

Register Write               Non-Speculative                          In-Order           Flush Side-Effect       Notes
NZCV                         No                                       No                 No                      1
SP_*                         No                                       No                 No                      1
SPSel                        Yes                                      Yes                Yes                     -
SPSR_*                       Yes                                      Yes                No                      -
FFR                          Yes                                      Yes                No                      -

Notes:
    1.   The NZCV and SP registers are fully renamed.
    2.   If the FPCR/FPSCR write is predicted to change the control field values, it will introduce a barrier which prevents
         subsequent instructions from executing. If the FPCR/FPSCR write is predicted to not change the control field
         values, it will execute without a barrier but trigger a flush if the values change.
    3.   FPSR/FPSCR writes must stall at dispatch if another FPSR/FPSCR write is still pending.
    4.   APSR writes that set the Q bit will introduce a barrier which prevents subsequent instructions from executing until
         the write completes.

4.11 Register forwarding hazards
The Armv8-A architecture allows FP/ASIMD instructions to read and write 32-bit S-registers. In
AArch32, each S-register corresponds to one half (upper or lower) of an overlaid 64-bit D-register. A
Q-register in turn consists of two overlaid D-register. Register forwarding hazards may occur when
one µOP reads a Q-register operand that has recently been written with one or more S-register
result. Consider the following scenario.
VADD     S0, S1, S2
VADD     Q6, Q5, Q0

The first instruction writes S0, which corresponds to the lowest part of Q0. The second instruction
then requires Q0 as an input operand. In this scenario, there is a RAW dependency between the first
and the second instructions. In most cases, Neoverse V1 performs slightly worse in such situations.

Neoverse V1 is able to avoid this register-hazard condition for certain cases. The following rules
describe the conditions under which a register-hazard can occur:
•   The producer writes an S-register (not a D[x] scalar)
•   The consumer reads an overlapping Q-register (not as a D[x] scalar)
•   The consumer is a FP/ASIMD µOP (not a store or MOV µOP)

To avoid unnecessary hazards, it is recommended that the programmer use D[x] scalar writes when
populating registers prior to ASIMD operations. For example, either of the following instruction
forms would safely prevent a subsequent hazard.
VLD1.32      D0[x], [address]
VADD         Q1, Q0, Q2F

                                                                                      4 Special considerations

4.12 IT blocks
The Armv8-A architecture performance deprecates some uses of the IT instruction in such a way that
software may be written using multiple naïve single instruction IT blocks. It is preferred that software
instead generate multi instruction IT blocks rather than single instruction blocks.

4.13 Instruction fusion
Neoverse V1 can accelerate certain instruction pairs in an operation called fusion. Specific Aarch64
instruction pairs that can be fused are as follows:
1. CMP/CMN (immediate) + B.cond
2. CMP/CMN (register) + B.cond
3. CMP (immediate) + CSEL
4. CMP (register) + CSEL
5. CMP (immediate) + CSET
6. CMP (register) + CSET
7. TST (immediate) + B.cond
8. TST (register) + B.cond
9. BICS (register) + B.cond
10. NOP + Any instruction

The following instruction pairs are fused in both Aarch32 and Aarch64 modes:
1. AESE + AESMC (see Section 4.6 on AES Encryption/Decryption)
2. AESD + AESIMC (see Section 4.6 on AES Encryption/Decryption)

These instruction pairs must be adjacent to each other in program code. For CMP, CMN, TST and
BICS, fusion is not allowed for shifted and/or extended register forms. For BICS, the destination
register should be XZR or WZR if fusion is to take place.

                                                                                     4 Special considerations

4.14 Zero Latency MOVs
A subset of register-to-register move operations and move immediate operations are executed with
zero latency. These instructions do not utilize the scheduling and execution resources of the machine.
These are as follows:

MOV Xd, #0

MOV Xd, XZR

MOV Wd, #0

MOV Wd, WZR

MOV Rd, #0 (AArch32)

MOV Wd, Wn

MOV Xd, Xn

MOV Rd, Rn (AArch32)

The last 3 instructions may not be executed with zero latency under certain conditions.

4.15 Cache maintenance operations
While using set way invalidation operations on L1 cache, it is recommended that software be written
to traverse the sets in the inner loop and ways in the out loop.

4.16 Mixing ASIMD and SVE instructions
Maximum issue bandwidth is sustained using one of the following combinations:
•   2 SVE Uops.
•   4 ASIMD Uops.
•   1 SVE Uop on V0 and 2 ASIMD Uops on VX13.
•   1 SVE Uop on V1 and 2 ASIMD Uops on V02.

                                                                                     4 Special considerations

4.17 Complex ASIMD and SVE instructions
The bandwidth of the following ASIMD and SVE instructions is limited by decode constraints and it is
advisable to avoid them when high performing code is desired.

ASIMD
1. LD4R, post-indexed addressing, element size = 64b.
2. LD4, single 4-element structure, post indexed addressing mode, element size = 64b.
3. LD4, multiple 4-element structures, quad form.
4. LD4, multiple structures, double word form.
5. ST4, multiple 4-element structures, quad form, element size less than 64b.
6. ST4, multiple 4-element structures, quad form, element size = 64b, post indexed addressing
   mode.

SVE
1. LD1B gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b unscaled offset.
2. LD1H gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b scaled or unscaled offset.
3. LD1W gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b scaled or unscaled offset.
4. LD3[B/H/W/D] contiguous (scalar + scalar addressing).
5. LD4[B/H/D/W] contiguous (scalar + immediate addressing).
6. LD4[B/H/D/W] contiguous (scalar + scalar addressing).
7. LDFF1B gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b unscaled offset.
8. LDFF1H gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b scaled or unscaled offset.
9. LDFF1W gather (scalar + vector addressing) where vector index register is the same as the
   destination register and element size = 32. Addressing mode is 32b scaled or unscaled offset.
10. ST3[B/H/W/D] contiguous (scalar + scalar addressing).
11. ST4[B/H/D/W] contiguous (scalar + immediate addressing).
12. ST4[B/H/D/W] contiguous (scalar + scalar addressing).

4.18 MOVPRFX fusion
Under certain conditions, a mechanism called MOVPRFX fusion can be used to accelerate the
execution of an instruction pair that consists of an SVE MOVPRFX instruction immediately followed
in program order by an SVE integer, floating point or BF16 instruction. The list of SVE instructions
and the conditions under which tis fusion can be applied is mentioned in the tables below.

                                                                                            4 Special considerations

Instruction Group                    SVE Instruction                            Notes
                                                    Integer Instructions
Arithmetic, basic                    ABS, ADD, CNOT, NEG, SABD, SQADD,          -
                                     SQSUB, SUB, SUBR, UABD, UQADD,
                                     UQSUB
Arithmetic, shift left               LSL, LSLR                                  -
Arithmetic, shift right              ASR, LSR                                   Only the immediate form is fusible.
Arithmetic, shift right for divide   ASRD                                       -
Count/reverse bits                   CLS, CLZ, CNT, RBIT                        -
Conditional extract operations       CLASTA, CLASTB, SPLICE                     -
Convert to floating point            SCVTF, UCVTF                               Only conversions where
                                                                                destination element size is greater
                                                                                than or equal to source element
                                                                                size are fusible.
Copy                                 CPY                                        Scalar form is not fusible.
Divides                              SDIV, SDIVR, UDIV, UDIVR                   -
Dot product                          SDOT, UDOT, USDOT, SUDOT                   -
Extend, sign or zero                 SXTB, SXTH, SXTW, UXTB, UXTH, UXTW         -
Extract/insert operation             EXT, INSR                                  Only SIMD and FP scalar form of
                                                                                INSR is fusible.
Logical                              AND, BIC, EON, EOR, NOT, ORN, ORR          -
Max/min, basic and pairwise          SMAX, SMIN, SMINP, UMAX, UMIN              -
Multiply                             MUL, SMULH, UMULH                          -
Matrix multiple accumulate           SMMLA, UMMLA, USMMLA                       -
Predicate counting, vector form      DECH, DECW, DECD, INCH, INCW, INCD,        -
                                     SQDECH, SQDECW, SQDECD, SQINCH,
                                     SQINCW, SQINCD, UQDECH, UQDECW,
                                     UQDECD, UQINCH, UQINCW, UQINCD
Reverse, vector                      REV, REVB, REVH, REVW                      -
                                                 Floating point Instructions
Floating point absolute              FABD, FABS                                 -
value/difference
Floating point arithmetic            FADD, FNEG, FSUB, FSUBR                    -
Floating point complex add           FCADD                                      -
Floating point convert               FCVT                                       Only conversions to a higher
                                                                                precision are fusible.
Floating point convert to integer FCVTZS, FCVTZU                                Only conversions where
                                                                                destination element size is greater
                                                                                than or equal to source element
                                                                                size are fusible.
Floating point copy                  FCPY, FMOV                                 -

                                                                                           4 Special considerations

Instruction Group                  SVE Instruction                              Notes
Floating point divide              FDIV, FDIVR                                  -
Floating point min/max             FMAX, FMIN, FMAXNM, FMINNM                   -
Floating point multiply            FMUL, FMULX                                  -
Floating point multiply            FMLA, FMLS                                   Only indexed forms are fusible.
Floating point complex multiple    FCMLA                                        Only indexed forms are fusible.
add
Floating point reciprocal          FRECPX                                       -
estimate
Floating point round to integral   FRINTA, FRINTI, FRINTM, FRINTN, FRINTP, -
                                   FRINTX, FRINTZ
Floating point square root         FSQRT                                        -
Floating point trigonometric       FTMAD                                        -
multiply add
                                                 BF16 Instructions
Dot product                        BFDOT                                        -
Matrix multiply accumulate         BFMMLA                                       -
Multiply accumulate long           BFMLALB, BFMLALT                             -