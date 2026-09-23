# SME/SME2 Architecture and State Notes

This is an agent-facing summary of the concepts needed to write correct SME code. It is not a complete ISA or ABI description. Consult current Arm specifications before relying on instruction details.

## Feature and dispatch

- SME is an optional Arm architecture extension; SME2 builds on SME. Do not infer support from an Armv9 label alone.
- The guide describes `ID_AA64PFR1_EL1.SME` as indicating SME, and SME2 through the SME field plus `ID_AA64SMFR0_EL1.SMEver`. Applications may not be able to read these system registers directly; OS interfaces such as Linux HWCAP may be the supported discovery route.
- Resolve feature discovery through the OS/runtime interface for the target environment. Keep an implementation fallback and dispatch only to instructions supported by the running CPU and OS context-management path.
- SME/SME2 variant features also matter. A kernel using optional data-type variants must check those exact features; a generic SME2 check is not enough.

## Streaming mode, ZA, and SVL

- SME adds streaming SVE mode and ZA matrix storage. `PSTATE.SM` controls streaming mode; `PSTATE.ZA` controls whether ZA is enabled. Software changes these through the architectural mechanisms such as `SMSTART`/`SMSTOP` and `SVCR` operations; do not treat them as ordinary application variables.
- `SMSTART SM` enters streaming mode without enabling ZA. `SMSTART ZA` enables ZA without entering streaming mode. The combined `SMSTART` form enables both. Corresponding stop forms clear state as specified by the architecture.
- Streaming vector length (`SVL`) is the vector length while in streaming mode. It is implementation/configuration dependent and independent of the non-streaming SVE vector length (`VL`). The guide describes supported effective SVLs as powers of two from 128 to 2048 bits; portable code must still query/use the effective runtime value and must not bake in a sample such as 128 bits.
- ZA is a square scalable array. It can be viewed as array vectors, element-sized tiles, and horizontal/vertical tile slices. The number of element lanes in one vector is `SVL / element_bits`; tile indices and slice relationships depend on element size and SVL. Treat diagrams or sample tile layouts as examples, not fixed dimensions.
- Entering or leaving streaming mode changes the streaming register context. The guide states that Z0–Z31 and P0–P15 in the newly selected mode are zeroed on transitions. Code must follow the ABI/compiler rules for preserving any non-streaming state it needs.
- ZA contents can be live across calls under the ABI's cooperative/lazy-save model. Do not casually clear, overwrite, or assume-zero ZA. In assembly, implement the current AAPCS64 SME rules, including lazy-save handling when required. In C/C++, prefer compiler-managed interfaces instead of copying an old prologue/epilogue from an example.
- SME2 also introduces ZT0. Treat its state and preservation/context-switch requirements according to the current ABI and operating system.

## ZA shape and precision map

The architectural ZA storage is reinterpreted according to the selected element width. The guide illustrates a single B tile, two H tiles, four S tiles, eight D tiles, and sixteen Q tiles. Each tile's lane dimensions scale with SVL. This is useful for understanding accumulator capacity, but code should use architecture-defined operand forms and runtime SVL-aware indexing rather than constants copied from an illustration.

An outer-product instruction conceptually combines a vector of row values and a vector of column values, then accumulates products into a ZA tile. Widening forms produce more/larger destination elements than the source element width. The accumulator type, signedness, saturation/rounding behavior, and required SME variant must match the algorithm exactly.

## SME2 operand and predication model

- SME2 can use groups of Z registers (commonly groups of 1, 2, or 4), groups of ZA tile slices, and groups of ZA array vectors (including larger groups). The legal grouping, register sequence, arrangement, and instruction-specific restrictions vary by instruction.
- Multi-vector operations can improve work per instruction and express interleaved/widening computations, but they increase register/data-layout constraints. Validate every operand group against the instruction reference.
- SME2 predicate-as-counter is distinct from ordinary SVE bitmask predication. It encodes a count of consecutive active or inactive elements for multi-vector operations. Do not pass a normal predicate mask where a predicate-as-counter form is required; use the relevant ACLE/compiler mechanism or construct it exactly per the specification.

## Function boundaries and ABI

- Use a current AAPCS64 release with SME support. Streaming-compatible, streaming, and non-streaming function types have distinct boundary behavior. In C/C++, use the compiler's supported ACLE function attributes and state-management facilities so mode transitions and state preservation are compiler-managed.
- Hand-written assembly functions must preserve the registers and SME state required by the active ABI, prepare for entry/exit, maintain stack and unwind conventions, and handle ZA lazy-save ownership where required. The example macros in the guide illustrate concepts, not drop-in universal prologues.
- Inline assembly must accurately declare clobbers and state effects to the compiler. Do not hide mode changes or ZA use from the compiler.
- Do not use SME instructions from a signal/interrupt/context where the OS or ABI does not permit them. Confirm OS support for SME context management before deployment.

## Toolchain notes

The guide's release snapshot says Arm Compiler for Embedded 6, Clang, and GCC support SME/SME2 to differing degrees; its listed minimum versions and ACLE support reflect the guide's publication date and can age. Check the current compiler release notes and ACLE documentation. For native assembly, enable the needed `-march`/`-mcpu` feature (for example an SME or SME2 feature where accepted by that compiler); add optional variant features only when needed. Validate assembler, linker, runtime library, OS, and target support as one chain.

The guide uses `__ARM_FEATURE_SME` and `<arm_sme.h>` as ACLE examples. Check current macro/header names and intrinsic availability for the exact compiler. Do not present one compiler's option syntax as universal.

## Correctness checklist

- [ ] Runtime dispatch verifies SME, SME2, and any required variants.
- [ ] Every tile size, loop increment, and predicate derives from runtime SVL or a proven invariant.
- [ ] SM/ZA state entry, exit, and function-boundary behavior follow current ACLE/AAPCS64 requirements.
- [ ] ZA initialization and accumulation lifetime are explicit.
- [ ] Tail rows/columns/K elements are masked or padded with mathematically safe values.
- [ ] Loads/stores respect bounds, alignment, strides, and aliasing rules.
- [ ] Accumulator width and overflow/rounding behavior match the reference operation.
- [ ] Non-SME fallback runs on machines without the required feature.
