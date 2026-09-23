# Arm SME/SME2 Agent Reference

Use this set when designing or tuning code for Arm's Scalable Matrix Extensions.
It is a decision aid for an implementation agent, not an instruction-set specification.

## Reading order

1. This page: scope, routing, and design loop.
2. [`architecture-and-state.md`](architecture-and-state.md): feature detection, streaming mode, ZA, SVL, ABI, and toolchain constraints.
3. [`optimization-patterns.md`](optimization-patterns.md): workload-to-instruction mapping, data-layout strategies, and worked algorithm patterns.

## Agent workflow

1. **Establish the target.** Identify the CPU/SoC, OS and runtime, compiler and version, target architecture flags, available SME/SME2 features, and whether the code is portable across different streaming vector lengths. If the deployment target is unknown, design a safe dispatch/fallback plan rather than assuming SME is present.
2. **Characterize the workload.** Record operation, dimensions, data types and signedness, layouts/strides, alignment, tail sizes, accumulation and rounding requirements, reuse, and expected matrix sizes. Estimate whether packing/preprocessing cost will be amortized.
3. **Choose the smallest fitting path.** Prefer ordinary scalar/SIMD/SVE for small or unsupported cases. Use SME outer products for matrix accumulation. Consider SME2 multi-vector and lookup-table operations only when the data shape and target feature set justify them.
4. **Design data movement with the compute kernel.** Specify input packing, padding values, tile blocking, ZA accumulation ownership, tails, and output extraction together. Validate all memory bounds and mathematical semantics before optimizing instruction count.
5. **Respect architectural and ABI state.** Read [`architecture-and-state.md`](architecture-and-state.md) before writing mode-changing code or crossing a function boundary. Prefer compiler-managed ACLE attributes and intrinsics where supported; hand-written assembly must satisfy the active AAPCS64 rules.
6. **Build and inspect.** Compile for the exact target with SME/SME2 explicitly enabled. Inspect diagnostics and disassembly to confirm the intended instructions are emitted; feature flags alone do not guarantee that the compiler selected a particular kernel.
7. **Validate correctness.** Compare against a trusted scalar/reference implementation for varied dimensions, tails, strides, signed/unsigned extremes, and floating-point tolerances. Test the minimum and maximum supported SVL when available. Check for unsupported-instruction dispatch and a non-SME fallback.
8. **Measure before claiming a speedup.** Benchmark end-to-end and kernel-only paths separately, including packing, memory traffic, launch/call overhead, and representative sizes. Use repeatable target hardware measurements. Report compiler, flags, CPU, SVL, shapes, baseline, and timing method.

## Fast routing

| Need | Read / investigate |
|---|---|
| Determine whether SME or SME2 is available | `architecture-and-state.md` → Feature and dispatch |
| Enter/leave streaming mode or use ZA safely | `architecture-and-state.md` → State model and ABI |
| Pick an instruction family for a workload | `optimization-patterns.md` → Workload map |
| Design packing, blocking, or tail handling | `optimization-patterns.md` → Data preparation and kernel structure |
| Check ACLE/compiler/assembly support | `architecture-and-state.md` → Toolchain notes; then current compiler docs |
| Confirm exact instruction operands/semantics | Current Arm Architecture Reference Manual and SME/SME2 ACLE specification |

## Source and limits

This reference summarizes Arm Ltd., *SME Programmer's Guide*, document 109246_0101_01_en,
Version 1.1, Issue 0101-01 (9 October 2025), especially chapters 2–9. It is an original
agent-oriented summary, not a reproduction of the guide. Some toolchain support statements
in the guide are release-specific and may now be stale. Verify all current support, feature
discovery APIs, exact instruction forms, and ABI requirements in the authoritative documents
for the actual target and toolchain. The Arm guide itself notes that software implementations
must provide appropriate correctness, security, and safety safeguards.
