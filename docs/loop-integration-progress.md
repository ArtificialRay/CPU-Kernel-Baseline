# SIMD-Loop Integration Progress

Persistent tracker for integrating the remaining simd-loop problems into the
`bench/` harness. Updated as work completes so progress survives context/credit
loss. See `CLAUDE.md` "Adding a new simd-loop problem" and
`scripts/gen_simd_loop_harness.py` for the mechanics.

**Status legend:** ✅ done · 🔵 in progress · ⬜ todo · ⛔ excluded

## Baseline

- **Integrated before this effort (23):** 001-004, 008, 010, 024, 027-029, 032,
  033, 035, 108, 113, 120-124, 126-128 — passing 140/140 workloads.
- **Excluded — true SME2 / MOPA matmul (13):** 201, 202, 204, 205, 206, 207,
  208, 210, 211, 212, 215, 231, 245. No AWS hardware; revisit when SME2 exists.
  These reuse the multi-axis (Cap D) infrastructure later.
- **In scope: 39 loops.**

## Prerequisite work (do before adding loops)

| # | Task | Status |
|---|------|--------|
| P1 | Rebase `feat/simd-loop-sort-123-124` onto origin/main (PR #22 + #24) | ✅ |
| P2 | Regenerate 23 loops (new `inputs` format) + confirm 140/140 | ✅ |

**Weak-coverage follow-ups** (pass but degenerate under uuid-random `[1,100]`/`uniform(-1,1)`):
- loop_027 fp32 sqrt — negative inputs → NaN on both sides (passes trivially).
- loop_108 rgba — high bytes zero under `[1,100]` → all-zero output both sides.
Both need a per-input value-range override mechanism (future small Cap).

PR #24 notes: workload format is now `inputs: {name:{"type":"random"|"scalar"}}`;
`inputs.py` uses uuid-seeded random (int `[1,100]`, float `uniform(-1,1)`); the
old uint32 LCG ramp is gone; `simd_loop.py` adapter (scratch/`array_pad`)
untouched. `bench-trace/` is gitignored → regenerate, don't migrate.

## Capabilities (build once, then loops within are mechanical)

Capabilities are **sequential**; loops within a capability are **parallelizable**
once the capability lands (fan out to subagents).

### Cap A — output-override + fixed-size aux input + multi-output + scalar params
Each loop here needs a **different** generator capability, so this is NOT a
homogeneous fan-out batch — each is bespoke infra. Heterogeneous; do serially.

| Loop | Description | Capability needed | Status |
|------|-------------|-------------------|--------|
| 105 | cascade summation | none — fit existing scalar pattern (`b`=scratch), custom ref + custom scalar kernel | ✅ |
| 106 | sheep-and-goats partition | output = MIDDLE ptr `b` + fixed-size-5 `perm` input | ⬜ |
| 101 | upscale filter | output `a` is FIRST ptr, size ~2N | ⬜ |
| 111 | fp64 overflow | two outputs (`output` + `exponent`) | ⬜ |
| 114 | auto-correlation | out size = `lags`; extra scalars `n,lags,scale` | ⬜ |

**Note:** loop_105 was the ONLY remaining loop that fits an existing pattern with
just a custom ref. All other in-scope loops need a capability built first — there
is no zero-infra parallel batch. Fan-out must follow each capability landing.

### Cap B — sentinel begin/end ABI + string/buffer input generator
Proof loop: **031**. Then 005, 006, 034, 022, 103, 026.

| Loop | Description | Status |
|------|-------------|--------|
| 031 | inline memcpy (two buffers) | ⬜ |
| 005 | strlen short strings | ⬜ |
| 006 | strlen long strings | ⬜ |
| 034 | short string compares | ⬜ |
| 022 | tcp checksum (packet-length buffers) | ⬜ |
| 103 | whitespace scan | ⬜ |
| 026 | utf-16 → ascii | ⬜ |

### Cap C — non-N output sizing + extra scalar params (partly from Cap A)
| Loop | Description | Status |
|------|-------------|--------|
| 102 | general histogram | ⬜ |
| 104 | byte histogram | ⬜ |

### Cap D — custom C types (interleaved complex first)
| Loop | Description | Status |
|------|-------------|--------|
| 037 | cfloat32 complex vector product | ⬜ |
| 109 | cuint32 complex addition | ⬜ |
| 110 | cint8/cint32 complex dot | ⬜ |
| 112 | cuint32 complex MAC | ⬜ |
| 107 | uint128→uint256 multiply | ⬜ |

### Cap E — multi-axis (m/n/k); SVE2-feasible, no SME needed
Proof loop: **223** or **220**. Then GEMV → matmul → conv.

| Loop | Description | Status |
|------|-------------|--------|
| 223 | matrix transposition (m,n) | ⬜ |
| 216 | fp32 col-major GEMV | ⬜ |
| 217 | int8 row-major GEMV | ⬜ |
| 218 | fp64 col-major GEMV | ⬜ |
| 219 | int8 col-major GEMV | ⬜ |
| 220 | fp32 row-major GEMV | ⬜ |
| 221 | fp64 row-major GEMV | ⬜ |
| 025 | fp32 small fixed-size matmul | ⬜ |
| 130 | fp32 matmul (m,n,k) | ⬜ |
| 135 | int8→int32 matmul (SDOT) | ⬜ |
| 136 | int4→int32 matmul (LUT) | ⬜ |
| 137 | bf16→fp32 matmul | ⬜ |
| 038 | fp16 1D convolution | ⬜ |
| 222 | fp16 convolution (multi-vector) | ⬜ |

### Deferred — bespoke pointer-structure / low ROI
| Loop | Reason | Status |
|------|--------|--------|
| 009 | linked-list pointer chasing (node_t) | ⬜ |
| 019 | scatter writes (object_t + indexes) | ⬜ |
| 012 | particle motion (multi-array struct + fixed arrays) | ⬜ |
| 023 | conjugate gradient (sparse indirect, OOB indices) | ⬜ |
| 036 | sparse gauss step (many indirect ptrs) | ⬜ |
| 040 | clamp — scalar-only struct, no array ptr | ⬜ |

## Change log
<!-- append one line per completed loop/capability: date — what — validation result -->
- 2026-06-15 — P1 rebase onto origin/main (PR #24); resolved inputs.py conflict (dropped legacy LCG ramp, #24's uuid-random wins).
- 2026-06-15 — P2 regenerated 23 loops to new `inputs` format; test_reference_scalars 140/140 passing.
- 2026-06-15 — loop_105 integrated (custom ref + custom scalar kernel; `b` treated as scratch input). 147/147 across 24 loops. Confirmed 105 is the only zero-infra win.
