# SIMD-Loop Integration Progress

Persistent tracker for integrating the remaining simd-loop problems into the
`bench/` harness. Updated as work completes so progress survives context/credit
loss. See `CLAUDE.md` "Adding a new simd-loop problem" and
`scripts/gen_simd_loop_harness.py` for the mechanics.

**Status legend:** ✅ done · 🔵 in progress · ⬜ todo · ⛔ excluded

## Current state (resume here)

- **Integrated this effort (14):** 105, 101, 114, 106 (Cap A); 223, 216, 217, 218,
  219, 220, 221, 038, 130, 135 (Cap E). Total harness loops: **37 — 223/223 workloads passing.**
- **Cap A DONE** except 111 (two outputs — needs evaluator multi-output support).
- **Cap E DONE** except 4 documented defers (025 no-scalar-impl, 136 int4/LUT,
  137 bf16, 222 scratch+asm).
- **Next, in order:** Cap B (strings — see `## Cap B handoff` at bottom) →
  Cap C (histograms) → Cap D (complex). Each: build the shared capability, then
  fan the batch out to subagents.
- **Subagent fan-out lesson:** worktree isolation branches from `origin/main`,
  NOT the branch tip — tell each subagent to fast-forward/reset to
  `feat/simd-loop-sort-123-124` (so it has the latest infra) before starting,
  or it will rebuild parallel infra under different names.

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
| 106 | sheep-and-goats partition | output-override (middle ptr `b`); perm baked as const | ✅ |
| 101 | upscale filter | output-override + derived axis (output 2*(N-1)) | ✅ |
| 111 | fp64 overflow | ⬜ defer: TWO outputs — needs evaluator multi-output support |
| 114 | auto-correlation | out-only axis (lags, from out_shape) + const scalar (scale) | ✅ |

**Note:** loop_105 was the ONLY remaining loop that fits an existing pattern with
just a custom ref. All other in-scope loops need a capability built first — there
is no zero-infra parallel batch. Fan-out must follow each capability landing.

### Cap B — sentinel begin/end ABI + string/buffer input generator
**Integrable: 005, 006, 034, 103, 031** (real scalar impls). **Exclude 022, 026**
— their HAVE_AUTOVEC block is `ABORT: No implementations` (no scalar reference,
like 025). Verify before excluding. Full handoff prompt: `## Cap B handoff` at the
bottom of this file.

| Loop | Description | Status |
|------|-------------|--------|
| 031 | chunked memcpy (two buffers, fixed `count[]` table, output = 10× copies) | ⬜ |
| 005 | strlen short strings (p/lmt sentinel; out=checksum=Σ lengths) | ⬜ |
| 006 | strlen long strings (same ABI as 005) | ⬜ |
| 034 | short string compares (a/b/lmt; out=checksum) | ⬜ |
| 103 | whitespace word-count (p/end; skip_whitespace/skip_word; out=count) | ⬜ |
| 022 | tcp checksum | ⛔ HAVE_AUTOVEC = ABORT (no scalar impl) |
| 026 | utf-16 → ascii | ⛔ HAVE_AUTOVEC = ABORT (no scalar impl) |

**Cap B is the messiest cap** — not a clean homogeneous batch. Sentinel pointers
(`p`+`lmt`/`end`) instead of `int n`; 031 has neither (fixed `count[]` table);
103 needs a faithful port of `skip_whitespace`/`skip_word`; 005/006 need
null-terminated-string buffers. The real new infra is a **byte-buffer input
generator**, not a uniform ABI.

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
**Infra DONE** (✅): `SimdLoopMeta.axes_order` field; adapter multi-axis branch
(recovers axis sizes from input shapes, passes each as int64, sizes output from
`out_shape`); generator `_MULTI_AXIS` registry + `MultiAxisInfo` + writers.
**To add a loop:** add one entry to `_MULTI_AXIS` in `gen_simd_loop_harness.py`
with `axes`, `inputs` (name→[axes]), `output` (name,[axes]), `reference` (numpy),
`sizes` (edge/perf axis dicts), and `scalar` (C kernel — extractor is unreliable
on the SME-template `.c` files, so supply it explicitly). ABI:
`armbench_entry_loop_NNN(in1.., int64_t <axes..>, void* res_out)`.

| Loop | Description | Status |
|------|-------------|--------|
| 223 | matrix transposition (m,n) | ✅ |
| 216 | fp32 col-major GEMV | ✅ |
| 217 | int8 row-major GEMV | ✅ |
| 218 | fp64 col-major GEMV | ✅ |
| 219 | int8 col-major GEMV | ✅ |
| 220 | fp32 row-major GEMV | ✅ |
| 221 | fp64 row-major GEMV | ✅ |
| 025 | fp32 small fixed-size matmul | ⛔ no scalar impl (kernel is ABORT); fixed-size, no m/n/k |
| 130 | fp32 matmul (m,n,k) | ✅ |
| 135 | int8→int32 matmul (SDOT) | ✅ |
| 136 | int4→int32 matmul (LUT) | ⬜ defer: rearranged `b_r` buffer + in-struct `lut[16]` |
| 137 | bf16→fp32 matmul | ⬜ defer: bfloat16 — numpy has no native bf16 |
| 038 | fp16 stencil convolution (_Float16) | ✅ |
| 222 | fp16 convolution (multi-vector) | ⬜ defer: scratch `buffer` + border/stride + inline asm in scalar |

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
- 2026-06-15 — Cap E multi-axis infra built (axes_order, adapter branch, generator _MULTI_AXIS path); loop_223 transpose integrated as proof. 153/153 across 25 loops.
- 2026-06-15 — GEMV proven: loop_220 (fp32 row) + loop_221 (fp64 row) integrated; added ARM type normalization (float32_t→float, float64_t→double). 165/165 across 27 loops.
- 2026-06-15 — GEMV family complete: loop_216/218 (fp32/fp64 col-major), 217/219 (uint8 row/col-major). col-major handled by storing `a` as (n,m) so flat layout matches. 189/189 across 31 loops.
- 2026-06-15 — loop_038 fp16 stencil conv integrated; added _Float16 support (norm float16_t→_Float16, dtype maps). 195/195 across 32 loops. 130/135 dispatched to parallel subagents.
- 2026-06-15 — matmul 130 (fp32) + 135 (int8→int32) integrated via two parallel subagents; merged their validated entries into canonical _MULTI_AXIS. 205/205 across 34 loops. Cap E complete except documented defers (025/136/137/222).
- 2026-06-16 — Cap A loop_101 (pixel upscale): added output-override + derived-axis + abi_axes to _MULTI_AXIS (output is first ptr, size 2*(N-1)). Validated via bench.cli + smoke-test. 211/211 across 35 loops.
- 2026-06-17 — Cap A loop_106 (sheep-and-goats): output-override (output is middle ptr `b`); compress/sag/permute baked into both C kernel and a faithful numpy port; perm constant baked (not passed). Integer-exact via bench.cli + smoke. 223/223 across 37 loops. Cap A complete (111 deferred).

## Cap B handoff

Paste this to a fresh agent/session to continue with Cap B (string/sentinel loops).

---

You are continuing the simd-loop integration effort in CPU-Kernel-Baseline, branch
`feat/simd-loop-sort-123-124`. **Read `docs/loop-integration-progress.md` first** —
it has the full state. 37 loops integrated, 223/223 workloads passing. Caps A and E
are done. Your job is **Cap B: string / sentinel-pointer loops**.

### Ground rules (standards)
- Validate EVERY loop two ways: `python -m bench.cli bench --definition loop_NNN
  --solution reference-scalar_loop_NNN` (all workloads PASSED) AND
  `python scripts/test_reference_scalars.py` (shows `[PASS] loop_NNN  N/N`, and the
  total count must not drop). Commit per loop with a clear message; never break the
  suite.
- The generator is `scripts/gen_simd_loop_harness.py`. Single-axis loops live in
  `_classify`/TARGET_LOOP_IDS; multi-axis/override loops live in the `_MULTI_AXIS`
  dict (study `loop_106` for output-override, `loop_114` for const axes + an
  output-only axis). Adapter: `bench/datasets/simd_loop.py`. Input gen:
  `bench/runtime/inputs.py`. Workload schema: `bench/data/workload.py`.
- `bench-trace/` is gitignored/generated — regenerate, don't hand-edit.
- If you use worktree subagents, tell each to `git reset --hard
  feat/simd-loop-sort-123-124` FIRST (worktrees branch from origin/main, missing
  this session's infra) and to RETURN their validated config entry (don't rely on
  their worktree commits).

### Scope
Integrate: **005, 006, 034, 103, 031**. Skip **022, 026** (their HAVE_AUTOVEC block
is `ABORT: No implementations` — confirm, then mark ⛔ in the tracker like 025).

### The two pieces of new infra to build first
1. **Byte-buffer / string input generation.** Today `inputs.py` only makes
   `{"type":"random"}` (uniform `[1,100]`) and `{"type":"scalar"}`. String loops need
   structured byte buffers. Add a new workload input type (e.g. `{"type":"bytes",
   ...}` or `{"type":"string", ...}`) in `bench/data/workload.py` (a new
   `WorkloadInput` variant) and generate it in `inputs.py`. Decide the layout per
   loop: 005/006 need concatenated NUL-terminated strings; 103 needs bytes that
   include whitespace codes (9,10,13,32) so word-count is non-trivial; 031 just needs
   raw bytes.
2. **Sentinel begin/end ABI.** These kernels take pointer pairs, not `int n`:
   - 005/006: `struct {uint8_t *p; uint8_t *lmt; uint32_t checksum;}` — p=begin,
     lmt=end; output = checksum (scalar). The ABI must pass begin and end pointers
     (or begin + a length the harness computes). Easiest: allocate the buffer, pass
     `p` = base and `lmt` = base+len.
   - 034: `{uint8_t *a; uint8_t *b; uint8_t *lmt; uint32_t checksum;}` — two buffers
     + end sentinel.
   - 103: `{uint8_t *p; uint8_t *end; int checksum;}` — port `skip_whitespace` /
     `skip_word` from `loops/loop_103.c` into BOTH the C scalar kernel and the numpy
     reference (the byte test is exact).
   - 031: `{uint8_t *a; uint8_t *b;}` — NO size/sentinel. The kernel copies in
     chunks using a fixed static `count[]` table (CHUNKS entries), 10 outer
     iterations; output `b` and input `a` are each `10 * sum(count)` bytes. Bake the
     `count[]` table into the scalar kernel; the reference just reproduces the copy.

Recommended order: **031 first** (simplest — raw bytes, no sentinel, output =
deterministic copy), to land the byte-buffer input type. Then **103** (proves the
sentinel begin/end ABI + a helper port). Then **005/006/034** (NUL-terminated
strings) — these three can fan out to parallel subagents once the string input type
and sentinel ABI exist.

### Definition of done
005, 006, 034, 103, 031 each passing via bench.cli + smoke-test; 022/026 marked ⛔;
tracker + change log updated; everything committed.
