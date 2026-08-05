# Case Study: `conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1`

**Definition**: Kh=3, Kw=3, Sh=1, Sw=1, Dh=1, Dw=1, pad=(1,1), int8 activations + int8
per-output-channel-quantized weights ("w8a8ch"), dequantized to float32 output via
`real = int8_value * (input_scale * weight_scales[c_out])`, with `input_scale = 0.02677`
fixed at generation time across all 20 workloads.
**Speedup** is defined as `ncnn_baseline_time / candidate_time` (>1 = candidate beats ncnn;
<1 = candidate is slower than ncnn). Run on a real remote Graviton4 (Neoverse V2, 128-bit
SVE2, `-march=armv9-a+sve2`) instance — the IPC and cache-miss counters below are genuine
hardware measurements, not simulated.

**Speedup trajectory** (vs ncnn-baseline, geomean across 20 workloads):

| Turn  | Version | Strategy | Time speedup | Cycle speedup | IPC   | Cache misses (mean) |
|-------|---------|----------|--------------|----------------|-------|----------------------|
| 1     | v1      | Scalar, per-tap boundary checks (compiled only, **never evaluated**) | — | — | — | — |
| 5–7   | v2      | im2col + SVE SDOT, 1 output channel at a time | 1.419× | 1.402× | 4.428 | 2,058,514 |
| 11–13 | v3 — **submitted** | im2col + SVE SDOT, OC\_TILE=4 | **2.528×** | **2.493×** | 3.614 | 1,364,219 |
| 17    | v4      | im2col + SVE SDOT, OC\_TILE=8 (compiled, **never evaluated**) | — | — | — | — |

Final submission (`results/conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1_ncnn_openrouter_anthropic_claude-sonnet-4-6.json`):
`time_speedup=2.5259`, `cycle_speedup=2.4891`, `max_absolute_error=0.03125`,
`max_relative_error=1.6e-7` (PASSED). The `source_file` recorded for the submit turn is
`v3.cpp` — **the agent compiled v4 at turn 17 but submitted v3's already-validated result
at turn 18 without ever running `evaluate` on v4.** v4 is therefore an unfinished experiment,
not a discarded regression; there is no timing data for the OC\_TILE=8 idea at all.

---

## 1. Reference-scalar vs ncnn-baseline: performance direction and cause

**No measured reference-scalar-vs-ncnn timing exists in this repository for this
definition.** `agent-runs/conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1/trajectory.jsonl` only
contains turns for the agent's own v1–v4; there is no separate reference-scalar `evaluate`
entry, and neither `bench-trace/traces/conv2d/conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1.jsonl`
(workload replicas only, no timing) nor `results/` (only the final candidate submission)
records one. Rather than inventing a ratio, this section gives a code-grounded qualitative
argument for the expected direction.

### What ncnn actually runs for this shape

The benchmark harness constructs every `ncnn::Option` itself, in
`bench/datasets/_ncnn_lib/_mat_factory.cpp:197-204`:

```cpp
opt->num_threads = 1;
opt->use_packing_layout = false;
opt->use_fp16_storage = false;
opt->use_bf16_storage = false;
opt->use_sgemm_convolution = false;
opt->use_winograd_convolution = false;
```

Both `use_sgemm_convolution` and `use_winograd_convolution` are forced off. Cross-checking
`ncnn/src/layer/arm/convolution_arm.cpp:1250-1277` (`create_pipeline_int8_arm`) and
`:1298-1415` (`forward_int8_arm`), the dispatch is an `if (winograd) / else if (sgemm) / else`
chain — with both flags off, **every workload for this definition falls to the final `else`
branch**, `convolution_packed_int8()` (`convolution_arm.cpp:1414`), regardless of channel
count. ncnn's more algorithmically-clever int8 paths (Winograd F(2,3)/F(4,3), im2col+GEMM)
are never exercised in this benchmark at all — this op-type's ncnn baseline is always the
"packed direct convolution" kernel.

### Why that kernel should still beat reference-scalar

`convolution_packed_int8()` (`ncnn/src/layer/arm/convolution_packed_int8.h:517-830`) is
genuine NEON int8 dot-product code — it uses `vdotq_lane_s32` (`:638-645`), which is
compiled in because `-march=armv9-a+sve2` implies `__ARM_FEATURE_DOTPROD`. Each `vdotq_lane_s32`
call retires 16 int8×int8 multiply-accumulates into a 4-lane int32 accumulator in one
instruction. The reference-scalar baseline
(`bench-trace/solutions/ncnn/reference-scalar/conv2d/conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1.json`,
embedded `kernel.cpp`) does the equivalent work with a 7-deep nested scalar loop
(n, co, oh, ow, ci, kh, kw), one `int64_t` multiply-accumulate at a time:

```cpp
for (int kh = 0; kh < Kh; ++kh) {
    for (int kw = 0; kw < Kw; ++kw) {
        int ih = oh * Sh - pad_top  + kh * Dh;
        int iw = ow * Sw - pad_left + kw * Dw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            acc += (int64_t)in_c[ih * W + iw] * (int64_t)w_c[kh * Kw + kw];
    }
}
```

So even though ncnn's exercised path (Section 2) pays a real gather/scatter tax from its
disabled packing layout, it is still doing ≥16 MACs per SIMD instruction versus
reference-scalar's 1 MAC per (mostly branch-guarded) scalar instruction. Reference-scalar
should be substantially slower than ncnn-baseline here — this is reasoning from the actual
code paths on both sides, not a measured claim.

---

## 2. ncnn-baseline's own bottlenecks

Because `use_packing_layout=false` forces `bottom_blob.elempack == 1` (no NEON-packed
activation layout), `convolution_packed_int8()` pays three costs that a packed-layout build
would avoid, all confirmed directly in the source it actually executes:

**Gather load** (`convolution_packed_int8.h:618-624`, inside the `inch` loop grouped by 8):

```cpp
else // if (elempack == 1)
{
    signed char tmp0[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
    signed char tmp1[8] = {r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7]};
    _r0 = vld1_s8(tmp0);
    _r1 = vld1_s8(tmp1);
}
```

`N = bottom_blob.cstep * elempack` is the stride between channel planes, so each of the 8
scalar reads that build `tmp0` lands on a different channel plane — a strided gather, not a
vector load, executed once per 8-input-channel group per kernel tap.

**kptr reset** (`:597`, at the top of the 2-output-pixel loop):

```cpp
const signed char* kptr = weight_data_tm.channel(p / 8);
```

This re-walks the entire `(inch/8 × maxk × 64B)` packed-weight block from its start for
every pair of output pixels — repeated `outh*outw/2` times per 8-output-channel group.

**Scatter store** (`:748-766`, the `out_elempack == 1` branch — forced here since
`use_packing_layout=false` also removes output packing):

```cpp
outptr[0] = vgetq_lane_s32(_sum0, 0);
outptr[1] = vgetq_lane_s32(_sum2, 0);
outptr[M] = vgetq_lane_s32(_sum0, 1);
outptr[M + 1] = vgetq_lane_s32(_sum2, 1);
...  // 16 scattered lane extractions total, across 8 non-contiguous channel planes
```

The 8 output channels' results, computed together in interleaved SIMD registers, must be
de-interleaved lane-by-lane back out to 8 separate non-contiguous channel-plane addresses
every 2 output pixels. Combined, these three patterns mean ncnn's baseline for this
definition does correct, real SIMD arithmetic but spends a large fraction of its time on
strided gather/scatter memory traffic and redundant weight re-reads that a packed-layout
build (`use_packing_layout=true`) would not incur.

---

## 3. Why the agent's submitted version (v3) beats ncnn-baseline

**v3 achieves `time_speedup = 2.528` (geomean), meaning it is 2.5x *faster* than the ncnn
baseline described above.**

### The SIMD widths are equal — the win is not from wider vectors

Graviton4 (Neoverse V2) implements SVE2 at a 128-bit vector length (per this repo's own
`CLAUDE.md` instance-type table). `v3.cpp:86-87` queries this at runtime:

```cpp
const int vl32 = (int)svcntw();   // = 4 on this target (128-bit / 32-bit lanes)
const int vl8  = vl32 * 4;        // = 16 int8 lanes per svdot_s32 call
```

16 int8 lanes per `svdot_s32` is the same per-instruction throughput as ncnn's
`vdotq_lane_s32` (also a 128-bit NEON operation). So the 2.5x gap is not "wider SIMD" — it's
memory-access pattern, mirroring the inverse of Section 2's findings.

### im2col gives contiguous loads where ncnn is forced into gather/scatter

`v3.cpp:99-103` reads both the activation window and all 4 output channels' weights with
straight contiguous loads:

```cpp
svint8_t va = svld1_s8(svptrue_b8(), col_m + k);
svint8_t vb0 = svld1_s8(svptrue_b8(), w0 + k);
svint8_t vb1 = svld1_s8(svptrue_b8(), w1 + k);
svint8_t vb2 = svld1_s8(svptrue_b8(), w2 + k);
svint8_t vb3 = svld1_s8(svptrue_b8(), w3 + k);
```

This works because the agent builds its own `col` buffer (`v2.cpp:15-44` / `v3.cpp:83-106`)
with each output pixel's `C_in*9` window stored contiguously — the candidate ABI has no
notion of ncnn's "elempack", so there's nothing forcing a strided layout. The agent's kernel
never pays ncnn's gather-load or scatter-store tax from Section 2; both sides run the same
direct-convolution arithmetic (no algorithmic shortcut like Winograd on either side — this
is a memory-layout win, not an algorithmic one).

### v1 → v2 → v3 progression

- **v1** (turn 1): scalar, per-tap boundary-checked accumulation, structurally identical to
  the reference-scalar baseline. Compiled but never evaluated — the agent moved straight to
  vectorizing after reading it back (turn 4 `read_code`).
- **v2** (turns 5–7): introduces `im2col_int8` (`v2.cpp:15-44`) plus per-output-channel SVE
  `svdot_s32` (`v2.cpp:80-106`), one channel at a time. `time_speedup=1.419`, `ipc=4.428`,
  `cache_misses=2,058,514`. High IPC here reflects that each of the `C_out` outer-loop passes
  re-streams the entire `col` buffer (`M*K` bytes) from memory sequentially and contiguously
  — good instruction efficiency, but `C_out` separate full re-reads of `col`.
- **v3** (turns 11–13): restructures to `CO_TILE=4` (`v3.cpp:62-127`) — four output channels'
  `svdot_s32` accumulations share a single `svld1_s8(col_m+k)` load per iteration
  (`v3.cpp:99`, `104-107`). This cuts the number of full `col`-buffer re-reads from `C_out`
  to `C_out/4`, which is exactly what `cache_misses_mean` shows: 2,058,514 → 1,364,219 (≈1.51x
  fewer misses, not the naive 4x, since weight traffic and the `col`-build pass itself are
  unaffected by this tiling). `ipc_mean` *drops* from 4.428 to 3.614 even as time improves —
  with 4 accumulators live per iteration instead of 1, the core has more concurrent
  load/FMA-equivalent work in flight per cycle, so the bottleneck partially shifts from
  "wait on `col` reads from L2/L3" (v2) toward "sustain 4 parallel `svdot_s32` chains plus
  4 horizontal `svaddv_s32` reductions" (v3) — fewer stalls overall (faster wall time) but a
  lower fraction of a now-busier pipeline is doing useful arithmetic per cycle.

### v4 was never measured

`v4.cpp` (turn 17) extends the same idea to `CO_TILE=8` (`v4.cpp:109-207`), explicitly
reasoning in its own header comment that v3's IPC of 3.6 indicated memory-bandwidth
pressure and that wider OC tiling should amortize the `col` re-read further. It compiled
successfully but the agent submitted v3's result at turn 18 without an `evaluate` call on
v4 — there is no `time_speedup`/`ipc`/`cache_misses` data for OC\_TILE=8 in this run at all,
so whether it would have beaten v3 is unknown.

---

## 4. Hardcoded parameters in the agent's submitted solution (v3.cpp)

| Parameter | Hardcode type | Effect of changing |
|-----------|---------------|---------------------|
| Kh=3, Kw=3 | Algorithmic | `K = C_in * 9` (`v3.cpp:25`) and the im2col build's literal `for (kh<3) for(kw<3)` (`v3.cpp:46-56`) bake in a 3×3 tap count structurally; any other kernel size produces a wrong-sized `col` buffer and wrong dot-product length. |
| Sh=1, Sw=1 | Algorithmic | The im2col index math `int ih = oh - 1 + kh; int iw = ow - 1 + kw;` (`v3.cpp:47,49`) maps output pixel `(oh,ow)` directly onto the input window start with no stride multiply (`oh*Sh`, `ow*Sw`). Any stride≠1 would read the wrong input window entirely — a silent correctness bug, not a crash. |
| Dh=1, Dw=1 | Algorithmic | Kernel offsets `kh`/`kw` are added to `oh-1`/`ow-1` directly (`v3.cpp:47,49`) with no dilation multiplier (`kh*Dh`). Any dilation≠1 would sample the wrong input taps. |
| pad_top=1, pad_left=1 | Algorithmic | The bounds check `if (ih >= 0 && ih < H && iw >= 0 && iw < W)` (`v3.cpp:50`) combined with the `oh-1`/`ow-1` offset only produces correct zero-padding behavior for exactly 1 pixel of padding on each side; the harness computes `H_out`/`W_out` for the actual `pad_top`/`pad_left` values, so a mismatched pad would misalign the zero-fill region. |
| input_scale=0.02677 | Not hardcoded (by design) | Baked as a `constexpr` in `conv2d.h` because it is genuinely constant across every workload for this definition (verified in `bench-trace/workloads/.../*.jsonl`: all 20 workloads use `input_scale.value == 0.02677`) — not a shortcut, this is the intended per-definition-constant convention documented in the solution header. |
| C_out (arbitrary) | Not hardcoded | v3 has a full remainder path: `CO_TILE=4` main loop (`v3.cpp:65-127`) followed by a scalar tail loop for `co < C_out` (`v3.cpp:131-158`) — any `C_out` value is handled correctly, including non-multiples of 4. All 20 workloads use `C_out ∈ {64,128,256,512}` (all divisible by 4), so the tail path is never actually exercised by this benchmark, but it is present and correct in the code. |
