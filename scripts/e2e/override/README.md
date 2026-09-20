# armbench kernel override hook for llama.cpp (CPU backend)

Lets agent-optimized standalone kernels (shared objects with a C ABI) replace
specific ggml ops at runtime, so a stock `llama-bench` / `llama-cli` binary can be
used for an end-to-end tokens/s measurement with the agent kernels swapped in.
Target: llama.cpp **v0.4.1** (commit `b29c606`).

## Files

| file | purpose |
|---|---|
| `armbench_override.h/.c` | self-contained C module compiled into `ggml-cpu`; loads the manifest, `dlopen`s kernels, dispatches |
| `ggml-cpu-override.patch` | `git diff` vs v0.4.1: adds the module to the ggml-cpu CMake sources (+ `${CMAKE_DL_LIBS}`), and the dispatch call |
| `apply.sh <llama.cpp dir>` | copies the .c/.h into `ggml/src/ggml-cpu/` and applies the patch (idempotent) |
| `manifest.example.json` | manifest format |
| `test_kernel.c`, `test_override.cpp`, `test_override.sh` | self-test (see below) |

## Usage

```sh
scripts/e2e/override/apply.sh /path/to/llama.cpp          # once per checkout
cmake -B build -DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF ...   # see "repack / KleidiAI"
cmake --build build -j
ARMBENCH_OVERRIDES=/abs/manifest.json ARMBENCH_OVERRIDE_LOG=1 build/bin/llama-bench -m model.gguf ...
```

* `ARMBENCH_OVERRIDES` — path to the JSON manifest. Unset = hook is a single static-bool check per op.
* `ARMBENCH_OVERRIDE_LOG=1` — at load: one line per manifest entry; then one stderr line per
  matched key on its first hit (`[armbench-override] hit mul_mat type=q4_K K=.. N=.. (M=.., nth=..)`),
  so coverage can be verified against the model's matmul shapes.
* Load failures (unreadable manifest, parse error, `dlopen`/`dlsym` failure) always print to
  stderr and **disable all overrides** (a partially-overridden run would be a misleading number).
* A kernel returning non-zero aborts the process (other threads have already passed the op;
  there is no fallback at that point).

### Manifest

```json
{"kernels":[
  {"op":"mul_mat","type":"q4_K","K":2560,"N":9216,"so":"/abs/lib.so","symbol":"armbench_llamacpp_gemm","abi":"llamacpp","threads":1},
  {"op":"mul_mat","type":"q4_K","K":9216,"N":2560,"so":"/abs/lib2.so","symbol":"armbench_entry_gemm","abi":"entry"}
]}
```

Match key for `mul_mat` = (`type` = ggml type name of `src0`, e.g. `q4_K`; `K` = `ne00`; `N` = `ne01`).
`threads` is parsed and stored but **reserved/unused** (kernels run single-threaded, see below).
`abi` selects the entry-point signature:

| `abi` | signature | notes |
|---|---|---|
| `llamacpp` (default) | `int armbench_llamacpp_gemm(const void* A, const void* B, float* C, int64_t M, int64_t N, int64_t K)` | N, K passed at runtime |
| `entry` | `int armbench_entry_gemm(const uint16_t* A_bf16, float* output, const uint8_t* B_blocks, int M)` | reference-scalar harness entry; N and K are `constexpr` in the kernel and must equal the manifest's `N`/`K`; `M` is `int` |

In both: `A` = row-major bf16 `[M,K]` (raw uint16 bits, converted from f32 by the hook with
round-to-nearest-even via `ggml_cpu_fp32_to_bf16`), `B` = the raw ggml quantized weight rows
(`src0->data` as-is, e.g. `block_q4_K[N][K/256]`), `C`/`output` = row-major f32 `[M,N]`, return 0 on success.

Mapping to ggml's `mul_mat`: `src0` = weights `[K,N]` (`ne00=K, ne01=N`), `src1` = activations
f32 `[K,M]` (`ne10=K, ne11=M`), `dst` = f32 `[N,M]` (`ne0=N, ne1=M`). For the 2-D case `dst` row `m`
is exactly `C` row `m`, so no transposition is needed. `src1` rows may be strided (`nb11`) but
elements must be packed (`nb10 == 4`).

The op is claimed only when: `src0->extra == NULL`, `src1`/`dst` are F32, `ne02==ne03==1`,
`ne12==ne13==1`, `src0` and `dst` contiguous, and a manifest entry matches. The decision is a
pure function of tensor metadata, so every ggml worker thread makes the same choice.

## Dispatch point (verified in v0.4.1)

`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward()` (line 1736), `case GGML_OP_MUL_MAT:` at
**lines 1861-1864**:

```c
case GGML_OP_MUL_MAT:
    {
        if (!armbench_override_mul_mat(params, tensor)) {   // added
            ggml_compute_forward_mul_mat(params, tensor);
        }
    } break;
```

Threading: the CPU backend calls `ggml_compute_forward` on every worker thread with
`params->ith/nth` and places `ggml_barrier` after every node (`ggml_graph_compute_thread`,
ggml-cpu.c:3157). The override runs the kernel on `ith == 0` only; other threads return `true`
immediately and wait at that barrier. The standalone ABI has no threading, so **override kernels
are single-threaded** even when llama.cpp runs with `-t 8`; the stock path for non-overridden
ops still uses all threads. (For a fair comparison, note the agent kernel's own internal
parallelism — none today — vs ggml's chunked multi-threaded mul_mat.)

Op fusion: `ggml_cpu_try_fuse_ops` (ggml-cpu.c:3079) only fuses `RMS_NORM+MUL`; `MUL_MAT` is
never fused, so the patched `case` is the sole path for plain-buffer matmuls.

## repack / KleidiAI interaction (what the code does)

`ggml_compute_forward` first calls `ggml_cpu_extra_compute_forward(params, tensor)`
(ggml-cpu.c:1744, defined in `traits.cpp`). It iterates the CPU device's *extra buffer types*
(`ggml-cpu.cpp:42`: KleidiAI if `GGML_USE_CPU_KLEIDIAI`, repack if `GGML_USE_CPU_REPACK`, AMX on
x86) and, if one returns tensor traits for the op, runs that implementation and **returns before
the switch** — i.e. before our hook.

* **Repack** (`repack.cpp:4778-4823`): claims `MUL_MAT` when `src0->buffer->buft ==
  ggml_backend_cpu_repack_buffer_type()`. Weights land in that buffer type when llama.cpp's
  `select_weight_buft` (`llama-model.cpp`) finds `supports_op` true, which for **`Q4_K` on any
  NEON+dotprod / NEON+i8mm CPU is the case whenever `N % 8 == 0`** (`repack.cpp:4605-4620`,
  `q4_K_8x8_q8_K` / `q4_K_8x4_q8_K`), also Q4_0, Q2_K, IQ4_NL, Q8_0 (`ggml_repack_get_optimal_repack_type`, :4528).
  `init_tensor` (:4731) sets `tensor->extra` to the repack traits and **re-lays out `src0->data`
  in an interleaved 4x/8x block format while leaving `src0->type` as `q4_K`**. So with repack
  active the override (a) is never reached for those tensors and (b) must not touch them even
  if it were — hence the `src0->extra != NULL` guard.
* **KleidiAI** (`kleidiai.cpp:1712-1730, 1863`): same mechanism (`tensor->extra`, own buffer
  type, own packed layout); in v0.4.1 it only has kernels for Q4_0/Q8_0 (and F16 for `M>1`), so
  it would not intercept `q4_K` matmuls but would still be an extra buffer type in the list.

**Recommendation for the measurement builds** (same source tree, two build dirs):

| build | cmake flags | meaning |
|---|---|---|
| `stock` baseline | default (`-DGGML_CPU_REPACK=ON`, `-DGGML_CPU_KLEIDIAI=OFF` default on macOS/Linux unless enabled) | best-effort stock llama.cpp; q4_K weights are repacked and use ggml's i8mm/dotprod tiles |
| `agent` kernels | `-DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF` + patch + `ARMBENCH_OVERRIDES` | weights stay in plain ggml block layout so the override sees them |

Alternative without a second build: run the patched default build with **`--no-repack`**
(`common/arg.cpp:2420`, sets `no_extra_bufts`), which keeps weights in plain CPU buffers and
lets the override match; with `ARMBENCH_OVERRIDES` unset the same binary is then "stock, no
repack" — a useful third data point since the agent kernels' fair comparison is arguably the
non-repacked stock path (both consume the same `block_q4_K` layout). Coverage check: with
`ARMBENCH_OVERRIDE_LOG=1`, every distinct (type,K,N) of the model's dense projections should
show a `hit` line; any shape listed in the manifest without a hit means it was intercepted or
mismatched.

Also note `GGML_USE_LLAMAFILE` (default ON): `llamafile_sgemm` is called *inside*
`ggml_compute_forward_mul_mat`, i.e. after our hook, so it cannot steal an overridden op.

## Self-test

```sh
scripts/e2e/override/test_override.sh /path/to/llama.cpp [build dir]
```
Requires the patch applied and `cmake --build build --target ggml ggml-base ggml-cpu` done
(static libs). It builds `libtest_kernel.$SOEXT` (dylib on macOS, so on Linux) exposing both ABIs
(dequantize `block_q4_K` via ggml's `dequantize_row_q4_K` + plain fp32 dot from the bf16 A), and
a harness that runs a graph with `C1 = mul_mat(q4_K [K=512,N=64], f32 [K=512,M=3])` (llamacpp ABI)
and `C2 = mul_mat(q4_K [K=256,N=32], f32 [K=256,M=3])` (entry ABI) on 4 threads, once without
the env var (dumps reference), once with. Asserts: outputs agree (abs 1e-2 + rel 1e-2; the stock
path quantizes activations to Q8_K, the override rounds them to bf16), 2 kernels loaded, 2
override hits, each test-kernel entry called exactly once, and two `hit mul_mat` log lines.
Result on this Mac (M2 Pro, v0.4.1): PASS, max abs diff 3.9e-3.

## Known limitations / TODO

* **Single-threaded override kernels** (see Threading). `threads` field reserved.
* **rms_norm is a stub** (`armbench_override_rms_norm` always returns false; not wired into the
  patch). ggml's `GGML_OP_RMS_NORM` has no weight — llama.cpp emits `rms_norm` then `mul`, and the
  CPU backend fuses the pair in `ggml_cpu_try_fuse_ops`. The standalone
  `armbench_llamacpp_rms_norm(x, weight, out, M, D, eps)` ABI needs the weight, so a correct
  override must match the `RMS_NORM+MUL` pair at graph level (natural place: that fusion site).
* Only 2-D `mul_mat` with F32 activations (no `MUL_MAT_ID`/MoE, no batched 3-D/4-D, no F16/BF16 src1).
* No Windows support (dlopen); the module compiles but stays disabled there.
* `src1 -> bf16` conversion is done by the hook on thread 0 (thread-local scratch grown on demand);
  its cost is included in the measured time, which matches what an integrated bf16 kernel would pay.
* Adding ops is additive: new enum value in `armbench_op`, name in `armbench_op_names`, a
  `find_*`/`armbench_override_*` pair, and a call site in the patch.
