# kernel_vs_repack: agent kernel vs llama.cpp's real CPU_REPACK path, per GEMM shape

Single-threaded timing of one quantized GEMM shape (llama.cpp v0.4.1, b29c606) on the same weights and activations:

| column | what runs |
|---|---|
| `t_repack_*` | `ggml_mul_mat`, weight in the **CPU_REPACK** extra buffer type (found via `ggml_backend_dev_get_extra_bufts`, as llama.cpp does); `ggml_backend_tensor_set` repacks it into the interleaved layout, and compute goes through `repack.cpp` gemm/gemv (i8mm `*_8x8` on Graviton4, dotprod `*_8x4` without i8mm). This is what stock `llama-bench` runs. |
| `t_norepack_*` | the same `ggml_mul_mat` with the weight in a plain CPU buffer. Same as `--no-repack` / `build-norepack`, and it is what `baseline-llamacpp-arm` measures. |
| `t_agent_*` | the agent `.so` loaded with `dlopen`, called as `scripts/e2e/override` calls it: f32 activations go to bf16 through `ggml_cpu_fp32_to_bf16` (this conversion is **included** in the time, and `t_bf16_convert_min_us` reports it separately). The kernel then gets the raw ggml block rows and writes row-major `[M,N]`. |

All three use the CPU backend with `ggml_backend_cpu_set_n_threads(1)`. Both ggml columns include ggml's own f32 to Q8_K/Q8_0 activation quantization. The timed rounds are interleaved (repack, norepack, agent, convert, and repeat) after the warmup. The results report min and median. Weights are Gaussian f32 quantized with `ggml_quantize_chunk`, or you can pass `--weights raw_rows.bin`. Activations are N(0,1), or you can pass `--act x.npy|x.bin`.

Speed ratios are `agent_vs_repack = t_repack_min / t_agent_min`, and a value > 1 means the agent is faster. `agent_vs_norepack` and `repack_vs_norepack` are computed the same way.

There are four sanity outputs:
- `sqnr_db` and `max_abs_agent` compare the agent output with the repack output.
- `sqnr_norepack_vs_repack_db` should be about 130 dB or more, because both paths quantize activations the same way.
- `sqnr_fp64ref_*_db` compare each path against an fp64 reference that uses dequantized weights and f32 activations on `--ref-rows` sampled columns.
- `repack_kernel` is ggml's chosen kernel, captured from its "repack tensor … with q4_K_8x8" log.

If ggml has no repack kernel for the type, N, and CPU combination (`tensor->extra == NULL` after allocation in CPU_REPACK), the line reports `repack_used:false` and `t_repack_*:null`. It does not time the plain path under the repack name.

## Build (on the Graviton4 e2e box)

The box already has `~/llama.cpp-e2e/build-stock`, which uses the defaults: `GGML_CPU_REPACK=ON` and static libs.

```sh
cd ~/arm-bench-e2e/scripts/e2e/kernel_vs_repack
./build.sh ~/llama.cpp-e2e ~/llama.cpp-e2e/build-stock        # -> ./bench_repack  (-O3 -mcpu=native)
```

`build.sh` links `libggml.a`, `libggml-cpu.a`, and `libggml-base.a`, or the matching `.so` files if that is what the build produced. It adds `-fopenmp` only if the build's CMakeCache says OpenMP was enabled. Set `CXXFLAGS_ARCH=-march=armv8.2-a+dotprod` to run a compile-only check on another machine. Do **not** link `build-agent` or `build-norepack`: they have `GGML_CPU_REPACK=OFF`, so they have no CPU_REPACK buffer type.

## Run

```sh
# one shape
./bench_repack --type q4_K --N 2560 --K 9216 --M 1,2,4,8,16,32,128,512 \
    --so ~/e2e/gatev2b/gemm_ggml_q4_K_n2560_k9216.so --symbol armbench_entry_gemm --abi entry_rows

# every kernel in a manifest -> JSONL (one line per shape x M) + summary table on stderr
./run_all.sh ~/e2e/gatev2b/manifest_gatev2b.json ~/e2e/kernel_vs_repack_gatev2b.jsonl
```

`run_all.sh` accepts these environment variables:

| variable | default | meaning |
|---|---|---|
| `MS` | `1,2,4,8,16,32,128,512` | M values to run |
| `BIG_N`, `BIG_MS` | `100000`, `1,2,4,8` | shapes with N > `BIG_N` use `BIG_MS`; this covers the lm_head (N=248320) |
| `PIN_CPU` | `2` | CPU passed to `taskset`; set it to `""` to disable pinning |
| `BENCH_ARGS` | none | extra arguments for `bench_repack` (see below) |

Useful `BENCH_ARGS` values:

| argument | default | meaning |
|---|---|---|
| `--reps` | `20` | timed rounds |
| `--warmup` | `3` | warmup calls before timing |
| `--max-sec` | `4` | time budget per M, with at least `--min-reps` 3 rounds |
| `--flush-mb` | `0` | stream through this many MB before every timed call, for a cold-cache mode (`256` is > L3) |
| `--ref-rows` | `64` | columns sampled for the fp64 reference |
| `--gen-threads` | all cores | threads for weight generation, which is untimed |

For `entry` and `entry_rows` kernels the bench calls the full-N `armbench_entry_gemm` from one thread. `run_all.sh` unsets `ARMBENCH_OVERRIDES` (so does the binary), because if the hook were compiled into `build-stock`'s libggml-cpu it would take over the plain-buffer path.

## Caveats (v0.4.1, verified in source)

- Repack exists on NEON for **q4_K, q5_K, q6_K** (N % 8 == 0; `*_8x8` with i8mm, `*_8x4` with dotprod only) and **q8_0** (N % 4 == 0; `4x8` with i8mm, `4x4` with dotprod). All 11 gatev2b shapes satisfy these conditions, including the q6_K lm_head (N=248320). Other N values show up as `repack_used:false`.
- `q4_K_8x8` and `q8_0_4x8` gemm have an SVE path, but only for 256-bit SVE. Graviton4 has 128-bit SVE, so it takes the NEON+i8mm path. Graviton3 would take the SVE path.
- The repack path uses gemm for rows in groups of 4 and gemv for the leftover rows. Expect a step at M = 4k+r.
- Weights are small (for example 1.5 MB for q4_K 1024x2560) and stay resident in L2 across repeated calls. Real decode streams weights from DRAM. Use `--flush-mb 256` for the cold-cache comparison.
- Weight generation is slow: ggml's reference quantizers run at about 1 µs/element per core. The q6_K lm_head (635M elements) takes about 1 minute on 16 cores. This time is not included in the measurements.
- Tested on the Graviton2 controller with `reference-scalar` kernels built for armv8.2 (dotprod path: `q4_K_8x4`, `q5_K_8x4`, `q6_K_8x4`, `q8_0_4x4`). That run produced `repack_used:false` for N=1020, repack vs norepack at 135 dB, and agent vs fp64 at about 43 dB, the same as ggml. The i8mm (`8x8`) timings and the SVE2 gatev2b `.so` files can only run on the Graviton4.
