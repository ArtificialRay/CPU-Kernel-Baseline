# armbench Kernel Override Hook (`llama.cpp` CPU Backend)

A lightweight runtime hook that allows standalone, agent-optimized C-ABI kernels (`.so`) to replace specific `ggml` operations in `llama.cpp`. This enables end-to-end `tokens/s` benchmarking via standard `llama-bench` or `llama-cli` binaries without rebuilding for every kernel variant.

* **Target Version:** `llama.cpp` **v0.4.1** (Commit `b29c606`)

---

## File Structure

| File | Description |
| --- | --- |
| **`armbench_override.h/.c`** | Self-contained C module compiled into `ggml-cpu`. Handles manifest loading, `dlopen` calls, and kernel dispatching. |
| **`ggml-cpu-override.patch`** | Source patch (`git diff` vs v0.4.1) linking the module to `ggml-cpu` CMake sources and inserting the dispatch hook. |
| **`apply.sh <llama.cpp dir>`** | Idempotent setup script that copies `.c`/`.h` files into `ggml/src/ggml-cpu/` and applies the patch. |
| **`manifest.example.json`** | Reference structure for configuration manifests. |

---

## Quick Start

### 1. Build & Setup

```bash
# Apply patch to target checkout (once per repo)
e2e/override/apply.sh /path/to/llama.cpp

# Configure CMake with repack/KleidiAI disabled and build
cmake -B build -DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF
cmake --build build -j

```

### 2. Execution

```bash
ARMBENCH_OVERRIDES=/abs/manifest.json \
ARMBENCH_OVERRIDE_LOG=1 \
build/bin/llama-bench -m model.gguf

```

---

## Environment Variables & Configuration

| Variable | Type | Description |
| --- | --- | --- |
| `ARMBENCH_OVERRIDES` | Path | Absolute path to the JSON manifest. Unset disables all hooks (single-bool overhead check). |
| `ARMBENCH_OVERRIDE_LOG` | Flag (`1`) | Enables stderr logs for manifest loading and key matches on first hit (e.g., `[armbench-override] hit mul_mat type=q4_K K=... N=...`). |
| `ARMBENCH_OVERRIDE_STRICT` | Flag (`1`) | Aborts execution on any loading error or skipped manifest entry to guarantee clean measurements. |
| `ARMBENCH_OVERRIDE_THREADS` | Flag (`1`) | Forces single-threaded (thread-0-only) dispatch for all ABIs (used for baseline ablations). |

> **Error Handling Policy:** Any load failure (e.g., invalid path, parse error, missing `dlsym`) prints an error to stderr and **disables all overrides** to prevent partial/misleading benchmarking runs.

---

## Manifest & ABIs

### Manifest Schema Example

```json
{
  "kernels": [
    {
      "op": "mul_mat",
      "type": "q4_K",
      "K": 2560,
      "N": 9216,
      "so": "/abs/lib.so",
      "symbol": "armbench_llamacpp_gemm",
      "abi": "llamacpp"
    },
    {
      "op": "mul_mat",
      "type": "q4_K",
      "K": 9216,
      "N": 2560,
      "so": "/abs/lib2.so",
      "symbol": "armbench_entry_gemm",
      "abi": "entry"
    },
    {
      "op": "mul_mat",
      "type": "q4_K",
      "K": 2560,
      "N": 2560,
      "so": "/abs/lib3.so",
      "symbol": "armbench_entry_gemm",
      "symbol_rows": "armbench_entry_gemm_rows",
      "abi": "entry_rows"
    }
  ]
}

```

* **Match Keys (`mul_mat`):** `type` (`src0` ggml type name), `K` (`ne00`), and `N` (`ne01`).
* **Note:** The `threads` field in JSON is currently reserved/unused (thread count is dynamically controlled by `ggml`'s `nth`).

### Supported ABI Variants

| ABI Type | Signature & Description | Threading Dispatch |
| --- | --- | --- |
| **`llamacpp`**<br>

<br>*(Default)* | `int armbench_llamacpp_gemm(const void* A, const void* B, float* C, int64_t M, int64_t N, int64_t K)` | Single-threaded (Thread 0) |
| **`entry`** | `int armbench_entry_gemm(const uint16_t* A_bf16, float* output, const uint8_t* B_blocks, int M)`<br>

<br>• $N$ and $K$ baked in as `constexpr`<br>

<br>• Reference-scalar harness layout | Single-threaded (Thread 0) |
| **`entry_rows`** | Requires two exported functions (`symbol` and `symbol_rows`):<br>

<br>`int armbench_entry_gemm_rows(const uint16_t* A_bf16, float* output, const uint8_t* B_blocks, int M, int n_rows)`<br>

<br>• Computes weight rows `[0, n_rows)` relative to `B_blocks`<br>

<br>• Writes output with stride `n_rows` | Multi-threaded |

#### Memory Layout Specifications

* **`A`:** Row-major `bf16` `[M, K]` (converted from F32 with round-to-nearest-even via `ggml_cpu_fp32_to_bf16`).
* **`B`:** Raw `ggml` quantized weight rows (`src0->data` as-is).
* **`C` / `output`:** Row-major `F32` `[M, N]`. Returns `0` on success.

---

## Technical Details

### Dispatch Hook

Hooked inside `ggml/src/ggml-cpu/ggml-cpu.c` (`ggml_compute_forward()`):

```c
case GGML_OP_MUL_MAT:
    {
        if (!armbench_override_mul_mat(params, tensor)) {   // Injected Hook
            ggml_compute_forward_mul_mat(params, tensor);
        }
    } break;

```

### Threading Execution Model

Uses thread-local scratchpads (`_Thread_local`) for F32 to BF16 casting without shared mutable state or internal barriers.

* **Single-Threaded (`llamacpp` / `entry`):** Thread 0 processes the node; other threads bypass and wait at `ggml`'s node barrier.
* **Multi-Threaded (`entry_rows`):**
* **Decode Phase ($M = 1$ / N-Split):** Splitted along weight rows across worker threads (`chunk = ceil(N / nth / 64) * 64`). Each thread casts its own activation row.
* **Prefill Phase ($M \ge 2$ / M-Split):** Splitted along activation rows (`mpt = ceil(M / nth)`). Each thread calls `entry_gemm` for its assigned subset.



---

## Repack & KleidiAI Compatibility

`ggml` features extra buffer types (e.g., Repack, KleidiAI) that intercept operations **before** `ggml_compute_forward` switches to our hook.

| Build Configuration | CMake Flags | Behavior |
| --- | --- | --- |
| **Stock Baseline** | `-DGGML_CPU_REPACK=ON`<br>

<br>`-DGGML_CPU_KLEIDIAI=OFF` | Repacks `q4_K` weights into dynamic interleaved blocks. Hook is bypassed. |
| **Agent Kernels** | `-DGGML_CPU_REPACK=OFF`<br>

<br>`-DGGML_CPU_KLEIDIAI=OFF` | Keeps weights in standard `block_q4_K` layout. Reaches override hook. |

> **Alternative Approach:** Run a patched default build with `--no-repack` to use standard CPU buffers dynamically without compiling a separate binary.

---

## Constraints & Limitations

* **Batch Decode Parallelism:** For batch sizes $M \in [2, \text{nth})$, execution is limited to $M$ threads under M-split logic.
* **RMS Norm Status:** Currently a stub (`armbench_override_rms_norm` returns `false`). `RMS_NORM+MUL` fusion logic is not yet integrated.
* **Supported Tensors:** Supports 2D `MUL_MAT` with F32 activations only (No MoE `MUL_MAT_ID`, 3D/4D batches, or FP16/BF16 `src1`).
* **OS Support:** Linux / macOS only (`dlopen` required; Windows is unsupported).