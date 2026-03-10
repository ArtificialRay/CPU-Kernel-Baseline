# sgl_cpu_kernels — Standalone AVX-512 BF16 CPU Kernel Library

Replaces SGLang's `sgl-kernel/csrc/cpu/` operators with **zero PyTorch/ATen/libtorch dependency**. Be careful that these kernels are implemented by **AVX512**

---

## What was removed and why

| Original dependency | Where it appeared | What replaced it |
|---|---|---|
| `at::Tensor` / `data_ptr<>()` | Every kernel entry point | Raw `bf16_t*` / `float*` pointers |
| `at::parallel_for` | Thread dispatch in moe.cpp, gemm | `#pragma omp parallel for` |
| `RECORD_FUNCTION(...)` | Profiling hooks in moe.cpp | Removed (add your own profiler) |
| `c10::IValue` arguments | RECORD_FUNCTION overhead | Removed |
| `torch::Tensor` schema defs | `common_extension.cc` registration | Isolated to `sgl_torch_binding.cpp` only |
| `<torch/extension.h>` | Every kernel | Removed from kernels; only in binding shim |

---

## Build (no torch required)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure    # runs test_kernels
```

Requires: GCC ≥ 11, `-mavx512bf16` support (Intel Sapphire Rapids or Icelake-SP+).

### With PyTorch binding shim

```bash
cmake .. -DBUILD_TORCH_BINDING=ON -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
```

This builds `sgl_cpu_kernels_torch.so` which is loadable as a Python extension. **Only `sgl_torch_binding.cpp` links against libtorch** — the kernel objects themselves remain framework-free.

---

## Integration into SGLang

### Drop-in replacement for `sgl-kernel` CPU ops

In `python/sglang/srt/layers/`:

```python
# Before (sgl-kernel ATen path):
import torch
torch.ops.sgl_kernel.rms_norm_cpu(out, x, weight, eps)

# After (call the standalone shim):
import sgl_cpu_kernels_torch as ck
ck.rms_norm(out, x, weight, eps)
```

Or for full framework removal, call via `ctypes`:

```python
import ctypes, numpy as np
lib = ctypes.CDLL("libsgl_cpu_kernels.so")
lib.sgl_rms_norm_bf16.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float
]
lib.sgl_rms_norm_bf16(out_ptr, x_ptr, w_ptr, n_tokens, hidden_dim, eps)
```

---

## File map

```
include/
  sgl_cpu_kernels.h   — Public C API (raw pointers, no framework types)
  sgl_vec.h           — AVX-512 BF16 intrinsic helpers

src/
  sgl_norm.cpp        — rms_norm, add_rms_norm
  sgl_activation.cpp  — silu_and_mul (split + inplace)
  sgl_rope.cpp        — RoPE Neox layout
  sgl_gemm.cpp        — GEMM, GEMM+bias, BMM
  sgl_moe.cpp         — fused_experts, shared_expert
  sgl_collective.cpp  — allreduce_sum (single-node TP)
  sgl_torch_binding.cpp  ← ONLY file that may include <torch/...>

tests/
  test_kernels.cpp    — Correctness unit tests (no torch)

CMakeLists.txt
```

---

## Performance notes (AVX-512 BF16 on SPR)

- **`_mm512_dpbf16_ps`** (BF16 dot-product): 2 bf16 pairs × 16 lanes = 32 MACs/cycle. Peak throughput on Sapphire Rapids is ~2 TFLOP/s per socket at BF16.
- **`sgl_gemm_bf16`**: uses `BLOCK_M=4, BLOCK_N=4, BLOCK_K=64` tiling. Tune `BLOCK_K` to match your L2 size (per-core L2 on SPR = 2MB).
- **`sgl_fused_experts_bf16`**: decode-phase (M=1 per token) uses a specialized `matvec_bf16_f32` path instead of the full tiled GEMM. For prefill (large M), swap to `sgl_gemm_bf16`.
- **`at::parallel_for` vs `#pragma omp parallel for`**: The ATen parallel backend ultimately calls OpenMP on Linux too, but adds ~500ns dispatch overhead per call. The direct OMP path eliminates this.
- **AllReduce**: The current `sgl_collective.cpp` uses per-element atomics which is correct but slow for large tensors. For production TP, replace with a ring-allreduce using per-rank staging buffers (preallocate once, reuse per forward pass).

---

## Remaining framework touchpoints (in SGLang Python layer)

These are outside the kernel library and are separate concerns:

- `torch.Tensor` memory allocation / device placement — keep as-is or replace with `posix_memalign` + a custom allocator
- `torch.ops` dispatch table — replaced by direct function calls or ctypes
- Python GIL overhead — use `torch.no_grad()` + `inference_mode()` to minimize, or bypass Python entirely with a C++ inference runner
