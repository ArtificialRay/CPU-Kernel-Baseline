# vLLM ARM CPU Kernel — Framework-Free Standalone Harness

Removes all PyTorch / LibTorch / c10 / pybind11 dependencies from the three
decode-phase bottleneck kernels so they can be compiled, tested, and
benchmarked on any AArch64 toolchain without a full vLLM build.

---

## ISA Feature Map

| Kernel | Base NEON | dotprod | BF16/BFMMLA | FP16 |
|--------|-----------|---------|-------------|------|
| rms_norm | ✅ | — | ✅ dispatch | ✅ |
| fused_add_rms_norm | ✅ | — | — | — |
| silu_and_mul | ✅ | — | — | ✅ |
| paged_attention_v1 | ✅ QK+V | ✅ (dotprod) | ✅ (bf16) | ✅ |
| paged_attention_v2 | ✅ | ✅ | ✅ | ✅ |

---

## Files

| File | Purpose |
|------|---------|
| `include/vllm_stub.h` | Stubs every PyTorch/c10 type and macro used by the kernels |
| `include/vllm_arm_kernels.h` | Framework-free, NEON-accelerated implementations |
| `test_vllm_arm_kernels.cpp` | Test + benchmark harness (17 tests, 0 deps) |
| `CMakeLists.txt` | Builds standalone target + optional real-kernel target |

---

## Framework Dependencies Removed

| Layer | Original | Stub |
|-------|----------|------|
| Tensor | `torch::Tensor` / `at::Tensor` | Flat `std::vector<uint8_t>` owner in `vllm_stub.h` |
| Scalar types | `c10::Half`, `c10::BFloat16`, `c10::ScalarType` | Minimal IEEE-correct implementations |
| Dispatch macros | `AT_DISPATCH_FLOATING_TYPES_AND2(...)` | C++17 `if constexpr` switch |
| Error checking | `TORCH_CHECK`, `TORCH_INTERNAL_ASSERT` | Throws `std::runtime_error` |
| Logging | `VLOG(n)` | No-op |
| Python bindings | `PYBIND11_MODULE`, `TORCH_LIBRARY` | No-op macros |
| OpenMP | `#pragma omp` | Falls back to serial if `_OPENMP` not defined |

---

## Kernels Provided

### 1. `rms_norm` — RMS Layer Normalization
- Used by every token produced in decode (LLaMA, Mistral, Qwen, Gemma …)
- Supports FP32, FP16, BF16
- NEON path: vectorized `vmlaq_f32` accumulation + `vaddvq_f32` reduce

### 2. `fused_add_rms_norm` — Residual-Add + RMSNorm
- Fuses `residual += x` and `rms_norm(residual)` into one pass over memory
- Saves one full `hidden_size` read/write vs calling separately
- Critical for decode latency: eliminates ~40% of memory bandwidth for the
  normalization stage

### 3. `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` — Gate Activations
- Applied inside every FFN block during both prefill and decode
- Supports FP32, FP16, BF16
- NEON path: batched 4-wide float32 lanes

### 4. `paged_attention_v1` — Single-Pass Decode Attention
- The decode-phase attention kernel over paged KV cache
- Handles MQA / GQA (`num_kv_heads < num_heads`)
- Respects vLLM's interleaved key cache layout `[blocks, kv_heads, H/x, block, x]`
- NEON path: `vmlaq_f32` for QK dot products and value accumulation

### 5. `paged_attention_v2` — Two-Pass Reduce Attention
- Used for long contexts where single-pass softmax overflows float32 range
- Pass 1: per-partition max, exp-sum, partial output
- Pass 2: globally re-weight and reduce all partitions
- Cross-checked against v1 in tests (max_diff < 1e-3)

---

## Build Instructions

### Host (x86/ARM, reference mode — zero deps)
```bash
g++ -std=c++17 -O2 -DVLLM_STANDALONE -I./include \
    test_vllm_arm_kernels.cpp -lm -o test_vllm_arm
./test_vllm_arm
```

### AArch64 native (NEON only, e.g. Graviton2 / Neoverse N1)
```bash
g++ -std=c++17 -O3 -march=armv8.2-a+dotprod+fp16 \
    -DVLLM_STANDALONE -I./include \
    test_vllm_arm_kernels.cpp -lm -lpthread -o test_vllm_arm_neon
./test_vllm_arm_neon
```

### AArch64 + BF16 (Graviton3 / Neoverse V1 / Apple M)
```bash
g++ -std=c++17 -O3 -march=armv8.2-a+dotprod+fp16+bf16 \
    -DVLLM_STANDALONE -DARM_BF16_SUPPORT -I./include \
    test_vllm_arm_kernels.cpp -lm -lpthread -o test_vllm_arm_bf16
./test_vllm_arm_bf16
```

### Cross-compile for Android AArch64 (via NDK)
```bash
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ \
    -std=c++17 -O3 -march=armv8.2-a+dotprod+fp16 \
    -DVLLM_STANDALONE -I./include \
    test_vllm_arm_kernels.cpp -lm -o test_vllm_arm_android
adb push test_vllm_arm_android /data/local/tmp/
adb shell /data/local/tmp/test_vllm_arm_android
```

### CMake (all targets)
```bash
# Standalone reference
cmake -B build && cmake --build build -j
./build/test_vllm_arm_standalone

# Real vLLM kernels on AArch64 with BF16
cmake -B build_real \
  -DVLLM_REAL_KERNELS=ON \
  -DVLLM_SRC_DIR=/path/to/vllm \
  -DARM_BF16=ON \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake
cmake --build build_real -j
```

### Wiring a real vLLM kernel (drop-in pattern)
```cpp
#define VLLM_STANDALONE
#define VLLM_REAL_KERNELS        // use actual csrc/cpu/attention.cpp
#include "vllm_stub.h"           // stubs PyTorch types
// #include "csrc/cpu/attention.cpp"   // compiled separately in CMake

// Then call exactly as vLLM does:
paged_attention_v1(out, query, key_cache, value_cache,
                   num_kv_heads, scale, block_tables, seq_lens,
                   block_size, max_seq_len, alibi_slopes,
                   "auto", 1.f);
```

