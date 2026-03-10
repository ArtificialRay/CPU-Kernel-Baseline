# Standalone ARM NEON Kernels Extracted from llama.cpp/ggml

> **Note**: This project provides ARM NEON-optimized operator implementations extracted from [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) (ggml tensor library). These are the **actual bottleneck kernels** used by LLaMA and other LLM inference engines, packaged as a **single header** with **zero dependencies**.

## Overview

**ggml** is the tensor library powering llama.cpp, whisper.cpp, and many other efficient LLM inference engines. Its ARM backend contains hand-optimized NEON/dotprod/i8mm kernels for transformer workloads (RMSNorm, RoPE, quantized MatMul).

This project extracts the **hottest code paths** from:
- `ggml/src/ggml-cpu/vec.h` — vectorized element-wise ops + reductions
- `ggml/src/ggml-cpu/ops.cpp` — rms_norm, norm, softmax, rope
- `ggml/src/ggml-cpu/ggml-quants.c` — Q4_0, Q8_0 quantized vec_dot (80%+ of LLM runtime)
- `ggml/src/ggml-impl.h` — fp16 ↔ fp32 conversion

All framework wiring (ggml_context, ggml_tensor, graph scheduling) is removed.

---

## Strategy

We extract **pure compute kernels** and package them as **header-only** NEON intrinsics. No ggml headers, no BLAS, no threading — just the math.

```
BEFORE (ggml framework)          AFTER (standalone)
───────────────────────          ─────────────────────────
ggml-backend.c                    llama_neon_ops.h
  ├── ggml.h                        ├── <arm_neon.h>
  │   ├── ggml-alloc.h              └── (no ggml deps)
  │   ├── ggml-backend.h
  │   └── ggml-quants.h
  └── NEON intrinsics            Pure NEON intrinsics (unchanged)
```

---

## Project Structure

```
ggml/
├── CMakeLists.txt                # Build configuration  
├── README.md                      # This document
├── llama_neon_ops.h               # Standalone NEON kernels header
└── test_llama_ops.cpp             # Unit tests
```

**Single header**: `llama_neon_ops.h` (~2000 lines) contains all operators critical for LLM inference.

---

## Operators Included

Focus on **transformer-specific** operations:

| Operator | Implementation | Purpose |
|----------|----------------|---------|
| **rms_norm** | NEON `vrsqrte_f32` + Newton-Raphson | LLaMA normalization |
| **rope** | NEON `sin/cos` + complex mul | Rotary Position Embedding |
| **softmax** | NEON max/exp/sum | Attention scores |
| **silu** | NEON `sigmoid` + `vmul` | SwiGLU activation |
| **mul_mat** (FP32) | NEON `vmlaq_f32` GEMV | Small batch inference |
| **vec_dot_q4_0_q8_0** | NEON `vdotq_s32` (dotprod) | Quantized MatMul (Q4×Q8) |
| **vec_dot_q8_0_q8_0** | NEON `vdotq_s32` or `vmmlaq_s32` (i8mm) | INT8 MatMul |
| **dequantize_q4_0** | NEON unpack + `vmulq_f32` | Q4_0 → FP32 |
| **quantize_q8_0** | NEON abs_max + scale + `vcvt_s8` | FP32 → Q8_0 |
| **fp16 ↔ fp32** | `vcvt_f16_f32` / `vcvt_f32_f16` | Mixed precision |

All operators have **scalar fallback** for x86 CI (`-DLLAMA_NO_NEON`).

---

## Build Instructions (ARM SIMD Optimized)

> **Note**: Building with NEON/dotprod/i8mm requires an ARM CPU or cross-compilation toolchain.

### Option 1: Cross-compile for AArch64 Linux (Baseline NEON)

Install cross-compiler:
```bash
sudo apt-get install g++-aarch64-linux-gnu
```

Build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
make -j$(nproc)

# Transfer to ARM device and run
scp test_llama_ops user@arm-device:/tmp/
ssh user@arm-device /tmp/test_llama_ops
```

Output: `test_llama_ops` (baseline ARMv8 NEON)

### Option 2: With DOTPROD Extension (Cortex-A55/A75+, Apple M1+)

Enables `vdotq_s32` for 4× faster INT8 MatMul:
```bash
mkdir build-dotprod && cd build-dotprod
cmake .. \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DENABLE_DOTPROD=ON
make -j$(nproc)
```

This sets `-march=armv8.2-a+dotprod`.

### Option 3: With I8MM Extension (Neoverse V1, Apple M2+)

Enables `vmmlaq_s32` for 8× faster INT8 MatMul:
```bash
mkdir build-i8mm && cd build-i8mm
cmake .. \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DENABLE_I8MM=ON
make -j$(nproc)
```

This sets `-march=armv8.6-a+dotprod+i8mm`.

### Option 4: Cross-compile for Android

Requires Android NDK:
```bash
mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DENABLE_DOTPROD=ON
make -j$(nproc)

adb push test_llama_ops /data/local/tmp/
adb shell /data/local/tmp/test_llama_ops
```

### Option 5: Native build on ARM device

If you're on an ARM64 Linux device (Raspberry Pi 4+, Jetson, MacBook M1/M2):
```bash
mkdir build && cd build
cmake .. -DENABLE_DOTPROD=ON  # or -DENABLE_I8MM=ON for M2+
make -j$(nproc)
./test_llama_ops
```

### Scalar Fallback (x86_64 CI/testing)

To compile on x86_64 with scalar fallback (no NEON):
```bash
mkdir build && cd build
cmake .. -DNEON_FALLBACK=ON
make -j$(nproc)
./test_llama_ops  # Runs scalar C++ implementation
```

---

## How to Use in Your Project

### Drop-in usage

Copy `llama_neon_ops.h` to your project:

```cpp
#include "llama_neon_ops.h"

int main() {
    using namespace llama_ops;
    
    // Example 1: RMSNorm (LLaMA's normalization)
    const int D = 4096;
    std::vector<float> x(D), weight(D), out(D);
    // ... fill x, weight ...
    rms_norm(out.data(), x.data(), weight.data(), D, 1e-6f);
    
    // Example 2: Quantized MatMul (Q4_0 weight × Q8_0 activation)
    const int M = 1, K = 4096, N = 11008;
    std::vector<block_q4_0> weight_q4(N * K / QK4_0);  // Quantized weights
    std::vector<block_q8_0> input_q8(M * K / QK8_0);   // Quantized input
    std::vector<float> output(M * N);
    // ... quantize weights and input ...
    mul_mat_q4_0_q8_0(output.data(), weight_q4.data(), input_q8.data(), M, N, K);
    
    // Example 3: Rotary Position Embedding (RoPE)
    const int n_tokens = 32, n_heads = 32, head_dim = 128;
    std::vector<float> q(n_tokens * n_heads * head_dim);
    std::vector<float> pos_freqs(head_dim / 2);
    // ... fill q, pos_freqs ...
    rope_neox(q.data(), pos_freqs.data(), n_tokens, n_heads, head_dim);
    
    return 0;
}
```

Compile:
```bash
# ARM64 with dotprod
aarch64-linux-gnu-g++ -O3 -std=c++11 -march=armv8.2-a+dotprod -I. my_llm.cpp -o my_llm

# x86 (scalar fallback)
g++ -O2 -std=c++11 -DLLAMA_NO_NEON -I. my_llm.cpp -o my_llm
```

---

## Implementation Details

### Quantization Formats

#### Q4_0: 4-bit grouped quantization (ggml's default)
```cpp
struct block_q4_0 {
    ggml_fp16_t d;      // Delta (scale)
    uint8_t qs[QK4_0/2]; // Quantized values (16 values packed in 8 bytes)
};
// QK4_0 = 32 (block size)
```
Each block stores 32 values in 18 bytes (16× compression vs FP32).

#### Q8_0: 8-bit quantization
```cpp
struct block_q8_0 {
    ggml_fp16_t d;      // Delta (scale)
    int8_t qs[QK8_0];    // Quantized values
};
// QK8_0 = 32
```
Used for activation quantization during inference.

### Critical Path: Quantized MatMul

**Q4_0 × Q8_0 dot product** is the hottest code path (70-80% of LLM runtime):

```cpp
// With ARM dotprod (4× faster than baseline NEON)
for (int i = 0; i < nb; i++) {
    const float d0 = GGML_FP16_TO_FP32(x[i].d);  // Weight scale
    const float d1 = GGML_FP16_TO_FP32(y[i].d);  // Activation scale
    
    const uint8x16_t v0_0 = vld1q_u8(x[i].qs);      // Load 16 Q4 values (packed)
    const int8x16_t  v1_0 = vld1q_s8(y[i].qs);      // Load 16 Q8 values
    
    // Unpack 4-bit to 8-bit, then dotprod
    int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    v0_0l = vsubq_s8(v0_0l, s8b);  // Remove 8 offset
    v0_0h = vsubq_s8(v0_0h, s8b);
    
    sumv0 = vdotq_s32(sumv0, v0_0l, v1_0);  // 4-way dot product
    sumv1 = vdotq_s32(sumv1, v0_0h, v1_1);
    
    summs += d0 * d1 * (horizontal_sum(sumv0) + horizontal_sum(sumv1));
}
```

**With i8mm** (`vmmlaq_s32`), this is another 2× faster.

### RMSNorm (LLaMA Normalization)

```cpp
// 1. Compute RMS = sqrt(mean(x²))
float sum_sq = 0.f;
for (int i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    sum_sq += horizontal_sum(vmulq_f32(v, v));  // x²
}
float rms = sqrtf(sum_sq / n + eps);

// 2. Normalize: y = (x / rms) * weight
float inv_rms = 1.f / rms;
float32x4_t vrms = vdupq_n_f32(inv_rms);
for (int i = 0; i < n; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    float32x4_t vw = vld1q_f32(weight + i);
    float32x4_t vy = vmulq_f32(vmulq_f32(vx, vrms), vw);
    vst1q_f32(y + i, vy);
}
```

### RoPE (Rotary Position Embedding)

For each token position, rotate Q/K vectors by angle θ:
```cpp
for (int p = 0; p < n_tokens; p++) {
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + (p * n_heads + h) * head_dim;
        for (int d = 0; d < head_dim; d += 2) {
            float cos_theta = pos_freqs[d/2];
            float sin_theta = pos_freqs[d/2 + head_dim/2];
            // Complex multiplication: (q0 + i*q1) * (cos + i*sin)
            float q0 = q_head[d+0], q1 = q_head[d+1];
            q_head[d+0] = q0 * cos_theta - q1 * sin_theta;
            q_head[d+1] = q0 * sin_theta + q1 * cos_theta;
        }
    }
}
```

---

## Testing

Unit tests in `test_llama_ops.cpp` compare NEON results against reference implementations:

```bash
cd build
./test_llama_ops
```

Expected output:
```
[PASS] rms_norm correctness
[PASS] rope_neox correctness
[PASS] softmax correctness
[PASS] silu correctness
[PASS] mul_mat_f32 correctness
[PASS] vec_dot_q4_0_q8_0 correctness
[PASS] vec_dot_q8_0_q8_0 correctness
[PASS] quantize_q8_0 correctness
[PASS] dequantize_q4_0 correctness
[PASS] fp16_fp32_conversion correctness

=== 10 passed, 0 failed ===
```

---

## Performance Notes

### Quantization Speedup (LLaMA-7B on Cortex-A76)

| Format | Instruction | Relative Speed |
|--------|-------------|----------------|
| **FP32** | `vmlaq_f32` | 1.0× (baseline) |
| **Q8×Q8** (NEON) | `vmull_s8` + reduce | ~2.5× |
| **Q8×Q8** (dotprod) | `vdotq_s32` | ~4.0× |
| **Q8×Q8** (i8mm) | `vmmlaq_s32` | ~8.0× |
| **Q4×Q8** (dotprod) | `vdotq_s32` + unpack | ~6.0× |

### Memory Bandwidth

Quantization reduces DRAM traffic by 4-8×:
- **FP32 weights**: 28 GB for 7B model
- **Q4_0 weights**: 3.5 GB for 7B model

This is critical for mobile/edge deployment where memory bandwidth is limited.

---

## Differences from llama.cpp/ggml

| Aspect | ggml Framework | This Project |
|--------|----------------|--------------|
| **Dependencies** | ggml.h, ggml-backend.h, threading | Header-only, zero deps |
| **Memory** | `ggml_context`, graph allocator | Plain `float*` pointers |
| **Operators** | 100+ ggml ops | ~10 LLM-critical ops |
| **Models** | Full LLaMA, GPT, Whisper support | Kernel-level only |
| **Quantization** | Q4_0/1, Q5_0/1, Q8_0, K-quants | Q4_0, Q8_0 only |
| **Backends** | CPU, CUDA, Metal, Vulkan | ARM NEON only |

This project is **educational** — for production LLM inference, use llama.cpp directly.

---

## ARM ISA Extensions

### ARMv8.0 (baseline)
- **NEON**: 128-bit SIMD, FP32/INT32/INT16/INT8 operations
- Used by: Cortex-A53, A72, Apple A10

### ARMv8.2-a+dotprod
- **Adds**: `SDOT/UDOT` 4-way INT8 dot product
- Used by: Cortex-A55, A75, A76, A77, Apple M1, Snapdragon 855+
- **Speedup**: 4× for INT8 MatMul

### ARMv8.6-a+i8mm
- **Adds**: `SMMLA/UMMLA` 8-way INT8 matrix multiply
- Used by: Neoverse V1, Apple M2/M3, Snapdragon 8 Gen 2+
- **Speedup**: 8× for INT8 MatMul

---

## References

- **llama.cpp GitHub**: https://github.com/ggerganov/llama.cpp
- **ggml tensor library**: `ggml/src/ggml-cpu/`
- **ARM NEON Intrinsics**: https://developer.arm.com/architectures/instruction-sets/intrinsics/
- **LLaMA Paper**: https://arxiv.org/abs/2302.13971 (RMSNorm, SwiGLU, RoPE)
- **GGML Quantization**: https://github.com/ggerganov/llama.cpp/pull/1684

---

## License

This code is extracted from llama.cpp/ggml, which is licensed under MIT License.  
See llama.cpp's [LICENSE](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE).
