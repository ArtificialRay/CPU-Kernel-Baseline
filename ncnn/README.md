# Standalone ARM NEON Operators Extracted from NCNN

> **Note**: This project provides standalone ARM NEON-optimized operator implementations extracted from the [Tencent/ncnn](https://github.com/Tencent/ncnn) inference framework. These are self-contained kernels with **zero framework dependencies** that can be dropped into any C++11 project.

## Overview

NCNN is a high-performance neural network inference framework optimized for mobile platforms. Its ARM backend contains hand-tuned NEON kernels for common operators. This project extracts the core computation kernels into a single header file for educational purposes and standalone testing.

## Strategy

We extract the **pure computational kernels** from ncnn's `src/layer/arm/` implementations and package them as **header-only** functions. All framework dependencies (Mat, Allocator, Option, ParamDict) are removed.

```
BEFORE (ncnn framework)          AFTER (standalone)
────────────────────────         ─────────────────────────
convolution_arm.cpp               arm_neon_ops.h
  ├── layer.h                       ├── <arm_neon.h>
  │   ├── mat.h                     └── (no ncnn deps)
  │   ├── option.h
  │   └── allocator.h
  └── neon intrinsics            Pure NEON intrinsics (unchanged)
```

---

## Project Structure

```
ncnn/
├── CMakeLists.txt                # Build configuration
├── README.md                      # This document
├── Operators.md                   # Operator coverage table
├── arm_neon_ops.h                 # Standalone NEON kernels
└── test_ops.cpp                   # Unit tests
```

**Single header**: `arm_neon_ops.h` contains all operators (conv2d, depthwise, linear, normalization, etc.)

---

## Operators Included

See [Operators.md](Operators.md) for detailed coverage. Summary:

| Operator | Implementation | Format |
|----------|----------------|--------|
| **conv2d** | NEON `vmlaq_f32` vectorized over output width | NCHW |
| **depthwise_conv2d** | Per-channel NEON accumulation | NCHW |
| **linear** | GEMV with NEON + horizontal reduction | [N, IC] × [OC, IC]ᵀ |
| **batchnorm** | Fused scale + shift | NCHW |
| **layernorm** | NEON mean/variance + normalize | Any dims |
| **softmax** | NEON max reduction + exp/normalize | Numerically stable |
| **relu** | `vmaxq_f32(v, vzero)` | In-place safe |
| **eltwise_add** / **mul** | `vaddq_f32` / `vmulq_f32` | Broadcasting |
| **gru_cell** | NEON-accelerated gates | Single timestep |
| **scaled_dot_product** | Q@Kᵀ softmax @V | Attention primitive |

All operators fall back to scalar C++ when compiled without NEON (`-DNCNN_NO_NEON`).

---

## Build Instructions (ARM SIMD Optimized)

> **Note**: Building with NEON requires an ARM CPU or cross-compilation toolchain.

### Option 1: Cross-compile for AArch64 Linux

Install the cross-compiler:
```bash
sudo apt-get install g++-aarch64-linux-gnu
```

Build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
make -j$(nproc)

# Transfer to ARM device and run
scp test_ops user@arm-device:/tmp/
ssh user@arm-device /tmp/test_ops
```

Output: `test_ops` (ARM64 binary with NEON optimization)

### Option 2: Cross-compile for AArch64 Android

Requires Android NDK:
```bash
mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24
make -j$(nproc)

# Push to device and run
adb push test_ops /data/local/tmp/
adb shell /data/local/tmp/test_ops
```

### Option 3: Cross-compile for ARMv7 (32-bit)

ARMv7 requires explicit `-mfpu=neon` flag:
```bash
sudo apt-get install g++-arm-linux-gnueabihf

mkdir build-armv7 && cd build-armv7
cmake .. \
  -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
  -DCMAKE_SYSTEM_PROCESSOR=armv7
make -j$(nproc)
```

### Option 4: Native build on ARM device

If you're already on an ARM64/ARMv7 Linux device (Raspberry Pi, Jetson, etc.):
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./test_ops
```

### Scalar Fallback (for x86_64 CI/testing)

To compile on x86_64 with scalar fallback (no NEON):
```bash
mkdir build && cd build
cmake .. -DNEON_FALLBACK=ON
make -j$(nproc)
./test_ops  # Runs scalar C++ implementation
```

---

## How to Use in Your Project

### Drop-in usage

Simply copy `arm_neon_ops.h` to your project:

```cpp
#include "arm_neon_ops.h"

int main() {
    // Allocate buffers
    std::vector<float> input(3 * 224 * 224);   // [IC=3, IH=224, IW=224]
    std::vector<float> weight(64 * 3 * 3 * 3); // [OC=64, IC=3, KH=3, KW=3]
    std::vector<float> bias(64);               // [OC=64]
    std::vector<float> output(64 * 222 * 222); // [OC=64, OH=222, OW=222]
    
    // ... fill input, weight, bias ...
    
    // Run NEON-accelerated convolution
    ops::conv2d_nchw(
        input.data(), weight.data(), bias.data(), output.data(),
        /*IC=*/3, /*IH=*/224, /*IW=*/224,
        /*OC=*/64, /*KH=*/3, /*KW=*/3
    );
    
    return 0;
}
```

Compile:
```bash
# ARM64
aarch64-linux-gnu-g++ -O3 -std=c++11 -I. my_app.cpp -o my_app

# x86 (scalar fallback)
g++ -O2 -std=c++11 -DNCNN_NO_NEON -I. my_app.cpp -o my_app
```

---

## Implementation Details

### Memory Layout

All operators use **NCHW** layout:
- Convolution: `[Batch, OutChannels, Height, Width]`
- Input tensors: contiguous row-major arrays
- No padding/alignment requirements

### Vectorization Strategy

**Conv2d example**:
```cpp
// Inner loop over output width (OW)
for (int ow = 0; ow + 3 < OW; ow += 4) {
    float32x4_t vin  = vld1q_f32(input + ow);   // Load 4 pixels
    float32x4_t vout = vld1q_f32(output + ow);  // Load current output
    float32x4_t vw   = vdupq_n_f32(weight);     // Broadcast weight
    vout = vmlaq_f32(vout, vin, vw);            // FMA: out += in * w
    vst1q_f32(output + ow, vout);               // Store result
}
// Handle tail with scalar code
```

**GRU cell**: Uses NEON for matrix-vector products (reset gate, update gate, new hidden state) with fused sigmoid/tanh activations.

**Softmax**: Numerically stable via `max` subtraction before `exp`, all vectorized with NEON.

---

## Testing

Unit tests in `test_ops.cpp` compare NEON results against **golden reference** implementations (plain C++):

```bash
cd build
./test_ops
```

Expected output:
```
[PASS] conv2d correctness
[PASS] depthwise_conv2d correctness
[PASS] linear correctness
[PASS] batchnorm correctness
[PASS] layernorm correctness
[PASS] softmax correctness
[PASS] relu correctness
[PASS] gru_cell correctness
[PASS] scaled_dot_product correctness

=== 9 passed, 0 failed ===
```

---

## Differences from NCNN Framework

| Aspect | NCNN Framework | This Project |
|--------|----------------|--------------|
| **Dependencies** | Requires full ncnn build | Header-only, zero deps |
| **Memory** | Uses `ncnn::Mat` allocator | Plain `float*` pointers |
| **Operators** | 100+ layers | ~10 core operators |
| **Optimization** | Winograd, im2col, assembly | Baseline NEON only |
| **Quantization** | INT8, FP16 support | FP32 only |
| **Layout** | NCHW, NCHW4, NCHW8 | NCHW only |

This project focuses on **educational clarity** rather than maximum performance.

---

## References

- **NCNN GitHub**: https://github.com/Tencent/ncnn
- **ARM NEON Intrinsics Guide**: https://developer.arm.com/architectures/instruction-sets/intrinsics/
- **NCNN Convolution Implementation**: `src/layer/arm/convolution_arm.cpp`

---
