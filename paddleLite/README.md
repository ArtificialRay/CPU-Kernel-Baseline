# Removing Paddle-Lite Framework Dependencies for ARM Kernel Unit Tests

> **Note**: This project provides ARM NEON/SVE optimized kernel implementations extracted from [PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite). The sources in `lite/backends/arm/math/` and `lite/kernels/arm/` contain hand-optimized SIMD code that **requires ARM CPU or cross-compilation toolchain** to build and run.

## Strategy

We **stub out every layer above `lite/backends/arm/math/`** in a single header
(`paddle_lite_stub.h`), leaving the actual NEON/SVE math untouched.

```
BEFORE                           AFTER
──────────────────────────       ─────────────────────────────
conv_compute.h                   conv_compute.h
  ├── kernel.h                     ├── paddle_lite_stub.h   ← single shim
  │   ├── op_params.h              │   contains all stubs
  │   ├── context.h                └── (no real paddle deps)
  │   ├── types.h
  │   └── utils/log.h
  └── backends/arm/math/…        backends/arm/math/…  (unchanged)
```

---

## What each stub replaces

### 1. `TargetType` / `PrecisionType` / `DataLayoutType` enums
**Original**: `lite/core/types.h`  
**Stub**: plain `enum class` in `namespace paddle::lite`  
**Why it matters**: Every `KernelLite<TARGET(kARM), PRECISION(kFloat)>` template
instantiation depends on these.

### 2. `DDim` and `Tensor`
**Original**: `lite/core/tensor.h` — depends on `lite/core/memory.h`,
`lite/core/target_wrapper.h`, and eventually the ARM memory allocator.  
**Stub**: `DDim` = `std::vector<int64_t>` wrapper; `Tensor` = owns a
`std::vector<uint8_t>` buffer.  
**Key interface preserved**:
- `Resize(DDim)` / `Resize(vector<int64_t>)`
- `mutable_data<T>()` / `data<T>()`
- `numel()`, `dim(i)`, `dims()`, `dims_size()`

### 3. `Scope`
**Original**: `lite/core/scope.h` — depends on protobuf `VarDesc`, the
variable-store, and reference counting.  
**Stub**: thin `std::map<std::string, Tensor>`.  
**Why it matters**: Some kernels call `scope->FindMutableTensor(name)` inside
`AttachImpl`. In standalone tests you set `param.x = &my_tensor` directly,
bypassing `AttachImpl` entirely.

### 4. `ConvParam`, `FcParam`, etc. (`op_params.h`)
**Original**: `lite/operators/op_params.h` — huge header, depends on `Tensor`,
`DDim`, activation enums, and various lite utilities.  
**Stub**: Plain POD structs with the exact same field names and types.  
**Fields included** (everything the ARM kernel's `Run()` actually reads):

```cpp
// ConvParam — fields accessed by conv_compute.cc
struct ConvParam {
    const Tensor* x, *filter, *bias;
    Tensor*       output;
    std::vector<int> strides, paddings, dilations;
    int           groups;
    bool          fuse_relu;
    ActivationParam activation_param;
    float         input_scale, output_scale;
    std::vector<float> weight_scale;   // per-channel INT8
    int           use_winograd;        // set by PrepareForRun
};
```

### 5. `KernelLite<Target, Precision>` base class
**Original**: `lite/core/kernel.h` — pulls in the entire type system, profiler,
and the `REGISTER_LITE_KERNEL` macro which links to the op-registry.  
**Stub**:
```cpp
template<TargetType T, PrecisionType P, DataLayoutType L = kNCHW>
class KernelLite {
public:
    virtual void PrepareForRun() {}
    virtual void Run() = 0;
    template<typename ParamT> void SetParam(const ParamT&);
    void SetContext(Context*);
protected:
    void*    param_raw_ = nullptr;
    Context* ctx_       = nullptr;
};
```
`REGISTER_LITE_KERNEL(...)` is `#define`-d to nothing.

### 6. `ARMContext` / `KernelContext<kARM>`
**Original**: `lite/core/context.h` — depends on the thread pool, device
detection at runtime, and the workspace allocator.  
**Stub**: Flat struct with `threads_`, `workspace_` (a `vector<uint8_t>`),
and capability flags (`has_dot`, `has_fp16`, `has_sve2`).  
**How to set capabilities** in your test:
```cpp
ARMContext ctx;
ctx.set_has_dot(true);   // simulate Cortex-A55/A510 with SDOT
ctx.set_has_fp16(true);  // simulate ARMv8.2 FP16
ctx.set_threads(4);
```

### 7. `DeviceInfo`
**Original**: `lite/backends/arm/math/device_info.h` — runs CPUID probing,
reads `/proc/cpuinfo`.  
**Stub**: Singleton with settable fields — no filesystem access.

### 8. Logging (`LOG`, `CHECK`, `VLOG`)
**Original**: `lite/utils/log.h` — custom log sink, glog dependency.  
**Stub**: `LOG(x)` → `std::cerr`, `CHECK(cond)` → `throw std::runtime_error`.

---

## Project Structure

```
paddleLite/
├── CMakeLists.txt                      # Build configuration
├── GUIDE.md                             # This document
├── paddle_lite_stub.h                   # Stub header (zero deps)
├── test_arm_kernels_standalone.cc       # Unit tests (reference impl)
├── examples.cc                          # Usage examples (with real kernels)
└── lite/                                # Paddle-Lite source subset
    ├── backends/arm/math/               # 97 .cc + 90 .h: NEON kernels
    │   ├── sgemm.cc, sgemv.cc
    │   ├── conv_direct*.cc, conv_winograd*.cc
    │   ├── gemm_prepacked_int8.cc
    │   ├── batch_norm.cc, norm.cc
    │   ├── lstm.cc, gru_utils.cc
    │   └── ... (activation, pool, elementwise)
    └── kernels/arm/                     # 81 .cc + 67 .h: kernel compute
        ├── conv_compute.cc
        ├── fc_compute.cc
        ├── batch_norm_compute.cc
        ├── layer_norm_compute.cc
        └── ... (all operators)
```

**Total**: 178 .cc files, 157 .h files from Paddle-Lite upstream

---

## Dependency graph after stubbing

```
test_arm_kernels_standalone.cc
  └── paddle_lite_stub.h          (zero external deps)
      └── <arm_neon.h>            (compiler-provided)

[when WITH_REAL_KERNELS=ON]
  └── examples.cc
      ├── paddle_lite_stub.h
      ├── lite/kernels/arm/conv_compute.cc
      └── lite/backends/arm/math/sgemm.cc   ← pure NEON, no framework deps
          └── lite/backends/arm/math/conv_winograd*.cc
```

---

## Build instructions (ARM SIMD Optimized)

> **Note**: Building with real ARM kernels requires an ARM CPU or cross-compilation toolchain. These instructions compile the actual NEON/SVE optimized implementations from `lite/`.

### Cross-compile for AArch64 Linux (using GNU toolchain)

Install the cross-compiler:
```bash
sudo apt-get install g++-aarch64-linux-gnu
```

Build:
```bash
mkdir build-arm64 && cd build-arm64
cmake .. \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_REAL_KERNELS=ON \
  -DPADDLE_LITE_ROOT=.. \
  -DWITH_ARM_DOTPROD=ON \
  -DWITH_ARM_FP16=ON
make -j$(nproc)

# Transfer to ARM device and run
scp test_arm_kernels_real user@arm-device:/tmp/
ssh user@arm-device /tmp/test_arm_kernels_real
```

Output: `test_arm_kernels_real` (ARM64 binary with NEON, SDOT, FP16)

### Build flags explained

- `-DWITH_REAL_KERNELS=ON`: Compile actual ARM kernels from `lite/` (required)
- `-DPADDLE_LITE_ROOT=..`: Path to local Paddle-Lite sources (use `..` for parent dir)
- `-DWITH_ARM_DOTPROD=ON`: Enable ARMv8.2 INT8 dot-product (SDOT/UDOT instructions)
- `-DWITH_ARM_FP16=ON`: Enable ARMv8.2 FP16 arithmetic

**To disable advanced features** (for older ARM CPUs), omit the corresponding flags:
```bash
cmake .. \
  -DWITH_REAL_KERNELS=ON \
  -DPADDLE_LITE_ROOT=..
  # This builds with baseline ARMv8 NEON only
```

---

## How to write a new kernel unit test

### Option 1: Reference implementation only

For testing algorithms without ARM SIMD (can run on x86_64):

```cpp
// test_my_op.cc
#include "paddle_lite_stub.h"
#include <cassert>
#include <cmath>

// Reference implementation
void ref_my_op(const Tensor& x, Tensor& y) {
    const float* px = x.data<float>();
    float* py = y.mutable_data<float>();
    for (int64_t i = 0; i < x.numel(); ++i)
        py[i] = std::sqrt(px[i]);  // example: element-wise sqrt
}

int main() {
    Tensor x, y;
    x.Resize({2, 3, 4}); y.Resize({2, 3, 4});
    // ... fill x with data ...
    ref_my_op(x, y);
    // ... verify y ...
    return 0;
}
```

Build: `g++ -std=c++14 -I. test_my_op.cc -lm -o test_my_op` (works on any platform)

### Option 2: With real ARM kernel (requires ARM CPU or cross-compiler)

```cpp
// 1. Include the stub (before any kernel header)
#include "paddle_lite_stub.h"

// 2. Include the kernel header (from local lite/ folder)
#ifdef PADDLE_LITE_USE_REAL_KERNEL
#include "lite/kernels/arm/batch_norm_compute.h"
#endif

// 3. Build param struct — field names match op_params.h exactly
void test_batch_norm() {
    BatchNormParam param;
    Tensor x, scale, bias, mean, var, y;
    
    x.Resize({1,4,8,8});
    scale.Resize({4});  fill_const(scale, 1.f);
    bias.Resize({4});   fill_const(bias,  0.f);
    mean.Resize({4});   fill_const(mean,  0.f);
    var.Resize({4});    fill_const(var,   1.f);
    y.Resize({1,4,8,8});
    
    param.x = &x; param.scale = &scale; param.bias = &bias;
    param.mean = &mean; param.variance = &var; param.y = &y;
    param.epsilon = 1e-5f; param.is_test = true;

    // 4. Instantiate context
    ARMContext ctx(/*threads=*/1);

#ifdef PADDLE_LITE_USE_REAL_KERNEL
    // 5. Instantiate, wire, run
    paddle::lite::kernels::arm::BatchNormCompute kernel;
    kernel.SetParam(param);
    kernel.SetContext(&ctx);
    kernel.PrepareForRun();   // weight packing, workspace alloc
    kernel.Run();
#endif

    // 6. Compare against reference
    Tensor y_ref; y_ref.Resize({1,4,8,8});
    ref_batch_norm(x, scale, bias, mean, var, y_ref);
    
#ifdef PADDLE_LITE_USE_REAL_KERNEL
    float diff = max_abs_diff(y, y_ref);
    assert(diff < 1e-5f);
#endif
}
```

Build with CMake (requires ARM toolchain):
```bash
# On ARM device
cmake .. -DWITH_REAL_KERNELS=ON -DPADDLE_LITE_ROOT=..
make

# Or cross-compile from x86_64
cmake .. \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DWITH_REAL_KERNELS=ON \
  -DPADDLE_LITE_ROOT=.. \
  -DWITH_ARM_DOTPROD=ON
make
```

---

## Operators covered by the stubs

| Category | Operators | Param struct |
|---|---|---|
| Convolution | conv2d, depthwise_conv2d, conv2d_transpose, deformable_conv | `ConvParam` |
| Linear | fc, mul, matmul, matmul_v2 | `FcParam`, `MatMulParam` |
| Normalization | batch_norm, layer_norm, instance_norm, group_norm | `BatchNormParam`, `LayerNormParam`, `InstanceNormParam`, `GroupNormParam` |
| Recurrent | lstm, gru, gru_unit | `LstmParam`, `GruParam` |
| Activation | relu, sigmoid, tanh, hard_swish, mish, clip, … | `ActivationComputeParam` |
| Elementwise | add, sub, mul, div, … | `ElementwiseParam` |
| Pooling | max_pool, avg_pool | `PoolParam` |
| Softmax | softmax | `SoftmaxParam` |
