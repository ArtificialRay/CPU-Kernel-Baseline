// paddle_lite_stub.h
// Drop-in replacement for all Paddle-Lite framework headers needed to
// compile and unit-test ARM kernels in isolation (no PaddlePaddle build).
//
// Covers the framework surface used by:
//   lite/kernels/arm/  (conv, fc/matmul, batch_norm, layer_norm, lstm/gru, …)
//   lite/backends/arm/math/  (NEON/SDOT primitives)
//
// Usage:
//   #define PADDLE_LITE_STANDALONE  (before including any kernel header)
//   #include "paddle_lite_stub.h"
//   … then include the kernel .h you want to test.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef PADDLE_LITE_STUB_H_
#define PADDLE_LITE_STUB_H_

// ---------------------------------------------------------------------------
// 0. Guard: prevent real Paddle-Lite headers from being pulled in
// ---------------------------------------------------------------------------
#define LITE_WITH_ARM 1

// Silence macro-heavy registration machinery — we don't need the op-registry.
#define REGISTER_LITE_KERNEL(...)
#define REGISTER_LITE_OP(...)
#define USE_LITE_KERNEL(...)
#define LITE_KERNEL_FUNC_IMPL(...)

// ---------------------------------------------------------------------------
// 1. Standard / platform headers
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// ARM NEON intrinsics (present when compiling for AArch32/AArch64)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SVE)
#  include <arm_sve.h>
#endif

// ---------------------------------------------------------------------------
// 2. Precision / Target / Layout enums  (mirrors lite/core/types.h)
// ---------------------------------------------------------------------------
namespace paddle { namespace lite {

enum class TargetType  : int { kUnk=0, kHost, kX86, kCUDA, kARM, kOpenCL, kNNAdapter };
enum class PrecisionType : int { kUnk=0, kFloat, kInt8, kInt32, kInt64, kFP16, kAny };
enum class DataLayoutType : int { kUnk=0, kNCHW, kNHWC, kImageDefault, kAny };

// Convenience aliases used throughout Paddle-Lite source
using TargetT    = TargetType;
using PrecisionT = PrecisionType;
using LayoutT    = DataLayoutType;

// Macros that appear in kernel registration (no-ops here)
#define TARGET(x)    paddle::lite::TargetType::x
#define PRECISION(x) paddle::lite::PrecisionType::x
#define LAYOUT(x)    paddle::lite::DataLayoutType::x

// ---------------------------------------------------------------------------
// 3. DDim  (mirrors lite/core/tensor.h DDim)
// ---------------------------------------------------------------------------
struct DDim {
    std::vector<int64_t> data_;

    DDim() = default;
    explicit DDim(std::vector<int64_t> d) : data_(std::move(d)) {}

    int64_t  operator[](int i)   const { return data_[i]; }
    int64_t& operator[](int i)         { return data_[i]; }
    size_t   size()              const { return data_.size(); }
    bool     empty()             const { return data_.empty(); }

    int64_t production() const {
        int64_t p = 1;
        for (auto v : data_) p *= v;
        return p;
    }
    int64_t count(int s, int e) const {
        int64_t p = 1;
        for (int i = s; i < e; ++i) p *= data_[i];
        return p;
    }
    DDim Slice(int s, int e) const {
        return DDim({data_.begin()+s, data_.begin()+e});
    }
    std::vector<int64_t> Vectorize() const { return data_; }
};

// ---------------------------------------------------------------------------
// 4. Tensor  (minimal, owns its buffer)
// ---------------------------------------------------------------------------
class Tensor {
public:
    Tensor() = default;

    // Resize / allocate
    void Resize(const DDim& dims) {
        dims_   = dims;
        size_t n = static_cast<size_t>(dims.production());
        buf_.resize(n * elem_bytes_);
    }
    void Resize(std::vector<int64_t> d) { Resize(DDim(d)); }

    const DDim& dims()  const { return dims_;  }
    DDim        ddim()  const { return dims_;  }
    int64_t     numel() const { return dims_.production(); }

    // typed access
    template<typename T>
    T* mutable_data() {
        elem_bytes_ = sizeof(T);
        buf_.resize(static_cast<size_t>(dims_.production()) * sizeof(T));
        return reinterpret_cast<T*>(buf_.data());
    }
    template<typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(buf_.data());
    }

    // raw pointer (used by some backends)
    void* raw_data() { return buf_.data(); }

    // shape helpers
    int dims_size() const { return static_cast<int>(dims_.size()); }
    int64_t dim(int i) const { return dims_[i]; }

    // layout (informational only in standalone mode)
    DataLayoutType layout() const { return layout_; }
    void set_layout(DataLayoutType l) { layout_ = l; }

    // memory is already allocated check
    bool IsInitialized() const { return !buf_.empty(); }

    void CopyDataFrom(const Tensor& other) {
        dims_       = other.dims_;
        layout_     = other.layout_;
        elem_bytes_ = other.elem_bytes_;
        buf_        = other.buf_;
    }

private:
    DDim            dims_;
    DataLayoutType  layout_     = DataLayoutType::kNCHW;
    size_t          elem_bytes_ = sizeof(float);
    std::vector<uint8_t> buf_;
};

// ---------------------------------------------------------------------------
// 5. Scope  (stub — kernels only call scope to fetch tensors they already own)
// ---------------------------------------------------------------------------
class Scope {
public:
    Tensor* FindMutableTensor(const std::string& name) {
        return &tensors_[name];
    }
    const Tensor* FindTensor(const std::string& name) const {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }
    Tensor* NewTensor(const std::string& name) {
        return &tensors_[name];
    }
private:
    mutable std::map<std::string, Tensor> tensors_;
};

// ---------------------------------------------------------------------------
// 6. ARMContext  (mirrors lite/core/context.h for ARM)
//    Kernels call ctx_->threads(), ctx_->workspace(), etc.
// ---------------------------------------------------------------------------
struct WorkSpace {
    void* get(size_t bytes) {
        if (bytes > buf_.size()) buf_.resize(bytes);
        return buf_.data();
    }
private:
    std::vector<uint8_t> buf_;
};

class ARMContext {
public:
    explicit ARMContext(int threads = 1) : threads_(threads) {}

    int  threads()    const { return threads_; }
    void set_threads(int t) { threads_ = t; }

    WorkSpace& workspace()  { return ws_; }

    // DeviceInfo helpers used by conv kernels
    bool has_dot()  const { return has_dot_;  }
    bool has_fp16() const { return has_fp16_; }
    bool has_sve2() const { return has_sve2_; }

    void set_has_dot (bool v) { has_dot_  = v; }
    void set_has_fp16(bool v) { has_fp16_ = v; }
    void set_has_sve2(bool v) { has_sve2_ = v; }

    // Some ARM math helpers query active thread id
    int active_thread_id() const { return 0; }

private:
    int       threads_  = 1;
    bool      has_dot_  = false;
    bool      has_fp16_ = false;
    bool      has_sve2_ = false;
    WorkSpace ws_;
};

// Context<kARM> alias used throughout kernel headers
template<TargetType> struct KernelContext;
template<> struct KernelContext<TargetType::kARM> : public ARMContext {
    using ARMContext::ARMContext;
};

// ---------------------------------------------------------------------------
// 7. Activation / ActivationParam  (used by conv fused-activation)
// ---------------------------------------------------------------------------
enum class ActivationType : int {
    kIndentity = 0,
    kRelu,
    kRelu6,
    kLeakyRelu,
    kSigmoid,
    kTanh,
    kHardSwish,
    kMish,
    kPRelu,
    kSwish,
};

struct ActivationParam {
    ActivationType active_type = ActivationType::kIndentity;
    bool   has_active  = false;
    float  Leaky_relu_alpha = 0.f;
    float  hard_swish_threshold = 6.f;
    float  hard_swish_scale     = 1.f / 6.f;
    float  hard_swish_offset    = 3.f;
    float  Relu6_threshold      = 6.f;
};

// ---------------------------------------------------------------------------
// 8. Op param structs  (mirrors lite/operators/op_params.h)
//    Only fields accessed by ARM kernels are replicated here.
// ---------------------------------------------------------------------------

// ---- Conv2d ---------------------------------------------------------------
struct ConvParam {
    const Tensor* x              = nullptr;
    const Tensor* filter         = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       output         = nullptr;
    // residual add
    const Tensor* residualData   = nullptr;

    std::vector<int> strides     = {1, 1};
    std::vector<int> paddings    = {0, 0};
    std::vector<int> dilations   = {1, 1};
    int   groups                 = 1;
    bool  fuse_relu              = false;
    ActivationParam activation_param;
    // quantization
    float input_scale            = 1.f;
    std::vector<float> weight_scale;
    float output_scale           = 1.f;
    // Winograd / direct / GEMM selector (set by PrepareForRun)
    int   use_winograd           = 0;
};

// ---- FC / Linear ----------------------------------------------------------
struct FcParam {
    const Tensor* input          = nullptr;
    const Tensor* w              = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       output         = nullptr;
    int           in_num_col_dims = 1;
    bool          weight_transposed = true;
    ActivationParam activation_param;
    // int8
    float input_scale            = 1.f;
    std::vector<float> weight_scale;
    float output_scale           = 1.f;
};

// ---- MatMul ---------------------------------------------------------------
struct MatMulParam {
    const Tensor* X              = nullptr;
    const Tensor* Y              = nullptr;
    Tensor*       Out            = nullptr;
    bool   transpose_X           = false;
    bool   transpose_Y           = false;
    float  alpha                 = 1.f;
};

// ---- BatchNorm ------------------------------------------------------------
struct BatchNormParam {
    const Tensor* x              = nullptr;
    const Tensor* scale          = nullptr;
    const Tensor* bias           = nullptr;
    const Tensor* mean           = nullptr;
    const Tensor* variance       = nullptr;
    Tensor*       y              = nullptr;
    Tensor*       mean_out       = nullptr;
    Tensor*       variance_out   = nullptr;
    Tensor*       saved_mean     = nullptr;
    Tensor*       saved_variance = nullptr;
    float  epsilon               = 1e-5f;
    float  momentum              = 0.9f;
    bool   is_test               = true;
    std::string data_layout      = "NCHW";
};

// ---- LayerNorm ------------------------------------------------------------
struct LayerNormParam {
    const Tensor* x              = nullptr;
    const Tensor* scale          = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       y              = nullptr;
    Tensor*       mean           = nullptr;
    Tensor*       variance       = nullptr;
    int           begin_norm_axis = 1;
    float         epsilon        = 1e-5f;
};

// ---- InstanceNorm ---------------------------------------------------------
struct InstanceNormParam {
    const Tensor* x              = nullptr;
    const Tensor* scale          = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       y              = nullptr;
    float         epsilon        = 1e-5f;
};

// ---- GroupNorm ------------------------------------------------------------
struct GroupNormParam {
    const Tensor* x              = nullptr;
    const Tensor* scale          = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       y              = nullptr;
    int           groups         = 1;
    float         epsilon        = 1e-5f;
    std::string   data_layout    = "NCHW";
};

// ---- LSTM -----------------------------------------------------------------
struct LstmParam {
    const Tensor* input          = nullptr;
    const Tensor* weight         = nullptr;
    const Tensor* bias           = nullptr;
    const Tensor* h0             = nullptr;
    const Tensor* c0             = nullptr;
    Tensor*       hidden         = nullptr;
    Tensor*       cell            = nullptr;
    bool          use_peepholes  = false;
    bool          is_reverse     = false;
    std::string   gate_activation  = "sigmoid";
    std::string   cell_activation  = "tanh";
    std::string   candidate_activation = "tanh";
};

// ---- GRU ------------------------------------------------------------------
struct GruParam {
    const Tensor* input          = nullptr;
    const Tensor* h0             = nullptr;
    const Tensor* weight         = nullptr;
    const Tensor* bias           = nullptr;
    Tensor*       hidden         = nullptr;
    Tensor*       batch_gate     = nullptr;
    Tensor*       batch_reset_hidden_prev = nullptr;
    bool          is_reverse     = false;
    std::string   activation       = "tanh";
    std::string   gate_activation  = "sigmoid";
    bool          origin_mode    = false;
};

// ---- Activation -----------------------------------------------------------
struct ActivationComputeParam {
    const Tensor* x              = nullptr;
    Tensor*       out            = nullptr;
    ActivationType active_type   = ActivationType::kRelu;
    float Relu6_threshold        = 6.f;
    float Leaky_relu_alpha       = 0.1f;
};

// ---- Elementwise ----------------------------------------------------------
struct ElementwiseParam {
    const Tensor* X              = nullptr;
    const Tensor* Y              = nullptr;
    Tensor*       Out            = nullptr;
    int           axis           = -1;
};

// ---- Pooling --------------------------------------------------------------
struct PoolParam {
    const Tensor* x              = nullptr;
    Tensor*       output         = nullptr;
    std::string   pooling_type   = "max";
    bool          global_pooling = false;
    std::vector<int> ksize       = {2, 2};
    std::vector<int> strides     = {1, 1};
    std::vector<int> paddings    = {0, 0};
    bool          ceil_mode      = false;
    bool          exclusive      = true;
};

// ---- Softmax --------------------------------------------------------------
struct SoftmaxParam {
    const Tensor* x              = nullptr;
    Tensor*       output         = nullptr;
    int           axis           = -1;
};

// ---------------------------------------------------------------------------
// 9. KernelLite base  (mirrors lite/core/kernel.h)
//    Kernels inherit: KernelLite<TARGET(kARM), PRECISION(kFloat)>
// ---------------------------------------------------------------------------
template<TargetType Target, PrecisionType Precision,
         DataLayoutType Layout = DataLayoutType::kNCHW>
class KernelLite {
public:
    using Context = KernelContext<Target>;

    virtual ~KernelLite() = default;

    // Called once before first Run(); subclasses override for weight packing.
    virtual void PrepareForRun() {}

    // Called once after param is set, before PrepareForRun
    virtual void ReInitWhenNeeded() {}

    // The main compute function.
    virtual void Run() = 0;

    // Param accessors -------------------------------------------------------
    // Each concrete kernel declares its own Param type via:
    //   using param_t = XxxParam;
    // and calls param_ as a member. We expose a raw void* here so the test
    // harness can do:
    //   kernel.SetParam(my_param);
    //   kernel.SetContext(&ctx);
    template<typename ParamT>
    void SetParam(const ParamT& p) {
        param_ptr_ = std::make_shared<ParamT>(p);
        param_raw_  = param_ptr_.get();
    }

    void SetContext(Context* ctx) { ctx_ = ctx; }
    Context* ctx() const { return ctx_; }

protected:
    // Subclasses access param via:   auto& param = *static_cast<MyParam*>(param_raw_);
    // OR they use their own typed member set by a cast in their constructor.
    void*     param_raw_  = nullptr;
    Context*  ctx_        = nullptr;

private:
    std::shared_ptr<void> param_ptr_;  // lifetime owner
};

// ---------------------------------------------------------------------------
// 10. Logging stubs  (lite/utils/log.h  →  fprintf)
// ---------------------------------------------------------------------------
}} // namespace paddle::lite

#ifndef VLOG
#  define VLOG(n)  if(false) std::cerr
#endif
#ifndef LOG
#  include <iostream>
#  define LOG(severity) std::cerr << "[" #severity "] "
#endif
#ifndef CHECK
#  define CHECK(cond)   if(!(cond)) throw std::runtime_error("CHECK failed: " #cond)
#endif
#ifndef CHECK_EQ
#  define CHECK_EQ(a,b) CHECK((a)==(b))
#endif
#ifndef CHECK_GT
#  define CHECK_GT(a,b) CHECK((a)>(b))
#endif
#ifndef CHECK_GE
#  define CHECK_GE(a,b) CHECK((a)>=(b))
#endif
#ifndef CHECK_LE
#  define CHECK_LE(a,b) CHECK((a)<=(b))
#endif
#ifndef DCHECK
#  define DCHECK(cond)
#endif
#ifndef DCHECK_EQ
#  define DCHECK_EQ(a,b)
#endif

// ---------------------------------------------------------------------------
// 11. DeviceInfo stub  (lite/backends/arm/math/device_info.h)
// ---------------------------------------------------------------------------
namespace paddle { namespace lite {

struct DeviceInfo {
    static DeviceInfo& Global() {
        static DeviceInfo inst;
        return inst;
    }
    bool has_fp16()    const { return has_fp16_; }
    bool has_dot()     const { return has_dot_; }
    bool has_sve2()    const { return has_sve2_; }
    int  arch()        const { return arch_; }   // e.g. 8 for ARMv8
    int  L1_cache()    const { return 32768; }
    int  L2_cache()    const { return 1048576; }
    int  core_num()    const { return 4; }

    void set_fp16(bool v)  { has_fp16_ = v; }
    void set_dot(bool v)   { has_dot_  = v; }
    void set_sve2(bool v)  { has_sve2_ = v; }
    void set_arch(int a)   { arch_     = a; }
private:
    bool has_fp16_ = false;
    bool has_dot_  = false;
    bool has_sve2_ = false;
    int  arch_     = 8;
};

}} // namespace paddle::lite

// ---------------------------------------------------------------------------
// 12. Convenience using-declarations (so kernel .h files compile unchanged)
// ---------------------------------------------------------------------------
namespace paddle { namespace lite { namespace kernels { namespace arm {} }}}

// Macros emitted by lite/core/type_system.h that kernels see
#define LITE_TYPE_SYSTEM_FORCE_LINK(...)

#endif // PADDLE_LITE_STUB_H_
