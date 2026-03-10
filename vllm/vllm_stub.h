// vllm_stub.h  — Drop-in replacement for all PyTorch/c10 dependencies
// used by vLLM's csrc/cpu bottleneck kernels.
//
// Lets you compile & test:
//   csrc/cpu/attention.cpp   (paged_attention_v1 / v2)
//   csrc/cpu/layernorm.cpp   (rms_norm, fused_add_rms_norm)
//   csrc/cpu/activation.cpp  (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul)
//   csrc/cpu/pos_encoding.cpp(rotary_embedding)
//   csrc/cpu/cache.cpp       (reshape_and_cache, copy_blocks)
//   csrc/cpu/quant.cpp       (dynamic_scaled_int8_quant, fp8_quant)
//
// WITHOUT linking against PyTorch / LibTorch / c10 / pybind11.
//
// Usage:
//   #define VLLM_STANDALONE
//   #include "vllm_stub.h"
//   // Then #include the vLLM .cpp you want to test
//
// Compile (host, reference-only):
//   g++ -std=c++17 -O2 -DVLLM_STANDALONE -I./include \
//       your_test.cpp -lm -o test
//
// Compile (AArch64, real ARM kernels):
//   aarch64-linux-gnu-g++ -std=c++17 -O3 \
//       -march=armv8.2-a+dotprod+fp16+bf16 \
//       -DVLLM_STANDALONE -DARM_BF16_SUPPORT \
//       -I./include -I/path/to/vllm/csrc \
//       your_test.cpp /path/to/vllm/csrc/cpu/attention.cpp ... \
//       -lm -lpthread -o test_arm

#pragma once
#ifndef VLLM_STUB_H
#define VLLM_STUB_H

// ── Standard headers the kernels need ──────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// ── ARM ISA headers (only on AArch64) ──────────────────────────────────────
#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  ifdef ARM_BF16_SUPPORT
#    include <arm_bf16.h>
#  endif
#endif

// ── Half-precision stub (fp16) ──────────────────────────────────────────────
// vLLM uses c10::Half; we stub it as a simple 16-bit wrapper.
namespace c10 {

struct Half {
    uint16_t x{0};
    Half() = default;
    Half(float f) {
        // Simple round-to-nearest float→fp16 (no flush-to-zero)
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        uint16_t sign   = (bits >> 16) & 0x8000;
        int32_t  expo   = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant   = bits & 0x7FFFFF;
        if (expo <= 0)       x = sign;
        else if (expo >= 31) x = sign | 0x7C00;
        else                 x = sign | (uint16_t(expo) << 10) | uint16_t(mant >> 13);
    }
    operator float() const {
        uint32_t sign = (x & 0x8000) << 16;
        int32_t  expo = (x >> 10) & 0x1F;
        uint32_t mant = x & 0x3FF;
        uint32_t bits;
        if (expo == 0)       bits = sign | (mant << 13);
        else if (expo == 31) bits = sign | 0x7F800000 | (mant << 13);
        else                 bits = sign | ((expo - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &bits, 4); return f;
    }
};

// BFloat16
struct BFloat16 {
    uint16_t x{0};
    BFloat16() = default;
    BFloat16(float f) { uint32_t b; std::memcpy(&b, &f, 4); x = uint16_t(b >> 16); }
    operator float() const {
        uint32_t b = uint32_t(x) << 16;
        float f; std::memcpy(&f, &b, 4); return f;
    }
};

// FP8 (E4M3 — used for KV cache quantization)
struct Float8_e4m3fn {
    uint8_t x{0};
    Float8_e4m3fn() = default;
    // Simple saturating float→fp8 conversion
    Float8_e4m3fn(float f) {
        // clamp to fp8 range ≈ ±448
        f = std::max(-448.f, std::min(448.f, f));
        uint32_t bits; std::memcpy(&bits, &f, 4);
        uint8_t sign = (bits >> 31) & 1;
        int32_t expo = ((bits >> 23) & 0xFF) - 127 + 7;  // bias 7 for E4M3
        uint32_t mant = (bits >> 20) & 0x7;               // 3 mantissa bits
        if (expo <= 0)     x = sign << 7;
        else if (expo > 15) x = (sign << 7) | 0x7E;       // NaN-free saturation
        else               x = (sign << 7) | (uint8_t(expo) << 3) | mant;
    }
    operator float() const {
        uint8_t sign = (x >> 7) & 1;
        int32_t expo = (x >> 3) & 0xF;
        uint32_t mant = x & 0x7;
        if (expo == 0 && mant == 0) return 0.f;
        uint32_t bits = (uint32_t(sign) << 31)
                      | (uint32_t(expo - 7 + 127) << 23)
                      | (mant << 20);
        float f; std::memcpy(&f, &bits, 4); return f;
    }
};

}  // namespace c10

// ── Scalar type enum (subset used by kernels) ──────────────────────────────
namespace c10 {
enum class ScalarType : int8_t {
    Byte      = 0,
    Char      = 1,
    Short     = 2,
    Int       = 3,
    Long      = 4,
    Half      = 5,
    Float     = 6,
    Double    = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDouble = 10,
    Bool      = 11,
    BFloat16  = 15,
    Float8_e4m3fn = 23,
    Undefined = -1,
};
}

// ── Tensor stub ─────────────────────────────────────────────────────────────
// A minimal, self-owned flat-memory tensor.  Supports the subset of the
// torch::Tensor / at::Tensor API called by vLLM's CPU kernels.
namespace at {

class Tensor {
public:
    Tensor() = default;

    // ---- Factory helpers ---------------------------------------------------
    static Tensor empty(std::vector<int64_t> shape, c10::ScalarType dtype = c10::ScalarType::Float) {
        Tensor t;
        t.shape_   = std::move(shape);
        t.dtype_   = dtype;
        t.numel_   = 1;
        for (auto d : t.shape_) t.numel_ *= d;
        t.itemsize_ = dtype_itemsize(t.dtype_);
        t.buf_.resize(t.numel_ * t.itemsize_, 0);
        t.data_ptr_ = t.buf_.data();
        return t;
    }

    // ---- Dimension queries ------------------------------------------------
    int64_t dim() const { return (int64_t)shape_.size(); }
    int64_t size(int64_t d) const {
        if (d < 0) d += dim();
        return shape_[d];
    }
    int64_t numel() const { return numel_; }
    c10::ScalarType scalar_type() const { return dtype_; }
    bool defined() const { return !buf_.empty() || data_ptr_ != nullptr; }
    bool is_contiguous() const { return true; }

    // ---- Data access -------------------------------------------------------
    void* data_ptr() const { return data_ptr_; }
    template<typename T> T* data_ptr() const { return reinterpret_cast<T*>(data_ptr_); }
    template<typename T> const T* const_data_ptr() const { return reinterpret_cast<const T*>(data_ptr_); }

    // ---- Stride (vLLM kernels sometimes call .stride(d)) ------------------
    int64_t stride(int64_t d) const {
        if (d < 0) d += dim();
        int64_t s = 1;
        for (int64_t i = (int64_t)shape_.size()-1; i > d; --i) s *= shape_[i];
        return s;
    }

    // ---- Reshape (view semantics, no copy) --------------------------------
    Tensor view(std::vector<int64_t> new_shape) const {
        Tensor t = *this;
        t.shape_ = std::move(new_shape);
        return t;
    }
    Tensor reshape(std::vector<int64_t> new_shape) const { return view(std::move(new_shape)); }

    // ---- Slice [dim][start:end] (shallow) ---------------------------------
    Tensor slice(int64_t dim, int64_t start, int64_t end) const {
        // Simplified: only dim=0 slice used by vLLM CPU kernels
        assert(dim == 0);
        Tensor t;
        t.dtype_ = dtype_;
        t.itemsize_ = itemsize_;
        t.shape_ = shape_;
        t.shape_[0] = end - start;
        t.numel_ = 1;
        for (auto d : t.shape_) t.numel_ *= d;
        int64_t row_bytes = stride(0) * itemsize_;
        t.data_ptr_ = static_cast<char*>(data_ptr_) + start * row_bytes;
        return t;
    }

    // ---- Index (Tensor[i]) ------------------------------------------------
    Tensor index(int64_t i) const { return slice(0, i, i+1).view({shape_.begin()+1, shape_.end()}); }

    // ---- Fill helpers (for tests) ----------------------------------------
    void fill_(float v) {
        if (dtype_ == c10::ScalarType::Float)
            for (int64_t i=0;i<numel_;++i) data_ptr<float>()[i] = v;
        else if (dtype_ == c10::ScalarType::BFloat16)
            for (int64_t i=0;i<numel_;++i) data_ptr<c10::BFloat16>()[i] = c10::BFloat16(v);
        else if (dtype_ == c10::ScalarType::Half)
            for (int64_t i=0;i<numel_;++i) data_ptr<c10::Half>()[i] = c10::Half(v);
    }

    // ---- Optional<Tensor> helper ------------------------------------------
    std::optional<Tensor> opt() const { return *this; }

private:
    std::vector<int64_t> shape_;
    c10::ScalarType dtype_{c10::ScalarType::Float};
    int64_t numel_{0};
    int64_t itemsize_{4};
    std::vector<uint8_t> buf_;
    void* data_ptr_{nullptr};

    static int64_t dtype_itemsize(c10::ScalarType dt) {
        switch (dt) {
            case c10::ScalarType::Float:   return 4;
            case c10::ScalarType::Half:    return 2;
            case c10::ScalarType::BFloat16:return 2;
            case c10::ScalarType::Byte:    return 1;
            case c10::ScalarType::Char:    return 1;
            case c10::ScalarType::Short:   return 2;
            case c10::ScalarType::Int:     return 4;
            case c10::ScalarType::Long:    return 8;
            case c10::ScalarType::Double:  return 8;
            case c10::ScalarType::Float8_e4m3fn: return 1;
            default: return 4;
        }
    }
};

// Tensor factory functions used by kernels
inline Tensor empty(std::vector<int64_t> shape, c10::ScalarType dtype = c10::ScalarType::Float) {
    return Tensor::empty(std::move(shape), dtype);
}

}  // namespace at

// torch:: aliases
namespace torch {
    using Tensor = at::Tensor;
    namespace kCPU {} // placeholder
}

// ── TORCH_CHECK / TORCH_INTERNAL_ASSERT / AT_DISPATCH ──────────────────────
#define TORCH_CHECK(cond, ...) \
    do { if (!(cond)) { \
        std::string msg = "TORCH_CHECK failed: " #cond; \
        throw std::runtime_error(msg); \
    }} while(0)

#define TORCH_INTERNAL_ASSERT(cond, ...) TORCH_CHECK(cond)
#define AT_ASSERT(cond) TORCH_CHECK(cond)
#define VLLM_CHECK(cond, ...) TORCH_CHECK(cond)

// ── AT_DISPATCH_FLOATING_TYPES_AND2 / AT_DISPATCH_... macros ───────────────
// vLLM uses these to dispatch over fp32/fp16/bf16.
// We replace them with a simple template-dispatch that calls the body
// with the right scalar_t type alias.

#define AT_DISPATCH_CASE(scalar_type_val, cpp_type, ...) \
    case scalar_type_val: { using scalar_t = cpp_type; __VA_ARGS__(); break; }

#define AT_DISPATCH_FLOATING_TYPES_AND2(extra1, extra2, tensor_dtype, name, ...) \
    [&]() { \
        switch (tensor_dtype) { \
            AT_DISPATCH_CASE(c10::ScalarType::Float,   float,         __VA_ARGS__) \
            AT_DISPATCH_CASE(c10::ScalarType::Half,    c10::Half,     __VA_ARGS__) \
            AT_DISPATCH_CASE(c10::ScalarType::BFloat16,c10::BFloat16, __VA_ARGS__) \
            default: throw std::runtime_error("Unsupported dtype in " name); \
        } \
    }()

#define AT_DISPATCH_FLOATING_TYPES(tensor_dtype, name, ...) \
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::ScalarType::Half, c10::ScalarType::BFloat16, tensor_dtype, name, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_dtype, name, ...) \
    AT_DISPATCH_FLOATING_TYPES_AND2(c10::ScalarType::Half, c10::ScalarType::BFloat16, tensor_dtype, name, __VA_ARGS__)

// For KV-cache / quant dispatch
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(e1,e2,...) \
    AT_DISPATCH_FLOATING_TYPES_AND2(e1,e2, __VA_ARGS__)

// ── Optional<Tensor> helpers used by kernels ──────────────────────────────
using OptionalTensor = std::optional<at::Tensor>;

// ── Scalar stub ─────────────────────────────────────────────────────────────
namespace c10 {
struct Scalar {
    double val;
    Scalar(double v=0.0) : val(v) {}
    Scalar(float  v)     : val(v) {}
    Scalar(int    v)     : val(v) {}
    template<typename T> T to() const { return static_cast<T>(val); }
};
}

// ── OpenMP (no-op if not available) ─────────────────────────────────────────
#ifdef _OPENMP
#  include <omp.h>
#else
inline int  omp_get_thread_num()  { return 0; }
inline int  omp_get_max_threads() { return 1; }
inline int  omp_get_num_threads() { return 1; }
#  define _Pragma(x)
// no-op: #pragma omp parallel for → just run serially
#endif

// ── cpu_types_arm.hpp compatibility layer ────────────────────────────────
// The vLLM cpu_types_arm.hpp defines vec_op:: types backed by NEON intrinsics.
// When building standalone (host x86 or without NEON), we provide scalar
// fallbacks so the .cpp files still compile and produce correct results.

namespace vec_op {

#if defined(__aarch64__) || defined(__ARM_NEON)
// ── Real NEON path ─────────────────────────────────────────────────────────
// Just include the real cpu_types_arm.hpp from vLLM when building with
// VLLM_STANDALONE + VLLM_REAL_KERNELS.  Otherwise provide scalar fallbacks.
#  ifdef VLLM_REAL_KERNELS
// Nothing to define here — cpu_types_arm.hpp is included by the .cpp files.
#  endif
#endif

// ── Scalar fallback (host testing on x86) ─────────────────────────────────
// Mimics the interface of cpu_types_arm.hpp / cpu_types_x86.hpp for
// float32 only, enough to run reference checks on a dev machine.
#ifndef VLLM_REAL_KERNELS

constexpr int VEC_ELEM_NUM = 4;  // Pretend SIMD width = 4 floats

struct FP32Vec4 {
    float v[4]{};
    FP32Vec4() = default;
    FP32Vec4(float a, float b, float c, float d) { v[0]=a;v[1]=b;v[2]=c;v[3]=d; }
    explicit FP32Vec4(float s) { v[0]=v[1]=v[2]=v[3]=s; }
    explicit FP32Vec4(const float* p) { std::memcpy(v, p, 16); }

    FP32Vec4 operator+(const FP32Vec4& o) const { return {v[0]+o.v[0],v[1]+o.v[1],v[2]+o.v[2],v[3]+o.v[3]}; }
    FP32Vec4 operator*(const FP32Vec4& o) const { return {v[0]*o.v[0],v[1]*o.v[1],v[2]*o.v[2],v[3]*o.v[3]}; }
    FP32Vec4 operator-(const FP32Vec4& o) const { return {v[0]-o.v[0],v[1]-o.v[1],v[2]-o.v[2],v[3]-o.v[3]}; }
    FP32Vec4& operator+=(const FP32Vec4& o) { for(int i=0;i<4;++i) v[i]+=o.v[i]; return *this; }

    void save(float* p) const { std::memcpy(p, v, 16); }
    float reduce_sum() const { return v[0]+v[1]+v[2]+v[3]; }
    float reduce_max() const { return std::max({v[0],v[1],v[2],v[3]}); }
};

struct FP32Vec8 {
    float v[8]{};
    FP32Vec8() = default;
    explicit FP32Vec8(float s) { for(auto& x:v) x=s; }
    explicit FP32Vec8(const float* p) { std::memcpy(v, p, 32); }
    FP32Vec8 operator+(const FP32Vec8& o) const { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=v[i]+o.v[i]; return r; }
    FP32Vec8 operator*(const FP32Vec8& o) const { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=v[i]*o.v[i]; return r; }
    FP32Vec8& operator+=(const FP32Vec8& o) { for(int i=0;i<8;++i) v[i]+=o.v[i]; return *this; }
    void save(float* p) const { std::memcpy(p, v, 32); }
    float reduce_sum() const { float s=0; for(auto x:v) s+=x; return s; }
    float reduce_max() const { float m=v[0]; for(auto x:v) m=std::max(m,x); return m; }
};

using FP32Vec16 = FP32Vec8;  // alias for kernels that use FP32Vec16

// BF16/FP16 vector stubs (convert through float)
struct BF16Vec8 {
    c10::BFloat16 v[8]{};
    BF16Vec8() = default;
    explicit BF16Vec8(const FP32Vec8& f) { for(int i=0;i<8;++i) v[i]=c10::BFloat16(f.v[i]); }
    FP32Vec8 to_fp32() const { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=(float)v[i]; return r; }
    void save(c10::BFloat16* p) const { std::memcpy(p, v, 16); }
};

struct FP16Vec8 {
    c10::Half v[8]{};
    FP16Vec8() = default;
    explicit FP16Vec8(const FP32Vec8& f) { for(int i=0;i<8;++i) v[i]=c10::Half(f.v[i]); }
    FP32Vec8 to_fp32() const { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=(float)v[i]; return r; }
    void save(c10::Half* p) const { std::memcpy(p, v, 16); }
};

// vec_op utility: element-wise ops on FP32Vec8
inline FP32Vec8 fma(FP32Vec8 a, FP32Vec8 b, FP32Vec8 c) {
    FP32Vec8 r;
    for(int i=0;i<8;++i) r.v[i] = a.v[i]*b.v[i]+c.v[i];
    return r;
}
inline FP32Vec8 exp(FP32Vec8 x) { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=std::exp(x.v[i]); return r; }
inline FP32Vec8 tanh(FP32Vec8 x) { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=std::tanh(x.v[i]); return r; }
inline FP32Vec8 max(FP32Vec8 a, FP32Vec8 b) { FP32Vec8 r; for(int i=0;i<8;++i) r.v[i]=std::max(a.v[i],b.v[i]); return r; }

#endif  // !VLLM_REAL_KERNELS
}  // namespace vec_op

// ── VLLM_DISPATCH_CASE / VLLM_DISPATCH_FLOATING_TYPES macros ───────────────
// Mirrors the real macros in vllm/core/scalar_type.hpp
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)  // no-op in standalone
#define VLLM_DISPATCH_FLOATING_TYPES(...)       // no-op

// ── Logging (no-op) ─────────────────────────────────────────────────────────
#define VLOG(n) if(false) std::cerr

// ── Pybind11 / module registration no-ops ──────────────────────────────────
// vLLM .cpp files sometimes have PYBIND11_MODULE at the bottom.
// We null it out so the .cpp compiles as a plain translation unit.
#define PYBIND11_MODULE(name, m) void _pybind_noop_##name(int& m)
#define TORCH_LIBRARY(ns, m)     void _torch_lib_noop_##ns(int& m)
#define TORCH_LIBRARY_IMPL(ns, d, m) void _torch_impl_noop_##ns(int& m)

// Convenience: tensor factory for tests
namespace vllm_test {

inline at::Tensor make_tensor(std::vector<int64_t> shape,
                               c10::ScalarType dtype = c10::ScalarType::Float,
                               float init_val = 0.f) {
    auto t = at::Tensor::empty(shape, dtype);
    t.fill_(init_val);
    return t;
}

inline at::Tensor rand_tensor(std::vector<int64_t> shape,
                               c10::ScalarType dtype = c10::ScalarType::Float,
                               float lo = -1.f, float hi = 1.f) {
    auto t = at::Tensor::empty(shape, dtype);
    auto* p = t.data_ptr<float>();
    // Simple LCG for reproducible randomness (no <random> dependency)
    uint32_t state = 0xDEADBEEF;
    auto lcg = [&]() -> float {
        state = state * 1664525u + 1013904223u;
        float r = (float)(state >> 8) / (float)(1 << 24);
        return lo + r * (hi - lo);
    };
    if (dtype == c10::ScalarType::Float) {
        for (int64_t i = 0; i < t.numel(); ++i) p[i] = lcg();
    } else if (dtype == c10::ScalarType::BFloat16) {
        auto* b = t.data_ptr<c10::BFloat16>();
        for (int64_t i = 0; i < t.numel(); ++i) b[i] = c10::BFloat16(lcg());
    } else if (dtype == c10::ScalarType::Half) {
        auto* h = t.data_ptr<c10::Half>();
        for (int64_t i = 0; i < t.numel(); ++i) h[i] = c10::Half(lcg());
    }
    return t;
}

inline float max_abs_diff(const at::Tensor& a, const at::Tensor& b) {
    assert(a.numel() == b.numel());
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    float d = 0.f;
    for (int64_t i = 0; i < a.numel(); ++i)
        d = std::max(d, std::abs(pa[i] - pb[i]));
    return d;
}

}  // namespace vllm_test

#endif  // VLLM_STUB_H
