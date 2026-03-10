// vllm_arm_kernels.h  — Framework-free implementations of vLLM's three
// ARM CPU inference bottleneck kernels.
//
// Each kernel matches the exact signature of its vLLM counterpart so you
// can swap the include without changing call sites.
//
// Operators provided:
//   1. paged_attention_v1   — single-pass decode attention (short contexts)
//   2. paged_attention_v2   — two-pass reduce decode attention (long contexts)
//   3. rms_norm             — RMSNorm (LLaMA / Mistral / Qwen)
//   4. fused_add_rms_norm   — residual-add fused into RMSNorm
//   5. silu_and_mul         — SiLU-gate activation (LLaMA MLP)
//   6. gelu_and_mul         — GELU-gate activation (GPT-NeoX)
//   7. gelu_tanh_and_mul    — tanh-GELU gate

#pragma once
#ifndef VLLM_ARM_KERNELS_H
#define VLLM_ARM_KERNELS_H

#include "vllm_stub.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

// ── NEON intrinsics (AArch64 only) ─────────────────────────────────────────
#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define VLLM_NEON 1
#endif

// ────────────────────────────────────────────────────────────────────────────
// Helper: float ↔ scalar type conversions
// ────────────────────────────────────────────────────────────────────────────
namespace vllm_detail {

template<typename T> inline float to_float(T v)      { return (float)v; }
template<>           inline float to_float(c10::Half v)     { return (float)v; }
template<>           inline float to_float(c10::BFloat16 v) { return (float)v; }

template<typename T> inline T from_float(float v)    { return (T)v; }
template<>           inline c10::Half    from_float(float v) { return c10::Half(v); }
template<>           inline c10::BFloat16 from_float(float v){ return c10::BFloat16(v); }

// SiLU: x * sigmoid(x)
inline float silu(float x) { return x / (1.f + std::exp(-x)); }

// GELU (exact)
inline float gelu(float x) {
    return 0.5f * x * (1.f + std::erf(x * 0.70710678118f));
}
// GELU tanh approximation  (OpenAI variant)
inline float gelu_tanh(float x) {
    float c = 0.044715f;
    float t = std::tanh(0.7978845608f * (x + c * x * x * x));
    return 0.5f * x * (1.f + t);
}

}  // namespace vllm_detail

// ────────────────────────────────────────────────────────────────────────────
// 1.  RMS Norm
//
//   y = x / sqrt( mean(x^2) + eps ) * weight
//
// Matches vLLM:  void rms_norm(Tensor& out, const Tensor& input,
//                              const Tensor& weight, double epsilon,
//                              bool use_quant=false)
// ────────────────────────────────────────────────────────────────────────────
template<typename T>
static void rms_norm_impl(T* __restrict__ out,
                          const T* __restrict__ inp,
                          const T* __restrict__ weight,
                          int64_t num_tokens,
                          int64_t hidden_size,
                          float   epsilon) {
#ifdef VLLM_NEON
    // ── NEON path ───────────────────────────────────────────────────────────
    // FP32 accumulation regardless of T for numerical stability
    for (int64_t t = 0; t < num_tokens; ++t) {
        const T* x = inp + t * hidden_size;
        T*       y = out + t * hidden_size;

        // Compute sum(x^2) with NEON
        float32x4_t vss = vdupq_n_f32(0.f);
        int64_t i = 0;
        for (; i + 4 <= hidden_size; i += 4) {
            float32x4_t vx;
            if constexpr (std::is_same_v<T, float>) {
                vx = vld1q_f32(reinterpret_cast<const float*>(x + i));
            } else {
                float tmp[4];
                for (int k=0;k<4;++k) tmp[k] = vllm_detail::to_float(x[i+k]);
                vx = vld1q_f32(tmp);
            }
            vss = vmlaq_f32(vss, vx, vx);
        }
        float ss = vaddvq_f32(vss);
        for (; i < hidden_size; ++i) { float v=vllm_detail::to_float(x[i]); ss+=v*v; }

        float inv = 1.f / std::sqrt(ss / (float)hidden_size + epsilon);
        float32x4_t vinv = vdupq_n_f32(inv);

        i = 0;
        for (; i + 4 <= hidden_size; i += 4) {
            float32x4_t vx, vw;
            if constexpr (std::is_same_v<T, float>) {
                vx = vld1q_f32(reinterpret_cast<const float*>(x + i));
                vw = vld1q_f32(reinterpret_cast<const float*>(weight + i));
                float32x4_t vy = vmulq_f32(vmulq_f32(vx, vinv), vw);
                vst1q_f32(reinterpret_cast<float*>(y + i), vy);
            } else {
                float xf[4], wf[4], yf[4];
                for (int k=0;k<4;++k) {
                    xf[k] = vllm_detail::to_float(x[i+k]);
                    wf[k] = vllm_detail::to_float(weight[i+k]);
                    yf[k] = xf[k] * inv * wf[k];
                    y[i+k] = vllm_detail::from_float<T>(yf[k]);
                }
            }
        }
        for (; i < hidden_size; ++i)
            y[i] = vllm_detail::from_float<T>(
                       vllm_detail::to_float(x[i]) * inv *
                       vllm_detail::to_float(weight[i]));
    }
#else
    // ── Scalar fallback ─────────────────────────────────────────────────────
    for (int64_t t = 0; t < num_tokens; ++t) {
        const T* x = inp + t * hidden_size;
        T*       y = out + t * hidden_size;
        float ss = 0.f;
        for (int64_t i = 0; i < hidden_size; ++i) { float v=vllm_detail::to_float(x[i]); ss+=v*v; }
        float inv = 1.f / std::sqrt(ss / (float)hidden_size + epsilon);
        for (int64_t i = 0; i < hidden_size; ++i)
            y[i] = vllm_detail::from_float<T>(
                       vllm_detail::to_float(x[i]) * inv *
                       vllm_detail::to_float(weight[i]));
    }
#endif
}

inline void rms_norm(at::Tensor&       out,
                     const at::Tensor& input,
                     const at::Tensor& weight,
                     double            epsilon,
                     bool              /*use_quant*/ = false) {
    int64_t num_tokens  = input.numel() / input.size(input.dim()-1);
    int64_t hidden_size = input.size(input.dim()-1);
    float   eps         = (float)epsilon;

    TORCH_CHECK(input.scalar_type() == out.scalar_type());
    switch (input.scalar_type()) {
        case c10::ScalarType::Float:
            rms_norm_impl<float>(out.data_ptr<float>(), input.data_ptr<float>(),
                                 weight.data_ptr<float>(), num_tokens, hidden_size, eps);
            break;
        case c10::ScalarType::BFloat16:
            rms_norm_impl<c10::BFloat16>(out.data_ptr<c10::BFloat16>(),
                                         input.data_ptr<c10::BFloat16>(),
                                         weight.data_ptr<c10::BFloat16>(),
                                         num_tokens, hidden_size, eps);
            break;
        case c10::ScalarType::Half:
            rms_norm_impl<c10::Half>(out.data_ptr<c10::Half>(),
                                     input.data_ptr<c10::Half>(),
                                     weight.data_ptr<c10::Half>(),
                                     num_tokens, hidden_size, eps);
            break;
        default: TORCH_CHECK(false, "rms_norm: unsupported dtype");
    }
}

// ── Fused add+rms_norm ─────────────────────────────────────────────────────
//   residual += input;   out = rms_norm(residual)
inline void fused_add_rms_norm(at::Tensor&       input,    // in-out residual
                                at::Tensor&       residual,
                                const at::Tensor& weight,
                                double            epsilon) {
    int64_t N = input.numel();
    int64_t hidden_size = input.size(input.dim()-1);
    int64_t num_tokens  = N / hidden_size;
    float   eps = (float)epsilon;

    // residual += input first
    {
        float* r = residual.data_ptr<float>();
        const float* x = input.data_ptr<float>();
        for (int64_t i = 0; i < N; ++i) r[i] += x[i];
    }
    // then rms_norm(residual) → input (in-place output)
    rms_norm_impl<float>(input.data_ptr<float>(),
                         residual.data_ptr<float>(),
                         weight.data_ptr<float>(),
                         num_tokens, hidden_size, eps);
}

// ────────────────────────────────────────────────────────────────────────────
// 2.  Activation gates
//
//   out[i] = gate(input[i]) * input[i + d/2]
//   where gate is SiLU, GELU, or GELU-tanh.
//
// Matches vLLM:  void silu_and_mul(Tensor& out, Tensor& input)
//                                 (input shape [*, 2d], out shape [*, d])
// ────────────────────────────────────────────────────────────────────────────
enum class GateType { SiLU, GELU, GELUTanh };

template<typename T, GateType G>
static void gate_and_mul_impl(T* __restrict__ out,
                               const T* __restrict__ inp,
                               int64_t num_tokens,
                               int64_t d) {
    for (int64_t t = 0; t < num_tokens; ++t) {
        const T* gate_in = inp + t * 2 * d;
        const T* mul_in  = inp + t * 2 * d + d;
        T*       o       = out + t * d;

#ifdef VLLM_NEON
        int64_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
            for (; i + 4 <= d; i += 4) {
                float32x4_t vg = vld1q_f32(reinterpret_cast<const float*>(gate_in + i));
                float32x4_t vm = vld1q_f32(reinterpret_cast<const float*>(mul_in  + i));
                float gf[4]; vst1q_f32(gf, vg);
                float of[4];
                for (int k=0;k<4;++k) {
                    if constexpr (G == GateType::SiLU)    of[k] = vllm_detail::silu(gf[k]) * ((const float*)mul_in)[i+k];
                    if constexpr (G == GateType::GELU)    of[k] = vllm_detail::gelu(gf[k]) * ((const float*)mul_in)[i+k];
                    if constexpr (G == GateType::GELUTanh) of[k] = vllm_detail::gelu_tanh(gf[k]) * ((const float*)mul_in)[i+k];
                }
                vst1q_f32(reinterpret_cast<float*>(o + i), vld1q_f32(of));
            }
        }
        for (; i < d; ++i) {
#else
        for (int64_t i = 0; i < d; ++i) {
#endif
            float gv = vllm_detail::to_float(gate_in[i]);
            float mv = vllm_detail::to_float(mul_in[i]);
            float rv;
            if constexpr (G == GateType::SiLU)    rv = vllm_detail::silu(gv) * mv;
            if constexpr (G == GateType::GELU)     rv = vllm_detail::gelu(gv) * mv;
            if constexpr (G == GateType::GELUTanh) rv = vllm_detail::gelu_tanh(gv) * mv;
            o[i] = vllm_detail::from_float<T>(rv);
        }
    }
}

template<GateType G>
static void gate_and_mul_dispatch(at::Tensor& out, at::Tensor& input) {
    int64_t d          = out.size(out.dim()-1);
    int64_t num_tokens = input.numel() / (2 * d);
    switch (input.scalar_type()) {
        case c10::ScalarType::Float:
            gate_and_mul_impl<float,   G>(out.data_ptr<float>(),         input.data_ptr<float>(),         num_tokens, d); break;
        case c10::ScalarType::BFloat16:
            gate_and_mul_impl<c10::BFloat16,G>(out.data_ptr<c10::BFloat16>(), input.data_ptr<c10::BFloat16>(), num_tokens, d); break;
        case c10::ScalarType::Half:
            gate_and_mul_impl<c10::Half,G>(out.data_ptr<c10::Half>(),    input.data_ptr<c10::Half>(),    num_tokens, d); break;
        default: TORCH_CHECK(false, "gate_and_mul: unsupported dtype");
    }
}

inline void silu_and_mul    (at::Tensor& out, at::Tensor& input) { gate_and_mul_dispatch<GateType::SiLU>    (out, input); }
inline void gelu_and_mul    (at::Tensor& out, at::Tensor& input) { gate_and_mul_dispatch<GateType::GELU>    (out, input); }
inline void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input){ gate_and_mul_dispatch<GateType::GELUTanh>(out, input); }

// ────────────────────────────────────────────────────────────────────────────
// 3.  Paged Attention  (decode phase — single token per sequence)
//
// Implements Multi-Head Attention over a paged KV cache.
// Matches vLLM's CPU attention.cpp signatures exactly.
//
//   paged_attention_v1:  compute in a single pass  (short contexts)
//   paged_attention_v2:  two-pass reduce            (long contexts)
//
// Tensor shapes (using vLLM's naming):
//   out          [num_seqs, num_heads, head_size]
//   query        [num_seqs, num_heads, head_size]
//   key_cache    [num_blocks, num_kv_heads, head_size/x, block_size, x]
//                  where x = 16/sizeof(T) — vLLM's interleaved layout
//   value_cache  [num_blocks, num_kv_heads, head_size, block_size]
//   block_tables [num_seqs, max_num_blocks_per_seq]
//   seq_lens     [num_seqs]  — actual context lengths
// ────────────────────────────────────────────────────────────────────────────

namespace paged_attn_detail {

// Compute softmax in-place over logits[0..ctx_len)
static void softmax_inplace(float* logits, int ctx_len) {
    float maxv = logits[0];
    for (int i=1;i<ctx_len;++i) maxv = std::max(maxv, logits[i]);
    float sum = 0.f;
    for (int i=0;i<ctx_len;++i) { logits[i] = std::exp(logits[i]-maxv); sum+=logits[i]; }
    float inv = 1.f / sum;
    for (int i=0;i<ctx_len;++i) logits[i] *= inv;
}

// QK dot product for one query head vs one key vector (head_size floats each)
static float qk_dot(const float* q, const float* k, int head_size, float scale) {
    float s = 0.f;
#ifdef VLLM_NEON
    float32x4_t vacc = vdupq_n_f32(0.f);
    int i=0;
    for (;i+4<=head_size;i+=4)
        vacc = vmlaq_f32(vacc, vld1q_f32(q+i), vld1q_f32(k+i));
    s = vaddvq_f32(vacc);
    for (;i<head_size;++i) s+=q[i]*k[i];
#else
    for (int i=0;i<head_size;++i) s+=q[i]*k[i];
#endif
    return s * scale;
}

// Accumulate weighted value into acc[head_size]
static void accum_value(float* acc, const float* v, float w, int head_size) {
#ifdef VLLM_NEON
    float32x4_t vw = vdupq_n_f32(w);
    int i=0;
    for (;i+4<=head_size;i+=4)
        vst1q_f32(acc+i, vmlaq_f32(vld1q_f32(acc+i), vld1q_f32(v+i), vw));
    for (;i<head_size;++i) acc[i]+=v[i]*w;
#else
    for (int i=0;i<head_size;++i) acc[i]+=v[i]*w;
#endif
}

// Convert key from vLLM's interleaved key cache layout to a plain float buffer
// vLLM key_cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
//   x = vector_size = 8 (float32) or 16 (float16/bf16)
template<typename KVT>
static void load_key(float* dst, const KVT* key_cache_block,
                     int block_offset, int head_size, int block_size, int x) {
    // key_cache_block: [num_kv_heads, head_size/x, block_size, x] for this block
    // For head h, slot s: key[d] is at [d/x][s][d%x]
    for (int d = 0; d < head_size; ++d) {
        int group  = d / x;
        int within = d % x;
        int idx    = group * block_size * x + block_offset * x + within;
        dst[d] = vllm_detail::to_float(key_cache_block[idx]);
    }
}

// Load value: value_cache layout [num_blocks, num_kv_heads, head_size, block_size]
template<typename KVT>
static void load_value(float* dst, const KVT* val_cache_block,
                       int block_offset, int head_size, int block_size) {
    // val_cache_block: [num_kv_heads, head_size, block_size]
    for (int d = 0; d < head_size; ++d) {
        dst[d] = vllm_detail::to_float(val_cache_block[d * block_size + block_offset]);
    }
}

}  // namespace paged_attn_detail

// ── paged_attention_v1 ─────────────────────────────────────────────────────
inline void paged_attention_v1(
        at::Tensor&             out,          // [num_seqs, num_heads, head_size]
        const at::Tensor&       query,        // [num_seqs, num_heads, head_size]
        const at::Tensor&       key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
        const at::Tensor&       value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
        int                     num_kv_heads,
        float                   scale,
        const at::Tensor&       block_tables, // [num_seqs, max_blocks_per_seq]
        const at::Tensor&       seq_lens,     // [num_seqs]
        int                     block_size,
        int                     /*max_seq_len*/,
        const std::optional<at::Tensor>& /*alibi_slopes*/,
        const std::string&      /*kv_cache_dtype*/,
        float                   /*kv_scale*/,
        int                     /*tp_rank*/       = 0,
        int                     /*blocksparse_local_blocks*/  = 0,
        float                   /*blocksparse_vert_stride*/   = 1.f,
        float                   /*blocksparse_block_size*/    = 64.f,
        float                   /*blocksparse_head_sliding_step*/ = 0.f) {

    int64_t num_seqs  = query.size(0);
    int64_t num_heads = query.size(1);
    int64_t head_size = query.size(2);
    int64_t max_blocks= block_tables.size(1);
    int     x         = 8;  // interleave factor for float32

    // x = 16/sizeof(KV element) in vLLM; we fix fp32 here
    if (key_cache.scalar_type() == c10::ScalarType::Half ||
        key_cache.scalar_type() == c10::ScalarType::BFloat16) x = 16;

    int64_t kv_head_stride_k = (head_size / x) * block_size * x;  // per kv-head in key block
    int64_t kv_head_stride_v =  head_size * block_size;             // per kv-head in val block
    int64_t num_blocks_total  = key_cache.size(0);
    int64_t heads_per_kv      = num_heads / num_kv_heads;

    std::vector<float> logits(block_size * max_blocks);
    std::vector<float> q_buf(head_size), k_buf(head_size), v_buf(head_size);

    const int32_t* seq_lens_ptr  = seq_lens.data_ptr<int32_t>();
    const int32_t* btables_ptr   = block_tables.data_ptr<int32_t>();
    float*         out_ptr       = out.data_ptr<float>();
    const float*   query_ptr     = query.data_ptr<float>();

    for (int64_t s = 0; s < num_seqs; ++s) {
        int ctx_len = seq_lens_ptr[s];
        if (ctx_len == 0) continue;

        const int32_t* btable = btables_ptr + s * max_blocks;

        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t kv_h = h / heads_per_kv;

            // Copy query
            const float* q = query_ptr + (s * num_heads + h) * head_size;
            std::memcpy(q_buf.data(), q, head_size * sizeof(float));

            // Compute QK logits over all context tokens
            int token_idx = 0;
            for (int bl = 0; bl < (ctx_len + block_size - 1) / block_size; ++bl) {
                int32_t phys_block  = btable[bl];
                int     slots_in_bl = std::min((int64_t)block_size,
                                               (int64_t)ctx_len - bl * block_size);

                // Pointer to key data for this kv-head in this physical block
                const float* k_cache_base =
                    key_cache.data_ptr<float>()
                    + phys_block * num_kv_heads * kv_head_stride_k
                    + kv_h * kv_head_stride_k;

                for (int sl = 0; sl < slots_in_bl; ++sl, ++token_idx) {
                    paged_attn_detail::load_key<float>(
                        k_buf.data(), k_cache_base, sl, head_size, block_size, x);
                    logits[token_idx] =
                        paged_attn_detail::qk_dot(q_buf.data(), k_buf.data(), head_size, scale);
                }
            }

            // Softmax over logits[0..ctx_len)
            paged_attn_detail::softmax_inplace(logits.data(), ctx_len);

            // Accumulate weighted values
            float* o = out_ptr + (s * num_heads + h) * head_size;
            std::fill(o, o + head_size, 0.f);

            token_idx = 0;
            for (int bl = 0; bl < (ctx_len + block_size - 1) / block_size; ++bl) {
                int32_t phys_block  = btable[bl];
                int     slots_in_bl = std::min((int64_t)block_size,
                                               (int64_t)ctx_len - bl * block_size);

                const float* v_cache_base =
                    value_cache.data_ptr<float>()
                    + phys_block * num_kv_heads * kv_head_stride_v
                    + kv_h * kv_head_stride_v;

                for (int sl = 0; sl < slots_in_bl; ++sl, ++token_idx) {
                    paged_attn_detail::load_value<float>(
                        v_buf.data(), v_cache_base, sl, head_size, block_size);
                    paged_attn_detail::accum_value(o, v_buf.data(), logits[token_idx], head_size);
                }
            }
        }
    }
}

// ── paged_attention_v2  (two-pass: per-partition softmax then reduce) ───────
inline void paged_attention_v2(
        at::Tensor&             out,
        at::Tensor&             exp_sums,       // [num_seqs, num_heads, max_partitions]
        at::Tensor&             max_logits,     // [num_seqs, num_heads, max_partitions]
        at::Tensor&             tmp_out,        // [num_seqs, num_heads, max_partitions, head_size]
        const at::Tensor&       query,
        const at::Tensor&       key_cache,
        const at::Tensor&       value_cache,
        int                     num_kv_heads,
        float                   scale,
        const at::Tensor&       block_tables,
        const at::Tensor&       seq_lens,
        int                     block_size,
        int                     max_seq_len,
        const std::optional<at::Tensor>& alibi_slopes,
        const std::string&      kv_cache_dtype,
        float                   kv_scale       = 1.f,
        int                     tp_rank        = 0,
        int                     blocksparse_local_blocks = 0,
        float                   blocksparse_vert_stride  = 1.f,
        float                   blocksparse_block_size   = 64.f,
        float                   blocksparse_head_sliding_step = 0.f) {

    // Partition size matches vLLM default: 512 tokens per partition
    constexpr int PARTITION_SIZE = 512;

    int64_t num_seqs       = query.size(0);
    int64_t num_heads      = query.size(1);
    int64_t head_size      = query.size(2);
    int64_t max_blocks_seq = block_tables.size(1);
    int64_t heads_per_kv   = num_heads / num_kv_heads;
    int     x              = 8;

    if (key_cache.scalar_type() == c10::ScalarType::Half ||
        key_cache.scalar_type() == c10::ScalarType::BFloat16) x = 16;

    int64_t kv_hs_k = (head_size / x) * block_size * x;
    int64_t kv_hs_v =  head_size * block_size;

    const int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
    const int32_t* btables_ptr  = block_tables.data_ptr<int32_t>();
    float*         out_ptr      = out.data_ptr<float>();
    const float*   q_ptr        = query.data_ptr<float>();
    float*         exp_ptr      = exp_sums.data_ptr<float>();
    float*         maxl_ptr     = max_logits.data_ptr<float>();
    float*         tmp_ptr      = tmp_out.data_ptr<float>();

    int64_t max_partitions = exp_sums.size(2);

    std::vector<float> q_buf(head_size), k_buf(head_size), v_buf(head_size);

    for (int64_t s = 0; s < num_seqs; ++s) {
        int ctx_len = seq_lens_ptr[s];
        if (ctx_len == 0) continue;

        const int32_t* btable = btables_ptr + s * max_blocks_seq;
        int num_partitions    = (ctx_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t kv_h = h / heads_per_kv;
            const float* q = q_ptr + (s * num_heads + h) * head_size;
            std::memcpy(q_buf.data(), q, head_size * sizeof(float));

            // Pass 1: per-partition softmax denominator + partial output
            for (int part = 0; part < num_partitions; ++part) {
                int token_start = part * PARTITION_SIZE;
                int token_end   = std::min(token_start + PARTITION_SIZE, ctx_len);
                int part_len    = token_end - token_start;

                std::vector<float> logits(part_len);
                int li = 0;
                for (int ti = token_start; ti < token_end; ++ti, ++li) {
                    int bl  = ti / block_size;
                    int sl  = ti % block_size;
                    int32_t pb = btable[bl];
                    const float* k_base =
                        key_cache.data_ptr<float>()
                        + pb * num_kv_heads * kv_hs_k
                        + kv_h * kv_hs_k;
                    paged_attn_detail::load_key<float>(
                        k_buf.data(), k_base, sl, head_size, block_size, x);
                    logits[li] =
                        paged_attn_detail::qk_dot(q_buf.data(), k_buf.data(), head_size, scale);
                }

                // Per-partition max / exp-sum
                float mx = logits[0];
                for (int i=1;i<part_len;++i) mx = std::max(mx, logits[i]);
                float esum = 0.f;
                for (int i=0;i<part_len;++i) { logits[i] = std::exp(logits[i]-mx); esum+=logits[i]; }

                int64_t part_idx = (s * num_heads + h) * max_partitions + part;
                exp_ptr [part_idx] = esum;
                maxl_ptr[part_idx] = mx;

                // Partial output accumulation
                float* tmp = tmp_ptr
                           + (s * num_heads + h) * max_partitions * head_size
                           + part * head_size;
                std::fill(tmp, tmp + head_size, 0.f);

                for (int ti = token_start, li2 = 0; ti < token_end; ++ti, ++li2) {
                    int bl = ti / block_size, sl = ti % block_size;
                    int32_t pb = btable[bl];
                    const float* v_base =
                        value_cache.data_ptr<float>()
                        + pb * num_kv_heads * kv_hs_v
                        + kv_h * kv_hs_v;
                    paged_attn_detail::load_value<float>(
                        v_buf.data(), v_base, sl, head_size, block_size);
                    paged_attn_detail::accum_value(tmp, v_buf.data(), logits[li2] / esum, head_size);
                }
            }

            // Pass 2: reduce partitions
            float* o = out_ptr + (s * num_heads + h) * head_size;
            std::fill(o, o + head_size, 0.f);

            // Global softmax: re-weight each partition's sum
            int64_t ph_base = (s * num_heads + h) * max_partitions;
            float global_max = maxl_ptr[ph_base];
            for (int p=1;p<num_partitions;++p)
                global_max = std::max(global_max, maxl_ptr[ph_base+p]);

            float global_sum = 0.f;
            for (int p=0;p<num_partitions;++p)
                global_sum += exp_ptr[ph_base+p]
                            * std::exp(maxl_ptr[ph_base+p] - global_max);

            for (int p=0;p<num_partitions;++p) {
                float w = exp_ptr[ph_base+p]
                        * std::exp(maxl_ptr[ph_base+p] - global_max)
                        / global_sum;
                const float* tmp = tmp_ptr
                                 + (s * num_heads + h) * max_partitions * head_size
                                 + p * head_size;
                paged_attn_detail::accum_value(o, tmp, w, head_size);
            }
        }
    }
}

#endif  // VLLM_ARM_KERNELS_H
