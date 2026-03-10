#pragma once
// sgl_cpu_kernels.h — Public API
// All functions operate on raw pointers.
// NO PyTorch, NO ATen, NO libtorch dependency.
// Thread-safety: kernels use OpenMP internally; callers must not share
// output buffers across concurrent calls to the same kernel.
//
// Data layout conventions (matching SGLang sgl-kernel originals):
//   bf16_t = uint16_t (brain float16, IEEE 754 upper 16 bits of f32)
//   All matrices are row-major unless noted.

#include <stdint.h>
#include <stddef.h>

typedef uint16_t bf16_t;

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// 1. RMSNorm
//    out[i] = (x[i] / rms(x)) * weight[i]   (+ optional residual add)
//    x, out:    [n_tokens, hidden_dim]  bf16
//    weight:    [hidden_dim]            bf16
//    eps:       float (typically 1e-6)
// ===========================================================================
void sgl_rms_norm_bf16(
    bf16_t*       __restrict__ out,          // [n_tokens, hidden_dim]
    const bf16_t* __restrict__ x,            // [n_tokens, hidden_dim]
    const bf16_t* __restrict__ weight,       // [hidden_dim]
    int n_tokens,
    int hidden_dim,
    float eps
);

// Fused residual-add + RMSNorm: residual += x, then rms-norm(residual)
void sgl_add_rms_norm_bf16(
    bf16_t*       __restrict__ residual,     // [n_tokens, hidden_dim]  in+out
    bf16_t*       __restrict__ out,          // [n_tokens, hidden_dim]  out
    const bf16_t* __restrict__ x,            // [n_tokens, hidden_dim]
    const bf16_t* __restrict__ weight,       // [hidden_dim]
    int n_tokens,
    int hidden_dim,
    float eps
);

// ===========================================================================
// 2. SiLU-and-Mul (gate activation used in SwiGLU FFN)
//    out[i] = silu(x[i]) * y[i]
//    x, y, out: [n_tokens, intermediate_dim]  bf16
//    In-place variant writes into x (y is the second half of the same buffer,
//    i.e. the packed gate layout used by SGLang).
// ===========================================================================
void sgl_silu_and_mul_bf16(
    bf16_t*       __restrict__ out,          // [n_tokens, d]
    const bf16_t* __restrict__ x,            // [n_tokens, d]   — gate
    const bf16_t* __restrict__ y,            // [n_tokens, d]   — up-proj
    int n_tokens,
    int d
);

// In-place version: input buffer is [n_tokens, 2*d], first half is gate,
// second half is up-proj; output overwrites first half [n_tokens, d].
void sgl_silu_and_mul_inplace_bf16(
    bf16_t*       __restrict__ buf,          // [n_tokens, 2*d] in, [n_tokens, d] out
    int n_tokens,
    int d
);

// ===========================================================================
// 3. Rotary Positional Embedding (RoPE) — Neox layout
//    q_out, k_out: rotated queries/keys  [n_tokens, n_heads, head_dim]
//    q,  k:        input                 [n_tokens, n_heads, head_dim]
//    cos, sin:     precomputed tables    [max_seq_len, head_dim/2]  f32
//    positions:    token position ids    [n_tokens]  int32
//    Handles head_dim rotary on first head_dim/2 elements (Neox style).
// ===========================================================================
void sgl_rope_neox_bf16(
    bf16_t*       __restrict__ q_out,
    bf16_t*       __restrict__ k_out,
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ k,
    const float*  __restrict__ cos_table,    // [max_pos, head_dim/2]
    const float*  __restrict__ sin_table,    // [max_pos, head_dim/2]
    const int32_t* __restrict__ positions,   // [n_tokens]
    int n_tokens,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_pos
);

// ===========================================================================
// 4. Packed GEMM (weight_packed_linear replacement)
//    C = A * B^T   (no bias variant for now)
//    A: [M, K]  bf16  (activations)
//    B: [N, K]  bf16  (packed weights, row = output neuron)
//    C: [M, N]  bf16
//    Uses AVX-512 BF16 dpbf16 dot-product instruction.
// ===========================================================================
void sgl_gemm_bf16(
    bf16_t*       __restrict__ C,            // [M, N]
    const bf16_t* __restrict__ A,            // [M, K]
    const bf16_t* __restrict__ B,            // [N, K]
    int M, int N, int K
);

// With bias: C = A*B^T + bias
void sgl_gemm_bias_bf16(
    bf16_t*       __restrict__ C,
    const bf16_t* __restrict__ A,
    const bf16_t* __restrict__ B,
    const bf16_t* __restrict__ bias,         // [N]
    int M, int N, int K
);

// ===========================================================================
// 5. BMM — Batched Matrix Multiply
//    C[b] = A[b] * B[b]^T
//    A: [batch, M, K]  bf16
//    B: [batch, N, K]  bf16
//    C: [batch, M, N]  bf16
// ===========================================================================
void sgl_bmm_bf16(
    bf16_t*       __restrict__ C,
    const bf16_t* __restrict__ A,
    const bf16_t* __restrict__ B,
    int batch, int M, int N, int K
);

// ===========================================================================
// 6. Fused MoE (fused_experts_cpu replacement) — BF16 weights
//
//   hidden_states:  [n_tokens, K]         bf16   input
//   w1:             [n_experts, 2*N, K]   bf16   gate+up proj weight (packed)
//   w2:             [n_experts, K, N]     bf16   down proj weight
//   topk_weights:   [n_tokens, topk]      f32
//   topk_ids:       [n_tokens, topk]      int32
//   out:            [n_tokens, K]         bf16   output
//
//   Algorithm:
//     For each selected expert e of each token t:
//       h1 = w1[e] @ x[t]          → [2N]
//       gate = silu(h1[:N]) * h1[N:]  → [N]
//       h2 = w2[e] @ gate           → [K]
//       out[t] += topk_weights[t,k] * h2
// ===========================================================================
void sgl_fused_experts_bf16(
    bf16_t*        __restrict__ out,           // [n_tokens, K]
    const bf16_t*  __restrict__ hidden_states, // [n_tokens, K]
    const bf16_t*  __restrict__ w1,            // [n_experts, 2*N, K]
    const bf16_t*  __restrict__ w2,            // [n_experts, K, N]
    const float*   __restrict__ topk_weights,  // [n_tokens, topk]
    const int32_t* __restrict__ topk_ids,      // [n_tokens, topk]
    int n_tokens,
    int n_experts,
    int K,           // hidden dim
    int N,           // intermediate dim (per expert, before gate split)
    int topk
);

// ===========================================================================
// 7. Shared Expert (shared_expert_cpu replacement)
//    Same as fused_experts but processes a single always-active expert.
//    w1: [2*N, K]  bf16
//    w2: [K, N]    bf16
// ===========================================================================
void sgl_shared_expert_bf16(
    bf16_t*        __restrict__ out,           // [n_tokens, K]
    const bf16_t*  __restrict__ hidden_states, // [n_tokens, K]
    const bf16_t*  __restrict__ w1,            // [2*N, K]
    const bf16_t*  __restrict__ w2,            // [K, N]
    float          scale,                      // routed_scaling_factor (1.0 if none)
    int n_tokens,
    int K,
    int N
);

// ===========================================================================
// 8. AllReduce (ring sum over shared-memory threads — for single-node TP)
//    buf:  [n_elements]  f32  — each TP rank writes its shard; after the call
//          every rank has the global sum.
//    Requires all TP threads to call this concurrently with the SAME buf ptr.
//    Uses a simple barrier + accumulate approach via OpenMP.
// ===========================================================================
void sgl_allreduce_sum_f32(
    float*  __restrict__ buf,
    int     n_elements,
    int     n_ranks,           // number of TP ranks (OpenMP threads)
    int     rank_id            // this thread's rank
);

#ifdef __cplusplus
} // extern "C"
#endif
