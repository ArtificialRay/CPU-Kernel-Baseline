// sgl_torch_binding.cpp — PyTorch dispatcher shim
//
// This is the ONLY file that may include <torch/...> headers.
// All computation is delegated to the pure C++ kernels above.
// Compile WITH libtorch only for this file; all other .cpp files are clean.
//
// Registration pattern mirrors the original sgl-kernel common_extension.cc
// but is minimal — just unwrap tensors, forward to C kernels, done.
//
// Build alongside the kernel objects:
//   g++ -O2 -shared -fPIC sgl_torch_binding.cpp sgl_norm.o sgl_activation.o \
//       sgl_rope.o sgl_gemm.o sgl_moe.o sgl_collective.o \
//       $(python3 -m torch.utils.cpp_extension --include-dirs) \
//       -ltorch -lc10 -o sgl_cpu_kernels_torch.so

#include <torch/extension.h>
#include "sgl_cpu_kernels.h"

// ---------------------------------------------------------------------------
// Helper: assert tensor is CPU, contiguous, correct dtype
// ---------------------------------------------------------------------------
static inline bf16_t* bf16_ptr(torch::Tensor& t) {
    TORCH_CHECK(t.device().is_cpu());
    TORCH_CHECK(t.dtype() == torch::kBFloat16);
    TORCH_CHECK(t.is_contiguous());
    return reinterpret_cast<bf16_t*>(t.data_ptr<at::BFloat16>());
}
static inline const bf16_t* cbf16_ptr(const torch::Tensor& t) {
    TORCH_CHECK(t.device().is_cpu());
    TORCH_CHECK(t.dtype() == torch::kBFloat16);
    TORCH_CHECK(t.is_contiguous());
    return reinterpret_cast<const bf16_t*>(t.data_ptr<at::BFloat16>());
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
void th_rms_norm(torch::Tensor out, torch::Tensor x,
                 torch::Tensor weight, float eps) {
    int n_tokens   = x.size(0);
    int hidden_dim = x.size(1);
    sgl_rms_norm_bf16(bf16_ptr(out), cbf16_ptr(x), cbf16_ptr(weight),
                      n_tokens, hidden_dim, eps);
}

void th_add_rms_norm(torch::Tensor residual, torch::Tensor out,
                     torch::Tensor x, torch::Tensor weight, float eps) {
    int n_tokens   = x.size(0);
    int hidden_dim = x.size(1);
    sgl_add_rms_norm_bf16(bf16_ptr(residual), bf16_ptr(out),
                          cbf16_ptr(x), cbf16_ptr(weight),
                          n_tokens, hidden_dim, eps);
}

// ---------------------------------------------------------------------------
// SiLU-and-mul
// ---------------------------------------------------------------------------
void th_silu_and_mul(torch::Tensor out, torch::Tensor x, torch::Tensor y) {
    int n_tokens = x.size(0);
    int d        = x.size(1);
    sgl_silu_and_mul_bf16(bf16_ptr(out), cbf16_ptr(x), cbf16_ptr(y),
                          n_tokens, d);
}

void th_silu_and_mul_inplace(torch::Tensor buf) {
    int n_tokens = buf.size(0);
    int d        = buf.size(1) / 2;
    sgl_silu_and_mul_inplace_bf16(bf16_ptr(buf), n_tokens, d);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------
void th_rope_neox(torch::Tensor q_out, torch::Tensor k_out,
                  torch::Tensor q,     torch::Tensor k,
                  torch::Tensor cos_t, torch::Tensor sin_t,
                  torch::Tensor pos) {
    int n_tokens   = q.size(0);
    int n_q_heads  = q.size(1);
    int n_kv_heads = k.size(1);
    int head_dim   = q.size(2);
    int max_pos    = cos_t.size(0);
    sgl_rope_neox_bf16(
        bf16_ptr(q_out), bf16_ptr(k_out),
        cbf16_ptr(q), cbf16_ptr(k),
        cos_t.data_ptr<float>(), sin_t.data_ptr<float>(),
        pos.data_ptr<int32_t>(),
        n_tokens, n_q_heads, n_kv_heads, head_dim, max_pos);
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------
torch::Tensor th_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(torch::kCPU));
    sgl_gemm_bf16(bf16_ptr(C), cbf16_ptr(A), cbf16_ptr(B), M, N, K);
    return C;
}

// ---------------------------------------------------------------------------
// Fused MoE
// ---------------------------------------------------------------------------
void th_fused_experts(torch::Tensor out, torch::Tensor hidden_states,
                      torch::Tensor w1,  torch::Tensor w2,
                      torch::Tensor topk_weights, torch::Tensor topk_ids) {
    int n_tokens   = hidden_states.size(0);
    int K          = hidden_states.size(1);
    int n_experts  = w1.size(0);
    int two_N      = w1.size(1);   // 2*N
    int N          = two_N / 2;
    int topk       = topk_ids.size(1);
    sgl_fused_experts_bf16(
        bf16_ptr(out), cbf16_ptr(hidden_states),
        cbf16_ptr(w1), cbf16_ptr(w2),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int32_t>(),
        n_tokens, n_experts, K, N, topk);
}

void th_shared_expert(torch::Tensor out, torch::Tensor hidden_states,
                      torch::Tensor w1,  torch::Tensor w2, float scale) {
    int n_tokens = hidden_states.size(0);
    int K        = hidden_states.size(1);
    int N        = w1.size(0) / 2;
    sgl_shared_expert_bf16(
        bf16_ptr(out), cbf16_ptr(hidden_states),
        cbf16_ptr(w1), cbf16_ptr(w2), scale,
        n_tokens, K, N);
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(sgl_cpu_kernels_torch, m) {
    m.doc() = "SGLang standalone CPU kernels (no sglang runtime dependency)";

    m.def("rms_norm",          &th_rms_norm,          "RMSNorm BF16");
    m.def("add_rms_norm",      &th_add_rms_norm,      "Add+RMSNorm BF16");
    m.def("silu_and_mul",      &th_silu_and_mul,      "SiLU*mul BF16");
    m.def("silu_and_mul_inplace", &th_silu_and_mul_inplace, "SiLU*mul inplace BF16");
    m.def("rope_neox",         &th_rope_neox,         "RoPE Neox BF16");
    m.def("gemm",              &th_gemm,              "GEMM BF16");
    m.def("fused_experts",     &th_fused_experts,     "Fused MoE BF16");
    m.def("shared_expert",     &th_shared_expert,     "Shared expert BF16");
}
