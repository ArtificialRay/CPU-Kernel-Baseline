This repo includes all cpu kernels for Large Language Model inference. Only attention has ARM instruction set optimized code in `cpu_attn_neon.hpp` and `cpu_attn_neon_bfmmla.hpp`. sgl kernels are cpu kernel implemtation for some components (attention, moe) in naive C/C++ or AVX instruction set extension.

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Activations** | silu_and_mul | SiLU activation with element-wise multiplication (used in SwiGLU) |
| | gelu_and_mul | GELU activation with element-wise multiplication (GeGLU, none approximation) |
| | gelu_tanh_and_mul | GELU activation with tanh approximation and multiplication (GeGLU) |
| | gelu_new | GELU implementation used in GPT-2 |
| | gelu_fast | Approximate GELU implementation |
| | gelu_quick | Quick GELU implementation |
| **Attention** | cpu_attn_reshape_and_cache | Reshape and cache key/value tensors for attention |
| | cpu_attention_with_kv_cache | CPU attention with KV cache (supports causal, sliding window, ALiBi) |
| | mla_decode_kvcache | Multi-head latent attention decode with KV cache |
| | get_scheduler_metadata | Generate attention scheduler metadata |
| **GEMM** | cpu_gemm_wna16 | Weight-only quantized GEMM (W4A16, supports GPTQ-style group quantization) |
| **Micro GEMM** | cpu_micro_gemm_amx | Low-level micro GEMM kernel with AMX support |
| | cpu_micro_gemm_vec | Low-level micro GEMM kernel with vector instructions |
| **MoE** | cpu_fused_moe | Fused Mixture of Experts kernel (expert selection + GEMM) |
| | prepack_moe_weight | Prepack MoE weights for optimal performance |
| | dynamic_4bit_int_moe | 4-bit integer quantized MoE operation |
| **Normalization** | rms_norm | Root Mean Square (RMS) Layer Normalization |
| | fused_add_rms_norm | Fused residual add and RMS normalization (in-place) |
| **Quantization/Dequantization** | static_scaled_int8_quant | Static per-tensor int8 quantization with scaling factor |
| | dynamic_scaled_int8_quant | Dynamic int8 quantization (compute scale on-the-fly) |
| **Rotary Embedding** | rotary_embedding | GPT-NeoX or GPT-J style rotary positional embedding |
| **SGLang Kernels - GEMM** | weight_packed_linear | Packed weight linear layer (BF16 with AMX-BF16/VNNI) |
| | convert_weight_packed | Convert weights to packed format |
| | int8_scaled_mm_with_quant | Int8 scaled matrix multiplication with quantization |
| **SGLang Kernels - MoE** | fused_experts_cpu | Fused experts with FP8/Int8 support (optimized for AMX) |
| **Utilities** | init_shm_manager | Initialize shared memory manager for distributed inference |
| | join_shm_manager | Join existing shared memory manager |
| | shm_allreduce | All-reduce operation via shared memory |
| | shm_gather | Gather operation via shared memory |
| | shm_all_gather | All-gather operation via shared memory |
| | shm_send_tensor_list | Send tensor list via shared memory |
| | shm_recv_tensor_list | Receive tensor list via shared memory |
| | compute_slot_mapping_kernel_impl | Compute slot mapping for paged attention |
| | init_cpu_threads_env | Initialize CPU thread environment with core pinning |
| | is_onednn_acl_supported | Check if oneDNN was built with ACL backend |