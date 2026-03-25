This repo includes cpu kernels from paddleLite. Kernels are implemented in `ggml/src/ggml-cpu`, while calls utilities in a bunch of files in `ggml/src/ggml-cpu` or `ggml/src`

One kernel implementation contains both naive C/C++ implementation and ARM NEON/FP16 intrinsic implementation, called by using different compile flags

Type of CPU kernels are:

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Activations** | abs | Absolute value |
| | sgn | Sign function |
| | neg | Negation |
| | step | Step function |
| | tanh | Hyperbolic tangent |
| | elu | Exponential Linear Unit |
| | relu | Rectified Linear Unit |
| | leaky_relu | Leaky ReLU |
| | sigmoid | Sigmoid activation |
| | hardsigmoid | Hard sigmoid approximation |
| | exp | Exponential function |
| | hardswish | Hard swish activation |
| | sqr | Square (x²) |
| | sqrt | Square root |
| | sin | Sine function |
| | cos | Cosine function |
| | log | Natural logarithm |
| | expm1 | exp(x) - 1 |
| | softplus | Softplus activation (log(1 + exp(x))) |
| | floor | Floor function |
| | ceil | Ceiling function |
| | round | Round to nearest integer |
| | silu | SiLU / Swish activation |
| | swiglu | SwiGLU (SiLU with gating) |
| | silu_back | Backpropagation for SiLU |
| | gelu | Gaussian Error Linear Unit |
| | unary | Generic unary operation dispatcher |
| **Attention** | flash_attn_ext | Flash attention extended version |
| | flash_attn_back | Flash attention backpropagation |
| | get_rel_pos | Get relative position embeddings |
| | add_rel_pos | Add relative position embeddings |
| **Convolution** | conv_transpose_1d | 1D transposed convolution |
| | conv_transpose_2d | 2D transposed convolution |
| | conv_1d | 1D convolution |
| | conv_2d | 2D convolution |
| | conv_3d | 3D convolution |
| | im2col | Image to column transformation |
| | im2col_back | Backpropagation for im2col |
| | pool_1d | 1D pooling |
| | pool_2d | 2D pooling |
| | upscale | Upscaling operation |
| | pad | Padding operation |
| | arange | Generate range of values |
| | timestep_embedding | Timestep embedding for diffusion models |
| **Elementwise** | dup | Duplicate/copy tensor |
| | cpy | Copy tensor (alias for dup) |
| | cont | Make tensor contiguous |
| | add | Element-wise addition |
| | add_non_quantized | Addition for non-quantized types |
| | add_id | Addition with ID indexing |
| | add1 | Add scalar or broadcast |
| | sub | Element-wise subtraction |
| | mul | Element-wise multiplication |
| | div | Element-wise division |
| | acc | Accumulate operation |
| | sum | Sum of tensors |
| | cumsum | Cumulative sum |
| | sum_rows | Sum along rows |
| | mean | Mean of tensor elements |
| | count_equal | Count equal elements |
| | fill | Fill tensor with value |
| | tri | Create triangular matrix |
| | scale | Scale tensor (multiply by constant) |
| | set | Set tensor values |
| | clamp | Clamp values to range |
| **GEMM/GEMV** | mul_mat | Matrix multiplication |
| | mul_mat_id | Matrix multiplication with ID |
| | out_prod | Outer product |
| | gemm | General Matrix Multiply |
| | gemv | General Matrix-Vector multiply |
| **Normalization** | norm | L2 normalization |
| | rms_norm | Root Mean Square normalization |
| | rms_norm_back | Backpropagation for RMS norm |
| | group_norm | Group normalization |
| | soft_max | Softmax activation |
| | soft_max_ext_back | Extended softmax backpropagation |
| **Quantization** | quantize | Quantization operations |
| | dequantize | Dequantization operations |
| | repack | Repack quantized data |
| **RoPE (Rotary Positional Embedding)** | rope | Rotary positional embedding |
| | rope_back | Backpropagation for RoPE |
| | mrope | Multi-dimensional RoPE |
| **Shape Manipulation** | repeat | Repeat tensor along axes |
| | repeat_back | Backpropagation for repeat |
| | concat | Concatenate tensors |
| | get_rows | Get rows by indices |
| | set_rows | Set rows by indices |
| | get_rows_back | Backpropagation for get_rows |
| | diag | Diagonal operations |
| | diag_mask_inf | Diagonal mask with infinity |
| | diag_mask_zero | Diagonal mask with zero |
| | roll | Roll tensor along axis |
| | permute | Permute tensor dimensions |
| | transpose | Transpose matrix |
| | view | Create view of tensor |
| | reshape | Reshape tensor |
| **Sort & Reduction** | argmax | Argument of maximum value |
| | argsort | Argument sort |
| | top_k | Top-k operation |
| **State Space Models (SSM)** | ssm_conv | SSM convolution |
| | ssm_scan | SSM scan operation |
| | win_part | Window partition |
| | win_unpart | Window unpartition |
| | gla | Gated Linear Attention |
| | gated_delta_net | Gated Delta Network |
| | solve_tri | Solve triangular system |
| **Loss Functions** | cross_entropy_loss | Cross entropy loss |
| | cross_entropy_loss_back | Backpropagation for cross entropy |
| **Other** | custom | Custom user-defined operations |
| | noop | No operation |
| **Accelerated Backends** | kleidiai | KleidiAI backend acceleration (ARM optimized) |
| | amx | Intel Advanced Matrix Extensions support |
| | llamafile | Llamafile optimizations |
| | spacemit | SpaceMIT accelerator support |

**Note:** GGML CPU backend includes:
- **SIMD support**: AVX512, AVX2, SSE2, ARM NEON, ARM SVE, RISC-V Vector, IBM VXE
- **Quantization formats**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M
- **Data types**: FP32, FP16, BF16, INT8, INT32
- **Hardware acceleration**: Intel AMX, ARM KleidiAI, specialized micro-kernels