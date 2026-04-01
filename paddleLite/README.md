This repo includes cpu kernels from paddleLite. Kernels are implemented in `paddleLite/kernels/arm`, while calls utilities in `paddleLite/backends/arm/math` or `paddleLite/backends/arm/arm_dnn_library` . 

One kernel implementation contains both naive C/C++ implementation and ARM NEON/FP16 intrinsic implementation, called by using different compile flags

Type of CPU kernels are:

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Activation** | sigmoid | Sigmoid activation function |
| | relu | Rectified Linear Unit activation |
| | relu6 | ReLU6 activation (clipped at 6) |
| | prelu | Parametric ReLU activation |
| | leaky_relu | Leaky ReLU with negative slope |
| | tanh | Hyperbolic tangent activation |
| | swish | Swish activation (x * sigmoid(beta * x)) |
| | hard_sigmoid | Hard sigmoid approximation |
| | relu_clipped | ReLU with clipping at max value |
| | thresholded_relu | ReLU with threshold parameter |
| | softmax | Softmax normalization |
| | softplus | Softplus activation (log(1 + exp(x))) |
| | mish | Mish activation (x * tanh(softplus(x))) |
| | gelu | Gaussian Error Linear Unit |
| | silu | Sigmoid Linear Unit (same as swish with beta=1) |
| | exp | Exponential function |
| | log | Natural logarithm |
| | floor | Floor function |
| | erf | Error function |
| | sign | Sign function |
| | reciprocal | Reciprocal (1/x) |
| | negative | Negation (-x) |
| | pow | Power function |
| | clip | Clip values to range [min, max] |
| | dropout | Dropout regularization |
| **Convolution & Pooling** | conv2d | 2D convolution |
| | depthwise_conv2d | Depthwise 2D convolution |
| | conv2d_transpose | 2D transposed convolution (deconvolution) |
| | depthwise_conv2d_transpose | Depthwise transposed convolution |
| | sparse_conv2d | Sparse 2D convolution |
| | deformable_conv | Deformable convolution |
| | pool2d | 2D pooling (max/average) |
| **GEMM** | fc | Fully connected layer |
| | matmul | Matrix multiplication |
| | matmul_v2 | Matrix multiplication v2 (with broadcasting) |
| | mul | Element-wise or matrix multiplication |
| | mul_grad | Gradient for mul operator |
| **Elementwise** | elementwise_add | Element-wise addition |
| | elementwise_sub | Element-wise subtraction |
| | elementwise_mul | Element-wise multiplication |
| | elementwise_div | Element-wise division |
| | elementwise_max | Element-wise maximum |
| | elementwise_min | Element-wise minimum |
| | elementwise_floordiv | Element-wise floor division |
| | fusion_elementwise_add_activation | Fused element-wise add + activation |
| | fusion_elementwise_sub_activation | Fused element-wise sub + activation |
| | fusion_elementwise_mul_activation | Fused element-wise mul + activation |
| | fusion_elementwise_div_activation | Fused element-wise div + activation |
| | elementwise_add_grad | Gradient for element-wise add |
| | elementwise_sub_grad | Gradient for element-wise sub |
| | elementwise_div_grad | Gradient for element-wise div |
| | elementwise_max_grad | Gradient for element-wise max |
| | scale | Scale tensor by constant (x * scale + bias) |
| | sum | Sum of input tensors |
| | axpy | AXPY operation (a*x + y) |
| **Normalization** | batch_norm | Batch normalization |
| | sync_batch_norm | Synchronized batch normalization |
| | layer_norm | Layer normalization |
| | group_norm | Group normalization |
| | instance_norm | Instance normalization |
| | affine_channel | Affine channel transformation |
| | lrn | Local Response Normalization |
| **Quantization** | calib | Calibration for quantization |
| | calib_inplace | In-place calibration |
| | dequantize_log | Dequantization with logarithmic scale |
| | lookup_table_dequant | Dequantization with lookup table |
| **Reduction** | reduce_sum | Reduce sum along axes |
| | reduce_mean | Reduce mean along axes |
| | reduce_max | Reduce max along axes |
| | reduce_min | Reduce min along axes |
| | reduce_prod | Reduce product along axes |
| | argmax | Argument of maximum value |
| | mean | Mean of all elements |
| | mean_grad | Gradient for mean operator |
| **Sequence & NLP** | lstm | Long Short-Term Memory cell |
| | gru | Gated Recurrent Unit cell |
| | gru_unit | GRU unit operation |
| | rnn | Basic RNN cell |
| | sequence_conv | Sequence convolution |
| | sequence_pool | Sequence pooling |
| | sequence_pool_grad | Gradient for sequence pooling |
| | sequence_expand_as | Expand sequence as another sequence |
| | lookup_table | Embedding lookup table |
| | viterbi_decode | Viterbi decoding algorithm |
| | fused_attention | Fused attention mechanism |
| **Tensor Manipulation** | concat | Concatenate tensors along axis |
| | split | Split tensor into multiple tensors |
| | slice | Slice tensor along axes |
| | transpose | Transpose tensor dimensions |
| | layout | Change tensor layout/format |
| | pad2d | 2D padding |
| | interpolate | Interpolation (resize) |
| | pixel_shuffle | Pixel shuffle (depth to space) |
| | affine_grid | Generate affine transformation grid |
| | grid_sampler | Grid sampling operation |
| | scatter | Scatter operation |
| | split_lod_tensor | Split LoD (Level of Detail) tensor |
| **Detection** | box_coder | Encode/decode bounding boxes |
| | decode_bboxes | Decode bounding box predictions |
| **Optimization** | sgd | Stochastic Gradient Descent optimizer |