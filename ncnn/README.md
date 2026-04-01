Remember to configure your working directory to `/your-clone-dir/ncnn`
Analyze dependency with claude code skills: [`.claude/skills/analyze-ncnn-dependency/SKILL.md`](.claude/skills/analyze-ncnn-dependency/SKILL.md). Remember to parse argument when call the skill: 
```bash
/analyze-ncnn-dependency working-directory
```
which could be: `c-partially-optimized` or `arm-heavy-optimized`

Generate testcases with claude code skills: [`.claude/skills/generate-testcase/SKILL.md`](.claude/skills/generate-testcase/SKILL.md)


This repo includes cpu kernels from ncnn, containing 2-levels of kernel implementations:
- cpu-partially-optimized: C/C++ naive or partially-optimized kernels in 9 categories

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Activation** | absval | Absolute value activation |
| | bnll | Binomial Normal Log Likelihood |
| | celu | Continuously Differentiable Exponential Linear Unit |
| | clip | Clip values to range [min, max] |
| | elu | Exponential Linear Unit |
| | erf | Error function |
| | exp | Exponential function |
| | gelu | Gaussian Error Linear Unit |
| | glu | Gated Linear Unit |
| | hardsigmoid | Hard sigmoid approximation |
| | hardswish | Hard swish activation |
| | log | Natural logarithm |
| | mish | Mish activation (x * tanh(softplus(x))) |
| | power | Power function (scale * x^power + shift) |
| | prelu | Parametric ReLU |
| | relu | Rectified Linear Unit |
| | selu | Scaled Exponential Linear Unit |
| | shrink | Soft shrinkage function |
| | sigmoid | Sigmoid activation |
| | softplus | Softplus activation (log(1 + exp(x))) |
| | swish | Swish activation (x * sigmoid(x)) |
| | tanh | Hyperbolic tangent |
| | threshold | Threshold activation |
| **Attention** | multiheadattention | Multi-head attention mechanism |
| | sdpa | Scaled Dot-Product Attention |
| **Convolution** | convolution | 2D convolution |
| | convolution1d | 1D convolution |
| | convolution3d | 3D convolution |
| | convolutiondepthwise | Depthwise 2D convolution |
| | convolutiondepthwise1d | Depthwise 1D convolution |
| | convolutiondepthwise3d | Depthwise 3D convolution |
| | deconvolution | 2D transposed convolution |
| | deconvolution1d | 1D transposed convolution |
| | deconvolution3d | 3D transposed convolution |
| | deconvolutiondepthwise | Depthwise 2D transposed convolution |
| | deconvolutiondepthwise1d | Depthwise 1D transposed convolution |
| | deconvolutiondepthwise3d | Depthwise 3D transposed convolution |
| | deformableconv2d | Deformable 2D convolution |
| **GEMM** | gemm | General Matrix Multiply (GEMM) |
| | innerproduct | Fully connected / inner product layer |
| | matmul | Matrix multiplication |
| | einsum | Einstein summation convention |
| **Normalization** | batchnorm | Batch normalization |
| | groupnorm | Group normalization |
| | instancenorm | Instance normalization |
| | layernorm | Layer normalization |
| | rmsnorm | Root Mean Square normalization |
| | lrn | Local Response Normalization |
| | mvn | Mean Variance Normalization |
| | normalize | L2 normalization |
| **Quantization** | quantize | Quantization (float to int8) |
| | dequantize | Dequantization (int8 to float) |
| | requantize | Requantization (int8 to int8 with different scale) |
| **Recurrent** | lstm | Long Short-Term Memory cell |
| | gru | Gated Recurrent Unit cell |
| | rnn | Basic Recurrent Neural Network cell |
| **Reduction & Pooling** | pooling | 2D pooling (max/average) |
| | pooling1d | 1D pooling |
| | pooling3d | 3D pooling |
| | spp | Spatial Pyramid Pooling |
| | statisticspooling | Statistics pooling (mean/std) |
| | reduction | Reduction operations (sum/mean/max/min/prod) |
| | softmax | Softmax activation |
| | argmax | Argument of maximum value |
| | cumulativesum | Cumulative sum along axis |
| **Tensor Manipulation** | bias | Add bias to tensor |
| | binaryop | Binary operations (add/sub/mul/div/max/min/pow) |
| | cast | Type casting |
| | concat | Concatenate tensors |
| | copyto | Copy tensor to another |
| | crop | Crop tensor |
| | deepcopy | Deep copy of tensor |
| | diag | Diagonal matrix operations |
| | dropout | Dropout regularization |
| | eltwise | Element-wise operations (sum/max/prod) |
| | expanddims | Expand dimensions of tensor |
| | flatten | Flatten tensor |
| | flip | Flip tensor along axes |
| | fold | Fold operation (inverse of unfold) |
| | gridsample | Grid sampling for spatial transformations |
| | interp | Interpolation / resize |
| | packing | Packing/unpacking for optimized layouts |
| | padding | Padding operation |
| | permute | Permute tensor dimensions |
| | pixelshuffle | Pixel shuffle (depth to space) |
| | reorg | Reorganize spatial data |
| | reshape | Reshape tensor |
| | scale | Scale tensor (x * scale + bias) |
| | shufflechannel | Shuffle channels |
| | slice | Slice tensor along axes |
| | split | Split tensor into multiple tensors |
| | squeeze | Remove dimensions of size 1 |
| | tile | Tile/repeat tensor |
| | unaryop | Unary operations (abs/neg/floor/ceil/sqrt/rsqrt/exp/log/sin/cos/tan) |
| | unfold | Unfold operation (extract sliding windows) |
| **Object Detection** | detectionoutput | Detection output processing |
| | yolodetectionoutput | YOLO detection output processing |
| | yolov3detectionoutput | YOLOv3 detection output processing |
| | priorbox | Generate prior boxes for SSD |
| | proposal | Region proposal generation |
| | roipooling | ROI (Region of Interest) pooling |
| | roialign | ROI align operation |
| | psroipooling | Position-Sensitive ROI pooling |
| **Other** | embed | Embedding layer / lookup table |
| | input | Input layer |
| | memorydata | Memory data holder |
| | noop | No operation (pass-through) |
| | rotaryembed | Rotary positional embedding |
| | spectrogram | Convert audio to spectrogram |
| | inversespectrogram | Convert spectrogram back to audio |
| **Common** | fused_activation | Fused activation utilities |

- Arm-heavy-optimized: heavily optimized cpu kernels with ARM intrinsics or ARM assembly in 9 categories. Some kernels call utility in `ncnn/arm-heavy-optimized/common`


| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Attention** | multiheadattention_arm | ARM-optimized multi-head attention mechanism |
| **Convolution** | convolution_arm | ARM-optimized 2D convolution (multiple kernel sizes: 1x1, 2x2, 3x3, 4x4, 5x5, 7x7) |
| | convolution1d_arm | ARM-optimized 1D convolution |
| | convolutiondepthwise_arm | ARM-optimized depthwise convolution (3x3, 5x5) |
| | deconvolution_arm | ARM-optimized transposed convolution (3x3, 4x4) |
| | deconvolutiondepthwise_arm | ARM-optimized depthwise transposed convolution |
| **GEMM** | gemm_arm | ARM-optimized General Matrix Multiply (FP32/FP16/BF16/INT8) |
| | innerproduct_arm | ARM-optimized fully connected layer (FP32/FP16) |
| | matmul_arm | ARM-optimized matrix multiplication |
| **Normalization** | batchnorm_arm | ARM-optimized batch normalization |
| | groupnorm_arm | ARM-optimized group normalization |
| | instancenorm_arm | ARM-optimized instance normalization |
| | layernorm_arm | ARM-optimized layer normalization |
| | rmsnorm_arm | ARM-optimized RMS normalization |
| | lrn_arm | ARM-optimized Local Response Normalization |
| **Quantization** | quantize_arm | ARM-optimized quantization (float to int8) |
| | dequantize_arm | ARM-optimized dequantization (int8 to float) |
| | requantize_arm | ARM-optimized requantization (int8 to int8 with scale change) |
| **Recurrent** | lstm_arm | ARM-optimized LSTM cell (FP32/FP16/INT8) |
| | gru_arm | ARM-optimized GRU cell (FP32/FP16/INT8) |
| | rnn_arm | ARM-optimized basic RNN cell (FP32/FP16/INT8) |
| **Reduction & Pooling** | pooling_arm | ARM-optimized 2D pooling (2x2, 3x3 kernels with pack optimization) |
| | softmax_arm | ARM-optimized softmax |
| **Tensor Manipulation** | absval_arm | ARM-optimized absolute value |
| | bias_arm | ARM-optimized bias addition |
| | binaryop_arm | ARM-optimized binary operations |
| | cast_arm | ARM-optimized type casting (FP32/FP16/BF16) |
| | clip_arm | ARM-optimized clip operation |
| | concat_arm | ARM-optimized tensor concatenation |
| | crop_arm | ARM-optimized crop operation |
| | dropout_arm | ARM-optimized dropout |
| | eltwise_arm | ARM-optimized element-wise operations |
| | flatten_arm | ARM-optimized flatten operation |
| | gelu_arm | ARM-optimized GELU activation |
| | hardsigmoid_arm | ARM-optimized hard sigmoid |
| | hardswish_arm | ARM-optimized hard swish |
| | interp_arm | ARM-optimized interpolation (bilinear/bicubic for packed data) |
| | mish_arm | ARM-optimized Mish activation |
| | packing_arm | ARM-optimized packing/unpacking (for NEON vector layouts) |
| | padding_arm | ARM-optimized padding (with pack4/pack8 support) |
| | pixelshuffle_arm | ARM-optimized pixel shuffle |
| | prelu_arm | ARM-optimized PReLU |
| | relu_arm | ARM-optimized ReLU |
| | reshape_arm | ARM-optimized reshape |
| | scale_arm | ARM-optimized scale operation |
| | selu_arm | ARM-optimized SELU |
| | shufflechannel_arm | ARM-optimized channel shuffle |
| | sigmoid_arm | ARM-optimized sigmoid |
| | slice_arm | ARM-optimized slice operation |
| | swish_arm | ARM-optimized swish activation |
| | tanh_arm | ARM-optimized tanh |
| | unaryop_arm | ARM-optimized unary operations |
| **Common Utilities** | arm_activation.h | ARM activation function utilities |
| | arm_usability.h | ARM instruction set detection and utilities |
| | neon_mathfun.h | NEON-optimized math functions (FP32) |
| | neon_mathfun_fp16s.h | NEON-optimized math functions (FP16) |
| | neon_mathfun_tanh.h | NEON-optimized tanh function |

**Note:** These operators are heavily optimized for ARM architectures with support for:
- **ASIMD (Advanced SIMD/NEON)**: Base ARM NEON instructions
- **ASIMDHP**: NEON FP16 arithmetic
- **ASIMDDP**: NEON dot product instructions
- **ASIMDFHM**: FP16 FML instructions
- **I8MM**: INT8 matrix multiply instructions
- **VFPv4**: Vector Floating Point v4
- Multiple data packing formats (pack4, pack8) for efficient SIMD processing
- INT8, BF16, and FP16 quantization support