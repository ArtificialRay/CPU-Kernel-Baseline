This baseline contains 2-levels of kernel implementations:
- cpu-partially-optimized: C/C++ naive or partially-optimized kernels in 10 categories, some kernels (like convolution and linear) call implementation in `naive-impl-and utils`:

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Convolution** | Conv1D | 1D convolution |
| | Conv2D | 2D convolution |
| | Conv3D | 3D convolution |
| | Deconv | Transposed convolution |
| | Pool1D | 1D pooling |
| | Pool2D | 2D pooling (max/average) |
| | Pool3D | 3D pooling |
| **Activation** | ReLU / ReLU6 | Rectified linear unit (standard and clamped) |
| | PReLU | Parametric ReLU |
| | Sigmoid / HardSigmoid | Sigmoid and its piecewise linear approximation |
| | Swish / HardSwish | Swish and its piecewise linear approximation |
| | ELU / SELU | Exponential linear unit variants |
| | GELU | Gaussian error linear unit |
| | Softmax / LogSoftmax | Softmax and log-domain variant |
| | Softplus / LogSigmoid | Smooth ReLU and log-sigmoid |
| | GLU | Gated linear unit |
| | Clip | Clamp values to [min, max] |
| | Tanh | Hyperbolic tangent |
| **Normalization** | BatchNorm | Batch normalization |
| | LayerNorm | Layer normalization |
| | GroupNorm | Group normalization |
| | InstanceNorm | Instance normalization |
| | Normalize | L1/L2 normalization |
| | Scale | Channel-wise affine scaling |
| | LRN | Local response normalization |
| **Elementwise** | Add / Sub / Mul / Div | Basic arithmetic ops |
| | Pow | Element-wise power |
| | Max / Min | Element-wise maximum/minimum |
| | Abs / Neg | Absolute value / negation |
| | Exp / Log | Exponential and logarithm |
| | Sqrt / Rsqrt | Square root and reciprocal square root |
| | Floor / Ceil | Floor and ceiling rounding |
| | Reciprocal | Element-wise reciprocal |
| | Sign / SignedMul | Sign function and signed multiplication |
| | Sin / Cos / Tan | Trigonometric functions |
| | Asin / Acos / Atan | Inverse trigonometric functions |
| | Erf | Gauss error function |
| | SquaredDiff | Squared difference (x-y)² |
| | BiasAdd | Bias addition |
| | Equal / Greater / Less | Element-wise comparison operators |
| | And / Not | Logical operators |
| | BitShift | Bitwise shift |
| | Where | Conditional selection (ternary) |
| | Unary / BinaryOp | Abstract base classes for unary/binary ops |
| **Tensor Manipulation** | Reshape / Flatten | Change tensor shape |
| | Permute | Transpose / reorder dimensions |
| | Concat | Concatenate tensors along an axis |
| | SplitV | Split tensor into multiple parts |
| | Squeeze / Unsqueeze | Remove or insert size-1 dimensions |
| | Expand / Tile | Broadcast or repeat tensor |
| | Pad / PadV2 | Pad tensor with constant or reflect |
| | StridedSlice / StridedSliceV2 | Strided slicing |
| | Shuffle / PixelShuffle | Channel shuffle and sub-pixel rearrangement |
| | Reorg | Reorg/space-to-depth rearrangement |
| | Upsample | Resize / interpolation upsampling |
| | Cast | Data type casting |
| | Reformat | Memory layout conversion |
| **Reduction** | ReduceMax / ReduceMin | Max/min reduction along axes |
| | ReduceMean / ReduceSum | Mean/sum reduction |
| | ReduceProd | Product reduction |
| | ReduceL1 / ReduceL2 | L1/L2 norm reduction |
| | ReduceLogSum / ReduceLogSumExp | Log-sum and log-sum-exp reduction |
| | ReduceSumSquare | Sum of squares reduction |
| **Indexing** | ArgMax / ArgMin | Index of max/min value along axis |
| | Gather / GatherND | Gather slices by indices |
| | Scatter / ScatterND / ScatterElements | Scatter values into tensor |
| | TopK | Top-K values and indices |
| | NonZero | Indices of non-zero elements |
| **Linear** | InnerProduct | Fully connected layer |
| | MatMul | General matrix multiplication |
| | Inverse | Matrix inverse |
| | Einsum | Einstein summation |
| **Other** | LSTM | Long short-term memory recurrent layer |
| | DetectionOutput / DetectionPostProcess | SSD-style detection post-processing |
| | PriorBox | Generate anchor/prior boxes |
| | ROIAlign | Region of interest alignment |
| | NonMaxSuppression | NMS for bounding box filtering |
| | GridSample | Spatial transformer sampling |
| | Shape / Size | Query tensor shape metadata |
| | Const / ConstantOfShape | Constant tensor generation |
| | OneHot | One-hot encoding |
| | Range | Generate range tensor (arange) |
| | HDRGuide | HDR image guided processing |
| | Histogram | Histogram computation |

-Arm-heavy-optimized: heavily optimized cpu kernels with ARM intrinsics or ARM assembly in 9 categories. Some kernels use implementation in /compute and /compute_arm82

| Category | Operator Name | Description |
|----------|---------------|-------------|
| **Convolution & Deconvolution** | Conv2D (1x1) | 1×1 optimized convolution |
| | Conv2D (3x3) | 3×3 Winograd/direct convolution |
| | Conv2D (c3) | 3-channel input convolution |
| | Conv2D (Depthwise) | Depthwise separable conv |
| | Conv2D (Depthwise S1) | Stride-1 depthwise conv |
| | Conv2D (Group) | Group convolution |
| | Conv2D (Int8 1x1) | Int8 quantized 1×1 conv |
| | Conv2D (Int8 Common) | Int8 quantized general conv |
| | Conv2D (Int8 Depthwise) | Int8 quantized depthwise conv |
| | Conv2D (Int8 SDOT Common) | SDOT-accelerated int8 conv |
| | Conv2D (Int8 SDOT DW 3x3) | SDOT-accelerated int8 depthwise 3×3 |
| | Conv1D | 1D convolution |
| | Pool2D | 2D max/average pooling |
| | *(fp16 variants via headers)* | FP16 3×3 / c3 / common / depthwise / depthwise-s1 |
| | Deconv (Common) | General transposed convolution |
| | Deconv (Depthwise) | Depthwise transposed convolution |
| | Deconv (Stride) | Strided transposed convolution |
| | *(fp16 variants via headers)* | FP16 common / depthwise deconv |
| **Activation** | ReLU | Rectified linear unit |
| | ReLU6 | Clamped ReLU [0, 6] |
| | PReLU | Parametric ReLU |
| | Sigmoid | Sigmoid activation |
| | HardSigmoid | Piecewise linear sigmoid |
| | Swish | Swish / SiLU |
| | HardSwish | Hard Swish |
| | ELU | Exponential linear unit |
| | SELU | Scaled ELU |
| | GELU | Gaussian error linear unit |
| | Softmax | Softmax along an axis |
| | Softplus | log(1 + exp(x)) |
| | LogSigmoid | log(sigmoid(x)) |
| | GLU | Gated linear unit |
| | Clip | Clamp to [min, max] |
| **Normalization** | BatchNorm | Batch normalization |
| | LayerNorm | Layer normalization |
| | GroupNorm | Group normalization |
| | InstanceNorm | Instance normalization |
| | Normalize | L1/L2 spatial normalize |
| | Scale | Channel-wise affine scale |
| **Element-wise** | Add | Element-wise addition |
| | Sub | Element-wise subtraction |
| | Mul | Element-wise multiplication |
| | Div | Element-wise division |
| | Pow | Element-wise power |
| | Max | Element-wise maximum |
| | Min | Element-wise minimum |
| | Abs | Absolute value |
| | Neg | Negation |
| | Exp | Exponential |
| | Log | Natural logarithm |
| | Sqrt | Square root |
| | Reciprocal | 1 / x |
| | Floor | Floor rounding |
| | Sign | Sign function |
| | SignedMul | Signed multiply |
| | Trig | Sin / Cos / Tan |
| | *(base classes)* | Unary / Binary base acc |
| **Tensor Manipulation** | Reshape | Reshape tensor |
| | Permute | Transpose / permute axes |
| | Concat | Concatenate along axis |
| | SplitV | Split tensor |
| | Squeeze | Remove size-1 dimensions |
| | Unsqueeze | Insert size-1 dimensions |
| | Expand | Broadcast-expand tensor |
| | Tile | Repeat tensor |
| | Pad | Pad with constant/reflect |
| | PadV2 | Pad v2 |
| | StridedSlice | Slice with strides |
| | Shuffle | Channel shuffle |
| | PixelShuffle | Sub-pixel convolution rearrange |
| | Reorg | Space-to-depth / depth-to-space |
| | Upsample | Nearest / bilinear upsample |
| | Cast | Type cast |
| | Reformat | Layout reformat (NC4HW4 ↔ NCHW) |
| **Reduction** | ReduceMax | Reduce maximum |
| | ReduceMin | Reduce minimum |
| | ReduceMean | Reduce mean |
| | ReduceSum | Reduce sum |
| | ReduceProd | Reduce product |
| | ReduceL1 | Reduce L1 norm |
| | ReduceL2 | Reduce L2 norm |
| | ReduceLogSum | Reduce log-sum |
| | ReduceLogSumExp | Reduce log-sum-exp |
| | ReduceSumSquare | Reduce sum of squares |
| **Indexing** | ArgMax / ArgMin | Index of max / min value |
| | Gather | Gather slices by index |
| | ScatterND | Scatter updates into tensor |
| **Linear** | InnerProduct (FC) | Fully connected layer |
| | MatMul | General matrix multiplication |
| | Inverse | Matrix inverse |
| **Other** | LSTM | Long short-term memory RNN |
| | DetectionOutput | SSD-style detection decode |
| | PriorBox | Anchor/prior box generator |
| | ROIAlign | RoI-aligned feature pooling |
| | GridSample | Spatial transformer sampling |
