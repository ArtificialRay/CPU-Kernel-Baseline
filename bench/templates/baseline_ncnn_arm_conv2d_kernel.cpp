// solutions/ncnn/baseline-ncnn-arm/conv2d/<def_name>/kernel.cpp (TEMPLATE)
//
// Stamped verbatim into every baseline-ncnn-arm conv2d Solution by
// scripts/extract_definitions.py (PHASE2.md deliverable #4).
//
// Implements the harness contract declared in
// solutions/ncnn/_harness/conv2d.h by delegating to ncnn::Convolution_arm
// (from libncnn_arm_heavy.a). Const params come from the Definition's
// const axes (out_c, kw, kh, sw, sh, dw, dh) and are passed in by
// armbench_entry_conv2d → convolution_kernel as runtime args, so this
// kernel.cpp is identical across all Definitions.
//
// create_pipeline is included in the timed path on purpose — matches
// today's baselines/ncnn.json semantics (run_conv2d_arm in
// starter/ncnn/baseline/convolution_arm.h calls it the same way).
// See PHASE2.md decision #4.

#include "conv2d.h"
#include "convolution_arm.h"

#include <cstring>

namespace ncnn {

int convolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                       const Mat& weight_data, const Mat& bias_data,
                       int kernel_w, int kernel_h,
                       int stride_w, int stride_h,
                       int dilation_w, int dilation_h,
                       int activation_type, const Mat& activation_params,
                       const Option& opt)
{
    Convolution_arm conv;
    conv.num_output       = top_blob.c;
    conv.kernel_w         = kernel_w;   conv.kernel_h   = kernel_h;
    conv.stride_w         = stride_w;   conv.stride_h   = stride_h;
    conv.dilation_w       = dilation_w; conv.dilation_h = dilation_h;
    // Harness already padded `bottom_blob` via copy_make_border, so disable
    // Convolution_arm's internal padding.
    conv.pad_left = 0; conv.pad_right  = 0;
    conv.pad_top  = 0; conv.pad_bottom = 0;
    conv.pad_value        = 0.f;
    conv.bias_term        = bias_data.empty() ? 0 : 1;
    conv.weight_data_size = weight_data.w;
    conv.int8_scale_term  = 0;
    conv.activation_type  = activation_type;
    conv.activation_params = const_cast<Mat&>(activation_params);
    conv.dynamic_weight   = 0;
    conv.weight_data      = const_cast<Mat&>(weight_data);
    if (!bias_data.empty()) conv.bias_data = const_cast<Mat&>(bias_data);

    if (conv.create_pipeline(opt) != 0) return -1;

    Mat local_top;
    int ret = conv.forward(bottom_blob, local_top, opt);
    if (ret != 0) return ret;
    if (local_top.empty()) return -1;

    // Copy into the caller-allocated top_blob (the harness pre-allocated it
    // to the expected output shape).
    if (local_top.c != top_blob.c || local_top.h != top_blob.h || local_top.w != top_blob.w)
        return -1;
    for (int cc = 0; cc < local_top.c; ++cc) {
        std::memcpy(top_blob.channel(cc), local_top.channel(cc),
                    local_top.h * local_top.w * sizeof(float));
    }
    return 0;
}

} // namespace ncnn
