#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../ncnn/mapped/convolution1d/convolution1d.h"
#include "../ncnn/mapped/convolution1d/convolution1d_arm.h"
// CANDIDATE_INJECT_START
// The ncnn Convolution1D treats the input as [w=length, h=channels] (2D mat)
static bool run_conv1d(int in_c, int out_c, int in_w, int kw,
                        int stride_w, int pad_left, int dil_w = 1,
                        bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    // Convolution1D input layout: w=length, h=channels (2D mat, no c dim)
    std::vector<float> in_flat(in_c * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;
    ncnn::Mat bottom = make_mat_2d(in_w, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Convolution1D conv1d;
    conv1d.num_output       = out_c;
    conv1d.kernel_w         = kw;
    conv1d.dilation_w       = dil_w;
    conv1d.stride_w         = stride_w;
    conv1d.pad_left         = pad_left; conv1d.pad_right = pad_left;
    conv1d.pad_value        = 0.f;
    conv1d.bias_term        = with_bias ? 1 : 0;
    conv1d.weight_data_size = wsize;
    conv1d.activation_type  = 0;
    conv1d.dynamic_weight   = 0;
    conv1d.weight_data      = make_weight(weight);
    if (with_bias) conv1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = conv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution1D::forward failed %d\n", ret); g_failed++; return false; }

    // Reference uses TestMat (h=in_c, w=in_w)
    TestMat in_tm(in_w, in_c, 1, in_flat);

    // Output: h=out_c, w=out_len (2D mat)
    std::vector<float> got;
    if (top.dims == 2) {
        int out_len = top.w;
        got.resize(out_c * out_len);
        for (int oc = 0; oc < out_c; ++oc)
            memcpy(got.data() + oc * out_len, top.row(oc), out_len * sizeof(float));
    } else {
        read_mat(top, got);
    }

}
// CANDIDATE_INJECT_END

// Convolution1D_arm
static bool run_conv1d_arm(int in_c, int out_c, int in_w, int kw,
                             int stride_w, int pad_left, int dil_w = 1,
                             bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;
    ncnn::Mat bottom = make_mat_2d(in_w, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Convolution1D_arm conv1d;
    conv1d.num_output       = out_c;
    conv1d.kernel_w         = kw;
    conv1d.dilation_w       = dil_w;
    conv1d.stride_w         = stride_w;
    conv1d.pad_left         = pad_left; conv1d.pad_right = pad_left;
    conv1d.pad_value        = 0.f;
    conv1d.bias_term        = with_bias ? 1 : 0;
    conv1d.weight_data_size = wsize;
    conv1d.activation_type  = 0;
    conv1d.dynamic_weight   = 0;
    conv1d.weight_data      = make_weight(weight);
    if (with_bias) conv1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (conv1d.create_pipeline(opt) != 0) {
        fprintf(stderr, "  Convolution1D_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = conv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution1D_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat in_tm(in_w, in_c, 1, in_flat);

    std::vector<float> got;
    if (top.dims == 2) {
        int out_len = top.w;
        got.resize(out_c * out_len);
        for (int oc = 0; oc < out_c; ++oc)
            memcpy(got.data() + oc * out_len, top.row(oc), out_len * sizeof(float));
    } else {
        read_mat(top, got);
    }

}