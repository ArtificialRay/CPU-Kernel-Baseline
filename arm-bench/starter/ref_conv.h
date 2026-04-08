#include "test_utils.h"
#include "ncnn_helpers.h"

static TestMat ref_conv2d(const TestMat& in,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_c, int kh, int kw,
                           int stride_h, int stride_w,
                           int pad_top, int pad_left,
                           int dil_h = 1, int dil_w = 1)
{
    int in_c = in.c, in_h = in.h, in_w = in.w;
    int out_h = (in_h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_h, out_c);
    for (int oc = 0; oc < out_c; ++oc)
    for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int ih = oh * stride_h - pad_top  + khi * dil_h;
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                       ? in.at(iw, ih, ic) : 0.f;
            int widx = ((oc * in_c + ic) * kh + khi) * kw + kwi;
            sum += px * weight[widx];
        }
        out.at(ow, oh, oc) = sum;
    }
    return out;
}

// Depthwise 2-D convolution reference (group == channels)
static TestMat ref_depthwise_conv2d(const TestMat& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int kh, int kw,
                                     int stride_h, int stride_w,
                                     int pad_top, int pad_left,
                                     int dil_h = 1, int dil_w = 1)
{
    int c = in.c;
    int out_h = (in.h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in.w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_h, c);
    for (int ch = 0; ch < c; ++ch)
    for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[ch];
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int ih = oh * stride_h - pad_top  + khi * dil_h;
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (ih >= 0 && ih < in.h && iw >= 0 && iw < in.w)
                       ? in.at(iw, ih, ch) : 0.f;
            sum += px * weight[(ch * kh + khi) * kw + kwi];
        }
        out.at(ow, oh, ch) = sum;
    }
    return out;
}

// 2-D transposed convolution (deconvolution) reference
// Weight layout: [out_c, in_c, kh, kw] (ncnn layout: per-outch block of in_c*kh*kw)
static TestMat ref_deconv2d(const TestMat& in,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              int in_c, int out_c,
                              int kh, int kw,
                              int stride_h, int stride_w,
                              int dil_h = 1, int dil_w = 1)
{
    int w = in.w, h = in.h;
    int ke_h = dil_h * (kh - 1) + 1;
    int ke_w = dil_w * (kw - 1) + 1;
    int out_h = (h - 1) * stride_h + ke_h;
    int out_w = (w - 1) * stride_w + ke_w;
    TestMat out(out_w, out_h, out_c, std::vector<float>(out_w * out_h * out_c, 0.f));
    // fill bias
    if (!bias.empty())
        for (int oc = 0; oc < out_c; ++oc)
            for (int i = 0; i < out_h * out_w; ++i)
                out.at(i % out_w, i / out_w, oc) = bias[oc];

    for (int oc = 0; oc < out_c; ++oc)
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
        for (int ic = 0; ic < in_c; ++ic) {
            float val = in.at(j, i, ic);
            for (int ki = 0; ki < kh; ++ki)
            for (int kj = 0; kj < kw; ++kj) {
                int oy = i * stride_h + ki * dil_h;
                int ox = j * stride_w + kj * dil_w;
                // Weight layout: [out_c, in_c, kh, kw] matching ncnn deconv forward
                float wt = weight[((oc * in_c + ic) * kh + ki) * kw + kj];
                out.at(ox, oy, oc) += val * wt;
            }
        }
    }
    return out;
}

// Depthwise 2-D transposed convolution reference (group == channels)
static TestMat ref_depthwise_deconv2d(const TestMat& in,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       int kh, int kw,
                                       int stride_h, int stride_w,
                                       int dil_h = 1, int dil_w = 1)
{
    int c = in.c, h = in.h, w = in.w;
    int ke_h = dil_h * (kh - 1) + 1;
    int ke_w = dil_w * (kw - 1) + 1;
    int out_h = (h - 1) * stride_h + ke_h;
    int out_w = (w - 1) * stride_w + ke_w;
    TestMat out(out_w, out_h, c, std::vector<float>(out_w * out_h * c, 0.f));
    if (!bias.empty())
        for (int ch = 0; ch < c; ++ch)
            for (int i = 0; i < out_h * out_w; ++i)
                out.at(i % out_w, i / out_w, ch) = bias[ch];

    for (int ch = 0; ch < c; ++ch)
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
        float val = in.at(j, i, ch);
        for (int ki = 0; ki < kh; ++ki)
        for (int kj = 0; kj < kw; ++kj) {
            int oy = i * stride_h + ki * dil_h;
            int ox = j * stride_w + kj * dil_w;
            out.at(ox, oy, ch) += val * weight[(ch * kh + ki) * kw + kj];
        }
    }
    return out;
}

// 1-D convolution reference
static TestMat ref_conv1d(const TestMat& in,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_c, int kw,
                           int stride_w, int pad_left,
                           int dil_w = 1)
{
    int in_c = in.h;   // Convolution1D: h=channels, w=sequence_length
    int in_w = in.w;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_c, 1);
    for (int oc = 0; oc < out_c; ++oc)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (iw >= 0 && iw < in_w) ? in.at(iw, ic) : 0.f;
            sum += px * weight[(oc * in_c + ic) * kw + kwi];
        }
        out.at(ow, oc) = sum;
    }
    return out;
}
