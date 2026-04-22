#pragma once
#include "ncnn_helpers.h"

// Reference implementations. Inputs/outputs are ncnn::Mat so starter headers
// don't need to pull in the test-side TestMat / test_utils.h.

// 2-D convolution reference. Input in: 3D Mat (w, h, c).
[[maybe_unused]] static ncnn::Mat ref_conv2d(const ncnn::Mat& in,
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
    ncnn::Mat out;
    out.create(out_w, out_h, out_c, 4u, (ncnn::Allocator*)0);
    for (int oc = 0; oc < out_c; ++oc) {
        float* outptr = out.channel(oc);
        for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow) {
            float sum = bias.empty() ? 0.f : bias[oc];
            for (int ic = 0; ic < in_c; ++ic) {
                const float* inptr_c = in.channel(ic);
                for (int khi = 0; khi < kh; ++khi)
                for (int kwi = 0; kwi < kw; ++kwi) {
                    int ih = oh * stride_h - pad_top  + khi * dil_h;
                    int iw = ow * stride_w - pad_left + kwi * dil_w;
                    float px = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                               ? inptr_c[ih * in_w + iw] : 0.f;
                    int widx = ((oc * in_c + ic) * kh + khi) * kw + kwi;
                    sum += px * weight[widx];
                }
            }
            outptr[oh * out_w + ow] = sum;
        }
    }
    return out;
}

// Depthwise 2-D convolution reference (group == channels)
[[maybe_unused]] static ncnn::Mat ref_depthwise_conv2d(const ncnn::Mat& in,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       int kh, int kw,
                                       int stride_h, int stride_w,
                                       int pad_top, int pad_left,
                                       int dil_h = 1, int dil_w = 1)
{
    int c = in.c, in_h = in.h, in_w = in.w;
    int out_h = (in_h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    ncnn::Mat out;
    out.create(out_w, out_h, c, 4u, (ncnn::Allocator*)0);
    for (int ch = 0; ch < c; ++ch) {
        const float* inptr = in.channel(ch);
        float* outptr = out.channel(ch);
        for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow) {
            float sum = bias.empty() ? 0.f : bias[ch];
            for (int khi = 0; khi < kh; ++khi)
            for (int kwi = 0; kwi < kw; ++kwi) {
                int ih = oh * stride_h - pad_top  + khi * dil_h;
                int iw = ow * stride_w - pad_left + kwi * dil_w;
                float px = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                           ? inptr[ih * in_w + iw] : 0.f;
                sum += px * weight[(ch * kh + khi) * kw + kwi];
            }
            outptr[oh * out_w + ow] = sum;
        }
    }
    return out;
}

// 2-D transposed convolution (deconvolution) reference
// Weight layout: [out_c, in_c, kh, kw] (ncnn layout: per-outch block of in_c*kh*kw)
[[maybe_unused]] static ncnn::Mat ref_deconv2d(const ncnn::Mat& in,
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
    ncnn::Mat out;
    out.create(out_w, out_h, out_c, 4u, (ncnn::Allocator*)0);
    // fill bias or zero
    for (int oc = 0; oc < out_c; ++oc) {
        float* outptr = out.channel(oc);
        float b = bias.empty() ? 0.f : bias[oc];
        for (int i = 0; i < out_h * out_w; ++i) outptr[i] = b;
    }

    for (int oc = 0; oc < out_c; ++oc) {
        float* outptr = out.channel(oc);
        for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) {
            for (int ic = 0; ic < in_c; ++ic) {
                const float* inptr_c = in.channel(ic);
                float val = inptr_c[i * w + j];
                for (int ki = 0; ki < kh; ++ki)
                for (int kj = 0; kj < kw; ++kj) {
                    int oy = i * stride_h + ki * dil_h;
                    int ox = j * stride_w + kj * dil_w;
                    float wt = weight[((oc * in_c + ic) * kh + ki) * kw + kj];
                    outptr[oy * out_w + ox] += val * wt;
                }
            }
        }
    }
    return out;
}

// Depthwise 2-D transposed convolution reference (group == channels)
[[maybe_unused]] static ncnn::Mat ref_depthwise_deconv2d(const ncnn::Mat& in,
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
    ncnn::Mat out;
    out.create(out_w, out_h, c, 4u, (ncnn::Allocator*)0);
    for (int ch = 0; ch < c; ++ch) {
        float* outptr = out.channel(ch);
        float b = bias.empty() ? 0.f : bias[ch];
        for (int i = 0; i < out_h * out_w; ++i) outptr[i] = b;
    }

    for (int ch = 0; ch < c; ++ch) {
        const float* inptr = in.channel(ch);
        float* outptr = out.channel(ch);
        for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) {
            float val = inptr[i * w + j];
            for (int ki = 0; ki < kh; ++ki)
            for (int kj = 0; kj < kw; ++kj) {
                int oy = i * stride_h + ki * dil_h;
                int ox = j * stride_w + kj * dil_w;
                outptr[oy * out_w + ox] += val * weight[(ch * kh + ki) * kw + kj];
            }
        }
    }
    return out;
}

// 1-D convolution reference. Input in: 2D Mat (w=seq_len, h=channels).
[[maybe_unused]] static ncnn::Mat ref_conv1d(const ncnn::Mat& in,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             int out_c, int kw,
                             int stride_w, int pad_left,
                             int dil_w = 1)
{
    int in_c = in.h;   // Convolution1D: h=channels, w=sequence_length
    int in_w = in.w;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    ncnn::Mat out;
    out.create(out_w, out_c, 4u, (ncnn::Allocator*)0);
    for (int oc = 0; oc < out_c; ++oc) {
        float* outptr = out.row(oc);
        for (int ow = 0; ow < out_w; ++ow) {
            float sum = bias.empty() ? 0.f : bias[oc];
            for (int ic = 0; ic < in_c; ++ic) {
                const float* inrow = in.row(ic);
                for (int kwi = 0; kwi < kw; ++kwi) {
                    int iw = ow * stride_w - pad_left + kwi * dil_w;
                    float px = (iw >= 0 && iw < in_w) ? inrow[iw] : 0.f;
                    sum += px * weight[(oc * in_c + ic) * kw + kwi];
                }
            }
            outptr[ow] = sum;
        }
    }
    return out;
}