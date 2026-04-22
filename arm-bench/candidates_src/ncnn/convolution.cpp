#include "starter/ncnn/candidate/convolution.h"

#include "common/fused_activation.h"

#include <vector>

namespace ncnn {

int convolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                       const Mat& weight_data, const Mat& bias_data,
                       int kernel_w, int kernel_h,
                       int stride_w, int stride_h,
                       int dilation_w, int dilation_h,
                       int activation_type, const Mat& activation_params,
                       const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float wt = kptr[k];
                        sum += val * wt;
                    }

                    kptr += maxk;
                }

                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }

    return 0;
}

} // namespace ncnn
