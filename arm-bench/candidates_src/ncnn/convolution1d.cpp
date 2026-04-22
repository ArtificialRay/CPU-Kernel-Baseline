#include "starter/ncnn/candidate/convolution1d.h"

#include "common/fused_activation.h"

#include <vector>

namespace ncnn {

int convolution1d_kernel(const Mat& bottom_blob, Mat& top_blob,
                         const Mat& weight_data, const Mat& bias_data,
                         int kernel_w, int stride_w, int dilation_w,
                         int activation_type, const Mat& activation_params,
                         const Option& opt)
{
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float* sptr = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val = *sptr;
                    float wt = kptr[k];
                    sum += val * wt;

                    sptr += dilation_w;
                }

                kptr += kernel_w;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[j] = sum;
        }
    }

    return 0;
}

} // namespace ncnn
