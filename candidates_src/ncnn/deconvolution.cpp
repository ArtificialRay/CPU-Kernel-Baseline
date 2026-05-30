#include "starter/ncnn/candidate/deconvolution.h"

#include "common/fused_activation.h"

#include <vector>

namespace ncnn {

int deconvolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                         const Mat& weight_data, const Mat& bias_data,
                         int kernel_w, int kernel_h,
                         int stride_w, int stride_h,
                         int dilation_w, int dilation_h,
                         int activation_type, const Mat& activation_params,
                         const Option& opt)
{
    const int outw = top_blob.w;
    const int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation_h - kernel_w * dilation_w;
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
        Mat out = top_blob.channel(p);

        const float bias = bias_data.empty() ? 0.f : bias_data[p];

        out.fill(bias);

        // shadowed variable for less openmp task args
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int inch = bottom_blob.c;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float* outptr = out.row(i * stride_h) + j * stride_w;

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const float val = bottom_blob.channel(q).row(i)[j];

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
                    }

                    kptr += maxk;
                }
            }
        }

        {
            float* outptr = out;
            int size = outw * outh;

            for (int i = 0; i < size; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}

} // namespace ncnn
