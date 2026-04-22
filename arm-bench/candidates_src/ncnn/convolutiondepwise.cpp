#include "starter/ncnn/candidate/convolutiondepthwise.h"

#include "common/fused_activation.h"

#include <vector>

namespace ncnn {

int convolutiondepthwise_kernel(const Mat& bottom_blob, Mat& top_blob,
                                const Mat& weight_data, const Mat& bias_data,
                                int kernel_w, int kernel_h,
                                int stride_w, int stride_h,
                                int dilation_w, int dilation_h,
                                int group,
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

    // depth-wise
    if (inch == group && group == outch)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = activation_ss(sum, activation_type, activation_params);
                }

                outptr += outw;
            }
        }
    }
    else
    {
        // group convolution
        const int inch_g = inch / group;
        const int outch_g = outch / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outch_g; p++)
            {
                float* outptr = top_blob.channel(g * outch_g + p);
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                // shadowed variable for less openmp task args
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[outch_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const Mat m = bottom_blob.channel(inch_g * g + q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        outptr[j] = activation_ss(sum, activation_type, activation_params);
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
