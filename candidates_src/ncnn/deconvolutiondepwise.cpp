#include "starter/ncnn/candidate/deconvolutiondepthwise.h"

#include "common/fused_activation.h"

#include <vector>

namespace ncnn {

int deconvolutiondepthwise_kernel(const Mat& bottom_blob, Mat& top_blob,
                                  const Mat& weight_data, const Mat& bias_data,
                                  int kernel_w, int kernel_h,
                                  int stride_w, int stride_h,
                                  int dilation_w, int dilation_h,
                                  int group,
                                  int activation_type, const Mat& activation_params,
                                  const Option& opt)
{
    const int inch = bottom_blob.c;

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

    // depth-wise
    if (inch == group && group == outch)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            Mat out = top_blob.channel(g);

            const float bias = bias_data.empty() ? 0.f : bias_data[g];

            out.fill(bias);

            // shadowed variable for less openmp task args
            const int w = bottom_blob.w;
            const int h = bottom_blob.h;
            const int outw = top_blob.w;
            const int outh = top_blob.h;

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float* outptr = out.row(i * stride_h) + j * stride_w;

                    const float val = inptr[i * w + j];

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
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
    }
    else
    {
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
                Mat out = top_blob.channel(g * outch_g + p);

                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                const float bias = bias_data.empty() ? 0.f : bias_data[g * outch_g + p];

                out.fill(bias);

                // shadowed variable for less openmp task args
                const int w = bottom_blob.w;
                const int h = bottom_blob.h;
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.row(i * stride_h) + j * stride_w;

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const float val = bottom_blob.channel(inch_g * g + q).row(i)[j];

                            for (int k = 0; k < maxk; k++)
                            {
                                outptr[space_ofs[k]] += val * kptr[k];
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
        }
    }

    return 0;
}

} // namespace ncnn
