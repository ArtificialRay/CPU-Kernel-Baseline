// test_mapped_scale.cpp — Scale base vs reference (Scale_arm only overrides two-blob variant)
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/scale/scale.h"

static bool run_scale(int c, int h, int w)
{
    std::vector<float> data(c * h * w);
    for (int i = 0; i < c * h * w; i++) data[i] = (float)(i + 1) * 0.1f;

    std::vector<float> scales(c);
    for (int i = 0; i < c; i++) scales[i] = 0.5f + 0.1f * i;

    ncnn::Mat m = make_mat(w, h, c, data);

    ncnn::Scale op;
    op.scale_data_size = c;
    op.bias_term = 0;
    op.scale_data = make_weight(scales);

    ncnn::Option opt = make_opt();
    if (op.forward_inplace(m, opt) != 0) { g_failed++; return false; }

    // reference: out[q][i] = in[q][i] * scale[q]
    std::vector<float> ref(c * h * w);
    for (int q = 0; q < c; q++)
        for (int i = 0; i < h * w; i++)
            ref[q * h * w + i] = data[q * h * w + i] * scales[q];

    std::vector<float> out;
    read_mat(m, out);
    int before = g_failed;
    ASSERT_VEC_NEAR(out, ref.data(), (int)out.size(), 1e-5f);
    return g_failed == before;
}

static void test_scale()
{
    run_scale(1, 1, 16);
    run_scale(3, 4, 8);
    run_scale(4, 5, 7);
}

int main()
{
    RUN_TEST(test_scale);
    print_summary("test_mapped_scale");
    return g_failed ? 1 : 0;
}
