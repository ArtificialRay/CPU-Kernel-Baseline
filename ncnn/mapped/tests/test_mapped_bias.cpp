// test_mapped_bias.cpp — base vs ARM comparison for Bias
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/bias/bias.h"
#include "../mapped/bias/bias_arm.h"

static bool run_bias(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);
    std::vector<float> biases(c);
    for (int i = 0; i < c; i++) biases[i] = 0.1f * (i + 1);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::Bias base;
    base.bias_data_size = c;
    base.bias_data = make_weight(biases);

    ncnn::Bias_arm arm;
    arm.bias_data_size = c;
    arm.bias_data = make_weight(biases);

    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_bias()
{
    run_bias(1, 1, 16);
    run_bias(3, 4, 8);
    run_bias(4, 5, 7);
}

int main()
{
    RUN_TEST(test_bias);
    print_summary("test_mapped_bias");
    return g_failed ? 1 : 0;
}
