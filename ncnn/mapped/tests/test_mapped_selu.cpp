// test_mapped_selu.cpp — base vs ARM comparison for SELU
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/selu/selu.h"
#include "../mapped/selu/selu_arm.h"

static bool run_selu(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);
    for (auto& v : data) v -= 0.3f;

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::SELU base;
    base.alpha = 1.67326324f; base.lambda = 1.050700987f;
    ncnn::SELU_arm arm;
    arm.alpha = 1.67326324f; arm.lambda = 1.050700987f;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_selu()
{
    run_selu(1, 1, 16);
    run_selu(3, 4, 8);
    run_selu(4, 5, 7);
}

int main()
{
    RUN_TEST(test_selu);
    print_summary("test_mapped_selu");
    return g_failed ? 1 : 0;
}
