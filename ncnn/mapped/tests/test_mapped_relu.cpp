// test_mapped_relu.cpp — base vs ARM comparison for ReLU
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/relu/relu.h"
#include "../mapped/relu/relu_arm.h"

static bool run_relu(int c, int h, int w, float slope = 0.f)
{
    std::vector<float> data = make_weights(c * h * w);
    for (auto& v : data) v -= 0.3f;

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::ReLU base; base.slope = slope;
    ncnn::ReLU_arm arm; arm.slope = slope;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_relu()
{
    run_relu(1, 1, 16);
    run_relu(3, 4, 8, 0.f);
    run_relu(4, 5, 7, 0.1f);
}

int main()
{
    RUN_TEST(test_relu);
    print_summary("test_mapped_relu");
    return g_failed ? 1 : 0;
}
