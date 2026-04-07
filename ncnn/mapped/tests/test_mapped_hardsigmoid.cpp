// test_mapped_hardsigmoid.cpp — base vs ARM comparison for HardSigmoid
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/hardsigmoid/hardsigmoid.h"
#include "../mapped/hardsigmoid/hardsigmoid_arm.h"

static bool run_hardsigmoid(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::HardSigmoid base;
    base.alpha = 0.2f; base.beta = 0.5f; base.lower = -2.5f; base.upper = 5.0f;
    ncnn::HardSigmoid_arm arm;
    arm.alpha = 0.2f; arm.beta = 0.5f; arm.lower = -2.5f; arm.upper = 5.0f;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_hardsigmoid()
{
    run_hardsigmoid(1, 1, 16);
    run_hardsigmoid(3, 4, 8);
    run_hardsigmoid(4, 5, 7);
}

int main()
{
    RUN_TEST(test_hardsigmoid);
    print_summary("test_mapped_hardsigmoid");
    return g_failed ? 1 : 0;
}
