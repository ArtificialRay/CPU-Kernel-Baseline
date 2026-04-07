// test_mapped_softmax.cpp — base vs ARM comparison for Softmax
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/softmax/softmax.h"
#include "../mapped/softmax/softmax_arm.h"

static bool run_softmax(int n)
{
    std::vector<float> data = make_weights(n);

    ncnn::Mat m1 = make_mat_1d(data);
    ncnn::Mat m2 = make_mat_1d(data);

    ncnn::Softmax base; base.axis = 0;
    ncnn::Softmax_arm arm; arm.axis = 0;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_softmax()
{
    run_softmax(8);
    run_softmax(16);
    run_softmax(32);
}

int main()
{
    RUN_TEST(test_softmax);
    print_summary("test_mapped_softmax");
    return g_failed ? 1 : 0;
}
