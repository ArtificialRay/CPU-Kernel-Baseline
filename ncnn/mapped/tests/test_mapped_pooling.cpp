// test_mapped_pooling.cpp — base vs ARM comparison for Pooling (global)
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/pooling/pooling.h"
#include "../mapped/pooling/pooling_arm.h"

static bool run_pooling(int c, int h, int w, int ptype)
{
    std::vector<float> data(c * h * w);
    for (int i = 0; i < c * h * w; i++) data[i] = (float)(i + 1) * 0.1f;

    ncnn::Mat in1 = make_mat(w, h, c, data);
    ncnn::Mat in2 = make_mat(w, h, c, data);
    ncnn::Mat out1, out2;

    ncnn::Pooling base;
    base.pooling_type = ptype;
    base.global_pooling = 1;
    base.adaptive_pooling = 0;

    ncnn::Pooling_arm arm;
    arm.pooling_type = ptype;
    arm.global_pooling = 1;
    arm.adaptive_pooling = 0;

    ncnn::Option opt = make_opt();
    arm.create_pipeline(opt);

    if (base.forward(in1, out1, opt) != 0 || arm.forward(in2, out2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(out1, o1); read_mat(out2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_pooling()
{
    run_pooling(3, 4, 8, ncnn::Pooling::PoolMethod_MAX);
    run_pooling(3, 4, 8, ncnn::Pooling::PoolMethod_AVE);
    run_pooling(4, 5, 7, ncnn::Pooling::PoolMethod_MAX);
}

int main()
{
    RUN_TEST(test_pooling);
    print_summary("test_mapped_pooling");
    return g_failed ? 1 : 0;
}
