// test_mapped_batchnorm.cpp — base vs ARM comparison for BatchNorm
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/batchnorm/batchnorm.h"
#include "../mapped/batchnorm/batchnorm_arm.h"

static bool run_batchnorm(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::BatchNorm base;
    base.channels = c;
    base.a_data.create(c, 4u, (ncnn::Allocator*)0);
    base.b_data.create(c, 4u, (ncnn::Allocator*)0);
    for (int i = 0; i < c; i++) {
        base.a_data[i] = 0.1f * i;
        base.b_data[i] = 0.5f + 0.1f * i;
    }

    ncnn::BatchNorm_arm arm;
    arm.channels = c;
    arm.a_data = base.a_data;
    arm.b_data = base.b_data;

    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_batchnorm()
{
    run_batchnorm(1, 1, 16);
    run_batchnorm(3, 4, 8);
    run_batchnorm(4, 5, 7);
}

int main()
{
    RUN_TEST(test_batchnorm);
    print_summary("test_mapped_batchnorm");
    return g_failed ? 1 : 0;
}
