// test_mapped_layernorm.cpp — base vs ARM comparison for LayerNorm
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/layernorm/layernorm.h"
#include "../mapped/layernorm/layernorm_arm.h"

static bool run_layernorm(int w)
{
    std::vector<float> data = make_weights(w);

    ncnn::Mat m1 = make_mat_1d(data);
    ncnn::Mat m2 = make_mat_1d(data);

    std::vector<float> gamma(w), beta(w);
    for (int i = 0; i < w; i++) {
        gamma[i] = 0.5f + 0.1f * i;
        beta[i]  = 0.01f * i;
    }

    ncnn::LayerNorm base;
    base.affine_size = w;
    base.eps = 1e-5f;
    base.affine = 1;
    base.gamma_data = make_weight(gamma);
    base.beta_data  = make_weight(beta);

    ncnn::LayerNorm_arm arm;
    arm.affine_size = w;
    arm.eps = 1e-5f;
    arm.affine = 1;
    arm.gamma_data = make_weight(gamma);
    arm.beta_data  = make_weight(beta);

    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_layernorm()
{
    run_layernorm(16);
    run_layernorm(32);
    run_layernorm(8);
}

int main()
{
    RUN_TEST(test_layernorm);
    print_summary("test_mapped_layernorm");
    return g_failed ? 1 : 0;
}
