// test_mapped_clip.cpp — base vs ARM comparison for Clip
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/clip/clip.h"
#include "../mapped/clip/clip_arm.h"

static bool run_clip(int c, int h, int w, float mn, float mx)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::Clip base; base.min = mn; base.max = mx;
    ncnn::Clip_arm arm; arm.min = mn; arm.max = mx;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_clip()
{
    run_clip(1, 1, 16, -0.3f, 0.3f);
    run_clip(3, 4, 8, -0.2f, 0.4f);
    run_clip(4, 5, 7, 0.f, 0.5f);
}

int main()
{
    RUN_TEST(test_clip);
    print_summary("test_mapped_clip");
    return g_failed ? 1 : 0;
}
