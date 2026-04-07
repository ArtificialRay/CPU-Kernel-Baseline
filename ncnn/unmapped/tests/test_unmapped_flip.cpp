// test_unmapped_flip.cpp — reference vs ncnn for Flip
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../unmapped/flip/flip.h"
#include <vector>

static ncnn::Mat make_int_mat(const std::vector<int>& vals)
{
    ncnn::Mat m;
    m.create((int)vals.size(), (size_t)4u, (ncnn::Allocator*)0);
    int* ptr = (int*)((float*)m);
    for (int i = 0; i < (int)vals.size(); i++) ptr[i] = vals[i];
    return m;
}

static bool run_flip_w(int h, int w)
{
    std::vector<float> data(h * w);
    for (int i = 0; i < h * w; i++) data[i] = (float)(i + 1);

    ncnn::Mat in = make_mat_2d(w, h, data);
    ncnn::Mat out;

    ncnn::Flip flip;
    flip.axes = make_int_mat({1});

    ncnn::Option opt = make_opt();
    if (flip.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref(h * w);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            ref[r * w + c] = data[r * w + (w - 1 - c)];

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-6f);
    return g_failed == before;
}

static bool run_flip_h(int h, int w)
{
    std::vector<float> data(h * w);
    for (int i = 0; i < h * w; i++) data[i] = (float)(i + 1);

    ncnn::Mat in = make_mat_2d(w, h, data);
    ncnn::Mat out;

    ncnn::Flip flip;
    flip.axes = make_int_mat({0});

    ncnn::Option opt = make_opt();
    if (flip.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref(h * w);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            ref[r * w + c] = data[(h - 1 - r) * w + c];

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-6f);
    return g_failed == before;
}

static bool run_flip_3d_w(int c, int h, int w)
{
    std::vector<float> data(c * h * w);
    for (int i = 0; i < c * h * w; i++) data[i] = (float)(i + 1);

    ncnn::Mat in = make_mat(w, h, c, data);
    ncnn::Mat out;

    ncnn::Flip flip;
    flip.axes = make_int_mat({2});

    ncnn::Option opt = make_opt();
    if (flip.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref(c * h * w);
    for (int ch = 0; ch < c; ch++)
        for (int r = 0; r < h; r++)
            for (int col = 0; col < w; col++)
                ref[ch * h * w + r * w + col] = data[ch * h * w + r * w + (w - 1 - col)];

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-6f);
    return g_failed == before;
}

static void test_flip()
{
    run_flip_w(4, 8);
    run_flip_w(3, 6);
    run_flip_h(4, 8);
    run_flip_3d_w(2, 3, 5);
}

int main()
{
    RUN_TEST(test_flip);
    print_summary("test_unmapped_flip");
    return g_failed ? 1 : 0;
}
