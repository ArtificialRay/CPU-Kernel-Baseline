// test_unmapped_pooling1d.cpp — reference vs ncnn for Pooling1D (global)
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../unmapped/pooling1d/pooling1d.h"
#include <algorithm>
#include <vector>

static bool run_pooling1d(int rows, int cols, int ptype)
{
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; i++) data[i] = (float)(i + 1) * 0.1f;

    ncnn::Mat in = make_mat_2d(cols, rows, data);
    ncnn::Mat out;

    ncnn::Pooling1D pool;
    pool.pooling_type = ptype;
    pool.global_pooling = 1;
    pool.adaptive_pooling = 0;
    pool.pad_left = 0;
    pool.pad_right = 0;

    ncnn::Option opt = make_opt();
    if (pool.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref(rows);
    for (int r = 0; r < rows; r++) {
        const float* row = data.data() + r * cols;
        if (ptype == ncnn::Pooling1D::PoolMethod_MAX) {
            ref[r] = *std::max_element(row, row + cols);
        } else {
            float s = 0.f;
            for (int c = 0; c < cols; c++) s += row[c];
            ref[r] = s / cols;
        }
    }

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-5f);
    return g_failed == before;
}

static void test_pooling1d()
{
    run_pooling1d(4, 8, ncnn::Pooling1D::PoolMethod_MAX);
    run_pooling1d(4, 8, ncnn::Pooling1D::PoolMethod_AVE);
    run_pooling1d(3, 6, ncnn::Pooling1D::PoolMethod_MAX);
    run_pooling1d(3, 6, ncnn::Pooling1D::PoolMethod_AVE);
}

int main()
{
    RUN_TEST(test_pooling1d);
    print_summary("test_unmapped_pooling1d");
    return g_failed ? 1 : 0;
}
