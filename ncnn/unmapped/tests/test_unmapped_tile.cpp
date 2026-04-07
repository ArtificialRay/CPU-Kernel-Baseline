// test_unmapped_tile.cpp — reference vs ncnn for Tile
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../unmapped/tile/tile.h"
#include <vector>

static bool run_tile_1d(int n, int tiles)
{
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) data[i] = (float)(i + 1);

    ncnn::Mat in = make_mat_1d(data);
    ncnn::Mat out;

    ncnn::Tile tile;
    tile.axis = 0;
    tile.tiles = tiles;

    ncnn::Option opt = make_opt();
    if (tile.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref;
    for (int t = 0; t < tiles; t++)
        for (int i = 0; i < n; i++)
            ref.push_back(data[i]);

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-6f);
    return g_failed == before;
}

static bool run_tile_3d_c(int c, int h, int w, int tiles)
{
    std::vector<float> data(c * h * w);
    for (int i = 0; i < c * h * w; i++) data[i] = (float)(i + 1);

    ncnn::Mat in = make_mat(w, h, c, data);
    ncnn::Mat out;

    ncnn::Tile tile;
    tile.axis = 0;
    tile.tiles = tiles;

    ncnn::Option opt = make_opt();
    if (tile.forward(in, out, opt) != 0) { g_failed++; return false; }

    std::vector<float> ref;
    for (int t = 0; t < tiles; t++)
        for (int i = 0; i < c * h * w; i++)
            ref.push_back(data[i]);

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-6f);
    return g_failed == before;
}

static void test_tile()
{
    run_tile_1d(8, 2);
    run_tile_1d(4, 3);
    run_tile_3d_c(2, 3, 4, 2);
    run_tile_3d_c(1, 2, 5, 3);
}

int main()
{
    RUN_TEST(test_tile);
    print_summary("test_unmapped_tile");
    return g_failed ? 1 : 0;
}
