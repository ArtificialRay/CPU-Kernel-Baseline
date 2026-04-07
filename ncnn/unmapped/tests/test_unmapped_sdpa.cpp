// test_unmapped_sdpa.cpp — reference vs ncnn for SDPA
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../unmapped/sdpa/sdpa.h"
#include <cmath>
#include <vector>

static void ref_sdpa(const float* Q, const float* K, const float* V,
                     float* out, int heads, int src_len, int cur_len,
                     int embed_dim, int v_dim, float scale)
{
    std::vector<float> scores(src_len * cur_len);
    for (int h = 0; h < heads; h++) {
        const float* q = Q + h * src_len * embed_dim;
        const float* k = K + h * cur_len * embed_dim;
        const float* v = V + h * cur_len * v_dim;
        float* o = out + h * src_len * v_dim;

        for (int i = 0; i < src_len; i++)
            for (int j = 0; j < cur_len; j++) {
                float s = 0.f;
                for (int d = 0; d < embed_dim; d++)
                    s += q[i * embed_dim + d] * k[j * embed_dim + d];
                scores[i * cur_len + j] = s * scale;
            }
        for (int i = 0; i < src_len; i++)
            softmax_inplace(scores.data() + i * cur_len, cur_len);
        for (int i = 0; i < src_len; i++)
            for (int j = 0; j < v_dim; j++) {
                float sum = 0.f;
                for (int k2 = 0; k2 < cur_len; k2++)
                    sum += scores[i * cur_len + k2] * v[k2 * v_dim + j];
                o[i * v_dim + j] = sum;
            }
    }
}

static bool run_sdpa(int heads, int src_len, int cur_len, int embed_dim, int v_dim)
{
    int q_size = heads * src_len * embed_dim;
    int k_size = heads * cur_len * embed_dim;
    int v_size = heads * cur_len * v_dim;

    std::vector<float> qdata = make_weights(q_size);
    std::vector<float> kdata = make_weights(k_size);
    std::vector<float> vdata = make_weights(v_size);
    for (auto& x : qdata) x *= 0.5f;
    for (auto& x : kdata) x *= 0.5f;
    for (auto& x : vdata) x *= 0.5f;

    ncnn::Mat query = make_mat(embed_dim, src_len, heads, qdata);
    ncnn::Mat key   = make_mat(embed_dim, cur_len, heads, kdata);
    ncnn::Mat value = make_mat(v_dim,     cur_len, heads, vdata);

    ncnn::SDPA sdpa;
    sdpa.attn_mask = 0;
    sdpa.scale = 0.f;
    sdpa.kv_cache = 0;
    sdpa.int8_scale_term = 0;

    ncnn::Option opt = make_opt();
    std::vector<ncnn::Mat> inputs = {query, key, value};
    std::vector<ncnn::Mat> outputs(1);

    if (sdpa.forward(inputs, outputs, opt) != 0) { g_failed++; return false; }

    float scale = 1.f / sqrtf((float)embed_dim);
    std::vector<float> ref(heads * src_len * v_dim);
    ref_sdpa(qdata.data(), kdata.data(), vdata.data(),
             ref.data(), heads, src_len, cur_len, embed_dim, v_dim, scale);

    std::vector<float> got;
    read_mat(outputs[0], got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)got.size(), 1e-4f);
    return g_failed == before;
}

static void test_sdpa()
{
    run_sdpa(1, 4, 4, 8, 8);
    run_sdpa(2, 3, 5, 4, 4);
    run_sdpa(1, 2, 3, 6, 6);
}

int main()
{
    RUN_TEST(test_sdpa);
    print_summary("test_unmapped_sdpa");
    return g_failed ? 1 : 0;
}
