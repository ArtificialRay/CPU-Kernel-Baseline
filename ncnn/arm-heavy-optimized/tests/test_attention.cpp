// test_attention.cpp — ARM attention kernel tests
// Tests MultiHeadAttention_arm (forward with identity weights)

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../attention/multiheadattention_arm.h"

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_mha_arm_output_shape()
{
    const int embed_dim = 8;
    const int num_heads = 2;
    const int T = 3;

    ncnn::MultiHeadAttention_arm mha;
    mha.embed_dim        = embed_dim;
    mha.num_heads        = num_heads;
    mha.weight_data_size = embed_dim * embed_dim;
    mha.kdim             = embed_dim;
    mha.vdim             = embed_dim;
    mha.attn_mask        = 0;
    mha.scale            = 1.f / sqrtf((float)(embed_dim / num_heads));
    mha.kv_cache         = 0;
    mha.int8_scale_term  = 0;

    std::vector<float> weight(embed_dim * embed_dim, 0.f);
    for (int i = 0; i < embed_dim; ++i) weight[i * embed_dim + i] = 0.5f;
    std::vector<float> bias(embed_dim, 0.f);

    auto make_w = [](const std::vector<float>& v) {
        ncnn::Mat m; m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
        memcpy((float*)m, v.data(), v.size() * sizeof(float)); return m;
    };
    mha.q_weight_data = make_w(weight); mha.q_bias_data = make_w(bias);
    mha.k_weight_data = make_w(weight); mha.k_bias_data = make_w(bias);
    mha.v_weight_data = make_w(weight); mha.v_bias_data = make_w(bias);
    mha.out_weight_data = make_w(weight); mha.out_bias_data = make_w(bias);

    ncnn::Mat q_blob, k_blob, v_blob;
    q_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    k_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    v_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    for (int t = 0; t < T; ++t) {
        float* r = q_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 1.f;
        r = k_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 1.f;
        r = v_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 1.f;
    }

    std::vector<ncnn::Mat> bottom = { q_blob, k_blob, v_blob };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();

    // Try create_pipeline first; if it fails, fall back to base forward
    int cp_ret = mha.create_pipeline(opt);
    if (cp_ret != 0) {
        printf("  (MultiHeadAttention_arm create_pipeline returned %d, trying base forward)\n", cp_ret);
    }

    int ret = mha.forward(bottom, top, opt);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(top[0].w, embed_dim);
    ASSERT_EQ(top[0].h, T);
}

void test_mha_arm_identity_weights()
{
    const int embed_dim = 4;
    const int num_heads = 2;
    const int T = 2;

    ncnn::MultiHeadAttention_arm mha;
    mha.embed_dim        = embed_dim;
    mha.num_heads        = num_heads;
    mha.weight_data_size = embed_dim * embed_dim;
    mha.kdim             = embed_dim;
    mha.vdim             = embed_dim;
    mha.attn_mask        = 0;
    mha.scale            = 1.f / sqrtf((float)(embed_dim / num_heads));
    mha.kv_cache         = 0;
    mha.int8_scale_term  = 0;

    std::vector<float> eye16(embed_dim * embed_dim, 0.f);
    for (int i = 0; i < embed_dim; ++i) eye16[i * embed_dim + i] = 1.f;
    std::vector<float> zeros4(embed_dim, 0.f);

    auto make_w = [](const std::vector<float>& v) {
        ncnn::Mat m; m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
        memcpy((float*)m, v.data(), v.size() * sizeof(float)); return m;
    };
    mha.q_weight_data  = make_w(eye16);  mha.q_bias_data  = make_w(zeros4);
    mha.k_weight_data  = make_w(eye16);  mha.k_bias_data  = make_w(zeros4);
    mha.v_weight_data  = make_w(eye16);  mha.v_bias_data  = make_w(zeros4);
    mha.out_weight_data = make_w(eye16); mha.out_bias_data = make_w(zeros4);

    std::vector<float> input_flat = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
    };
    ncnn::Mat q_blob, k_blob, v_blob;
    q_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    k_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    v_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    for (int t = 0; t < T; ++t) {
        memcpy(q_blob.row(t), input_flat.data() + t * embed_dim, embed_dim * sizeof(float));
        memcpy(k_blob.row(t), input_flat.data() + t * embed_dim, embed_dim * sizeof(float));
        memcpy(v_blob.row(t), input_flat.data() + t * embed_dim, embed_dim * sizeof(float));
    }

    std::vector<ncnn::Mat> bottom = { q_blob, k_blob, v_blob };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();

    int cp_ret = mha.create_pipeline(opt);
    if (cp_ret != 0) {
        printf("  (MultiHeadAttention_arm create_pipeline returned %d, trying base forward)\n", cp_ret);
    }

    int ret = mha.forward(bottom, top, opt);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(top[0].w, embed_dim);
    ASSERT_EQ(top[0].h, T);

    // Verify no NaNs in output
    for (int t = 0; t < T; ++t) {
        const float* row = top[0].row(t);
        for (int d = 0; d < embed_dim; ++d) {
            ASSERT_TRUE(row[d] == row[d]);  // not NaN
        }
    }
}

void test_mha_arm_no_crash()
{
    // Minimal smoke test: MHA with small dimensions should not crash
    const int embed_dim = 4;
    const int num_heads = 1;
    const int T = 1;

    ncnn::MultiHeadAttention_arm mha;
    mha.embed_dim        = embed_dim;
    mha.num_heads        = num_heads;
    mha.weight_data_size = embed_dim * embed_dim;
    mha.kdim             = embed_dim;
    mha.vdim             = embed_dim;
    mha.attn_mask        = 0;
    mha.scale            = 1.f / sqrtf((float)embed_dim);
    mha.kv_cache         = 0;
    mha.int8_scale_term  = 0;

    std::vector<float> w(embed_dim * embed_dim, 0.f);
    for (int i = 0; i < embed_dim; ++i) w[i * embed_dim + i] = 1.f;
    std::vector<float> b(embed_dim, 0.f);

    auto mk = [](const std::vector<float>& v) {
        ncnn::Mat m; m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
        memcpy((float*)m, v.data(), v.size() * sizeof(float)); return m;
    };
    mha.q_weight_data = mk(w); mha.q_bias_data = mk(b);
    mha.k_weight_data = mk(w); mha.k_bias_data = mk(b);
    mha.v_weight_data = mk(w); mha.v_bias_data = mk(b);
    mha.out_weight_data = mk(w); mha.out_bias_data = mk(b);

    ncnn::Mat q_blob, k_blob, v_blob;
    q_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    k_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    v_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    for (int t = 0; t < T; ++t) {
        float* r;
        r = q_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 0.1f * (d + 1);
        r = k_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 0.1f * (d + 1);
        r = v_blob.row(t); for (int d = 0; d < embed_dim; ++d) r[d] = 0.1f * (d + 1);
    }

    std::vector<ncnn::Mat> bottom = { q_blob, k_blob, v_blob };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();

    mha.create_pipeline(opt);  // ignore return value
    int ret = mha.forward(bottom, top, opt);
    ASSERT_EQ(ret, 0);
    ASSERT_TRUE(top[0].dims > 0);
}

int main() {
    printf("=== test_attention (ARM) ===\n");
    printf("\n-- Real ARM MultiHeadAttention --\n");
    RUN_TEST(test_mha_arm_no_crash);
    RUN_TEST(test_mha_arm_output_shape);
    RUN_TEST(test_mha_arm_identity_weights);

    print_summary("attention_arm");
    return g_failed > 0 ? 1 : 0;
}
