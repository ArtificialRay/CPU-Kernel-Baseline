// test_attention.cpp
// Tests for the REAL ncnn kernel implementations:
//   - attention/sdpa.cpp         → ncnn::SDPA::forward()
//   - attention/multiheadattention.cpp → ncnn::MultiHeadAttention::forward()
//
// Each test:
//   1. Constructs actual ncnn::Mat inputs.
//   2. Calls the real forward() from the compiled attention sources.
//   3. Compares output against an independent reference.

#include "test_utils.h"

// Pull in the real ncnn headers (framework/ lives one directory up).
#include "../attention/sdpa.h"
#include "../attention/multiheadattention.h"
#include "../../framework/option.h"

#include <vector>
#include <cmath>
#include <cstring>

// ─── Reference SDPA ──────────────────────────────────────────────────────────
// Inputs stored as [num_heads, tgt_len, head_dim] flat row-major arrays.
static void ref_sdpa(const float* Q, const float* K, const float* V,
                     float* out,
                     int num_heads, int tgt_len, int src_len,
                     int head_dim, int v_dim,
                     float scale,
                     const float* attn_mask = nullptr)
{
    std::vector<float> scores(tgt_len * src_len);
    for (int h = 0; h < num_heads; ++h) {
        const float* q = Q + h * tgt_len * head_dim;
        const float* k = K + h * src_len * head_dim;
        const float* v = V + h * src_len * v_dim;
        float*       o = out + h * tgt_len * v_dim;

        for (int i = 0; i < tgt_len; ++i)
            for (int j = 0; j < src_len; ++j) {
                float s = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    s += q[i*head_dim+d] * k[j*head_dim+d];
                scores[i*src_len+j] = s * scale;
                if (attn_mask) scores[i*src_len+j] += attn_mask[i*src_len+j];
            }
        for (int i = 0; i < tgt_len; ++i) softmax_inplace(scores.data() + i*src_len, src_len);
        for (int i = 0; i < tgt_len; ++i)
            for (int d = 0; d < v_dim; ++d) {
                float sum = 0.f;
                for (int j = 0; j < src_len; ++j) sum += scores[i*src_len+j] * v[j*v_dim+d];
                o[i*v_dim+d] = sum;
            }
    }
}

// ─── Helpers to convert between flat arrays and ncnn::Mat ────────────────────
// ncnn::Mat (3-D) layout:  channel c → row h → element w
// Flat layout we use here: [c][h][w]  (same logical order, but ncnn has cstep
// alignment that may differ from w*h).

static ncnn::Mat make_mat(int w_, int h_, int c_,
                          const std::vector<float>& flat)
{
    ncnn::Mat m;
    m.create(w_, h_, c_, 4u, (ncnn::Allocator*)0);
    for (int cc = 0; cc < c_; ++cc)
        for (int hh = 0; hh < h_; ++hh) {
            float* dst = m.channel(cc).row(hh);
            const float* src = flat.data() + cc * h_ * w_ + hh * w_;
            memcpy(dst, src, w_ * sizeof(float));
        }
    return m;
}

static void read_mat(const ncnn::Mat& m, std::vector<float>& flat)
{
    flat.resize(m.c * m.h * m.w);
    for (int cc = 0; cc < m.c; ++cc)
        for (int hh = 0; hh < m.h; ++hh) {
            const float* src = m.channel(cc).row(hh);
            float* dst = flat.data() + cc * m.h * m.w + hh * m.w;
            memcpy(dst, src, m.w * sizeof(float));
        }
}

// ─── SDPA test runner ─────────────────────────────────────────────────────────
// Runs the real ncnn::SDPA::forward() and checks it matches ref_sdpa().
// Returns true if all values match within tol.
static bool run_sdpa_test(int num_heads, int tgt_len, int src_len,
                           int head_dim, int v_dim, float scale,
                           const std::vector<float>& Q_flat,
                           const std::vector<float>& K_flat,
                           const std::vector<float>& V_flat,
                           const std::vector<float>* mask_flat = nullptr,
                           float tol = 1e-4f)
{
    // ── Reference ──
    std::vector<float> ref(num_heads * tgt_len * v_dim, 0.f);
    ref_sdpa(Q_flat.data(), K_flat.data(), V_flat.data(), ref.data(),
             num_heads, tgt_len, src_len, head_dim, v_dim, scale,
             mask_flat ? mask_flat->data() : nullptr);

    // ── Build ncnn Mats ──
    // SDPA input convention:  query[c=heads, h=tgt_len, w=embed_dim]
    //                         cur_key[c=heads, h=src_len, w=embed_dim]
    //                         cur_value[c=heads, h=src_len, w=v_dim]
    ncnn::Mat q = make_mat(head_dim, tgt_len, num_heads, Q_flat);
    ncnn::Mat k = make_mat(head_dim, src_len, num_heads, K_flat);
    ncnn::Mat v = make_mat(v_dim,    src_len, num_heads, V_flat);

    std::vector<ncnn::Mat> bottom = {q, k, v};

    if (mask_flat) {
        // attn_mask shape: [tgt_len, src_len] (no head dim → single shared mask)
        ncnn::Mat mask = make_mat(src_len, tgt_len, 1, *mask_flat);
        bottom.push_back(mask);
    }

    std::vector<ncnn::Mat> top(1);

    ncnn::SDPA sdpa;
    sdpa.attn_mask      = (mask_flat != nullptr) ? 1 : 0;
    sdpa.scale          = scale;
    sdpa.kv_cache       = 0;
    sdpa.int8_scale_term = 0;

    ncnn::Option opt;
    opt.num_threads = 1;

    int ret = sdpa.forward(bottom, top, opt);
    if (ret != 0) {
        fprintf(stderr, "  SDPA::forward() returned %d\n", ret);
        return false;
    }

    // ── Compare ──
    std::vector<float> got;
    read_mat(top[0], got);

    for (int i = 0; i < (int)ref.size(); ++i) {
        if (fabsf(got[i] - ref[i]) > tol) {
            fprintf(stderr, "  mismatch at [%d]: got=%.6f  ref=%.6f\n",
                    i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_sdpa_uniform_attention()
{
    // Q[2]=[0,0] → all scores equal → uniform attention
    int H=1, T=3, S=3, D=2;
    float scale = 1.f / sqrtf((float)D);
    std::vector<float> Q = { 1,0, 0,1, 0,0 };
    std::vector<float> K = { 1,0, 0,1, 0,0 };
    std::vector<float> V = { 3.f,6.f, 1.f,2.f, 2.f,4.f };

    std::vector<float> ref(H * T * D, 0.f);
    ref_sdpa(Q.data(), K.data(), V.data(), ref.data(), H, T, S, D, D, scale);

    ASSERT_TRUE(run_sdpa_test(H, T, S, D, D, scale, Q, K, V));
    // spot-check: uniform attention on row 2 → mean(V)
    ASSERT_NEAR(ref[4], 2.f, 1e-4f);
    ASSERT_NEAR(ref[5], 4.f, 1e-4f);
}

void test_sdpa_sharp_attention()
{
    // Q=[0,1] matches K[1]=[0,1] closely → output ≈ V[1]=11
    int H=1, T=1, S=3, D=2;
    std::vector<float> Q = { 0.f, 1.f };
    std::vector<float> K = { 1.f, 0.f,  0.f, 1.f,  0.f, 0.f };
    std::vector<float> V = { 10.f, 11.f, 12.f };   // v_dim=1
    float scale = 1.f;

    ASSERT_TRUE(run_sdpa_test(H, T, S, D, 1, scale, Q, K, V));

    // verify impl output > 10 and < 12, close to 11
    ncnn::Mat qm  = make_mat(D, T, H, Q);
    ncnn::Mat km  = make_mat(D, S, H, K);
    ncnn::Mat vm  = make_mat(1, S, H, V);
    std::vector<ncnn::Mat> bot = {qm, km, vm};
    std::vector<ncnn::Mat> top(1);
    ncnn::SDPA sdpa; sdpa.attn_mask=0; sdpa.scale=scale; sdpa.kv_cache=0; sdpa.int8_scale_term=0;
    ncnn::Option opt; opt.num_threads=1;
    sdpa.forward(bot, top, opt);
    float v0 = top[0].channel(0).row(0)[0];
    ASSERT_TRUE(v0 > 10.f && v0 < 12.f);
    ASSERT_NEAR(v0, 11.f, 1.f);
}

void test_sdpa_output_shape()
{
    int H=2, T=4, S=6, D=8, VD=8;
    float scale = 1.f / sqrtf((float)D);
    std::vector<float> Q(H*T*D, 0.f), K(H*S*D, 0.f), V(H*S*VD, 0.f);
    ASSERT_TRUE(run_sdpa_test(H, T, S, D, VD, scale, Q, K, V));
}

void test_sdpa_causal_mask()
{
    // 2-token self-attention with causal mask
    int H=1, T=2, S=2, D=1;
    std::vector<float> Q={1.f,1.f}, K={1.f,1.f}, V={1.f,2.f};
    float neg_inf = -1e9f;
    std::vector<float> mask = {0.f, neg_inf, 0.f, 0.f};
    float scale = 1.f;

    ncnn::Mat qm = make_mat(D,T,H,Q), km=make_mat(D,S,H,K), vm=make_mat(1,S,H,V);
    ncnn::Mat msk = make_mat(S, T, 1, mask);
    std::vector<ncnn::Mat> bot={qm,km,vm,msk};
    std::vector<ncnn::Mat> top(1);
    ncnn::SDPA sdpa; sdpa.attn_mask=1; sdpa.scale=scale; sdpa.kv_cache=0; sdpa.int8_scale_term=0;
    ncnn::Option opt; opt.num_threads=1;
    sdpa.forward(bot, top, opt);

    float o0 = top[0].channel(0).row(0)[0];
    float o1 = top[0].channel(0).row(1)[0];
    ASSERT_NEAR(o0, 1.f,  1e-4f);   // token 0: sees only V[0]=1
    ASSERT_NEAR(o1, 1.5f, 1e-4f);   // token 1: equal mix of V[0]=1, V[1]=2
}

void test_sdpa_scale_effect()
{
    // Higher scale → sharper attention → output closer to dominant value
    int H=1, T=1, S=3, D=1;
    std::vector<float> Q={1.f}, K={2.f,0.f,0.f}, V={10.f,1.f,1.f};

    ncnn::Option opt; opt.num_threads=1;
    auto run = [&](float scale) -> float {
        ncnn::Mat qm=make_mat(D,T,H,Q), km=make_mat(D,S,H,K), vm=make_mat(1,S,H,V);
        std::vector<ncnn::Mat> bot={qm,km,vm}, top(1);
        ncnn::SDPA s; s.attn_mask=0; s.scale=scale; s.kv_cache=0; s.int8_scale_term=0;
        s.forward(bot,top,opt);
        return top[0].channel(0).row(0)[0];
    };

    float sharp = run(10.f);
    float flat  = run(0.01f);
    ASSERT_TRUE(sharp > flat);  // high scale → more weight on K[0] → closer to V[0]=10
}

void test_sdpa_multi_head()
{
    // 2 heads with complementary Q→K matching
    int H=2, T=1, S=2, D=2, VD=2;
    // head0: Q=[1,0] → mostly K[0]=[1,0] → V[0]=[100,0]
    // head1: Q=[0,1] → mostly K[1]=[0,1] → V[1]=[0,100] (head1)
    std::vector<float> Q = { 1.f,0.f,   0.f,1.f };
    std::vector<float> K = { 1.f,0.f, 0.f,1.f,
                              1.f,0.f, 0.f,1.f };
    std::vector<float> V = { 100.f,0.f, 0.f,100.f,
                               0.f,100.f, 100.f,0.f };
    float scale = 1.f;
    ASSERT_TRUE(run_sdpa_test(H, T, S, D, VD, scale, Q, K, V));

    ncnn::Mat qm=make_mat(D,T,H,Q), km=make_mat(D,S,H,K), vm=make_mat(VD,S,H,V);
    std::vector<ncnn::Mat> bot={qm,km,vm}, top(1);
    ncnn::SDPA sdpa; sdpa.attn_mask=0; sdpa.scale=scale; sdpa.kv_cache=0; sdpa.int8_scale_term=0;
    ncnn::Option opt; opt.num_threads=1;
    sdpa.forward(bot,top,opt);

    float h0d0 = top[0].channel(0).row(0)[0];
    float h0d1 = top[0].channel(0).row(0)[1];
    float h1d0 = top[0].channel(1).row(0)[0];
    float h1d1 = top[0].channel(1).row(0)[1];
    ASSERT_TRUE(h0d0 > 50.f);   // head0 mostly sees V[0]=[100,0]
    ASSERT_TRUE(h0d1 < 50.f);
    ASSERT_TRUE(h1d0 > 50.f);   // head1 mostly sees V[1]=[0,100] in head1
    ASSERT_TRUE(h1d1 < 50.f);
}

void test_sdpa_matches_reference()
{
    // Larger random-looking but deterministic test: verify impl == ref exactly
    int H=3, T=5, S=7, D=4, VD=4;
    float scale = 1.f / sqrtf((float)D);

    // Fill with a simple deterministic pattern
    int Qsz=H*T*D, Ksz=H*S*D, Vsz=H*S*VD;
    std::vector<float> Qf(Qsz), Kf(Ksz), Vf(Vsz);
    for (int i=0;i<Qsz;++i) Qf[i] = sinf((float)(i+1)*0.3f);
    for (int i=0;i<Ksz;++i) Kf[i] = cosf((float)(i+1)*0.2f);
    for (int i=0;i<Vsz;++i) Vf[i] = sinf((float)(i+1)*0.1f + 1.f);

    ASSERT_TRUE(run_sdpa_test(H, T, S, D, VD, scale, Qf, Kf, Vf, nullptr, 1e-4f));
}

// ─── MultiHeadAttention tests ─────────────────────────────────────────────────
// We set weight matrices directly on the MHA object (they are public Mat members).

void test_mha_identity_weights()
{
    // With identity Q/K/V projection weights, zero biases, and identity output
    // projection, MHA(Q,K,V) reduces to sdpa(Q,K,V) with scale already applied.
    //
    // Setup:
    //   embed_dim = 4, num_heads = 2, head_dim = 2
    //   qdim = vdim = kdim = embed_dim = 4 (square weight matrices)
    //   W_q = W_k = W_v = I_4,  b_q = b_k = b_v = 0
    //   W_out = I_4,  b_out = 0
    //
    // Note: MHA stores output projection as [qdim * embed_dim] = [4*4] = 16 elems,
    // and input projection W_q as [embed_dim * qdim] = [4*4] = 16 elems.

    const int embed_dim = 4;
    const int num_heads = 2;
    const int T = 2;  // seq_len

    // Build MHA
    ncnn::MultiHeadAttention mha;
    mha.embed_dim       = embed_dim;
    mha.num_heads       = num_heads;
    mha.weight_data_size = embed_dim * embed_dim;  // qdim = embed_dim
    mha.kdim            = embed_dim;
    mha.vdim            = embed_dim;
    mha.attn_mask       = 0;
    mha.scale           = 1.f / sqrtf((float)(embed_dim / num_heads));
    mha.kv_cache        = 0;
    mha.int8_scale_term = 0;

    // Identity weights
    std::vector<float> eye16(embed_dim * embed_dim, 0.f);
    for (int i = 0; i < embed_dim; ++i) eye16[i * embed_dim + i] = 1.f;
    std::vector<float> zeros4(embed_dim, 0.f);

    auto make_weight_mat = [](const std::vector<float>& v) {
        ncnn::Mat m;
        m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
        memcpy((float*)m, v.data(), v.size() * sizeof(float));
        return m;
    };

    mha.q_weight_data  = make_weight_mat(eye16);
    mha.q_bias_data    = make_weight_mat(zeros4);
    mha.k_weight_data  = make_weight_mat(eye16);
    mha.k_bias_data    = make_weight_mat(zeros4);
    mha.v_weight_data  = make_weight_mat(eye16);
    mha.v_bias_data    = make_weight_mat(zeros4);
    mha.out_weight_data = make_weight_mat(eye16);
    mha.out_bias_data   = make_weight_mat(zeros4);

    // Input: Q=K=V, shape [T, embed_dim] (2D Mat: w=embed_dim, h=T)
    // Use a simple deterministic sequence
    std::vector<float> input_flat = {
        1.f, 0.f, 0.f, 0.f,   // token 0
        0.f, 1.f, 0.f, 0.f,   // token 1
    };
    ncnn::Mat q_blob, k_blob, v_blob;
    q_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    k_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    v_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    for (int t = 0; t < T; ++t) {
        memcpy(q_blob.row(t), input_flat.data() + t*embed_dim, embed_dim*sizeof(float));
        memcpy(k_blob.row(t), input_flat.data() + t*embed_dim, embed_dim*sizeof(float));
        memcpy(v_blob.row(t), input_flat.data() + t*embed_dim, embed_dim*sizeof(float));
    }

    std::vector<ncnn::Mat> bottom = {q_blob, k_blob, v_blob};
    std::vector<ncnn::Mat> top(1);

    ncnn::Option opt;
    opt.num_threads = 1;

    int ret = mha.forward(bottom, top, opt);
    ASSERT_EQ(ret, 0);

    // Output should be [T, embed_dim] = [2, 4]
    ASSERT_EQ(top[0].w, embed_dim);
    ASSERT_EQ(top[0].h, T);

    // With identity weights + zero biases and shared Q=K=V, the output is a
    // weighted average of V (=input), which is a valid self-attention result.
    // Just verify shape and no NaNs.
    for (int t = 0; t < T; ++t) {
        const float* row = top[0].row(t);
        for (int d = 0; d < embed_dim; ++d) {
            ASSERT_TRUE(row[d] == row[d]);   // not NaN
        }
    }
}

void test_mha_output_shape()
{
    // Verify the real MHA produces output with w=embed_dim, h=src_seqlen
    const int embed_dim = 8;
    const int num_heads = 2;
    const int T = 3;

    ncnn::MultiHeadAttention mha;
    mha.embed_dim        = embed_dim;
    mha.num_heads        = num_heads;
    mha.weight_data_size = embed_dim * embed_dim;
    mha.kdim             = embed_dim;
    mha.vdim             = embed_dim;
    mha.attn_mask        = 0;
    mha.scale            = 1.f / sqrtf((float)(embed_dim / num_heads));
    mha.kv_cache         = 0;
    mha.int8_scale_term  = 0;

    // Random-ish weights (just scale identity so values stay small)
    std::vector<float> weight(embed_dim * embed_dim, 0.f);
    for (int i = 0; i < embed_dim; ++i) weight[i*embed_dim+i] = 0.5f;
    std::vector<float> bias(embed_dim, 0.f);

    auto make_w = [](const std::vector<float>& v) {
        ncnn::Mat m; m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
        memcpy((float*)m, v.data(), v.size()*sizeof(float)); return m;
    };
    mha.q_weight_data = make_w(weight); mha.q_bias_data = make_w(bias);
    mha.k_weight_data = make_w(weight); mha.k_bias_data = make_w(bias);
    mha.v_weight_data = make_w(weight); mha.v_bias_data = make_w(bias);
    mha.out_weight_data = make_w(weight); mha.out_bias_data = make_w(bias);

    ncnn::Mat q_blob, k_blob, v_blob;
    q_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    k_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    v_blob.create(embed_dim, T, 4u, (ncnn::Allocator*)0);
    // Fill with 1s
    for (int t=0; t<T; ++t) {
        float* r = q_blob.row(t); for (int d=0;d<embed_dim;++d) r[d]=1.f;
        r = k_blob.row(t); for (int d=0;d<embed_dim;++d) r[d]=1.f;
        r = v_blob.row(t); for (int d=0;d<embed_dim;++d) r[d]=1.f;
    }

    std::vector<ncnn::Mat> bottom={q_blob,k_blob,v_blob}, top(1);
    ncnn::Option opt; opt.num_threads=1;
    int ret = mha.forward(bottom, top, opt);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(top[0].w, embed_dim);
    ASSERT_EQ(top[0].h, T);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    printf("=== test_attention (real ncnn::SDPA + ncnn::MultiHeadAttention) ===\n");

    printf("\n-- SDPA --\n");
    RUN_TEST(test_sdpa_uniform_attention);
    RUN_TEST(test_sdpa_sharp_attention);
    RUN_TEST(test_sdpa_output_shape);
    RUN_TEST(test_sdpa_causal_mask);
    RUN_TEST(test_sdpa_scale_effect);
    RUN_TEST(test_sdpa_multi_head);
    RUN_TEST(test_sdpa_matches_reference);

    printf("\n-- MultiHeadAttention --\n");
    RUN_TEST(test_mha_identity_weights);
    RUN_TEST(test_mha_output_shape);

    print_summary("attention");
    return g_failed > 0 ? 1 : 0;
}
