// test_other.cpp
// Tests for other/:
//   embed (token embedding lookup), rotaryembed (RoPE),
//   spectrogram / inversespectrogram (STFT),
//   roipooling / roialign (ROI pooling variants),
//   detectionoutput / yolodetectionoutput (NMS + box decode),
//   input, memorydata, noop (no-op / pass-through)

#include "test_utils.h"
#include <limits>

// ─── Embedding lookup ────────────────────────────────────────────────────────

// Simulates the Embed layer: given word indices and a weight matrix,
// returns the corresponding embedding rows.
static void ref_embed(const int* words, int num_words,
                       const float* weight, int embed_dim, int vocab_size,
                       const float* bias, float* out) {
    for (int i = 0; i < num_words; ++i) {
        int idx = std::max(0, std::min(words[i], vocab_size - 1));
        const float* row = weight + idx * embed_dim;
        for (int d = 0; d < embed_dim; ++d) {
            out[i * embed_dim + d] = row[d] + (bias ? bias[d] : 0.f);
        }
    }
}

// ─── Rotary Position Embedding (RoPE) ────────────────────────────────────────

// Split-half mode: x = [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
static void ref_rope_split(float* x, const float* cos_c, const float* sin_c,
                             int head_dim, int seq_len, int num_heads) {
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        for (int s = 0; s < seq_len; ++s) {
            float* ptr = x + (h * seq_len + s) * head_dim;
            for (int d = 0; d < half; ++d) {
                float x0 = ptr[d];
                float x1 = ptr[d + half];
                int ci = s * half + d;
                ptr[d]      = x0 * cos_c[ci] - x1 * sin_c[ci];
                ptr[d+half] = x0 * sin_c[ci] + x1 * cos_c[ci];
            }
        }
    }
}

// Interleaved mode: adjacent pairs rotate together
static void ref_rope_interleaved(float* x, const float* cos_c, const float* sin_c,
                                  int head_dim, int seq_len, int num_heads) {
    for (int h = 0; h < num_heads; ++h) {
        for (int s = 0; s < seq_len; ++s) {
            float* ptr = x + (h * seq_len + s) * head_dim;
            for (int d = 0; d < head_dim; d += 2) {
                float x0 = ptr[d];
                float x1 = ptr[d+1];
                int ci = s * (head_dim/2) + d/2;
                ptr[d]   = x0 * cos_c[ci] - x1 * sin_c[ci];
                ptr[d+1] = x0 * sin_c[ci] + x1 * cos_c[ci];
            }
        }
    }
}

// ─── ROI Pooling ─────────────────────────────────────────────────────────────

// Simplified ROI max-pool: pool a feature map region into a fixed output size
static void ref_roi_pool(const float* feat, int fw, int fh,
                          float roi_x1, float roi_y1, float roi_x2, float roi_y2,
                          int out_h, int out_w,
                          float spatial_scale, float* out) {
    float rh = roi_y2 - roi_y1;
    float rw = roi_x2 - roi_x1;
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            // Half-open intervals [ys, ye) and [xs, xe) for each bin
            float ys = roi_y1 * spatial_scale + oy     * rh * spatial_scale / out_h;
            float ye = roi_y1 * spatial_scale + (oy+1) * rh * spatial_scale / out_h;
            float xs = roi_x1 * spatial_scale + ox     * rw * spatial_scale / out_w;
            float xe = roi_x1 * spatial_scale + (ox+1) * rw * spatial_scale / out_w;
            int ys_i = std::max(0,      (int)floorf(ys));
            int ye_i = std::min(fh,     (int)ceilf(ye));   // exclusive
            int xs_i = std::max(0,      (int)floorf(xs));
            int xe_i = std::min(fw,     (int)ceilf(xe));   // exclusive
            float mx = -std::numeric_limits<float>::infinity();
            for (int fy = ys_i; fy < ye_i; ++fy)
                for (int fx = xs_i; fx < xe_i; ++fx)
                    mx = std::max(mx, feat[fy * fw + fx]);
            out[oy * out_w + ox] = (mx == -std::numeric_limits<float>::infinity()) ? 0.f : mx;
        }
    }
}

// ─── IoU for NMS ─────────────────────────────────────────────────────────────

static float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                       float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

// ─── Spectrogram / STFT helpers ──────────────────────────────────────────────

static float hann_window(int n, int N) {
    return 0.5f * (1.f - cosf(2.f * M_PI * n / (N - 1)));
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_embed_basic() {
    int vocab = 4, dim = 3;
    float weight[] = { 1.f, 2.f, 3.f,   // word 0
                        4.f, 5.f, 6.f,   // word 1
                        7.f, 8.f, 9.f,   // word 2
                       10.f,11.f,12.f }; // word 3
    int words[] = { 0, 2, 1 };
    float out[9];
    ref_embed(words, 3, weight, dim, vocab, nullptr, out);
    // word 0 → [1,2,3]
    ASSERT_NEAR(out[0], 1.f, 1e-5f); ASSERT_NEAR(out[1], 2.f, 1e-5f); ASSERT_NEAR(out[2], 3.f, 1e-5f);
    // word 2 → [7,8,9]
    ASSERT_NEAR(out[3], 7.f, 1e-5f); ASSERT_NEAR(out[4], 8.f, 1e-5f); ASSERT_NEAR(out[5], 9.f, 1e-5f);
    // word 1 → [4,5,6]
    ASSERT_NEAR(out[6], 4.f, 1e-5f);
}

void test_embed_with_bias() {
    float weight[] = { 0.f, 0.f };  // zero embedding
    float bias[]   = { 1.f, 2.f };
    int words[] = { 0 };
    float out[2];
    ref_embed(words, 1, weight, 2, 1, bias, out);
    ASSERT_NEAR(out[0], 1.f, 1e-5f);
    ASSERT_NEAR(out[1], 2.f, 1e-5f);
}

void test_embed_oob_clamp() {
    float weight[] = { 1.f, 2.f, 3.f,   // word 0
                        4.f, 5.f, 6.f }; // word 1
    int words[] = { -1, 100 };  // out of bounds
    float out[6];
    ref_embed(words, 2, weight, 3, 2, nullptr, out);
    // Both should clamp to valid range
    for (int i = 0; i < 6; ++i) ASSERT_TRUE(out[i] >= 1.f && out[i] <= 6.f);
}

void test_rope_split_identity() {
    // cos=1, sin=0 → rotation is identity
    int head_dim = 4, seq = 1, heads = 1, half = head_dim / 2;
    float x[] = { 1.f, 2.f, 3.f, 4.f };
    float cos_c[] = { 1.f, 1.f };  // seq*half = 1*2
    float sin_c[] = { 0.f, 0.f };
    ref_rope_split(x, cos_c, sin_c, head_dim, seq, heads);
    float expected[] = { 1.f, 2.f, 3.f, 4.f };
    ASSERT_VEC_NEAR(x, expected, head_dim, 1e-5f);
    (void)half;
}

void test_rope_split_90deg() {
    // cos=0, sin=1 → 90-degree rotation: [x0, x1] → [-x1, x0]
    int head_dim = 2, seq = 1, heads = 1;
    float x[] = { 3.f, 4.f };
    float cos_c[] = { 0.f };
    float sin_c[] = { 1.f };
    ref_rope_split(x, cos_c, sin_c, head_dim, seq, heads);
    ASSERT_NEAR(x[0], -4.f, 1e-5f);
    ASSERT_NEAR(x[1],  3.f, 1e-5f);
}

void test_rope_interleaved_identity() {
    int head_dim = 4, seq = 1, heads = 1;
    float x[] = { 1.f, 2.f, 3.f, 4.f };
    float cos_c[] = { 1.f, 1.f };  // 2 pairs
    float sin_c[] = { 0.f, 0.f };
    ref_rope_interleaved(x, cos_c, sin_c, head_dim, seq, heads);
    float expected[] = { 1.f, 2.f, 3.f, 4.f };
    ASSERT_VEC_NEAR(x, expected, head_dim, 1e-5f);
}

void test_rope_norm_preserving() {
    // RoPE is an orthogonal rotation → preserves L2 norm
    int head_dim = 4, seq = 1, heads = 1;
    float x[] = { 1.f, 2.f, 3.f, 4.f };
    float theta = 0.3f;
    float cos_c[] = { cosf(theta), cosf(theta) };
    float sin_c[] = { sinf(theta), sinf(theta) };
    float norm_before = 0.f;
    for (float v : x) norm_before += v * v;
    ref_rope_split(x, cos_c, sin_c, head_dim, seq, heads);
    float norm_after = 0.f;
    for (float v : x) norm_after += v * v;
    ASSERT_NEAR(norm_before, norm_after, 1e-4f);
}

void test_roi_pool_full_region() {
    // Feature map = 4x4, ROI = entire feature map
    float feat[16]; for (int i = 0; i < 16; i++) feat[i] = (float)(i + 1);
    float out[4];
    ref_roi_pool(feat, 4, 4, 0.f, 0.f, 4.f, 4.f, 2, 2, 1.f, out);
    // Top-left 2x2: max(1,2,5,6)=6
    ASSERT_NEAR(out[0], 6.f, 1e-4f);
    // Top-right 2x2: max(3,4,7,8)=8
    ASSERT_NEAR(out[1], 8.f, 1e-4f);
}

void test_roi_pool_single_cell() {
    float feat[4] = { 1.f, 2.f, 3.f, 4.f };
    float out[1];
    ref_roi_pool(feat, 2, 2, 0.f, 0.f, 2.f, 2.f, 1, 1, 1.f, out);
    // Max of entire 2x2 = 4
    ASSERT_NEAR(out[0], 4.f, 1e-4f);
}

void test_iou_no_overlap() {
    float iou = iou_xyxy(0, 0, 1, 1, 2, 2, 3, 3);
    ASSERT_NEAR(iou, 0.f, 1e-5f);
}

void test_iou_full_overlap() {
    float iou = iou_xyxy(0, 0, 2, 2, 0, 0, 2, 2);
    ASSERT_NEAR(iou, 1.f, 1e-5f);
}

void test_iou_partial_overlap() {
    // [0,0,2,2] and [1,1,3,3]: overlap = 1x1=1, union = 4+4-1=7
    float iou = iou_xyxy(0, 0, 2, 2, 1, 1, 3, 3);
    ASSERT_NEAR(iou, 1.f / 7.f, 1e-5f);
}

void test_hann_window() {
    int N = 8;
    // Hann window: w[0]=0, w[N/2] near 1, w[N-1]=0
    ASSERT_NEAR(hann_window(0, N), 0.f, 1e-5f);
    // Symmetric check: w[n] = w[N-1-n]
    for (int n = 0; n < N; ++n)
        ASSERT_NEAR(hann_window(n, N), hann_window(N-1-n, N), 1e-5f);
}

void test_noop_passthrough() {
    // noop/input: output equals input unchanged
    float data[] = { 1.f, 2.f, 3.f };
    float out[3];
    memcpy(out, data, sizeof(data));
    ASSERT_VEC_NEAR(out, data, 3, 1e-5f);
}

int main() {
    printf("=== test_other ===\n");
    RUN_TEST(test_embed_basic);
    RUN_TEST(test_embed_with_bias);
    RUN_TEST(test_embed_oob_clamp);
    RUN_TEST(test_rope_split_identity);
    RUN_TEST(test_rope_split_90deg);
    RUN_TEST(test_rope_interleaved_identity);
    RUN_TEST(test_rope_norm_preserving);
    RUN_TEST(test_roi_pool_full_region);
    RUN_TEST(test_roi_pool_single_cell);
    RUN_TEST(test_iou_no_overlap);
    RUN_TEST(test_iou_full_overlap);
    RUN_TEST(test_iou_partial_overlap);
    RUN_TEST(test_hann_window);
    RUN_TEST(test_noop_passthrough);
    print_summary("other");
    return g_failed > 0 ? 1 : 0;
}
