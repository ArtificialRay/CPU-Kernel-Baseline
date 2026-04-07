// test_utils.h — shared test infrastructure for ncnn kernel tests
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>

// ─── Assertion macros ────────────────────────────────────────────────────────

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                        \
        if ((a) != (b)) {                                                       \
            fprintf(stderr, "  FAIL [%s:%d]  %s == %s  (%d != %d)\n",         \
                    __FILE__, __LINE__, #a, #b, (int)(a), (int)(b));           \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_TRUE(cond)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "  FAIL [%s:%d]  %s is false\n",                  \
                    __FILE__, __LINE__, #cond);                                 \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                  \
    do {                                                                        \
        float _a = (float)(a), _b = (float)(b), _t = (float)(tol);            \
        if (fabsf(_a - _b) > _t) {                                             \
            fprintf(stderr, "  FAIL [%s:%d]  |%s - %s| = %.6f > %.6f\n",      \
                    __FILE__, __LINE__, #a, #b, fabsf(_a - _b), _t);          \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_VEC_NEAR(got, ref, n, tol)                                       \
    do {                                                                        \
        for (int _i = 0; _i < (int)(n); ++_i) {                               \
            float _g = (float)(got)[_i], _r = (float)(ref)[_i];               \
            if (fabsf(_g - _r) > (float)(tol)) {                              \
                fprintf(stderr, "  FAIL [%s:%d]  vec[%d]: got %.6f  ref %.6f\n",\
                        __FILE__, __LINE__, _i, _g, _r);                       \
                g_failed++;                                                     \
                break;                                                          \
            }                                                                   \
        }                                                                       \
    } while (0)

// ─── Test runner ─────────────────────────────────────────────────────────────

static int g_failed = 0;
static int g_passed = 0;

#define RUN_TEST(fn)                                                            \
    do {                                                                        \
        int _before = g_failed;                                                 \
        fn();                                                                   \
        if (g_failed == _before) {                                             \
            printf("  PASS  %s\n", #fn);                                        \
            g_passed++;                                                         \
        } else {                                                                \
            printf("  FAIL  %s\n", #fn);                                        \
        }                                                                       \
    } while (0)

inline void print_summary(const char* suite) {
    int total = g_passed + g_failed;
    printf("\n[%s]  %d / %d passed\n", suite, g_passed, total);
}

// ─── Lightweight Mat helper (float32 only) ───────────────────────────────────
// Mirrors ncnn Mat semantics: shape stored as (w, h, c), flat row-major storage.

struct TestMat {
    int w, h, c;          // spatial width, height, channels
    std::vector<float> data;

    TestMat() : w(0), h(0), c(0) {}
    TestMat(int w_, int h_ = 1, int c_ = 1) : w(w_), h(h_), c(c_), data(w_ * h_ * c_, 0.f) {}
    TestMat(int w_, int h_, int c_, const std::vector<float>& d) : w(w_), h(h_), c(c_), data(d) {}

    float& at(int x, int y = 0, int ch = 0) { return data[ch * h * w + y * w + x]; }
    float  at(int x, int y = 0, int ch = 0) const { return data[ch * h * w + y * w + x]; }

    int total() const { return w * h * c; }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void fill_range() { for (int i = 0; i < total(); ++i) data[i] = (float)(i + 1) * 0.1f; }
    void fill_ramp()  { for (int i = 0; i < total(); ++i) data[i] = (float)(i + 1); }

    // Channel slice
    float* channel_ptr(int ch) { return data.data() + ch * h * w; }
    const float* channel_ptr(int ch) const { return data.data() + ch * h * w; }
};

// ─── Reference math helpers ──────────────────────────────────────────────────

inline float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }
inline float tanh_f(float x)    { return tanhf(x); }
inline float relu_f(float x)    { return x > 0.f ? x : 0.f; }
inline float softplus_f(float x) { return logf(1.f + expf(x)); }
inline float mish_f(float x)    { return x * tanh_f(softplus_f(x)); }
inline float swish_f(float x)   { return x * sigmoid_f(x); }

// Numerically-stable softmax over a flat array in-place
inline void softmax_inplace(float* p, int n) {
    float mx = *std::max_element(p, p + n);
    float s = 0.f;
    for (int i = 0; i < n; ++i) { p[i] = expf(p[i] - mx); s += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= s;
}

// int8 clamp
inline int8_t float_to_int8(float v) {
    int i = (int)roundf(v);
    if (i >  127) i =  127;
    if (i < -128) i = -128;
    return (int8_t)i;
}

// Dot product
inline float dot(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Generate repeatable weight data
inline std::vector<float> make_weights(int n, float scale = 1.f) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = ((float)((i * 1234567 + 7654321) % 1000) / 1000.f - 0.5f) * scale;
    return w;
}
