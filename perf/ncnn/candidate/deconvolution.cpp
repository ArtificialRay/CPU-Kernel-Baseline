#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolution.h"

// CANDIDATE_TESTCASE_START
// Mirrors perf/ncnn/baseline/deconvolution.cpp: setup once per shape, time
// PERF_INNER_REPS forwards. Same INNER_REPS as baseline so candidate vs
// baseline ms compare apples-to-apples (both report INNER_REPS forwards).
// EXPECT_MATCH is intentionally NOT used in perf binaries — correctness is
// the test binary's job; perf must time forward() alone, not ref + cmp.
static constexpr int PERF_INNER_REPS = 1;

void perf_deconv_base_3x3_s1() {
    auto c0 = setup_deconv2d(1, 1, 4, 4, 3, 3, 1, 1);
    auto c1 = setup_deconv2d(3, 4, 5, 5, 3, 3, 1, 1);
    auto c2 = setup_deconv2d( 64,  128, 56, 56, 3, 3, 1, 1);
    auto c3 = setup_deconv2d(128,  256, 28, 28, 3, 3, 1, 1);
    // auto c4 = setup_deconv2d(256,  512, 224,224, 3, 3, 1, 1);
    auto c5 = setup_deconv2d(  3,   32, 112, 112, 3, 3, 1, 1);
    // auto c6 = setup_deconv2d(512, 1024,  7,  7, 3, 3, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d(c0);
        forward_deconv2d(c1);
        forward_deconv2d(c2);
        forward_deconv2d(c3);
        // forward_deconv2d(c4);
        forward_deconv2d(c5);
        // forward_deconv2d(c6);
    }
}

void perf_deconv_base_3x3_s2() {
    auto c0 = setup_deconv2d(1, 1, 4, 4, 3, 3, 2, 2);
    auto c1 = setup_deconv2d(3, 4, 5, 5, 3, 3, 2, 2);
    auto c2 = setup_deconv2d( 64,  128, 28, 28, 3, 3, 2, 2);
    auto c3 = setup_deconv2d(128,  256, 14, 14, 3, 3, 2, 2);
    // auto c4 = setup_deconv2d(256,  512, 224,224, 3, 3, 2, 2);
    auto c5 = setup_deconv2d(  3,   32, 56, 56, 3, 3, 2, 2);
    // auto c6 = setup_deconv2d(512, 1024,  7,  7, 3, 3, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d(c0);
        forward_deconv2d(c1);
        forward_deconv2d(c2);
        forward_deconv2d(c3);
        // forward_deconv2d(c4);
        forward_deconv2d(c5);
        // forward_deconv2d(c6);
    }
}

void perf_deconv_base_4x4_s1() {
    auto c0 = setup_deconv2d(1, 1, 4, 4, 4, 4, 1, 1);
    auto c1 = setup_deconv2d(3, 4, 5, 5, 4, 4, 1, 1);
    auto c2 = setup_deconv2d( 64,  128, 56, 56, 4, 4, 1, 1);
    auto c3 = setup_deconv2d(128,  256, 28, 28, 4, 4, 1, 1);
    // auto c4 = setup_deconv2d(256,  512, 224,224, 4, 4, 1, 1);
    auto c5 = setup_deconv2d(  3,   32, 112, 112, 4, 4, 1, 1);
    // auto c6 = setup_deconv2d(512, 1024,  7,  7, 4, 4, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d(c0);
        forward_deconv2d(c1);
        forward_deconv2d(c2);
        forward_deconv2d(c3);
        // forward_deconv2d(c4);
        forward_deconv2d(c5);
        // forward_deconv2d(c6);
    }
}

void perf_deconv_base_4x4_s2() {
    auto c0 = setup_deconv2d(1, 1, 3, 3, 4, 4, 2, 2);
    auto c1 = setup_deconv2d(2, 4, 4, 4, 4, 4, 2, 2);
    auto c2 = setup_deconv2d( 32,   64, 56, 56, 4, 4, 2, 2);
    auto c3 = setup_deconv2d( 64,  128, 28, 28, 4, 4, 2, 2);
    // auto c4 = setup_deconv2d(128,  256, 224,224, 4, 4, 2, 2);
    auto c5 = setup_deconv2d(  3,   32, 56, 56, 4, 4, 2, 2);
    auto c6 = setup_deconv2d(512,  512,  7,  7, 4, 4, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d(c0);
        forward_deconv2d(c1);
        forward_deconv2d(c2);
        forward_deconv2d(c3);
        //forward_deconv2d(c4);
        forward_deconv2d(c5);
        // forward_deconv2d(c6);
    }
}

void perf_deconv_base_bias() {
    auto c0 = setup_deconv2d( 64, 128, 56, 56, 3, 3, 1, 1, true);
    auto c1 = setup_deconv2d( 64, 128, 28, 28, 3, 3, 2, 2, true);
    auto c2 = setup_deconv2d( 64, 128, 56, 56, 4, 4, 1, 1, true);
    auto c3 = setup_deconv2d( 64, 128, 28, 28, 4, 4, 2, 2, true);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d(c0);
        forward_deconv2d(c1);
        forward_deconv2d(c2);
        forward_deconv2d(c3);
    }
}
// CANDIDATE_TESTCASE_END