#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolutiondepthwise.h"

// CANDIDATE_TESTCASE_START
// Mirrors perf/ncnn/baseline/deconvolutiondepwise.cpp: setup once per shape,
// time PERF_INNER_REPS forwards. Same INNER_REPS as baseline so candidate vs
// baseline ms compare apples-to-apples (both report INNER_REPS forwards).
// EXPECT_MATCH is intentionally NOT used in perf binaries — correctness is
// the test binary's job; perf must time forward() alone, not ref + cmp.
static constexpr int PERF_INNER_REPS = 1;

void perf_dw_deconv_base_2x2_s2() {
    auto c0 = setup_depthwise_deconv2d(2, 3, 3, 2, 2, 2, 2);
    auto c1 = setup_depthwise_deconv2d(4, 4, 4, 2, 2, 2, 2);
    auto c2 = setup_depthwise_deconv2d(  64,  56, 56, 2, 2, 2, 2);
    auto c3 = setup_depthwise_deconv2d( 128,  28, 28, 2, 2, 2, 2);
    auto c4 = setup_depthwise_deconv2d( 256,  14, 14, 2, 2, 2, 2);
    auto c5 = setup_depthwise_deconv2d( 512,   7,  7, 2, 2, 2, 2);
    auto c6 = setup_depthwise_deconv2d(  32, 112, 112, 2, 2, 2, 2);
    auto c7 = setup_depthwise_deconv2d(1024,  14, 14, 2, 2, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_depthwise_deconv2d(c0);
        forward_depthwise_deconv2d(c1);
        forward_depthwise_deconv2d(c2);
        forward_depthwise_deconv2d(c3);
        forward_depthwise_deconv2d(c4);
        forward_depthwise_deconv2d(c5);
        forward_depthwise_deconv2d(c6);
        forward_depthwise_deconv2d(c7);
    }
}

void perf_dw_deconv_base_3x3_s1() {
    auto c0 = setup_depthwise_deconv2d(2, 4, 4, 3, 3, 1, 1);
    auto c1 = setup_depthwise_deconv2d(4, 5, 5, 3, 3, 1, 1);
    auto c2 = setup_depthwise_deconv2d(  64,  56, 56, 3, 3, 1, 1);
    auto c3 = setup_depthwise_deconv2d( 128,  28, 28, 3, 3, 1, 1);
    auto c4 = setup_depthwise_deconv2d( 512,  14, 14, 3, 3, 1, 1);
    auto c5 = setup_depthwise_deconv2d(  32, 224, 224, 3, 3, 1, 1);
    auto c6 = setup_depthwise_deconv2d(2048,   7,  7, 3, 3, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_depthwise_deconv2d(c0);
        forward_depthwise_deconv2d(c1);
        forward_depthwise_deconv2d(c2);
        forward_depthwise_deconv2d(c3);
        forward_depthwise_deconv2d(c4);
        forward_depthwise_deconv2d(c5);
        forward_depthwise_deconv2d(c6);
    }
}
// CANDIDATE_TESTCASE_END