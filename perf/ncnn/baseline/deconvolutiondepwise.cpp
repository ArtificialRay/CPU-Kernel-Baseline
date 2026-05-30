#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── DeconvolutionDepthWise_arm — perf ─────────────────────────────
// Each perf function:
//   1. setup_depthwise_deconv2d_arm(...) once per shape — pays create_pipeline
//      (and per-group weight packing) OUTSIDE the timed forward loop.
//   2. forward_depthwise_deconv2d_arm(ctx) PERF_INNER_REPS times — the NEON
//      kernel cost, dominating per-binary wall time.
// Candidate-side perf file uses the same INNER_REPS so candidate vs baseline
// ms remain apples-to-apples (both report INNER_REPS forwards).
static constexpr int PERF_INNER_REPS = 1;

void perf_dw_deconv_arm_2x2_s2() {
    auto c0 = setup_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2);
    auto c1 = setup_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2);
    auto c2 = setup_depthwise_deconv2d_arm(  64,  56, 56, 2, 2, 2, 2);
    auto c3 = setup_depthwise_deconv2d_arm( 128,  28, 28, 2, 2, 2, 2);
    auto c4 = setup_depthwise_deconv2d_arm( 256,  14, 14, 2, 2, 2, 2);
    auto c5 = setup_depthwise_deconv2d_arm( 512,   7,  7, 2, 2, 2, 2);
    auto c6 = setup_depthwise_deconv2d_arm(  32, 112, 112, 2, 2, 2, 2);
    auto c7 = setup_depthwise_deconv2d_arm(1024,  14, 14, 2, 2, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_depthwise_deconv2d_arm(c0);
        forward_depthwise_deconv2d_arm(c1);
        forward_depthwise_deconv2d_arm(c2);
        forward_depthwise_deconv2d_arm(c3);
        forward_depthwise_deconv2d_arm(c4);
        forward_depthwise_deconv2d_arm(c5);
        forward_depthwise_deconv2d_arm(c6);
        forward_depthwise_deconv2d_arm(c7);
    }
}

void perf_dw_deconv_arm_3x3_s1() {
    auto c0 = setup_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1);
    auto c1 = setup_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1);
    auto c2 = setup_depthwise_deconv2d_arm(  64,  56, 56, 3, 3, 1, 1);
    auto c3 = setup_depthwise_deconv2d_arm( 128,  28, 28, 3, 3, 1, 1);
    auto c4 = setup_depthwise_deconv2d_arm( 512,  14, 14, 3, 3, 1, 1);
    auto c5 = setup_depthwise_deconv2d_arm(  32, 224, 224, 3, 3, 1, 1);
    auto c6 = setup_depthwise_deconv2d_arm(2048,   7,  7, 3, 3, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_depthwise_deconv2d_arm(c0);
        forward_depthwise_deconv2d_arm(c1);
        forward_depthwise_deconv2d_arm(c2);
        forward_depthwise_deconv2d_arm(c3);
        forward_depthwise_deconv2d_arm(c4);
        forward_depthwise_deconv2d_arm(c5);
        forward_depthwise_deconv2d_arm(c6);
    }
}
// BASELINE_TESTCASE_END