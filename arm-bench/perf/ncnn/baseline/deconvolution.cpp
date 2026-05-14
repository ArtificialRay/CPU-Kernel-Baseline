#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolution_arm.h"

// BASELINE_TESTCASE_START
// ── Deconvolution_arm — perf ──────────────────────────────────────
// Shapes mirror tests/ncnn/baseline/deconvolution.cpp (only the four
// NEON-specialized (k, s) variants). Each perf function:
//   1. setup_deconv2d_arm(...) once per shape — pays create_pipeline cost
//      (which is largely wasted work on these (k, s); see header comment)
//      OUTSIDE the timed forward loop.
//   2. forward_deconv2d_arm(ctx) PERF_INNER_REPS times per shape — the
//      actual NEON kernel cost, dominating the per-binary wall time.
// The candidate-side perf file uses the same INNER_REPS so candidate vs
// baseline ms remain apples-to-apples (both report INNER_REPS forwards).
static constexpr int PERF_INNER_REPS = 1;

void perf_deconv_arm_3x3_s1() {
    auto c0 = setup_deconv2d_arm(1, 1, 4, 4, 3, 3, 1, 1);
    auto c1 = setup_deconv2d_arm(3, 4, 5, 5, 3, 3, 1, 1);
    auto c2 = setup_deconv2d_arm( 64,  128, 56, 56, 3, 3, 1, 1);
    auto c3 = setup_deconv2d_arm(128,  256, 28, 28, 3, 3, 1, 1);
    // auto c4 = setup_deconv2d_arm(256,  512, 224,224, 3, 3, 1, 1);
    auto c5 = setup_deconv2d_arm(  3,   32, 112, 112, 3, 3, 1, 1);
    // auto c6 = setup_deconv2d_arm(512, 1024,  7,  7, 3, 3, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d_arm(c0);
        forward_deconv2d_arm(c1);
        forward_deconv2d_arm(c2);
        forward_deconv2d_arm(c3);
        // forward_deconv2d_arm(c4);
        forward_deconv2d_arm(c5);
        // forward_deconv2d_arm(c6);
    }
}

void perf_deconv_arm_3x3_s2() {
    auto c0 = setup_deconv2d_arm(1, 1, 4, 4, 3, 3, 2, 2);
    auto c1 = setup_deconv2d_arm(3, 4, 5, 5, 3, 3, 2, 2);
    auto c2 = setup_deconv2d_arm( 64,  128, 28, 28, 3, 3, 2, 2);
    auto c3 = setup_deconv2d_arm(128,  256, 14, 14, 3, 3, 2, 2);
    // auto c4 = setup_deconv2d_arm(256,  512, 224,224,  3, 3, 2, 2);
    auto c5 = setup_deconv2d_arm(  3,   32, 56, 56, 3, 3, 2, 2);
    // auto c6 = setup_deconv2d_arm(512, 1024,  7,  7, 3, 3, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d_arm(c0);
        forward_deconv2d_arm(c1);
        forward_deconv2d_arm(c2);
        forward_deconv2d_arm(c3);
        // forward_deconv2d_arm(c4);
        forward_deconv2d_arm(c5);
        // forward_deconv2d_arm(c6);
    }
}

void perf_deconv_arm_4x4_s1() {
    auto c0 = setup_deconv2d_arm(1, 1, 4, 4, 4, 4, 1, 1);
    auto c1 = setup_deconv2d_arm(3, 4, 5, 5, 4, 4, 1, 1);
    auto c2 = setup_deconv2d_arm( 64,  128, 56, 56, 4, 4, 1, 1);
    auto c3 = setup_deconv2d_arm(128,  256, 28, 28, 4, 4, 1, 1);
    // auto c4 = setup_deconv2d_arm(256,  512, 224,224,  4, 4, 1, 1);
    auto c5 = setup_deconv2d_arm(  3,   32, 112, 112, 4, 4, 1, 1);
    // auto c6 = setup_deconv2d_arm(512, 1024,  7,  7, 4, 4, 1, 1);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d_arm(c0);
        forward_deconv2d_arm(c1);
        forward_deconv2d_arm(c2);
        forward_deconv2d_arm(c3);
        // forward_deconv2d_arm(c4);
        forward_deconv2d_arm(c5);
        // forward_deconv2d_arm(c6);
    }
}

void perf_deconv_arm_4x4_s2() {
    auto c0 = setup_deconv2d_arm(1, 1, 3, 3, 4, 4, 2, 2);
    auto c1 = setup_deconv2d_arm(2, 4, 4, 4, 4, 4, 2, 2);
    auto c2 = setup_deconv2d_arm( 32,   64, 56, 56, 4, 4, 2, 2);
    auto c3 = setup_deconv2d_arm( 64,  128, 28, 28, 4, 4, 2, 2);
    // auto c4 = setup_deconv2d_arm(128,  256, 224,224,  4, 4, 2, 2);
    auto c5 = setup_deconv2d_arm(  3,   32, 56, 56, 4, 4, 2, 2);
    // auto c6 = setup_deconv2d_arm(512,  512,  7,  7, 4, 4, 2, 2);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d_arm(c0);
        forward_deconv2d_arm(c1);
        forward_deconv2d_arm(c2);
        forward_deconv2d_arm(c3);
        // forward_deconv2d_arm(c4);
        forward_deconv2d_arm(c5);
        // forward_deconv2d_arm(c6);
    }
}

void perf_deconv_arm_bias() {
    auto c0 = setup_deconv2d_arm( 64, 128, 56, 56, 3, 3, 1, 1, true);
    auto c1 = setup_deconv2d_arm( 64, 128, 28, 28, 3, 3, 2, 2, true);
    auto c2 = setup_deconv2d_arm( 64, 128, 56, 56, 4, 4, 1, 1, true);
    auto c3 = setup_deconv2d_arm( 64, 128, 28, 28, 4, 4, 2, 2, true);

    for (int rep = 0; rep < PERF_INNER_REPS; ++rep) {
        forward_deconv2d_arm(c0);
        forward_deconv2d_arm(c1);
        forward_deconv2d_arm(c2);
        forward_deconv2d_arm(c3);
    }
}
// BASELINE_TESTCASE_END
