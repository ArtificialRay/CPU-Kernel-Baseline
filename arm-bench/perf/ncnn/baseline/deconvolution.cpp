#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolution_arm.h"

// BASELINE_TESTCASE_START
// ── Deconvolution_arm ─────────────────────────────────────────────
void perf_deconv_arm_2x2_s2() {
    run_deconv2d_arm(1, 1, 3, 3, 2, 2, 2, 2);
    run_deconv2d_arm(2, 4, 4, 4, 2, 2, 2, 2);
    run_deconv2d_arm(   32,   64, 56, 56, 2, 2, 2, 2);
    run_deconv2d_arm(   64,  128, 28, 28, 2, 2, 2, 2);
    run_deconv2d_arm(  128,  256, 14, 14, 2, 2, 2, 2);
    run_deconv2d_arm(  256,  512,  7,  7, 2, 2, 2, 2);
    run_deconv2d_arm(    3,   32, 112, 112, 2, 2, 2, 2);
    run_deconv2d_arm( 1024, 1024,  7,  7, 2, 2, 2, 2);
}

void perf_deconv_arm_3x3_s1() {
    run_deconv2d_arm(1, 1, 4, 4, 3, 3, 1, 1);
    run_deconv2d_arm(3, 4, 5, 5, 3, 3, 1, 1);
    run_deconv2d_arm(  64,  128, 56, 56, 3, 3, 1, 1);
    run_deconv2d_arm( 128,  256, 28, 28, 3, 3, 1, 1);
    run_deconv2d_arm( 256,  512, 14, 14, 3, 3, 1, 1);
    run_deconv2d_arm(   3,   32, 112, 112, 3, 3, 1, 1);
    run_deconv2d_arm( 512, 1024,  7,  7, 3, 3, 1, 1);
}

void perf_deconv_arm_bias() {
    run_deconv2d_arm(2, 4, 4, 4, 3, 3, 1, 1, true);
    run_deconv2d_arm(  64,  128, 56, 56, 3, 3, 1, 1, true);
    run_deconv2d_arm( 128,  256, 28, 28, 3, 3, 1, 1, true);
    run_deconv2d_arm( 256,  512, 14, 14, 3, 3, 1, 1, true);
    run_deconv2d_arm(   3,   32, 112, 112, 3, 3, 1, 1, true);
    run_deconv2d_arm( 512, 1024,  7,  7, 3, 3, 1, 1, true);
}
// BASELINE_TESTCASE_END