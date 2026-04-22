#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── DeconvolutionDepthWise_arm ────────────────────────────────────
void perf_dw_deconv_arm_2x2_s2() {
    run_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(   64,  56, 56, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(  128,  28, 28, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(  256,  14, 14, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(  512,   7,  7, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(   32, 112, 112, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm( 1024,  14, 14, 2, 2, 2, 2);
}

void perf_dw_deconv_arm_3x3_s1() {
    run_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(   64,  56, 56, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(  128,  28, 28, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(  512,  14, 14, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(   32, 224, 224, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm( 2048,   7,  7, 3, 3, 1, 1);
}
