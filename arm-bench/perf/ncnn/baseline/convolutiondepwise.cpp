#include "test_utils.h"
#include "starter/ncnn/baseline/convolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── ConvolutionDepthWise_arm ──────────────────────────────────────
void perf_dw_arm_3x3() {
    run_depthwise_conv2d_arm(2, 6, 6, 3, 3, 1, 1, 0, 0);
    run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm(8, 12, 12, 3, 3, 2, 2, 0, 0);
    run_depthwise_conv2d_arm(  64, 112, 112, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm( 128,  56,  56, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm( 256,  28,  28, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm( 512,  14,  14, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm(1024,   7,   7, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm(  32, 224, 224, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm(2048,  14,  14, 3, 3, 1, 1, 1, 1);
}

void perf_dw_arm_5x5() {
    run_depthwise_conv2d_arm(4, 8, 8, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm(  64,  56,  56, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm( 128,  28,  28, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm( 256,  14,  14, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm( 512,   7,   7, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm(  32, 224, 224, 5, 5, 1, 1, 2, 2);
    run_depthwise_conv2d_arm(1024,  14,  14, 5, 5, 1, 1, 2, 2);
}

void perf_dw_arm_bias() {
    run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_depthwise_conv2d_arm(  64, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_depthwise_conv2d_arm( 256,  28,  28, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_depthwise_conv2d_arm(1024,   7,   7, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_depthwise_conv2d_arm(  32, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_depthwise_conv2d_arm(2048,  14,  14, 3, 3, 1, 1, 1, 1, 1, 1, true);
}