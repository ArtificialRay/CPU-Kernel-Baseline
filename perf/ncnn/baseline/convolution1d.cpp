#include "starter/ncnn/baseline/convolution1d_arm.h"

// BASELINE_TESTCASE_START
void perf_conv1d_arm_k3() {
    run_conv1d_arm(2, 4, 8, 3, 1, 0);
    run_conv1d_arm(4, 8, 16, 3, 1, 1);
    run_conv1d_arm(8, 4, 12, 3, 2, 0);
    run_conv1d_arm(  64,  128, 112, 3, 1, 1);
    run_conv1d_arm( 128,  256,  56, 3, 1, 1);
    run_conv1d_arm( 256,  512,  28, 3, 1, 1);
    run_conv1d_arm( 512, 1024,  14, 3, 1, 1);
    run_conv1d_arm(   3,   64, 224, 3, 1, 1);
    run_conv1d_arm(1024, 1024,  14, 3, 1, 1);
}

void perf_conv1d_arm_k1() {
    run_conv1d_arm(3, 4, 8, 1, 1, 0);
    run_conv1d_arm(  64,  256, 112, 1, 1, 0);
    run_conv1d_arm( 128,  512,  56, 1, 1, 0);
    run_conv1d_arm( 256, 1024,  28, 1, 1, 0);
    run_conv1d_arm( 512, 2048,  14, 1, 1, 0);
    run_conv1d_arm(  64,   64, 224, 1, 1, 0);
    run_conv1d_arm(1024, 1024,  14, 1, 1, 0);
}

void perf_conv1d_arm_bias() {
    run_conv1d_arm(4, 8, 8, 3, 1, 1, 1, true);
    run_conv1d_arm(  64,  128, 112, 3, 1, 1, 1, true);
    run_conv1d_arm( 128,  256,  56, 3, 1, 1, 1, true);
    run_conv1d_arm( 256,  512,  28, 3, 1, 1, 1, true);
    run_conv1d_arm(   3,   64, 224, 3, 1, 1, 1, true);
    run_conv1d_arm( 512, 1024,  14, 3, 1, 1, 1, true);
}
