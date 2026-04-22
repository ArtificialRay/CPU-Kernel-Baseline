#include "starter/ncnn/baseline/convolution_arm.h"

// BASELINE_TESTCASE_START
// ── Convolution_arm ───────────────────────────────────────────────
void perf_conv_arm_1x1_s1() {
    run_conv2d_arm(1, 1, 4, 4, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(3, 4, 5, 5, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(8, 8, 7, 7, 1, 1, 1, 1, 0, 0);
    // ResNet bottleneck 1×1 channel expansion
    run_conv2d_arm( 64,  256, 56, 56, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(128,  512, 28, 28, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(256, 1024, 14, 14, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(512, 2048,  7,  7, 1, 1, 1, 1, 0, 0);
    // Large in_w (224)
    run_conv2d_arm( 64,   64, 224, 224, 1, 1, 1, 1, 0, 0);
    // Large Cin (1024)
    run_conv2d_arm(1024, 1024, 14, 14, 1, 1, 1, 1, 0, 0);
}

void perf_conv_arm_3x3_s1() {
    run_conv2d_arm(1, 1, 5, 5, 3, 3, 1, 1, 0, 0);
    run_conv2d_arm(3, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    run_conv2d_arm(4, 8, 10, 10, 3, 3, 1, 1, 1, 1);
}

void perf_conv_arm_3x3_s2() {
    run_conv2d_arm(1, 1, 8, 8, 3, 3, 2, 2, 0, 0);
    run_conv2d_arm(4, 8, 12, 12, 3, 3, 2, 2, 1, 1);
    // ResNet-style 3×3 stride-2 downsampling stages
    run_conv2d_arm( 64,  128, 112, 112, 3, 3, 2, 2, 1, 1);
    run_conv2d_arm(128,  256,  56,  56, 3, 3, 2, 2, 1, 1);
    run_conv2d_arm(256,  512,  28,  28, 3, 3, 2, 2, 1, 1);
    run_conv2d_arm(512, 1024,  14,  14, 3, 3, 2, 2, 1, 1);
}

void perf_conv_base_dilation() {
    run_conv2d_arm(2, 4, 8, 8, 3, 3, 1, 1, 0, 0, 2, 2);
    // Atrous 3×3 dilation=2 pad=2 (same-size output) — DeepLab-style stages
    run_conv2d_arm( 64, 128, 56, 56, 3, 3, 1, 1, 2, 2, 2, 2);
    run_conv2d_arm(128, 256, 28, 28, 3, 3, 1, 1, 2, 2, 2, 2);
    run_conv2d_arm(256, 256, 14, 14, 3, 3, 1, 1, 2, 2, 2, 2);
    run_conv2d_arm(512, 512,  7,  7, 3, 3, 1, 1, 2, 2, 2, 2);
    // Large in_w (224) — small C, dilation=2
    run_conv2d_arm(  3,  32, 224, 224, 3, 3, 1, 1, 2, 2, 2, 2);
    // Larger dilation=4 — moderate C/spatial
    run_conv2d_arm( 32,  64, 112, 112, 3, 3, 1, 1, 4, 4, 4, 4);
}

void perf_conv_arm_5x5() {
    run_conv2d_arm(3, 4, 10, 10, 5, 5, 1, 1, 2, 2);
    // Larger 5×5 pad=2 stride=1
    run_conv2d_arm( 32,  64, 32, 32, 5, 5, 1, 1, 2, 2);
    run_conv2d_arm( 64, 128, 28, 28, 5, 5, 1, 1, 2, 2);
    run_conv2d_arm(128, 128, 14, 14, 5, 5, 1, 1, 2, 2);
    // Large in_w (224) — small C
    run_conv2d_arm(  3,  32, 224, 224, 5, 5, 1, 1, 2, 2);
    // Large Cin (256/512) — small spatial
    run_conv2d_arm(256, 256, 14, 14, 5, 5, 1, 1, 2, 2);
    run_conv2d_arm(512, 512,  7,  7, 5, 5, 1, 1, 2, 2);
}

void test_conv_arm_bias() {
    run_conv2d_arm(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
    // ResNet-style 3×3 pad=1 stride=1 stages with bias
    run_conv2d_arm( 64,  128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_conv2d_arm(128,  256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, true);
    run_conv2d_arm(256,  512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, true);
    // Large in_w (224) — small C
    run_conv2d_arm(  3,   64, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, true);
    // Large Cin (1024) — small spatial
    run_conv2d_arm(512, 1024,  7,  7, 3, 3, 1, 1, 1, 1, 1, 1, true);
}

// ResNet-style 7×7 stride-2 stem (224×224×3 → 112×112×64)
void perf_conv_arm_7x7_s2() {
    run_conv2d_arm(3, 64, 224, 224, 7, 7, 2, 2, 3, 3);
}
