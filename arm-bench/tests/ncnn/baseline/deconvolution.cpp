#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolution_arm.h"

// BASELINE_TESTCASE_START
// ── Deconvolution_arm ─────────────────────────────────────────────
// Only the four (kernel, stride) combinations that ncnn's
// arm-heavy-optimized deconvolution_arm.cpp specializes with NEON kernels:
//   deconv3x3s1_neon / deconv3x3s2_neon / deconv4x4s1_neon / deconv4x4s2_neon
// (see arm-heavy-optimized/conv/deconvolution_arm.cpp:627-665, gated on
//  elempack==1 && out_elempack==1 && dilation==1).
// Other (k, s) configs fall through to the scalar omp loop — same algorithm
// as the candidate, so they wouldn't exercise the optimized path.
void test_deconv_arm_3x3_s1() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 1, 1);
    // ResNet/U-Net-style 3×3 s=1 (output = in+2)
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64,  128, 56, 56, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 128,  256, 28, 28, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 256,  512, 14, 14, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,   3,   32, 112, 112, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 1, 1);
}

void test_deconv_arm_3x3_s2() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 2, 2);
    // U-Net-style 3×3 s=2 upsampling (output = 2·in + 1)
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64,  128, 28, 28, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 128,  256, 14, 14, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 256,  512,  7,  7, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,   3,   32, 56, 56, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 2, 2);
}

void test_deconv_arm_4x4_s1() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 4, 4, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 3, 4, 5, 5, 4, 4, 1, 1);
    // 4×4 s=1 (output = in+3)
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64,  128, 56, 56, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 128,  256, 28, 28, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 256,  512, 14, 14, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,   3,   32, 112, 112, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 512, 1024,  7,  7, 4, 4, 1, 1);
}

void test_deconv_arm_4x4_s2() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 3, 3, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 2, 4, 4, 4, 4, 4, 2, 2);
    // 4×4 s=2 upsampling (output = 2·in + 2) — common in GAN/segmentation upsampling
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  32,   64, 56, 56, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64,  128, 28, 28, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 128,  256, 14, 14, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,   3,   32, 56, 56, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 512,  512,  7,  7, 4, 4, 2, 2);
}

void test_deconv_arm_bias() {
    // One bias case per (k, s) variant to verify bias on every NEON path.
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64, 128, 56, 56, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64, 128, 28, 28, 3, 3, 2, 2, true);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64, 128, 56, 56, 4, 4, 1, 1, true);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d,  64, 128, 28, 28, 4, 4, 2, 2, true);
}
// BASELINE_TESTCASE_END
