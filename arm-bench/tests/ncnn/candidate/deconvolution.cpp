#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolution.h"
// CANDIDATE_TESTCASE_START
void test_deconv_base_2x2_s2() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 3, 3, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 2, 4, 4, 4, 2, 2, 2, 2);
    // U-Net-style 2×2 s=2 upsampling
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   32,   64, 56, 56, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   64,  128, 28, 28, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  128,  256, 14, 14, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  256,  512,  7,  7, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,    3,   32, 112, 112, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1024, 1024,  7,  7, 2, 2, 2, 2);
}

void test_deconv_base_3x3_s1() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 1, 1);
    // Larger 3×3 stride=1 (output = in+2)
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 56, 56, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 28, 28, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 256,  512, 14, 14, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 112, 112, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 1, 1);
}

void test_deconv_base_bias() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 2, 4, 4, 4, 3, 3, 1, 1, true);
    // Larger 3×3 s=1 with bias
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 56, 56, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 28, 28, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 256,  512, 14, 14, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 112, 112, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
