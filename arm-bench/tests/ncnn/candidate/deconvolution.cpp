#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolution.h"
// CANDIDATE_TESTCASE_START
// Mirrors tests/ncnn/baseline/deconvolution.cpp shape coverage so candidate and
// baseline are exercised on identical (k, s) variants — the four ones that
// Deconvolution_arm specializes with NEON kernels (3×3 s1/s2, 4×4 s1/s2).
void test_deconv_base_3x3_s1() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 56, 56, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 28, 28, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 256,  512, 14, 14, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 112, 112, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 1, 1);
}

void test_deconv_base_3x3_s2() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 28, 28, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 14, 14, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 256,  512,  7,  7, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 56, 56, 3, 3, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512, 1024,  7,  7, 3, 3, 2, 2);
}

void test_deconv_base_4x4_s1() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 4, 4, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 3, 4, 5, 5, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 56, 56, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 28, 28, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 256,  512, 14, 14, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 112, 112, 4, 4, 1, 1);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512, 1024,  7,  7, 4, 4, 1, 1);
}

void test_deconv_base_4x4_s2() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 1, 1, 3, 3, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 2, 4, 4, 4, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  32,   64, 56, 56, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64,  128, 28, 28, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 128,  256, 14, 14, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,   3,   32, 56, 56, 4, 4, 2, 2);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, 512,  512,  7,  7, 4, 4, 2, 2);
}

void test_deconv_base_bias() {
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64, 128, 56, 56, 3, 3, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64, 128, 28, 28, 3, 3, 2, 2, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64, 128, 56, 56, 4, 4, 1, 1, true);
    EXPECT_MATCH(run_deconv2d, run_ref_deconv2d,  64, 128, 28, 28, 4, 4, 2, 2, true);
}
// CANDIDATE_TESTCASE_END
