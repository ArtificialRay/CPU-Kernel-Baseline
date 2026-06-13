#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolutiondepthwise.h"
// CANDIDATE_TESTCASE_START
void test_dw_deconv_base_2x2_s2() {
    // Depthwise 2×2 s=2 upsampling at typical stages
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,   64,  56, 56, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,  128,  28, 28, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,  256,  14, 14, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,  512,   7,  7, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,   32, 112, 112, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 1024,  14, 14, 2, 2, 2, 2);
}

void test_dw_deconv_base_3x3_s1() {
    // Depthwise 3×3 s=1 at typical stages
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,   64,  56, 56, 3, 3, 1, 1);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,  128,  28, 28, 3, 3, 1, 1);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,  512,  14, 14, 3, 3, 1, 1);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d,   32, 224, 224, 3, 3, 1, 1);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 2048,   7,  7, 3, 3, 1, 1);
}
// CANDIDATE_TESTCASE_END
