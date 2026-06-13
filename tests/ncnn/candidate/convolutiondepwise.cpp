#include "test_utils.h"
#include "starter/ncnn/candidate/convolutiondepthwise.h"
// CANDIDATE_TESTCASE_START
void test_dw_base_3x3() {
    // MobileNet-style depthwise 3×3 (pad=1 stride=1)
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   64, 112, 112, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  128,  56,  56, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  256,  28,  28, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  512,  14,  14, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 1024,   7,   7, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   32, 224, 224, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 2048,  14,  14, 3, 3, 1, 1, 1, 1);
}

void test_dw_base_5x5() {
    // Larger 5×5 (pad=2 stride=1)
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   64,  56,  56, 5, 5, 1, 1, 2, 2);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  128,  28,  28, 5, 5, 1, 1, 2, 2);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  256,  14,  14, 5, 5, 1, 1, 2, 2);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  512,   7,   7, 5, 5, 1, 1, 2, 2);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   32, 224, 224, 5, 5, 1, 1, 2, 2);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 1024,  14,  14, 5, 5, 1, 1, 2, 2);
}

void test_dw_base_bias() {
    // Depthwise 3×3 with bias at typical stages
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   64, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, true);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,  256,  28,  28, 3, 3, 1, 1, 1, 1, 1, 1, true);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 1024,   7,   7, 3, 3, 1, 1, 1, 1, 1, 1, true);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d,   32, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, true);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 2048,  14,  14, 3, 3, 1, 1, 1, 1, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
