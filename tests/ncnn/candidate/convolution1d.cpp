#include "test_utils.h"
#include "starter/ncnn/candidate/convolution1d.h"
// CANDIDATE_TESTCASE_START
void test_conv1d_base_k3() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 2, 4, 8, 3, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 4, 8, 16, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 8, 4, 12, 3, 2, 0);
    // ResNet-1D channel ladder (kernel=3 pad=1 stride=1)
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,   64,  128, 112, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  128,  256,  56, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  256,  512,  28, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  512, 1024,  14, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,    3,   64, 224, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 1024, 1024,  14, 3, 1, 1);
}

void test_conv1d_base_k1() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 3, 4, 8, 1, 1, 0);
    // ResNet-1D bottleneck 1×1 channel expansion
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,   64,  256, 112, 1, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  128,  512,  56, 1, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  256, 1024,  28, 1, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  512, 2048,  14, 1, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,   64,   64, 224, 1, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 1024, 1024,  14, 1, 1, 0);
}

void test_conv1d_base_bias() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 4, 8, 8, 3, 1, 1, 1, true);
    // ResNet-1D 3×1 with bias at typical stages
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,   64,  128, 112, 3, 1, 1, 1, true);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  128,  256,  56, 3, 1, 1, 1, true);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  256,  512,  28, 3, 1, 1, 1, true);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,    3,   64, 224, 3, 1, 1, 1, true);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d,  512, 1024,  14, 3, 1, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
