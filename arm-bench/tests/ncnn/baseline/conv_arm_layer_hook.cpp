// Strong override of test_make_layer_cpu_hook (weak default in
// ncnn_framework_stub.cpp) so Convolution_arm::forwardDilation_arm's
// sub-layer is a real Convolution_arm instead of a no-op base Layer.
//
// Linked into test_baseline_conv ONLY — other test binaries continue to
// use the weak default that returns 0.

#include "starter/ncnn/baseline/convolution_arm.h"
#include "layer_type.h"

namespace ncnn {

Layer* test_make_layer_cpu_hook(int index)
{
    if (index == LayerType::Convolution) return new Convolution_arm();
    return 0;
}

} // namespace ncnn
