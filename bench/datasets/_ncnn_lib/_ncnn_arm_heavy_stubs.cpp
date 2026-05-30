// bench/datasets/_ncnn_lib/_ncnn_arm_heavy_stubs.cpp
//
// Replaces _ncnn_unused_stubs.cpp when solution.spec.dependencies contains
// "ncnn_arm_heavy". The difference:
//
//   _ncnn_unused_stubs.cpp (Phase 1, scalar-only):
//       create_layer() aborts — no Solution should ever exercise mat.cpp's
//       Layer-factory paths (e.g. copy_make_border with pad > 0).
//
//   _ncnn_arm_heavy_stubs.cpp (Phase 2+, baseline-ncnn-arm):
//       create_layer(Padding) returns a real Padding (so make_padding /
//       copy_make_border work for pad > 0 workloads).
//       create_layer(Convolution) returns Convolution_arm (so
//       Convolution_arm::forwardDilation_arm's sub-layer is itself, mirroring
//       conv_arm_layer_hook.cpp in the legacy CMake build).
//       All other LayerType indices return a no-op base Layer (whose forward
//       returns -1, signalling failure — these paths shouldn't fire in Phase 2).
//
// Also re-supplies Layer:: base-class symbols + cpu helpers that the rest of
// the .so needs at link time (framework/layer.cpp itself is unusable here
// because it #includes the CMake-generated layer_declaration.h which isn't
// in our slim checkout). All trivial defaults — kernels that need real
// versions live in libncnn_arm_heavy.a.

#include "convolution_arm.h"
#include "padding.h"
#include "layer.h"
#include "layer_type.h"

#include <cstdio>

namespace ncnn {

// ── Layer base ───────────────────────────────────────────────────────────────
// Initialize every public bool/int field in framework/layer.h so Convolution_arm
// doesn't read garbage during create_pipeline / forward. Earlier attempts to
// only set a subset caused a segfault on Graviton3.
// Body-init (not initializer list) tolerates upstream field additions.
Layer::Layer() {
    one_blob_only           = false;
    support_inplace         = false;
    support_vulkan          = false;
    support_packing         = false;
    support_bf16_storage    = false;
    support_fp16_storage    = false;
    support_int8_storage    = false;
    support_tensor_storage  = false;
    support_vulkan_packing  = false;
    support_any_packing     = false;
    support_vulkan_any_packing = false;
    support_reserved_1 = false;
    support_reserved_2 = false;
    support_reserved_3 = false;
    support_reserved_4 = false;
    support_reserved_5 = false;
    support_reserved_6 = false;
    support_reserved_7 = false;
    support_reserved_8 = false;
    support_reserved_9 = false;
    featmask = 0;
    userdata = nullptr;
    typeindex = -1;
}
Layer::~Layer() {}
int Layer::load_param(const ParamDict& /*pd*/) { return 0; }
int Layer::load_model(const ModelBin& /*mb*/) { return 0; }
int Layer::create_pipeline(const Option& /*opt*/) { return 0; }
int Layer::destroy_pipeline(const Option& /*opt*/) { return 0; }
int Layer::forward(const Mat& /*bb*/, Mat& /*tb*/, const Option& /*opt*/) const { return -1; }
int Layer::forward(const std::vector<Mat>& /*bbs*/, std::vector<Mat>& /*tbs*/,
                   const Option& /*opt*/) const { return -1; }
int Layer::forward_inplace(Mat& /*b*/, const Option& /*opt*/) const { return -1; }
int Layer::forward_inplace(std::vector<Mat>& /*bs*/, const Option& /*opt*/) const { return -1; }

// ── Layer factory ────────────────────────────────────────────────────────────

static Layer* _make_layer(int index) {
    if (index == LayerType::Padding) return new Padding();
    if (index == LayerType::Convolution) return new Convolution_arm();
    return new Layer();
}

Layer* create_layer(int index)          { return _make_layer(index); }
Layer* create_layer_cpu(int index)      { return _make_layer(index); }
Layer* create_layer_naive(int index)    { return _make_layer(index); }
Layer* create_layer(const char*)        { return new Layer(); }
Layer* create_layer_cpu(const char*)    { return new Layer(); }
Layer* create_layer_naive(const char*)  { return new Layer(); }

} // namespace ncnn
