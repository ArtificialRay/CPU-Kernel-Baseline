// bench/datasets/_ncnn_lib/_ncnn_unused_stubs.cpp
//
// Stubs for ncnn symbols that mat.cpp references (transitively, via
// copy_make_border → ParamDict → Layer factory) but that our harness
// never reaches at runtime.
//
// Why this exists: the full ncnn build provides these via CMake-generated
// `layer_registry.h.in` + the `mapped/<layer>/<layer>.cpp` tree, neither
// of which is in our slim checkout. Rather than vendoring all of that,
// we link these no-op stubs to satisfy the linker. The stubs abort if
// ever called — that would mean someone tried to use ncnn Layer/Padding
// machinery from a harness path that's outside Phase 1's scope.
//
// When that day comes (multi-threaded kernels, padded inputs that hit
// copy_make_border, real Layer dispatch in baseline-ncnn-arm), wire in
// the real ncnn build instead of this file.

#include "layer.h"
#include <cstdio>
#include <cstdlib>

namespace ncnn {

[[noreturn]] static void _stub_abort(const char* sym)
{
    std::fprintf(stderr,
        "armbench: ncnn symbol '%s' was called but is only stubbed in the "
        "Phase-1 build. This means your Solution exercised a code path "
        "(probably padding > 0 or full Layer dispatch) that needs the full "
        "ncnn framework. See bench/datasets/_ncnn_lib/_ncnn_unused_stubs.cpp.\n",
        sym);
    std::abort();
}

// Used by mat.cpp's copy_make_border (Padding layer factory)
Layer* create_layer(int /*index*/) { _stub_abort("create_layer(int)"); }
Layer* create_layer(const char* /*type*/) { _stub_abort("create_layer(const char*)"); }
Layer* create_layer_cpu(int /*index*/) { _stub_abort("create_layer_cpu(int)"); }
Layer* create_layer_cpu(const char* /*type*/) { _stub_abort("create_layer_cpu(const char*)"); }

// Layer base — used transitively. Provide trivial defaults so that, if some
// path constructs a Layer (we hope not), at least it doesn't crash inside the
// vtable. These are no-ops.
// Set members in body (not initializer list) so this compiles against minor
// ncnn versions that have slightly different Layer member sets.
Layer::Layer() {
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
    support_packing = false;
    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;
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

} // namespace ncnn
