// solutions/ncnn/_harness/conv2d.cpp
//
// Shared C-ABI entry point for the ncnn dataset's conv2d kernel-type.
// bench/compile.py compiles this together with the Solution's kernel.cpp
// (which defines ncnn::convolution_kernel) into one .so. bench/runner.py
// then dlopens the .so and calls armbench_entry_conv2d via ctypes.
//
// Responsibilities of this shim (taking the burden off Solution authors):
//   1. unwrap opaque void* into ncnn::Mat& / ncnn::Option& references
//   2. pad bottom_blob using ncnn::copy_make_border (the ncnn equivalent of
//      what Convolution::make_padding does in starter/ncnn/candidate/convolution.h)
//   3. allocate top_blob to the computed output shape
//   4. call into ncnn::convolution_kernel (provided by the Solution)
//
// Notes:
//   - top_mat is passed in as an empty ncnn::Mat constructed by the Python
//     side via _mat_factory.armbench_ncnn_mat_create_empty(). We .create()
//     it here so the Solution's kernel always sees a pre-allocated output.
//   - The output channel count must be derivable from weight shape:
//     out_c = weight.total() / (in_c * kw * kh). We pass it through scalar
//     args (out_c) rather than re-deriving to avoid surprises.

#include "conv2d.h"
#include "mat.h"
#include "option.h"

#ifndef BORDER_CONSTANT
#define BORDER_CONSTANT 0
#endif

extern "C" {

int armbench_entry_conv2d(
    // tensors (opaque ncnn::Mat*)
    void* bottom_mat_v,        // const ncnn::Mat*   — original (un-padded) input
    void* top_mat_v,           // ncnn::Mat*         — empty Mat; we create() output here
    void* weight_mat_v,        // const ncnn::Mat*   — flat 1D Mat
    void* bias_mat_v,          // const ncnn::Mat*   — flat 1D Mat or empty
    void* activation_params_v, // const ncnn::Mat*   — empty for type=0/1
    void* opt_v,               // const ncnn::Option*
    // kernel params (caller knows these from Definition.const_axes + Workload.scalar_inputs)
    int out_c,
    int kernel_w, int kernel_h,
    int stride_w, int stride_h,
    int dilation_w, int dilation_h,
    int pad_left, int pad_top,
    int activation_type)
{
    const auto& bottom    = *reinterpret_cast<const ncnn::Mat*>(bottom_mat_v);
    auto&       top       = *reinterpret_cast<ncnn::Mat*>(top_mat_v);
    const auto& weight    = *reinterpret_cast<const ncnn::Mat*>(weight_mat_v);
    const auto& bias      = *reinterpret_cast<const ncnn::Mat*>(bias_mat_v);
    const auto& act_par   = *reinterpret_cast<const ncnn::Mat*>(activation_params_v);
    const auto& opt       = *reinterpret_cast<const ncnn::Option*>(opt_v);

    if (bottom.empty()) return -100;

    // 1. Pad → bordered (matches Convolution::make_padding in convolution.h:65-110)
    ncnn::Mat bordered;
    if (pad_left > 0 || pad_top > 0) {
        ncnn::Option pad_opt = opt;
        pad_opt.blob_allocator = opt.workspace_allocator;
        ncnn::copy_make_border(bottom, bordered,
                               pad_top, pad_top, pad_left, pad_left,
                               BORDER_CONSTANT, 0.f, pad_opt);
        if (bordered.empty()) return -100;
    } else {
        bordered = bottom;
    }

    // 2. Compute output dims (matches convolution.h:126-127)
    const int ext_kw = dilation_w * (kernel_w - 1) + 1;
    const int ext_kh = dilation_h * (kernel_h - 1) + 1;
    const int outw   = (bordered.w - ext_kw) / stride_w + 1;
    const int outh   = (bordered.h - ext_kh) / stride_h + 1;

    // 3. Allocate top (matches convolution.h:129)
    top.create(outw, outh, out_c, (size_t)4u, opt.blob_allocator);
    if (top.empty()) return -100;

    // 4. Dispatch to the Solution-supplied kernel
    return ncnn::convolution_kernel(
        bordered, top, weight, bias,
        kernel_w, kernel_h,
        stride_w, stride_h,
        dilation_w, dilation_h,
        activation_type, act_par,
        opt);
}

} // extern "C"
