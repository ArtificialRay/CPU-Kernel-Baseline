// solutions/ncnn/_harness/conv2d.h
//
// Calling-convention contract for every conv2d Solution in the "ncnn" dataset.
//
// Every Solution under solutions/ncnn/<author>/conv2d/ must define the
// `ncnn::convolution_kernel` symbol with EXACTLY this signature. The shared
// shim `armbench_entry_conv2d` in conv2d.cpp constructs a Convolution layer,
// pads input via copy_make_border, allocates output, then calls into this
// symbol. The Solution author writes the kernel body; everything else
// (padding, allocation, layer setup) is the harness's job.
//
// This matches today's starter/ncnn/candidate/convolution.h:57-63 signature
// byte-for-byte — so the existing baseline_arm kernel and any agent kernel
// can be migrated by just wrapping their .cpp into a Solution JSON.

#ifndef ARMBENCH_NCNN_HARNESS_CONV2D_H
#define ARMBENCH_NCNN_HARNESS_CONV2D_H

#include "mat.h"
#include "option.h"

namespace ncnn {

// The solution-supplied kernel.
//
//   bottom_blob:   already-padded input, dims=3, layout (c, h, w), float32
//   top_blob:      pre-allocated output, dims=3, layout (out_c, out_h, out_w), float32
//   weight_data:   weights as a flat 1D Mat of size out_c*in_c*kh*kw
//   bias_data:     bias as a 1D Mat of size out_c, or empty Mat if no bias
//   activation_params: per-activation params Mat (empty for type=0/1=relu)
//   opt:           ncnn options (num_threads=1 enforced by bench/runner.py)
//
// Returns 0 on success, non-zero on error.
int convolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                       const Mat& weight_data, const Mat& bias_data,
                       int kernel_w, int kernel_h,
                       int stride_w, int stride_h,
                       int dilation_w, int dilation_h,
                       int activation_type, const Mat& activation_params,
                       const Option& opt);

} // namespace ncnn

#endif // ARMBENCH_NCNN_HARNESS_CONV2D_H
