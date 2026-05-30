// bench/compile/builders/candidate_harness/conv2d.h
//
// Raw-`float*` calling-convention contract for candidate conv2d kernels.
//
// Candidates are built by CandidateBuilder with NO ncnn dependency: no
// ncnn::Mat, no ncnn::Option, no framework sources. A candidate sees bare
// float buffers and explicit shape ints. This is the entry the runner binds
// via ctypes (`armbench_entry_conv2d`); the matching ctypes argtypes live in
// bench/datasets/raw.py SIGNATURES["conv2d"] — edit one, edit the other.
//
// Layout conventions (match bench/runtime/inputs.py + the PyTorch reference):
//   input:   NCHW, contiguous, UN-padded. The kernel applies implicit zero
//            padding using pad_top/pad_left (exactly like ref_conv2d), it is
//            NOT pre-padded by a harness.
//   output:  N x Cout x H_out x W_out, contiguous, caller-allocated.
//            H_out = (H + 2*pad_top  - (Dh*(Kh-1)+1)) / Sh + 1
//            W_out = (W + 2*pad_left - (Dw*(Kw-1)+1)) / Sw + 1
//   weight:  Cout x Cin x Kh x Kw, contiguous.
//   bias:    length Cout, or NULL when there is no bias. (The conv2d reference
//            never adds bias, so correctness does not depend on it.)
//   act:     activation_type 0=none, 1=relu; act_params is optional (may be
//            NULL) with n_act entries.
//
// Returns 0 on success, non-zero on error.

#ifndef ARMBENCH_CANDIDATE_HARNESS_CONV2D_H
#define ARMBENCH_CANDIDATE_HARNESS_CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif

// The C-ABI symbol the runner binds. A candidate may implement this directly
// (entry_point: "kernel.cpp::armbench_entry_conv2d"), in which case the
// forwarder conv2d.cpp is NOT compiled.
int armbench_entry_conv2d(
    const float* input, float* output,
    const float* weight, const float* bias,   // bias may be NULL
    int N, int Cin, int H, int W,
    int Cout, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw,
    int pad_top, int pad_left,
    int activation_type, const float* act_params, int n_act);

// Alternate author style: implement just the kernel body under this fixed
// name (entry_point: "kernel.cpp::armbench_conv2d_kernel"). CandidateBuilder
// then compiles the conv2d.cpp forwarder, which defines armbench_entry_conv2d
// and calls this symbol. Same argument list as armbench_entry_conv2d.
int armbench_conv2d_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int N, int Cin, int H, int W,
    int Cout, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw,
    int pad_top, int pad_left,
    int activation_type, const float* act_params, int n_act);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ARMBENCH_CANDIDATE_HARNESS_CONV2D_H
