// bench/compile/builders/candidate_harness/conv2d.cpp
//
// Thin C-ABI forwarder for the alternate candidate author style: a kernel that
// implements only `armbench_conv2d_kernel` (the body) and lets this shim export
// the runner-facing `armbench_entry_conv2d`.
//
// CandidateBuilder compiles this file ONLY when the solution's entry symbol is
// NOT `armbench_entry_conv2d` (i.e. the kernel did not export the C-ABI entry
// itself). When the kernel exports `armbench_entry_conv2d` directly, this file
// is skipped to avoid a duplicate-symbol link error.

#include "conv2d.h"

extern "C" int armbench_entry_conv2d(
    const float* input, float* output,
    const float* weight, const float* bias,
    int N, int Cin, int H, int W,
    int Cout, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw,
    int pad_top, int pad_left,
    int activation_type, const float* act_params, int n_act)
{
    return armbench_conv2d_kernel(
        input, output, weight, bias,
        N, Cin, H, W,
        Cout, Kh, Kw, Sh, Sw, Dh, Dw,
        pad_top, pad_left,
        activation_type, act_params, n_act);
}
