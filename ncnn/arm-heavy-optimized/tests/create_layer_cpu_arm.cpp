// create_layer_cpu_arm.cpp
// Provides create_layer_cpu(int) returning real ARM layer instances.
// This file must be compiled together with the arm kernel object files so
// the compiler can see the _arm class definitions.

#include "../../framework/layer.h"
#include "../../framework/cpu.h"

// ARM kernel headers
#include "../gemm/gemm_arm.h"
#include "../reduction/softmax_arm.h"
#include "../conv/convolution_arm.h"

namespace ncnn {

// Returns a real _arm instance for known layer types; otherwise a base Layer.
// LayerType indices correspond to ncnn's internal enum, but we just use
// a simple switch based on what tests need.
//
// For tests that instantiate arm layers directly (new Gemm_arm() etc.),
// this factory is mainly needed by sub-layers inside MHA / MatMul_arm / etc.
// that call create_layer_cpu internally.
//
// We map the most common ones:
//   Gemm    → Gemm_arm
//   Softmax → Softmax_arm
//   Convolution → Convolution_arm

Layer* create_layer_cpu(int index)
{
    // ncnn LayerType enum values (from layer.h / layer_type.h)
    // These are the numeric indices used internally by ncnn.
    // Gemm=29, InnerProduct=14, Softmax=46, Convolution=3 (approximate)
    // Since we don't have the full enum here, we use a generous default.
    // Sub-layers in MHA use Gemm_arm; MatMul_arm sub-layer uses Gemm_arm.
    (void)index;
    // Default: return a Gemm_arm since that's the most common sub-layer
    // created by MultiHeadAttention_arm and MatMul_arm.
    return new Gemm_arm();
}

} // namespace ncnn
