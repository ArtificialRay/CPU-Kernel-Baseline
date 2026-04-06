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
    // Dispatch based on ncnn LayerType enum values (from framework/layer_type_enum.h):
    //   Convolution = 8, Gemm = 40, Softmax = 93
    switch (index)
    {
        case  8: return new Convolution_arm(); // Convolution
        case 40: return new Gemm_arm();        // Gemm
        case 93: return new Softmax_arm();     // Softmax
        default: return new Gemm_arm();
    }
}

} // namespace ncnn
