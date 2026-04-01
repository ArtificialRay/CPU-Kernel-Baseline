cmake_minimum_required(VERSION 3.10)
project(ncnn_kernel_tests CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

# Include paths:
#  - tests/     : for test_utils.h, ncnn_helpers.h
#  - ../ (c-partially-optimized/) : so that "../framework/layer.h" resolves
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ── ncnn framework stub library ───────────────────────────────────────────────
# Provides the non-inline ncnn symbols (Mat::create, Layer, ParamDict, Option,
# ModelBin, cpu helpers) without requiring the full ncnn CMake build.
# Also includes expression.cpp for kernels that use dynamic shape expressions.

add_library(ncnn_stub STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_framework_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../framework/expression.cpp
)
target_include_directories(ncnn_stub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ── Kernel implementation libraries ──────────────────────────────────────────

add_library(attention_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../attention/sdpa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../attention/multiheadattention.cpp
)
target_include_directories(attention_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(activation_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/relu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/sigmoid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/gelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/clip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/tanh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/absval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/swish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/hardswish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/hardsigmoid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/elu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/selu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/mish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/softplus.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/power.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/threshold.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/shrink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/bnll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/celu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/erf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../activation/glu.cpp
)
target_include_directories(activation_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(norm_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/layernorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/rmsnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/batchnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/instancenorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/groupnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/lrn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/mvn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/normalize.cpp
)
target_include_directories(norm_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(gemm_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/innerproduct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/einsum.cpp
)
target_include_directories(gemm_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(conv_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolutiondepthwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolutiondepthwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolution1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolutiondepthwise1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolution1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolutiondepthwise1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolution3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolutiondepthwise3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolution3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolutiondepthwise3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deformableconv2d.cpp
)
target_include_directories(conv_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(reduction_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/reduction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/pooling1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/pooling3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/cumulativesum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/spp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/statisticspooling.cpp
)
target_include_directories(reduction_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(recurrent_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/lstm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/gru.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/rnn.cpp
)
target_include_directories(recurrent_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(tensor_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/binaryop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/padding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/unaryop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/permute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/flip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/crop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/dropout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/bias.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/scale.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/expanddims.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/shufflechannel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/pixelshuffle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/interp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/packing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/copyto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/deepcopy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/diag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/fold.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/reorg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/unfold.cpp
)
target_include_directories(tensor_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(quant_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/dequantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/requantize.cpp
)
target_include_directories(quant_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ── Helper macro for tests that link the actual ncnn implementations ──────────
macro(add_ncnn_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${ARGN} ncnn_stub m)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# ── Test targets ─────────────────────────────────────────────────────────────
enable_testing()

add_ncnn_test(test_activation activation_impl)
add_ncnn_test(test_conv       conv_impl)
add_ncnn_test(test_gemm       gemm_impl)
add_ncnn_test(test_norm       norm_impl)
add_ncnn_test(test_recurrent  recurrent_impl)
add_ncnn_test(test_reduction  reduction_impl)
add_ncnn_test(test_tensor     tensor_impl)
add_ncnn_test(test_quant      quant_impl)
add_ncnn_test(test_attention  attention_impl)

# test_other has no kernel deps beyond stub
add_executable(test_other test_other.cpp)
target_link_libraries(test_other ncnn_stub m)
add_test(NAME test_other COMMAND test_other)

# ── Summary target ────────────────────────────────────────────────────────────
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS
        test_activation test_conv test_gemm test_norm
        test_attention test_recurrent test_reduction
        test_tensor test_quant test_other
    COMMENT "Running all kernel tests"
)
