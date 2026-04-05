cmake_minimum_required(VERSION 3.10)
project(ncnn_arm_kernel_tests CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

# ARM-specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(-march=armv8.2-a+fp16+dotprod)
endif()

# Include paths
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ── ncnn framework stub library ───────────────────────────────────────────────
set(FRAMEWORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../framework)
add_library(ncnn_stub STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_framework_stub.cpp
    ${FRAMEWORK_DIR}/expression.cpp
    ${FRAMEWORK_DIR}/mat.cpp
)
target_include_directories(ncnn_stub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${FRAMEWORK_DIR}
)

# ── ARM kernel implementation libraries ───────────────────────────────────────

add_library(norm_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/layernorm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/rmsnorm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/batchnorm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/instancenorm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/groupnorm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../norm/lrn_arm.cpp
)
target_include_directories(norm_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Base class cpp files live in c-partially-optimized (shared implementations)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../c-partially-optimized)

# Base layernorm/rmsnorm etc. (the non-arm base classes)
add_library(norm_base STATIC
    ${BASE_DIR}/norm/layernorm.cpp
    ${BASE_DIR}/norm/rmsnorm.cpp
    ${BASE_DIR}/norm/batchnorm.cpp
    ${BASE_DIR}/norm/instancenorm.cpp
    ${BASE_DIR}/norm/groupnorm.cpp
    ${BASE_DIR}/norm/lrn.cpp
)
target_include_directories(norm_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Reduction (softmax, pooling)
add_library(reduction_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/softmax_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../reduction/pooling_arm.cpp
)
target_include_directories(reduction_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(reduction_base STATIC
    ${BASE_DIR}/reduction/softmax.cpp
    ${BASE_DIR}/reduction/pooling.cpp
)
target_include_directories(reduction_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Gemm (matmul_arm, innerproduct_arm, gemm_arm)
add_library(gemm_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/gemm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/innerproduct_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/matmul_arm.cpp
)
target_include_directories(gemm_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(gemm_base STATIC
    ${BASE_DIR}/gemm/gemm.cpp
    ${BASE_DIR}/gemm/innerproduct.cpp
    ${BASE_DIR}/gemm/matmul.cpp
)
target_include_directories(gemm_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Attention (multiheadattention_arm + base class from c-partially-optimized)
add_library(attention_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../attention/multiheadattention_arm.cpp
    ${BASE_DIR}/attention/multiheadattention.cpp
)
target_include_directories(attention_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Conv (convolution_arm, convolutiondepthwise_arm, deconvolution_arm)
add_library(conv_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolution_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/convolutiondepthwise_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolution_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../conv/deconvolutiondepthwise_arm.cpp
)
target_include_directories(conv_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(conv_base STATIC
    ${BASE_DIR}/conv/convolution.cpp
    ${BASE_DIR}/conv/convolutiondepthwise.cpp
    ${BASE_DIR}/conv/deconvolution.cpp
    ${BASE_DIR}/conv/deconvolutiondepthwise.cpp
    ${BASE_DIR}/activation/relu.cpp
    ${BASE_DIR}/activation/sigmoid.cpp
    ${BASE_DIR}/activation/tanh.cpp
    ${BASE_DIR}/activation/clip.cpp
    ${BASE_DIR}/activation/hardswish.cpp
    ${BASE_DIR}/activation/hardsigmoid.cpp
    ${BASE_DIR}/activation/gelu.cpp
    ${BASE_DIR}/activation/prelu.cpp
    ${BASE_DIR}/activation/selu.cpp
    ${BASE_DIR}/activation/mish.cpp
    ${BASE_DIR}/activation/swish.cpp
)
target_include_directories(conv_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Quant (quantize_arm, dequantize_arm)
add_library(quant_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quantize_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/dequantize_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/requantize_arm.cpp
)
target_include_directories(quant_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(quant_base STATIC
    ${BASE_DIR}/quant/quantize.cpp
    ${BASE_DIR}/quant/dequantize.cpp
    ${BASE_DIR}/quant/requantize.cpp
)
target_include_directories(quant_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Recurrent (lstm_arm, gru_arm, rnn_arm)
add_library(recurrent_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/lstm_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/gru_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../recurrent/rnn_arm.cpp
)
target_include_directories(recurrent_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(recurrent_base STATIC
    ${BASE_DIR}/recurrent/lstm.cpp
    ${BASE_DIR}/recurrent/gru.cpp
    ${BASE_DIR}/recurrent/rnn.cpp
)
target_include_directories(recurrent_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# Tensor ops (binaryop_arm, unaryop_arm, flatten_arm, concat_arm)
add_library(tensor_impl STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/binaryop_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/unaryop_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/flatten_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/concat_arm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensor/packing_arm.cpp
)
target_include_directories(tensor_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(tensor_base STATIC
    ${BASE_DIR}/tensor/binaryop.cpp
    ${BASE_DIR}/tensor/unaryop.cpp
    ${BASE_DIR}/tensor/flatten.cpp
    ${BASE_DIR}/tensor/concat.cpp
    ${BASE_DIR}/tensor/padding.cpp
    ${BASE_DIR}/tensor/packing.cpp
)
target_include_directories(tensor_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${BASE_DIR}
)

# create_layer_cpu_arm: provides create_layer_cpu(int) returning Gemm_arm
add_library(layer_cpu_arm STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/create_layer_cpu_arm.cpp
)
target_include_directories(layer_cpu_arm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(layer_cpu_arm gemm_impl gemm_base ncnn_stub)

# ── Helper macro ──────────────────────────────────────────────────────────────
macro(add_ncnn_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${ARGN} ncnn_stub m)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# ── Test targets ──────────────────────────────────────────────────────────────
enable_testing()

add_ncnn_test(test_norm       norm_impl      norm_base)
add_ncnn_test(test_reduction  reduction_impl reduction_base)
add_ncnn_test(test_quant      quant_impl     quant_base)
add_ncnn_test(test_recurrent  recurrent_impl recurrent_base)
add_ncnn_test(test_tensor     tensor_impl    tensor_base)
add_ncnn_test(test_gemm       gemm_impl      gemm_base      layer_cpu_arm)
add_ncnn_test(test_conv       conv_impl      conv_base      gemm_impl gemm_base layer_cpu_arm)
add_ncnn_test(test_attention  attention_impl gemm_impl      gemm_base layer_cpu_arm reduction_impl reduction_base)

# ── Summary target ────────────────────────────────────────────────────────────
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS
        test_norm test_reduction test_quant test_recurrent
        test_tensor test_gemm test_conv test_attention
    COMMENT "Running all ARM kernel tests"
)
