cmake_minimum_required(VERSION 3.10)
project(ncnn_kernel_tests CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

# OpenMP — apply globally so all impl libraries get -fopenmp
find_package(OpenMP REQUIRED)
add_compile_options(${OpenMP_CXX_FLAGS})
link_libraries(OpenMP::OpenMP_CXX)

# ARM-specific flags (enables NEON + FP16 + dotprod intrinsics)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(-march=armv8.2-a+fp16+dotprod)
endif()

# ── Directory aliases ─────────────────────────────────────────────
set(ROOT_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(FRAMEWORK_DIR  ${ROOT_DIR}/framework)
set(MAPPED_DIR     ${ROOT_DIR}/mapped)
set(UNMAPPED_DIR   ${ROOT_DIR}/unmapped)
set(ARM_CONV_DIR   ${ROOT_DIR}/arm-heavy-optimized/conv)

# ── ncnn framework stub library ───────────────────────────────────
# Provides Mat, copy_make_border, flatten, Layer, Option, ParamDict,
# ModelBin, cpu helpers, and all layer factory stubs.
add_library(ncnn_stub STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_framework_stub.cpp
    ${FRAMEWORK_DIR}/mat.cpp
    ${FRAMEWORK_DIR}/expression.cpp
    ${MAPPED_DIR}/padding/padding.cpp
)
target_include_directories(ncnn_stub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ROOT_DIR}
    ${FRAMEWORK_DIR}
)

# ── Mapped base implementations ───────────────────────────────────
# These are the c-partially-optimized base classes from mapped/.
# Includes: mapped/common/fused_activation.h (via ../common/) and framework.

add_library(mapped_conv_base STATIC
    ${MAPPED_DIR}/convolution/convolution.cpp
    ${MAPPED_DIR}/convolutiondepthwise/convolutiondepthwise.cpp
    ${MAPPED_DIR}/deconvolution/deconvolution.cpp
    ${MAPPED_DIR}/deconvolutiondepthwise/deconvolutiondepthwise.cpp
    ${MAPPED_DIR}/convolution1d/convolution1d.cpp
)
target_include_directories(mapped_conv_base PUBLIC ${ROOT_DIR})

add_library(mapped_ip_base STATIC
    ${MAPPED_DIR}/innerproduct/innerproduct.cpp
    ${MAPPED_DIR}/requantize/requantize.cpp
)
target_include_directories(mapped_ip_base PUBLIC ${ROOT_DIR})

# ── Mapped ARM implementations ────────────────────────────────────
# These are the ARM-optimized versions. The conv ARM files include headers
# (convolution_packed.h, deconvolution_3x3.h, etc.) that live in
# arm-heavy-optimized/conv/, so we add that to the include path.

add_library(mapped_conv_arm STATIC
    ${MAPPED_DIR}/convolution/convolution_arm.cpp
    ${MAPPED_DIR}/convolutiondepthwise/convolutiondepthwise_arm.cpp
    ${MAPPED_DIR}/deconvolution/deconvolution_arm.cpp
    ${MAPPED_DIR}/deconvolutiondepthwise/deconvolutiondepthwise_arm.cpp
    ${MAPPED_DIR}/convolution1d/convolution1d_arm.cpp
)
target_include_directories(mapped_conv_arm PUBLIC
    ${ROOT_DIR}
    ${ARM_CONV_DIR}          # for convolution_packed.h, deconvolution_3x3.h, etc.
)

add_library(mapped_ip_arm STATIC
    ${MAPPED_DIR}/innerproduct/innerproduct_arm.cpp
    ${MAPPED_DIR}/requantize/requantize_arm.cpp
)
target_include_directories(mapped_ip_arm PUBLIC ${ROOT_DIR})

# ── Unmapped implementations ──────────────────────────────────────
# Plain c-partially-optimized kernels with no ARM variant.
# They use unmapped/common/fused_activation.h (via ../common/).

add_library(unmapped_conv_impl STATIC
    ${UNMAPPED_DIR}/convolution3d/convolution3d.cpp
    ${UNMAPPED_DIR}/convolutiondepthwise1d/convolutiondepthwise1d.cpp
    ${UNMAPPED_DIR}/convolutiondepthwise3d/convolutiondepthwise3d.cpp
    ${UNMAPPED_DIR}/deconvolution1d/deconvolution1d.cpp
    ${UNMAPPED_DIR}/deconvolution3d/deconvolution3d.cpp
    ${UNMAPPED_DIR}/deconvolutiondepthwise1d/deconvolutiondepthwise1d.cpp
    ${UNMAPPED_DIR}/deconvolutiondepthwise3d/deconvolutiondepthwise3d.cpp
    ${UNMAPPED_DIR}/deformableconv2d/deformableconv2d.cpp
)
target_include_directories(unmapped_conv_impl PUBLIC ${ROOT_DIR})

# ── Helper macro: add a test executable ──────────────────────────
macro(add_ncnn_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} ${ARGN} ncnn_stub m)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# ── Helper macro: per-kernel mapped test (base + arm) ────────────
# Creates a library with base.cpp + base_arm.cpp, then the test target.
macro(add_mapped_kernel_test kernel)
    add_library(mapped_${kernel}_impl STATIC
        ${MAPPED_DIR}/${kernel}/${kernel}.cpp
        ${MAPPED_DIR}/${kernel}/${kernel}_arm.cpp
    )
    target_include_directories(mapped_${kernel}_impl PUBLIC ${ROOT_DIR})
    add_ncnn_test(test_mapped_${kernel}
        mapped_${kernel}_impl
    )
endmacro()

# ── Helper macro: per-kernel unmapped test (base only) ───────────
macro(add_unmapped_kernel_test kernel)
    add_library(unmapped_${kernel}_impl STATIC
        ${UNMAPPED_DIR}/${kernel}/${kernel}.cpp
    )
    target_include_directories(unmapped_${kernel}_impl PUBLIC ${ROOT_DIR})
    add_ncnn_test(test_unmapped_${kernel}
        unmapped_${kernel}_impl
    )
endmacro()

# ── Test targets ──────────────────────────────────────────────────
enable_testing()

# test_mapped_conv: base + arm conv kernels from mapped/
add_ncnn_test(test_mapped_conv
    mapped_conv_arm
    mapped_conv_base
)

# test_mapped_ip: InnerProduct and Requantize (base + arm)
add_ncnn_test(test_mapped_ip
    mapped_ip_arm
    mapped_ip_base
)

# test_unmapped_conv: all unmapped conv kernels
add_ncnn_test(test_unmapped_conv
    unmapped_conv_impl
)

# ── Per-kernel mapped tests ───────────────────────────────────────
add_mapped_kernel_test(absval)
add_mapped_kernel_test(relu)
add_mapped_kernel_test(sigmoid)
add_mapped_kernel_test(tanh)
add_mapped_kernel_test(clip)
add_mapped_kernel_test(gelu)
add_mapped_kernel_test(swish)
add_mapped_kernel_test(mish)
add_mapped_kernel_test(hardsigmoid)
add_mapped_kernel_test(hardswish)
add_mapped_kernel_test(selu)
add_mapped_kernel_test(dropout)
add_mapped_kernel_test(unaryop)
add_mapped_kernel_test(prelu)
add_mapped_kernel_test(batchnorm)
add_mapped_kernel_test(bias)
add_mapped_kernel_test(layernorm)
add_mapped_kernel_test(binaryop)
add_mapped_kernel_test(eltwise)
add_mapped_kernel_test(softmax)
add_mapped_kernel_test(flatten)
add_mapped_kernel_test(concat)

# pooling_arm.cpp conditionally includes pooling_2x2.h etc. from arm-heavy-optimized/reduction/
add_library(mapped_pooling_impl STATIC
    ${MAPPED_DIR}/pooling/pooling.cpp
    ${MAPPED_DIR}/pooling/pooling_arm.cpp
)
target_include_directories(mapped_pooling_impl PUBLIC
    ${ROOT_DIR}
    ${ROOT_DIR}/arm-heavy-optimized/reduction
)
add_ncnn_test(test_mapped_pooling mapped_pooling_impl)

# scale: base only (Scale_arm only overrides the two-blob variant)
add_library(mapped_scale_impl STATIC
    ${MAPPED_DIR}/scale/scale.cpp
)
target_include_directories(mapped_scale_impl PUBLIC ${ROOT_DIR})
add_ncnn_test(test_mapped_scale mapped_scale_impl)

# ── Per-kernel unmapped tests ─────────────────────────────────────
add_unmapped_kernel_test(sdpa)
add_unmapped_kernel_test(reduction)
add_unmapped_kernel_test(flip)
add_unmapped_kernel_test(tile)

# pooling1d uses layer_type.h (for copy_make_border via Padding)
add_library(unmapped_pooling1d_impl STATIC
    ${UNMAPPED_DIR}/pooling1d/pooling1d.cpp
)
target_include_directories(unmapped_pooling1d_impl PUBLIC ${ROOT_DIR})
add_ncnn_test(test_unmapped_pooling1d unmapped_pooling1d_impl)

# ── Summary target ────────────────────────────────────────────────
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS
        test_mapped_conv test_mapped_ip test_unmapped_conv
        test_mapped_absval test_mapped_relu test_mapped_sigmoid
        test_mapped_tanh test_mapped_clip test_mapped_gelu
        test_mapped_swish test_mapped_mish test_mapped_hardsigmoid
        test_mapped_hardswish test_mapped_selu test_mapped_dropout
        test_mapped_unaryop test_mapped_prelu test_mapped_batchnorm
        test_mapped_bias test_mapped_layernorm test_mapped_binaryop
        test_mapped_eltwise test_mapped_softmax test_mapped_flatten
        test_mapped_concat test_mapped_pooling test_mapped_scale
        test_unmapped_sdpa test_unmapped_reduction test_unmapped_pooling1d
        test_unmapped_flip test_unmapped_tile
    COMMENT "Running all ncnn kernel tests"
)
