cmake_minimum_required(VERSION 3.14)
project(PaddleLiteStandaloneTests CXX)

# -----------------------------------------------------------------------
# User-configurable options
# -----------------------------------------------------------------------
set(PADDLE_LITE_ROOT "" CACHE PATH
    "Path to Paddle-Lite source tree (required to compile real kernels)")

option(WITH_REAL_KERNELS
    "Also compile the actual ARM kernel .cc files from Paddle-Lite"
    OFF)

option(WITH_ARM_FP16
    "Enable ARMv8.2 FP16 (requires -march=armv8.2-a+fp16)"
    OFF)

option(WITH_ARM_DOTPROD
    "Enable ARMv8.2 INT8 dot-product (requires -march=armv8-a+dotprod)"
    OFF)

# -----------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Base NEON for AArch64 is always present; add extras when requested
set(ARCH_FLAGS "-march=armv8-a")
if(WITH_ARM_DOTPROD)
    string(APPEND ARCH_FLAGS "+dotprod")
    add_compile_definitions(LITE_WITH_ARM82_INT8_SDOT)
endif()
if(WITH_ARM_FP16)
    string(APPEND ARCH_FLAGS "+fp16")
    add_compile_definitions(ENABLE_ARM_FP16)
endif()

add_compile_options(${ARCH_FLAGS} -O2 -Wall -Wno-unused-variable)
add_compile_definitions(LITE_WITH_ARM=1 PADDLE_LITE_STANDALONE)

# -----------------------------------------------------------------------
# Include paths
# -----------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)          # paddle_lite_stub.h

if(WITH_REAL_KERNELS AND PADDLE_LITE_ROOT)
    include_directories(${PADDLE_LITE_ROOT})
    add_compile_definitions(PADDLE_LITE_USE_REAL_KERNEL)
endif()

# -----------------------------------------------------------------------
# Standalone test (reference only, zero Paddle deps)
# -----------------------------------------------------------------------
add_executable(test_arm_kernels_standalone
    test_arm_kernels_standalone.cc
)
target_link_libraries(test_arm_kernels_standalone m)

# -----------------------------------------------------------------------
# Real-kernel test (requires Paddle-Lite sources + compiled math libs)
# -----------------------------------------------------------------------
if(WITH_REAL_KERNELS AND PADDLE_LITE_ROOT)
    # Collect only the math backend files needed (no op-registry, no MIR)
    file(GLOB ARM_MATH_SRCS
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/gemm_prepacked_int8.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/gemv_arm_int8.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/packed_sgemm.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/sgemm.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/sgemv.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/conv3x3s1_depthwise_fp32.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/conv3x3s2_depthwise_fp32.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/conv_direct*.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/conv_winograd*.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/batch_norm.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/norm.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/gru_utils.cc"
        "${PADDLE_LITE_ROOT}/lite/backends/arm/math/lstm.cc"
    )

    # Kernel compute files
    file(GLOB ARM_KERNEL_SRCS
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/conv_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/fc_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/matmul_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/batch_norm_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/layer_norm_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/instance_norm_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/group_norm_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/lstm_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/gru_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/activation_compute.cc"
        "${PADDLE_LITE_ROOT}/lite/kernels/arm/elementwise_compute.cc"
    )

    add_executable(test_arm_kernels_real
        test_arm_kernels_standalone.cc
        ${ARM_MATH_SRCS}
        ${ARM_KERNEL_SRCS}
    )
    target_link_libraries(test_arm_kernels_real m)
endif()
