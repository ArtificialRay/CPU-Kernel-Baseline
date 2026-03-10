cmake_minimum_required(VERSION 3.18)
project(sgl_cpu_kernels CXX)

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Compiler flags — no CUDA, no torch, pure C++
# ---------------------------------------------------------------------------
set(AVX512_FLAGS "-mavx512f -mavx512bf16 -mavx512vl -mavx512bw")
set(OPT_FLAGS    "-O3 -funroll-loops -ffast-math")
set(WARN_FLAGS   "-Wall -Wextra -Wno-unused-parameter")

# Check if compiler supports AVX-512 BF16
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512bf16" HAS_AVX512BF16)
if(NOT HAS_AVX512BF16)
    message(FATAL_ERROR "Compiler does not support -mavx512bf16. Use GCC >= 11.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX512_FLAGS} ${OPT_FLAGS} ${WARN_FLAGS}")

find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Kernel sources (zero framework dependency)
# ---------------------------------------------------------------------------
set(KERNEL_SRCS
    src/sgl_norm.cpp
    src/sgl_activation.cpp
    src/sgl_rope.cpp
    src/sgl_gemm.cpp
    src/sgl_moe.cpp
    src/sgl_collective.cpp
)

# Static library (use this from C/C++ inference runtimes directly)
add_library(sgl_cpu_kernels_static STATIC ${KERNEL_SRCS})
target_include_directories(sgl_cpu_kernels_static PUBLIC include)
target_link_libraries(sgl_cpu_kernels_static PUBLIC OpenMP::OpenMP_CXX m)

# Shared library (dlopen-able from Python ctypes or other runtimes)
add_library(sgl_cpu_kernels SHARED ${KERNEL_SRCS})
target_include_directories(sgl_cpu_kernels PUBLIC include)
target_link_libraries(sgl_cpu_kernels PUBLIC OpenMP::OpenMP_CXX m)
set_target_properties(sgl_cpu_kernels PROPERTIES
    OUTPUT_NAME "sgl_cpu_kernels"
    VERSION     "1.0.0"
    SOVERSION   "1"
)

# ---------------------------------------------------------------------------
# Tests (also zero framework dependency)
# ---------------------------------------------------------------------------
enable_testing()
add_executable(test_kernels tests/test_kernels.cpp)
target_include_directories(test_kernels PRIVATE include)
target_link_libraries(test_kernels PRIVATE sgl_cpu_kernels_static)
add_test(NAME unit_tests COMMAND test_kernels)

# ---------------------------------------------------------------------------
# Optional: PyTorch binding shim (only if torch is available)
# ---------------------------------------------------------------------------
option(BUILD_TORCH_BINDING "Build PyTorch extension shim" OFF)
if(BUILD_TORCH_BINDING)
    find_package(Torch REQUIRED)
    add_library(sgl_cpu_kernels_torch SHARED src/sgl_torch_binding.cpp)
    target_include_directories(sgl_cpu_kernels_torch PRIVATE include)
    target_link_libraries(sgl_cpu_kernels_torch PRIVATE sgl_cpu_kernels_static ${TORCH_LIBRARIES})
    set_target_properties(sgl_cpu_kernels_torch PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "AVX-512 BF16 flags: ${AVX512_FLAGS}")
message(STATUS "Torch binding:      ${BUILD_TORCH_BINDING}")
