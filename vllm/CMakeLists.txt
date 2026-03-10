cmake_minimum_required(VERSION 3.18)
project(vllm_arm_standalone CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(VLLM_REAL_KERNELS
    "Compile actual vLLM csrc/cpu kernels instead of the standalone reference"
    OFF)
option(ARM_BF16
    "Enable ARM BF16 (armv8.2-a+bf16); requires Graviton3 / Neoverse V1 / Apple M"
    OFF)

set(VLLM_SRC_DIR "" CACHE PATH
    "Path to the vLLM repo root (required when VLLM_REAL_KERNELS=ON)")

# ── Include dirs ─────────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── ARM ISA flags ─────────────────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(ARM_BF16)
        set(ARM_FLAGS "-march=armv8.2-a+dotprod+fp16+bf16")
        add_compile_definitions(ARM_BF16_SUPPORT)
        message(STATUS "ARM ISA: armv8.2-a+dotprod+fp16+bf16")
    else()
        set(ARM_FLAGS "-march=armv8.2-a+dotprod+fp16")
        message(STATUS "ARM ISA: armv8.2-a+dotprod+fp16")
    endif()
    add_compile_options(${ARM_FLAGS})
else()
    message(STATUS "Non-ARM host — using scalar fallback")
endif()

# ── Compile definitions ───────────────────────────────────────────────────────
add_compile_definitions(VLLM_STANDALONE)

# ── Target: standalone reference tests (zero framework dep) ──────────────────
add_executable(test_vllm_arm_standalone
    test_vllm_arm_kernels.cpp
)
target_compile_options(test_vllm_arm_standalone PRIVATE -O3)
target_link_libraries(test_vllm_arm_standalone m)
if(UNIX)
    target_link_libraries(test_vllm_arm_standalone pthread)
endif()

# ── Target: real vLLM kernels (requires vLLM source + PyTorch headers) ────────
if(VLLM_REAL_KERNELS)
    if(NOT VLLM_SRC_DIR)
        message(FATAL_ERROR "Set VLLM_SRC_DIR to the vLLM repo root")
    endif()

    # Find PyTorch include dirs (torch.utils.cmake_prefix_path)
    find_package(Torch REQUIRED)

    add_compile_definitions(VLLM_REAL_KERNELS)

    add_executable(test_vllm_arm_real
        test_vllm_arm_kernels.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/attention.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/layernorm.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/activation.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/pos_encoding.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/cache.cpp
        ${VLLM_SRC_DIR}/csrc/cpu/quant.cpp
    )
    target_include_directories(test_vllm_arm_real PRIVATE
        ${VLLM_SRC_DIR}/csrc
        ${VLLM_SRC_DIR}/csrc/cpu
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_options(test_vllm_arm_real PRIVATE -O3)
    target_link_libraries(test_vllm_arm_real ${TORCH_LIBRARIES} m pthread)
    message(STATUS "Real-kernel target: test_vllm_arm_real")
endif()

# ── Quick build instructions ──────────────────────────────────────────────────
message(STATUS "")
message(STATUS "── Build: standalone reference (host) ──────────────────")
message(STATUS "  cmake -B build && cmake --build build")
message(STATUS "  ./build/test_vllm_arm_standalone")
message(STATUS "")
message(STATUS "── Build: standalone reference (AArch64 cross) ─────────")
message(STATUS "  cmake -B build -DCMAKE_TOOLCHAIN_FILE=.../aarch64.cmake")
message(STATUS "  cmake --build build")
message(STATUS "")
message(STATUS "── Build: real vLLM kernels (AArch64) ──────────────────")
message(STATUS "  cmake -B build_real \\")
message(STATUS "    -DVLLM_REAL_KERNELS=ON \\")
message(STATUS "    -DVLLM_SRC_DIR=/path/to/vllm \\")
message(STATUS "    -DARM_BF16=ON \\")
message(STATUS "    -DCMAKE_TOOLCHAIN_FILE=.../aarch64.cmake")
message(STATUS "  cmake --build build_real")
message(STATUS "")
