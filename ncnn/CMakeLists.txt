cmake_minimum_required(VERSION 3.10)
project(arm_neon_ops CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------
option(NEON_FALLBACK "Compile with scalar fallback (no NEON)" OFF)

# -----------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O2 -Wall -Wextra)
endif()

# ARMv7 needs explicit -mfpu=neon
if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    add_compile_options(-mfpu=neon -mfloat-abi=hard)
endif()

if(NEON_FALLBACK)
    add_compile_definitions(NCNN_NO_NEON)
    message(STATUS "Building with scalar fallback (NCNN_NO_NEON defined)")
else()
    message(STATUS "Building with ARM NEON intrinsics (if __ARM_NEON is set by toolchain)")
endif()

# -----------------------------------------------------------------------
# Test executable
# -----------------------------------------------------------------------
add_executable(test_ops test/test_ops.cpp)
target_include_directories(test_ops PRIVATE include)

# -----------------------------------------------------------------------
# CTest integration
# -----------------------------------------------------------------------
enable_testing()
add_test(NAME ops_unit_tests COMMAND test_ops)

message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  # ARM64 native:")
message(STATUS "  cmake .. && make && ./test_ops")
message(STATUS "  # ARMv7 cross-compile:")
message(STATUS "  cmake .. -DCMAKE_TOOLCHAIN_FILE=<your-armv7-toolchain.cmake> && make")
message(STATUS "  # x86 host (scalar fallback):")
message(STATUS "  cmake .. -DNEON_FALLBACK=ON && make && ./test_ops")
message(STATUS "")
