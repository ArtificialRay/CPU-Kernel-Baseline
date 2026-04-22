cmake_minimum_required(VERSION 3.14)
project(arm_bench_ncnn CXX)

# ── Toolchain / language ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -O2)

find_package(OpenMP REQUIRED)

# ARM NEON / FP16 / dotprod flags apply only when the target is aarch64.
# On x86 the kernels still compile via the scalar fallback branches.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(-march=armv8.2-a+fp16+dotprod)
endif()

# ── Paths ────────────────────────────────────────────────────────────
# This CMake expects to live alongside arm-bench:
#   <repo>/arm-bench/                 (this dir)
#   <repo>/ncnn/  (baseline sources)
set(ARM_BENCH ${CMAKE_CURRENT_SOURCE_DIR})
# BASE_ROOT can be overridden via -DBASE_ROOT=<path> (e.g. on a remote instance
# where the baseline is rsynced to ~/ncnn instead of the local sibling layout).
if(NOT DEFINED BASE_ROOT)
    set(BASE_ROOT ${ARM_BENCH}/../ncnn)
endif()

if(NOT EXISTS ${BASE_ROOT}/framework/mat.cpp)
    message(FATAL_ERROR
        "Expected CPU-Kernel-Baseline/ncnn at ${BASE_ROOT}, but framework/mat.cpp is missing.\n"
        "Either place CPU-Kernel-Baseline as a sibling of arm-bench, or set -DBASE_ROOT=<path>.")
endif()

# ── ncnn framework static lib (shared by every test binary) ──────────
# Supplies:
#   - Mat / Expression / Allocator / Option / Layer / ParamDict / ModelBin
#     (implementations from CPU-Kernel-Baseline's framework; same ABI as the
#     class declarations our starter / arm-baseline headers see).
#   - Padding layer (mapped/padding) because Convolution::make_padding calls
#     copy_make_border, which is implemented via this layer.
#   - ncnn_framework_stub.cpp — fills in Allocator::~Allocator, Option::Option,
#     Layer::* defaults and the create_layer_cpu() factory (a trimmed stand-in
#     for what ncnn's full build-time codegen would provide).
add_library(ncnn_framework STATIC
    ${BASE_ROOT}/framework/mat.cpp
    ${BASE_ROOT}/framework/expression.cpp
    ${BASE_ROOT}/mapped/padding/padding.cpp
    ${BASE_ROOT}/mapped/tests/ncnn_framework_stub.cpp
)
target_include_directories(ncnn_framework PUBLIC
    ${BASE_ROOT}
    ${BASE_ROOT}/framework
)
target_link_libraries(ncnn_framework PUBLIC OpenMP::OpenMP_CXX)

# ── Include paths reused by every ncnn kernel test ──────────────────
set(NCNN_TEST_INCLUDES
    ${ARM_BENCH}                        # "starter/ncnn/…" + "candidates_src/…" from tests and kernels
    ${ARM_BENCH}/tests/ncnn             # "test_utils.h"
    ${ARM_BENCH}/starter/ncnn           # unqualified "ncnn_helpers.h" / "ref_conv.h"
    ${ARM_BENCH}/candidates_src/ncnn    # "common/fused_activation.h" inside candidate .cpp
    ${BASE_ROOT}                        # "framework/…" / "mapped/…" from baseline .cpp
    ${BASE_ROOT}/framework              # unqualified "mat.h" / "option.h" / "layer.h"
)

# ── Auto-generate a main() per test binary ──────────────────────────
# Each test .cpp defines void test_xxx() but no main(). CMake parses the file
# at configure time, emits a combined .cpp = test body + RUN_TEST(fn) calls
# for every discovered test function, and that single TU is what gets compiled.
#
# Why one combined TU instead of linking two: `g_failed` in test_utils.h is
# `static int g_failed = 0;` — file-local per TU. If the test body and main()
# lived in separate TUs they'd each have their own counter, and assertions
# inside the test would never reach main()'s accounting.
#
# Re-configure on test .cpp change via CMAKE_CONFIGURE_DEPENDS so CI/IDE
# pick up edits without a manual cmake re-run.
function(_ncnn_gen_test_main test_src generated_src suite)
    file(READ ${test_src} _content)
    string(REGEX MATCHALL "void[ \t]+test_[A-Za-z0-9_]+[ \t]*\\(" _raw "${_content}")
    set(_calls "")
    foreach(m IN LISTS _raw)
        string(REGEX REPLACE "void[ \t]+(test_[A-Za-z0-9_]+)[ \t]*\\(" "\\1" _fn "${m}")
        string(APPEND _calls "    RUN_TEST(${_fn});\n")
    endforeach()
    file(WRITE ${generated_src}
"${_content}

// ── Auto-generated main (by CMake) ─────────────────────────────────
int main() {
${_calls}    print_summary(\"${suite}\");
    return g_failed ? 1 : 0;
}
")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${test_src})
endfunction()

# ── Candidate test binaries ──────────────────────────────────────────
# A candidate binary contains only:
#   1. candidates_src/ncnn/<kernel>.cpp   — the *_kernel() function under test
#   2. generated test TU (test body + auto main)
#   3. ncnn_framework (Mat, Layer defaults, padding, stub)
#
# The Convolution / ConvolutionDepthWise / … classes are fully defined
# INLINE inside starter/ncnn/candidate/<kernel>.h. So the candidate binary
# never touches CPU-Kernel-Baseline/ncnn/mapped/<kernel>/<kernel>.cpp.
function(add_candidate_test name candidate_src test_src)
    set(stem "test_candidate_${name}")
    set(generated "${CMAKE_BINARY_DIR}/generated/${stem}.cpp")
    _ncnn_gen_test_main(
        ${ARM_BENCH}/tests/ncnn/candidate/${test_src}
        ${generated}
        "${name}_candidate")
    add_executable(${stem}
        ${ARM_BENCH}/candidates_src/ncnn/${candidate_src}
        ${generated}
    )
    target_include_directories(${stem} PRIVATE ${NCNN_TEST_INCLUDES})
    target_link_libraries(${stem} PRIVATE ncnn_framework m)
    add_test(NAME ${stem} COMMAND ${stem})
endfunction()

add_candidate_test(conv       convolution.cpp          convolution.cpp)
add_candidate_test(conv1d     convolution1d.cpp        convolution1d.cpp)
add_candidate_test(convdw     convolutiondepwise.cpp   convolutiondepwise.cpp)
add_candidate_test(deconv     deconvolution.cpp        deconvolution.cpp)
add_candidate_test(deconvdw   deconvolutiondepwise.cpp deconvolutiondepwise.cpp)

# ── Baseline test binaries ───────────────────────────────────────────
# A baseline binary links:
#   1. arm-heavy-optimized/conv/<kernel>_arm.cpp — the ARM NEON subclass
#   2. mapped/<kernel>/<kernel>.cpp               — the base class Convolution::
#      methods, consumed verbatim (the "use baseline source with less mod"
#      principle: we don't touch the original .cpp to provide the base vtable)
#   3. generated test TU + ncnn_framework
#
# arm-heavy-optimized/conv/<kernel>_arm.cpp includes its OWN
# arm-heavy-optimized/conv/<kernel>.h which declares ncnn::Convolution
# identically to our starter header — same class layout, same vtable slots,
# so the test (which sees our starter declaration) dispatches correctly.
function(add_baseline_test name arm_src mapped_subdir mapped_src test_src)
    set(stem "test_baseline_${name}")
    set(generated "${CMAKE_BINARY_DIR}/generated/${stem}.cpp")
    _ncnn_gen_test_main(
        ${ARM_BENCH}/tests/ncnn/baseline/${test_src}
        ${generated}
        "${name}_baseline")
    add_executable(${stem}
        ${BASE_ROOT}/arm-heavy-optimized/conv/${arm_src}
        ${BASE_ROOT}/mapped/${mapped_subdir}/${mapped_src}
        ${generated}
    )
    target_include_directories(${stem} PRIVATE
        ${NCNN_TEST_INCLUDES}
        ${BASE_ROOT}/arm-heavy-optimized/conv   # "convolution_3x3_winograd.h" etc.
    )
    target_link_libraries(${stem} PRIVATE ncnn_framework m)
    add_test(NAME ${stem} COMMAND ${stem})
endfunction()

add_baseline_test(conv     convolution_arm.cpp             convolution          convolution.cpp          convolution.cpp)
add_baseline_test(conv1d   convolution1d_arm.cpp           convolution1d        convolution1d.cpp        convolution1d.cpp)
add_baseline_test(convdw   convolutiondepthwise_arm.cpp    convolutiondepthwise convolutiondepthwise.cpp convolutiondepwise.cpp)
add_baseline_test(deconv   deconvolution_arm.cpp           deconvolution        deconvolution.cpp        deconvolution.cpp)
add_baseline_test(deconvdw deconvolutiondepthwise_arm.cpp  deconvolutiondepthwise deconvolutiondepthwise.cpp deconvolutiondepwise.cpp)

# ── Perf binaries ────────────────────────────────────────────────────
# Perf .cpp files define void perf_xxx() / void test_xxx() that invoke
# run_X(...) only (no run_ref_X, no assertions). Generator emits a main
# that calls each discovered function; the return value of run_X is
# discarded. These binaries link the same kernel sources as the
# corresponding test binaries, so timing measurement can be wrapped
# around them later without re-plumbing the build.
function(_ncnn_gen_perf_main perf_src generated_src)
    file(READ ${perf_src} _content)
    string(REGEX MATCHALL "void[ \t]+(perf|test)_[A-Za-z0-9_]+[ \t]*\\(" _raw "${_content}")
    set(_calls "")
    foreach(m IN LISTS _raw)
        string(REGEX REPLACE "void[ \t]+((perf|test)_[A-Za-z0-9_]+)[ \t]*\\(" "\\1" _fn "${m}")
        string(APPEND _calls "    ${_fn}();\n")
    endforeach()
    file(WRITE ${generated_src}
"${_content}

// ── Auto-generated main (by CMake) ─────────────────────────────────
int main() {
${_calls}    return 0;
}
")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${perf_src})
endfunction()

function(add_candidate_perf name candidate_src perf_src)
    set(stem "perf_candidate_${name}")
    set(generated "${CMAKE_BINARY_DIR}/generated/${stem}.cpp")
    _ncnn_gen_perf_main(
        ${ARM_BENCH}/perf/ncnn/candidate/${perf_src}
        ${generated})
    add_executable(${stem}
        ${ARM_BENCH}/candidates_src/ncnn/${candidate_src}
        ${generated}
    )
    target_include_directories(${stem} PRIVATE ${NCNN_TEST_INCLUDES})
    target_link_libraries(${stem} PRIVATE ncnn_framework m)
endfunction()

function(add_baseline_perf name arm_src mapped_subdir mapped_src perf_src)
    set(stem "perf_baseline_${name}")
    set(generated "${CMAKE_BINARY_DIR}/generated/${stem}.cpp")
    _ncnn_gen_perf_main(
        ${ARM_BENCH}/perf/ncnn/baseline/${perf_src}
        ${generated})
    add_executable(${stem}
        ${BASE_ROOT}/arm-heavy-optimized/conv/${arm_src}
        ${BASE_ROOT}/mapped/${mapped_subdir}/${mapped_src}
        ${generated}
    )
    target_include_directories(${stem} PRIVATE
        ${NCNN_TEST_INCLUDES}
        ${BASE_ROOT}/arm-heavy-optimized/conv
    )
    target_link_libraries(${stem} PRIVATE ncnn_framework m)
endfunction()

add_candidate_perf(conv       convolution.cpp          convolution.cpp)
add_candidate_perf(conv1d     convolution1d.cpp        convolution1d.cpp)
add_candidate_perf(convdw     convolutiondepwise.cpp   convolutiondepwise.cpp)
add_candidate_perf(deconv     deconvolution.cpp        deconvolution.cpp)
add_candidate_perf(deconvdw   deconvolutiondepwise.cpp deconvolutiondepwise.cpp)

add_baseline_perf(conv     convolution_arm.cpp             convolution          convolution.cpp          convolution.cpp)
add_baseline_perf(conv1d   convolution1d_arm.cpp           convolution1d        convolution1d.cpp        convolution1d.cpp)
add_baseline_perf(convdw   convolutiondepthwise_arm.cpp    convolutiondepthwise convolutiondepthwise.cpp convolutiondepwise.cpp)
add_baseline_perf(deconv   deconvolution_arm.cpp           deconvolution        deconvolution.cpp        deconvolution.cpp)
add_baseline_perf(deconvdw deconvolutiondepthwise_arm.cpp  deconvolutiondepthwise deconvolutiondepthwise.cpp deconvolutiondepwise.cpp)

# ── Convenience: run all tests ───────────────────────────────────────
enable_testing()
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all ncnn kernel tests"
)
