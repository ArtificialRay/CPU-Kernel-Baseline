// Example: Cross-platform SIMD profiling demo
// Compiles and runs on both Linux (Arm64) and macOS (Apple Silicon)
//
// Compile:
//   Linux:  g++ -std=c++17 -march=armv8-a+fp16 -I../src cross_platform_example.cpp -o example
//   macOS:  clang++ -std=c++17 -I../src cross_platform_example.cpp -o example
//
// Run:
//   ./example 10000

#include "micro_profiler.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <vector>

using namespace microprof;

// Reference: scalar FP16 to FP32 conversion
__attribute__((noinline))
void reference_fp16_to_fp32(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Using FP16 arithmetic extension if available
        #if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
            _Float16 h;
            std::memcpy(&h, &src[i], sizeof(h));
            dst[i] = (float)h;
        #else
            // Fallback: bitcast via vcvt intrinsic
            float16_t h = vld1_dup_f16((const __fp16*)&src[i]);
            dst[i] = vgetq_lane_f32(vcvt_f32_f16(h), 0);
        #endif
    }
}

// NEON: vectorized FP16 to FP32 conversion
__attribute__((noinline))
void NEON_fp16_to_fp32(const uint16_t* src, float* dst, size_t count) {
    size_t i = 0;
    // Process 4 FP16 values at a time -> 4 FP32 values
    for (; i + 4 <= count; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)&src[i]);
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(&dst[i], f);
    }
    // Handle remainder
    for (; i < count; ++i) {
        float16x4_t h = vld1_dup_f16((const __fp16*)&src[i]);
        float32x4_t f = vcvt_f32_f16(h);
        dst[i] = vgetq_lane_f32(f, 0);
    }
}

int main(int argc, char* argv[]) {
    // Parse iterations
    std::int64_t iterations = 1;
    if (argc > 1) {
        try {
            std::size_t pos = 0;
            std::string arg = argv[1];
            long long parsed = std::stoll(arg, &pos, 10);
            if (pos != arg.size() || parsed <= 0) {
                throw std::invalid_argument("invalid");
            }
            iterations = static_cast<std::int64_t>(parsed);
        } catch (const std::exception&) {
            std::cerr << "Usage: " << argv[0] << " [iterations]\n";
            return EXIT_FAILURE;
        }
    }

    // Test data
    constexpr size_t N = 256;
    std::vector<uint16_t> src(N);
    std::vector<float> ref_dst(N), neon_dst(N);

    // Initialize with FP16 bit patterns (1.0, 2.0, 3.0, ...)
    for (size_t i = 0; i < N; ++i) {
        __fp16 val = (__fp16)(i + 1.0f);
        std::memcpy(&src[i], &val, sizeof(uint16_t));
    }

    // Register profiling handles
    auto h_ref = prof_register("reference_fp16_to_fp32");
    auto h_neon = prof_register("NEON_fp16_to_fp32");

    // Profile reference
    prof_start(h_ref);
    for (std::int64_t i = 0; i < iterations; ++i) {
        reference_fp16_to_fp32(src.data(), ref_dst.data(), N);
    }
    prof_stop(h_ref);

    // Profile NEON
    prof_start(h_neon);
    for (std::int64_t i = 0; i < iterations; ++i) {
        NEON_fp16_to_fp32(src.data(), neon_dst.data(), N);
    }
    prof_stop(h_neon);

    // Verify correctness
    bool pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(ref_dst[i] - neon_dst[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref_dst[i]
                      << " neon=" << neon_dst[i] << "\n";
            pass = false;
            break;
        }
    }

    std::cout << "Correctness test: " << (pass ? "PASS" : "FAIL") << "\n\n";

    // Report profiling results
    std::cout << "Profiling results (" << iterations << " iterations):\n";
    prof_report_json(stdout, true, iterations);

    return pass ? 0 : 1;
}
