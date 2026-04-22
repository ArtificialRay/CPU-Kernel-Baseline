#include <arm_neon.h>
#include "micro_profiler.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <sys/syslimits.h>
#include <unistd.h>
#endif

using namespace microprof;

static constexpr int N = 8;
static constexpr int ELEMS = N * N;

__attribute__((noinline))
void matmul8x8_fp32_outer_reference(const float* __restrict A,
                                    const float* __restrict B,
                                    float* __restrict C) {
    for (int i = 0; i < ELEMS; ++i) {
        C[i] = 0.0f;
    }

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            const float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

__attribute__((noinline))
void matmul8x8_fp32_outer_NEON(const float* __restrict A,
                               const float* __restrict B,
                               float* __restrict C) {
    float32x4_t c_lo[8];
    float32x4_t c_hi[8];

    for (int i = 0; i < N; ++i) {
        c_lo[i] = vdupq_n_f32(0.0f);
        c_hi[i] = vdupq_n_f32(0.0f);
    }

    for (int k = 0; k < N; ++k) {
        const float32x4_t b_lo = vld1q_f32(B + k * N + 0);
        const float32x4_t b_hi = vld1q_f32(B + k * N + 4);

        for (int i = 0; i < N; ++i) {
            const float32x4_t a_broadcast = vdupq_n_f32(A[i * N + k]);
            c_lo[i] = vfmaq_f32(c_lo[i], a_broadcast, b_lo);
            c_hi[i] = vfmaq_f32(c_hi[i], a_broadcast, b_hi);
        }
    }

    for (int i = 0; i < N; ++i) {
        vst1q_f32(C + i * N + 0, c_lo[i]);
        vst1q_f32(C + i * N + 4, c_hi[i]);
    }
}

static std::string get_executable_path(const char* argv0) {
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) == 0) {
        char resolved[PATH_MAX];
        if (realpath(buf.data(), resolved) != nullptr) {
            return std::string(resolved);
        }
        return std::string(buf.data());
    }
#endif
    char resolved[PATH_MAX];
    if (realpath(argv0, resolved) != nullptr) {
        return std::string(resolved);
    }
    return std::string(argv0 ? argv0 : "program");
}

static std::string dirname_of(const std::string& path) {
    const std::size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return std::string(".");
    }
    if (pos == 0) {
        return std::string("/");
    }
    return path.substr(0, pos);
}

static std::string basename_of(const std::string& path) {
    const std::size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

static bool nearly_equal(float a, float b) {
    const float diff = std::fabs(a - b);
    const float scale = std::max(std::fabs(a), std::fabs(b));
    const float tol = std::max(1e-5f, 0.01f * scale);
    return diff <= tol;
}

[[gnu::noinline]]
int main(int argc, char* argv[]) {
    std::int64_t number_of_iterations = 1;
    if (argc > 1) {
        try {
            std::size_t pos = 0;
            std::string arg = argv[1];
            long long parsed = std::stoll(arg, &pos, 10);
            if (pos != arg.size() || parsed <= 0) {
                throw std::invalid_argument("non-positive or extra chars");
            }
            number_of_iterations = static_cast<std::int64_t>(parsed);
        } catch (const std::exception&) {
            std::cerr << "Usage: " << argv[0] << " [number_of_iterations]\n"
                      << "  number_of_iterations must be a positive integer.\n";
            return EXIT_FAILURE;
        }
    }

    alignas(16) std::array<float, ELEMS> A;
    alignas(16) std::array<float, ELEMS> B;
    alignas(16) std::array<float, ELEMS> C_ref;
    alignas(16) std::array<float, ELEMS> C_neon;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 0.25f * static_cast<float>(i + 1) - 0.15f * static_cast<float>(j + 1) + 0.01f * static_cast<float>((i * j) % 5);
            B[i * N + j] = -0.05f * static_cast<float>(i + 1) + 0.30f * static_cast<float>(j + 1) - 0.02f * static_cast<float>((i + j) % 7);
        }
    }

    C_ref.fill(0.0f);
    C_neon.fill(0.0f);

    if (number_of_iterations > 1) {
        volatile float warm = 0.0f;
        for (int t = 0; t < 4; ++t) {
            for (int i = 0; i < ELEMS; ++i) {
                warm += A[i] + B[i];
            }
        }
        (void)warm;
    }

    auto h_reference = prof_register("matmul8x8_fp32_outer_reference");
    auto h_neon = prof_register("matmul8x8_fp32_outer_NEON");

    prof_start(h_reference);
    for (std::int64_t it = 0; it < number_of_iterations; ++it) {
        matmul8x8_fp32_outer_reference(A.data(), B.data(), C_ref.data());
    }
    prof_stop(h_reference);

    prof_start(h_neon);
    for (std::int64_t it = 0; it < number_of_iterations; ++it) {
        matmul8x8_fp32_outer_NEON(A.data(), B.data(), C_neon.data());
    }
    prof_stop(h_neon);

    bool neon_ok = true;
    for (int i = 0; i < ELEMS; ++i) {
        if (!nearly_equal(C_ref[i], C_neon[i])) {
            neon_ok = false;
            break;
        }
    }

    std::cout << "matmul8x8_fp32_outer_NEON: " << (neon_ok ? "pass" : "fail") << "\n";

    const std::string exe_path = get_executable_path(argv[0]);
    const std::string exe_dir = dirname_of(exe_path);
    const std::string exe_base = basename_of(exe_path);

    const std::string results_path = exe_dir + "/" + exe_base + "_results.json";
    FILE* results_file = std::fopen(results_path.c_str(), "w");
    if (results_file) {
        std::fprintf(results_file,
                     "{\n  \"results\": {\n    \"matmul8x8_fp32_outer_NEON\": \"%s\"\n  }\n}\n",
                     neon_ok ? "pass" : "fail");
        std::fclose(results_file);
    } else {
        std::cerr << "Failed to open results file: " << results_path << "\n";
    }

    const std::string prof_path = exe_dir + "/" + exe_base + "_profiling_hw.json";
    FILE* prof_file = std::fopen(prof_path.c_str(), "w");
    if (prof_file) {
        prof_report_json(prof_file, true, number_of_iterations);
        std::fclose(prof_file);
    } else {
        std::cerr << "Failed to open profiling file: " << prof_path << "\n";
    }

    return neon_ok ? 0 : 1;
}