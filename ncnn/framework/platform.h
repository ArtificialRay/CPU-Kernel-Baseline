// Manually configured platform.h for standalone test builds (no CMake).
// Derived from platform.h.in with all optional features disabled.
#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#define NCNN_STDIO             1
#define NCNN_STRING            1
#define NCNN_SIMPLEOCV         0
#define NCNN_SIMPLEOMP         0
#define NCNN_SIMPLESTL         0
#define NCNN_SIMPLEMATH        0
#define NCNN_THREADS           0
#define NCNN_BENCHMARK         0
#define NCNN_C_API             0
#define NCNN_PLATFORM_API      0
#define NCNN_WINXP             0
#define NCNN_PIXEL             0
#define NCNN_PIXEL_ROTATE      0
#define NCNN_PIXEL_AFFINE      0
#define NCNN_PIXEL_DRAWING     0
#define NCNN_VULKAN            0
#define NCNN_SIMPLEVK          0
#define NCNN_SYSTEM_GLSLANG    0
#define NCNN_RUNTIME_CPU       0
#define NCNN_GNU_INLINE_ASM    0
#define NCNN_AVX               0
#define NCNN_XOP               0
#define NCNN_FMA               0
#define NCNN_F16C              0
#define NCNN_AVX2              0
#define NCNN_AVXVNNI           0
#define NCNN_AVXVNNIINT8       0
#define NCNN_AVXVNNIINT16      0
#define NCNN_AVXNECONVERT      0
#define NCNN_AVX512            0
#define NCNN_AVX512VNNI        0
#define NCNN_AVX512BF16        0
#define NCNN_AVX512FP16        0
#define NCNN_VFPV4             0
#define NCNN_ARM82             0
#define NCNN_ARM82DOT          0
#define NCNN_ARM82FP16FML      0
#define NCNN_ARM84BF16         0
#define NCNN_ARM84I8MM         0
#define NCNN_ARM86SVE          0
#define NCNN_ARM86SVE2         0
#define NCNN_ARM86SVEBF16      0
#define NCNN_ARM86SVEI8MM      0
#define NCNN_ARM86SVEF32MM     0
#define NCNN_MSA               0
#define NCNN_LSX               0
#define NCNN_MMI               0
#define NCNN_RVV               0
#define NCNN_ZFH               0
#define NCNN_ZVFH              0
#define NCNN_XTHEADVECTOR      0
#define NCNN_INT8              0
#define NCNN_BF16              0
#define NCNN_FORCE_INLINE      1

#include "ncnn_export.h"

#ifdef __cplusplus

#include <stddef.h>
#include <stdio.h>

// ── No-op thread primitives (NCNN_THREADS=0) ─────────────────────────────────
namespace ncnn {

class NCNN_EXPORT Mutex
{
public:
    Mutex() {}
    ~Mutex() {}
    void lock() {}
    void unlock() {}
};

class NCNN_EXPORT ConditionVariable
{
public:
    ConditionVariable() {}
    ~ConditionVariable() {}
    void wait(Mutex& /*m*/) {}
    void broadcast() {}
    void signal() {}
};

class NCNN_EXPORT Thread
{
public:
    Thread(void* (*/*fn*/)(void*), void* /*arg*/ = 0) {}
    ~Thread() {}
    void join() {}
};

class NCNN_EXPORT ThreadLocalStorage
{
public:
    ThreadLocalStorage() : data(0) {}
    ~ThreadLocalStorage() {}
    void set(void* value) { data = value; }
    void* get() { return data; }
private:
    void* data;
};

class NCNN_EXPORT MutexLockGuard
{
public:
    MutexLockGuard(Mutex& m) : mutex(m) { mutex.lock(); }
    ~MutexLockGuard() { mutex.unlock(); }
private:
    Mutex& mutex;
};

} // namespace ncnn

#include <algorithm>
#include <list>
#include <vector>
#include <stack>
#include <string>
#include <math.h>

#endif // __cplusplus

#if NCNN_STDIO
#include <stdio.h>
#define NCNN_LOGE(...) do { fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#else
#define NCNN_LOGE(...)
#endif

#if NCNN_FORCE_INLINE
#ifdef __GNUC__
    #define NCNN_FORCEINLINE inline __attribute__((__always_inline__))
#else
    #define NCNN_FORCEINLINE inline
#endif
#else
    #define NCNN_FORCEINLINE inline
#endif

#endif // NCNN_PLATFORM_H
