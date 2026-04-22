# Cross-Platform Profiler Migration Guide

## Summary

The profiling system has been upgraded to support both **Linux** and **macOS** with a unified API. Generated code now works on both platforms without modification.

## What Changed

### New Header Files
- **`micro_profiler.hpp`** - New unified header (use this in your code)
- **`micro_profiler_mach.hpp`** - macOS implementation using `mach_absolute_time()`
- **`micro_profiler_perf.hpp`** - Existing Linux implementation (unchanged)

### Updated Prompt Configurations
All prompt config files have been updated to reference `micro_profiler.hpp` instead of `micro_profiler_perf.hpp`:
- `system_prompt_from_NL_multiple_simd_ext_profile_code.yaml`
- `system_prompt_from_NL_multiple_simd_ext_profile_code_return_volatile.yaml`
- `system_prompt_profile_batch_results_on_graviton.yaml`
- `system_prompt_profile_current_results_on_graviton.yaml`

## Migration for Existing Code

### Option 1: Use the Unified Header (Recommended)
Replace:
```cpp
#include "micro_profiler_perf.hpp"
```

With:
```cpp
#include "micro_profiler.hpp"
```

The API remains identical. The header will automatically select the correct implementation at compile time.

### Option 2: Keep Existing Code
Your existing code will continue to work on Linux without changes. The `micro_profiler_perf.hpp` header is still available and unchanged.

## Platform Detection

The unified header uses these preprocessor conditionals:
```cpp
#if defined(__APPLE__) && defined(__MACH__)
    // macOS - uses micro_profiler_mach.hpp
#elif defined(__linux__)
    // Linux - uses micro_profiler_perf.hpp
#else
    #error "Unsupported platform"
#endif
```

## Metrics Collected

### Linux (via perf_event)
```json
{
  "Function": "example_func",
  "PERF_COUNT_HW_CPU_CYCLES": 12345,
  "PERF_COUNT_HW_INSTRUCTIONS": 5678,
  "PERF_COUNT_SW_TASK_CLOCK_NS": 1234567
}
```

### macOS (via mach_absolute_time)
```json
{
  "Function": "example_func",
  "MACH_ABSOLUTE_TIME_TICKS": 12345,
  "TIME_NS": 1234567
}
```

## AI Code Generation

The AI will now generate code using `micro_profiler.hpp` by default. When you specify macOS as a target platform via `arm_target_platforms: ["macos"]` in your config, the generated code will compile and run correctly.

## Testing

To verify the cross-platform profiler works on macOS:

```bash
cd src

# Create a minimal test
cat > test.cpp << 'EOF'
#include "micro_profiler.hpp"
using namespace microprof;

void test_func() {
    volatile int sum = 0;
    for (int i = 0; i < 1000; ++i) sum += i;
}

int main() {
    auto h = prof_register("test_func");
    prof_start(h);
    test_func();
    prof_stop(h);
    prof_report_json(stdout, true, 1);
    return 0;
}
EOF

# Compile and run
clang++ -std=c++17 -I. test.cpp -o test
./test

# Clean up
rm test test.cpp
```

Expected output:
```json
[
  {
    "Function": "test_func",
    "MACH_ABSOLUTE_TIME_TICKS": <some value>,
    "TIME_NS": <nanoseconds>
  }
]
```

## Backward Compatibility

✅ Existing Linux code using `micro_profiler_perf.hpp` continues to work  
✅ New code using `micro_profiler.hpp` works on both Linux and macOS  
✅ All API functions remain identical across platforms  
✅ JSON output structure preserved (with platform-specific metric names)

## Benefits

1. **Single Codebase** - Write once, profile anywhere
2. **Automatic Selection** - No manual platform detection needed
3. **Consistent API** - Same function calls on all platforms
4. **Zero Runtime Overhead** - Platform selection at compile time
5. **Future-Ready** - Easy to add new platforms (BSD, Windows ARM, etc.)

## See Also

- [`PROFILER_README.md`](PROFILER_README.md) - Comprehensive API documentation
- [`micro_profiler.hpp`](micro_profiler.hpp) - Unified header source
- [`micro_profiler_mach.hpp`](micro_profiler_mach.hpp) - macOS implementation
- [`micro_profiler_perf.hpp`](micro_profiler_perf.hpp) - Linux implementation
