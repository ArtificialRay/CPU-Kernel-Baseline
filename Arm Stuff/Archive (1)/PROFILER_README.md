# Cross-Platform Micro Profiler

This directory contains a lightweight, header-only profiling library for Arm SIMD code generation that works on both Linux and macOS.

## Header Files

### `micro_profiler.hpp` (Unified Interface)
The main header you should include in your code. It automatically selects the appropriate platform-specific implementation at compile time.

### `micro_profiler_perf.hpp` (Linux Implementation)
Uses Linux `perf_event` subsystem to collect:
- **CPU cycles** (`PERF_COUNT_HW_CPU_CYCLES`)
- **Instructions retired** (`PERF_COUNT_HW_INSTRUCTIONS`)
- **Task clock** (`PERF_COUNT_SW_TASK_CLOCK_NS`) - nanoseconds of on-CPU time

### `micro_profiler_mach.hpp` (macOS Implementation)
Uses macOS `mach_absolute_time()` to collect:
- **Wall-clock time ticks** (`MACH_ABSOLUTE_TIME_TICKS`)
- **Wall-clock time in nanoseconds** (`TIME_NS`)

## API (Identical on All Platforms)

```cpp
#include "micro_profiler.hpp"
using namespace microprof;

// Register a profiling handle (typically done once per function)
ProfHandle h = prof_register("function_name");

// Start profiling
prof_start(h);
// ... your code here ...
prof_stop(h);

// Generate JSON report
prof_report_json(stdout, /*pretty=*/true, /*iterations=*/1);

// Optional: Reset all profiling data
prof_reset();
```

## Usage Example

```cpp
#include "micro_profiler.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace microprof;

void NEON_func();      // generated SIMD function
void reference_func(); // generated reference function

int main(int argc, char* argv[]) {
    // Parse number of iterations
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

    // Register profiling handles
    auto h_NEON_func = prof_register("NEON_func");
    auto h_reference_func = prof_register("reference_func");

    // Profile NEON function
    prof_start(h_NEON_func);
    for (std::int64_t i = 0; i < number_of_iterations; ++i) {
        NEON_func();
    }
    prof_stop(h_NEON_func);

    // Profile reference function
    prof_start(h_reference_func);
    for (std::int64_t i = 0; i < number_of_iterations; ++i) {
        reference_func();
    }
    prof_stop(h_reference_func);

    // Output profiling report to file
    FILE* prof_file = fopen("profiling_results.json", "w");
    if (prof_file) {
        prof_report_json(prof_file, true, number_of_iterations);
        fclose(prof_file);
    }

    // Test for equality between reference and SIMD functions...
    
    return 0;
}
```

## JSON Output Format

### Linux (perf_event)
```json
[
  {
    "Function": "NEON_func",
    "PERF_COUNT_HW_CPU_CYCLES": 12345,
    "PERF_COUNT_HW_INSTRUCTIONS": 5678,
    "PERF_COUNT_SW_TASK_CLOCK_NS": 1234567,
    "_iterations": 1000
  },
  {
    "Function": "reference_func",
    "PERF_COUNT_HW_CPU_CYCLES": 67890,
    "PERF_COUNT_HW_INSTRUCTIONS": 23456,
    "PERF_COUNT_SW_TASK_CLOCK_NS": 7654321,
    "_iterations": 1000
  }
]
```

### macOS (mach_absolute_time)
```json
[
  {
    "Function": "NEON_func",
    "MACH_ABSOLUTE_TIME_TICKS": 12345,
    "TIME_NS": 1234567,
    "_iterations": 1000
  },
  {
    "Function": "reference_func",
    "MACH_ABSOLUTE_TIME_TICKS": 67890,
    "TIME_NS": 7654321,
    "_iterations": 1000
  }
]
```

## Platform-Specific Notes

### Linux
- Requires `perf_event` support (kernel 2.6.32+)
- Hardware counters may require elevated privileges or adjusted `perf_event_paranoid` settings
- Fallback to `CNTVCT_EL0` available on AArch64 if perf_event fails (enable with `-DMICROPROF_FALLBACK_CNTVCT`)

### macOS
- Uses `mach_absolute_time()` which provides monotonic, high-resolution timing
- Does not provide hardware counters (cycles, instructions) as these require kernel extensions
- Suitable for wall-clock performance comparisons

## Compilation

```bash
# Linux
g++ -std=c++17 -I/path/to/src your_code.cpp -o your_binary

# macOS
clang++ -std=c++17 -I/path/to/src your_code.cpp -o your_binary
```

## Advanced Features

### Per-Iteration Averaging
Pass the `iterations` parameter to `prof_report_json()` to automatically divide metrics by the iteration count:

```cpp
prof_report_json(stdout, true, 1000);  // Divide all metrics by 1000
```

### RAII Helper
Use `ProfScope` for automatic start/stop:

```cpp
void my_function() {
    static auto h = prof_register("my_function");
    ProfScope scope(h);  // Starts profiling; stops on scope exit
    // ... your code ...
}
```

Or use the `PROF_FN()` macro:

```cpp
void my_function() {
    PROF_FN();  // Automatically registers and profiles using __func__
    // ... your code ...
}
```

## Thread Safety
- All operations are thread-safe
- Per-thread accounting is maintained automatically
- Aggregated results combine metrics from all threads

## Limitations

### macOS
- No hardware performance counters (cycles, instructions)
- Time measurements are wall-clock, not CPU time
- For accurate profiling, minimize background activity

### Linux
- Requires kernel support for perf_event
- Hardware counters may be unavailable in virtualized environments
- Some counters require elevated privileges

## Integration with AI Code Generator

When generating code for both Linux and macOS targets, include this in your prompt configuration:

```yaml
profiling_method: |
  Include "micro_profiler.hpp" which automatically selects the platform-specific implementation.
  Use prof_register(), prof_start(), prof_stop(), and prof_report_json() as shown in the example.
```

The AI will generate code that works on both platforms without modification.
