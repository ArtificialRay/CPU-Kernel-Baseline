#pragma once
// micro_profiler.hpp — Unified cross-platform profiling header
// Automatically selects the appropriate implementation based on the target platform.
//
// API (identical on all platforms):
//   ProfHandle h = prof_register("name");
//   prof_start(h); /* ... work ... */ prof_stop(h);
//   prof_report_json(stdout, /*pretty=*/true, /*iterations=*/1);
//
// Metrics collected:
//   - Linux: CPU cycles, instructions retired, task clock (via perf_event)
//   - macOS: Wall-clock time (via mach_absolute_time)
//
// Usage:
//   #include "micro_profiler.hpp"
//   using namespace microprof;
//
// The profiler automatically detects the platform at compile time.

#if defined(__APPLE__) && defined(__MACH__)
    // macOS or iOS
    #include "micro_profiler_mach.hpp"
#elif defined(__linux__)
    // Linux
    #include "micro_profiler_perf.hpp"
#else
    #error "Unsupported platform for micro_profiler. Only Linux and macOS are supported."
#endif
