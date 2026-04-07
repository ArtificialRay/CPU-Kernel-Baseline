---
name: verify-testcase
description: Step-by-step tutorial to verify test cases that is generated in 'ncnn/tests' in an auto-research fusion
---


## step 1: Configure CMakeLists.txt to compile and link the real implementations

Configure `ncnn/tests/CMakeLists.txt` to compile and link the real implementations with the test case starter code.

```cmake
add_library(ncnn_stub STATIC ncnn_framework_stub.cpp) # adding ncnn framework dependency stubs, could be real implementations if the dependency is not heavy
target_include_directories(ncnn_stub PUBLIC ..)   # c-partially-optimized/

add_library(attention_impl STATIC
    ../attention/sdpa.cpp
    ../attention/multiheadattention.cpp)
target_include_directories(attention_impl PUBLIC ..)

# test_attention links the REAL implementations
add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention attention_impl ncnn_stub m)
```

**key points:**
- Make sure the real implementations are compiled and linked in the test target
- If there are framework dependencies, add stubs or real implementations to allow compiling and linking without the full ncnn framework build
- If an optimized kernel implementation depends on its partially optimized version (e.g. `arm-heavy-optimized/conv/convolution_arm.cpp` depends on `c-partially-optimized/conv/convolution.cpp`), make sure to compile and link both implementations in the test target
- Make sure all optimization skills (not only ARM SIMD intrinsics/assembly, but also multi-threading, cache optimization, etc.) are properly tested with the real implementations in the test target. 

## step 2: Build and test your generated test cases with CMakeList.txt

Build the test target with CMake and run the generated test cases to verify they pass successfully.

```bash
# configure
# cd /your-clone-di/ncnn/tests/
mkdir -p build && cd build
cmake ..
make -j$(nproc)
# Run all tests
ctest --output-on-failure
# Or use the summary target
make run_all_tests
```

**key points:**
- Make sure the testcases for all types of kernel (e.g. attention, convolution, gemm) are built and run successfully
- If there are any test failures, debug and fix the issues on generated test cases until all tests pass successfully
- Double check if all optimization skills are properly compiled and tested, you can find warning messages in the build output (e.g. warning: ignoring ‘#pragma omp parallel’) helpful.

→ **For ncnn dependency analysis guide, see  [`.claude/skills/map-kernels-with-arm-baseline/SKILL.md`](../map-kernels-with-arm-baseline/SKILL.md)**

## Summary of Directory/File changes
```
ncnn/tests/CMakeLists.txt # NEW: compile and link real kernel implementations
```