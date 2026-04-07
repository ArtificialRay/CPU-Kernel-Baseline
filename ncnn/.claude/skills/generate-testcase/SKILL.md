---
name: generate-testcase
description: Step-by-step tutorial to generate test cases for ncnn kernel implementations using the new test setup that compiles and links the actual .cpp files
---

# Tutorial: Generate test cases for ncnn kernel implementations

## Goal 
Generate test cases for ncnn kernel implementations (e.g. `sdpa.cpp`, `multiheadattention.cpp`) using the new test setup that compiles and links the actual .cpp files.

## Notice
Do not changes of the original kernel and dependency implementation

## Directory structure
```ncnn/
├── c-partially-optimized/
│   ├── attention/
│   ├── activation/
│   ├── convolution/
│   ├── gemm/
│   ├── ...
├── arm-heavy-optimized/
│   ├── attention/
│   ├── convolution/
│   ├── gemm/
│   ├── common/
│   ├── ...
├── framework/
├── mapped/ 
│   ├── batchnorm/
│   ├── cast/
│   ├── attention/
│   ├── ...
├── unmapped/
│   ├── argmax/
│   ├── bnll/
│   ├── deconvollution1d/
│   ├── ...
```

## Step 1: Identify kernel implementations 
Fetch all kernel implementation directories (e.g. `attention/`, `activation/`, `convolution/`) in both `ncnn/mapped/` and `ncnn/unmapped/` and read the implementations to understand their functionality and expected behavior. 


## Step 2: Write per kernel reference test cases in `ncnn/tests/`
Based on the understanding of kernel implementations, initialize reference test cases in `ncnn/tests/` for each type of kernel in both `ncnn/mapped/` and `ncnn/unmapped/`. 
For kernels in `ncnn/mapped/`, each kernel has a corresponding arm baseline , you can use the arm baseline as reference and write a test case to compare the output of c-partially-optimized ncnn kernelw with the arm baseline.

For kernels in `ncnn/unmapped/`, since there is no arm baseline, you can write a reference implementation in the test case based on the kernel implementation and use it as reference to compare with the output of ncnn kernel implementations. Here is an example:

```cpp
static void ref_sdpa(const float* Q, const float* K, const float* V,
                     float* out,
                     int num_heads, int tgt_len, int src_len,
                     int head_dim, int v_dim,
                     float scale,
                     const float* attn_mask = nullptr)
{
    std::vector<float> scores(tgt_len * src_len);
    for (int h = 0; h < num_heads; ++h) {
        const float* q = Q + h * tgt_len * head_dim;
        const float* k = K + h * src_len * head_dim;
        const float* v = V + h * src_len * v_dim;
        float*       o = out + h * tgt_len * v_dim;

        for (int i = 0; i < tgt_len; ++i)
            for (int j = 0; j < src_len; ++j) {
                float s = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    s += q[i*head_dim+d] * k[j*head_dim+d];
                scores[i*src_len+j] = s * scale;
                if (attn_mask) scores[i*src_len+j] += attn_mask[i*src_len+j];
            }
        for (int i = 0; i < tgt_len; ++i) softmax_inplace(scores.data() + i*src_len, src_len);
        for (int i = 0; i < tgt_len; ++i)
            for (int d = 0; d < v_dim; ++d) {
                float sum = 0.f;
                for (int j = 0; j < src_len; ++j) sum += scores[i*src_len+j] * v[j*v_dim+d];
                o[i*v_dim+d] = sum;
            }
    }
}
```

**key points:**
- Make sure each kernel has its own reference test case
- The reference test cases should be self-contained, do not add any dependency on ncnn codebase
- Write all reference testcases for each kernel in a single file
- After the ref cases are ready, verify if it could be compiled and run successfully 

## step 3: Configure testcase starter code to call the real implementations
Fetch dependencies of kernel implementations in `/ncnn/framework` ,`/ncnn/common`; Initialize the test case starter code to call the real implementations. 

```cpp
// Runs the real ncnn::SDPA::forward() and checks it matches ref_sdpa().
// Returns true if all values match within tol.
static bool run_sdpa_test(int num_heads, int tgt_len, int src_len,
                           int head_dim, int v_dim, float scale,
                           const std::vector<float>& Q_flat,
                           const std::vector<float>& K_flat,
                           const std::vector<float>& V_flat,
                           const std::vector<float>* mask_flat = nullptr,
                           float tol = 1e-4f)
{
    // ── Reference ──
    std::vector<float> ref(num_heads * tgt_len * v_dim, 0.f);
    ref_sdpa(Q_flat.data(), K_flat.data(), V_flat.data(), ref.data(),
             num_heads, tgt_len, src_len, head_dim, v_dim, scale,
             mask_flat ? mask_flat->data() : nullptr);

    // ── Build ncnn Mats ──
    // SDPA input convention:  query[c=heads, h=tgt_len, w=embed_dim]
    //                         cur_key[c=heads, h=src_len, w=embed_dim]
    //                         cur_value[c=heads, h=src_len, w=v_dim]
    ncnn::Mat q = make_mat(head_dim, tgt_len, num_heads, Q_flat);
    ncnn::Mat k = make_mat(head_dim, src_len, num_heads, K_flat);
    ncnn::Mat v = make_mat(v_dim,    src_len, num_heads, V_flat);

    std::vector<ncnn::Mat> bottom = {q, k, v};

    if (mask_flat) {
        // attn_mask shape: [tgt_len, src_len] (no head dim → single shared mask)
        ncnn::Mat mask = make_mat(src_len, tgt_len, 1, *mask_flat);
        bottom.push_back(mask);
    }

    std::vector<ncnn::Mat> top(1);

    ncnn::SDPA sdpa;
    sdpa.attn_mask      = (mask_flat != nullptr) ? 1 : 0;
    sdpa.scale          = scale;
    sdpa.kv_cache       = 0;
    sdpa.int8_scale_term = 0;

    ncnn::Option opt;
    opt.num_threads = 1;

    int ret = sdpa.forward(bottom, top, opt);
    if (ret != 0) {
        fprintf(stderr, "  SDPA::forward() returned %d\n", ret);
        return false;
    }

    // ── Compare ──
    std::vector<float> got;
    read_mat(top[0], got);

    for (int i = 0; i < (int)ref.size(); ++i) {
        if (fabsf(got[i] - ref[i]) > tol) {
            fprintf(stderr, "  mismatch at [%d]: got=%.6f  ref=%.6f\n",
                    i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}
```

**key points:**
- To make your work simple, try to implement starter code that could be reused for different kernels, and put your implementation in a common directory or file (e.g. `ncnn/tests/test_utils.cpp/h`)
- If you think the dependency implementation is too heavy, you are free to provide a **Minimal implementations of ncnn framework symbols** needed to compile and link with real kernel implementations without full ncnn framework build.
- Make sure the testcases starter code calls real ncnn kernel implementation 
- Make sure the testcases starter code compare the output of real ncnn kernel implementations with the reference implementation with a reasonable tolerance (e.g. 1e-4 for float)

## step 4: Enrich test cases to dynamic input size
Enrich the test cases with dynamic input size, and make sure the testcases could be run successfully with different input sizes. 

## step 4: Configure CMakeLists.txt to compile and link the real implementations

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

## step 5: Build and test your generated test cases with CMakeList.txt

Build the test target with CMake and run the generated test cases to verify they pass successfully.

```bash
# configure
# cd /your-clone-di/ncnn/$ARGUMENTS/tests/
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

→ **For ncnn dependency analysis guide, see  [`.claude/skills/analyze-ncnn-dependency/SKILL.md`](../analyze-ncnn-dependency/SKILL.md)**

## Summary of Directory/File changes
```
ncnn/$ARGUMENTS/tests/ # NEW: test cases for kernel implementations
ncnn/$ARGUMENTS/tests/CMakeLists.txt # NEW: compile and link real kernel implementations
```
