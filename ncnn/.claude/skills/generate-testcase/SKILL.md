---
name: generate-testcase
description: Step-by-step tutorial to generate test cases for ncnn kernel implementations using the new test setup that compiles and links the actual .cpp files
---

# Tutorial: Generate test cases for ncnn kernel implementations

## Goal 
Generate test cases for ncnn kernel implementations (e.g. `sdpa.cpp`, `multiheadattention.cpp`) using the new test setup that compiles and links the actual .cpp files.

## Notice
- Do not changes of the original kernel and dependency implementation
- Make sure number of test cases file equals to number of type of kernel in both `ncnn/mapped/` and `ncnn/unmapped/`, for example, if there are 5 types of kernels in `ncnn/mapped/` and 3 types of kernels in `ncnn/unmapped/`, there should be 5 test case files in total in `ncnn/mapped/tests/` and 3 test case files in `ncnn/unmapped/tests/`

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
Based on the understanding of kernel implementations, initialize reference test cases for each type of kernel in both `ncnn/mapped/` and `ncnn/unmapped/`. 
For kernels in `ncnn/mapped/`, each kernel has a corresponding arm baseline , you can use the arm baseline as reference and write a test case to compare the output of c-partially-optimized ncnn kernelw with the arm baseline. Write your test to `ncnn/mapped/tests`

For kernels in `ncnn/unmapped/`, since there is no arm baseline, you need to write a reference implementation in the test case based on the kernel implementation and use it as reference to compare with the output of ncnn kernel implementations. Write your test to `ncnn/unmapped/tests`
Here is an example:

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
- Make sure each kernel has its own reference test case **in separate file**, for example, `/ncnn/mapped/attention` has testcase file
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
Enrich the test cases with dynamic input size, and make sure the testcases could be run successfully with different input sizes. The implementation varies between different types of kernels. For example, for convolution kernels, you can write test cases with different input/output channels, kernel sizes, strides, etc. For gemm kernels, you can write test cases with different matrix sizes. For attention kernels, you can write test cases with different number of heads, sequence lengths, head dimensions, etc.
Here is an example of convolution test case with dynamic input size:

```cpp
// Run Convolution_arm, compare against ref; returns false on mismatch.
static bool run_conv2d_arm(int in_c, int out_c,
                            int in_h, int in_w,
                            int kh, int kw,
                            int stride_h, int stride_w,
                            int pad_top,  int pad_left,
                            int dil_h = 1, int dil_w = 1,
                            bool with_bias = false) {
    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, in_c);
    in.fill_range();

    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Convolution_arm conv;
    conv.num_output       = out_c;
    conv.kernel_w         = kw;  conv.kernel_h  = kh;
    conv.dilation_w       = dil_w; conv.dilation_h = dil_h;
    conv.stride_w         = stride_w; conv.stride_h = stride_h;
    conv.pad_left         = pad_left; conv.pad_right  = pad_left;
    conv.pad_top          = pad_top;  conv.pad_bottom = pad_top;
    conv.pad_value        = 0.f;
    conv.bias_term        = with_bias ? 1 : 0;
    conv.weight_data_size = wsize;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);
    if (with_bias) conv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (conv.create_pipeline(opt) != 0) {
        fprintf(stderr, "  FAIL create_pipeline in_c=%d out_c=%d %dx%d k=%dx%d s=%dx%d p=%d,%d d=%dx%d\n",
                in_c, out_c, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
        g_failed++; return false;
    }
    if (conv.forward(bottom, top, opt) != 0) {
        fprintf(stderr, "  FAIL forward in_c=%d out_c=%d %dx%d k=%dx%d s=%dx%d p=%d,%d d=%dx%d\n",
                in_c, out_c, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
        g_failed++; return false;
    }

    TestMat ref = ref_conv2d(in, weight, bias, out_c, kh, kw,
                              stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
```

**key points:**
- Setting dynamic input size inside the test file

→ **For ncnn kernel maps with arm baseline guide, see  [`.claude/skills/map-kernels-with-arm-baseline/SKILL.md`](../map-kernels-with-arm-baseline/SKILL.md)**

## Summary of Directory/File changes
```
ncnn/tests/ # NEW: test cases for kernel implementations
```
