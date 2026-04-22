"""
loop_105: Cascade summation

Purpose: Use of pairwise FP add instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_105",
    "num": "105",
    "name": "Cascade summation",
    "description": "Sum adjacent FP value pairs in a cascading pairwise reduction",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_105_cascade_summation",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_105_data {
  float *restrict a;
  float *restrict b;
  int n;
  float res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
|| defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS) ||\
          defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME) || defined(__ARM_FEATURE_SVE) || defined(__ARM_NEON)
static float cascade_summation_16(float *restrict a)
LOOP_ATTR
{
  float t0 = a[0] + a[1];
  float t1 = a[2] + a[3];
  float t2 = a[4] + a[5];
  float t3 = a[6] + a[7];
  float t4 = a[8] + a[9];
  float t5 = a[10] + a[11];
  float t6 = a[12] + a[13];
  float t7 = a[14] + a[15];
  float t10 = t0 + t1;
  float t11 = t2 + t3;
  float t12 = t4 + t5;
  float t13 = t6 + t7;
  float t100 = t10 + t11;
  float t101 = t12 + t13;
  return t100 + t101;
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Cascade summation
Purpose: Use of pairwise FP add instruction
Target: {isa_upper} on {isa_desc}

Struct definition:
```c
{struct_def}
```

Scalar implementation to optimize:
```c
{scalar_code}
```

Write an optimized {isa_upper} implementation. Output only the C function.
"""

# Input sizes for edge-case correctness testing at submit time.
# Empty list = skip (loop uses non-SIZE parameters or fixed dimensions).
# SIZE must be a power of 2 >= 16 (cascade_summation base case is n==16).
EDGE_SIZES = [16, 32, 64, 4096, 8192]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [1048576, 16777216]
