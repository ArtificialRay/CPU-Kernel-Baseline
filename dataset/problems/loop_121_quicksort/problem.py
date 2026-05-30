"""
loop_121: Quicksort

Purpose: Use of CMPLT with COMPACT and CNTP instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_121",
    "num": "121",
    "name": "Quicksort",
    "description": "Partition an integer array around a pivot (one pass of quicksort)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_121_quicksort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_121_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
|| defined(HAVE_SVE_INTRINSICS) || defined(__ARM_FEATURE_SVE)
static inline uint32_t find_pivot(uint32_t n, const int32_t *restrict data) {
  struct tuple_32 {
    uint32_t idx;
    int32_t val;
  } t, candidates[3];

  candidates[0].idx = 0;
  candidates[1].idx = n - 1;
  candidates[2].idx = n / 2;

  candidates[0].val = data[candidates[0].idx];
  candidates[1].val = data[candidates[1].idx];
  candidates[2].val = data[candidates[2].idx];

  int i, j;
  for (i = 1; i < 3; i++) {
    t = candidates[i];
    for (j = i - 1; j >= 0 && candidates[j].val > t.val; j--)
      candidates[j + 1] = candidates[j];
    candidates[j + 1] = t;
  }

  return candidates[1].idx;
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Quicksort
Purpose: Use of CMPLT with COMPACT and CNTP instructions
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
EDGE_SIZES = [0, 1, 2, 255, 257]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [50000, 160000]
