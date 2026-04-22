"""
loop_102: General histogram

Purpose: Use of HISTCNT instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_102",
    "num": "102",
    "name": "General histogram",
    "description": "Count the frequency of each byte value in a large buffer (histogram)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_102_general_histogram",
    "tags": ['sve2', 'histogram'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_102_data {
  uint32_t *restrict histogram;
  uint64_t histogram_size;
  uint32_t *restrict records;
  int64_t num_records;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void NOINLINE update(uint32_t *histogram, uint32_t *records,
                            int64_t num_records) {
  for (int i = 0; i < num_records; i++) {
    uint32_t entry = records[i];
    histogram[entry] += 1;
  }
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: General histogram
Purpose: Use of HISTCNT instruction
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
EDGE_SIZES = [0, 1, 7, 9999, 10001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [8000000, 32000000]
