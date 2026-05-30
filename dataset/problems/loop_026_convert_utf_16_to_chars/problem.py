"""
loop_026: Convert UTF-16 to chars

Purpose: Use of gathers load instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_026",
    "num": "026",
    "name": "Convert UTF-16 to chars",
    "description": "Convert a UTF-16 string to narrow ASCII chars, skipping high surrogate pairs",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_026_convert_utf_16_to_chars",
    "tags": ['sve2', 'string'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_026_data {
  uint16_t *p;
  uint8_t *d;
  uint16_t *lmt;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void NOINLINE convert_utf16_to_bytes(uint16_t *restrict a,
                                            uint8_t *restrict b, int64_t n) {
  for (int i = 0; i < n; i++) {
    uint32_t raw = a[i];
    uint32_t first = table1[raw >> 10];
    first += (raw >> 4) & 0x3f;
    uint32_t second = table2[first];
    second += raw & 0xf;
    uint32_t result = table3[second];
    if (result >= BAD_VALUE) {  // very unlikely
      break;
    }
    b[i] = result;
  }
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Convert UTF-16 to chars
Purpose: Use of gathers load instruction
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
EDGE_SIZES = [0, 1, 3, 1999, 2001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [2000000, 8000000]
PERF_SIZES_C8G = [8000000, 32000000]  # DRAM-bound on Graviton4 (64MB L3)
