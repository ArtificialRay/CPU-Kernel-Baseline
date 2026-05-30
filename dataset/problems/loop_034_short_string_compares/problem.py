"""
loop_034: Short string compares

Purpose: Use of FFR for strcmp

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_034",
    "num": "034",
    "name": "Short string compares",
    "description": "Compare many short null-terminated strings for equality",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_034_short_string_compares",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_034_data {
  uint8_t *a;
  uint8_t *b;
  uint8_t *lmt;
  uint32_t checksum;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_034(struct loop_034_data *restrict input) {
  uint8_t *p1 = input->a;
  uint8_t *p2 = input->b;
  uint8_t *lmt = input->lmt;

  uint32_t res = 0;
  uint32_t cnt = 0;
  int length = 13;
  while (p1 < lmt) {
    int64_t r = strcmp_opt(p1, p2);
    uint32_t cmp = 1;
    if (r > 0) cmp = 2;
    if (r < 0) cmp = 3;
    res += cnt * cmp;
    p1 += length;
    p2 += length;
    cnt++;
    length = 3 + (length + 11) % 43;
  }
  input->checksum = res;
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Short string compares
Purpose: Use of FFR for strcmp
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
EDGE_SIZES = [1, 5, 10, 5999, 6001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
# Capped at ~18000 (sample_json_size is ~20KB; SIZE >= sample_json_size aborts).
PERF_SIZES = [10000, 18000]
