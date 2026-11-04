"""
loop_104: Byte historgram

Purpose: Use of HISTSEG instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_104",
    "num": "104",
    "name": "Byte historgram",
    "description": "Compute byte-value frequency histogram using segmented counting",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_104_byte_historgram",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_104_data {
  uint32_t *histogram;
  uint64_t histogram_size;
  uint8_t *data;
  int n;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
void NOINLINE update(uint32_t *restrict histogram, uint8_t *data, int n) {
  for (int i = 0; i < n; i++) histogram[data[i]] += 1;
}
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Byte historgram
Purpose: Use of HISTSEG instruction
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

# Input sizes for edge-case correctness testing.
# Empty list = skip (loop uses non-SIZE parameters or fixed dimensions).
EDGE_SIZES = []

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = []
