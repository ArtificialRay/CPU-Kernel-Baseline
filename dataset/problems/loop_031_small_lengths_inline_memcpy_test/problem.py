"""
loop_031: small-lengths inline memcpy test

Purpose: Use of simd-based memcpy for small lengths and at varied alignments

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_031",
    "num": "031",
    "name": "small-lengths inline memcpy test",
    "description": "Copy small byte buffers of varied lengths and alignments",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_031_small_lengths_inline_memcpy_test",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_031_data {
  uint8_t *a;
  uint8_t *b;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
//Do not define
//   static void inline_memcpy(uint8_t *restrict dst, uint8_t *restrict src,int64_t count)
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: small-lengths inline memcpy test
Purpose: Use of simd-based memcpy for small lengths and at varied alignments
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
EDGE_SIZES = [0, 1, 7, 15599, 15601]

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [16000000, 64000000]
