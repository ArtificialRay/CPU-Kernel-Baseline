"""
loop_036: Sparse matrix Gauss Step

Purpose: Use of gather-based matrix processing

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_036",
    "num": "036",
    "name": "Sparse matrix Gauss Step",
    "description": "Apply one Gaussian elimination step to a sparse row using indirect column indices",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_036_sparse_matrix_gauss_step",
    "tags": ['sve2', 'matmul'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_036_data {
  double *restrict matrix;
  int32_t *restrict indexes;
  int32_t *restrict non_zeros;
  double *restrict diag;
  double *restrict xv;
  double *restrict rv;
  double *restrict res;
  int64_t dim;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_036(struct loop_036_data *restrict input) {
  double *matrix = input->matrix;
  int32_t *indexes = input->indexes;
  int32_t *non_zeros = input->non_zeros;
  double *diag = input->diag;
  double *xv = input->xv;
  double *rv = input->rv;
  double *res = input->res;
  int64_t dim = input->dim;

  for (int i = 0; i < dim; i++) {
    const double *const row = matrix + (i * dim);
    const int32_t *const row_indexes = indexes + (i * dim);
    const int32_t row_non_zeros = non_zeros[i];
    const double row_diag = diag[i];
    double sum = rv[i];

    for (int32_t j = 0; j < row_non_zeros; j++) {
      int32_t col = row_indexes[j];
      sum -= row[col] * xv[j];
    }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Sparse matrix Gauss Step
Purpose: Use of gather-based matrix processing
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
EDGE_SIZES = []

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = []
