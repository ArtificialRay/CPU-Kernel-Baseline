"""
scripts/update_descriptions.py — Replace instruction-hint descriptions with
algorithmic descriptions in all dataset/problems/*/problem.py files.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROBLEMS_DIR = REPO_ROOT / "dataset" / "problems"

# Maps dir_name prefix → realistic description
DESCRIPTIONS = {
    "loop_001": "Compute the FP32 dot product of two float arrays",
    "loop_002": "Compute the integer dot product of two uint32 arrays",
    "loop_003": "Compute the FP64 dot product of two double arrays",
    "loop_004": "Compute the integer dot product of two uint64 arrays",
    "loop_005": "Compute the length of many short null-terminated strings",
    "loop_006": "Compute the length of many long null-terminated strings",
    "loop_008": "Sum a double array with exact scalar FP ordering — result must be bit-identical to sequential scalar addition",
    "loop_009": "Traverse a linked list and XOR-accumulate the payload from each node",
    "loop_010": "Find the last active element of an FP array under a computed predicate",
    "loop_012": "Advance 3D particle positions by a time step with modular boundary wrapping",
    "loop_019": "Write a marker value to a set of scattered (non-contiguous) memory locations",
    "loop_022": "Compute one's complement checksum across a buffer of variable-length TCP packets",
    "loop_023": "Sparse matrix-vector multiply where column indices are stored as an indirect index array",
    "loop_024": "Sum of absolute differences between two byte arrays, accumulated into a uint32",
    "loop_025": "Multiply two small fixed-size FP32 matrices",
    "loop_026": "Convert a UTF-16 string to narrow ASCII chars, skipping high surrogate pairs",
    "loop_027": "Compute element-wise square root of an FP32 array",
    "loop_028": "Compute element-wise fast approximate reciprocal division of FP64 arrays",
    "loop_029": "Scale each FP64 element by an integer power of two",
    "loop_031": "Copy small byte buffers of varied lengths and alignments",
    "loop_032": "Solve a step of a banded linear system with non-unit-strided coefficient access",
    "loop_033": "Compute FP64 inner product of two arrays",
    "loop_034": "Compare many short null-terminated strings for equality",
    "loop_035": "Element-wise addition of two arrays with arbitrary length (non-multiple-of-vector-width)",
    "loop_036": "Apply one Gaussian elimination step to a sparse row using indirect column indices",
    "loop_037": "Element-wise complex multiplication of two FP32 complex-number arrays (interleaved re/im)",
    "loop_038": "1D convolution of an FP16 signal with an FP16 filter kernel",
    "loop_040": "Clamp each element of an integer sequence to data-dependent [min, max] bounds",
    "loop_101": "Upscale a pixel buffer by splitting each element into its high and low halves",
    "loop_102": "Count the frequency of each byte value in a large buffer (histogram)",
    "loop_103": "Find all whitespace character positions in a byte string",
    "loop_104": "Compute byte-value frequency histogram using segmented counting",
    "loop_105": "Sum adjacent FP value pairs in a cascading pairwise reduction",
    "loop_106": "Partition a vector by bit flag, concentrating set-bit elements to one side",
    "loop_107": "Multiply 64-bit integer pairs producing full 128-bit results with carry",
    "loop_108": "Deinterleave RGBA pixel data and accumulate channel values with bit shifts",
    "loop_109": "Element-wise addition of complex numbers stored as interleaved uint32 (re, im) pairs",
    "loop_110": "Compute complex dot product of two arrays of uint32 complex numbers",
    "loop_111": "Normalise FP64 values that may overflow by detecting and rescaling their exponents",
    "loop_112": "Complex multiply-accumulate over arrays of uint32 complex numbers",
    "loop_113": "Sum adjacent uint32 pairs across an array (horizontal pairwise add)",
    "loop_114": "Compute auto-correlation of an integer array with widening accumulation",
    "loop_120": "Sort a small integer array using insertion sort",
    "loop_121": "Partition an integer array around a pivot (one pass of quicksort)",
    "loop_122": "Sort element pairs using odd-even transposition compare-and-swap",
    "loop_123": "Merge two sorted halves using a bitonic compare-and-swap network",
    "loop_124": "Sort integers by digit using a single radix counting pass",
    "loop_126": "Conditionally update array elements where a per-element predicate holds",
    "loop_127": "Search an array for the first element satisfying a condition, exit immediately on match",
    "loop_128": "Shift array elements in place where source and destination ranges may overlap",
    "loop_130": "Multiply two FP32 matrices using tiled register accumulation",
    "loop_135": "Multiply INT8 matrices accumulating into INT32 using tiled 4-element dot products",
    "loop_136": "Multiply 4-bit quantized matrices into INT32 using lookup-table dequantization",
    "loop_137": "Multiply BF16 matrices accumulating into FP32 using tiled dot products",
    "loop_201": "Multiply two FP64 matrices using tiled outer-product accumulation",
    "loop_202": "Multiply two FP32 matrices using tiled outer-product accumulation",
    "loop_204": "Multiply two FP16 matrices accumulating into FP16 using tiled outer products",
    "loop_205": "Multiply INT8 matrices accumulating into INT32 using tiled outer products or dot products",
    "loop_206": "Multiply INT16 matrices accumulating into INT64 using tiled outer products",
    "loop_207": "Multiply 1-bit binary matrices accumulating into INT32",
    "loop_208": "Multiply BF16 matrices accumulating into BF16 using tiled outer products",
    "loop_210": "Multiply BF16 matrices accumulating into FP32 using tiled outer products or dot products",
    "loop_211": "Multiply INT16 matrices accumulating into INT32 using tiled outer products or dot products",
    "loop_212": "Multiply 4-bit quantized matrices into FP32 using lookup dequantization and dot products",
    "loop_215": "Multiply uint8 matrices accumulating into uint32 using column-major tiled dot products",
    "loop_216": "Multiply an FP32 matrix by an FP32 vector (column-major GEMV)",
    "loop_217": "Multiply an INT8 matrix by an INT8 vector accumulating into INT32 (row-major GEMV)",
    "loop_218": "Multiply an FP64 matrix by an FP64 vector (column-major GEMV)",
    "loop_219": "Multiply an INT8 matrix by an INT8 vector accumulating into INT32 (column-major GEMV)",
    "loop_220": "Multiply an FP32 matrix by an FP32 vector (row-major GEMV)",
    "loop_221": "Multiply an FP64 matrix by an FP64 vector (row-major GEMV)",
    "loop_222": "1D convolution of an FP16 signal with an FP16 filter using multi-vector loads",
    "loop_223": "Transpose a matrix in-place using interleaved load and store",
    "loop_231": "Multiply a BF16 matrix by a BF16 vector accumulating into FP32 (column-major interleaved GEMV)",
    "loop_245": "Multiply INT8 matrices into INT32 using a mix of tiled outer products, dot products, and matrix rearrangement",
}


def update_file(path: Path) -> bool:
    dir_name = path.parent.name          # e.g. "loop_023_conjugate_gradient"
    loop_key = "_".join(dir_name.split("_")[:2])  # e.g. "loop_023"

    new_desc = DESCRIPTIONS.get(loop_key)
    if new_desc is None:
        print(f"  SKIP {dir_name} — no mapping defined")
        return False

    text = path.read_text()
    # Replace in METADATA dict
    new_text = re.sub(
        r'("description":\s*)"[^"]*"',
        rf'\1"{new_desc}"',
        text,
    )
    if new_text == text:
        print(f"  UNCHANGED {dir_name}")
        return False

    path.write_text(new_text)
    print(f"  UPDATED  {dir_name}")
    return True


if __name__ == "__main__":
    updated = 0
    for problem_py in sorted(PROBLEMS_DIR.glob("*/problem.py")):
        if update_file(problem_py):
            updated += 1
    print(f"\nDone — {updated} files updated.")
