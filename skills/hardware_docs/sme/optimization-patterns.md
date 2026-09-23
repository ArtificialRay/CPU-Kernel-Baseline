# SME/SME2 Optimization Patterns

These patterns summarize the algorithmic ideas in Arm's SME Programmer's Guide examples. They are starting points for design, not guaranteed fastest kernels. Benchmark on the deployment target.

## Workload map

| Workload / data | Candidate approach | Main design question |
|---|---|---|
| Dense FP32 matrix × matrix | SME FP32 outer-product accumulation (guide example: `FMOPA`) | Can panel packing and ZA tile reuse amortize over the K reduction? |
| Unsigned 8-bit matrix × matrix, 32-bit output | SME2 four-way widening outer-product accumulation (guide example: `UMOPA`) | Can both operands be packed/interleaved and K padded safely to the instruction grouping? |
| Matrix × vector, column-major unsigned 8-bit | Accumulate input columns against vector values; SME2 multi-vector operations can cover row groups | Does source layout support efficient contiguous loads, and how large is the output tail? |
| Row-major 2-bit quantized matrix × vector | SME2 lookup-table decompression (`LUTI2`) combined with accumulation | Is on-the-fly unpack/decompression cheaper than materializing expanded data? |
| Complex FP16 matrix × matrix, FP32 accumulation | Two widening outer-product accumulators for real/imaginary components (guide example uses `FMOPA`) | Is the IQ interleaved layout and dual-accumulator organization worthwhile for this workload? |

Instruction names above identify guide examples, not an exhaustive list or a guarantee that every CPU implements the required form. Check the feature variant, operand arrangement, signedness, and exact architecture semantics.

## Data preparation is part of the kernel

SME matrix kernels often gain efficiency by packing inputs into the order consumed by outer-product instructions. Treat preprocessing as part of the algorithm and measure it end to end.

1. Select a tile/block shape in units derived from SVL and the element type.
2. Pack the left and right panels in the order required by the chosen outer-product instruction. A transpose-like panel transform may make the reduction dimension contiguous for one operand; the other may need interleaving.
3. Pad row/column tails and reduction tails with neutral values. For additive matrix products, zero padding preserves results. For quantized formats, decode/sign-extend/zero-point handling must be mathematically correct; raw zero bits are not always the neutral value.
4. Align buffers where useful, but keep loads legal for unaligned/tail inputs. Do not read beyond the allocation just because masked lanes will not contribute.
5. Reuse packed panels when the same operands feed several output tiles. Avoid repacking for each small tile if the preprocessing cost dominates.

## Outer-product matrix-multiply structure

The guide's FP32 and integer examples use the identity that matrix multiplication is a sum of outer products. A practical high-level kernel structure is:

1. Initialize/own the relevant ZA accumulator tile or tiles.
2. Load packed vectors for a block of rows and columns.
3. Issue outer-product accumulate operations across the reduction block, reusing the loaded vectors where possible.
4. Advance using SVL-derived strides and handle the remaining reduction elements safely.
5. Store/extract the completed ZA slices into the output layout, with a correct output-tail predicate.

For integer widening operations, match source signedness and destination width. The guide's unsigned 8-bit example uses a four-way operation, so it rearranges/zero-pads input data and groups four consecutive reduction elements before accumulation. Never assume the grouping of a different instruction is the same.

For floating point, define whether accumulation is FP32/FP64 and the allowed error/rounding tolerance before choosing a kernel. Fused multiply-add and reassociation can change results compared with a scalar reference.

## SME2 multi-vector operations

SME2's grouped Z, ZA-slice, and ZA-array-vector operands can process several vectors in one instruction. Evaluate them when:

- The operation naturally has independent adjacent vectors or widening results.
- Input/output layout can provide those vectors with low shuffle/packing overhead.
- Grouping improves instruction throughput without causing register pressure or spills.
- Predicate-as-counter can express the final partial group correctly.

Prototype the single-vector form first when unsure. Then compare grouped variants and inspect generated assembly for spills, extra moves, and unexpected mode transitions.

## Lookup-table decompression

The guide's compressed matrix-vector example stores four 2-bit values per byte and uses `LUTI2` with SME2's `ZT0` table to expand values while processing. The general pattern is:

1. Define the exact quantization map (including signedness, scale, and zero point).
2. Populate ZT0 with the corresponding decode values and ensure ZT0 state is valid for the function.
3. Load packed indices and use the correct element-size/segment selection for the lookup operation.
4. Feed expanded values directly to arithmetic where possible, avoiding a full intermediate expanded matrix.
5. Compare lookup cost with scalar/SVE unpacking and pre-expansion for realistic reuse and cache behavior.

Do not reuse the guide's particular decode table unless it matches the user's quantization format exactly.

## Tail, layout, and numeric edge cases

- Dimensions need not be multiples of SVL or an instruction's reduction grouping. Use predicates, safe padding, or a scalar cleanup path.
- Account separately for row-major/column-major layouts and leading dimensions. A transpose for computation does not imply that the public API's output layout should change.
- Define aliasing and in-place behavior. Packing buffers must not overlap live inputs unless proven safe.
- Bound integer accumulation. A 32-bit output can overflow even if each product is valid; establish input bounds or widen the accumulator/output.
- Complex arithmetic needs a precise convention for interleaving, conjugation, and output precision. Do not infer conjugation from a complex multiply example.
- Validate degenerate dimensions (zero, one, smaller than a vector), odd K, and every remainder class around the chosen tile/group size.

## Benchmark and review checklist

- Compare scalar/SVE baseline, SME kernel-only, and full path including preprocessing.
- Include small, medium, and large dimensions; hot-cache and representative memory-resident cases.
- Record exact CPU, firmware/OS, compiler version, flags, effective SVL, feature dispatch, data type, layout, and shape.
- Inspect generated assembly and profile memory traffic, packing cost, spills, and mode transition overhead.
- Keep correctness tests independent of the optimized code; use randomized tests plus adversarial values and a tolerance appropriate to the numeric contract.
- State the measured speedup and workload range. Do not generalize one shape or one simulator result to all hardware.

## Guide example index

The source guide's later chapters include these demonstrations:

- `matmul_fp32`: FP32 matrix-by-matrix multiply; left-operand preprocessing and outer-product accumulation.
- `matmul_int8`: unsigned 8-bit inputs to unsigned 32-bit output; right/left preprocessing and four-way widening accumulation.
- `gemv_cm_int8`: column-major unsigned 8-bit matrix-vector multiply.
- `lut_gemv_rm_int8`: row-major 2-bit compressed matrix-vector multiply using lookup-table decompression.
- `cplx_matmul_fp16fp32`: IQ-interleaved complex FP16 inputs with FP32 accumulation and conversion behavior as described in the guide.

Consult the original example for details before translating its code. The examples have explicit layout and input assumptions and should not be copied into a different data contract without adaptation.
