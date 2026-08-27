# SVE / SVE2 C-intrinsics cheat-sheet (arm_sve.h, clang-18, Neoverse V2 / Graviton4)

This is the **signature** reference — how to *call* SVE intrinsics correctly in C.
(The Neoverse SWOG docs tell you which instruction is *fast*; this tells you how to
*write* it. Read this before hand-writing SVE — it prevents the most common
compile errors.) `#include <arm_sve.h>`. All types are SVE **scalable vectors**.

## The 3 rules that cause ~90% of SVE compile errors

1. **SVE vector types are SIZELESS** (`svfloat32_t`, `svint32_t`, `svbfloat16_t`, `svbool_t`, …).
   You **cannot** put them in an **array**, a **struct**, `sizeof`, or a global/static.
   - ❌ `svint32_t acc[8];`  → *"array has sizeless element type 'svint32_t'"*
   - ✅ declare them as separate named locals: `svint32_t acc0, acc1, acc2, ...;`
     (or restructure so accumulators are distinct variables, not indexed storage).

2. **Predication suffix `_x` / `_z` / `_m` is mandatory** on most math intrinsics and
   picks what happens in inactive lanes: `_x` = don't-care (fastest, use this for
   dense compute), `_z` = zero, `_m` = merge (keep the first operand's lanes).
   Nearly all take a governing predicate `svbool_t pg` as the **first** argument.

3. **The `_n` suffix means "the last vector operand is a SCALAR"** (broadcast).
   The scalar goes in the position of the operand it replaces — usually **last**.
   Getting this position wrong is the other classic error (see svmad below).

## Loads / stores (predicate first, then pointer)
```c
svfloat32_t svld1_f32 (svbool_t pg, const float32_t *p);      // contiguous f32 load
void        svst1_f32 (svbool_t pg, float32_t *p, svfloat32_t v);
svint32_t   svld1sb_s32(svbool_t pg, const int8_t  *p);       // WIDENING load: int8 -> int32 lanes
svbfloat16_t svld1_bf16(svbool_t pg, const bfloat16_t *p);    // bf16 load (reinterpret uint16_t* if needed)
svuint16_t  svld1_u16 (svbool_t pg, const uint16_t *p);
```

## Fused multiply-add family (watch the operand order + `_n`)
```c
// svmla: acc = acc + a*b     (accumulator is the FIRST vector operand)
svfloat32_t svmla_f32_x  (svbool_t pg, svfloat32_t acc, svfloat32_t a, svfloat32_t b);
svfloat32_t svmla_n_f32_x(svbool_t pg, svfloat32_t acc, svfloat32_t a, float32_t  b); // b is a SCALAR

// svmad: acc = a*b + c       (order is a, b, c — different from svmla!)
svfloat32_t svmad_f32_x  (svbool_t pg, svfloat32_t a, svfloat32_t b, svfloat32_t c);
svfloat32_t svmad_n_f32_x(svbool_t pg, svfloat32_t a, svfloat32_t b, float32_t  c); // c (LAST) is the SCALAR
```
❌ Real mistake seen: `svmad_n_f32_x(pg, v0, scaleScalar, vb)` →
*"no known conversion from 'const float' to 'svfloat32_t' for 3rd argument"*.
The `_n` scalar must be the **last** arg; the 3rd arg (`b`) is still a **vector**.
✅ `svmad_n_f32_x(pg, v0, vb, scaleScalar)`  (or drop `_n` and pass a broadcast vector).

## Dot products — NO predicate argument (this trips people up)
```c
svint32_t   svdot_s32  (svint32_t   acc, svint8_t      a, svint8_t      b); // int8·int8 -> int32, 4-way
svuint32_t  svdot_u32  (svuint32_t  acc, svuint8_t     a, svuint8_t     b);
svfloat32_t svbfdot_f32(svfloat32_t acc, svbfloat16_t  a, svbfloat16_t  b); // bf16·bf16 -> f32, 2-way
```
These take **(accumulator, a, b)** and **no `svbool_t`** — they operate on the whole
vector. `acc` is both input and result (accumulate across calls).

## Matmul (widening outer-product accumulate) — also no predicate
```c
svint32_t   svmmla_s32   (svint32_t   acc, svint8_t     a, svint8_t     b); // needs +i8mm
svfloat32_t svbfmmla_f32 (svfloat32_t acc, svbfloat16_t a, svbfloat16_t b); // needs +bf16
```

## Predicates, counts, tails
```c
svbool_t svptrue_b32 (void);                       // all-true predicate for 32-bit lanes
svbool_t svwhilelt_b32(int32_t i, int32_t n);      // active while i < n — use for loop tails, no scalar cleanup
uint64_t svcntw (void);                            // # of 32-bit lanes this vector holds (VL/32)
// idiomatic strided loop:
for (uint64_t i = 0; i < n; i += svcntw()) {
    svbool_t pg = svwhilelt_b32((int32_t)i, (int32_t)n);
    svfloat32_t v = svld1_f32(pg, p + i);
    ...
    svst1_f32(pg, q + i, v);
}
```

## Conversions / reinterprets
```c
svfloat32_t svcvt_f32_bf16_x(svbool_t pg, svbfloat16_t a); // bf16 -> f32 widen
svbfloat16_t svcvt_bf16_f32_x(svbool_t pg, svfloat32_t a); // f32  -> bf16 narrow (rounds)
float32_t   svaddv_f32(svbool_t pg, svfloat32_t a);        // horizontal sum (reduction)
```

## Required `-march` for the instruction families
- bf16 (`svbfdot`, `svbfmmla`): `+bf16`
- int8 matmul (`svmmla_s32`): `+i8mm`
- typical: `-march=armv9-a+sve2+bf16+i8mm` (or `-mcpu=neoverse-v2`).

**When a call won't compile:** it's almost always (1) a sizeless type in an
array/struct, (2) a missing/`wrong` `_x/_z/_m` suffix, (3) the `_n` scalar in the
wrong position, or (4) a missing `-march` feature for that instruction.
