/*----------------------------------------------------------------------------
#
#   Loop 012: Particle motion
#
#   Purpose:
#     Use of FP compare instructions.
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "helpers.h"
#include "loops.h"


struct loop_012_data {
  int64_t step;
  double direction[3];
  int64_t magnitude[3];
  double *restrict vx;
  double *restrict vy;
  double *restrict vz;
  double *restrict nx;
  double *restrict ny;
  double *restrict nz;
  uint64_t n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_012(struct loop_012_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static double next_pos(int64_t step, double direction, int64_t magnitude,
                       double value) {
  double pos = direction;
  double vabs = value < 0 ? -value : value;
  double vabsstep =
      (vabs * step) - ((int64_t)(vabs * step) / magnitude) * magnitude;
  double vstep = value < 0 ? -vabsstep : vabsstep;
  pos -= vstep;
  return pos < 0.0 ? pos + magnitude : pos >= magnitude ? pos - magnitude : pos;
}

static void inner_loop_012(struct loop_012_data *restrict data) {
  int64_t step = data->step;
  double *direction = data->direction;
  int64_t *magnitude = data->magnitude;
  double *vx = data->vx;
  double *vy = data->vy;
  double *vz = data->vz;
  double *nx = data->nx;
  double *ny = data->ny;
  double *nz = data->nz;
  uint64_t n = data->n;

  for (int p = 0; p < n; p++) {
    nx[p] = next_pos(step, direction[0], magnitude[0], vx[p]);
    ny[p] = next_pos(step, direction[1], magnitude[1], vy[p]);
    nz[p] = next_pos(step, direction[2], magnitude[2], vz[p]);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_012(struct loop_012_data *restrict data)
LOOP_ATTR
{
  int64_t step = data->step;
  double *direction = data->direction;
  int64_t *magnitude = data->magnitude;
  double *vx = data->vx;
  double *vy = data->vy;
  double *vz = data->vz;
  double *nx = data->nx;
  double *ny = data->ny;
  double *nz = data->nz;
  uint64_t n = data->n;

  svbool_t p0 = svptrue_b64();
  int64_t a = 0;

  float step_d = (float)step;
  svfloat64_t step_vec = svdup_f64(step_d);

  svfloat64_t direction0_vec = svdup_f64(direction[0]);
  svfloat64_t direction1_vec = svdup_f64(direction[1]);
  svfloat64_t direction2_vec = svdup_f64(direction[2]);

  svint64_t magnitude0_int_vec = svdup_s64(magnitude[0]);
  svint64_t magnitude1_int_vec = svdup_s64(magnitude[1]);
  svint64_t magnitude2_int_vec = svdup_s64(magnitude[2]);

  svfloat64_t magnitude0_vec = svcvt_f64_z(p0, magnitude0_int_vec);
  svfloat64_t magnitude1_vec = svcvt_f64_z(p0, magnitude1_int_vec);
  svfloat64_t magnitude2_vec = svcvt_f64_z(p0, magnitude2_int_vec);

  svfloat64_t vx_vec, vy_vec, vz_vec;
  svfloat64_t vmul_vec;

  svfloat64_t absstep_vec, vstep_vec, res_add_vec, res_sub_vec;
  svfloat64_t direction0_tmp_vec, direction1_tmp_vec, direction2_tmp_vec;

  svint64_t vmul_int_vec;

  svbool_t p1, p2;

  for (uint64_t p = 0; p < n; p += svcntd()) {
    svbool_t p4 = svwhilelt_b64(p, n);
    vx_vec = svld1_vnum(p4, vx, a);
    p1 = svcmplt(p0, vx_vec, 0.0);
    vx_vec = svabs_x(p0, vx_vec);
    vmul_vec = svmul_x(p0, vx_vec, step_vec);
    vmul_int_vec = svcvt_s64_x(p0, vmul_vec);
    vmul_int_vec = svdiv_x(p0, vmul_int_vec, magnitude0_int_vec);
    vmul_int_vec = svmul_x(p0, vmul_int_vec, magnitude0_int_vec);
    vx_vec = svcvt_f64_x(p0, vmul_int_vec);
    absstep_vec = svsub_x(p0, vmul_vec, vx_vec);
    vstep_vec = svneg_m(absstep_vec, p1, absstep_vec);
    direction0_tmp_vec = svsub_x(p0, direction0_vec, vstep_vec);
    p1 = svcmplt(p4, direction0_tmp_vec, 0.0);
    p2 = svcmpge(p4, direction0_tmp_vec, magnitude0_vec);
    res_add_vec = svadd_x(p0, direction0_tmp_vec, magnitude0_vec);
    res_sub_vec = svsub_m(p2, direction0_tmp_vec, magnitude0_vec);
    vx_vec = svsel(p1, res_add_vec, res_sub_vec);
    svst1_vnum(p4, nx, a, vx_vec);

    vy_vec = svld1_vnum(p4, vy, a);
    p1 = svcmplt(p0, vy_vec, 0.0);
    vy_vec = svabs_x(p0, vy_vec);
    vmul_vec = svmul_x(p0, vy_vec, step_vec);
    vmul_int_vec = svcvt_s64_x(p0, vmul_vec);
    vmul_int_vec = svdiv_x(p0, vmul_int_vec, magnitude1_int_vec);
    vmul_int_vec = svmul_x(p0, vmul_int_vec, magnitude1_int_vec);
    vy_vec = svcvt_f64_x(p0, vmul_int_vec);
    absstep_vec = svsub_x(p0, vmul_vec, vy_vec);
    vstep_vec = svneg_m(absstep_vec, p1, absstep_vec);
    direction1_tmp_vec = svsub_x(p0, direction1_vec, vstep_vec);
    p1 = svcmplt(p4, direction1_tmp_vec, 0.0);
    p2 = svcmpge(p4, direction1_tmp_vec, magnitude1_vec);
    res_add_vec = svadd_x(p0, direction1_tmp_vec, magnitude1_vec);
    res_sub_vec = svsub_m(p2, direction1_tmp_vec, magnitude1_vec);
    vy_vec = svsel(p1, res_add_vec, res_sub_vec);
    svst1_vnum(p4, ny, a, vy_vec);

    vz_vec = svld1_vnum(p4, vz, a);
    p1 = svcmplt(p0, vz_vec, 0.0);
    vz_vec = svabs_x(p0, vz_vec);
    vmul_vec = svmul_x(p0, vz_vec, step_vec);
    vmul_int_vec = svcvt_s64_x(p0, vmul_vec);
    vmul_int_vec = svdiv_x(p0, vmul_int_vec, magnitude2_int_vec);
    vmul_int_vec = svmul_x(p0, vmul_int_vec, magnitude2_int_vec);
    vz_vec = svcvt_f64_x(p0, vmul_int_vec);
    absstep_vec = svsub_x(p0, vmul_vec, vz_vec);
    vstep_vec = svneg_m(absstep_vec, p1, absstep_vec);
    direction2_tmp_vec = svsub_x(p0, direction2_vec, vstep_vec);
    p1 = svcmplt(p4, direction2_tmp_vec, 0.0);
    p2 = svcmpge(p4, direction2_tmp_vec, magnitude2_vec);
    res_add_vec = svadd_x(p0, direction2_tmp_vec, magnitude2_vec);
    res_sub_vec = svsub_x(p2, direction2_tmp_vec, magnitude2_vec);
    vz_vec = svsel(p1, res_add_vec, res_sub_vec);
    svst1_vnum(p4, nz, a, vz_vec);

    a++;
  }
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))
static void inner_loop_012(struct loop_012_data *restrict data)
LOOP_ATTR
{
  int64_t step = data->step;
  double *direction = data->direction;
  int64_t *magnitude = data->magnitude;
  double *vx = data->vx;
  double *vy = data->vy;
  double *vz = data->vz;
  double *nx = data->nx;
  double *ny = data->ny;
  double *nz = data->nz;
  uint64_t n = data->n;

  uint64_t p;

  asm volatile(
      "     mov     %[p], #0                                \n"
      "     ptrue   p0.d                                    \n"
      "     scvtf   d29, %[step]                            \n"
      "     mov     z29.d, d29                              \n"
      "     ld1rd   {z20.d}, p0/z, [%[direction]]           \n"
      "     ld1rd   {z21.d}, p0/z, [%[direction], #8]       \n"
      "     ld1rd   {z22.d}, p0/z, [%[direction], #16]      \n"
      "     ld1rd   {z23.d}, p0/z, [%[magnitude]]           \n"
      "     ld1rd   {z24.d}, p0/z, [%[magnitude], #8]       \n"
      "     ld1rd   {z25.d}, p0/z, [%[magnitude], #16]      \n"
      "     scvtf   z26.d, p0/m, z23.d                      \n"
      "     scvtf   z27.d, p0/m, z24.d                      \n"
      "     scvtf   z28.d, p0/m, z25.d                      \n"
      "     b       2f                                      \n"

      "1:   ld1d    {z0.d}, p4/z, [%[vx], %[p], lsl #3]     \n"  // load value
      "     fcmlt   p1.d, p0/z, z0.d, #0.0                  \n"  // value < 0 ?
      "     fabs    z0.d, p0/m, z0.d                        \n"  // vabs
      "     fmul    z2.d, z0.d, z29.d                       \n"  // vabs * step
      "     fcvtzs  z0.d, p0/m, z2.d                        \n"  // to int64_t
      "     sdiv    z0.d, p0/m, z0.d, z23.d                 \n"  // / magnitude
      "     mul     z0.d, p0/m, z0.d, z23.d                 \n"  // * magnitude
      "     scvtf   z0.d, p0/m, z0.d                        \n"  // to double
      "     fsub    z4.d, z2.d, z0.d                        \n"  // vabsstep
      "     fneg    z4.d, p1/m, z4.d                        \n"  // vstep
      "     fsub    z4.d, z20.d, z4.d                       \n"  // pos -= vstep
      "     fcmlt   p1.d, p4/z, z4.d, #0.0                  \n"  // pos < 0.0 ?
      "     fcmge   p2.d, p4/z, z4.d, z26.d                 \n"  // pos >= mgnt?
      "     fadd    z1.d, z4.d, z26.d                       \n"  // pos + magnt
      "     fsub    z4.d, p2/m, z4.d, z26.d                 \n"  // pos - magnt
      "     sel     z2.d, p1, z1.d, z4.d                    \n"
      "     st1d    {z2.d}, p4, [%[nx], %[p], lsl #3]       \n"

      "     ld1d    {z0.d}, p4/z, [%[vy], %[p], lsl #3]     \n"  // load value
      "     fcmlt   p1.d, p0/z, z0.d, #0.0                  \n"  // value < 0 ?
      "     fabs    z0.d, p0/m, z0.d                        \n"  // vabs
      "     fmul    z2.d, z0.d, z29.d                       \n"  // vabs * step
      "     fcvtzs  z0.d, p0/m, z2.d                        \n"  // to int64_t
      "     sdiv    z0.d, p0/m, z0.d, z24.d                 \n"  // / magnitude
      "     mul     z0.d, p0/m, z0.d, z24.d                 \n"  // * magnitude
      "     scvtf   z0.d, p0/m, z0.d                        \n"  // to double
      "     fsub    z4.d, z2.d, z0.d                        \n"  // vabsstep
      "     fneg    z4.d, p1/m, z4.d                        \n"  // vstep
      "     fsub    z4.d, z21.d, z4.d                       \n"  // pos -= vstep
      "     fcmlt   p1.d, p4/z, z4.d, #0.0                  \n"  // pos < 0.0 ?
      "     fcmge   p3.d, p4/z, z4.d, z27.d                 \n"  // pos >= mgnt?
      "     fadd    z1.d, z4.d, z27.d                       \n"  // pos + magnt
      "     fsub    z4.d, p3/m, z4.d, z27.d                 \n"  // pos - magnt
      "     sel     z0.d, p1, z1.d, z4.d                    \n"
      "     st1d    {z0.d}, p4, [%[ny], %[p], lsl #3]       \n"

      "     ld1d    {z0.d}, p4/z, [%[vz], %[p], lsl #3]     \n"  // load value
      "     fcmlt   p1.d, p0/z, z0.d, #0.0                  \n"  // value < 0 ?
      "     fabs    z0.d, p0/m, z0.d                        \n"  // vabs
      "     fmul    z2.d, z0.d, z29.d                       \n"  // vabs * step
      "     fcvtzs  z0.d, p0/m, z2.d                        \n"  // to int64_t
      "     sdiv    z0.d, p0/m, z0.d, z25.d                 \n"  // / magnitude
      "     mul     z0.d, p0/m, z0.d, z25.d                 \n"  // * magnitude
      "     scvtf   z0.d, p0/m, z0.d                        \n"  // to double
      "     fsub    z4.d, z2.d, z0.d                        \n"  // vabsstep
      "     fneg    z4.d, p1/m, z4.d                        \n"  // vstep
      "     fsub    z4.d, z22.d, z4.d                       \n"  // pos -= vstep
      "     fcmlt   p1.d, p4/z, z4.d, #0.0                  \n"  // pos < 0.0 ?
      "     fcmge   p5.d, p4/z, z4.d, z28.d                 \n"  // pos >= mgnt?
      "     fadd    z1.d, z4.d, z28.d                       \n"  // pos + magnt
      "     fsub    z4.d, p5/m, z4.d, z28.d                 \n"  // pos - magnt
      "     sel     z0.d, p1, z1.d, z4.d                    \n"
      "     st1d    {z0.d}, p4, [%[nz], %[p], lsl #3]       \n"

      "     incd    %[p]                                    \n"
      "2:   whilelo p4.d, %[p], %[n]                        \n"
      "     b.any   1b                                      \n"
      // output operands, source operands, and clobber list
      : [p] "=&r"(p)
      : [step] "r"(step), [direction] "r"(direction),
        [magnitude] "r"(magnitude), [vx] "r"(vx), [vy] "r"(vy), [vz] "r"(vz),
        [nx] "r"(nx), [ny] "r"(ny), [nz] "r"(nz), [n] "r"(n)
      : "v1", "v2", "v4", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "p0", "p1", "p4", "cc", "memory");
}
#else
static void inner_loop_012(struct loop_012_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 1000
#endif

LOOP_DECL(012, SC_SVE_LOOP_ATTR)
{
  struct loop_012_data data = {
    .step = 7,
    .direction = {1, 2, 3},
    .magnitude = {2, 4, 6},
    .n = SIZE,
  };

  ALLOC_64B(data.vx, SIZE, "velocity X-component");
  ALLOC_64B(data.vy, SIZE, "velocity Y-component");
  ALLOC_64B(data.vz, SIZE, "velocity Z-component");
  ALLOC_64B(data.nx, SIZE, "position X-component");
  ALLOC_64B(data.ny, SIZE, "position Y-component");
  ALLOC_64B(data.nz, SIZE, "position Z-component");

  fill_double_range(data.vx, SIZE, -2.0, 2.0);
  fill_double_range(data.vy, SIZE, -2.0, 2.0);
  fill_double_range(data.vz, SIZE, -2.0, 2.0);
  fill_double(data.nx, SIZE);
  fill_double(data.ny, SIZE);
  fill_double(data.nz, SIZE);

  inner_loops_012(iters, &data);

  double res = 0;
  for (int i = 0; i < SIZE; i++) {
    res += i * (data.nx[i] + data.ny[i] + data.nz[i]);
  }

  double correct = 2951635.0;

  bool passed = check_double(res, correct, 1.0);
#ifndef STANDALONE
  FINALISE_LOOP_F(12, passed, "%9.6f", correct, 1.0f, res)
#endif
  return passed ? 0 : 1;
}
