#include "../ops.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "../elementwise/binary-ops.h"
#include "../simd-gemm.h"
#include "ggml.h"
#include "../activations/unary-ops.h"
#include "../gemm_gemv/vec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

static void ggml_compute_forward_gelu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

static void ggml_compute_forward_gelu_erf_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_erf_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_erf(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

static void ggml_compute_forward_gelu_quick_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_quick_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_quick(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

static void ggml_compute_forward_leaky_relu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_contiguous_1(src0));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src0, dst));

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

static void ggml_compute_forward_leaky_relu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_contiguous_1(src0));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src0, dst));

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(ggml_fp16_t));
    assert(src0->nb[0] == sizeof(ggml_fp16_t));

    for (int i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

void ggml_compute_forward_leaky_relu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

static void ggml_compute_forward_silu_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * grad = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous_1(grad));
    assert(ggml_is_contiguous_1(src1));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src1, dst));
    assert(ggml_are_same_shape(src1, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1->ne[0];
    const int nr = ggml_nrows(src1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src1->data + i1*(src1->nb[1])),
                (float *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu_back_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * grad = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous_1(grad));
    assert(ggml_is_contiguous_1(src1));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src1, dst));
    assert(ggml_are_same_shape(src1, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1->ne[0];
    const int nr = ggml_nrows(src1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((char *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

void ggml_compute_forward_silu_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

static void ggml_compute_forward_reglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_reglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_reglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

static void ggml_compute_forward_geglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

static void ggml_compute_forward_swiglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

static void ggml_compute_forward_swiglu_oai_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);
        float * dst_p  = (float *) ((char *) dst->data + i1*(dst->nb[1]));

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        for (int k = 0; k < nc; k++) {
            const float x = std::min(src0_p[k], limit);
            const float y = std::clamp(src1_p[k], -limit, limit);
            const float out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = dst_p[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu_oai(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_erf

static void ggml_compute_forward_geglu_erf_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_erf_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_erf(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

static void ggml_compute_forward_geglu_quick_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_quick_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_quick(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

void ggml_compute_forward_unary(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_unary_op op = ggml_get_unary_op(dst);

    switch (op) {
        case GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } break;
        case GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } break;
        case GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } break;
        case GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } break;
        case GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } break;
        case GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } break;
        case GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } break;
        case GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } break;
        case GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } break;
        case GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } break;
        case GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } break;
        case GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } break;
        case GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } break;
        case GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } break;
        case GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } break;
        case GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } break;
        case GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } break;
        case GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } break;
        case GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } break;
        case GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } break;
        case GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } break;
        case GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

void ggml_compute_forward_glu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_glu_op op = ggml_get_glu_op(dst);

    switch (op) {
        case GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } break;
        case GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } break;
        case GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

