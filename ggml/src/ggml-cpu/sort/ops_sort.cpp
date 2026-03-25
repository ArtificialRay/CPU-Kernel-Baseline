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

static void ggml_compute_forward_argmax_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    for (int64_t i1 = 0; i1 < ne01; i1++) {
        float * src = (float *) ((char *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((char *)  dst->data + i1*nb0);
        int v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

void ggml_compute_forward_argmax(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

struct cmp_argsort {
    const float * data;
    bool operator()(int32_t a, int32_t b) const {
        if constexpr (order == GGML_SORT_ORDER_ASC) {
            return data[a] < data[b];
        } else {
            return data[a] > data[b];
        }
    }
};

static void ggml_compute_forward_argsort_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    for (int64_t i = ith; i < nr; i += nth) {
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        switch (order) {
            case GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                break;

            case GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                break;

            default:
                GGML_ABORT("invalid sort order");
        }
    }
}

void ggml_compute_forward_argsort(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

struct cmp_top_k {
    const float * data;
    bool operator()(int32_t a, int32_t b) const {
        return data[a] > data[b];
    }
};

static void ggml_compute_forward_top_k_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = ggml_nrows(src0);

    const int top_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t i = ith; i < nr; i += nth) {
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        for (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + top_k, tmp + ne00, cmp_top_k{src_data});

        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);

        std::copy(tmp, tmp + top_k, dst_data);

        // emphasize that the order is not important
        if (top_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

void ggml_compute_forward_top_k(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_top_k_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

