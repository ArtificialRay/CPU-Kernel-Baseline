#include "cpu_types.hpp"

namespace {
template <typename scalar_t>
struct KernelVecType {
  using load_vec_type = void;
  using cvt_vec_type = void;
};

template <>
struct KernelVecType<float> {
  using load_vec_type = vec_op::FP32Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using load_vec_type = vec_op::BF16Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::Half> {
#if defined(__powerpc64__) || defined(__s390x__)
  // Power architecture-specific vector type
  using load_vec_type = vec_op::FP32Vec16;
#else
  // Fallback for other architectures
  using load_vec_type = vec_op::FP16Vec16;
#endif
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <bool AZP, typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int32_t* azp,
                                   const int64_t num_tokens,
                                   const int64_t input_stride,
                                   const int64_t hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int64_t vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  const cvt_vec_t inv_scale(1.0 / *scale);
  const cvt_vec_t i8_min_vec(i8_min);
  const cvt_vec_t i8_max_vec(i8_max);

  cvt_vec_t zp_vec;
  if constexpr (AZP) {
    zp_vec = cvt_vec_t(static_cast<float>(*azp));
  }

#pragma omp parallel for
  for (int64_t i = 0; i < num_tokens; ++i) {
    int64_t j = 0;
    const scalar_t* input_ptr = input + i * input_stride;
    int8_t* output_ptr = output + i * hidden_size;
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = elems_fp32 * inv_scale;

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + zp_vec;
      }

      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output_ptr + j);
    }

    load_vec_t elems(input_ptr + j);
    cvt_vec_t elems_fp32(elems);
    elems_fp32 = elems_fp32 * inv_scale;

    if constexpr (AZP) {
      elems_fp32 = elems_fp32 + zp_vec;
    }

    elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
    vec_op::INT8Vec16 elems_int8(elems_fp32);
    elems_int8.save(output_ptr + j, hidden_size - j);
  }
}

template <bool AZP, typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, int32_t* azp,
                                    const int64_t num_tokens,
                                    const int64_t input_stride,
                                    const int64_t hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  const cvt_vec_t i8_min_vec(i8_min);
  const cvt_vec_t i8_max_vec(i8_max);

#pragma omp parallel for
  for (int64_t i = 0; i < num_tokens; ++i) {
    cvt_vec_t max_value(std::numeric_limits<float>::lowest());
    cvt_vec_t min_value(std::numeric_limits<float>::max());
    {
      int64_t j = 0;
      const scalar_t* input_ptr = input + i * input_stride;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input_ptr + j);
        cvt_vec_t elems_fp32(elems);
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32);
          min_value = min_value.min(elems_fp32);
        } else {
          max_value = max_value.max(elems_fp32.abs());
        }
      }

      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);

      if (j + vec_elem_num == hidden_size) {
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32);
          min_value = min_value.min(elems_fp32);
        } else {
          max_value = max_value.max(elems_fp32.abs());
        }
      } else {
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32, hidden_size - j);
          min_value = min_value.min(elems_fp32, hidden_size - j);
        } else {
          max_value = max_value.max(elems_fp32.abs(), hidden_size - j);
        }
      }
    }

    float scale_val;
    float azp_val = 0.0f;
    if constexpr (AZP) {
      float max_scalar = max_value.reduce_max();
      float min_scalar = min_value.reduce_min();
      scale_val = (max_scalar - min_scalar) / 255.0f;
      azp_val = std::nearbyint(-128.0f - min_scalar / scale_val);
      azp[i] = azp_val;
      scale[i] = scale_val;
    } else {
      scale_val = max_value.reduce_max() / 127.0f;
      scale[i] = scale_val;
    }

    const cvt_vec_t inv_scale(1.0 / scale_val);
    const cvt_vec_t azp_vec(azp_val);

    {
      int64_t j = 0;
      const scalar_t* input_ptr = input + i * input_stride;
      int8_t* output_ptr = output + i * hidden_size;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input_ptr + j);
        cvt_vec_t elems_fp32(elems);
        elems_fp32 = (elems_fp32 * inv_scale);

        if constexpr (AZP) {
          elems_fp32 = elems_fp32 + azp_vec;
        }
        elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
        vec_op::INT8Vec16 elems_int8(elems_fp32);
        elems_int8.save(output_ptr + j);
      }

      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = (elems_fp32 * inv_scale);

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + azp_vec;
      }
      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output_ptr + j, hidden_size - j);
    }
  }
}
}  // namespace

// static-per-tensor quantization.
void static_scaled_int8_quant(
    torch::Tensor& out,          // [batch, hidden_size]
    const torch::Tensor& input,  // [batch, hidden_size]
    const torch::Tensor& scale, std::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(static_scaled_int8_quant)
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK_EQ(input.dim(), 2);
  TORCH_CHECK_EQ(input.stride(1), 1);
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp.has_value() || azp->numel() == 1);

  const int64_t stride = input.stride(0);
  const int64_t hidden_size = input.size(1);
  const int64_t num_tokens = input.size(0);
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          static_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              stride, hidden_size);
        } else {
          static_scaled_int8_quant_impl<false>(input.data_ptr<scalar_t>(),
                                               out.data_ptr<int8_t>(),
                                               scale.data_ptr<float>(), nullptr,
                                               num_tokens, stride, hidden_size);
        }
      });
}

// dynamic-per-token quantization.
void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [batch, hidden_size]
    const torch::Tensor& input,  // [batch, hidden_size]
    torch::Tensor& scale,        // [batch, 1]
    std::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(dynamic_scaled_int8_quant)
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK_EQ(input.dim(), 2);
  TORCH_CHECK_EQ(input.stride(1), 1);

  const int64_t hidden_size = input.size(1);
  const int64_t num_tokens = input.size(0);
  const int64_t stride = input.stride(0);
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          dynamic_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              stride, hidden_size);
        } else {
          dynamic_scaled_int8_quant_impl<false>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), nullptr, num_tokens, stride,
              hidden_size);
        }
      });
}
