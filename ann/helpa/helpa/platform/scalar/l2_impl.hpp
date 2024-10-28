#pragma once

#include "helpa/ref/l2_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float l2_fp32_fp32(const float *x, const float *y, const int32_t d) {
  return l2_fp32_fp32_ref(x, y, d);
}

inline float l2_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  return l2_fp32_fp16_ref(x, y, d);
}

inline float l2_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  return l2_fp16_fp16_ref(x, y, d);
}

inline float l2_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  return l2_fp32_bf16_ref(x, y, d);
}

inline float l2_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  return l2_bf16_bf16_ref(x, y, d);
}

inline int32_t l2_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  return l2_s8_s8_ref(x, y, d);
}

inline float l2a_fp32_fp32(const float *x, const float *y, const int32_t d) {
  return l2_fp32_fp32(x, y, d);
}

inline float l2a_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  return l2_fp32_fp16(x, y, d);
}

inline float l2a_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  return l2_fp16_fp16(x, y, d);
}

inline float l2a_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  return l2_fp32_bf16(x, y, d);
}

inline float l2a_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  return l2_bf16_bf16(x, y, d);
}

inline int32_t l2a_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  return l2_s8_s8(x, y, d);
}

} // namespace helpa
