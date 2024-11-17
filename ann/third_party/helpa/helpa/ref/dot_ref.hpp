#pragma once

#include "ann/third_party/helpa/helpa/types.hpp"
#include "immintrin.h"

namespace helpa {

inline float reduce_add_f32x16(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum = _mm256_add_ps(low, high);
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm256_castps256_ps128(sum));
}

inline float dot_fp32_fp32_ref(const float *x, const float *y,
                               const int32_t d) {
    __m512 sum = _mm512_setzero_ps();
    int aligned_d = d & ~15;
    for (int i = 0; i < aligned_d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto yy = _mm512_loadu_ps(y + i);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }

    float tail_sum = 0.0f;
    for (int i = aligned_d; i < d; ++i) {
        tail_sum += x[i] * y[i];
    }

    return -reduce_add_f32x16(sum) - tail_sum;
}

inline float dot_fp32_fp16_ref(const float *x, const fp16 *y, const int32_t d) {
  auto ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    ans += x[i] * float(y[i]);
  }
  return -ans;
}

inline float dot_fp16_fp16_ref(const fp16 *x, const fp16 *y, const int32_t d) {
  auto ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    ans += float(x[i]) * float(y[i]);
  }
  return -ans;
}

inline float dot_fp32_bf16_ref(const float *x, const bf16 *y, const int32_t d) {
  float ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    ans += x[i] * float(y[i]);
  }
  return -ans;
}

inline float dot_bf16_bf16_ref(const bf16 *x, const bf16 *y, const int32_t d) {
  float ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    ans += float(x[i]) * float(y[i]);
  }
  return -ans;
}

inline int32_t dot_u8_s8_ref(const uint8_t *x, const int8_t *y,
                             const int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    ans += int32_t(x[i]) * int32_t(y[i]);
  }
  return -ans;
}

inline int32_t dot_s8_s8_ref(const int8_t *x, const int8_t *y,
                             const int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    ans += int32_t(x[i]) * int32_t(y[i]);
  }
  return -ans;
}

inline int32_t dot_u4_u4_ref(const uint8_t *x, const uint8_t *y,
                             const int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    int32_t xx = x[i] >> ((i & 1) * 4) & 15;
    int32_t yy = y[i] >> ((i & 1) * 4) & 15;
    ans += xx * yy;
  }
  return -ans;
}

} // namespace helpa
