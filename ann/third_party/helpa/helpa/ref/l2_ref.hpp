#pragma once
#include "ann/third_party/helpa/helpa/types.hpp"
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "../ann/helpa/helpa/common.hpp"
namespace helpa {

inline float
l2_fp32_fp32_ref(const float* x, const float* y, const int32_t d) {
#if defined(__aarch64__)
    float32x4_t sum = vdupq_n_f32(0);
    for (int32_t i = 0; i < d; i += 4) {
        auto xx = vld1q_f32(x + i);
        auto yy = vld1q_f32(y + i);
        auto t = vsubq_f32(xx, yy);
        sum = vmlaq_f32(sum, t, t);
    }
    return vaddvq_f32(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
#endif
}

inline float
l2_fp32_fp16_ref(const float* x, const fp16* y, const int32_t d) {
    auto ans = 0.0f;
    for (int32_t i = 0; i < d; ++i) {
        auto t = x[i] - float(y[i]);
        ans += t * t;
    }
    return ans;
}

inline float
l2_fp16_fp16_ref(const fp16* x, const fp16* y, const int32_t d) {
    auto ans = 0.0f;
    for (int32_t i = 0; i < d; ++i) {
        auto t = float(x[i]) - float(y[i]);
        ans += t * t;
    }
    return ans;
}

inline float
l2_fp32_bf16_ref(const float* x, const bf16* y, const int32_t d) {
    float ans = 0.0f;
    for (int32_t i = 0; i < d; ++i) {
        auto t = x[i] - float(y[i]);
        ans += t * t;
    }
    return ans;
}

inline float
l2_bf16_bf16_ref(const bf16* x, const bf16* y, const int32_t d) {
    float ans = 0.0f;
    for (int32_t i = 0; i < d; ++i) {
        auto t = float(x[i]) - float(y[i]);
        ans += t * t;
    }
    return ans;
}

inline int32_t
l2_u8_s8_ref(const uint8_t* x, const int8_t* y, const int32_t d) {
    int32_t ans = 0;
    for (int32_t i = 0; i < d; ++i) {
        auto t = int32_t(x[i]) - int32_t(y[i]);
        ans += t * t;
    }
    return ans;
}

inline int32_t
l2_s8_s8_ref(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t ans = 0;
    for (int32_t i = 0; i < d; ++i) {
        auto t = int32_t(x[i]) - int32_t(y[i]);
        ans += t * t;
    }
    return ans;
}

inline int32_t
l2_u4_u4_ref(const uint8_t* x, const uint8_t* y, const int32_t d) {
    int32_t ans = 0;
    for (int32_t i = 0; i < d; ++i) {
        int32_t xx = x[i / 2] >> ((i & 1) * 4) & 15;
        int32_t yy = y[i / 2] >> ((i & 1) * 4) & 15;
        auto t = xx - yy;
        ans += t * t;
    }
    return ans;
}

}  // namespace helpa
