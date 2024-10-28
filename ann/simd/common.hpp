#pragma once

#include "ann/common.hpp"
#include "ann/simd/prefetch.hpp"

#include "ann/third_party/helpa/helpa/core.hpp"

#if defined(__AVX512F__)
#include "ann/simd/avx512.hpp"
#elif defined(__AVX2__)
#include "ann/simd/avx2.hpp"
#elif defined(__aarch64__)
#include "ann/simd/neon.hpp"
#else
#include "ann/simd/scalar.hpp"
#endif

namespace ann {

template <typename X, typename Y, typename U>
using DistFunc = U (*)(const X *x, const Y *y, const int d);

// e5m2
inline float L2SqrE5M2(const float *x, const e5m2 *y, int d);
inline float IPE5M2(const float *x, const e5m2 *y, int d);

// sq8
inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif);
inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif);
// sq6
inline float L2SqrSQ6_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif);
inline float IPSQ6_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif);

// sq4
inline float L2SqrSQ4_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif);
inline float IPSQ4_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif);
inline int32_t L2SqrSQ8SQ4(const uint8_t *x, const uint8_t *y, int d);
// sq2
inline int32_t L2SqrSQ2(const uint8_t *x, const uint8_t *y, int d);
// sq1
inline int32_t L2SqrSQ1(const uint8_t *x, const uint8_t *y, int d);

} // namespace ann
