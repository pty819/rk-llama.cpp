#include "rknpu2-calibration.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define RKNPU2_HAS_NEON 1
#endif

namespace rknpu2_calibration {

// --- Hadamard Transform Implementations ---

// Helper to check if a number is a power of two
static bool is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Iterative Fast Walsh-Hadamard Transform (in-place)
static void fwht_iterative(float* data, int size) {
    for (int h = 1; h < size; h <<= 1) {
        for (int i = 0; i < size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

int next_power_of_two(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Fused activation-side Hadamard prep: out = FWHT(pad_zero(src * s)).
// Element-wise NEON mul/add/sub round identically to the scalar reference, so the
// result is bit-identical to the original signed_row -> hadamard_transform sequence.
// `out` must hold K_op floats; callers reuse a thread-local scratch buffer.
void hadamard_signed_fwht(float* out, const float* src, const float* s, int K, int K_op) {
#ifdef RKNPU2_HAS_NEON
    if (K_op < 8) {  // vector stages below need a full 8-element block
        for (int k = 0; k < K; ++k) out[k] = src[k] * s[k];
        for (int k = K; k < K_op; ++k) out[k] = 0.0f;
        fwht_iterative(out, K_op);
        return;
    }
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        vst1q_f32(out + k, vmulq_f32(vld1q_f32(src + k), vld1q_f32(s + k)));
    }
    for (; k < K; ++k) out[k] = src[k] * s[k];
    for (; k < K_op; ++k) out[k] = 0.0f;

    // FWHT stage h == 1: butterfly on adjacent pairs, 8 elements per iteration.
    {
        float* p = out;
        float* end = out + K_op;
        for (; p + 8 <= end; p += 8) {
            float32x4_t a = vld1q_f32(p);
            float32x4_t b = vld1q_f32(p + 4);
            float32x4_t ev = vuzp1q_f32(a, b);   // x0 x2 x4 x6
            float32x4_t od = vuzp2q_f32(a, b);   // x1 x3 x5 x7
            float32x4_t sm = vaddq_f32(ev, od);
            float32x4_t df = vsubq_f32(ev, od);
            vst1q_f32(p,     vzip1q_f32(sm, df));  // (x0+x1) (x0-x1) (x2+x3) (x2-x3)
            vst1q_f32(p + 4, vzip2q_f32(sm, df));  // (x4+x5) (x4-x5) (x6+x7) (x6-x7)
        }
    }
    // FWHT stage h == 2: pairs (i, i+2), 8 elements per iteration.
    {
        float* p = out;
        float* end = out + K_op;
        for (; p + 8 <= end; p += 8) {
            float32x4_t lo = vld1q_f32(p);
            float32x4_t hi = vld1q_f32(p + 4);
            float32x4_t f  = vcombine_f32(vget_low_f32(lo),  vget_low_f32(hi));   // x0 x1 x4 x5
            float32x4_t sc = vcombine_f32(vget_high_f32(lo), vget_high_f32(hi));  // x2 x3 x6 x7
            float32x4_t sm = vaddq_f32(f, sc);
            float32x4_t df = vsubq_f32(f, sc);
            vst1_f32(p,     vget_low_f32(sm));   // -> p+0, p+1
            vst1_f32(p + 2, vget_low_f32(df));   // -> p+2, p+3
            vst1_f32(p + 4, vget_high_f32(sm));  // -> p+4, p+5
            vst1_f32(p + 6, vget_high_f32(df));  // -> p+6, p+7
        }
    }
    // FWHT stages h >= 4: plain 4-wide butterfly between blocks h apart.
    for (int h = 4; h < K_op; h <<= 1) {
        for (int i = 0; i < K_op; i += h * 2) {
            float* x = out + i;
            float* y = out + i + h;
            for (int j = 0; j < h; j += 4) {
                float32x4_t xv = vld1q_f32(x + j);
                float32x4_t yv = vld1q_f32(y + j);
                vst1q_f32(x + j, vaddq_f32(xv, yv));
                vst1q_f32(y + j, vsubq_f32(xv, yv));
            }
        }
    }
#else
    for (int k = 0; k < K; ++k) out[k] = src[k] * s[k];
    for (int k = K; k < K_op; ++k) out[k] = 0.0f;
    fwht_iterative(out, K_op);
#endif
}

void hadamard_transform(float* dst, const float* src, int K, int padded_size) {
    // If no padding is needed, copy and perform in-place.
    if (K == padded_size) {
        memcpy(dst, src, K * sizeof(float));
        fwht_iterative(dst, K);
        return;
    }

    // Using a thread-local buffer to avoid repeated heap allocations.
    thread_local static std::vector<float> padded_data;
    
    // Resizing the buffer only if the current one is too small.
    if (padded_data.size() < (size_t)padded_size) {
        padded_data.resize(padded_size);
    }
    
    // Copying source data and zero-fill the rest (padding).
    memcpy(padded_data.data(), src, K * sizeof(float));
    if (padded_size > K) {
        memset(padded_data.data() + K, 0, (padded_size - K) * sizeof(float));
    }

    // Applying the transform to our temporary buffer
    fwht_iterative(padded_data.data(), padded_size);

    // Copying the result to the destination buffer
    memcpy(dst, padded_data.data(), padded_size * sizeof(float));
}

} // namespace rknpu2_calibration