#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace voxel {

// ============================================================================
// FastInvSqrt — Quake III fast inverse square root
// ============================================================================

inline f64 FastInvSqrt(f64 x) {
    u64 i = std::bit_cast<u64>(x);
    i = 0x5FE6EB50C7B537A9ULL - (i >> 1);
    f64 y = std::bit_cast<f64>(i);
    y = y * (1.5 - (0.5 * x * y * y));
    y = y * (1.5 - (0.5 * x * y * y));
    return y;
}

// ============================================================================
// FastSqrt
// ============================================================================

inline f64 FastSqrt(f64 x) {
    if (x <= 0.0) return 0.0;
    return 1.0 / FastInvSqrt(x);
}

// ============================================================================
// Lerp — linear interpolation
// ============================================================================

inline f64 Lerp(f64 a, f64 b, f64 t) {
    return a + (b - a) * t;
}

// ============================================================================
// FloorDiv / CeilDiv — correct floor division for negatives
// ============================================================================

inline i64 FloorDiv(i64 a, i64 b) {
    if (b == 0) return 0;
    i64 q = a / b;
    i64 r = a % b;
    if (r != 0 && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

inline i64 CeilDiv(i64 a, i64 b) {
    if (b == 0) return 0;
    i64 q = a / b;
    i64 r = a % b;
    if (r != 0 && ((a > 0) == (b > 0))) q += 1;
    return q;
}

// ============================================================================
// IsNaN / IsFinite
// ============================================================================

inline bool IsNaN(f64 x) {
    return x != x;
}

inline bool IsFinite(f64 x) {
    return !std::isinf(x) && !IsNaN(x);
}

// ============================================================================
// SafeDiv / SafeMod — division with zero guard
// ============================================================================

inline f64 SafeDiv(f64 a, f64 b) {
    if (b == 0.0) return 0.0;
    return a / b;
}

inline i64 SafeMod(i64 a, i64 b) {
    if (b == 0) return 0;
    return a % b;
}

// ============================================================================
// Clamp / ClampI
// ============================================================================

inline f64 Clamp(f64 x, f64 lo, f64 hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

inline i64 ClampI(i64 x, i64 lo, i64 hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// ============================================================================
// Min3 / Max3
// ============================================================================

template<typename T>
inline T Min3(T a, T b, T c) {
    return std::min(std::min(a, b), c);
}

template<typename T>
inline T Max3(T a, T b, T c) {
    return std::max(std::max(a, b), c);
}

// ============================================================================
// HypotF — sqrt(a^2 + b^2) without overflow
// ============================================================================

inline f64 HypotF(f64 a, f64 b) {
    a = std::abs(a);
    b = std::abs(b);
    if (a < b) std::swap(a, b);
    if (a == 0.0) return 0.0;
    f64 ratio = b / a;
    return a * std::sqrt(1.0 + ratio * ratio);
}

// ============================================================================
// NextPowerOfTwo
// ============================================================================

inline u64 NextPowerOfTwo(u64 x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

} // namespace voxel
