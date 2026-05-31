#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

#include <cstdint>
#include <bit>

namespace voxel {

// ============================================================================
// PopCount
// ============================================================================

inline u32 PopCount32(u32 x) {
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u32>(__builtin_popcount(x));
#elif VOXEL_CC_MSVC
    return __popcnt(x);
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3Fu;
#endif
}

inline u64 PopCount64(u64 x) {
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u64>(__builtin_popcountll(x));
#elif VOXEL_CC_MSVC
    return __popcnt64(x);
#else
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = x + (x >> 8);
    x = x + (x >> 16);
    x = x + (x >> 32);
    return x & 0x7Fu;
#endif
}

// ============================================================================
// CountLeadingZeros
// ============================================================================

inline u32 CountLeadingZeros32(u32 x) {
    if (x == 0) return 32;
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u32>(__builtin_clz(x));
#elif VOXEL_CC_MSVC
    return __lzcnt(x);
#else
    u32 n = 1;
    if ((x >> 16) == 0) { n += 16; x <<= 16; }
    if ((x >> 24) == 0) { n += 8;  x <<= 8;  }
    if ((x >> 28) == 0) { n += 4;  x <<= 4;  }
    if ((x >> 30) == 0) { n += 2;  x <<= 2;  }
    n -= (x >> 31);
    return n;
#endif
}

inline u64 CountLeadingZeros64(u64 x) {
    if (x == 0) return 64;
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u64>(__builtin_clzll(x));
#elif VOXEL_CC_MSVC
    return __lzcnt64(x);
#else
    u64 n = 1;
    if ((x >> 32) == 0) { n += 32; x <<= 32; }
    if ((x >> 48) == 0) { n += 16; x <<= 16; }
    if ((x >> 56) == 0) { n += 8;  x <<= 8;  }
    if ((x >> 60) == 0) { n += 4;  x <<= 4;  }
    if ((x >> 62) == 0) { n += 2;  x <<= 2;  }
    n -= (x >> 63);
    return n;
#endif
}

// ============================================================================
// CountTrailingZeros
// ============================================================================

inline u32 CountTrailingZeros32(u32 x) {
    if (x == 0) return 32;
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u32>(__builtin_ctz(x));
#elif VOXEL_CC_MSVC
    return _tzcnt_u32(x);
#else
    u32 n = 31;
    x &= (~x + 1);
    if (x & 0x0000FFFFu) n -= 16;
    if (x & 0x00FF00FFu) n -= 8;
    if (x & 0x0F0F0F0Fu) n -= 4;
    if (x & 0x33333333u) n -= 2;
    if (x & 0x55555555u) n -= 1;
    return n;
#endif
}

inline u64 CountTrailingZeros64(u64 x) {
    if (x == 0) return 64;
#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    return static_cast<u64>(__builtin_ctzll(x));
#elif VOXEL_CC_MSVC
    return _tzcnt_u64(x);
#else
    u64 n = 63;
    x &= (~x + 1);
    if (x & 0x00000000FFFFFFFFULL) n -= 32;
    if (x & 0x0000FFFF0000FFFFULL) n -= 16;
    if (x & 0x00FF00FF00FF00FFULL) n -= 8;
    if (x & 0x0F0F0F0F0F0F0F0FULL) n -= 4;
    if (x & 0x3333333333333333ULL) n -= 2;
    if (x & 0x5555555555555555ULL) n -= 1;
    return n;
#endif
}

// ============================================================================
// BitReverse
// ============================================================================

inline u32 BitReverse32(u32 x) {
    x = ((x & 0x55555555u) << 1)  | ((x & 0xAAAAAAAAu) >> 1);
    x = ((x & 0x33333333u) << 2)  | ((x & 0xCCCCCCCCu) >> 2);
    x = ((x & 0x0F0F0F0Fu) << 4)  | ((x & 0xF0F0F0F0u) >> 4);
    x = ((x & 0x00FF00FFu) << 8)  | ((x & 0xFF00FF00u) >> 8);
    x = (x << 16) | (x >> 16);
    return x;
}

inline u64 BitReverse64(u64 x) {
    x = ((x & 0x5555555555555555ULL) << 1)  | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2)  | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4)  | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8)  | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
    x = (x << 32) | (x >> 32);
    return x;
}

// ============================================================================
// RoundUpPow2 — next power of 2
// ============================================================================

inline u32 RoundUpPow2(u32 x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline u64 RoundUpPow2(u64 x) {
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

// ============================================================================
// IsPow2
// ============================================================================

inline bool IsPow2(u64 x) {
    return (x & (x - 1)) == 0 && x != 0;
}

// ============================================================================
// Log2Floor / Log2Ceil
// ============================================================================

inline u32 Log2Floor(u64 x) {
    if (x == 0) return 0;
    return 63 - static_cast<u32>(CountLeadingZeros64(x));
}

inline u32 Log2Ceil(u64 x) {
    if (x <= 1) return 0;
    return 64 - static_cast<u32>(CountLeadingZeros64(x - 1));
}

// ============================================================================
// AlignUp / AlignDown
// ============================================================================

inline u64 AlignUp(u64 x, u64 alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

inline u64 AlignDown(u64 x, u64 alignment) {
    return x & ~(alignment - 1);
}

// ============================================================================
// RotateLeft / RotateRight
// ============================================================================

inline u64 RotateLeft64(u64 x, u32 n) {
    n &= 63;
    return (x << n) | (x >> (64 - n));
}

inline u64 RotateRight64(u64 x, u32 n) {
    n &= 63;
    return (x >> n) | (x << (64 - n));
}

// ============================================================================
// HasSingleBit
// ============================================================================

inline bool HasSingleBit(u64 x) {
    return PopCount64(x) == 1;
}

} // namespace voxel
