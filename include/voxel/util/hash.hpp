#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

#include <cstring>
#include <bit>

namespace voxel {

// ============================================================================
// HashU64 — MurmurHash3 finalizer avalanche
// ============================================================================

inline u64 HashU64(u64 x) {
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

// ============================================================================
// HashCombine — boost-style hash combine (golden ratio spin)
// ============================================================================

inline u64 HashCombine(u64 seed, u64 h) {
    seed ^= h + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

// ============================================================================
// HashBytes — hash arbitrary byte sequence
// ============================================================================

inline u64 HashBytes(const void* data, sz len, u64 seed = 0) {
    const u8* p = static_cast<const u8*>(data);
    u64 h = seed ^ (static_cast<u64>(len) * 0x9E3779B97F4A7C15ULL);

    sz i = 0;
    for (; i + 8 <= len; i += 8) {
        u64 chunk;
        std::memcpy(&chunk, p + i, sizeof(chunk));
        h = HashCombine(h, HashU64(chunk));
    }

    u64 tail = 0;
    sz remaining = len - i;
    if (remaining > 0) {
        std::memcpy(&tail, p + i, remaining);
        h = HashCombine(h, HashU64(tail));
    }

    return h;
}

// ============================================================================
// HashString — hash null-terminated string
// ============================================================================

inline u64 HashString(const char* str) {
    if (!str) return 0;
    return HashBytes(str, std::strlen(str));
}

// ============================================================================
// HashF64 — hash double by reinterpretation
// ============================================================================

inline u64 HashF64(f64 x) {
    return HashU64(std::bit_cast<u64>(x));
}

// ============================================================================
// HashValue — generic hash for arithmetic and pointer types
// ============================================================================

template<typename T>
inline u64 HashValue(T x) {
    if constexpr (std::is_floating_point_v<T>) {
        return HashU64(std::bit_cast<u64>(static_cast<f64>(x)));
    } else if constexpr (std::is_pointer_v<T>) {
        return HashU64(reinterpret_cast<u64>(x));
    } else if constexpr (sizeof(T) <= 8) {
        return HashU64(static_cast<u64>(x));
    } else {
        return HashBytes(&x, sizeof(T));
    }
}

} // namespace voxel
