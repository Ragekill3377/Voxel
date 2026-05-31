#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include <cstring>
#include <bit>
#include <type_traits>

namespace voxel {
namespace ops {

namespace detail {

inline u64 RotL(u64 x, int r) {
    return (x << r) | (x >> (64 - r));
}

inline u64 ShiftMix(u64 val) {
    return val ^ (val >> 47);
}

inline u64 Len16(u64 u, u64 v, u64 mul) {
    u64 a = (u ^ v) * mul;
    a ^= (a >> 47);
    u64 b = (v ^ a) * mul;
    b ^= (b >> 47);
    return b * mul;
}

inline u64 XXHRound(u64 acc, u64 input, u64 prime1, u64 prime2) {
    acc += input * prime2;
    acc  = RotL(acc, 31);
    acc *= prime1;
    return acc;
}

inline u64 XXHMergeRound(u64 acc, u64 val, u64 prime1) {
    val  = XXHRound(0, val, prime1, 0);
    acc ^= val;
    acc  = acc * prime1;
    return acc;
}

inline void Fetch64(const u8* p, u64& v) {
    std::memcpy(&v, p, 8);
}

inline void Fetch32(const u8* p, u32& v) {
    std::memcpy(&v, p, 4);
}

inline u64 FarmHash64Len0to16(const u8* s, sz len) {
    if (len >= 8) {
        u64 mul = 0x9ddfea08eb382d69ULL;
        u64 a, b;
        Fetch64(s, a);
        Fetch64(s + len - 8, b);
        u64 c = RotL(b, static_cast<int>(len)) * mul;
        (void)c;
        a = RotL(a, 43) * mul;
        b += mul + a;
        a ^= b >> 47;
        b *= mul;
        b ^= a;
        return b;
    }
    if (len >= 4) {
        u64 mul = 0x9ddfea08eb382d69ULL;
        u32 a32, b32;
        Fetch32(s, a32);
        Fetch32(s + len - 4, b32);
        u64 a = static_cast<u64>(a32) << 3;
        u64 b = (static_cast<u64>(b32) << (len * 8)) >> (len * 8);
        u64 c = RotL(b, static_cast<int>(len)) * mul;
        (void)c;
        a = RotL(a, 43) * mul;
        b += mul + a;
        a ^= b >> 47;
        b *= mul;
        b ^= a;
        return b;
    }
    if (len > 0) {
        u8 a = s[0];
        u8 b = s[len >> 1];
        u8 c = s[len - 1];
        u32 y = static_cast<u32>(a) + (static_cast<u32>(b) << 8);
        u32 z = static_cast<u32>(len) + (static_cast<u32>(c) << 2);
        return ShiftMix(y * 0x9ddfea08eb382d69ULL ^ z * 0x9ae16a3b2f90404fULL) * 0x9ae16a3b2f90404fULL;
    }
    return 0x9ae16a3b2f90404fULL;
}

inline u64 FarmHash64Len17to32(const u8* s, sz len) {
    u64 mul = 0x9ddfea08eb382d69ULL;
    u64 a, b, c, d;
    Fetch64(s, a);
    Fetch64(s + 8, b);
    Fetch64(s + len - 8, c);
    Fetch64(s + len - 16, d);
    u64 e = RotL(a + b, 43)
          + RotL(c, 30)
          + d + mul;
    u64 f = a + RotL(b + mul, 18) + c + mul;
    e = Len16(e, f, mul);
    f = Len16(f + c, e, mul);
    return f;
}

inline u64 FarmHash64Len33to64(const u8* s, sz len) {
    u64 mul = 0x9ddfea08eb382d69ULL;
    u64 a, b, c, d, e, f, g, h;
    Fetch64(s,      a); Fetch64(s + 8,  b);
    Fetch64(s + 16, c); Fetch64(s + 24, d);
    Fetch64(s + len - 32, e); Fetch64(s + len - 24, f);
    Fetch64(s + len - 16, g); Fetch64(s + len - 8,  h);

    u64 x = RotL(a + b + c + d + ShiftMix(d + mul + e + f + g + h), 37) * mul;
    u64 y = RotL(e + f + g + h + ShiftMix(a + b + c + mul), 37) * mul;
    x = Len16(x, y, mul);
    y = Len16(e + h, f + g, mul);
    return Len16(x + d, y + d, mul) ^ y;
}

} // namespace detail

inline u64 MurmurHash64A(const void* key, sz len, u64 seed = 0) {
    const u64 m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    u64 h = seed ^ (static_cast<u64>(len) * m);

    const u64* data = static_cast<const u64*>(key);
    const u64* end = data + (len / 8);

    while (data != end) {
        u64 k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const u8* tail = reinterpret_cast<const u8*>(data);
    switch (len & 7) {
    case 7: h ^= static_cast<u64>(tail[6]) << 48; [[fallthrough]];
    case 6: h ^= static_cast<u64>(tail[5]) << 40; [[fallthrough]];
    case 5: h ^= static_cast<u64>(tail[4]) << 32; [[fallthrough]];
    case 4: h ^= static_cast<u64>(tail[3]) << 24; [[fallthrough]];
    case 3: h ^= static_cast<u64>(tail[2]) << 16; [[fallthrough]];
    case 2: h ^= static_cast<u64>(tail[1]) << 8;  [[fallthrough]];
    case 1: h ^= static_cast<u64>(tail[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

inline u64 XXHash64(const void* key, sz len, u64 seed = 0) {
    const u64 PRIME64_1 = 11400714785074694791ULL;
    const u64 PRIME64_2 = 14029467366897019727ULL;
    const u64 PRIME64_3 =  1609587929392839161ULL;
    const u64 PRIME64_4 =  9650029242287828579ULL;
    const u64 PRIME64_5 =  2870177450012600261ULL;

    const u8* p = static_cast<const u8*>(key);
    const u8* const end = p + len;
    u64 h64;

    if (len >= 32) {
        const u8* const limit = end - 32;
        u64 v1 = seed + PRIME64_1 + PRIME64_2;
        u64 v2 = seed + PRIME64_2;
        u64 v3 = seed + 0;
        u64 v4 = seed - PRIME64_1;

        do {
            u64 k1, k2, k3, k4;
            std::memcpy(&k1, p, 8); p += 8;
            std::memcpy(&k2, p, 8); p += 8;
            std::memcpy(&k3, p, 8); p += 8;
            std::memcpy(&k4, p, 8); p += 8;

            v1 = detail::XXHRound(v1, k1, PRIME64_1, PRIME64_2);
            v2 = detail::XXHRound(v2, k2, PRIME64_1, PRIME64_2);
            v3 = detail::XXHRound(v3, k3, PRIME64_1, PRIME64_2);
            v4 = detail::XXHRound(v4, k4, PRIME64_1, PRIME64_2);
        } while (p <= limit);

        h64 = detail::RotL(v1, 1) + detail::RotL(v2, 7) + detail::RotL(v3, 12) + detail::RotL(v4, 18);
        h64 = detail::XXHMergeRound(h64, v1, PRIME64_1);
        h64 = detail::XXHMergeRound(h64, v2, PRIME64_1);
        h64 = detail::XXHMergeRound(h64, v3, PRIME64_1);
        h64 = detail::XXHMergeRound(h64, v4, PRIME64_1);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += static_cast<u64>(len);

    while (p + 8 <= end) {
        u64 k1;
        std::memcpy(&k1, p, 8);
        k1 *= PRIME64_2;
        k1 = detail::RotL(k1, 31);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = detail::RotL(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= end) {
        u32 k1;
        std::memcpy(&k1, p, 4);
        h64 ^= static_cast<u64>(k1) * PRIME64_1;
        h64 = detail::RotL(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < end) {
        h64 ^= static_cast<u64>(*p) * PRIME64_5;
        h64 = detail::RotL(h64, 11) * PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

inline u64 FarmHash64(const void* key, sz len) {
    const u8* s = static_cast<const u8*>(key);

    if (len <= 16) {
        return detail::FarmHash64Len0to16(s, len);
    }
    if (len <= 32) {
        return detail::FarmHash64Len17to32(s, len);
    }
    if (len <= 64) {
        return detail::FarmHash64Len33to64(s, len);
    }

    u64 seed = 81;
    const u64 mul = 0x9ddfea08eb382d69ULL;
    u64 x = seed;
    u64 y = seed * mul + 113;
    u64 z = detail::Len16(y ^ static_cast<u64>(len) * mul, x, mul);

    const u8* p = s;
    const u8* end = s + len;

    u64 v[2] = {0, 0};
    u64 w[2] = {0, 0};
    (void)w;
    (void)end;

    v[0] = detail::RotL(y ^ mul, 49) * mul + detail::ShiftMix(x);
    v[1] = detail::RotL(v[0], 42) * mul + detail::ShiftMix(y);
    w[0] = y + mul + detail::ShiftMix(x);
    w[1] = x * mul + detail::ShiftMix(y);

    const u8* limit = end - 64;
    do {
        u64 a, b, c, d, e, f, g, h;
        std::memcpy(&a, p,      8); std::memcpy(&b, p + 8,  8);
        std::memcpy(&c, p + 16, 8); std::memcpy(&d, p + 24, 8);
        std::memcpy(&e, p + 32, 8); std::memcpy(&f, p + 40, 8);
        std::memcpy(&g, p + 48, 8); std::memcpy(&h, p + 56, 8);

        x = detail::RotL(x + y + v[0] + a + e, 37) * mul;
        y = detail::RotL(y + v[1] + b + f, 42) * mul;
        z = detail::Len16(z, detail::RotL(v[0] + c + g, 33) * mul, mul);
        v[0] = detail::RotL(v[0] + d + h, 51) * mul;
        v[1] = detail::Len16(v[1] + y + mul, a + e, mul);

        x += v[1];
        y += v[0];
        z += y;

        p += 64;
    } while (p <= limit);

    for (int i = 0; i < 2; ++i) {
        sz remaining = static_cast<sz>(end - p);
        sz chunk = std::min(remaining, static_cast<sz>(64));
        u64 a, b, c, d, e, f, g, h;
        a = b = c = d = e = f = g = h = 0;

        if (chunk >= 8)  { std::memcpy(&a, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&b, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&c, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&d, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&e, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&f, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&g, p, 8); p += 8; chunk -= 8; }
        if (chunk >= 8)  { std::memcpy(&h, p, 8); p += 8; chunk -= 8; }

        x = detail::RotL(x + y + v[0] + a + e, 37) * mul;
        y = detail::RotL(y + v[1] + b + f, 42) * mul;
        z = detail::Len16(z, detail::RotL(v[0] + c + g, 33) * mul, mul);
        v[0] = detail::RotL(v[0] + d + h, 51) * mul;
        v[1] = detail::Len16(v[1] + y + mul, a + e, mul);
    }

    return detail::Len16(z + x, y + v[1], mul);
}

template<typename T>
u64 HashValue(T value) {
    if constexpr (std::is_same_v<T, u64>) {
        u64 x = value;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    } else if constexpr (std::is_same_v<T, i64>) {
        u64 x = static_cast<u64>(value);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    } else if constexpr (std::is_same_v<T, u32>) {
        u64 x = static_cast<u64>(value);
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        x = x ^ (x >> 33);
        return x;
    } else if constexpr (std::is_same_v<T, i32>) {
        u64 x = static_cast<u64>(static_cast<u32>(value));
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        x = x ^ (x >> 33);
        return x;
    } else if constexpr (std::is_same_v<T, f64>) {
        u64 bits;
        std::memcpy(&bits, &value, 8);
        if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL && (bits & 0x000FFFFFFFFFFFFFULL) != 0) {
            bits = 0x7FF8000000000000ULL;
        }
        if ((bits & 0x7FFFFFFFFFFFFFFFULL) == 0 && (bits >> 63)) {
            bits = 0;
        }
        u64 x = bits;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    } else if constexpr (std::is_same_v<T, f32>) {
        u32 bits;
        std::memcpy(&bits, &value, 4);
        if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
            bits = 0x7FC00000U;
        }
        if ((bits & 0x7FFFFFFFU) == 0 && (bits >> 31)) {
            bits = 0;
        }
        u64 x = static_cast<u64>(bits);
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        x = x ^ (x >> 33);
        return x;
    } else if constexpr (std::is_same_v<T, u16>) {
        u64 x = static_cast<u64>(value);
        x *= 0x9e3779b97f4a7c15ULL;
        x ^= x >> 31;
        x *= 0xbf58476d1ce4e5b9ULL;
        return x;
    } else if constexpr (std::is_same_v<T, i16>) {
        u64 x = static_cast<u64>(static_cast<u16>(value));
        x *= 0x9e3779b97f4a7c15ULL;
        x ^= x >> 31;
        x *= 0xbf58476d1ce4e5b9ULL;
        return x;
    } else if constexpr (std::is_same_v<T, u8>) {
        u64 x = static_cast<u64>(value);
        x *= 0x9e3779b97f4a7c15ULL;
        x ^= value ^ (x >> 31);
        return x;
    } else if constexpr (std::is_same_v<T, i8>) {
        u64 x = static_cast<u64>(static_cast<u8>(value));
        x *= 0x9e3779b97f4a7c15ULL;
        x ^= value ^ (x >> 31);
        return x;
    } else {
        static_assert(sizeof(T) == 0, "HashValue: unsupported type");
        return 0;
    }
}

inline u64 HashCombine(u64 a, u64 b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

inline u64 HashBytes(const u8* data, sz len) {
    u64 h = 0x9ae16a3b2f90404fULL;

    sz i = 0;
    for (; i + 8 <= len; i += 8) {
        u64 v;
        std::memcpy(&v, data + i, 8);
        h = HashCombine(h, v);
    }

    u64 tail = 0;
    for (sz j = i; j < len; ++j) {
        tail = (tail << 8) | static_cast<u64>(data[j]);
    }
    if (i < len) {
        h = HashCombine(h, tail);
    }

    h ^= static_cast<u64>(len);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}

} // namespace ops
} // namespace voxel
