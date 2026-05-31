#pragma once

// ============================================================================
// Compiler detection
// ============================================================================

#if defined(__GNUC__) && !defined(__clang__)
    #define VOXEL_CC_GCC     1
    #define VOXEL_CC_CLANG   0
    #define VOXEL_CC_MSVC    0
#elif defined(__clang__)
    #define VOXEL_CC_GCC     0
    #define VOXEL_CC_CLANG   1
    #define VOXEL_CC_MSVC    0
#elif defined(_MSC_VER)
    #define VOXEL_CC_GCC     0
    #define VOXEL_CC_CLANG   0
    #define VOXEL_CC_MSVC    1
#else
    #define VOXEL_CC_GCC     0
    #define VOXEL_CC_CLANG   0
    #define VOXEL_CC_MSVC    0
#endif

// ============================================================================
// Operating system detection
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define VOXEL_OS_WINDOWS 1
    #define VOXEL_OS_LINUX   0
    #define VOXEL_OS_MACOS   0
#elif defined(__APPLE__)
    #define VOXEL_OS_WINDOWS 0
    #define VOXEL_OS_LINUX   0
    #define VOXEL_OS_MACOS   1
#elif defined(__linux__)
    #define VOXEL_OS_WINDOWS 0
    #define VOXEL_OS_LINUX   1
    #define VOXEL_OS_MACOS   0
#else
    #define VOXEL_OS_WINDOWS 0
    #define VOXEL_OS_LINUX   0
    #define VOXEL_OS_MACOS   0
#endif

// ============================================================================
// Architecture detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    #define VOXEL_ARCH_X86_64  1
    #define VOXEL_ARCH_ARM64   0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define VOXEL_ARCH_X86_64  0
    #define VOXEL_ARCH_ARM64   1
#else
    #define VOXEL_ARCH_X86_64  0
    #define VOXEL_ARCH_ARM64   0
#endif

// ============================================================================
// SIMD feature detection
// ============================================================================

#if VOXEL_ARCH_X86_64
    #if defined(__AVX512F__)
        #define VOXEL_SIMD_AVX512 1
    #else
        #define VOXEL_SIMD_AVX512 0
    #endif
    #if defined(__AVX2__)
        #define VOXEL_SIMD_AVX2   1
    #else
        #define VOXEL_SIMD_AVX2   0
    #endif
    #if defined(__SSE4_2__)
        #define VOXEL_SIMD_SSE42  1
    #else
        #define VOXEL_SIMD_SSE42  0
    #endif
    #define VOXEL_SIMD_NEON     0
    #define VOXEL_SIMD_SVE      0
#elif VOXEL_ARCH_ARM64
    #if defined(__ARM_NEON)
        #define VOXEL_SIMD_NEON    1
    #else
        #define VOXEL_SIMD_NEON    0
    #endif
    #if defined(__ARM_FEATURE_SVE)
        #define VOXEL_SIMD_SVE     1
    #else
        #define VOXEL_SIMD_SVE     0
    #endif
    #define VOXEL_SIMD_AVX512   0
    #define VOXEL_SIMD_AVX2     0
    #define VOXEL_SIMD_SSE42    0
#else
    #define VOXEL_SIMD_AVX512   0
    #define VOXEL_SIMD_AVX2     0
    #define VOXEL_SIMD_SSE42    0
    #define VOXEL_SIMD_NEON     0
    #define VOXEL_SIMD_SVE      0
#endif

// ============================================================================
// SIMD vector width constants
// ============================================================================

namespace voxel {

inline constexpr sz kSimdWidthAVX512 = 64;
inline constexpr sz kSimdWidthAVX2   = 32;
inline constexpr sz kSimdWidthSSE42  = 16;
inline constexpr sz kSimdWidthNEON   = 16;
inline constexpr sz kSimdWidthSVE    = 0; // runtime query

#if VOXEL_SIMD_AVX512
    inline constexpr sz kSimdNativeWidth = 64;
#elif VOXEL_SIMD_AVX2
    inline constexpr sz kSimdNativeWidth = 32;
#elif VOXEL_SIMD_SSE42 || VOXEL_SIMD_NEON
    inline constexpr sz kSimdNativeWidth = 16;
#else
    inline constexpr sz kSimdNativeWidth = 16;
#endif

// ============================================================================
// Cache line constants
// ============================================================================

inline constexpr sz kCacheLineSize     = 64;
inline constexpr sz kCacheLineMask     = 63;
inline constexpr sz kPageSize          = 4096;
inline constexpr sz kHugePageSize2MB   = 2097152;
inline constexpr sz kHugePageSize1GB   = 1073741824;

}

// ============================================================================
// Compiler attribute macros
// ============================================================================

#if VOXEL_CC_GCC || VOXEL_CC_CLANG
    #define VOXEL_ALWAYS_INLINE  __attribute__((always_inline)) inline
    #define VOXEL_NOINLINE       __attribute__((noinline))
    #define VOXEL_HOT            __attribute__((hot))
    #define VOXEL_COLD           __attribute__((cold))
    #define VOXEL_FLATTEN        __attribute__((flatten))
    #define VOXEL_PURE           __attribute__((pure))
    #define VOXEL_CONST          __attribute__((const))
    #define VOXEL_RESTRICT       __restrict__
    #define VOXEL_LIKELY(x)      __builtin_expect(!!(x), 1)
    #define VOXEL_UNLIKELY(x)    __builtin_expect(!!(x), 0)
    #define VOXEL_PREFETCH(p)    __builtin_prefetch(p)
    #define VOXEL_PREFETCH_W(p)  __builtin_prefetch(p, 1)
    #define VOXEL_CLZ(x)         __builtin_clzll(x)
    #define VOXEL_CTZ(x)         __builtin_ctzll(x)
    #define VOXEL_POPCOUNT(x)    __builtin_popcountll(x)
    #define VOXEL_ASSUME_ALIGNED(p, n) __builtin_assume_aligned(p, n)
#elif VOXEL_CC_MSVC
    #define VOXEL_ALWAYS_INLINE  __forceinline
    #define VOXEL_NOINLINE       __declspec(noinline)
    #define VOXEL_HOT
    #define VOXEL_COLD
    #define VOXEL_FLATTEN
    #define VOXEL_PURE
    #define VOXEL_CONST
    #define VOXEL_RESTRICT       __restrict
    #define VOXEL_LIKELY(x)      (x)
    #define VOXEL_UNLIKELY(x)    (x)
    #define VOXEL_PREFETCH(p)
    #define VOXEL_PREFETCH_W(p)
    #define VOXEL_CLZ(x)         __lzcnt64(x)
    #define VOXEL_CTZ(x)         _tzcnt_u64(x)
    #define VOXEL_POPCOUNT(x)    __popcnt64(x)
    #define VOXEL_ASSUME_ALIGNED(p, n) __assume(((uintptr_t)(p) & ((n)-1)) == 0)
#else
    #define VOXEL_ALWAYS_INLINE  inline
    #define VOXEL_NOINLINE
    #define VOXEL_HOT
    #define VOXEL_COLD
    #define VOXEL_FLATTEN
    #define VOXEL_PURE
    #define VOXEL_CONST
    #define VOXEL_RESTRICT
    #define VOXEL_LIKELY(x)      (x)
    #define VOXEL_UNLIKELY(x)    (x)
    #define VOXEL_PREFETCH(p)
    #define VOXEL_PREFETCH_W(p)
    #define VOXEL_CLZ(x)         (x == 0 ? 64 : __builtin_clzll(x))
    #define VOXEL_CTZ(x)         (x == 0 ? 64 : __builtin_ctzll(x))
    #define VOXEL_POPCOUNT(x)    __builtin_popcountll(x)
    #define VOXEL_ASSUME_ALIGNED(p, n) (p)
#endif

// ============================================================================
// Debug and assertion
// ============================================================================

#if !defined(NDEBUG)
    #include <cstdio>
    #include <cstdlib>
    #define VOXEL_ASSERT(cond, msg) \
        do { if (!(cond)) { std::fprintf(stderr, "VOXEL ASSERTION FAILED [%s:%d]: %s\n", __FILE__, __LINE__, msg); std::abort(); } } while(0)
    #define VOXEL_DEBUG_ONLY(x) x
#else
    #define VOXEL_ASSERT(cond, msg) ((void)0)
    #define VOXEL_DEBUG_ONLY(x) ((void)0)
#endif

namespace voxel {
namespace platform {

inline constexpr bool IsX86_64()  { return VOXEL_ARCH_X86_64 != 0; }
inline constexpr bool IsARM64()   { return VOXEL_ARCH_ARM64  != 0; }
inline constexpr bool IsWindows() { return VOXEL_OS_WINDOWS  != 0; }
inline constexpr bool IsLinux()   { return VOXEL_OS_LINUX    != 0; }
inline constexpr bool IsMacOS()   { return VOXEL_OS_MACOS    != 0; }
inline constexpr bool HasAVX2()   { return VOXEL_SIMD_AVX2   != 0; }
inline constexpr bool HasAVX512() { return VOXEL_SIMD_AVX512 != 0; }
inline constexpr bool HasNEON()   { return VOXEL_SIMD_NEON   != 0; }

struct CpuInfo {
    u64 LogicalCores;
    u64 PhysicalCores;
    sz  L1CacheSize;
    sz  L2CacheSize;
    sz  L3CacheSize;
    sz  PageSize;
    bool HasAVX;
    bool HasAVX2;
    bool HasAVX512F;
    bool HasAVX512BW;
    bool HasAVX512DQ;
    bool HasBMI1;
    bool HasBMI2;
    bool HasFMA;
    bool HasPOPCNT;
    bool HasLZCNT;
};

inline CpuInfo QueryCpuInfo() {
    CpuInfo info{};
    info.LogicalCores  = 1;
    info.PhysicalCores = 1;
    info.L1CacheSize   = 32768;
    info.L2CacheSize   = 262144;
    info.L3CacheSize   = 8388608;
    info.PageSize      = 4096;
    info.HasAVX        = VOXEL_SIMD_AVX2  || VOXEL_SIMD_AVX512;
    info.HasAVX2       = VOXEL_SIMD_AVX2  || VOXEL_SIMD_AVX512;
    info.HasAVX512F    = VOXEL_SIMD_AVX512;
    info.HasAVX512BW   = VOXEL_SIMD_AVX512;
    info.HasAVX512DQ   = VOXEL_SIMD_AVX512;
    info.HasBMI1       = VOXEL_CC_GCC || VOXEL_CC_CLANG;
    info.HasBMI2       = VOXEL_CC_GCC || VOXEL_CC_CLANG;
    info.HasFMA        = VOXEL_SIMD_AVX2 || VOXEL_SIMD_AVX512 || VOXEL_SIMD_NEON;
    info.HasPOPCNT     = true;
    info.HasLZCNT      = true;
    return info;
}

} // namespace platform
} // namespace voxel
