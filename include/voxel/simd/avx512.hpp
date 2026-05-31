#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include "voxel/core/types.hpp"
#include "voxel/simd/x86.hpp"

namespace voxel {
namespace simd {

inline void PrefetchNTA(const void* p) { _mm_prefetch(static_cast<const char*>(p), _MM_HINT_NTA); }
inline void PrefetchT0(const void* p) { _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0); }
inline void PrefetchT1(const void* p) { _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T1); }
inline void PrefetchT2(const void* p) { _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T2); }

inline void CompressStoreEpI32(i32* VOXEL_RESTRICT dst, __mmask16 k, const i32* VOXEL_RESTRICT src) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    _mm512_mask_compressstoreu_epi32(dst, k, v);
}

inline void CompressStoreEpI64(i64* VOXEL_RESTRICT dst, __mmask8 k, const i64* VOXEL_RESTRICT src) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    _mm512_mask_compressstoreu_epi64(dst, k, v);
}

inline void CompressStorePs(f32* VOXEL_RESTRICT dst, __mmask16 k, const f32* VOXEL_RESTRICT src) {
    __m512 v = _mm512_loadu_ps(src);
    _mm512_mask_compressstoreu_ps(dst, k, v);
}

inline void CompressStorePd(f64* VOXEL_RESTRICT dst, __mmask8 k, const f64* VOXEL_RESTRICT src) {
    __m512d v = _mm512_loadu_pd(src);
    _mm512_mask_compressstoreu_pd(dst, k, v);
}

inline __m512i ExpandEpI32(const i32* VOXEL_RESTRICT src, __mmask16 k) {
    return _mm512_mask_expand_epi32(_mm512_setzero_si512(), k, _mm512_loadu_si512(reinterpret_cast<const void*>(src)));
}

inline __m512i ExpandEpI64(const i64* VOXEL_RESTRICT src, __mmask8 k) {
    return _mm512_mask_expand_epi64(_mm512_setzero_si512(), k, _mm512_loadu_si512(reinterpret_cast<const void*>(src)));
}

inline __m512 ExpandPs(const f32* VOXEL_RESTRICT src, __mmask16 k) {
    return _mm512_mask_expand_ps(_mm512_setzero_ps(), k, _mm512_loadu_ps(src));
}

inline __m512d ExpandPd(const f64* VOXEL_RESTRICT src, __mmask8 k) {
    return _mm512_mask_expand_pd(_mm512_setzero_pd(), k, _mm512_loadu_pd(src));
}

inline __m512i CompressEpI32(__m512i v, __mmask16 k) {
    i32 buf[16] = {};
    _mm512_mask_compressstoreu_epi32(buf, k, v);
    u32 cnt = static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u16>(k))));
    __mmask16 load_mask = (cnt > 0) ? static_cast<__mmask16>((1u << cnt) - 1u) : static_cast<__mmask16>(0);
    return _mm512_maskz_loadu_epi32(load_mask, buf);
}

inline __m512i CompressEpI64(__m512i v, __mmask8 k) {
    i64 buf[8] = {};
    _mm512_mask_compressstoreu_epi64(buf, k, v);
    u32 cnt = static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u8>(k))));
    __mmask8 load_mask = (cnt > 0) ? static_cast<__mmask8>((1u << cnt) - 1u) : static_cast<__mmask8>(0);
    return _mm512_maskz_loadu_epi64(load_mask, buf);
}

inline __m512 CompressPs(__m512 v, __mmask16 k) {
    f32 buf[16] = {};
    _mm512_mask_compressstoreu_ps(buf, k, v);
    u32 cnt = static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u16>(k))));
    __mmask16 load_mask = (cnt > 0) ? static_cast<__mmask16>((1u << cnt) - 1u) : static_cast<__mmask16>(0);
    return _mm512_maskz_loadu_ps(load_mask, buf);
}

inline __m512d CompressPd(__m512d v, __mmask8 k) {
    f64 buf[8] = {};
    _mm512_mask_compressstoreu_pd(buf, k, v);
    u32 cnt = static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u8>(k))));
    __mmask8 load_mask = (cnt > 0) ? static_cast<__mmask8>((1u << cnt) - 1u) : static_cast<__mmask8>(0);
    return _mm512_maskz_loadu_pd(load_mask, buf);
}

inline __m512i DetectConflictsEpI32(__m512i v) { return _mm512_conflict_epi32(v); }

inline __m512i DetectConflictsEpI64(__m512i v) { return _mm512_conflict_epi64(v); }

inline u32 PopCountMask8(__mmask8 k) { return static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u8>(k)))); }

inline u32 PopCountMask16(__mmask16 k) { return static_cast<u32>(__builtin_popcount(static_cast<unsigned int>(static_cast<u16>(k)))); }

inline __m512i TernaryLogicEpI64(__m512i a, __m512i b, __m512i c, u8 table) {
    return _mm512_ternarylogic_epi64(a, b, c, static_cast<int>(table));
}

} // namespace simd
} // namespace voxel
