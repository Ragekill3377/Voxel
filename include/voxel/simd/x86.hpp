#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

namespace voxel {
namespace simd {

using i64x = long long;
using u64x = unsigned long long;

template<typename T> struct alignas(32) Vec256;
template<typename T> struct alignas(64) Vec512;

template<>
struct alignas(32) Vec256<f64> {
    static constexpr sz kLanes = 4;
    __m256d v;

    static Vec256 Load(const f64* p) { return {_mm256_load_pd(p)}; }
    static Vec256 LoadU(const f64* p) { return {_mm256_loadu_pd(p)}; }
    void Store(f64* p) const { _mm256_store_pd(p, v); }
    void StoreU(f64* p) const { _mm256_storeu_pd(p, v); }
    static Vec256 Set1(f64 x) { return {_mm256_set1_pd(x)}; }
    static Vec256 Zero() { return {_mm256_setzero_pd()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_pd(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_pd(v, b.v)}; }
    Vec256 Mul(Vec256 b) const { return {_mm256_mul_pd(v, b.v)}; }
    Vec256 Div(Vec256 b) const { return {_mm256_div_pd(v, b.v)}; }
    Vec256 Min(Vec256 b) const { return {_mm256_min_pd(v, b.v)}; }
    Vec256 Max(Vec256 b) const { return {_mm256_max_pd(v, b.v)}; }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_EQ_OQ)}; }
    Vec256 CmpNe(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_NEQ_OQ)}; }
    Vec256 CmpLt(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_LT_OQ)}; }
    Vec256 CmpLe(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_LE_OQ)}; }
    Vec256 CmpGt(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_GT_OQ)}; }
    Vec256 CmpGe(Vec256 b) const { return {_mm256_cmp_pd(v, b.v, _CMP_GE_OQ)}; }

    Vec256 Blend(Vec256 a, Vec256 mask) const { return {_mm256_blendv_pd(v, a.v, mask.v)}; }

    Vec256 And(Vec256 b) const { return {_mm256_and_pd(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_pd(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_pd(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_pd(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) { return {_mm256_fmadd_pd(a.v, b.v, c.v)}; }

    Vec256 Sqrt() const { return {_mm256_sqrt_pd(v)}; }

    Vec256 Abs() const { return {_mm256_andnot_pd(_mm256_set1_pd(-0.0), v)}; }

    Vec256 Neg() const { return {_mm256_xor_pd(v, _mm256_set1_pd(-0.0))}; }

    f64 HorizontalSum() const {
        __m256d t = _mm256_hadd_pd(v, v);
        __m128d lo = _mm256_castpd256_pd128(t);
        __m128d hi = _mm256_extractf128_pd(t, 1);
        __m128d sum = _mm_add_pd(lo, hi);
        return _mm_cvtsd_f64(sum);
    }

    f64 HorizontalMin() const {
        __m256d t0 = _mm256_permute_pd(v, 0x5);
        __m256d t1 = _mm256_min_pd(v, t0);
        __m256d t2 = _mm256_permute2f128_pd(t1, t1, 0x01);
        __m256d t3 = _mm256_min_pd(t1, t2);
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(t3));
    }

    f64 HorizontalMax() const {
        __m256d t0 = _mm256_permute_pd(v, 0x5);
        __m256d t1 = _mm256_max_pd(v, t0);
        __m256d t2 = _mm256_permute2f128_pd(t1, t1, 0x01);
        __m256d t3 = _mm256_max_pd(t1, t2);
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(t3));
    }

    static Vec256 Gather(const f64* base, const i32* indices, int scale) {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        return {_mm256_i32gather_pd(base, idx, scale)};
    }

    void Scatter(f64* base, const i32* indices, int scale) const {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        f64 tmp[4];
        _mm256_storeu_pd(tmp, v);
        base[_mm_extract_epi32(idx, 0) * scale] = tmp[0];
        base[_mm_extract_epi32(idx, 1) * scale] = tmp[1];
        base[_mm_extract_epi32(idx, 2) * scale] = tmp[2];
        base[_mm_extract_epi32(idx, 3) * scale] = tmp[3];
    }
};

template<>
struct alignas(32) Vec256<f32> {
    static constexpr sz kLanes = 8;
    __m256 v;

    static Vec256 Load(const f32* p) { return {_mm256_load_ps(p)}; }
    static Vec256 LoadU(const f32* p) { return {_mm256_loadu_ps(p)}; }
    void Store(f32* p) const { _mm256_store_ps(p, v); }
    void StoreU(f32* p) const { _mm256_storeu_ps(p, v); }
    static Vec256 Set1(f32 x) { return {_mm256_set1_ps(x)}; }
    static Vec256 Zero() { return {_mm256_setzero_ps()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_ps(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_ps(v, b.v)}; }
    Vec256 Mul(Vec256 b) const { return {_mm256_mul_ps(v, b.v)}; }
    Vec256 Div(Vec256 b) const { return {_mm256_div_ps(v, b.v)}; }
    Vec256 Min(Vec256 b) const { return {_mm256_min_ps(v, b.v)}; }
    Vec256 Max(Vec256 b) const { return {_mm256_max_ps(v, b.v)}; }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_EQ_OQ)}; }
    Vec256 CmpNe(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_NEQ_OQ)}; }
    Vec256 CmpLt(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_LT_OQ)}; }
    Vec256 CmpLe(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_LE_OQ)}; }
    Vec256 CmpGt(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_GT_OQ)}; }
    Vec256 CmpGe(Vec256 b) const { return {_mm256_cmp_ps(v, b.v, _CMP_GE_OQ)}; }

    Vec256 Blend(Vec256 a, Vec256 mask) const { return {_mm256_blendv_ps(v, a.v, mask.v)}; }

    Vec256 And(Vec256 b) const { return {_mm256_and_ps(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_ps(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_ps(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_ps(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) { return {_mm256_fmadd_ps(a.v, b.v, c.v)}; }

    Vec256 Sqrt() const { return {_mm256_sqrt_ps(v)}; }

    Vec256 Abs() const { return {_mm256_andnot_ps(_mm256_set1_ps(-0.0f), v)}; }

    Vec256 Neg() const { return {_mm256_xor_ps(v, _mm256_set1_ps(-0.0f))}; }

    f32 HorizontalSum() const {
        __m256 t0 = _mm256_hadd_ps(v, v);
        __m256 t1 = _mm256_hadd_ps(t0, t0);
        __m128 lo = _mm256_castps256_ps128(t1);
        __m128 hi = _mm256_extractf128_ps(t1, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        return _mm_cvtss_f32(sum);
    }

    f32 HorizontalMin() const {
        __m256 t0 = _mm256_permute_ps(v, 0xB1);
        __m256 t1 = _mm256_min_ps(v, t0);
        __m256 t2 = _mm256_permute_ps(t1, 0x4E);
        __m256 t3 = _mm256_min_ps(t1, t2);
        __m256 t4 = _mm256_permute2f128_ps(t3, t3, 0x01);
        __m256 t5 = _mm256_min_ps(t3, t4);
        return _mm_cvtss_f32(_mm256_castps256_ps128(t5));
    }

    f32 HorizontalMax() const {
        __m256 t0 = _mm256_permute_ps(v, 0xB1);
        __m256 t1 = _mm256_max_ps(v, t0);
        __m256 t2 = _mm256_permute_ps(t1, 0x4E);
        __m256 t3 = _mm256_max_ps(t1, t2);
        __m256 t4 = _mm256_permute2f128_ps(t3, t3, 0x01);
        __m256 t5 = _mm256_max_ps(t3, t4);
        return _mm_cvtss_f32(_mm256_castps256_ps128(t5));
    }

    static Vec256 Gather(const f32* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm256_i32gather_ps(base, idx, scale)};
    }

    void Scatter(f32* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        f32 tmp[8];
        _mm256_storeu_ps(tmp, v);
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 0) * scale] = tmp[0];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 1) * scale] = tmp[1];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 2) * scale] = tmp[2];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 3) * scale] = tmp[3];
        base[_mm256_extract_epi32(idx, 4) * scale] = tmp[4];
        base[_mm256_extract_epi32(idx, 5) * scale] = tmp[5];
        base[_mm256_extract_epi32(idx, 6) * scale] = tmp[6];
        base[_mm256_extract_epi32(idx, 7) * scale] = tmp[7];
    }
};

template<>
struct alignas(32) Vec256<i64> {
    static constexpr sz kLanes = 4;
    __m256i v;

    using i64_avx = long long;
    using u64_avx = unsigned long long;

    static Vec256 Load(const i64* p) { return {_mm256_load_si256(reinterpret_cast<const __m256i*>(p))}; }
    static Vec256 LoadU(const i64* p) { return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))}; }
    void Store(i64* p) const { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    void StoreU(i64* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
    static Vec256 Set1(i64 x) { return {_mm256_set1_epi64x(static_cast<i64_avx>(x))}; }
    static Vec256 Zero() { return {_mm256_setzero_si256()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_epi64(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_epi64(v, b.v)}; }
    Vec256 Mul(Vec256 b) const {
        i64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] * b_arr[0];
        r_arr[1] = a_arr[1] * b_arr[1];
        r_arr[2] = a_arr[2] * b_arr[2];
        r_arr[3] = a_arr[3] * b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Div(Vec256 b) const {
        i64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0];
        r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2];
        r_arr[3] = a_arr[3] / b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Min(Vec256 b) const {
        i64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] < b_arr[0] ? a_arr[0] : b_arr[0];
        r_arr[1] = a_arr[1] < b_arr[1] ? a_arr[1] : b_arr[1];
        r_arr[2] = a_arr[2] < b_arr[2] ? a_arr[2] : b_arr[2];
        r_arr[3] = a_arr[3] < b_arr[3] ? a_arr[3] : b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Max(Vec256 b) const {
        i64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] > b_arr[0] ? a_arr[0] : b_arr[0];
        r_arr[1] = a_arr[1] > b_arr[1] ? a_arr[1] : b_arr[1];
        r_arr[2] = a_arr[2] > b_arr[2] ? a_arr[2] : b_arr[2];
        r_arr[3] = a_arr[3] > b_arr[3] ? a_arr[3] : b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmpeq_epi64(v, b.v)}; }
    Vec256 CmpNe(Vec256 b) const {
        __m256i eq = _mm256_cmpeq_epi64(v, b.v);
        return {_mm256_andnot_si256(eq, _mm256_cmpeq_epi64(v, v))};
    }
    Vec256 CmpLt(Vec256 b) const { return {_mm256_cmpgt_epi64(b.v, v)}; }
    Vec256 CmpLe(Vec256 b) const {
        __m256i gt = _mm256_cmpgt_epi64(v, b.v);
        __m256i all = _mm256_cmpeq_epi64(v, v);
        return {_mm256_andnot_si256(gt, all)};
    }
    Vec256 CmpGt(Vec256 b) const { return {_mm256_cmpgt_epi64(v, b.v)}; }
    Vec256 CmpGe(Vec256 b) const {
        __m256i lt = _mm256_cmpgt_epi64(b.v, v);
        __m256i all = _mm256_cmpeq_epi64(v, v);
        return {_mm256_andnot_si256(lt, all)};
    }

    Vec256 Blend(Vec256 a, Vec256 mask) const {
        return {_mm256_blendv_epi8(v, a.v, mask.v)};
    }

    Vec256 And(Vec256 b) const { return {_mm256_and_si256(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_si256(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_si256(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_si256(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) {
        i64 a_arr[4], b_arr[4], c_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), a.v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_arr), c.v);
        r_arr[0] = a_arr[0] * b_arr[0] + c_arr[0];
        r_arr[1] = a_arr[1] * b_arr[1] + c_arr[1];
        r_arr[2] = a_arr[2] * b_arr[2] + c_arr[2];
        r_arr[3] = a_arr[3] * b_arr[3] + c_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }

    Vec256 Sqrt() const { return *this; }

    Vec256 Abs() const {
        i64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        a_arr[0] = a_arr[0] < 0 ? -a_arr[0] : a_arr[0];
        a_arr[1] = a_arr[1] < 0 ? -a_arr[1] : a_arr[1];
        a_arr[2] = a_arr[2] < 0 ? -a_arr[2] : a_arr[2];
        a_arr[3] = a_arr[3] < 0 ? -a_arr[3] : a_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_arr))};
    }

    Vec256 Neg() const { return {_mm256_sub_epi64(_mm256_setzero_si256(), v)}; }

    i64 HorizontalSum() const {
        i64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        return a_arr[0] + a_arr[1] + a_arr[2] + a_arr[3];
    }

    i64 HorizontalMin() const {
        i64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        i64 m = a_arr[0];
        if (a_arr[1] < m) m = a_arr[1];
        if (a_arr[2] < m) m = a_arr[2];
        if (a_arr[3] < m) m = a_arr[3];
        return m;
    }

    i64 HorizontalMax() const {
        i64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        i64 m = a_arr[0];
        if (a_arr[1] > m) m = a_arr[1];
        if (a_arr[2] > m) m = a_arr[2];
        if (a_arr[3] > m) m = a_arr[3];
        return m;
    }

    static Vec256 Gather(const i64* base, const i32* indices, int scale) {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        return {_mm256_i32gather_epi64(reinterpret_cast<const i64_avx*>(base), idx, scale)};
    }

    void Scatter(i64* base, const i32* indices, int scale) const {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        i64 tmp[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v);
        base[_mm_extract_epi32(idx, 0) * scale] = tmp[0];
        base[_mm_extract_epi32(idx, 1) * scale] = tmp[1];
        base[_mm_extract_epi32(idx, 2) * scale] = tmp[2];
        base[_mm_extract_epi32(idx, 3) * scale] = tmp[3];
    }
};

template<>
struct alignas(32) Vec256<u64> {
    static constexpr sz kLanes = 4;
    __m256i v;

    using i64_avx = long long;
    using u64_avx = unsigned long long;

    static Vec256 Load(const u64* p) { return {_mm256_load_si256(reinterpret_cast<const __m256i*>(p))}; }
    static Vec256 LoadU(const u64* p) { return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))}; }
    void Store(u64* p) const { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    void StoreU(u64* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
    static Vec256 Set1(u64 x) { return {_mm256_set1_epi64x(static_cast<i64_avx>(x))}; }
    static Vec256 Zero() { return {_mm256_setzero_si256()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_epi64(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_epi64(v, b.v)}; }
    Vec256 Mul(Vec256 b) const {
        u64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] * b_arr[0];
        r_arr[1] = a_arr[1] * b_arr[1];
        r_arr[2] = a_arr[2] * b_arr[2];
        r_arr[3] = a_arr[3] * b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Div(Vec256 b) const {
        u64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0];
        r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2];
        r_arr[3] = a_arr[3] / b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Min(Vec256 b) const {
        u64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] < b_arr[0] ? a_arr[0] : b_arr[0];
        r_arr[1] = a_arr[1] < b_arr[1] ? a_arr[1] : b_arr[1];
        r_arr[2] = a_arr[2] < b_arr[2] ? a_arr[2] : b_arr[2];
        r_arr[3] = a_arr[3] < b_arr[3] ? a_arr[3] : b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Max(Vec256 b) const {
        u64 a_arr[4], b_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] > b_arr[0] ? a_arr[0] : b_arr[0];
        r_arr[1] = a_arr[1] > b_arr[1] ? a_arr[1] : b_arr[1];
        r_arr[2] = a_arr[2] > b_arr[2] ? a_arr[2] : b_arr[2];
        r_arr[3] = a_arr[3] > b_arr[3] ? a_arr[3] : b_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmpeq_epi64(v, b.v)}; }
    Vec256 CmpNe(Vec256 b) const {
        __m256i eq = _mm256_cmpeq_epi64(v, b.v);
        return {_mm256_andnot_si256(eq, _mm256_cmpeq_epi64(v, v))};
    }
    Vec256 CmpLt(Vec256 b) const {
        __m256i sign = _mm256_set1_epi64x(0x8000000000000000LL);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        return {_mm256_cmpgt_epi64(b_xor, a_xor)};
    }
    Vec256 CmpLe(Vec256 b) const {
        __m256i sign = _mm256_set1_epi64x(0x8000000000000000LL);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        __m256i gt = _mm256_cmpgt_epi64(a_xor, b_xor);
        __m256i all = _mm256_cmpeq_epi64(v, v);
        return {_mm256_andnot_si256(gt, all)};
    }
    Vec256 CmpGt(Vec256 b) const {
        __m256i sign = _mm256_set1_epi64x(0x8000000000000000LL);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        return {_mm256_cmpgt_epi64(a_xor, b_xor)};
    }
    Vec256 CmpGe(Vec256 b) const {
        __m256i sign = _mm256_set1_epi64x(0x8000000000000000LL);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        __m256i lt = _mm256_cmpgt_epi64(b_xor, a_xor);
        __m256i all = _mm256_cmpeq_epi64(v, v);
        return {_mm256_andnot_si256(lt, all)};
    }

    Vec256 Blend(Vec256 a, Vec256 mask) const {
        return {_mm256_blendv_epi8(v, a.v, mask.v)};
    }

    Vec256 And(Vec256 b) const { return {_mm256_and_si256(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_si256(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_si256(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_si256(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) {
        u64 a_arr[4], b_arr[4], c_arr[4], r_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), a.v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_arr), c.v);
        r_arr[0] = a_arr[0] * b_arr[0] + c_arr[0];
        r_arr[1] = a_arr[1] * b_arr[1] + c_arr[1];
        r_arr[2] = a_arr[2] * b_arr[2] + c_arr[2];
        r_arr[3] = a_arr[3] * b_arr[3] + c_arr[3];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }

    Vec256 Sqrt() const { return *this; }

    Vec256 Abs() const { return *this; }

    Vec256 Neg() const { return {_mm256_sub_epi64(_mm256_setzero_si256(), v)}; }

    u64 HorizontalSum() const {
        u64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        return a_arr[0] + a_arr[1] + a_arr[2] + a_arr[3];
    }

    u64 HorizontalMin() const {
        u64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        u64 m = a_arr[0];
        if (a_arr[1] < m) m = a_arr[1];
        if (a_arr[2] < m) m = a_arr[2];
        if (a_arr[3] < m) m = a_arr[3];
        return m;
    }

    u64 HorizontalMax() const {
        u64 a_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        u64 m = a_arr[0];
        if (a_arr[1] > m) m = a_arr[1];
        if (a_arr[2] > m) m = a_arr[2];
        if (a_arr[3] > m) m = a_arr[3];
        return m;
    }

    static Vec256 Gather(const u64* base, const i32* indices, int scale) {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        return {_mm256_i32gather_epi64(reinterpret_cast<const i64_avx*>(base), idx, scale)};
    }

    void Scatter(u64* base, const i32* indices, int scale) const {
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
        u64 tmp[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v);
        base[_mm_extract_epi32(idx, 0) * scale] = tmp[0];
        base[_mm_extract_epi32(idx, 1) * scale] = tmp[1];
        base[_mm_extract_epi32(idx, 2) * scale] = tmp[2];
        base[_mm_extract_epi32(idx, 3) * scale] = tmp[3];
    }
};

template<>
struct alignas(32) Vec256<i32> {
    static constexpr sz kLanes = 8;
    __m256i v;

    static Vec256 Load(const i32* p) { return {_mm256_load_si256(reinterpret_cast<const __m256i*>(p))}; }
    static Vec256 LoadU(const i32* p) { return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))}; }
    void Store(i32* p) const { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    void StoreU(i32* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
    static Vec256 Set1(i32 x) { return {_mm256_set1_epi32(x)}; }
    static Vec256 Zero() { return {_mm256_setzero_si256()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_epi32(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_epi32(v, b.v)}; }
    Vec256 Mul(Vec256 b) const { return {_mm256_mullo_epi32(v, b.v)}; }
    Vec256 Div(Vec256 b) const {
        i32 a_arr[8], b_arr[8], r_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0]; r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2]; r_arr[3] = a_arr[3] / b_arr[3];
        r_arr[4] = a_arr[4] / b_arr[4]; r_arr[5] = a_arr[5] / b_arr[5];
        r_arr[6] = a_arr[6] / b_arr[6]; r_arr[7] = a_arr[7] / b_arr[7];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Min(Vec256 b) const { return {_mm256_min_epi32(v, b.v)}; }
    Vec256 Max(Vec256 b) const { return {_mm256_max_epi32(v, b.v)}; }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmpeq_epi32(v, b.v)}; }
    Vec256 CmpNe(Vec256 b) const {
        __m256i eq = _mm256_cmpeq_epi32(v, b.v);
        return {_mm256_andnot_si256(eq, _mm256_cmpeq_epi32(v, v))};
    }
    Vec256 CmpLt(Vec256 b) const { return {_mm256_cmpgt_epi32(b.v, v)}; }
    Vec256 CmpLe(Vec256 b) const {
        __m256i gt = _mm256_cmpgt_epi32(v, b.v);
        __m256i all = _mm256_cmpeq_epi32(v, v);
        return {_mm256_andnot_si256(gt, all)};
    }
    Vec256 CmpGt(Vec256 b) const { return {_mm256_cmpgt_epi32(v, b.v)}; }
    Vec256 CmpGe(Vec256 b) const {
        __m256i lt = _mm256_cmpgt_epi32(b.v, v);
        __m256i all = _mm256_cmpeq_epi32(v, v);
        return {_mm256_andnot_si256(lt, all)};
    }

    Vec256 Blend(Vec256 a, Vec256 mask) const {
        return {_mm256_blendv_epi8(v, a.v, mask.v)};
    }

    Vec256 And(Vec256 b) const { return {_mm256_and_si256(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_si256(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_si256(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_si256(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) {
        return {_mm256_add_epi32(_mm256_mullo_epi32(a.v, b.v), c.v)};
    }

    Vec256 Sqrt() const { return *this; }

    Vec256 Abs() const { return {_mm256_abs_epi32(v)}; }

    Vec256 Neg() const { return {_mm256_sub_epi32(_mm256_setzero_si256(), v)}; }

    i32 HorizontalSum() const {
        i32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        return a_arr[0] + a_arr[1] + a_arr[2] + a_arr[3] + a_arr[4] + a_arr[5] + a_arr[6] + a_arr[7];
    }

    i32 HorizontalMin() const {
        i32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        i32 m = a_arr[0];
if (a_arr[1] < m) m = a_arr[1]; 
if (a_arr[2] < m) m = a_arr[2];
if (a_arr[3] < m) m = a_arr[3]; 
if (a_arr[4] < m) m = a_arr[4];
if (a_arr[5] < m) m = a_arr[5]; 
if (a_arr[6] < m) m = a_arr[6];
        if (a_arr[7] < m) m = a_arr[7];
        return m;
    }

    i32 HorizontalMax() const {
        i32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        i32 m = a_arr[0];
if (a_arr[1] > m) m = a_arr[1]; 
if (a_arr[2] > m) m = a_arr[2];
if (a_arr[3] > m) m = a_arr[3]; 
if (a_arr[4] > m) m = a_arr[4];
if (a_arr[5] > m) m = a_arr[5]; 
if (a_arr[6] > m) m = a_arr[6];
        if (a_arr[7] > m) m = a_arr[7];
        return m;
    }

    static Vec256 Gather(const i32* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm256_i32gather_epi32(base, idx, scale)};
    }

    void Scatter(i32* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        i32 tmp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v);
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 0) * scale] = tmp[0];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 1) * scale] = tmp[1];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 2) * scale] = tmp[2];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 3) * scale] = tmp[3];
        base[_mm256_extract_epi32(idx, 4) * scale] = tmp[4];
        base[_mm256_extract_epi32(idx, 5) * scale] = tmp[5];
        base[_mm256_extract_epi32(idx, 6) * scale] = tmp[6];
        base[_mm256_extract_epi32(idx, 7) * scale] = tmp[7];
    }
};

template<>
struct alignas(32) Vec256<u32> {
    static constexpr sz kLanes = 8;
    __m256i v;

    static Vec256 Load(const u32* p) { return {_mm256_load_si256(reinterpret_cast<const __m256i*>(p))}; }
    static Vec256 LoadU(const u32* p) { return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))}; }
    void Store(u32* p) const { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    void StoreU(u32* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
    static Vec256 Set1(u32 x) { return {_mm256_set1_epi32(static_cast<i32>(x))}; }
    static Vec256 Zero() { return {_mm256_setzero_si256()}; }

    Vec256 Add(Vec256 b) const { return {_mm256_add_epi32(v, b.v)}; }
    Vec256 Sub(Vec256 b) const { return {_mm256_sub_epi32(v, b.v)}; }
    Vec256 Mul(Vec256 b) const { return {_mm256_mullo_epi32(v, b.v)}; }
    Vec256 Div(Vec256 b) const {
        u32 a_arr[8], b_arr[8], r_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0]; r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2]; r_arr[3] = a_arr[3] / b_arr[3];
        r_arr[4] = a_arr[4] / b_arr[4]; r_arr[5] = a_arr[5] / b_arr[5];
        r_arr[6] = a_arr[6] / b_arr[6]; r_arr[7] = a_arr[7] / b_arr[7];
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_arr))};
    }
    Vec256 Min(Vec256 b) const { return {_mm256_min_epu32(v, b.v)}; }
    Vec256 Max(Vec256 b) const { return {_mm256_max_epu32(v, b.v)}; }

    Vec256 CmpEq(Vec256 b) const { return {_mm256_cmpeq_epi32(v, b.v)}; }
    Vec256 CmpNe(Vec256 b) const {
        __m256i eq = _mm256_cmpeq_epi32(v, b.v);
        return {_mm256_andnot_si256(eq, _mm256_cmpeq_epi32(v, v))};
    }
    Vec256 CmpLt(Vec256 b) const {
        __m256i sign = _mm256_set1_epi32(0x80000000);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        return {_mm256_cmpgt_epi32(b_xor, a_xor)};
    }
    Vec256 CmpLe(Vec256 b) const {
        __m256i sign = _mm256_set1_epi32(0x80000000);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        __m256i gt = _mm256_cmpgt_epi32(a_xor, b_xor);
        __m256i all = _mm256_cmpeq_epi32(v, v);
        return {_mm256_andnot_si256(gt, all)};
    }
    Vec256 CmpGt(Vec256 b) const {
        __m256i sign = _mm256_set1_epi32(0x80000000);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        return {_mm256_cmpgt_epi32(a_xor, b_xor)};
    }
    Vec256 CmpGe(Vec256 b) const {
        __m256i sign = _mm256_set1_epi32(0x80000000);
        __m256i a_xor = _mm256_xor_si256(v, sign);
        __m256i b_xor = _mm256_xor_si256(b.v, sign);
        __m256i lt = _mm256_cmpgt_epi32(b_xor, a_xor);
        __m256i all = _mm256_cmpeq_epi32(v, v);
        return {_mm256_andnot_si256(lt, all)};
    }

    Vec256 Blend(Vec256 a, Vec256 mask) const {
        return {_mm256_blendv_epi8(v, a.v, mask.v)};
    }

    Vec256 And(Vec256 b) const { return {_mm256_and_si256(v, b.v)}; }
    Vec256 Or(Vec256 b) const  { return {_mm256_or_si256(v, b.v)}; }
    Vec256 Xor(Vec256 b) const { return {_mm256_xor_si256(v, b.v)}; }
    Vec256 AndNot(Vec256 b) const { return {_mm256_andnot_si256(b.v, v)}; }

    static Vec256 FMA(Vec256 a, Vec256 b, Vec256 c) {
        return {_mm256_add_epi32(_mm256_mullo_epi32(a.v, b.v), c.v)};
    }

    Vec256 Sqrt() const { return *this; }

    Vec256 Abs() const { return *this; }

    Vec256 Neg() const { return {_mm256_sub_epi32(_mm256_setzero_si256(), v)}; }

    u32 HorizontalSum() const {
        u32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        return a_arr[0] + a_arr[1] + a_arr[2] + a_arr[3] + a_arr[4] + a_arr[5] + a_arr[6] + a_arr[7];
    }

    u32 HorizontalMin() const {
        u32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        u32 m = a_arr[0];
if (a_arr[1] < m) m = a_arr[1]; 
if (a_arr[2] < m) m = a_arr[2];
if (a_arr[3] < m) m = a_arr[3]; 
if (a_arr[4] < m) m = a_arr[4];
if (a_arr[5] < m) m = a_arr[5]; 
if (a_arr[6] < m) m = a_arr[6];
        if (a_arr[7] < m) m = a_arr[7];
        return m;
    }

    u32 HorizontalMax() const {
        u32 a_arr[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a_arr), v);
        u32 m = a_arr[0];
if (a_arr[1] > m) m = a_arr[1]; 
if (a_arr[2] > m) m = a_arr[2];
if (a_arr[3] > m) m = a_arr[3]; 
if (a_arr[4] > m) m = a_arr[4];
if (a_arr[5] > m) m = a_arr[5]; 
if (a_arr[6] > m) m = a_arr[6];
        if (a_arr[7] > m) m = a_arr[7];
        return m;
    }

    static Vec256 Gather(const u32* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), idx, scale)};
    }

    void Scatter(u32* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        u32 tmp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v);
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 0) * scale] = tmp[0];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 1) * scale] = tmp[1];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 2) * scale] = tmp[2];
        base[_mm_extract_epi32(_mm256_castsi256_si128(idx), 3) * scale] = tmp[3];
        base[_mm256_extract_epi32(idx, 4) * scale] = tmp[4];
        base[_mm256_extract_epi32(idx, 5) * scale] = tmp[5];
        base[_mm256_extract_epi32(idx, 6) * scale] = tmp[6];
        base[_mm256_extract_epi32(idx, 7) * scale] = tmp[7];
    }
};

#if VOXEL_SIMD_AVX512
template<>
struct alignas(64) Vec512<f64> {
    static constexpr sz kLanes = 8;
    __m512d v;

    static Vec512 Load(const f64* p) { return {_mm512_load_pd(p)}; }
    static Vec512 LoadU(const f64* p) { return {_mm512_loadu_pd(p)}; }
    void Store(f64* p) const { _mm512_store_pd(p, v); }
    void StoreU(f64* p) const { _mm512_storeu_pd(p, v); }
    static Vec512 Set1(f64 x) { return {_mm512_set1_pd(x)}; }
    static Vec512 Zero() { return {_mm512_setzero_pd()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_pd(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_pd(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mul_pd(v, b.v)}; }
    Vec512 Div(Vec512 b) const { return {_mm512_div_pd(v, b.v)}; }
    Vec512 Min(Vec512 b) const { return {_mm512_min_pd(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_pd(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_EQ_OQ)}; }
    Vec512 CmpNe(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_NEQ_OQ)}; }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_LT_OQ)}; }
    Vec512 CmpLe(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_LE_OQ)}; }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_GT_OQ)}; }
    Vec512 CmpGe(Vec512 b) const { return {_mm512_cmp_pd(v, b.v, _CMP_GE_OQ)}; }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask8 k = _mm512_movepi64_mask(_mm512_castpd_si512(mask.v));
        return {_mm512_mask_blend_pd(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_pd(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_pd(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_pd(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_pd(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) { return {_mm512_fmadd_pd(a.v, b.v, c.v)}; }

    Vec512 Sqrt() const { return {_mm512_sqrt_pd(v)}; }

    Vec512 Abs() const { return {_mm512_abs_pd(v)}; }

    Vec512 Neg() const { return {_mm512_xor_pd(v, _mm512_set1_pd(-0.0))}; }

    f64 HorizontalSum() const { return _mm512_reduce_add_pd(v); }

    f64 HorizontalMin() const { return _mm512_reduce_min_pd(v); }

    f64 HorizontalMax() const { return _mm512_reduce_max_pd(v); }

    static Vec512 Gather(const f64* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm512_i32gather_pd(idx, base, scale)};
    }

    void Scatter(f64* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        _mm512_i32scatter_pd(base, idx, v, scale);
    }

    static Vec512 MaskedLoad(const f64* p, __mmask8 k) { return {_mm512_mask_load_pd(_mm512_setzero_pd(), k, p)}; }
    void MaskedStore(f64* p, __mmask8 k) const { _mm512_mask_store_pd(p, k, v); }
    void CompressStore(f64* p, __mmask8 k) const { _mm512_mask_compressstoreu_pd(p, k, v); }
};

template<>
struct alignas(64) Vec512<f32> {
    static constexpr sz kLanes = 16;
    __m512 v;

    static Vec512 Load(const f32* p) { return {_mm512_load_ps(p)}; }
    static Vec512 LoadU(const f32* p) { return {_mm512_loadu_ps(p)}; }
    void Store(f32* p) const { _mm512_store_ps(p, v); }
    void StoreU(f32* p) const { _mm512_storeu_ps(p, v); }
    static Vec512 Set1(f32 x) { return {_mm512_set1_ps(x)}; }
    static Vec512 Zero() { return {_mm512_setzero_ps()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_ps(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_ps(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mul_ps(v, b.v)}; }
    Vec512 Div(Vec512 b) const { return {_mm512_div_ps(v, b.v)}; }
    Vec512 Min(Vec512 b) const { return {_mm512_min_ps(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_ps(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_EQ_OQ)}; }
    Vec512 CmpNe(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_NEQ_OQ)}; }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_LT_OQ)}; }
    Vec512 CmpLe(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_LE_OQ)}; }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_GT_OQ)}; }
    Vec512 CmpGe(Vec512 b) const { return {_mm512_cmp_ps(v, b.v, _CMP_GE_OQ)}; }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask16 k = _mm512_movepi32_mask(_mm512_castps_si512(mask.v));
        return {_mm512_mask_blend_ps(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_ps(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_ps(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_ps(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_ps(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) { return {_mm512_fmadd_ps(a.v, b.v, c.v)}; }

    Vec512 Sqrt() const { return {_mm512_sqrt_ps(v)}; }

    Vec512 Abs() const { return {_mm512_abs_ps(v)}; }

    Vec512 Neg() const { return {_mm512_xor_ps(v, _mm512_set1_ps(-0.0f))}; }

    f32 HorizontalSum() const { return _mm512_reduce_add_ps(v); }

    f32 HorizontalMin() const { return _mm512_reduce_min_ps(v); }

    f32 HorizontalMax() const { return _mm512_reduce_max_ps(v); }

    static Vec512 Gather(const f32* base, const i32* indices, int scale) {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        return {_mm512_i32gather_ps(idx, base, scale)};
    }

    void Scatter(f32* base, const i32* indices, int scale) const {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        _mm512_i32scatter_ps(base, idx, v, scale);
    }

    static Vec512 MaskedLoad(const f32* p, __mmask16 k) { return {_mm512_mask_load_ps(_mm512_setzero_ps(), k, p)}; }
    void MaskedStore(f32* p, __mmask16 k) const { _mm512_mask_store_ps(p, k, v); }
    void CompressStore(f32* p, __mmask16 k) const { _mm512_mask_compressstoreu_ps(p, k, v); }
};

template<>
struct alignas(64) Vec512<i64> {
    static constexpr sz kLanes = 8;
    __m512i v;

    using i64_avx = long long;
    using u64_avx = unsigned long long;

    static Vec512 Load(const i64* p) { return {_mm512_load_si512(reinterpret_cast<const void*>(p))}; }
    static Vec512 LoadU(const i64* p) { return {_mm512_loadu_si512(reinterpret_cast<const void*>(p))}; }
    void Store(i64* p) const { _mm512_store_si512(reinterpret_cast<void*>(p), v); }
    void StoreU(i64* p) const { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    static Vec512 Set1(i64 x) { return {_mm512_set1_epi64(static_cast<i64_avx>(x))}; }
    static Vec512 Zero() { return {_mm512_setzero_si512()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_epi64(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_epi64(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mullo_epi64(v, b.v)}; }
    Vec512 Div(Vec512 b) const {
        i64 a_arr[8], b_arr[8], r_arr[8];
        _mm512_storeu_si512(reinterpret_cast<void*>(a_arr), v);
        _mm512_storeu_si512(reinterpret_cast<void*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0]; r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2]; r_arr[3] = a_arr[3] / b_arr[3];
        r_arr[4] = a_arr[4] / b_arr[4]; r_arr[5] = a_arr[5] / b_arr[5];
        r_arr[6] = a_arr[6] / b_arr[6]; r_arr[7] = a_arr[7] / b_arr[7];
        return {_mm512_loadu_si512(reinterpret_cast<const void*>(r_arr))};
    }
    Vec512 Min(Vec512 b) const { return {_mm512_min_epi64(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_epi64(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmpeq_epi64(v, b.v)}; }
    Vec512 CmpNe(Vec512 b) const {
        __m512i eq = _mm512_cmpeq_epi64(v, b.v);
        __m512i all = _mm512_set1_epi64(-1);
        return {_mm512_andnot_si512(eq, all)};
    }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmpgt_epi64(b.v, v)}; }
    Vec512 CmpLe(Vec512 b) const {
        __m512i gt = _mm512_cmpgt_epi64(v, b.v);
        __m512i all = _mm512_set1_epi64(-1);
        return {_mm512_andnot_si512(gt, all)};
    }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmpgt_epi64(v, b.v)}; }
    Vec512 CmpGe(Vec512 b) const {
        __m512i lt = _mm512_cmpgt_epi64(b.v, v);
        __m512i all = _mm512_set1_epi64(-1);
        return {_mm512_andnot_si512(lt, all)};
    }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask8 k = _mm512_movepi64_mask(mask.v);
        return {_mm512_mask_blend_epi64(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_si512(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_si512(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_si512(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_si512(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) {
        return {_mm512_add_epi64(_mm512_mullo_epi64(a.v, b.v), c.v)};
    }

    Vec512 Sqrt() const { return *this; }

    Vec512 Abs() const { return {_mm512_abs_epi64(v)}; }

    Vec512 Neg() const { return {_mm512_sub_epi64(_mm512_setzero_si512(), v)}; }

    i64 HorizontalSum() const { return _mm512_reduce_add_epi64(v); }

    i64 HorizontalMin() const { return _mm512_reduce_min_epi64(v); }

    i64 HorizontalMax() const { return _mm512_reduce_max_epi64(v); }

    static Vec512 Gather(const i64* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm512_i32gather_epi64(idx, reinterpret_cast<const i64_avx*>(base), scale)};
    }

    void Scatter(i64* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        _mm512_i32scatter_epi64(reinterpret_cast<void*>(base), idx, v, scale);
    }

    static Vec512 MaskedLoad(const i64* p, __mmask8 k) { return {_mm512_mask_load_epi64(_mm512_setzero_si512(), k, reinterpret_cast<const void*>(p))}; }
    void MaskedStore(i64* p, __mmask8 k) const { _mm512_mask_store_epi64(p, k, v); }
    void CompressStore(i64* p, __mmask8 k) const { _mm512_mask_compressstoreu_epi64(p, k, v); }
};

template<>
struct alignas(64) Vec512<u64> {
    static constexpr sz kLanes = 8;
    __m512i v;

    using i64_avx = long long;
    using u64_avx = unsigned long long;

    static Vec512 Load(const u64* p) { return {_mm512_load_si512(reinterpret_cast<const void*>(p))}; }
    static Vec512 LoadU(const u64* p) { return {_mm512_loadu_si512(reinterpret_cast<const void*>(p))}; }
    void Store(u64* p) const { _mm512_store_si512(reinterpret_cast<void*>(p), v); }
    void StoreU(u64* p) const { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    static Vec512 Set1(u64 x) { return {_mm512_set1_epi64(static_cast<i64_avx>(x))}; }
    static Vec512 Zero() { return {_mm512_setzero_si512()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_epi64(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_epi64(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mullo_epi64(v, b.v)}; }
    Vec512 Div(Vec512 b) const {
        u64 a_arr[8], b_arr[8], r_arr[8];
        _mm512_storeu_si512(reinterpret_cast<void*>(a_arr), v);
        _mm512_storeu_si512(reinterpret_cast<void*>(b_arr), b.v);
        r_arr[0] = a_arr[0] / b_arr[0]; r_arr[1] = a_arr[1] / b_arr[1];
        r_arr[2] = a_arr[2] / b_arr[2]; r_arr[3] = a_arr[3] / b_arr[3];
        r_arr[4] = a_arr[4] / b_arr[4]; r_arr[5] = a_arr[5] / b_arr[5];
        r_arr[6] = a_arr[6] / b_arr[6]; r_arr[7] = a_arr[7] / b_arr[7];
        return {_mm512_loadu_si512(reinterpret_cast<const void*>(r_arr))};
    }
    Vec512 Min(Vec512 b) const { return {_mm512_min_epu64(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_epu64(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmpeq_epi64(v, b.v)}; }
    Vec512 CmpNe(Vec512 b) const {
        __m512i eq = _mm512_cmpeq_epi64(v, b.v);
        __m512i all = _mm512_set1_epi64(-1);
        return {_mm512_andnot_si512(eq, all)};
    }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmplt_epu64(v, b.v)}; }
    Vec512 CmpLe(Vec512 b) const { return {_mm512_cmple_epu64(v, b.v)}; }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmpgt_epu64(v, b.v)}; }
    Vec512 CmpGe(Vec512 b) const { return {_mm512_cmpge_epu64(v, b.v)}; }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask8 k = _mm512_movepi64_mask(mask.v);
        return {_mm512_mask_blend_epi64(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_si512(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_si512(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_si512(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_si512(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) {
        return {_mm512_add_epi64(_mm512_mullo_epi64(a.v, b.v), c.v)};
    }

    Vec512 Sqrt() const { return *this; }

    Vec512 Abs() const { return *this; }

    Vec512 Neg() const { return {_mm512_sub_epi64(_mm512_setzero_si512(), v)}; }

    u64 HorizontalSum() const { return static_cast<u64>(_mm512_reduce_add_epi64(v)); }

    u64 HorizontalMin() const { return static_cast<u64>(_mm512_reduce_min_epu64(v)); }

    u64 HorizontalMax() const { return static_cast<u64>(_mm512_reduce_max_epu64(v)); }

    static Vec512 Gather(const u64* base, const i32* indices, int scale) {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        return {_mm512_i32gather_epi64(idx, reinterpret_cast<const i64_avx*>(base), scale)};
    }

    void Scatter(u64* base, const i32* indices, int scale) const {
        __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
        _mm512_i32scatter_epi64(reinterpret_cast<void*>(base), idx, v, scale);
    }

    static Vec512 MaskedLoad(const u64* p, __mmask8 k) { return {_mm512_mask_load_epi64(_mm512_setzero_si512(), k, reinterpret_cast<const void*>(p))}; }
    void MaskedStore(u64* p, __mmask8 k) const { _mm512_mask_store_epi64(p, k, v); }
    void CompressStore(u64* p, __mmask8 k) const { _mm512_mask_compressstoreu_epi64(p, k, v); }
};

template<>
struct alignas(64) Vec512<i32> {
    static constexpr sz kLanes = 16;
    __m512i v;

    static Vec512 Load(const i32* p) { return {_mm512_load_si512(reinterpret_cast<const void*>(p))}; }
    static Vec512 LoadU(const i32* p) { return {_mm512_loadu_si512(reinterpret_cast<const void*>(p))}; }
    void Store(i32* p) const { _mm512_store_si512(reinterpret_cast<void*>(p), v); }
    void StoreU(i32* p) const { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    static Vec512 Set1(i32 x) { return {_mm512_set1_epi32(x)}; }
    static Vec512 Zero() { return {_mm512_setzero_si512()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_epi32(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_epi32(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mullo_epi32(v, b.v)}; }
    Vec512 Div(Vec512 b) const {
        i32 a_arr[16], b_arr[16], r_arr[16];
        _mm512_storeu_si512(reinterpret_cast<void*>(a_arr), v);
        _mm512_storeu_si512(reinterpret_cast<void*>(b_arr), b.v);
        for (int i = 0; i < 16; ++i) r_arr[i] = a_arr[i] / b_arr[i];
        return {_mm512_loadu_si512(reinterpret_cast<const void*>(r_arr))};
    }
    Vec512 Min(Vec512 b) const { return {_mm512_min_epi32(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_epi32(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmpeq_epi32(v, b.v)}; }
    Vec512 CmpNe(Vec512 b) const {
        __m512i eq = _mm512_cmpeq_epi32(v, b.v);
        __m512i all = _mm512_set1_epi32(-1);
        return {_mm512_andnot_si512(eq, all)};
    }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmpgt_epi32(b.v, v)}; }
    Vec512 CmpLe(Vec512 b) const {
        __m512i gt = _mm512_cmpgt_epi32(v, b.v);
        __m512i all = _mm512_set1_epi32(-1);
        return {_mm512_andnot_si512(gt, all)};
    }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmpgt_epi32(v, b.v)}; }
    Vec512 CmpGe(Vec512 b) const {
        __m512i lt = _mm512_cmpgt_epi32(b.v, v);
        __m512i all = _mm512_set1_epi32(-1);
        return {_mm512_andnot_si512(lt, all)};
    }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask16 k = _mm512_movepi32_mask(mask.v);
        return {_mm512_mask_blend_epi32(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_si512(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_si512(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_si512(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_si512(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) {
        return {_mm512_add_epi32(_mm512_mullo_epi32(a.v, b.v), c.v)};
    }

    Vec512 Sqrt() const { return *this; }

    Vec512 Abs() const { return {_mm512_abs_epi32(v)}; }

    Vec512 Neg() const { return {_mm512_sub_epi32(_mm512_setzero_si512(), v)}; }

    i32 HorizontalSum() const { return _mm512_reduce_add_epi32(v); }

    i32 HorizontalMin() const { return _mm512_reduce_min_epi32(v); }

    i32 HorizontalMax() const { return _mm512_reduce_max_epi32(v); }

    static Vec512 Gather(const i32* base, const i32* indices, int scale) {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        return {_mm512_i32gather_epi32(idx, base, scale)};
    }

    void Scatter(i32* base, const i32* indices, int scale) const {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        _mm512_i32scatter_epi32(base, idx, v, scale);
    }

    static Vec512 MaskedLoad(const i32* p, __mmask16 k) { return {_mm512_mask_load_epi32(_mm512_setzero_si512(), k, p)}; }
    void MaskedStore(i32* p, __mmask16 k) const { _mm512_mask_store_epi32(p, k, v); }
    void CompressStore(i32* p, __mmask16 k) const { _mm512_mask_compressstoreu_epi32(p, k, v); }
};

template<>
struct alignas(64) Vec512<u32> {
    static constexpr sz kLanes = 16;
    __m512i v;

    static Vec512 Load(const u32* p) { return {_mm512_load_si512(reinterpret_cast<const void*>(p))}; }
    static Vec512 LoadU(const u32* p) { return {_mm512_loadu_si512(reinterpret_cast<const void*>(p))}; }
    void Store(u32* p) const { _mm512_store_si512(reinterpret_cast<void*>(p), v); }
    void StoreU(u32* p) const { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    static Vec512 Set1(u32 x) { return {_mm512_set1_epi32(static_cast<i32>(x))}; }
    static Vec512 Zero() { return {_mm512_setzero_si512()}; }

    Vec512 Add(Vec512 b) const { return {_mm512_add_epi32(v, b.v)}; }
    Vec512 Sub(Vec512 b) const { return {_mm512_sub_epi32(v, b.v)}; }
    Vec512 Mul(Vec512 b) const { return {_mm512_mullo_epi32(v, b.v)}; }
    Vec512 Div(Vec512 b) const {
        u32 a_arr[16], b_arr[16], r_arr[16];
        _mm512_storeu_si512(reinterpret_cast<void*>(a_arr), v);
        _mm512_storeu_si512(reinterpret_cast<void*>(b_arr), b.v);
        for (int i = 0; i < 16; ++i) r_arr[i] = a_arr[i] / b_arr[i];
        return {_mm512_loadu_si512(reinterpret_cast<const void*>(r_arr))};
    }
    Vec512 Min(Vec512 b) const { return {_mm512_min_epu32(v, b.v)}; }
    Vec512 Max(Vec512 b) const { return {_mm512_max_epu32(v, b.v)}; }

    Vec512 CmpEq(Vec512 b) const { return {_mm512_cmpeq_epi32(v, b.v)}; }
    Vec512 CmpNe(Vec512 b) const {
        __m512i eq = _mm512_cmpeq_epi32(v, b.v);
        __m512i all = _mm512_set1_epi32(-1);
        return {_mm512_andnot_si512(eq, all)};
    }
    Vec512 CmpLt(Vec512 b) const { return {_mm512_cmplt_epu32(v, b.v)}; }
    Vec512 CmpLe(Vec512 b) const { return {_mm512_cmple_epu32(v, b.v)}; }
    Vec512 CmpGt(Vec512 b) const { return {_mm512_cmpgt_epu32(v, b.v)}; }
    Vec512 CmpGe(Vec512 b) const { return {_mm512_cmpge_epu32(v, b.v)}; }

    Vec512 Blend(Vec512 a, Vec512 mask) const {
        __mmask16 k = _mm512_movepi32_mask(mask.v);
        return {_mm512_mask_blend_epi32(k, v, a.v)};
    }

    Vec512 And(Vec512 b) const { return {_mm512_and_si512(v, b.v)}; }
    Vec512 Or(Vec512 b) const  { return {_mm512_or_si512(v, b.v)}; }
    Vec512 Xor(Vec512 b) const { return {_mm512_xor_si512(v, b.v)}; }
    Vec512 AndNot(Vec512 b) const { return {_mm512_andnot_si512(b.v, v)}; }

    static Vec512 FMA(Vec512 a, Vec512 b, Vec512 c) {
        return {_mm512_add_epi32(_mm512_mullo_epi32(a.v, b.v), c.v)};
    }

    Vec512 Sqrt() const { return *this; }

    Vec512 Abs() const { return *this; }

    Vec512 Neg() const { return {_mm512_sub_epi32(_mm512_setzero_si512(), v)}; }

    u32 HorizontalSum() const { return static_cast<u32>(_mm512_reduce_add_epi32(v)); }

    u32 HorizontalMin() const { return static_cast<u32>(_mm512_reduce_min_epu32(v)); }

    u32 HorizontalMax() const { return static_cast<u32>(_mm512_reduce_max_epu32(v)); }

    static Vec512 Gather(const u32* base, const i32* indices, int scale) {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        return {_mm512_i32gather_epi32(idx, reinterpret_cast<const void*>(base), scale)};
    }

    void Scatter(u32* base, const i32* indices, int scale) const {
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(indices));
        _mm512_i32scatter_epi32(base, idx, v, scale);
    }

    static Vec512 MaskedLoad(const u32* p, __mmask16 k) { return {_mm512_mask_load_epi32(_mm512_setzero_si512(), k, reinterpret_cast<const void*>(p))}; }
    void MaskedStore(u32* p, __mmask16 k) const { _mm512_mask_store_epi32(reinterpret_cast<void*>(p), k, v); }
    void CompressStore(u32* p, __mmask16 k) const { _mm512_mask_compressstoreu_epi32(p, k, v); }
};

void simd_add_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Add(vb).Store(dst + i * 4);
    }
}

void simd_add_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Add(vb).Store(dst + i * 8);
    }
}

void simd_add_i64(i64* VOXEL_RESTRICT dst, const i64* VOXEL_RESTRICT a, const i64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i64>::Load(a + i * 4);
        auto vb = Vec256<i64>::Load(b + i * 4);
        va.Add(vb).Store(dst + i * 4);
    }
}

void simd_add_i32(i32* VOXEL_RESTRICT dst, const i32* VOXEL_RESTRICT a, const i32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i32>::Load(a + i * 8);
        auto vb = Vec256<i32>::Load(b + i * 8);
        va.Add(vb).Store(dst + i * 8);
    }
}

void simd_sub_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Sub(vb).Store(dst + i * 4);
    }
}

void simd_sub_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Sub(vb).Store(dst + i * 8);
    }
}

void simd_sub_i64(i64* VOXEL_RESTRICT dst, const i64* VOXEL_RESTRICT a, const i64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i64>::Load(a + i * 4);
        auto vb = Vec256<i64>::Load(b + i * 4);
        va.Sub(vb).Store(dst + i * 4);
    }
}

void simd_sub_i32(i32* VOXEL_RESTRICT dst, const i32* VOXEL_RESTRICT a, const i32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i32>::Load(a + i * 8);
        auto vb = Vec256<i32>::Load(b + i * 8);
        va.Sub(vb).Store(dst + i * 8);
    }
}

void simd_mul_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Mul(vb).Store(dst + i * 4);
    }
}

void simd_mul_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Mul(vb).Store(dst + i * 8);
    }
}

void simd_mul_i64(i64* VOXEL_RESTRICT dst, const i64* VOXEL_RESTRICT a, const i64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i64>::Load(a + i * 4);
        auto vb = Vec256<i64>::Load(b + i * 4);
        va.Mul(vb).Store(dst + i * 4);
    }
}

void simd_mul_i32(i32* VOXEL_RESTRICT dst, const i32* VOXEL_RESTRICT a, const i32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<i32>::Load(a + i * 8);
        auto vb = Vec256<i32>::Load(b + i * 8);
        va.Mul(vb).Store(dst + i * 8);
    }
}

void simd_div_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Div(vb).Store(dst + i * 4);
    }
}

void simd_div_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Div(vb).Store(dst + i * 8);
    }
}

void simd_min_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Min(vb).Store(dst + i * 4);
    }
}

void simd_min_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Min(vb).Store(dst + i * 8);
    }
}

void simd_max_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz n = count / 4;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f64>::Load(a + i * 4);
        auto vb = Vec256<f64>::Load(b + i * 4);
        va.Max(vb).Store(dst + i * 4);
    }
}

void simd_max_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz n = count / 8;
    for (sz i = 0; i < n; ++i) {
        auto va = Vec256<f32>::Load(a + i * 8);
        auto vb = Vec256<f32>::Load(b + i * 8);
        va.Max(vb).Store(dst + i * 8);
    }
}

void simd_filter_gt_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT src, f64 threshold, sz count) {
    sz n = count / 4;
    auto zero = Vec256<f64>::Zero();
    auto thresh = Vec256<f64>::Set1(threshold);
    for (sz i = 0; i < n; ++i) {
        auto v = Vec256<f64>::Load(src + i * 4);
        auto mask = v.CmpGt(thresh);
        zero.Blend(v, mask).Store(dst + i * 4);
    }
}

void simd_filter_gt_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT src, f32 threshold, sz count) {
    sz n = count / 8;
    auto zero = Vec256<f32>::Zero();
    auto thresh = Vec256<f32>::Set1(threshold);
    for (sz i = 0; i < n; ++i) {
        auto v = Vec256<f32>::Load(src + i * 8);
        auto mask = v.CmpGt(thresh);
        zero.Blend(v, mask).Store(dst + i * 8);
    }
}

void simd_filter_gt_i64(i64* VOXEL_RESTRICT dst, const i64* VOXEL_RESTRICT src, i64 threshold, sz count) {
    sz n = count / 4;
    auto zero = Vec256<i64>::Zero();
    auto thresh = Vec256<i64>::Set1(threshold);
    for (sz i = 0; i < n; ++i) {
        auto v = Vec256<i64>::Load(src + i * 4);
        auto mask = v.CmpGt(thresh);
        zero.Blend(v, mask).Store(dst + i * 4);
    }
}

void simd_filter_gt_i32(i32* VOXEL_RESTRICT dst, const i32* VOXEL_RESTRICT src, i32 threshold, sz count) {
    sz n = count / 8;
    auto zero = Vec256<i32>::Zero();
    auto thresh = Vec256<i32>::Set1(threshold);
    for (sz i = 0; i < n; ++i) {
        auto v = Vec256<i32>::Load(src + i * 8);
        auto mask = v.CmpGt(thresh);
        zero.Blend(v, mask).Store(dst + i * 8);
    }
}

f64 simd_sum_f64(const f64* VOXEL_RESTRICT src, sz count) {
    sz n = count / 4;
    auto sum = Vec256<f64>::Zero();
    for (sz i = 0; i < n; ++i) {
        sum = sum.Add(Vec256<f64>::Load(src + i * 4));
    }
    f64 result = sum.HorizontalSum();
    for (sz i = n * 4; i < count; ++i) {
        result += src[i];
    }
    return result;
}

f32 simd_sum_f32(const f32* VOXEL_RESTRICT src, sz count) {
    sz n = count / 8;
    auto sum = Vec256<f32>::Zero();
    for (sz i = 0; i < n; ++i) {
        sum = sum.Add(Vec256<f32>::Load(src + i * 8));
    }
    f32 result = sum.HorizontalSum();
    for (sz i = n * 8; i < count; ++i) {
        result += src[i];
    }
    return result;
}

i64 simd_sum_i64(const i64* VOXEL_RESTRICT src, sz count) {
    sz n = count / 4;
    auto sum = Vec256<i64>::Zero();
    for (sz i = 0; i < n; ++i) {
        sum = sum.Add(Vec256<i64>::Load(src + i * 4));
    }
    i64 result = sum.HorizontalSum();
    for (sz i = n * 4; i < count; ++i) {
        result += src[i];
    }
    return result;
}

i32 simd_sum_i32(const i32* VOXEL_RESTRICT src, sz count) {
    sz n = count / 8;
    auto sum = Vec256<i32>::Zero();
    for (sz i = 0; i < n; ++i) {
        sum = sum.Add(Vec256<i32>::Load(src + i * 8));
    }
    i32 result = sum.HorizontalSum();
    for (sz i = n * 8; i < count; ++i) {
        result += src[i];
    }
    return result;
}

#endif // VOXEL_SIMD_AVX512

} // namespace simd
} // namespace voxel
