#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include <cstring>
#include <algorithm>

namespace voxel {
namespace simd {
namespace scalar {

// ============================================================================
// Primary template (generic, unused directly; specializations follow)
// ============================================================================

template<typename T, sz N>
struct ScalarVec {
    T Lanes[N];
    static constexpr sz kLanes = N;

    ScalarVec() = default;

    static ScalarVec Load(const T* p) { ScalarVec v; for (sz i=0;i<N;++i) v.Lanes[i]=p[i]; return v; }
    static void Store(T* p, ScalarVec v) { for (sz i=0;i<N;++i) p[i]=v.Lanes[i]; }
    static ScalarVec Set1(T x) { ScalarVec v; for (sz i=0;i<N;++i) v.Lanes[i]=x; return v; }
    static ScalarVec Zero() { ScalarVec v; for (sz i=0;i<N;++i) v.Lanes[i]=T{}; return v; }

    ScalarVec Add(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]+o.Lanes[i]; return r; }
    ScalarVec Sub(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]-o.Lanes[i]; return r; }
    ScalarVec Mul(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]*o.Lanes[i]; return r; }
    ScalarVec Div(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]/o.Lanes[i]; return r; }
    ScalarVec Min(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]<o.Lanes[i]?Lanes[i]:o.Lanes[i]; return r; }
    ScalarVec Max(ScalarVec o) const { ScalarVec r; for(sz i=0;i<N;++i) r.Lanes[i]=Lanes[i]>o.Lanes[i]?Lanes[i]:o.Lanes[i]; return r; }
};

// ============================================================================
// ScalarVec<f64, 4> — 4-lane double-precision
// ============================================================================

template <>
struct ScalarVec<f64, 4> {
    f64 Lanes[4];
    static constexpr sz kLanes = 4;

    ScalarVec() = default;

    static ScalarVec Load(const f64* p) { ScalarVec v; v.Lanes[0]=p[0];v.Lanes[1]=p[1];v.Lanes[2]=p[2];v.Lanes[3]=p[3]; return v; }
    static ScalarVec LoadU(const f64* p) { return Load(p); }
    static void Store(f64* p, ScalarVec v) { p[0]=v.Lanes[0];p[1]=v.Lanes[1];p[2]=v.Lanes[2];p[3]=v.Lanes[3]; }
    static void StoreU(f64* p, ScalarVec v) { Store(p, v); }
    static ScalarVec Set1(f64 x) { ScalarVec v; v.Lanes[0]=x;v.Lanes[1]=x;v.Lanes[2]=x;v.Lanes[3]=x; return v; }
    static ScalarVec Zero() { return Set1(0.0); }

    ScalarVec Add(ScalarVec o) const { ScalarVec r; r.Lanes[0]=Lanes[0]+o.Lanes[0];r.Lanes[1]=Lanes[1]+o.Lanes[1];r.Lanes[2]=Lanes[2]+o.Lanes[2];r.Lanes[3]=Lanes[3]+o.Lanes[3]; return r; }
    ScalarVec Sub(ScalarVec o) const { ScalarVec r; r.Lanes[0]=Lanes[0]-o.Lanes[0];r.Lanes[1]=Lanes[1]-o.Lanes[1];r.Lanes[2]=Lanes[2]-o.Lanes[2];r.Lanes[3]=Lanes[3]-o.Lanes[3]; return r; }
    ScalarVec Mul(ScalarVec o) const { ScalarVec r; r.Lanes[0]=Lanes[0]*o.Lanes[0];r.Lanes[1]=Lanes[1]*o.Lanes[1];r.Lanes[2]=Lanes[2]*o.Lanes[2];r.Lanes[3]=Lanes[3]*o.Lanes[3]; return r; }
    ScalarVec Div(ScalarVec o) const { ScalarVec r; r.Lanes[0]=Lanes[0]/o.Lanes[0];r.Lanes[1]=Lanes[1]/o.Lanes[1];r.Lanes[2]=Lanes[2]/o.Lanes[2];r.Lanes[3]=Lanes[3]/o.Lanes[3]; return r; }
    ScalarVec Min(ScalarVec o) const { ScalarVec r; r.Lanes[0]=std::min(Lanes[0],o.Lanes[0]);r.Lanes[1]=std::min(Lanes[1],o.Lanes[1]);r.Lanes[2]=std::min(Lanes[2],o.Lanes[2]);r.Lanes[3]=std::min(Lanes[3],o.Lanes[3]); return r; }
    ScalarVec Max(ScalarVec o) const { ScalarVec r; r.Lanes[0]=std::max(Lanes[0],o.Lanes[0]);r.Lanes[1]=std::max(Lanes[1],o.Lanes[1]);r.Lanes[2]=std::max(Lanes[2],o.Lanes[2]);r.Lanes[3]=std::max(Lanes[3],o.Lanes[3]); return r; }

    ScalarVec Neg() const { ScalarVec r; r.Lanes[0]=-Lanes[0];r.Lanes[1]=-Lanes[1];r.Lanes[2]=-Lanes[2];r.Lanes[3]=-Lanes[3]; return r; }
    ScalarVec Abs() const { ScalarVec r; r.Lanes[0]=std::fabs(Lanes[0]);r.Lanes[1]=std::fabs(Lanes[1]);r.Lanes[2]=std::fabs(Lanes[2]);r.Lanes[3]=std::fabs(Lanes[3]); return r; }
    ScalarVec Sqrt() const { ScalarVec r; r.Lanes[0]=std::sqrt(Lanes[0]);r.Lanes[1]=std::sqrt(Lanes[1]);r.Lanes[2]=std::sqrt(Lanes[2]);r.Lanes[3]=std::sqrt(Lanes[3]); return r; }

    u64 CmpGtMask(ScalarVec o) const {
        u64 mask = 0;
        if (Lanes[0] > o.Lanes[0]) mask |= 1;
        if (Lanes[1] > o.Lanes[1]) mask |= 2;
        if (Lanes[2] > o.Lanes[2]) mask |= 4;
        if (Lanes[3] > o.Lanes[3]) mask |= 8;
        return mask;
    }

    f64 HorizontalSum() const { return Lanes[0]+Lanes[1]+Lanes[2]+Lanes[3]; }
    f64 HorizontalMin() const { return std::min({Lanes[0],Lanes[1],Lanes[2],Lanes[3]}); }
    f64 HorizontalMax() const { return std::max({Lanes[0],Lanes[1],Lanes[2],Lanes[3]}); }
};

// ============================================================================
// Bulk operations on f64 arrays
// ============================================================================

inline void scalar_add_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Add(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] + b[i];
}

inline void scalar_sub_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Sub(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] - b[i];
}

inline void scalar_mul_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Mul(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] * b[i];
}

inline void scalar_div_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Div(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] / b[i];
}

inline void scalar_min_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Min(vb));
    }
    for (; i < count; ++i) dst[i] = std::min(a[i], b[i]);
}

inline void scalar_max_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a,
                           const f64* VOXEL_RESTRICT b, sz count)
{
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> va = ScalarVec<f64, 4>::Load(&a[i]);
        ScalarVec<f64, 4> vb = ScalarVec<f64, 4>::Load(&b[i]);
        ScalarVec<f64, 4>::Store(&dst[i], va.Max(vb));
    }
    for (; i < count; ++i) dst[i] = std::max(a[i], b[i]);
}

inline sz scalar_filter_gt_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT src,
                                f64 threshold, sz count)
{
    sz outIdx = 0;
    sz i = 0;
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> v = ScalarVec<f64, 4>::Load(&src[i]);
        if (v.Lanes[0] > threshold) dst[outIdx++] = v.Lanes[0];
        if (v.Lanes[1] > threshold) dst[outIdx++] = v.Lanes[1];
        if (v.Lanes[2] > threshold) dst[outIdx++] = v.Lanes[2];
        if (v.Lanes[3] > threshold) dst[outIdx++] = v.Lanes[3];
    }
    for (; i < count; ++i)
        if (src[i] > threshold) dst[outIdx++] = src[i];
    return outIdx;
}

inline f64 scalar_sum_f64(const f64* VOXEL_RESTRICT src, sz count)
{
    f64 sum = 0.0;
    sz i = 0;
    ScalarVec<f64, 4> vacc = ScalarVec<f64, 4>::Zero();
    for (; i + 4 <= count; i += 4) {
        ScalarVec<f64, 4> v = ScalarVec<f64, 4>::Load(&src[i]);
        vacc = vacc.Add(v);
    }
    sum = vacc.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

// ============================================================================
// ScalarVec<f32, 8> — 8-lane single-precision
// ============================================================================

template <>
struct ScalarVec<f32, 8> {
    f32 Lanes[8];
    static constexpr sz kLanes = 8;

    ScalarVec() = default;

    static ScalarVec Load(const f32* p) {
        ScalarVec v;
        for (sz i=0;i<8;++i) v.Lanes[i]=p[i];
        return v;
    }
    static void Store(f32* p, ScalarVec v) {
        for (sz i=0;i<8;++i) p[i]=v.Lanes[i];
    }
    static ScalarVec Set1(f32 x) { ScalarVec v; for(sz i=0;i<8;++i) v.Lanes[i]=x; return v; }
    static ScalarVec Zero() { return Set1(0.0f); }

    ScalarVec Add(ScalarVec o) const { ScalarVec r; for(sz i=0;i<8;++i) r.Lanes[i]=Lanes[i]+o.Lanes[i]; return r; }
    ScalarVec Sub(ScalarVec o) const { ScalarVec r; for(sz i=0;i<8;++i) r.Lanes[i]=Lanes[i]-o.Lanes[i]; return r; }
    ScalarVec Mul(ScalarVec o) const { ScalarVec r; for(sz i=0;i<8;++i) r.Lanes[i]=Lanes[i]*o.Lanes[i]; return r; }
    f32 HorizontalSum() const { f32 s=0; for(sz i=0;i<8;++i) s+=Lanes[i]; return s; }
};

inline sz scalar_filter_gt_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT src,
                                f32 threshold, sz count)
{
    sz outIdx = 0;
    for (sz i = 0; i + 8 <= count; i += 8) {
        ScalarVec<f32, 8> v = ScalarVec<f32, 8>::Load(&src[i]);
        for (sz j = 0; j < 8; ++j)
            if (v.Lanes[j] > threshold) dst[outIdx++] = v.Lanes[j];
    }
    for (sz i = (count / 8) * 8; i < count; ++i)
        if (src[i] > threshold) dst[outIdx++] = src[i];
    return outIdx;
}

inline f32 scalar_sum_f32(const f32* VOXEL_RESTRICT src, sz count)
{
    ScalarVec<f32, 8> vacc = ScalarVec<f32, 8>::Zero();
    sz i = 0;
    for (; i + 8 <= count; i += 8)
        vacc = vacc.Add(ScalarVec<f32, 8>::Load(&src[i]));
    f32 sum = vacc.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

// ============================================================================
// ScalarVec<i64, 4> — 4-lane 64-bit integer
// ============================================================================

template <>
struct ScalarVec<i64, 4> {
    i64 Lanes[4];
    static constexpr sz kLanes = 4;

    ScalarVec() = default;

    static ScalarVec Load(const i64* p) { ScalarVec v; for(sz i=0;i<4;++i) v.Lanes[i]=p[i]; return v; }
    static void Store(i64* p, ScalarVec v) { for(sz i=0;i<4;++i) p[i]=v.Lanes[i]; }
    static ScalarVec Set1(i64 x) { ScalarVec v; for(sz i=0;i<4;++i) v.Lanes[i]=x; return v; }
    static ScalarVec Zero() { return Set1(0); }

    ScalarVec Add(ScalarVec o) const { ScalarVec r; for(sz i=0;i<4;++i) r.Lanes[i]=Lanes[i]+o.Lanes[i]; return r; }
    ScalarVec Sub(ScalarVec o) const { ScalarVec r; for(sz i=0;i<4;++i) r.Lanes[i]=Lanes[i]-o.Lanes[i]; return r; }
    i64 HorizontalSum() const { return Lanes[0]+Lanes[1]+Lanes[2]+Lanes[3]; }
};

inline sz scalar_filter_gt_i64(i64* VOXEL_RESTRICT dst, const i64* VOXEL_RESTRICT src,
                                i64 threshold, sz count)
{
    sz outIdx = 0;
    for (sz i = 0; i + 4 <= count; i += 4) {
        ScalarVec<i64, 4> v = ScalarVec<i64, 4>::Load(&src[i]);
        for (sz j = 0; j < 4; ++j)
            if (v.Lanes[j] > threshold) dst[outIdx++] = v.Lanes[j];
    }
    for (sz i = (count / 4) * 4; i < count; ++i)
        if (src[i] > threshold) dst[outIdx++] = src[i];
    return outIdx;
}

inline i64 scalar_sum_i64(const i64* VOXEL_RESTRICT src, sz count)
{
    ScalarVec<i64, 4> vacc = ScalarVec<i64, 4>::Zero();
    sz i = 0;
    for (; i + 4 <= count; i += 4)
        vacc = vacc.Add(ScalarVec<i64, 4>::Load(&src[i]));
    i64 sum = vacc.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

// ============================================================================
// ScalarVec<i32, 8> — 8-lane 32-bit integer
// ============================================================================

template <>
struct ScalarVec<i32, 8> {
    i32 Lanes[8];
    static constexpr sz kLanes = 8;

    ScalarVec() = default;

    static ScalarVec Load(const i32* p) { ScalarVec v; for(sz i=0;i<8;++i) v.Lanes[i]=p[i]; return v; }
    static void Store(i32* p, ScalarVec v) { for(sz i=0;i<8;++i) p[i]=v.Lanes[i]; }
    static ScalarVec Set1(i32 x) { ScalarVec v; for(sz i=0;i<8;++i) v.Lanes[i]=x; return v; }
    static ScalarVec Zero() { return Set1(0); }

    ScalarVec Add(ScalarVec o) const { ScalarVec r; for(sz i=0;i<8;++i) r.Lanes[i]=Lanes[i]+o.Lanes[i]; return r; }
    ScalarVec Sub(ScalarVec o) const { ScalarVec r; for(sz i=0;i<8;++i) r.Lanes[i]=Lanes[i]-o.Lanes[i]; return r; }
    i32 HorizontalSum() const { i32 s=0; for(sz i=0;i<8;++i) s+=Lanes[i]; return s; }
};

inline sz scalar_filter_gt_i32(i32* VOXEL_RESTRICT dst, const i32* VOXEL_RESTRICT src,
                                i32 threshold, sz count)
{
    sz outIdx = 0;
    for (sz i = 0; i + 8 <= count; i += 8) {
        ScalarVec<i32, 8> v = ScalarVec<i32, 8>::Load(&src[i]);
        for (sz j = 0; j < 8; ++j)
            if (v.Lanes[j] > threshold) dst[outIdx++] = v.Lanes[j];
    }
    for (sz i = (count / 8) * 8; i < count; ++i)
        if (src[i] > threshold) dst[outIdx++] = src[i];
    return outIdx;
}

inline i32 scalar_sum_i32(const i32* VOXEL_RESTRICT src, sz count)
{
    ScalarVec<i32, 8> vacc = ScalarVec<i32, 8>::Zero();
    sz i = 0;
    for (; i + 8 <= count; i += 8)
        vacc = vacc.Add(ScalarVec<i32, 8>::Load(&src[i]));
    i32 sum = vacc.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

} // namespace scalar
} // namespace simd
} // namespace voxel
