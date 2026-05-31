#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

namespace voxel {
namespace simd {

template <typename T>
struct alignas(16) Vec128;

template <>
struct alignas(16) Vec128<f64> {
    static constexpr sz kLanes = 2;
    using Native = float64x2_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const f64* p) { return vld1q_f64(p); }
    static void Store(f64* p, Vec128 v) { vst1q_f64(p, v.V); }
    static Vec128 LoadU(const f64* p) { return vld1q_f64(p); }
    static void StoreU(f64* p, Vec128 v) { vst1q_f64(p, v.V); }
    static Vec128 Set1(f64 x) { return vdupq_n_f64(x); }
    static Vec128 Zero() { return vdupq_n_f64(0.0); }

    Vec128 Add(Vec128 o) const { return vaddq_f64(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_f64(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_f64(V, o.V); }
    Vec128 Div(Vec128 o) const { return vdivq_f64(V, o.V); }
    Vec128 Min(Vec128 o) const { return vminq_f64(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_f64(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_f64_u64(vceqq_f64(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_f64_u64(vcltq_f64(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_f64_u64(vcleq_f64(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_f64_u64(vcgtq_f64(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_f64_u64(vcgeq_f64(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_f64(vreinterpretq_u64_f64(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(V), vreinterpretq_u64_f64(o.V))); }
    Vec128 Or(Vec128 o) const { return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(V), vreinterpretq_u64_f64(o.V))); }
    Vec128 Xor(Vec128 o) const { return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(V), vreinterpretq_u64_f64(o.V))); }
    Vec128 AndNot(Vec128 o) const { return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(V), vreinterpretq_u64_f64(o.V))); }

    Vec128 FMA(Vec128 a, Vec128 b) const { return vfmaq_f64(V, a.V, b.V); }
    Vec128 Sqrt() const { return vsqrtq_f64(V); }
    Vec128 Abs() const { return vabsq_f64(V); }
    Vec128 Neg() const { return vnegq_f64(V); }

    f64 HorizontalSum() const { return vaddvq_f64(V); }
    f64 HorizontalMin() const { return vminvq_f64(V); }
    f64 HorizontalMax() const { return vmaxvq_f64(V); }
};

template <>
struct alignas(16) Vec128<f32> {
    static constexpr sz kLanes = 4;
    using Native = float32x4_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const f32* p) { return vld1q_f32(p); }
    static void Store(f32* p, Vec128 v) { vst1q_f32(p, v.V); }
    static Vec128 LoadU(const f32* p) { return vld1q_f32(p); }
    static void StoreU(f32* p, Vec128 v) { vst1q_f32(p, v.V); }
    static Vec128 Set1(f32 x) { return vdupq_n_f32(x); }
    static Vec128 Zero() { return vdupq_n_f32(0.0f); }

    Vec128 Add(Vec128 o) const { return vaddq_f32(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_f32(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_f32(V, o.V); }
    Vec128 Div(Vec128 o) const { return vdivq_f32(V, o.V); }
    Vec128 Min(Vec128 o) const { return vminq_f32(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_f32(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_f32_u32(vceqq_f32(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_f32_u32(vcltq_f32(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_f32_u32(vcleq_f32(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_f32_u32(vcgtq_f32(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_f32_u32(vcgeq_f32(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_f32(vreinterpretq_u32_f32(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(V), vreinterpretq_u32_f32(o.V))); }
    Vec128 Or(Vec128 o) const { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(V), vreinterpretq_u32_f32(o.V))); }
    Vec128 Xor(Vec128 o) const { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(V), vreinterpretq_u32_f32(o.V))); }
    Vec128 AndNot(Vec128 o) const { return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(V), vreinterpretq_u32_f32(o.V))); }

    Vec128 FMA(Vec128 a, Vec128 b) const { return vfmaq_f32(V, a.V, b.V); }
    Vec128 Sqrt() const { return vsqrtq_f32(V); }
    Vec128 Abs() const { return vabsq_f32(V); }
    Vec128 Neg() const { return vnegq_f32(V); }

    f32 HorizontalSum() const { return vaddvq_f32(V); }
    f32 HorizontalMin() const { return vminvq_f32(V); }
    f32 HorizontalMax() const { return vmaxvq_f32(V); }
};

template <>
struct alignas(16) Vec128<i64> {
    static constexpr sz kLanes = 2;
    using Native = int64x2_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const i64* p) { return vld1q_s64(p); }
    static void Store(i64* p, Vec128 v) { vst1q_s64(p, v.V); }
    static Vec128 LoadU(const i64* p) { return vld1q_s64(p); }
    static void StoreU(i64* p, Vec128 v) { vst1q_s64(p, v.V); }
    static Vec128 Set1(i64 x) { return vdupq_n_s64(x); }
    static Vec128 Zero() { return vdupq_n_s64(0); }

    Vec128 Add(Vec128 o) const { return vaddq_s64(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_s64(V, o.V); }
    Vec128 Mul(Vec128 o) const {
        int64x2_t r;
        r = vsetq_lane_s64(vgetq_lane_s64(V, 0) * vgetq_lane_s64(o.V, 0), r, 0);
        r = vsetq_lane_s64(vgetq_lane_s64(V, 1) * vgetq_lane_s64(o.V, 1), r, 1);
        return r;
    }
    Vec128 Div(Vec128 o) const {
        int64x2_t r;
        r = vsetq_lane_s64(vgetq_lane_s64(V, 0) / vgetq_lane_s64(o.V, 0), r, 0);
        r = vsetq_lane_s64(vgetq_lane_s64(V, 1) / vgetq_lane_s64(o.V, 1), r, 1);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_s64(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_s64(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_s64_u64(vceqq_s64(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_s64_u64(vmvnq_u64(vceqq_s64(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_s64_u64(vcltq_s64(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_s64_u64(vcleq_s64(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_s64_u64(vcgtq_s64(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_s64_u64(vcgeq_s64(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_s64(vreinterpretq_u64_s64(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_s64(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_s64(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_s64(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_s64(V, o.V); }

    Vec128 Neg() const { return vnegq_s64(V); }
    Vec128 Abs() const { return vabsq_s64(V); }

    i64 HorizontalSum() const { return vaddvq_s64(V); }
    i64 HorizontalMin() const { return vminvq_s64(V); }
    i64 HorizontalMax() const { return vmaxvq_s64(V); }
};

template <>
struct alignas(16) Vec128<u64> {
    static constexpr sz kLanes = 2;
    using Native = uint64x2_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const u64* p) { return vld1q_u64(p); }
    static void Store(u64* p, Vec128 v) { vst1q_u64(p, v.V); }
    static Vec128 LoadU(const u64* p) { return vld1q_u64(p); }
    static void StoreU(u64* p, Vec128 v) { vst1q_u64(p, v.V); }
    static Vec128 Set1(u64 x) { return vdupq_n_u64(x); }
    static Vec128 Zero() { return vdupq_n_u64(0); }

    Vec128 Add(Vec128 o) const { return vaddq_u64(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_u64(V, o.V); }
    Vec128 Mul(Vec128 o) const {
        uint64x2_t r;
        r = vsetq_lane_u64(vgetq_lane_u64(V, 0) * vgetq_lane_u64(o.V, 0), r, 0);
        r = vsetq_lane_u64(vgetq_lane_u64(V, 1) * vgetq_lane_u64(o.V, 1), r, 1);
        return r;
    }
    Vec128 Div(Vec128 o) const {
        uint64x2_t r;
        r = vsetq_lane_u64(vgetq_lane_u64(V, 0) / vgetq_lane_u64(o.V, 0), r, 0);
        r = vsetq_lane_u64(vgetq_lane_u64(V, 1) / vgetq_lane_u64(o.V, 1), r, 1);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_u64(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_u64(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vceqq_u64(V, o.V); }
    Vec128 CmpNe(Vec128 o) const { return vmvnq_u64(vceqq_u64(V, o.V)); }
    Vec128 CmpLt(Vec128 o) const { return vcltq_u64(V, o.V); }
    Vec128 CmpLe(Vec128 o) const { return vcleq_u64(V, o.V); }
    Vec128 CmpGt(Vec128 o) const { return vcgtq_u64(V, o.V); }
    Vec128 CmpGe(Vec128 o) const { return vcgeq_u64(V, o.V); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_u64(V, a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_u64(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_u64(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_u64(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_u64(V, o.V); }

    u64 HorizontalSum() const { return vaddvq_u64(V); }
    u64 HorizontalMin() const { return vminvq_u64(V); }
    u64 HorizontalMax() const { return vmaxvq_u64(V); }
};

template <>
struct alignas(16) Vec128<i32> {
    static constexpr sz kLanes = 4;
    using Native = int32x4_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const i32* p) { return vld1q_s32(p); }
    static void Store(i32* p, Vec128 v) { vst1q_s32(p, v.V); }
    static Vec128 LoadU(const i32* p) { return vld1q_s32(p); }
    static void StoreU(i32* p, Vec128 v) { vst1q_s32(p, v.V); }
    static Vec128 Set1(i32 x) { return vdupq_n_s32(x); }
    static Vec128 Zero() { return vdupq_n_s32(0); }

    Vec128 Add(Vec128 o) const { return vaddq_s32(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_s32(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_s32(V, o.V); }
    Vec128 Div(Vec128 o) const {
        int32x4_t r;
        r = vsetq_lane_s32(vgetq_lane_s32(V, 0) / vgetq_lane_s32(o.V, 0), r, 0);
        r = vsetq_lane_s32(vgetq_lane_s32(V, 1) / vgetq_lane_s32(o.V, 1), r, 1);
        r = vsetq_lane_s32(vgetq_lane_s32(V, 2) / vgetq_lane_s32(o.V, 2), r, 2);
        r = vsetq_lane_s32(vgetq_lane_s32(V, 3) / vgetq_lane_s32(o.V, 3), r, 3);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_s32(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_s32(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_s32_u32(vceqq_s32(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_s32_u32(vcltq_s32(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_s32_u32(vcleq_s32(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_s32_u32(vcgtq_s32(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_s32_u32(vcgeq_s32(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_s32(vreinterpretq_u32_s32(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_s32(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_s32(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_s32(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_s32(V, o.V); }

    Vec128 Neg() const { return vnegq_s32(V); }
    Vec128 Abs() const { return vabsq_s32(V); }

    i32 HorizontalSum() const { return vaddvq_s32(V); }
    i32 HorizontalMin() const { return vminvq_s32(V); }
    i32 HorizontalMax() const { return vmaxvq_s32(V); }
};

template <>
struct alignas(16) Vec128<u32> {
    static constexpr sz kLanes = 4;
    using Native = uint32x4_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const u32* p) { return vld1q_u32(p); }
    static void Store(u32* p, Vec128 v) { vst1q_u32(p, v.V); }
    static Vec128 LoadU(const u32* p) { return vld1q_u32(p); }
    static void StoreU(u32* p, Vec128 v) { vst1q_u32(p, v.V); }
    static Vec128 Set1(u32 x) { return vdupq_n_u32(x); }
    static Vec128 Zero() { return vdupq_n_u32(0); }

    Vec128 Add(Vec128 o) const { return vaddq_u32(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_u32(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_u32(V, o.V); }
    Vec128 Div(Vec128 o) const {
        uint32x4_t r;
        r = vsetq_lane_u32(vgetq_lane_u32(V, 0) / vgetq_lane_u32(o.V, 0), r, 0);
        r = vsetq_lane_u32(vgetq_lane_u32(V, 1) / vgetq_lane_u32(o.V, 1), r, 1);
        r = vsetq_lane_u32(vgetq_lane_u32(V, 2) / vgetq_lane_u32(o.V, 2), r, 2);
        r = vsetq_lane_u32(vgetq_lane_u32(V, 3) / vgetq_lane_u32(o.V, 3), r, 3);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_u32(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_u32(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vceqq_u32(V, o.V); }
    Vec128 CmpNe(Vec128 o) const { return vmvnq_u32(vceqq_u32(V, o.V)); }
    Vec128 CmpLt(Vec128 o) const { return vcltq_u32(V, o.V); }
    Vec128 CmpLe(Vec128 o) const { return vcleq_u32(V, o.V); }
    Vec128 CmpGt(Vec128 o) const { return vcgtq_u32(V, o.V); }
    Vec128 CmpGe(Vec128 o) const { return vcgeq_u32(V, o.V); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_u32(V, a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_u32(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_u32(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_u32(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_u32(V, o.V); }

    u32 HorizontalSum() const { return vaddvq_u32(V); }
    u32 HorizontalMin() const { return vminvq_u32(V); }
    u32 HorizontalMax() const { return vmaxvq_u32(V); }
};

template <>
struct alignas(16) Vec128<i16> {
    static constexpr sz kLanes = 8;
    using Native = int16x8_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const i16* p) { return vld1q_s16(p); }
    static void Store(i16* p, Vec128 v) { vst1q_s16(p, v.V); }
    static Vec128 LoadU(const i16* p) { return vld1q_s16(p); }
    static void StoreU(i16* p, Vec128 v) { vst1q_s16(p, v.V); }
    static Vec128 Set1(i16 x) { return vdupq_n_s16(x); }
    static Vec128 Zero() { return vdupq_n_s16(0); }

    Vec128 Add(Vec128 o) const { return vaddq_s16(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_s16(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_s16(V, o.V); }
    Vec128 Div(Vec128 o) const {
        int16x8_t r;
        r = vsetq_lane_s16(vgetq_lane_s16(V, 0) / vgetq_lane_s16(o.V, 0), r, 0);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 1) / vgetq_lane_s16(o.V, 1), r, 1);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 2) / vgetq_lane_s16(o.V, 2), r, 2);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 3) / vgetq_lane_s16(o.V, 3), r, 3);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 4) / vgetq_lane_s16(o.V, 4), r, 4);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 5) / vgetq_lane_s16(o.V, 5), r, 5);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 6) / vgetq_lane_s16(o.V, 6), r, 6);
        r = vsetq_lane_s16(vgetq_lane_s16(V, 7) / vgetq_lane_s16(o.V, 7), r, 7);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_s16(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_s16(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_s16_u16(vceqq_s16(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_s16_u16(vcltq_s16(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_s16_u16(vcleq_s16(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_s16_u16(vcgtq_s16(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_s16_u16(vcgeq_s16(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_s16(vreinterpretq_u16_s16(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_s16(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_s16(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_s16(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_s16(V, o.V); }

    Vec128 Neg() const { return vnegq_s16(V); }
    Vec128 Abs() const { return vabsq_s16(V); }

    i16 HorizontalSum() const { return vaddvq_s16(V); }
    i16 HorizontalMin() const { return vminvq_s16(V); }
    i16 HorizontalMax() const { return vmaxvq_s16(V); }
};

template <>
struct alignas(16) Vec128<u16> {
    static constexpr sz kLanes = 8;
    using Native = uint16x8_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const u16* p) { return vld1q_u16(p); }
    static void Store(u16* p, Vec128 v) { vst1q_u16(p, v.V); }
    static Vec128 LoadU(const u16* p) { return vld1q_u16(p); }
    static void StoreU(u16* p, Vec128 v) { vst1q_u16(p, v.V); }
    static Vec128 Set1(u16 x) { return vdupq_n_u16(x); }
    static Vec128 Zero() { return vdupq_n_u16(0); }

    Vec128 Add(Vec128 o) const { return vaddq_u16(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_u16(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_u16(V, o.V); }
    Vec128 Div(Vec128 o) const {
        uint16x8_t r;
        r = vsetq_lane_u16(vgetq_lane_u16(V, 0) / vgetq_lane_u16(o.V, 0), r, 0);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 1) / vgetq_lane_u16(o.V, 1), r, 1);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 2) / vgetq_lane_u16(o.V, 2), r, 2);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 3) / vgetq_lane_u16(o.V, 3), r, 3);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 4) / vgetq_lane_u16(o.V, 4), r, 4);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 5) / vgetq_lane_u16(o.V, 5), r, 5);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 6) / vgetq_lane_u16(o.V, 6), r, 6);
        r = vsetq_lane_u16(vgetq_lane_u16(V, 7) / vgetq_lane_u16(o.V, 7), r, 7);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_u16(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_u16(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vceqq_u16(V, o.V); }
    Vec128 CmpNe(Vec128 o) const { return vmvnq_u16(vceqq_u16(V, o.V)); }
    Vec128 CmpLt(Vec128 o) const { return vcltq_u16(V, o.V); }
    Vec128 CmpLe(Vec128 o) const { return vcleq_u16(V, o.V); }
    Vec128 CmpGt(Vec128 o) const { return vcgtq_u16(V, o.V); }
    Vec128 CmpGe(Vec128 o) const { return vcgeq_u16(V, o.V); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_u16(V, a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_u16(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_u16(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_u16(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_u16(V, o.V); }

    u16 HorizontalSum() const { return vaddvq_u16(V); }
    u16 HorizontalMin() const { return vminvq_u16(V); }
    u16 HorizontalMax() const { return vmaxvq_u16(V); }
};

template <>
struct alignas(16) Vec128<i8> {
    static constexpr sz kLanes = 16;
    using Native = int8x16_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const i8* p) { return vld1q_s8(p); }
    static void Store(i8* p, Vec128 v) { vst1q_s8(p, v.V); }
    static Vec128 LoadU(const i8* p) { return vld1q_s8(p); }
    static void StoreU(i8* p, Vec128 v) { vst1q_s8(p, v.V); }
    static Vec128 Set1(i8 x) { return vdupq_n_s8(x); }
    static Vec128 Zero() { return vdupq_n_s8(0); }

    Vec128 Add(Vec128 o) const { return vaddq_s8(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_s8(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_s8(V, o.V); }
    Vec128 Div(Vec128 o) const {
        int8x16_t r;
        r = vsetq_lane_s8(vgetq_lane_s8(V,  0) / vgetq_lane_s8(o.V,  0), r,  0);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  1) / vgetq_lane_s8(o.V,  1), r,  1);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  2) / vgetq_lane_s8(o.V,  2), r,  2);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  3) / vgetq_lane_s8(o.V,  3), r,  3);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  4) / vgetq_lane_s8(o.V,  4), r,  4);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  5) / vgetq_lane_s8(o.V,  5), r,  5);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  6) / vgetq_lane_s8(o.V,  6), r,  6);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  7) / vgetq_lane_s8(o.V,  7), r,  7);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  8) / vgetq_lane_s8(o.V,  8), r,  8);
        r = vsetq_lane_s8(vgetq_lane_s8(V,  9) / vgetq_lane_s8(o.V,  9), r,  9);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 10) / vgetq_lane_s8(o.V, 10), r, 10);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 11) / vgetq_lane_s8(o.V, 11), r, 11);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 12) / vgetq_lane_s8(o.V, 12), r, 12);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 13) / vgetq_lane_s8(o.V, 13), r, 13);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 14) / vgetq_lane_s8(o.V, 14), r, 14);
        r = vsetq_lane_s8(vgetq_lane_s8(V, 15) / vgetq_lane_s8(o.V, 15), r, 15);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_s8(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_s8(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vreinterpretq_s8_u8(vceqq_s8(V, o.V)); }
    Vec128 CmpNe(Vec128 o) const { return vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(V, o.V))); }
    Vec128 CmpLt(Vec128 o) const { return vreinterpretq_s8_u8(vcltq_s8(V, o.V)); }
    Vec128 CmpLe(Vec128 o) const { return vreinterpretq_s8_u8(vcleq_s8(V, o.V)); }
    Vec128 CmpGt(Vec128 o) const { return vreinterpretq_s8_u8(vcgtq_s8(V, o.V)); }
    Vec128 CmpGe(Vec128 o) const { return vreinterpretq_s8_u8(vcgeq_s8(V, o.V)); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_s8(vreinterpretq_u8_s8(V), a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_s8(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_s8(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_s8(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_s8(V, o.V); }

    Vec128 Neg() const { return vnegq_s8(V); }
    Vec128 Abs() const { return vabsq_s8(V); }

    i8 HorizontalSum() const { return vaddvq_s8(V); }
    i8 HorizontalMin() const { return vminvq_s8(V); }
    i8 HorizontalMax() const { return vmaxvq_s8(V); }
};

template <>
struct alignas(16) Vec128<u8> {
    static constexpr sz kLanes = 16;
    using Native = uint8x16_t;
    Native V;

    Vec128() = default;
    Vec128(Native v) : V(v) {}

    static Vec128 Load(const u8* p) { return vld1q_u8(p); }
    static void Store(u8* p, Vec128 v) { vst1q_u8(p, v.V); }
    static Vec128 LoadU(const u8* p) { return vld1q_u8(p); }
    static void StoreU(u8* p, Vec128 v) { vst1q_u8(p, v.V); }
    static Vec128 Set1(u8 x) { return vdupq_n_u8(x); }
    static Vec128 Zero() { return vdupq_n_u8(0); }

    Vec128 Add(Vec128 o) const { return vaddq_u8(V, o.V); }
    Vec128 Sub(Vec128 o) const { return vsubq_u8(V, o.V); }
    Vec128 Mul(Vec128 o) const { return vmulq_u8(V, o.V); }
    Vec128 Div(Vec128 o) const {
        uint8x16_t r;
        r = vsetq_lane_u8(vgetq_lane_u8(V,  0) / vgetq_lane_u8(o.V,  0), r,  0);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  1) / vgetq_lane_u8(o.V,  1), r,  1);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  2) / vgetq_lane_u8(o.V,  2), r,  2);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  3) / vgetq_lane_u8(o.V,  3), r,  3);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  4) / vgetq_lane_u8(o.V,  4), r,  4);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  5) / vgetq_lane_u8(o.V,  5), r,  5);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  6) / vgetq_lane_u8(o.V,  6), r,  6);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  7) / vgetq_lane_u8(o.V,  7), r,  7);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  8) / vgetq_lane_u8(o.V,  8), r,  8);
        r = vsetq_lane_u8(vgetq_lane_u8(V,  9) / vgetq_lane_u8(o.V,  9), r,  9);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 10) / vgetq_lane_u8(o.V, 10), r, 10);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 11) / vgetq_lane_u8(o.V, 11), r, 11);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 12) / vgetq_lane_u8(o.V, 12), r, 12);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 13) / vgetq_lane_u8(o.V, 13), r, 13);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 14) / vgetq_lane_u8(o.V, 14), r, 14);
        r = vsetq_lane_u8(vgetq_lane_u8(V, 15) / vgetq_lane_u8(o.V, 15), r, 15);
        return r;
    }
    Vec128 Min(Vec128 o) const { return vminq_u8(V, o.V); }
    Vec128 Max(Vec128 o) const { return vmaxq_u8(V, o.V); }

    Vec128 CmpEq(Vec128 o) const { return vceqq_u8(V, o.V); }
    Vec128 CmpNe(Vec128 o) const { return vmvnq_u8(vceqq_u8(V, o.V)); }
    Vec128 CmpLt(Vec128 o) const { return vcltq_u8(V, o.V); }
    Vec128 CmpLe(Vec128 o) const { return vcleq_u8(V, o.V); }
    Vec128 CmpGt(Vec128 o) const { return vcgtq_u8(V, o.V); }
    Vec128 CmpGe(Vec128 o) const { return vcgeq_u8(V, o.V); }

    Vec128 Blend(Vec128 a, Vec128 b) const { return vbslq_u8(V, a.V, b.V); }
    Vec128 And(Vec128 o) const { return vandq_u8(V, o.V); }
    Vec128 Or(Vec128 o) const { return vorrq_u8(V, o.V); }
    Vec128 Xor(Vec128 o) const { return veorq_u8(V, o.V); }
    Vec128 AndNot(Vec128 o) const { return vbicq_u8(V, o.V); }

    u8 HorizontalSum() const { return vaddvq_u8(V); }
    u8 HorizontalMin() const { return vminvq_u8(V); }
    u8 HorizontalMax() const { return vmaxvq_u8(V); }
};

VOXEL_ALWAYS_INLINE void simd_add_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT a, const f64* VOXEL_RESTRICT b, sz count) {
    sz i = 0;
    for (; i + 1 < count; i += 2) {
        Vec128<f64> va = Vec128<f64>::Load(&a[i]);
        Vec128<f64> vb = Vec128<f64>::Load(&b[i]);
        Vec128<f64>::Store(&dst[i], va.Add(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] + b[i];
}

VOXEL_ALWAYS_INLINE void simd_add_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT a, const f32* VOXEL_RESTRICT b, sz count) {
    sz i = 0;
    for (; i + 3 < count; i += 4) {
        Vec128<f32> va = Vec128<f32>::Load(&a[i]);
        Vec128<f32> vb = Vec128<f32>::Load(&b[i]);
        Vec128<f32>::Store(&dst[i], va.Add(vb));
    }
    for (; i < count; ++i) dst[i] = a[i] + b[i];
}

VOXEL_ALWAYS_INLINE void simd_filter_gt_f64(f64* VOXEL_RESTRICT dst, const f64* VOXEL_RESTRICT src, f64 threshold, sz count) {
    Vec128<f64> vthresh = Vec128<f64>::Set1(threshold);
    Vec128<f64> vzero = Vec128<f64>::Zero();
    sz i = 0;
    for (; i + 1 < count; i += 2) {
        Vec128<f64> vsrc = Vec128<f64>::Load(&src[i]);
        Vec128<f64> mask = vsrc.CmpGt(vthresh);
        Vec128<f64>::Store(&dst[i], mask.Blend(vsrc, vzero));
    }
    for (; i < count; ++i) dst[i] = (src[i] > threshold) ? src[i] : 0.0;
}

VOXEL_ALWAYS_INLINE void simd_filter_gt_f32(f32* VOXEL_RESTRICT dst, const f32* VOXEL_RESTRICT src, f32 threshold, sz count) {
    Vec128<f32> vthresh = Vec128<f32>::Set1(threshold);
    Vec128<f32> vzero = Vec128<f32>::Zero();
    sz i = 0;
    for (; i + 3 < count; i += 4) {
        Vec128<f32> vsrc = Vec128<f32>::Load(&src[i]);
        Vec128<f32> mask = vsrc.CmpGt(vthresh);
        Vec128<f32>::Store(&dst[i], mask.Blend(vsrc, vzero));
    }
    for (; i < count; ++i) dst[i] = (src[i] > threshold) ? src[i] : 0.0f;
}

VOXEL_ALWAYS_INLINE f64 simd_sum_f64(const f64* VOXEL_RESTRICT src, sz count) {
    Vec128<f64> vsum = Vec128<f64>::Zero();
    sz i = 0;
    for (; i + 1 < count; i += 2) vsum = vsum.Add(Vec128<f64>::Load(&src[i]));
    f64 sum = vsum.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

VOXEL_ALWAYS_INLINE f32 simd_sum_f32(const f32* VOXEL_RESTRICT src, sz count) {
    Vec128<f32> vsum = Vec128<f32>::Zero();
    sz i = 0;
    for (; i + 3 < count; i += 4) vsum = vsum.Add(Vec128<f32>::Load(&src[i]));
    f32 sum = vsum.HorizontalSum();
    for (; i < count; ++i) sum += src[i];
    return sum;
}

} // namespace simd
} // namespace voxel
