#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include <cstring>
#include <bit>
#include <cmath>
#include <ostream>
#include <iomanip>
#include <array>

namespace voxel {

// ============================================================================
// RegFile — 64 scalar + 16 vector (256-bit) + 8 mask registers + condition flags
// ============================================================================

class RegFile {
public:
    static constexpr sz kScalarCount  = 64;
    static constexpr sz kVectorCount  = 16;
    static constexpr sz kMaskCount    = 8;
    static constexpr sz kVecWidth     = 32;  // 256 bits = 32 bytes
    static constexpr sz kMaxVecLanes  = kVecWidth / 1; // at 1-byte granularity

    enum class Flag : u32 {
        None     = 0,
        Zero     = 1u << 0,
        Carry    = 1u << 1,
        Sign     = 1u << 2,
        Overflow = 1u << 3,
        Parity   = 1u << 4,
        NaN      = 1u << 5,
        Underflow  = 1u << 6,
        Inexact    = 1u << 7,
        DivByZero  = 1u << 8,
        InvalidOp  = 1u << 9,
    };

    RegFile() : Flags_(0) { Reset(); }

    // ---- Scalar access ----
    u64&       Scalar(sz i)       { return Scalar_[i & 0x3F]; }
    const u64& Scalar(sz i) const { return Scalar_[i & 0x3F]; }
    u64&       operator[](sz i)       { return Scalar_[i & 0x3F]; }
    const u64& operator[](sz i) const { return Scalar_[i & 0x3F]; }

    template<typename T> requires (sizeof(T) <= 8)
    T ScalarAs(sz i) const {
        u64 v = Scalar_[i & 0x3F];
        if constexpr (std::is_floating_point_v<T>)
            return std::bit_cast<T>(v);
        else
            return static_cast<T>(v);
    }

    template<typename T> requires (sizeof(T) <= 8)
    void SetScalarT(sz i, T value) {
        if constexpr (std::is_floating_point_v<T>)
            Scalar_[i & 0x3F] = std::bit_cast<u64>(value);
        else
            Scalar_[i & 0x3F] = static_cast<u64>(value);
    }

    // ---- Vector access ----
    template<typename T> requires (sizeof(T) <= kVecWidth)
    T* VecLanes(sz i) {
        return reinterpret_cast<T*>(&Vector_[i & 0xF][0]);
    }

    template<typename T> requires (sizeof(T) <= kVecWidth)
    const T* VecLanes(sz i) const {
        return reinterpret_cast<const T*>(&Vector_[i & 0xF][0]);
    }

    void ClearVector(sz i) { std::memset(&Vector_[i & 0xF][0], 0, kVecWidth); }

    u8*       VecRaw(sz i)       { return &Vector_[i & 0xF][0]; }
    const u8* VecRaw(sz i) const { return &Vector_[i & 0xF][0]; }

    // ---- Mask access (for predicated vector operations) ----
    u32&       Mask(sz i)       { return Masks_[i & 0x7]; }
    const u32& Mask(sz i) const { return Masks_[i & 0x7]; }

    // ---- Temporary/scratch vector register for intermediate results ----
    u8*       ScratchVec()       { return ScratchVec_; }
    const u8* ScratchVec() const { return ScratchVec_; }

    // ---- Flag manipulation ----
    void SetFlag(Flag f)     { Flags_ |= static_cast<u32>(f); }
    void ClearFlag(Flag f)   { Flags_ &= ~static_cast<u32>(f); }
    void ClearAllFlags()     { Flags_ = 0; }
    bool Test(Flag f)   const { return (Flags_ & static_cast<u32>(f)) != 0; }
    u32  RawFlags()     const { return Flags_; }

    void UpdateArithFlags(i64 result) {
        ClearFlag(Flag::Zero);  ClearFlag(Flag::Sign);
        ClearFlag(Flag::Carry); ClearFlag(Flag::Overflow);
        if (result == 0)         SetFlag(Flag::Zero);
        if (result < 0)          SetFlag(Flag::Sign);
    }

    void UpdateFloatFlags(f64 result) {
        ClearFlag(Flag::NaN);     ClearFlag(Flag::Underflow);
        ClearFlag(Flag::Overflow); ClearFlag(Flag::Inexact);
        ClearFlag(Flag::DivByZero); ClearFlag(Flag::InvalidOp);
        if (result == 0.0)  { SetFlag(Flag::Zero); ClearFlag(Flag::Sign); }
        if (result < 0.0)   SetFlag(Flag::Sign);
        if (std::isnan(result)) SetFlag(Flag::NaN);
    }

    // ---- Full reset ----
    void Reset() {
        std::memset(Scalar_, 0, sizeof(Scalar_));
        std::memset(Vector_, 0, sizeof(Vector_));
        std::memset(Masks_,  0, sizeof(Masks_));
        std::memset(ScratchVec_, 0, kVecWidth);
        Flags_ = 0;
    }

    // ---- Telemetry: hex dump of entire register file ----
    void DumpScalars(std::ostream& os, sz maxRegs = 16) const {
        os << "=== Scalar Registers (first " << maxRegs << ") ===\n";
        for (sz i = 0; i < maxRegs && i < kScalarCount; ++i) {
            os << "  R" << std::setw(2) << i << ": 0x"
               << std::hex << std::setfill('0') << std::setw(16) << Scalar_[i]
               << "  (" << std::dec << std::setfill(' ') << std::setw(20) << static_cast<i64>(Scalar_[i])
               << " as i64, " << std::bit_cast<f64>(Scalar_[i]) << " as f64)\n";
        }
    }

    void DumpVectors(std::ostream& os, sz vecIdx = 0, sz maxLanes = 8) const {
        os << "=== Vector Register V" << vecIdx << " (256-bit) ===\n";
        os << "  As u64: ";
        const u64* p = reinterpret_cast<const u64*>(&Vector_[vecIdx & 0xF][0]);
        for (sz i = 0; i < 4; ++i) os << std::hex << "0x" << p[i] << " ";
        os << "\n  As f64: ";
        const f64* pf = reinterpret_cast<const f64*>(&Vector_[vecIdx & 0xF][0]);
        for (sz i = 0; i < 4; ++i) os << pf[i] << " ";
        os << "\n  As i64: ";
        const i64* pi = reinterpret_cast<const i64*>(&Vector_[vecIdx & 0xF][0]);
        for (sz i = 0; i < 4; ++i) os << pi[i] << " ";
        os << std::dec << "\n";
    }

    void DumpAll(std::ostream& os) const {
        DumpScalars(os, 16);
        for (sz v = 0; v < 4; ++v) DumpVectors(os, v);
        os << "=== Flags ===\n";
        os << "  Zero:" << Test(Flag::Zero) << " Carry:" << Test(Flag::Carry)
           << " Sign:" << Test(Flag::Sign) << " Overflow:" << Test(Flag::Overflow)
           << " NaN:" << Test(Flag::NaN) << "\n";
        os << "=== Mask Registers ===\n";
        for (sz i = 0; i < kMaskCount; ++i)
            os << "  M" << i << ": 0x" << std::hex << std::setw(8) << Masks_[i] << std::dec << "\n";
    }

private:
    alignas(64) u64 Scalar_[kScalarCount];
    alignas(32) u8  Vector_[kVectorCount][kVecWidth];
    alignas(32) u32 Masks_[kMaskCount];
    alignas(32) u8  ScratchVec_[kVecWidth];
    u32 Flags_;
};

// ============================================================================
// Vector lane count for a given element type
// ============================================================================

template<typename T>
inline constexpr sz kVectorLanes = RegFile::kVecWidth / sizeof(T);

static_assert(kVectorLanes<f64> == 4, "f64 vector must have 4 lanes");
static_assert(kVectorLanes<f32> == 8, "f32 vector must have 8 lanes");
static_assert(kVectorLanes<i64> == 4, "i64 vector must have 4 lanes");
static_assert(kVectorLanes<i32> == 8, "i32 vector must have 8 lanes");
static_assert(kVectorLanes<u64> == 4, "u64 vector must have 4 lanes");
static_assert(kVectorLanes<u32> == 8, "u32 vector must have 8 lanes");

} // namespace voxel
