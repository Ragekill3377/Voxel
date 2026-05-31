#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/registers.hpp"
#include "voxel/data/segment.hpp"
#include <vector>
#include <cstring>
#include <algorithm>
#include <bit>
#include <functional>
#include <numeric>

namespace voxel {

// ============================================================================
// NullBitmap — compact bitmask for nullable columns
// ============================================================================
//
// Bit i is 1 = valid (not null), 0 = null.  Bitmap is initialized to all 1s
// (all valid) so that columns without explicit nulls default to non-nullable.
// ============================================================================

class NullBitmap {
public:
    std::vector<u64> Bits;
    sz               RowCount = 0;

    NullBitmap() = default;

    explicit NullBitmap(sz count)
        : RowCount(count)
    {
        sz wordCount = (count + 63) / 64;
        Bits.assign(wordCount, ~0ULL);
        if (count > 0) {
            sz trailing = count % 64;
            if (trailing != 0) {
                Bits.back() &= (1ULL << trailing) - 1;
            }
        }
    }

    bool IsNull(sz row) const {
        sz wordIdx = row >> 6;
        sz bitIdx  = row & 63;
        return ((Bits[wordIdx] >> bitIdx) & 1ULL) == 0;
    }

    bool IsValid(sz row) const {
        return !IsNull(row);
    }

    void SetNull(sz row) {
        sz wordIdx = row >> 6;
        sz bitIdx  = row & 63;
        Bits[wordIdx] &= ~(1ULL << bitIdx);
    }

    void SetValid(sz row) {
        sz wordIdx = row >> 6;
        sz bitIdx  = row & 63;
        Bits[wordIdx] |= (1ULL << bitIdx);
    }

    void SetRow(sz row, bool valid) {
        if (valid) SetValid(row); else SetNull(row);
    }

    void SetAllValid() {
        for (u64& w : Bits) w = ~0ULL;
        if (RowCount > 0) {
            sz trailing = RowCount % 64;
            if (trailing != 0) {
                Bits.back() &= (1ULL << trailing) - 1;
            }
        }
    }

    void SetAllNull() {
        for (u64& w : Bits) w = 0;
    }

    sz NullCount() const {
        sz count = 0;
        for (sz i = 0; i < Bits.size(); ++i) {
            u64 word = Bits[i];
            count += VOXEL_POPCOUNT(~word);
        }
        if (RowCount > 0) {
            sz trailing = RowCount % 64;
            if (trailing != 0 && !Bits.empty()) {
                u64 extra = ~Bits.back() & ~((1ULL << trailing) - 1);
                count -= static_cast<sz>(VOXEL_POPCOUNT(extra));
            }
        }
        return count;
    }

    sz ValidCount() const {
        return RowCount - NullCount();
    }

    u64 GetWord(sz wordIdx) const {
        return Bits[wordIdx];
    }

    void SetWord(sz wordIdx, u64 word) {
        Bits[wordIdx] = word;
    }

    sz WordCount() const {
        return (RowCount + 63) / 64;
    }

    void ApplyFilter(const NullBitmap& mask, const NullBitmap& filter) {
        sz maxRowCount = std::max({RowCount, mask.RowCount, filter.RowCount});
        sz wordCount   = (maxRowCount + 63) / 64;

        std::vector<u64> newBits(wordCount, ~0ULL);
        for (sz w = 0; w < wordCount; ++w) {
            u64 mWord = (w < mask.WordCount())  ? mask.GetWord(w)  : ~0ULL;
            u64 fWord = (w < filter.WordCount()) ? filter.GetWord(w) : ~0ULL;
            newBits[w] = mWord & fWord;
        }

        Bits     = std::move(newBits);
        RowCount = maxRowCount;

        sz trailing = maxRowCount % 64;
        if (trailing != 0 && !Bits.empty()) {
            Bits.back() &= (1ULL << trailing) - 1;
        }
    }

    NullBitmap Combine(const NullBitmap& other) const {
        sz maxCount = std::max(RowCount, other.RowCount);
        NullBitmap result(maxCount);
        sz wordCount = result.WordCount();
        for (sz w = 0; w < wordCount; ++w) {
            u64 a = (w < WordCount())        ? Bits[w]         : ~0ULL;
            u64 b = (w < other.WordCount())  ? other.Bits[w]   : ~0ULL;
            result.Bits[w] = a & b;
        }
        return result;
    }

    void UnionWith(const NullBitmap& other) {
        sz maxCount = std::max(RowCount, other.RowCount);
        sz wordCount = (maxCount + 63) / 64;
        Bits.resize(wordCount, ~0ULL);
        for (sz w = 0; w < wordCount; ++w) {
            u64 otherWord = (w < other.WordCount()) ? other.Bits[w] : ~0ULL;
            Bits[w] &= otherWord;
        }
        RowCount = maxCount;
        TrimTrailingBits();
    }

    void IntersectWith(const NullBitmap& other) {
        sz minCount = std::min(RowCount, other.RowCount);
        sz wordCount = (minCount + 63) / 64;
        for (sz w = 0; w < wordCount; ++w) {
            Bits[w] |= other.Bits[w];
        }
    }

    void Negate() {
        for (u64& w : Bits) w = ~w;
        TrimTrailingBits();
    }

    NullBitmap Invert() const {
        NullBitmap result = *this;
        result.Negate();
        return result;
    }

    u32 PopcountWord(sz wordIdx) const {
        return static_cast<u32>(VOXEL_POPCOUNT(Bits[wordIdx]));
    }

    void CopyTo(u64* dst, sz wordCount) const {
        sz toCopy = std::min(wordCount, Bits.size());
        std::memcpy(dst, Bits.data(), toCopy * sizeof(u64));
    }

    void CopyFrom(const u64* src, sz wordCount) {
        sz toCopy = std::min(wordCount, Bits.size());
        std::memcpy(Bits.data(), src, toCopy * sizeof(u64));
        TrimTrailingBits();
    }

    void Resize(sz newCount) {
        sz newWordCount = (newCount + 63) / 64;
        Bits.resize(newWordCount, ~0ULL);
        RowCount = newCount;

        if (newCount > 0) {
            sz trailing = newCount % 64;
            if (trailing != 0) {
                Bits.back() &= (1ULL << trailing) - 1;
            }
        }
    }

    void CopyFrom(const NullBitmap& other) {
        RowCount = other.RowCount;
        Bits     = other.Bits;
    }

    bool AnyNull() const {
        return NullCount() > 0;
    }

    bool AllNull() const {
        return ValidCount() == 0;
    }

    bool AllValid() const {
        return NullCount() == 0;
    }

    void Print(std::ostream& os, sz limit = 64) const {
        os << "NullBitmap[" << RowCount << "] ";
        sz show = std::min(limit, RowCount);
        for (sz i = 0; i < show; ++i)
            os << (IsValid(i) ? '1' : '0');
        if (RowCount > limit) os << " ...";
    }

private:
    void TrimTrailingBits() {
        if (RowCount > 0 && !Bits.empty()) {
            sz trailing = RowCount % 64;
            if (trailing != 0) {
                Bits.back() &= (1ULL << trailing) - 1;
            }
        }
    }
};

// ============================================================================
// NullableSegment — segment with per-element null indicators
// ============================================================================

template<typename T>
class NullableSegment {
public:
    Segment<T>   Data;
    NullBitmap   Nulls;
    T            NullSentinel = std::numeric_limits<T>::lowest();

    NullableSegment() = default;

    NullableSegment(T* externalData, sz count)
        : Data(externalData, count), Nulls(count)
    {
        SetDefaultSentinel();
    }

    NullableSegment(Arena& arena, sz count)
        : Data(arena, count), Nulls(count)
    {
        SetDefaultSentinel();
    }

    NullableSegment(const NullableSegment&)            = delete;
    NullableSegment& operator=(const NullableSegment&) = delete;

    NullableSegment(NullableSegment&& other) noexcept
        : Data(std::move(other.Data))
        , Nulls(std::move(other.Nulls))
        , NullSentinel(other.NullSentinel)
    {}

    NullableSegment& operator=(NullableSegment&& other) noexcept {
        if (this != &other) {
            Data         = std::move(other.Data);
            Nulls        = std::move(other.Nulls);
            NullSentinel = other.NullSentinel;
        }
        return *this;
    }

    bool IsNull(sz row) const {
        return Nulls.IsNull(row);
    }

    bool IsValid(sz row) const {
        return Nulls.IsValid(row);
    }

    T GetValue(sz row) const {
        return Nulls.IsNull(row) ? NullSentinel : Data[row];
    }

    T GetValueOr(sz row, T fallback) const {
        return Nulls.IsNull(row) ? fallback : Data[row];
    }

    void SetValue(sz row, T value) {
        Data[row] = value;
        Nulls.SetValid(row);
    }

    void SetNull(sz row) {
        Nulls.SetNull(row);
        if constexpr (std::is_floating_point_v<T>) {
            Data[row] = std::numeric_limits<T>::quiet_NaN();
        } else {
            Data[row] = NullSentinel;
        }
    }

    sz Size() const        { return Data.Count; }
    bool Empty() const     { return Data.Empty(); }
    sz ValidCount() const  { return Nulls.ValidCount(); }
    sz NullCountVal() const { return Nulls.NullCount(); }

    void SetAllValid() {
        Nulls.SetAllValid();
    }

    void SetAllNull() {
        Nulls.SetAllNull();
        for (sz i = 0; i < Data.Count; ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                Data[i] = std::numeric_limits<T>::quiet_NaN();
            } else {
                Data[i] = NullSentinel;
            }
        }
    }

    void CompactValid(Arena& arena) {
        if (!Nulls.AnyNull()) return;

        sz validCount = Nulls.ValidCount();
        T* newData = arena.AllocMany<T>(validCount);
        NullBitmap newNulls(validCount);

        sz pos = 0;
        for (sz i = 0; i < Data.Count; ++i) {
            if (Nulls.IsValid(i)) {
                newData[pos++] = Data[i];
            }
        }

        Data.Data      = newData;
        Data.Count     = validCount;
        Data.Capacity  = validCount;
        Data.Owned     = true;
        Data.OwnerArena = &arena;
        Nulls = std::move(newNulls);
    }

    void ApplyFilter(const NullBitmap& filter) {
        Nulls.ApplyFilter(Nulls, filter);
    }

    T SumValid() const {
        T total = T{};
        for (sz i = 0; i < Data.Count; ++i) {
            if (Nulls.IsValid(i)) total += Data[i];
        }
        return total;
    }

    T MinValid() const {
        if (Nulls.ValidCount() == 0) return NullSentinel;
        T val;
        bool first = true;
        for (sz i = 0; i < Data.Count; ++i) {
            if (Nulls.IsValid(i)) {
                if (first) { val = Data[i]; first = false; }
                else if (Data[i] < val) val = Data[i];
            }
        }
        return val;
    }

    T MaxValid() const {
        if (Nulls.ValidCount() == 0) return NullSentinel;
        T val;
        bool first = true;
        for (sz i = 0; i < Data.Count; ++i) {
            if (Nulls.IsValid(i)) {
                if (first) { val = Data[i]; first = false; }
                else if (Data[i] > val) val = Data[i];
            }
        }
        return val;
    }

    void ForEachValid(const std::function<void(sz idx, T value)>& fn) const {
        for (sz i = 0; i < Data.Count; ++i) {
            if (Nulls.IsValid(i)) fn(i, Data[i]);
        }
    }

    void Print(std::ostream& os, sz limit = 20) const {
        os << "NullableSegment[" << Data.Count << "] nulls=" << Nulls.NullCount()
           << " sentinel=" << NullSentinel << "\n  ";
        sz shown = 0;
        for (sz i = 0; i < Data.Count && shown < limit; ++i) {
            if (shown > 0) os << ", ";
            if (Nulls.IsNull(i)) os << "NULL";
            else os << Data[i];
            ++shown;
        }
        if (Data.Count > limit) os << ", ...";
    }

private:
    void SetDefaultSentinel() {
        if constexpr (std::is_floating_point_v<T>) {
            NullSentinel = std::numeric_limits<T>::quiet_NaN();
        }
    }
};

// ============================================================================
// NullPropagation — engine-level null propagation utilities
// ============================================================================

template<typename T>
void PropagateNullsScalar(RegFile& regs, u8 resultRd, u8 ra, u8 rb) {
    u64 aVal = regs.Scalar(ra);
    u64 bVal = regs.Scalar(rb);

    bool aIsNull = false;
    bool bIsNull = false;

    if constexpr (std::is_floating_point_v<T>) {
        f64 a = std::bit_cast<f64>(aVal);
        f64 b = std::bit_cast<f64>(bVal);
        aIsNull = std::isnan(a);
        bIsNull = std::isnan(b);
    } else {
        T a = static_cast<T>(aVal);
        T b = static_cast<T>(bVal);
        T sentinel = std::numeric_limits<T>::lowest();
        aIsNull = (a == sentinel);
        bIsNull = (b == sentinel);
    }

    if (aIsNull || bIsNull) {
        regs.SetFlag(RegFile::Flag::NaN);
        if constexpr (std::is_floating_point_v<T>) {
            regs.Scalar(resultRd) = std::bit_cast<u64>(
                std::numeric_limits<f64>::quiet_NaN());
        } else {
            regs.Scalar(resultRd) = static_cast<u64>(
                std::numeric_limits<T>::lowest());
        }
    }
}

template<typename T>
void PropagateNullsVector(RegFile& regs, u8 resultRd, u8 va, u8 vb) {
    const T* a = regs.VecLanes<T>(va);
    const T* b = regs.VecLanes<T>(vb);
    T*       r = regs.VecLanes<T>(resultRd);

    constexpr sz kLanes = RegFile::kVecWidth / sizeof(T);

    for (sz i = 0; i < kLanes; ++i) {
        bool aNull = false, bNull = false;
        if constexpr (std::is_floating_point_v<T>) {
            aNull = std::isnan(static_cast<f64>(a[i]));
            bNull = std::isnan(static_cast<f64>(b[i]));
        } else {
            T sentinel = std::numeric_limits<T>::lowest();
            aNull = (a[i] == sentinel);
            bNull = (b[i] == sentinel);
        }

        if (aNull || bNull) {
            if constexpr (std::is_floating_point_v<T>) {
                r[i] = std::numeric_limits<T>::quiet_NaN();
            } else {
                r[i] = std::numeric_limits<T>::lowest();
            }
        }
    }

    u32 mask = 0;
    for (sz i = 0; i < kLanes; ++i) {
        bool isNull = false;
        if constexpr (std::is_floating_point_v<T>) {
            isNull = std::isnan(static_cast<f64>(r[i]));
        } else {
            isNull = (r[i] == std::numeric_limits<T>::lowest());
        }
        if (isNull) mask |= (1u << i);
    }
    regs.Mask(0) = mask;
}

inline void PropagateNulls(RegFile& regs, u8 resultRd, u8 ra, u8 rb) {
    u64 aVal = regs.Scalar(ra);
    u64 bVal = regs.Scalar(rb);

    bool aNull = (aVal == static_cast<u64>(-1));
    bool bNull = (bVal == static_cast<u64>(-1));

    if (aNull || bNull) {
        regs.Scalar(resultRd) = static_cast<u64>(-1);
        regs.SetFlag(RegFile::Flag::NaN);
    }
}

// ============================================================================
// NullHandling helpers — batch null propagation for vector lanes
// ============================================================================

template<typename T>
void NullMaskFromVector(RegFile& regs, u8 vecIdx, u8 maskIdx) {
    const T* data = regs.VecLanes<T>(vecIdx);
    constexpr sz kLanes = RegFile::kVecWidth / sizeof(T);
    u32 mask = 0;

    for (sz i = 0; i < kLanes; ++i) {
        bool isNull = false;
        if constexpr (std::is_floating_point_v<T>) {
            isNull = std::isnan(static_cast<f64>(data[i]));
        } else {
            isNull = (data[i] == std::numeric_limits<T>::lowest());
        }
        if (!isNull) mask |= (1u << i);
    }
    regs.Mask(maskIdx) = mask;
}

inline void MaskNullBitmap(const NullBitmap& bm, u64* words, sz wordCount) {
    sz toCopy = std::min(wordCount, bm.WordCount());
    for (sz i = 0; i < toCopy; ++i)
        words[i] &= bm.GetWord(i);
}

} // namespace voxel
