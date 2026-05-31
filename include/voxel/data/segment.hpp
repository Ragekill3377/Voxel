#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/arena.hpp"
#include <vector>
#include <cstring>
#include <memory>
#include <span>
#include <ostream>
#include <algorithm>
#include <utility>
#include <numeric>
#include <functional>

namespace voxel {

// ============================================================================
// Segment — zero-copy typed columnar data segment
// ============================================================================

template<typename T>
class Segment {
public:
    using Iterator      = T*;
    using ConstIterator = const T*;
    using Span          = std::span<T>;

    T*     Data     = nullptr;
    sz     Count    = 0;
    sz     Capacity = 0;
    bool   Owned    = false;
    Arena* OwnerArena = nullptr;

    DataType ElementType  = TypeTraits<T>::kTypeId;
    bool     IsEncoded    = false;
    u8       EncodingType = 0;
    Segment<T>* DecodedView = nullptr;

    Segment() = default;

    Segment(T* externalData, sz count)
        : Data(externalData), Count(count), Capacity(count), Owned(false), OwnerArena(nullptr)
    {}

    Segment(Arena& arena, sz count)
        : Owned(true), OwnerArena(&arena)
    {
        Capacity = count;
        Data = static_cast<T*>(arena.Alloc(count * sizeof(T)));
        Count = count;
        std::memset(Data, 0, count * sizeof(T));
    }

    Segment(const Segment&)            = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : Data(other.Data)
        , Count(other.Count)
        , Capacity(other.Capacity)
        , Owned(other.Owned)
        , OwnerArena(other.OwnerArena)
        , ElementType(other.ElementType)
        , IsEncoded(other.IsEncoded)
        , EncodingType(other.EncodingType)
        , DecodedView(other.DecodedView)
    {
        other.Data        = nullptr;
        other.Count       = 0;
        other.Capacity    = 0;
        other.Owned       = false;
        other.OwnerArena  = nullptr;
        other.IsEncoded   = false;
        other.EncodingType = 0;
        other.DecodedView  = nullptr;
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            Data        = other.Data;
            Count       = other.Count;
            Capacity    = other.Capacity;
            Owned       = other.Owned;
            OwnerArena  = other.OwnerArena;
            ElementType = other.ElementType;
            IsEncoded   = other.IsEncoded;
            EncodingType = other.EncodingType;
            DecodedView  = other.DecodedView;

            other.Data        = nullptr;
            other.Count       = 0;
            other.Capacity    = 0;
            other.Owned       = false;
            other.OwnerArena  = nullptr;
            other.IsEncoded   = false;
            other.EncodingType = 0;
            other.DecodedView  = nullptr;
        }
        return *this;
    }

    T&       operator[](sz i)       { return Data[i]; }
    const T& operator[](sz i) const { return Data[i]; }

    Iterator      begin()       { return Data; }
    Iterator      end()         { return Data + Count; }
    ConstIterator begin() const { return Data; }
    ConstIterator end()   const { return Data + Count; }
    ConstIterator cbegin() const { return Data; }
    ConstIterator cend()   const { return Data + Count; }

    T*       DataPtr()       { return Data; }
    const T* DataPtr() const { return Data; }

    sz   Size()  const { return Count; }
    bool Empty() const { return Count == 0; }

    void LoadVector(void* dst, sz offset, sz count) const {
        sz available = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        std::memcpy(dst, Data + offset, available * sizeof(T));
        if (available < count) {
            std::memset(static_cast<u8*>(dst) + available * sizeof(T), 0,
                        (count - available) * sizeof(T));
        }
    }

    void StoreVector(const void* src, sz offset, sz count) {
        sz toWrite = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        std::memcpy(Data + offset, src, toWrite * sizeof(T));
    }

    void LoadStrided(void* dst, sz offset, sz stride, sz count) const {
        u8* d = static_cast<u8*>(dst);
        for (sz i = 0; i < count; ++i) {
            sz idx = offset + i * stride;
            if (idx < Count) {
                std::memcpy(d + i * sizeof(T), &Data[idx], sizeof(T));
            } else {
                T zero{};
                std::memcpy(d + i * sizeof(T), &zero, sizeof(T));
            }
        }
    }

    void StoreStrided(const void* src, sz offset, sz stride, sz count) {
        const u8* s = static_cast<const u8*>(src);
        for (sz i = 0; i < count; ++i) {
            sz idx = offset + i * stride;
            if (idx < Count) {
                std::memcpy(&Data[idx], s + i * sizeof(T), sizeof(T));
            }
        }
    }

    void Scatter(const void* src, const i32* indices, sz count) {
        const u8* s = static_cast<const u8*>(src);
        for (sz i = 0; i < count; ++i) {
            sz idx = static_cast<sz>(indices[i]);
            if (idx < Count) {
                std::memcpy(&Data[idx], s + i * sizeof(T), sizeof(T));
            }
        }
    }

    void Gather(void* dst, const i32* indices, sz count) const {
        u8* d = static_cast<u8*>(dst);
        for (sz i = 0; i < count; ++i) {
            sz idx = static_cast<sz>(indices[i]);
            if (idx < Count) {
                std::memcpy(d + i * sizeof(T), &Data[idx], sizeof(T));
            } else {
                T zero{};
                std::memcpy(d + i * sizeof(T), &zero, sizeof(T));
            }
        }
    }

    Span CreateSpan(sz offset, sz count) const {
        sz effective = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        return Span(Data + offset, effective);
    }

    Segment Slice(sz offset, sz count) const {
        sz effective = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        Segment result(Data + offset, effective);
        result.ElementType = ElementType;
        result.IsEncoded   = IsEncoded;
        result.EncodingType = EncodingType;
        result.DecodedView  = DecodedView;
        return result;
    }

    void Fill(T value) {
        for (sz i = 0; i < Count; ++i) Data[i] = value;
    }

    void FillRange(sz offset, sz count, T value) {
        sz end = (offset + count <= Count) ? (offset + count) : Count;
        for (sz i = offset; i < end; ++i) Data[i] = value;
    }

    void CopyTo(T* dst, sz count) const {
        sz toCopy = (count <= Count) ? count : Count;
        std::memcpy(dst, Data, toCopy * sizeof(T));
    }

    void CopyFrom(const T* src, sz offset, sz count) {
        sz toCopy = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        std::memcpy(Data + offset, src, toCopy * sizeof(T));
    }

    void Add(const Segment<T>& rhs) { ElemwiseBinary(rhs, std::plus<T>{}); }
    void Sub(const Segment<T>& rhs) { ElemwiseBinary(rhs, std::minus<T>{}); }
    void Mul(const Segment<T>& rhs) { ElemwiseBinary(rhs, std::multiplies<T>{}); }
    void Div(const Segment<T>& rhs) {
        for (sz i = 0; i < Count && i < rhs.Count; ++i) {
            if (rhs.Data[i] != T{}) Data[i] /= rhs.Data[i];
            else Data[i] = T{};
        }
    }

    void Scale(T scalar) {
        for (sz i = 0; i < Count; ++i) Data[i] *= scalar;
    }

    void Offset(T scalar) {
        for (sz i = 0; i < Count; ++i) Data[i] += scalar;
    }

    bool Equals(const Segment<T>& rhs) const {
        if (Count != rhs.Count) return false;
        for (sz i = 0; i < Count; ++i)
            if (Data[i] != rhs.Data[i]) return false;
        return true;
    }

    i32 Compare(const Segment<T>& rhs) const {
        sz n = (Count < rhs.Count) ? Count : rhs.Count;
        for (sz i = 0; i < n; ++i) {
            if (Data[i] < rhs.Data[i]) return -1;
            if (Data[i] > rhs.Data[i]) return 1;
        }
        if (Count < rhs.Count) return -1;
        if (Count > rhs.Count) return 1;
        return 0;
    }

    u64 Hash() const {
        u64 h = static_cast<u64>(Count);
        for (sz i = 0; i < Count; ++i) {
            u64 v = std::bit_cast<u64>(Data[i]);
            h = ((h << 7) | (h >> 57)) ^ (v * 0x9E3779B97F4A7C15ULL);
        }
        return h;
    }

    T Sum() const {
        T total = T{};
        for (sz i = 0; i < Count; ++i) total += Data[i];
        return total;
    }

    T Min() const {
        if (Count == 0) return T{};
        T val = Data[0];
        for (sz i = 1; i < Count; ++i)
            if (Data[i] < val) val = Data[i];
        return val;
    }

    T Max() const {
        if (Count == 0) return T{};
        T val = Data[0];
        for (sz i = 1; i < Count; ++i)
            if (Data[i] > val) val = Data[i];
        return val;
    }

    f64 Mean() const {
        if (Count == 0) return 0.0;
        return static_cast<f64>(Sum()) / static_cast<f64>(Count);
    }

    sz CountMatching(const std::function<bool(const T&)>& pred) const {
        sz c = 0;
        for (sz i = 0; i < Count; ++i)
            if (pred(Data[i])) ++c;
        return c;
    }

    sz CountEqualTo(T value) const {
        sz c = 0;
        for (sz i = 0; i < Count; ++i)
            if (Data[i] == value) ++c;
        return c;
    }

    void Reverse() {
        for (sz i = 0; i < Count / 2; ++i)
            std::swap(Data[i], Data[Count - 1 - i]);
    }

    Segment Clone(Arena& arena) const {
        Segment result(arena, Count);
        std::memcpy(result.Data, Data, Count * sizeof(T));
        result.ElementType  = ElementType;
        result.IsEncoded    = IsEncoded;
        result.EncodingType = EncodingType;
        return result;
    }

    void Print(std::ostream& os, sz limit = 20) const {
        os << "Segment<T>[" << Count << "] = [";
        for (sz i = 0; i < Count && i < limit; ++i) {
            if (i > 0) os << ", ";
            os << Data[i];
        }
        if (Count > limit) os << ", ... (" << (Count - limit) << " more)";
        os << "]";
    }

private:
    template<typename Op>
    void ElemwiseBinary(const Segment<T>& rhs, Op op) {
        sz n = (Count < rhs.Count) ? Count : rhs.Count;
        for (sz i = 0; i < n; ++i) Data[i] = op(Data[i], rhs.Data[i]);
    }
};

// ============================================================================
// MutableSegment — owned, growable columnar segment
// ============================================================================

template<typename T>
class MutableSegment : public Segment<T> {
public:
    MutableSegment() : Segment<T>() {}

    MutableSegment(Arena& arena, sz initialCapacity)
        : Segment<T>(arena, initialCapacity)
    {
        this->Owned = true;
        this->OwnerArena = &arena;
        this->Count = 0;
    }

    void Reserve(sz capacity) {
        if (capacity <= this->Capacity) return;
        VOXEL_ASSERT(this->Owned && this->OwnerArena != nullptr,
                     "Reserve requires an owned segment with an arena");
        Arena& arena = *this->OwnerArena;
        T* newData = arena.AllocMany<T>(capacity);
        if (this->Data && this->Count > 0) {
            std::memcpy(newData, this->Data, this->Count * sizeof(T));
        }
        this->Data     = newData;
        this->Capacity = capacity;
    }

    void Append(T value) {
        if (this->Count >= this->Capacity) {
            sz newCap = this->Capacity == 0 ? 64 : this->Capacity * 2;
            Reserve(newCap);
        }
        this->Data[this->Count++] = value;
    }

    void Append(const T* values, sz count) {
        if (this->Count + count > this->Capacity) {
            sz newCap = this->Capacity == 0 ? 64 : this->Capacity;
            while (newCap < this->Count + count) newCap *= 2;
            Reserve(newCap);
        }
        std::memcpy(this->Data + this->Count, values, count * sizeof(T));
        this->Count += count;
    }

    void AppendSegment(const Segment<T>& seg) {
        Append(seg.DataPtr(), seg.Count);
    }

    void Clear() {
        this->Count = 0;
    }

    void Truncate(sz newSize) {
        if (newSize < this->Count) {
            this->Count = newSize;
        }
    }

    void PopBack() {
        if (this->Count > 0) --this->Count;
    }

    T& Back()       { return this->Data[this->Count - 1]; }
    T  Back() const { return this->Data[this->Count - 1]; }
    T& Front()       { return this->Data[0]; }
    T  Front() const { return this->Data[0]; }

    void Insert(sz pos, T value) {
        VOXEL_ASSERT(pos <= this->Count, "Insert position out of bounds");
        if (this->Count >= this->Capacity) {
            sz newCap = this->Capacity == 0 ? 64 : this->Capacity * 2;
            Reserve(newCap);
        }
        for (sz i = this->Count; i > pos; --i)
            this->Data[i] = this->Data[i - 1];
        this->Data[pos] = value;
        ++this->Count;
    }

    void Remove(sz pos) {
        VOXEL_ASSERT(pos < this->Count, "Remove position out of bounds");
        for (sz i = pos; i + 1 < this->Count; ++i)
            this->Data[i] = this->Data[i + 1];
        --this->Count;
    }

    void RemoveRange(sz pos, sz count) {
        VOXEL_ASSERT(pos + count <= this->Count, "RemoveRange out of bounds");
        for (sz i = pos; i + count < this->Count; ++i)
            this->Data[i] = this->Data[i + count];
        this->Count -= count;
    }

    void Sort() {
        std::sort(this->Data, this->Data + this->Count);
    }

    void SortDescending() {
        std::sort(this->Data, this->Data + this->Count, std::greater<T>{});
    }

    void Unique() {
        if (this->Count <= 1) return;
        Sort();
        sz write = 1;
        for (sz read = 1; read < this->Count; ++read) {
            if (this->Data[read] != this->Data[write - 1]) {
                this->Data[write++] = this->Data[read];
            }
        }
        this->Count = write;
    }

    void Shuffle() {
        for (sz i = this->Count - 1; i > 0; --i) {
            sz j = static_cast<sz>(std::rand()) % (i + 1);
            std::swap(this->Data[i], this->Data[j]);
        }
    }

    void Reset() {
        this->Count = 0;
        this->Capacity = 0;
        if (this->Owned && this->OwnerArena) {
            this->OwnerArena->Reset();
        }
        this->Data = nullptr;
    }
};

// ============================================================================
// ColumnDescriptor — per-column metadata and statistics
// ============================================================================

class ColumnDescriptor {
public:
    std::string Name;
    DataType     Type       = DataType::Null;
    sz           NullCount  = 0;
    sz           DistinctCount = 0;
    f64          MinValue   = 0.0;
    f64          MaxValue   = 0.0;
    bool         HasNulls   = false;

    ColumnDescriptor() = default;

    ColumnDescriptor(std::string name, DataType type)
        : Name(std::move(name)), Type(type)
    {}

    void UpdateStatistics(f64 value, bool isNull) {
        if (isNull) {
            HasNulls = true;
            ++NullCount;
            return;
        }
        ++DistinctCount;
        if (DistinctCount == 1) {
            MinValue = value;
            MaxValue = value;
        } else {
            if (value < MinValue) MinValue = value;
            if (value > MaxValue) MaxValue = value;
        }
    }

    void ComputeFromSegment(const f64* data, sz count, const bool* nullFlags = nullptr) {
        NullCount      = 0;
        DistinctCount  = 0;
        HasNulls       = false;
        MinValue       = 0.0;
        MaxValue       = 0.0;

        if (count == 0) return;

        bool firstValid = false;
        for (sz i = 0; i < count; ++i) {
            bool isNull = (nullFlags != nullptr) && !nullFlags[i];
            if (isNull) {
                HasNulls = true;
                ++NullCount;
                continue;
            }
            ++DistinctCount;
            f64 v = data[i];
            if (!firstValid) {
                MinValue   = v;
                MaxValue   = v;
                firstValid = true;
            } else {
                if (v < MinValue) MinValue = v;
                if (v > MaxValue) MaxValue = v;
            }
        }
    }

    f64 Range() const { return MaxValue - MinValue; }
    f64 DistinctnessRatio(sz rowCount) const {
        if (rowCount == 0) return 0.0;
        return static_cast<f64>(DistinctCount) / static_cast<f64>(rowCount);
    }
    f64 NullRatio(sz rowCount) const {
        if (rowCount == 0) return 0.0;
        return static_cast<f64>(NullCount) / static_cast<f64>(rowCount);
    }

    void Merge(const ColumnDescriptor& other) {
        if (other.NullCount > 0) HasNulls = true;
        NullCount     += other.NullCount;
        DistinctCount += other.DistinctCount;
        if (other.DistinctCount > 0) {
            if (DistinctCount == other.DistinctCount || DistinctCount == 0) {
                MinValue = other.MinValue;
                MaxValue = other.MaxValue;
            } else {
                if (other.MinValue < MinValue) MinValue = other.MinValue;
                if (other.MaxValue > MaxValue) MaxValue = other.MaxValue;
            }
        }
    }

    void Reset() {
        NullCount     = 0;
        DistinctCount = 0;
        HasNulls      = false;
        MinValue      = 0.0;
        MaxValue      = 0.0;
    }

    void Print(std::ostream& os) const {
        os << "Column: " << Name
           << "  Type: " << TypeName(Type)
           << "  Nulls: " << NullCount
           << "  Distinct: " << DistinctCount
           << "  Range: [" << MinValue << ", " << MaxValue << "]"
           << "  HasNulls: " << (HasNulls ? "yes" : "no");
    }
};

// ============================================================================
// Segment utilities
// ============================================================================

template<typename T>
Segment<T> MakeSegment(Arena& arena, sz count) {
    return Segment<T>(arena, count);
}

template<typename T>
Segment<T> WrapSegment(T* data, sz count) {
    return Segment<T>(data, count);
}

template<typename T>
void SegmentCopy(Segment<T>& dst, sz dstOff, const Segment<T>& src, sz srcOff, sz count) {
    sz maxCopy = count;
    if (dstOff + maxCopy > dst.Count) maxCopy = dst.Count - dstOff;
    if (srcOff + maxCopy > src.Count) maxCopy = src.Count - srcOff;
    std::memcpy(dst.DataPtr() + dstOff, src.DataPtr() + srcOff, maxCopy * sizeof(T));
}

template<typename T>
void SegmentAdd(Segment<T>& dst, const Segment<T>& a, const Segment<T>& b) {
    sz n = dst.Count;
    if (a.Count < n) n = a.Count;
    if (b.Count < n) n = b.Count;
    for (sz i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}

template<typename T>
void SegmentFilter(const Segment<T>& src, const std::function<bool(const T&)>& pred,
                   MutableSegment<T>& dst) {
    for (sz i = 0; i < src.Count; ++i) {
        if (pred(src[i])) dst.Append(src[i]);
    }
}

} // namespace voxel
