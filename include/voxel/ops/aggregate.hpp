#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/data/segment.hpp"
#include "voxel/bytecode/opcodes.hpp"
#include "voxel/ops/hash.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>
#include <ostream>

namespace voxel {
namespace ops {

struct WelfordState {
    sz  Count;
    f64 Mean;
    f64 M2;

    WelfordState() : Count(0), Mean(0.0), M2(0.0) {}

    void Reset() {
        Count = 0;
        Mean  = 0.0;
        M2    = 0.0;
    }
};

inline void WelfordUpdate(WelfordState& s, f64 value) {
    s.Count++;
    f64 delta = value - s.Mean;
    s.Mean += delta / static_cast<f64>(s.Count);
    f64 delta2 = value - s.Mean;
    s.M2 += delta * delta2;
}

inline void WelfordRemove(WelfordState& s, f64 value) {
    if (s.Count <= 1) {
        s.Reset();
        return;
    }
    f64 oldMean = (s.Mean * s.Count - value) / static_cast<f64>(s.Count - 1);
    f64 delta = value - oldMean;
    f64 delta2 = value - s.Mean;
    s.M2 -= delta * delta2;
    s.Count--;
    s.Mean = oldMean;
}

inline f64 WelfordVariance(const WelfordState& s) {
    if (s.Count < 2) return 0.0;
    return s.M2 / static_cast<f64>(s.Count - 1);
}

inline f64 WelfordStdDev(const WelfordState& s) {
    return std::sqrt(WelfordVariance(s));
}

inline f64 WelfordPopulationVariance(const WelfordState& s) {
    if (s.Count == 0) return 0.0;
    return s.M2 / static_cast<f64>(s.Count);
}

inline f64 WelfordPopulationStdDev(const WelfordState& s) {
    return std::sqrt(WelfordPopulationVariance(s));
}

template<typename TKey, typename TValue>
class HashAggregator {
public:
    struct AggregateState {
        TValue Sum;
        TValue Min;
        TValue Max;
        sz     Count;
        f64    M2;
        f64    Mean;

        AggregateState() : Sum{}, Min{}, Max{}, Count(0), M2(0.0), Mean(0.0) {}

        void Init(TValue firstValue) {
            Sum   = firstValue;
            Min   = firstValue;
            Max   = firstValue;
            Count = 1;
            Mean  = static_cast<f64>(firstValue);
            M2    = 0.0;
        }

        void Update(TValue value) {
            Sum += value;
            if (value < Min || Count == 0) Min = value;
            if (value > Max || Count == 0) Max = value;

            Count++;
            f64 v = static_cast<f64>(value);
            f64 delta = v - Mean;
            Mean += delta / static_cast<f64>(Count);
            f64 delta2 = v - Mean;
            M2 += delta * delta2;
        }

        void Merge(const AggregateState& other) {
            if (other.Count == 0) return;
            if (Count == 0) {
                *this = other;
                return;
            }

            f64 oldCount = static_cast<f64>(Count);
            f64 newCount = static_cast<f64>(other.Count);

            Sum += other.Sum;
            if (other.Min < Min) Min = other.Min;
            if (other.Max > Max) Max = other.Max;

            f64 delta = other.Mean - Mean;
            M2 += other.M2 + delta * delta * oldCount * newCount / static_cast<f64>(Count + other.Count);
            Mean = (Mean * oldCount + other.Mean * newCount) / static_cast<f64>(Count + other.Count);
            Count += other.Count;
        }

        f64 Variance() const {
            if (Count < 2) return 0.0;
            return M2 / static_cast<f64>(Count - 1);
        }

        f64 StdDev() const {
            return std::sqrt(Variance());
        }

        TValue Avg() const {
            if (Count == 0) return TValue{};
            return Sum / static_cast<TValue>(Count);
        }
    };

    struct HashEntry {
        TKey           Key;
        AggregateState State;
        u64            Hash;
        HashEntry*     Next;
    };

    HashAggregator(Arena& arena)
        : Table_(nullptr), TableSize_(0), EntryCount_(0),
          EntryArena_(nullptr), EntryLimit_(nullptr), EntryCursor_(nullptr),
          Arena_(arena), LoadFactorLimit_(0.75)
    {}

    ~HashAggregator() = default;

    HashAggregator(const HashAggregator&) = delete;
    HashAggregator& operator=(const HashAggregator&) = delete;

    void Init(sz expectedGroups) {
        TableSize_ = NextPowerOfTwo(std::max(expectedGroups * 2, static_cast<sz>(16)));
        sz tableBytes = TableSize_ * sizeof(HashEntry*);
        Table_ = static_cast<HashEntry**>(Arena_.Alloc(tableBytes));
        std::memset(Table_, 0, tableBytes);

        sz entryBytes = expectedGroups * sizeof(HashEntry);
        EntryArena_ = static_cast<HashEntry*>(Arena_.Alloc(entryBytes));
        EntryLimit_ = EntryArena_ + expectedGroups;
        EntryCursor_ = EntryArena_;

        EntryCount_ = 0;
        LoadFactorLimit_ = 0.75;
    }

    VOXEL_HOT void Accumulate(const TKey* VOXEL_RESTRICT keys,
                               const TValue* VOXEL_RESTRICT values,
                               sz count) {
        for (sz i = 0; i < count; ++i) {
            AccumulateOne(keys[i], values[i]);
        }
    }

    VOXEL_HOT void AccumulateOne(TKey key, TValue value) {
        u64 h = HashValue(key);

        sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));
        HashEntry* entry = Table_[slot];

        while (entry) {
            if (entry->Hash == h && entry->Key == key) {
                entry->State.Update(value);
                return;
            }
            entry = entry->Next;
        }

        if (EntryCursor_ >= EntryLimit_) {
            Grow();
            slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));
        }

        HashEntry* newEntry = EntryCursor_++;
        newEntry->Key  = key;
        newEntry->Hash = h;
        newEntry->Next = Table_[slot];
        newEntry->State.Init(value);
        Table_[slot] = newEntry;
        ++EntryCount_;
    }

    void Merge(const HashAggregator& other) {
        for (sz slot = 0; slot < other.TableSize_; ++slot) {
            HashEntry* entry = other.Table_[slot];
            while (entry) {
                u64 h = entry->Hash;
                sz mySlot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

                HashEntry* existing = Table_[mySlot];
                HashEntry* found = nullptr;
                while (existing) {
                    if (existing->Hash == h && existing->Key == entry->Key) {
                        found = existing;
                        break;
                    }
                    existing = existing->Next;
                }

                if (found) {
                    found->State.Merge(entry->State);
                } else {
                    if (EntryCursor_ >= EntryLimit_) {
                        Grow();
                        mySlot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));
                    }

                    HashEntry* newEntry = EntryCursor_++;
                    newEntry->Key  = entry->Key;
                    newEntry->Hash = h;
                    newEntry->Next = Table_[mySlot];
                    newEntry->State = entry->State;
                    Table_[mySlot] = newEntry;
                    ++EntryCount_;
                }
                entry = entry->Next;
            }
        }
    }

    void Finalize(TKey* VOXEL_RESTRICT outKeys,
                  TValue* VOXEL_RESTRICT outSums,
                  sz* VOXEL_RESTRICT outCounts,
                  sz maxGroups) {
        sz written = 0;

        for (sz slot = 0; slot < TableSize_ && written < maxGroups; ++slot) {
            HashEntry* entry = Table_[slot];
            while (entry && written < maxGroups) {
                outKeys[written]   = entry->Key;
                outSums[written]   = entry->State.Sum;
                outCounts[written] = entry->State.Count;
                ++written;
                entry = entry->Next;
            }
        }
    }

    sz GroupCount() const { return EntryCount_; }

    AggregateState* Lookup(TKey key) const {
        if (!Table_) return nullptr;

        u64 h = HashValue(key);
        sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

        HashEntry* entry = Table_[slot];
        while (entry) {
            if (entry->Hash == h && entry->Key == key)
                return &entry->State;
            entry = entry->Next;
        }
        return nullptr;
    }

    void Clear() {
        if (Table_ && TableSize_ > 0) {
            std::memset(Table_, 0, TableSize_ * sizeof(HashEntry*));
        }
        EntryCount_ = 0;
        EntryCursor_ = EntryArena_;
    }

private:
    HashEntry** Table_;
    sz          TableSize_;
    sz          EntryCount_;

    HashEntry*  EntryArena_;
    HashEntry*  EntryLimit_;
    HashEntry*  EntryCursor_;

    Arena&      Arena_;
    f64         LoadFactorLimit_;

    static sz NextPowerOfTwo(sz v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

    VOXEL_NOINLINE void Grow() {
        sz oldSize = TableSize_;
        HashEntry** oldTable = Table_;

        sz newSize = std::max(oldSize * 2, static_cast<sz>(16));
        sz newTableBytes = newSize * sizeof(HashEntry*);
        Table_ = static_cast<HashEntry**>(Arena_.Alloc(newTableBytes));
        std::memset(Table_, 0, newTableBytes);
        TableSize_ = newSize;

        sz newEntryCount = EntryCount_ * 2;
        sz newEntryBytes = newEntryCount * sizeof(HashEntry);
        HashEntry* newArena = static_cast<HashEntry*>(Arena_.Alloc(newEntryBytes));
        EntryLimit_ = newArena + newEntryCount;
        EntryCursor_ = newArena;

        sz rehashed = 0;
        for (sz slot = 0; slot < oldSize; ++slot) {
            HashEntry* entry = oldTable[slot];
            while (entry) {
                HashEntry* next = entry->Next;

                sz newSlot = static_cast<sz>(entry->Hash & (static_cast<u64>(newSize) - 1));

                HashEntry* newEntry = EntryCursor_++;
                newEntry->Key   = entry->Key;
                newEntry->Hash  = entry->Hash;
                newEntry->State = entry->State;
                newEntry->Next  = Table_[newSlot];
                Table_[newSlot] = newEntry;
                ++rehashed;

                entry = next;
            }
        }

        EntryArena_ = newArena;
    }
};

template<typename T>
class StreamingAggregator {
public:
    StreamingAggregator() : Sum_{}, Min_{}, Max_{}, Count_(0) {
        Welford_.Reset();
    }

    VOXEL_HOT void Accumulate(const T* VOXEL_RESTRICT values, sz count) {
        if (count == 0) return;

        for (sz i = 0; i < count; ++i) {
            T v = values[i];
            Sum_ += v;
            if (Count_ == 0) {
                Min_ = v;
                Max_ = v;
            } else {
                if (v < Min_) Min_ = v;
                if (v > Max_) Max_ = v;
            }
            WelfordUpdate(Welford_, static_cast<f64>(v));
            Count_++;
        }
    }

    void Finalize(T& outSum, T& outMin, T& outMax, T& outAvg, T& outStdDev, sz& outCount) {
        outSum    = Sum_;
        outMin    = Min_;
        outMax    = Max_;
        outCount  = Count_;

        if constexpr (std::is_floating_point_v<T>) {
            outAvg    = (Count_ > 0) ? Sum_ / static_cast<T>(Count_) : T{};
            outStdDev = static_cast<T>(WelfordStdDev(Welford_));
        } else {
            outAvg    = (Count_ > 0) ? static_cast<T>(Sum_ / Count_) : T{};
            outStdDev = static_cast<T>(WelfordStdDev(Welford_));
        }
    }

    void Merge(const StreamingAggregator& other) {
        if (other.Count_ == 0) return;
        if (Count_ == 0) {
            *this = other;
            return;
        }

        Sum_ += other.Sum_;
        if (other.Min_ < Min_) Min_ = other.Min_;
        if (other.Max_ > Max_) Max_ = other.Max_;

        f64 oldCount = static_cast<f64>(Count_);
        f64 newCount = static_cast<f64>(other.Count_);

        f64 delta = other.Welford_.Mean - Welford_.Mean;
        Welford_.M2 += other.Welford_.M2 + delta * delta * oldCount * newCount / static_cast<f64>(Count_ + other.Count_);
        Welford_.Mean = (Welford_.Mean * oldCount + other.Welford_.Mean * newCount) / static_cast<f64>(Count_ + other.Count_);
        Welford_.Count += other.Welford_.Count;
        Count_ += other.Count_;
    }

    void Clear() {
        Sum_ = T{};
        Min_ = T{};
        Max_ = T{};
        Count_ = 0;
        Welford_.Reset();
    }

    T  GetSum() const { return Sum_; }
    T  GetMin() const { return Min_; }
    T  GetMax() const { return Max_; }
    sz GetCount() const { return Count_; }

private:
    T            Sum_;
    T            Min_;
    T            Max_;
    sz           Count_;
    WelfordState Welford_;
};

class AggregateOperator {
public:
    static void AggSum(RegFile& regs, u8 rd, const Segment<f64>& seg,
                       u8 ra, u8 rb) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        static constexpr sz kLanes = RegFile::kVecWidth / sizeof(f64);
        sz lanes = std::min(count, kLanes);

        f64 sum = 0.0;
        if (lanes == count && count <= kLanes) {
            for (sz i = 0; i < count; ++i) {
                sum += data[i];
            }
        } else {
            for (sz i = 0; i < count; ++i) {
                sum += data[i];
            }
        }

        if constexpr (std::is_floating_point_v<f64>)
            regs.Scalar(rd) = std::bit_cast<u64>(sum);
        else
            regs.Scalar(rd) = static_cast<u64>(sum);
    }

    static void AggCount(RegFile& regs, u8 rd, const Segment<f64>& seg, sz count) {
        sz actualCount = std::min(count, seg.Count);
        regs.Scalar(rd) = static_cast<u64>(actualCount);
    }

    static void AggMin(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        if (count == 0) {
            regs.Scalar(rd) = 0;
            return;
        }

        f64 minVal = data[0];
        for (sz i = 1; i < count; ++i) {
            if (data[i] < minVal) minVal = data[i];
        }

        regs.Scalar(rd) = std::bit_cast<u64>(minVal);
    }

    static void AggMax(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        if (count == 0) {
            regs.Scalar(rd) = 0;
            return;
        }

        f64 maxVal = data[0];
        for (sz i = 1; i < count; ++i) {
            if (data[i] > maxVal) maxVal = data[i];
        }

        regs.Scalar(rd) = std::bit_cast<u64>(maxVal);
    }

    static void AggAvg(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        if (count == 0) {
            regs.Scalar(rd) = 0;
            return;
        }

        f64 sum = 0.0;
        for (sz i = 0; i < count; ++i) {
            sum += data[i];
        }

        regs.Scalar(rd) = std::bit_cast<u64>(sum / static_cast<f64>(count));
    }

    static void AggStdDev(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        if (count < 2) {
            regs.Scalar(rd) = 0;
            return;
        }

        WelfordState ws;
        for (sz i = 0; i < count; ++i) {
            WelfordUpdate(ws, data[i]);
        }

        regs.Scalar(rd) = std::bit_cast<u64>(WelfordStdDev(ws));
    }

    static void AggVar(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        const f64* data  = seg.Data;
        sz         count = seg.Count;

        if (count < 2) {
            regs.Scalar(rd) = 0;
            return;
        }

        f64 mean = 0.0;
        for (sz i = 0; i < count; ++i) {
            mean += data[i];
        }
        mean /= static_cast<f64>(count);

        f64 variance = 0.0;
        for (sz i = 0; i < count; ++i) {
            f64 diff = data[i] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<f64>(count - 1);

        regs.Scalar(rd) = std::bit_cast<u64>(variance);
    }

    static void AggSumI64(RegFile& regs, u8 rd, const Segment<i64>& seg) {
        const i64* data  = seg.Data;
        sz         count = seg.Count;

        i64 sum = 0;
        for (sz i = 0; i < count; ++i) {
            sum += data[i];
        }

        regs.Scalar(rd) = static_cast<u64>(sum);
    }

    static void AggCount64(RegFile& regs, u8 rd, const Segment<f64>& seg) {
        sz count = seg.Count;
        regs.Scalar(rd) = static_cast<u64>(count);
    }
};

template<typename TKey, typename TValue>
class GroupedAggregator {
public:
    GroupedAggregator(Arena& arena)
        : Aggregator_(arena), SortedKeys_(nullptr), SortedValues_(nullptr), Count_(0) {}

    void AccumulateSorted(const TKey* VOXEL_RESTRICT sortedKeys,
                          const TValue* VOXEL_RESTRICT sortedValues,
                          sz count) {
        SortedKeys_   = sortedKeys;
        SortedValues_ = sortedValues;
        Count_        = count;
    }

    void FinalizeGroups(TKey* VOXEL_RESTRICT outKeys,
                        TValue* VOXEL_RESTRICT outSums,
                        TValue* VOXEL_RESTRICT outMins,
                        TValue* VOXEL_RESTRICT outMaxs,
                        sz* VOXEL_RESTRICT outCounts,
                        f64* VOXEL_RESTRICT outAvgs,
                        sz& groupCount) {
        groupCount = 0;
        if (Count_ == 0) return;

        sz groupStart = 0;
        while (groupStart < Count_) {
            TKey groupKey = SortedKeys_[groupStart];

            typename HashAggregator<TKey, TValue>::AggregateState state;
            state.Init(SortedValues_[groupStart]);

            sz idx = groupStart + 1;
            while (idx < Count_ && SortedKeys_[idx] == groupKey) {
                state.Update(SortedValues_[idx]);
                ++idx;
            }

            outKeys[groupCount]   = groupKey;
            outSums[groupCount]   = state.Sum;
            outMins[groupCount]   = state.Min;
            outMaxs[groupCount]   = state.Max;
            outCounts[groupCount] = state.Count;
            outAvgs[groupCount]   = state.Count > 0 ? static_cast<f64>(state.Sum) / static_cast<f64>(state.Count) : 0.0;

            ++groupCount;
            groupStart = idx;
        }
    }

private:
    HashAggregator<TKey, TValue> Aggregator_;
    const TKey*   SortedKeys_;
    const TValue* SortedValues_;
    sz            Count_;
};

template<typename TKey, typename TValue>
class HybridAggregator {
public:
    HybridAggregator(Arena& arena)
        : HashAgg_(arena), SortKeys_(nullptr), SortValues_(nullptr), Count_(0),
          UseHash_(true), HashThreshold_(64), Arena_(arena) {}

    void SetHashThreshold(sz threshold) { HashThreshold_ = threshold; }

    void Accumulate(const TKey* VOXEL_RESTRICT keys,
                    const TValue* VOXEL_RESTRICT values,
                    sz count) {
        Count_ = count;

        if (count <= HashThreshold_) {
            UseHash_ = true;
            HashAgg_.Init(count);
            HashAgg_.Accumulate(keys, values, count);
        } else {
            UseHash_ = false;
            SortKeys_   = static_cast<TKey*>(Arena_.Alloc(count * sizeof(TKey)));
            SortValues_ = static_cast<TValue*>(Arena_.Alloc(count * sizeof(TValue)));
            std::memcpy(SortKeys_,   keys,   count * sizeof(TKey));
            std::memcpy(SortValues_, values, count * sizeof(TValue));
            std::sort(SortKeys_, SortKeys_ + count, [](const TKey& a, const TKey& b) { return a < b; });
        }
    }

    void Finalize(TKey* VOXEL_RESTRICT outKeys,
                  TValue* VOXEL_RESTRICT outSums,
                  sz* VOXEL_RESTRICT outCounts,
                  sz maxGroups) {
        if (UseHash_) {
            HashAgg_.Finalize(outKeys, outSums, outCounts, maxGroups);
        }
    }

    sz GroupCount() const {
        return UseHash_ ? HashAgg_.GroupCount() : 0;
    }

private:
    HashAggregator<TKey, TValue> HashAgg_;
    TKey*   SortKeys_;
    TValue* SortValues_;
    sz      Count_;
    bool    UseHash_;
    sz      HashThreshold_;
    Arena&  Arena_;
};

} // namespace ops
} // namespace voxel
