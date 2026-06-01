#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/data/segment.hpp"
#include "voxel/bytecode/opcodes.hpp"
#include "voxel/ops/hash.hpp"
#include <cstring>
#include <algorithm>
#include <vector>
#include <ostream>

namespace voxel {
namespace ops {

template<typename TKey>
class HashJoin {
public:
    struct BuildEntry {
        TKey        Key;
        u32         RowId;
        BuildEntry* Next;
    };

    HashJoin(Arena& arena)
        : HashTable_(nullptr), TableSize_(0), EntryCount_(0),
          EntryArena_(nullptr), EntryCursor_(nullptr), EntryLimit_(nullptr),
          Arena_(arena)
    {}

    ~HashJoin() = default;

    HashJoin(const HashJoin&) = delete;
    HashJoin& operator=(const HashJoin&) = delete;

    void Init(sz expectedRows) {
        TableSize_ = NextPowerOfTwo(std::max(expectedRows * 2, static_cast<sz>(64)));
        sz tableBytes = TableSize_ * sizeof(BuildEntry*);
        HashTable_ = static_cast<BuildEntry**>(Arena_.Alloc(tableBytes));
        std::memset(HashTable_, 0, tableBytes);

        sz entryBytes = expectedRows * sizeof(BuildEntry);
        EntryArena_ = static_cast<BuildEntry*>(Arena_.Alloc(entryBytes));
        EntryCursor_ = EntryArena_;
        EntryLimit_  = EntryArena_ + expectedRows;

        EntryCount_ = 0;
    }

    VOXEL_HOT void Build(const TKey* VOXEL_RESTRICT buildKeys, sz buildCount) {
        Init(buildCount);

        for (sz i = 0; i < buildCount; ++i) {
            Insert(buildKeys[i], static_cast<u32>(i));
        }
    }

    VOXEL_HOT void Probe(const TKey* VOXEL_RESTRICT probeKeys, sz probeCount,
                          u32* VOXEL_RESTRICT leftMatches,
                          u32* VOXEL_RESTRICT rightMatches,
                          sz& matchCount) {
        matchCount = 0;
        sz maxMatches = probeCount;

        for (sz i = 0; i < probeCount && matchCount < maxMatches; ++i) {
            u64 h = HashValue(probeKeys[i]);
            sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

            BuildEntry* entry = HashTable_[slot];
            while (entry) {
                if (entry->Key == probeKeys[i]) {
                    leftMatches[matchCount]  = static_cast<u32>(i);
                    rightMatches[matchCount] = entry->RowId;
                    ++matchCount;
                    break;
                }
                entry = entry->Next;
            }
        }
    }

    VOXEL_HOT void ProbeAllMatches(const TKey* VOXEL_RESTRICT probeKeys, sz probeCount,
                                    u32* VOXEL_RESTRICT leftMatches,
                                    u32* VOXEL_RESTRICT rightMatches,
                                    sz& matchCount, sz maxMatches) {
        matchCount = 0;

        for (sz i = 0; i < probeCount && matchCount < maxMatches; ++i) {
            u64 h = HashValue(probeKeys[i]);
            sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

            BuildEntry* entry = HashTable_[slot];
            while (entry && matchCount < maxMatches) {
                if (entry->Key == probeKeys[i]) {
                    leftMatches[matchCount]  = static_cast<u32>(i);
                    rightMatches[matchCount] = entry->RowId;
                    ++matchCount;
                }
                entry = entry->Next;
            }
        }
    }

    bool Contains(TKey key) const {
        u64 h = HashValue(key);
        sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

        BuildEntry* entry = HashTable_[slot];
        while (entry) {
            if (entry->Key == key) return true;
            entry = entry->Next;
        }
        return false;
    }

    u32 Lookup(TKey key) const {
        u64 h = HashValue(key);
        sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

        BuildEntry* entry = HashTable_[slot];
        while (entry) {
            if (entry->Key == key) return entry->RowId;
            entry = entry->Next;
        }
        return static_cast<u32>(-1);
    }

    sz GetTableSize() const { return TableSize_; }
    sz GetEntryCount() const { return EntryCount_; }

    void Clear() {
        if (HashTable_ && TableSize_ > 0) {
            std::memset(HashTable_, 0, TableSize_ * sizeof(BuildEntry*));
        }
        EntryCount_ = 0;
        EntryCursor_ = EntryArena_;
    }

private:
    BuildEntry** HashTable_;
    sz           TableSize_;
    sz           EntryCount_;

    BuildEntry*  EntryArena_;
    BuildEntry*  EntryCursor_;
    BuildEntry*  EntryLimit_;

    Arena&       Arena_;

    void Insert(TKey key, u32 rowId) {
        u64 h = HashValue(key);
        sz slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));

        if (EntryCursor_ >= EntryLimit_) {
            Grow();
            slot = static_cast<sz>(h & (static_cast<u64>(TableSize_) - 1));
        }

        BuildEntry* newEntry = EntryCursor_++;
        newEntry->Key  = key;
        newEntry->RowId = rowId;
        newEntry->Next = HashTable_[slot];
        HashTable_[slot] = newEntry;
        ++EntryCount_;
    }

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
        BuildEntry** oldTable = HashTable_;

        sz newSize = std::max(oldSize * 2, static_cast<sz>(64));
        sz newTableBytes = newSize * sizeof(BuildEntry*);
        HashTable_ = static_cast<BuildEntry**>(Arena_.Alloc(newTableBytes));
        std::memset(HashTable_, 0, newTableBytes);
        TableSize_ = newSize;

        sz newEntryCount = (EntryCursor_ - EntryArena_) * 2;
        sz newEntryBytes = newEntryCount * sizeof(BuildEntry);
        BuildEntry* newArena = static_cast<BuildEntry*>(Arena_.Alloc(newEntryBytes));
        BuildEntry* newCursor = newArena;

        for (sz slot = 0; slot < oldSize; ++slot) {
            BuildEntry* entry = oldTable[slot];
            while (entry) {
                BuildEntry* next = entry->Next;

                sz newSlot = static_cast<sz>(HashValue(entry->Key) & (static_cast<u64>(newSize) - 1));

                BuildEntry* newEntry = newCursor++;
                newEntry->Key   = entry->Key;
                newEntry->RowId = entry->RowId;
                newEntry->Next  = HashTable_[newSlot];
                HashTable_[newSlot] = newEntry;

                entry = next;
            }
        }

        EntryArena_  = newArena;
        EntryCursor_ = newCursor;
        EntryLimit_  = newArena + newEntryCount;
    }
};

template<typename TKey>
class MergeJoin {
public:
    void Join(const TKey* VOXEL_RESTRICT leftKeys, sz leftCount,
              const TKey* VOXEL_RESTRICT rightKeys, sz rightCount,
              u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
              sz& matchCount) {
        matchCount = 0;
        sz li = 0;
        sz ri = 0;

        while (li < leftCount && ri < rightCount) {
            if (leftKeys[li] < rightKeys[ri]) {
                ++li;
            } else if (rightKeys[ri] < leftKeys[li]) {
                ++ri;
            } else {
                sz rightStart = ri;
                while (ri < rightCount && rightKeys[ri] == leftKeys[li]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                    ++ri;
                }
                ++li;
                ri = rightStart;
            }
        }
    }

    void JoinUnique(const TKey* VOXEL_RESTRICT leftKeys, sz leftCount,
                    const TKey* VOXEL_RESTRICT rightKeys, sz rightCount,
                    u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                    sz& matchCount) {
        matchCount = 0;
        sz li = 0;
        sz ri = 0;

        while (li < leftCount && ri < rightCount) {
            if (leftKeys[li] < rightKeys[ri]) {
                ++li;
            } else if (rightKeys[ri] < leftKeys[li]) {
                ++ri;
            } else {
                leftOut[matchCount]  = static_cast<u32>(li);
                rightOut[matchCount] = static_cast<u32>(ri);
                ++matchCount;
                ++li;
                ++ri;
            }
        }
    }
};

template<typename TKey>
class AntiJoin {
public:
    void Join(const TKey* VOXEL_RESTRICT leftKeys, sz leftCount,
              const TKey* VOXEL_RESTRICT rightKeys, sz rightCount,
              u32* VOXEL_RESTRICT outLeft, sz& matchCount,
              Arena& arena) {
        matchCount = 0;

        HashJoin<TKey> hashJoin(arena);
        hashJoin.Build(rightKeys, rightCount);

        for (sz i = 0; i < leftCount; ++i) {
            if (!hashJoin.Contains(leftKeys[i])) {
                outLeft[matchCount] = static_cast<u32>(i);
                ++matchCount;
            }
        }
    }
};

template<typename TKey>
class SemiJoin {
public:
    void Join(const TKey* VOXEL_RESTRICT leftKeys, sz leftCount,
              const TKey* VOXEL_RESTRICT rightKeys, sz rightCount,
              u32* VOXEL_RESTRICT outLeft, sz& matchCount,
              Arena& arena) {
        matchCount = 0;

        HashJoin<TKey> hashJoin(arena);
        hashJoin.Build(rightKeys, rightCount);

        for (sz i = 0; i < leftCount; ++i) {
            if (hashJoin.Contains(leftKeys[i])) {
                outLeft[matchCount] = static_cast<u32>(i);
                ++matchCount;
            }
        }
    }
};

class NestedLoopJoin {
public:
    void Join(const Segment<f64>& left, const Segment<f64>& right,
              u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
              sz& matchCount,
              bool (*predicate)(f64, f64)) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (predicate(lData[li], rData[ri])) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }

    void JoinEq(const Segment<f64>& left, const Segment<f64>& right,
                u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                sz& matchCount) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (lData[li] == rData[ri]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }

    void JoinLT(const Segment<f64>& left, const Segment<f64>& right,
                u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                sz& matchCount) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (lData[li] < rData[ri]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }

    void JoinLE(const Segment<f64>& left, const Segment<f64>& right,
                u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                sz& matchCount) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (lData[li] <= rData[ri]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }

    void JoinGT(const Segment<f64>& left, const Segment<f64>& right,
                u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                sz& matchCount) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (lData[li] > rData[ri]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }

    void JoinNE(const Segment<f64>& left, const Segment<f64>& right,
                u32* VOXEL_RESTRICT leftOut, u32* VOXEL_RESTRICT rightOut,
                sz& matchCount) {
        matchCount = 0;

        const f64* VOXEL_RESTRICT lData = left.Data;
        const f64* VOXEL_RESTRICT rData = right.Data;
        sz lCount = left.Count;
        sz rCount = right.Count;

        for (sz li = 0; li < lCount; ++li) {
            for (sz ri = 0; ri < rCount; ++ri) {
                if (lData[li] != rData[ri]) {
                    leftOut[matchCount]  = static_cast<u32>(li);
                    rightOut[matchCount] = static_cast<u32>(ri);
                    ++matchCount;
                }
            }
        }
    }
};

} // namespace ops
} // namespace voxel
