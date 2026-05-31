#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/arena.hpp"
#include "voxel/data/nulls.hpp"
#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <thread>
#include <deque>
#include <cassert>

namespace voxel {

// ============================================================================
// DataBlock<T> — fixed-size block of columnar data for vectorized batches
// ============================================================================

template<typename T>
class DataBlock {
public:
    static constexpr sz kDefaultBlockSize = 4096;

    T*          Data     = nullptr;
    sz          Count    = 0;
    sz          Capacity = 0;
    NullBitmap* Nulls    = nullptr;

    DataBlock() : Data(nullptr), Count(0), Capacity(0), Nulls(nullptr) {}

    explicit DataBlock(sz capacity)
        : Capacity(capacity)
    {
        Data = new T[capacity];
        Count = 0;
        Nulls = new NullBitmap(capacity);
    }

    ~DataBlock() {
        delete[] Data;
        delete Nulls;
    }

    DataBlock(const DataBlock&)            = delete;
    DataBlock& operator=(const DataBlock&) = delete;

    DataBlock(DataBlock&& other) noexcept
        : Data(other.Data)
        , Count(other.Count)
        , Capacity(other.Capacity)
        , Nulls(other.Nulls)
    {
        other.Data     = nullptr;
        other.Count    = 0;
        other.Capacity = 0;
        other.Nulls    = nullptr;
    }

    DataBlock& operator=(DataBlock&& other) noexcept {
        if (this != &other) {
            delete[] Data;
            delete Nulls;
            Data     = other.Data;
            Count    = other.Count;
            Capacity = other.Capacity;
            Nulls    = other.Nulls;

            other.Data     = nullptr;
            other.Count    = 0;
            other.Capacity = 0;
            other.Nulls    = nullptr;
        }
        return *this;
    }

    void Append(const T* values, sz count) {
        VOXEL_ASSERT(Count + count <= Capacity && values != nullptr,
                     "Append would overflow DataBlock capacity");
        std::memcpy(Data + Count, values, count * sizeof(T));
        Count += count;
    }

    void Append(const T* values, sz count, const bool* nullFlags) {
        VOXEL_ASSERT(Count + count <= Capacity && values != nullptr,
                     "Append would overflow DataBlock capacity");
        std::memcpy(Data + Count, values, count * sizeof(T));
        if (nullFlags && Nulls) {
            for (sz i = 0; i < count; ++i) {
                if (!nullFlags[i]) {
                    Nulls->SetNull(Count + i);
                }
            }
        }
        Count += count;
    }

    void Append(T value) {
        VOXEL_ASSERT(Count < Capacity, "Append would overflow DataBlock capacity");
        Data[Count++] = value;
    }

    void AppendNull() {
        VOXEL_ASSERT(Count < Capacity, "Append would overflow DataBlock capacity");
        Data[Count] = T{};
        if (Nulls) Nulls->SetNull(Count);
        ++Count;
    }

    void Clear() {
        Count = 0;
        if (Nulls) {
            Nulls->SetAllValid();
        }
    }

    T* GetRow(sz row) {
        return &Data[row];
    }

    const T* GetRow(sz row) const {
        return &Data[row];
    }

    T& operator[](sz row) {
        return Data[row];
    }

    const T& operator[](sz row) const {
        return Data[row];
    }

    bool IsNull(sz row) const {
        if (!Nulls) return false;
        return Nulls->IsNull(row);
    }

    void SetNull(sz row) {
        if (Nulls) Nulls->SetNull(row);
    }

    void SetValid(sz row) {
        if (Nulls) Nulls->SetValid(row);
    }

    bool HasNulls() const {
        if (!Nulls) return false;
        return Nulls->NullCount() > 0;
    }

    sz NullCountVal() const {
        if (!Nulls) return 0;
        return Nulls->NullCount();
    }

    sz ValidCount() const {
        if (!Nulls) return Count;
        return Nulls->ValidCount();
    }

    bool IsFull() const {
        return Count >= Capacity;
    }

    sz Remaining() const {
        return Capacity - Count;
    }

    void ZeroFill(sz start, sz count) {
        VOXEL_ASSERT(start + count <= Capacity,
                     "ZeroFill would overflow DataBlock capacity");
        std::memset(Data + start, 0, count * sizeof(T));
    }

    void Fill(T value, sz start, sz count) {
        VOXEL_ASSERT(start + count <= Capacity,
                     "Fill would overflow DataBlock capacity");
        for (sz i = 0; i < count; ++i) {
            Data[start + i] = value;
        }
    }

    void FillAll(T value) {
        for (sz i = 0; i < Count; ++i) Data[i] = value;
    }

    void Compact() {
        if (!Nulls || Nulls->NullCount() == 0) return;

        sz write = 0;
        for (sz read = 0; read < Count; ++read) {
            if (Nulls->IsValid(read)) {
                if (write != read) {
                    Data[write] = Data[read];
                }
                ++write;
            }
        }

        Count = write;
        Nulls->SetAllValid();
        Nulls->Resize(Capacity);
    }

    void Reorder(const i32* permutation, sz permCount) {
        VOXEL_ASSERT(permCount <= Capacity, "Permutation overflow");
        T* temp = new T[permCount];
        for (sz i = 0; i < permCount; ++i)
            temp[i] = Data[permutation[i]];
        std::memcpy(Data, temp, permCount * sizeof(T));
        delete[] temp;
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

    void Sort() {
        std::sort(Data, Data + Count);
    }

    void SortWithNulls() {
        if (!Nulls) { Sort(); return; }
        std::vector<std::pair<T, bool>> pairs(Count);
        for (sz i = 0; i < Count; ++i)
            pairs[i] = {Data[i], Nulls->IsNull(i)};
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) {
                      if (a.second != b.second) return !a.second;
                      return a.first < b.first;
                  });
        for (sz i = 0; i < Count; ++i) {
            Data[i] = pairs[i].first;
            Nulls->SetRow(i, !pairs[i].second);
        }
    }

    void Deduplicate() {
        if (Count <= 1) return;
        Sort();
        sz write = 1;
        for (sz read = 1; read < Count; ++read) {
            if (Data[read] != Data[write - 1]) {
                if (write != read) Data[write] = Data[read];
                ++write;
            }
        }
        Count = write;
    }

    Segment<T> Slice(sz offset, sz count) const {
        sz effective = (offset + count <= Count) ? count : (Count > offset ? Count - offset : 0);
        return Segment<T>(Data + offset, effective);
    }

    void CopyTo(Segment<T>& dst, sz dstOff) const {
        sz toCopy = (Count + dstOff <= dst.Capacity) ? Count : (dst.Capacity > dstOff ? dst.Capacity - dstOff : 0);
        std::memcpy(dst.DataPtr() + dstOff, Data, toCopy * sizeof(T));
    }

    void CopyFrom(const Segment<T>& src, sz srcOff) {
        sz toCopy = (srcOff + Count <= src.Count) ? Count : (src.Count > srcOff ? src.Count - srcOff : 0);
        std::memcpy(Data, src.DataPtr() + srcOff, toCopy * sizeof(T));
        Count = toCopy;
    }
};

// ============================================================================
// BlockIterator — segmented iterator over multiple DataBlock<T> instances
// ============================================================================

template<typename T>
class BlockIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = T;
    using difference_type   = isz;
    using pointer           = const T*;
    using reference         = const T&;

    BlockIterator() = default;

    BlockIterator(const std::vector<DataBlock<T>*>& blocks, sz blockIdx, sz rowIdx)
        : Blocks_(&blocks), BlockIdx_(blockIdx), RowIdx_(rowIdx)
    {
        SkipEmptyBlocks();
    }

    reference operator*() const {
        return (*Blocks_)[BlockIdx_]->operator[](RowIdx_);
    }

    pointer operator->() const {
        return &(*Blocks_)[BlockIdx_]->operator[](RowIdx_);
    }

    BlockIterator& operator++() {
        ++RowIdx_;
        if (RowIdx_ >= (*Blocks_)[BlockIdx_]->Count) {
            ++BlockIdx_;
            RowIdx_ = 0;
            SkipEmptyBlocks();
        }
        return *this;
    }

    BlockIterator operator++(int) {
        BlockIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    BlockIterator& operator+=(isz n) {
        for (isz i = 0; i < n; ++i) ++(*this);
        return *this;
    }

    bool operator==(const BlockIterator& other) const {
        return BlockIdx_ == other.BlockIdx_ && RowIdx_ == other.RowIdx_
            && Blocks_ == other.Blocks_;
    }

    bool operator!=(const BlockIterator& other) const {
        return !(*this == other);
    }

    bool IsNull() const {
        return (*Blocks_)[BlockIdx_]->IsNull(RowIdx_);
    }

    sz GlobalRowIndex() const {
        sz global = 0;
        for (sz b = 0; b < BlockIdx_ && b < Blocks_->size(); ++b) {
            global += (*Blocks_)[b]->Count;
        }
        global += RowIdx_;
        return global;
    }

    bool IsEnd() const {
        return !Blocks_ || BlockIdx_ >= Blocks_->size();
    }

private:
    const std::vector<DataBlock<T>*>* Blocks_   = nullptr;
    sz BlockIdx_ = 0;
    sz RowIdx_   = 0;

    void SkipEmptyBlocks() {
        while (Blocks_ && BlockIdx_ < Blocks_->size() &&
               (*Blocks_)[BlockIdx_]->Count == 0) {
            ++BlockIdx_;
        }
    }
};

// ============================================================================
// BlockManager — thread-safe pool of pre-allocated DataBlock instances
// ============================================================================

template<typename T>
class BlockManager {
public:
    static constexpr sz kDefaultPoolSize = 64;

    explicit BlockManager(sz prealloc = kDefaultPoolSize)
    {
        Preallocate(prealloc);
    }

    ~BlockManager() {
        std::lock_guard<std::mutex> lock(Mutex_);
        for (auto* block : FreeList_) delete block;
        for (auto* block : InUseList_) delete block;
    }

    BlockManager(const BlockManager&)            = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    DataBlock<T>* AcquireBlock() {
        std::lock_guard<std::mutex> lock(Mutex_);

        if (FreeList_.empty()) {
            Preallocate(kDefaultPoolSize / 2);
            if (FreeList_.empty()) {
                DataBlock<T>* block = new DataBlock<T>(DataBlock<T>::kDefaultBlockSize);
                InUseList_.push_back(block);
                return block;
            }
        }

        DataBlock<T>* block = FreeList_.back();
        FreeList_.pop_back();
        block->Clear();
        InUseList_.push_back(block);
        return block;
    }

    DataBlock<T>* AcquireBlock(sz capacity) {
        std::lock_guard<std::mutex> lock(Mutex_);

        DataBlock<T>* block = nullptr;
        for (sz i = 0; i < FreeList_.size(); ++i) {
            if (FreeList_[i]->Capacity >= capacity) {
                block = FreeList_[i];
                FreeList_[i] = FreeList_.back();
                FreeList_.pop_back();
                break;
            }
        }

        if (!block) {
            block = new DataBlock<T>(capacity);
        }

        block->Clear();
        InUseList_.push_back(block);
        return block;
    }

    void ReleaseBlock(DataBlock<T>* block) {
        std::lock_guard<std::mutex> lock(Mutex_);

        auto it = std::find(InUseList_.begin(), InUseList_.end(), block);
        if (it != InUseList_.end()) {
            InUseList_.erase(it);
            block->Clear();
            FreeList_.push_back(block);
        }
    }

    sz AvailableBlocks() const {
        std::lock_guard<std::mutex> lock(Mutex_);
        return FreeList_.size();
    }

    sz InUseBlocks() const {
        std::lock_guard<std::mutex> lock(Mutex_);
        return InUseList_.size();
    }

    void Preallocate(sz count) {
        std::lock_guard<std::mutex> lock(Mutex_);
        for (sz i = 0; i < count; ++i) {
            DataBlock<T>* block = new DataBlock<T>(DataBlock<T>::kDefaultBlockSize);
            FreeList_.push_back(block);
        }
    }

    void Preallocate(sz count, sz capacity) {
        std::lock_guard<std::mutex> lock(Mutex_);
        for (sz i = 0; i < count; ++i) {
            DataBlock<T>* block = new DataBlock<T>(capacity);
            FreeList_.push_back(block);
        }
    }

    void ReleaseAll() {
        std::lock_guard<std::mutex> lock(Mutex_);
        for (auto* block : InUseList_) {
            block->Clear();
            FreeList_.push_back(block);
        }
        InUseList_.clear();
    }

    void ShrinkPool() {
        std::lock_guard<std::mutex> lock(Mutex_);
        while (FreeList_.size() > kDefaultPoolSize) {
            delete FreeList_.back();
            FreeList_.pop_back();
        }
    }

    sz TotalBlocks() const {
        std::lock_guard<std::mutex> lock(Mutex_);
        return FreeList_.size() + InUseList_.size();
    }

    std::vector<DataBlock<T>*> AcquireBatch(sz count) {
        std::vector<DataBlock<T>*> result;
        result.reserve(count);
        for (sz i = 0; i < count; ++i)
            result.push_back(AcquireBlock());
        return result;
    }

    void ReleaseBatch(const std::vector<DataBlock<T>*>& blocks) {
        for (auto* b : blocks) ReleaseBlock(b);
    }

    static BlockManager<T>& Instance() {
        static BlockManager<T> instance;
        return instance;
    }

private:
    std::vector<DataBlock<T>*> FreeList_;
    std::vector<DataBlock<T>*> InUseList_;
    mutable std::mutex         Mutex_;
};

// ============================================================================
// BlockedSegment — data partitioned into fixed-size blocks for cache efficiency
// ============================================================================

template<typename T>
class BlockedSegment {
public:
    std::vector<DataBlock<T>*> Blocks;
    sz TotalCount = 0;

    BlockedSegment() = default;

    ~BlockedSegment() {
        for (auto* b : Blocks) delete b;
    }

    BlockedSegment(const BlockedSegment&)            = delete;
    BlockedSegment& operator=(const BlockedSegment&) = delete;

    BlockedSegment(BlockedSegment&& other) noexcept
        : Blocks(std::move(other.Blocks)), TotalCount(other.TotalCount)
    {
        other.Blocks.clear();
        other.TotalCount = 0;
    }

    BlockedSegment& operator=(BlockedSegment&& other) noexcept {
        if (this != &other) {
            for (auto* b : Blocks) delete b;
            Blocks = std::move(other.Blocks);
            TotalCount = other.TotalCount;
            other.Blocks.clear();
            other.TotalCount = 0;
        }
        return *this;
    }

    sz Size() const { return TotalCount; }

    T Get(sz row) const {
        sz blockIdx = 0;
        sz remaining = row;
        for (auto* block : Blocks) {
            if (remaining < block->Count) return (*block)[remaining];
            remaining -= block->Count;
        }
        return T{};
    }

    BlockIterator<T> begin() const {
        return BlockIterator<T>(Blocks, 0, 0);
    }

    BlockIterator<T> end() const {
        return BlockIterator<T>(Blocks, Blocks.size(), 0);
    }

    void AppendBlock(DataBlock<T>* block) {
        Blocks.push_back(block);
        TotalCount += block->Count;
    }

    void AppendFrom(const T* data, sz count, sz blockSize = DataBlock<T>::kDefaultBlockSize) {
        sz offset = 0;
        while (offset < count) {
            sz chunk = std::min(blockSize, count - offset);
            auto* block = new DataBlock<T>(blockSize);
            block->Append(data + offset, chunk);
            AppendBlock(block);
            offset += chunk;
        }
    }

    void Clear() {
        for (auto* b : Blocks) delete b;
        Blocks.clear();
        TotalCount = 0;
    }
};

} // namespace voxel
