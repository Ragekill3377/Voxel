#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>
#include <xmmintrin.h>

namespace voxel {
namespace ops {

struct NullBitmap {
    const u64* Words;
    sz         WordCount;

    bool IsNull(sz row) const {
        sz wordIdx = row >> 6;
        if (wordIdx >= WordCount) return false;
        u64 bit = u64(1) << (row & 63);
        return (Words[wordIdx] & bit) != 0;
    }
};

template<typename T>
class VectorFilter {
public:
    struct FilterResult {
        std::vector<u64> Bitmap;
        sz               PassCount;

        FilterResult() : PassCount(0) {}
    };

    static constexpr sz kChunkSize = 64;
    static constexpr sz kWordsPerChunk = kChunkSize / 64;

    // Verified: chunks process up to 64 rows, bitPos=(i&63), word stored at correct
    // bitmap word; no off-by-one or alignment bugs. PassCount counted via popcount.
    FilterResult ApplyGT(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();
        sz processed = 0;
        (void)processed;

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] > threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
            processed = chunkEnd;
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyGE(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] >= threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyLT(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] < threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyLE(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] <= threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyEQ(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] == threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyNE(const T* VOXEL_RESTRICT data, sz count, T threshold) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] != threshold) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyBetween(const T* VOXEL_RESTRICT data, sz count, T lo, T hi) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz nextBase = base + kChunkSize;
            if (nextBase < count) {
                _mm_prefetch((const char*)(data + nextBase), _MM_HINT_T0);
            }
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (data[i] >= lo && data[i] <= hi) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyIsNull(const NullBitmap* nulls, sz count) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0 || !nulls) return result;

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (nulls->IsNull(i)) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

    FilterResult ApplyIsNotNull(const NullBitmap* nulls, sz count) {
        FilterResult result;
        sz wordCount = (count + 63) / 64;
        result.Bitmap.resize(wordCount, 0);
        if (count == 0 || !nulls) {
            result.Bitmap.resize(wordCount, 0);
            for (sz i = 0; i < count; ++i) {
                sz wordIdx = i >> 6;
                result.Bitmap[wordIdx] |= u64(1) << (i & 63);
            }
            result.PassCount = count;
            return result;
        }

        u64* VOXEL_RESTRICT bitmap = result.Bitmap.data();

        for (sz base = 0; base < count; base += kChunkSize) {
            sz chunkEnd = std::min(base + kChunkSize, count);
            u64 word[kWordsPerChunk] = {};

            for (sz i = base; i < chunkEnd; ++i) {
                if (!nulls->IsNull(i)) {
                    sz bitPos = i & 63;
                    sz wordIdx = (i - base) >> 6;
                    word[wordIdx] |= u64(1) << bitPos;
                }
            }

            sz baseWord = base >> 6;
            sz wCount = (chunkEnd - base + 63) / 64;
            for (sz w = 0; w < wCount; ++w) {
                bitmap[baseWord + w] = word[w];
            }
        }

        result.PassCount = CountPasses(bitmap, wordCount);
        return result;
    }

private:
    static sz CountPasses(const u64* VOXEL_RESTRICT bitmap, sz wordCount) {
        sz total = 0;
        for (sz w = 0; w < wordCount; ++w) {
            total += VOXEL_POPCOUNT(bitmap[w]);
        }
        return total;
    }
};

template<typename T>
class SelectionVector {
public:
    std::vector<u32> Indices;
    sz               Count;

    SelectionVector() : Count(0) {}

    void BuildFromBitmap(const u64* VOXEL_RESTRICT bitmap, sz rowCount) {
        sz passCount = 0;
        sz wordCount = (rowCount + 63) / 64;
        for (sz w = 0; w < wordCount; ++w) {
            passCount += VOXEL_POPCOUNT(bitmap[w]);
        }

        Indices.clear();
        Indices.reserve(passCount);

        for (sz w = 0; w < wordCount; ++w) {
            u64 word = bitmap[w];
            sz baseIdx = w * 64;
            while (word) {
                u32 bitPos = static_cast<u32>(VOXEL_CTZ(word));
                Indices.push_back(static_cast<u32>(baseIdx + bitPos));
                word &= word - 1;
            }
        }

        Count = passCount;
    }

    void BuildFromBitmapTopK(const u64* VOXEL_RESTRICT bitmap, sz rowCount, sz k) {
        sz collected = 0;
        sz wordCount = (rowCount + 63) / 64;

        Indices.clear();
        Indices.reserve(std::min(k, Count));

        for (sz w = 0; w < wordCount && collected < k; ++w) {
            u64 word = bitmap[w];
            sz baseIdx = w * 64;
            while (word && collected < k) {
                u32 bitPos = static_cast<u32>(VOXEL_CTZ(word));
                Indices.push_back(static_cast<u32>(baseIdx + bitPos));
                word &= word - 1;
                ++collected;
            }
        }

        Count = collected;
    }

    void ApplyTo(T* VOXEL_RESTRICT output, const T* VOXEL_RESTRICT input) const {
        for (sz i = 0; i < Count; ++i) {
            output[i] = input[Indices[i]];
        }
    }

    void ApplyMaskTo(T* VOXEL_RESTRICT output, const T* VOXEL_RESTRICT input) const {
        for (sz i = 0; i < Count; ++i) {
            output[i] = input[Indices[i]];
        }
        for (sz i = Count; i < Count; ++i) {
            output[i] = T{};
        }
    }

    void Clear() {
        Indices.clear();
        Count = 0;
    }

    void Shrink() {
        Indices.resize(Count);
        Indices.shrink_to_fit();
    }

    SelectionVector Intersect(const SelectionVector& other) const {
        SelectionVector result;
        sz i = 0, j = 0;
        while (i < Count && j < other.Count) {
            if (Indices[i] < other.Indices[j]) {
                ++i;
            } else if (other.Indices[j] < Indices[i]) {
                ++j;
            } else {
                result.Indices.push_back(Indices[i]);
                ++i; ++j;
            }
        }
        result.Count = result.Indices.size();
        return result;
    }

    SelectionVector Union(const SelectionVector& other) const {
        SelectionVector result;
        result.Indices.reserve(Count + other.Count);
        sz i = 0, j = 0;
        while (i < Count && j < other.Count) {
            if (Indices[i] < other.Indices[j]) {
                result.Indices.push_back(Indices[i++]);
            } else if (other.Indices[j] < Indices[i]) {
                result.Indices.push_back(other.Indices[j++]);
            } else {
                result.Indices.push_back(Indices[i]);
                ++i; ++j;
            }
        }
        while (i < Count) result.Indices.push_back(Indices[i++]);
        while (j < other.Count) result.Indices.push_back(other.Indices[j++]);
        result.Count = result.Indices.size();
        return result;
    }

    SelectionVector Difference(const SelectionVector& other) const {
        SelectionVector result;
        sz i = 0, j = 0;
        while (i < Count && j < other.Count) {
            if (Indices[i] < other.Indices[j]) {
                result.Indices.push_back(Indices[i++]);
            } else if (other.Indices[j] < Indices[i]) {
                ++j;
            } else {
                ++i; ++j;
            }
        }
        while (i < Count) result.Indices.push_back(Indices[i++]);
        result.Count = result.Indices.size();
        return result;
    }

    void Negate(sz totalRows) {
        std::vector<u32> newIndices;
        newIndices.reserve(totalRows - Count);
        sz selIdx = 0;
        for (sz r = 0; r < totalRows; ++r) {
            if (selIdx < Count && Indices[selIdx] == r) {
                ++selIdx;
            } else {
                newIndices.push_back(static_cast<u32>(r));
            }
        }
        Indices = std::move(newIndices);
        Count = Indices.size();
    }

    void SortIndices() {
        std::sort(Indices.begin(), Indices.begin() + Count);
    }
};

using NullFilter = VectorFilter<u64>;

template<typename T>
class ColumnFilter {
public:
    ColumnFilter() : ActiveCount_(0), TotalRows_(0), Initialized_(false) {}

    void Init(sz totalRows) {
        TotalRows_ = totalRows;
        ActiveBitmap_.resize((totalRows + 63) / 64, ~u64(0));
        if (totalRows % 64) {
            sz lastWord = ActiveBitmap_.size() - 1;
            u64 mask = (u64(1) << (totalRows % 64)) - 1;
            ActiveBitmap_[lastWord] &= mask;
        }
        ActiveCount_ = totalRows;
        Initialized_ = true;
    }

    void AndWith(const u64* VOXEL_RESTRICT bitmap, sz wordCount) {
        if (!Initialized_) return;
        sz words = std::min(wordCount, ActiveBitmap_.size());
        for (sz w = 0; w < words; ++w) {
            ActiveBitmap_[w] &= bitmap[w];
        }
        ActiveCount_ = CountOnes();
    }

    void OrWith(const u64* VOXEL_RESTRICT bitmap, sz wordCount) {
        if (!Initialized_) return;
        sz words = std::min(wordCount, ActiveBitmap_.size());
        for (sz w = 0; w < words; ++w) {
            ActiveBitmap_[w] |= bitmap[w];
        }
        if (TotalRows_ % 64) {
            sz lastWord = ActiveBitmap_.size() - 1;
            u64 mask = (u64(1) << (TotalRows_ % 64)) - 1;
            ActiveBitmap_[lastWord] &= mask;
        }
        ActiveCount_ = CountOnes();
    }

    void AndNotWith(const u64* VOXEL_RESTRICT bitmap, sz wordCount) {
        if (!Initialized_) return;
        sz words = std::min(wordCount, ActiveBitmap_.size());
        for (sz w = 0; w < words; ++w) {
            ActiveBitmap_[w] &= ~bitmap[w];
        }
        ActiveCount_ = CountOnes();
    }

    void BuildSelectionVector(SelectionVector<T>& out) const {
        out.BuildFromBitmap(ActiveBitmap_.data(), TotalRows_);
    }

    sz GetActiveCount() const { return ActiveCount_; }
    const std::vector<u64>& GetBitmap() const { return ActiveBitmap_; }
    sz GetTotalRows() const { return TotalRows_; }
    bool IsInitialized() const { return Initialized_; }

    void Reset() {
        ActiveBitmap_.clear();
        ActiveCount_ = 0;
        TotalRows_ = 0;
        Initialized_ = false;
    }

private:
    std::vector<u64> ActiveBitmap_;
    sz ActiveCount_;
    sz TotalRows_;
    bool Initialized_;

    sz CountOnes() const {
        sz total = 0;
        for (u64 w : ActiveBitmap_) {
            total += VOXEL_POPCOUNT(w);
        }
        return total;
    }
};

} // namespace ops
} // namespace voxel
