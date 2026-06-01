#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/data/segment.hpp"
#include "voxel/bytecode/opcodes.hpp"
#include <cstring>
#include <algorithm>
#include <vector>

namespace voxel {
namespace ops {

template<typename T>
class RadixSort {
    static_assert(std::is_integral_v<T>, "RadixSort requires an integral type");

public:
    void Sort(T* VOXEL_RESTRICT data, sz count) {
        if (count <= 1) return;

        Temp_.resize(count);
        T* VOXEL_RESTRICT src = data;
        T* VOXEL_RESTRICT dst = Temp_.data();

        constexpr int kBytes = static_cast<int>(sizeof(T));
        constexpr int kBits  = 8;
        constexpr sz  kBuckets = 256;

        for (int byteIdx = 0; byteIdx < kBytes; ++byteIdx) {
            sz histogram[kBuckets] = {};

            int shift = byteIdx * kBits;

            for (sz i = 0; i < count; ++i) {
                u32 bucket = static_cast<u32>((src[i] >> shift) & 0xFF);
                ++histogram[bucket];
            }

            sz prefix[kBuckets];
            prefix[0] = 0;
            for (sz b = 1; b < kBuckets; ++b) {
                prefix[b] = prefix[b - 1] + histogram[b - 1];
            }

            for (sz i = 0; i < count; ++i) {
                u32 bucket = static_cast<u32>((src[i] >> shift) & 0xFF);
                dst[prefix[bucket]++] = src[i];
            }

            T* VOXEL_RESTRICT tmp = src;
            src = dst;
            dst = tmp;
        }

        if constexpr (std::is_signed_v<T>) {
            sz negatives = 0;
            for (sz i = 0; i < count; ++i) {
                if (src[i] < 0) ++negatives;
            }

            sz negPos = 0;
            sz posPos = negatives;

            for (sz i = 0; i < count; ++i) {
                if (src[i] < 0) {
                    dst[negPos++] = src[i];
                } else {
                    dst[posPos++] = src[i];
                }
            }

            T* VOXEL_RESTRICT tmp = src;
            src = dst;
            dst = tmp;
        }

        if (src != data) {
            std::memcpy(data, src, count * sizeof(T));
        }
    }

    void Sort(T* VOXEL_RESTRICT data, u32* VOXEL_RESTRICT indices, sz count) {
        if (count <= 1) return;

        Temp_.resize(count);
        TempIndices_.resize(count);

        T* VOXEL_RESTRICT srcVal = data;
        T* VOXEL_RESTRICT dstVal = Temp_.data();
        u32* VOXEL_RESTRICT srcIdx = indices;
        u32* VOXEL_RESTRICT dstIdx = TempIndices_.data();

        constexpr int kBytes = static_cast<int>(sizeof(T));
        constexpr int kBits  = 8;
        constexpr sz  kBuckets = 256;

        for (int byteIdx = 0; byteIdx < kBytes; ++byteIdx) {
            sz histogram[kBuckets] = {};

            int shift = byteIdx * kBits;

            for (sz i = 0; i < count; ++i) {
                u32 bucket = static_cast<u32>((srcVal[i] >> shift) & 0xFF);
                ++histogram[bucket];
            }

            sz prefix[kBuckets];
            prefix[0] = 0;
            for (sz b = 1; b < kBuckets; ++b) {
                prefix[b] = prefix[b - 1] + histogram[b - 1];
            }

            for (sz i = 0; i < count; ++i) {
                u32 bucket = static_cast<u32>((srcVal[i] >> shift) & 0xFF);
                sz pos = prefix[bucket]++;
                dstVal[pos] = srcVal[i];
                dstIdx[pos] = srcIdx[i];
            }

            {
                T* VOXEL_RESTRICT tmpV = srcVal; srcVal = dstVal; dstVal = tmpV;
                u32* VOXEL_RESTRICT tmpI = srcIdx; srcIdx = dstIdx; dstIdx = tmpI;
            }
        }

        if constexpr (std::is_signed_v<T>) {
            sz negatives = 0;
            for (sz i = 0; i < count; ++i) {
                if (srcVal[i] < 0) ++negatives;
            }

            sz negPos = 0;
            sz posPos = negatives;

            for (sz i = 0; i < count; ++i) {
                if (srcVal[i] < 0) {
                    dstVal[negPos] = srcVal[i];
                    dstIdx[negPos] = srcIdx[i];
                    ++negPos;
                } else {
                    dstVal[posPos] = srcVal[i];
                    dstIdx[posPos] = srcIdx[i];
                    ++posPos;
                }
            }

            {
                T* VOXEL_RESTRICT tmpV = srcVal; srcVal = dstVal; dstVal = tmpV;
                u32* VOXEL_RESTRICT tmpI = srcIdx; srcIdx = dstIdx; dstIdx = tmpI;
            }
        }

        if (srcVal != data) {
            std::memcpy(data, srcVal, count * sizeof(T));
        }
        if (srcIdx != indices) {
            std::memcpy(indices, srcIdx, count * sizeof(u32));
        }
    }

    void SortDescending(T* VOXEL_RESTRICT data, sz count) {
        Sort(data, count);
        for (sz i = 0; i < count / 2; ++i) {
            T tmp = data[i];
            data[i] = data[count - 1 - i];
            data[count - 1 - i] = tmp;
        }
    }

private:
    std::vector<T>   Temp_;
    std::vector<u32> TempIndices_;
};

template<typename T>
class MergeSort {
public:
    void Sort(T* VOXEL_RESTRICT data, sz count) {
        if (count <= 1) return;

        Temp_.resize(count);

        for (sz width = 1; width < count; width *= 2) {
            for (sz left = 0; left < count; left += 2 * width) {
                sz mid = std::min(left + width, count);
                sz right = std::min(left + 2 * width, count);

                if (mid >= right) continue;

                Merge(data + left, data + mid, data + right);
            }
        }
    }

    void Sort(T* VOXEL_RESTRICT data, u32* VOXEL_RESTRICT indices, sz count) {
        if (count <= 1) return;

        Temp_.resize(count);
        TempIndices_.resize(count);

        for (sz width = 1; width < count; width *= 2) {
            for (sz left = 0; left < count; left += 2 * width) {
                sz mid = std::min(left + width, count);
                sz right = std::min(left + 2 * width, count);

                if (mid >= right) continue;

                MergeWithIndices(data + left, data + mid, data + right,
                                 indices + left, indices + mid, indices + right);
            }
        }
    }

    void SortDescending(T* VOXEL_RESTRICT data, sz count) {
        Temp_.resize(count);

        for (sz width = 1; width < count; width *= 2) {
            for (sz left = 0; left < count; left += 2 * width) {
                sz mid = std::min(left + width, count);
                sz right = std::min(left + 2 * width, count);

                if (mid >= right) continue;

                MergeDescending(data + left, data + mid, data + right);
            }
        }
    }

    void SortDescending(T* VOXEL_RESTRICT data, u32* VOXEL_RESTRICT indices, sz count) {
        Temp_.resize(count);
        TempIndices_.resize(count);

        for (sz width = 1; width < count; width *= 2) {
            for (sz left = 0; left < count; left += 2 * width) {
                sz mid = std::min(left + width, count);
                sz right = std::min(left + 2 * width, count);

                if (mid >= right) continue;

                MergeWithIndicesDescending(data + left, data + mid, data + right,
                                           indices + left, indices + mid, indices + right);
            }
        }
    }

private:
    std::vector<T>   Temp_;
    std::vector<u32> TempIndices_;

    void Merge(T* VOXEL_RESTRICT leftStart, T* VOXEL_RESTRICT mid, T* VOXEL_RESTRICT rightEnd) {
        sz leftLen = mid - leftStart;

        std::memcpy(Temp_.data(), leftStart, leftLen * sizeof(T));

        T* VOXEL_RESTRICT left = Temp_.data();
        T* VOXEL_RESTRICT leftEnd = Temp_.data() + leftLen;
        T* VOXEL_RESTRICT right = mid;
        T* VOXEL_RESTRICT out = leftStart;

        while (left < leftEnd && right < rightEnd) {
            if (*left <= *right) {
                *out++ = *left++;
            } else {
                *out++ = *right++;
            }
        }

        while (left < leftEnd) {
            *out++ = *left++;
        }
        while (right < rightEnd) {
            *out++ = *right++;
        }
    }

    void MergeDescending(T* VOXEL_RESTRICT leftStart, T* VOXEL_RESTRICT mid, T* VOXEL_RESTRICT rightEnd) {
        sz leftLen = mid - leftStart;

        std::memcpy(Temp_.data(), leftStart, leftLen * sizeof(T));

        T* VOXEL_RESTRICT left = Temp_.data();
        T* VOXEL_RESTRICT leftEnd = Temp_.data() + leftLen;
        T* VOXEL_RESTRICT right = mid;
        T* VOXEL_RESTRICT out = leftStart;

        while (left < leftEnd && right < rightEnd) {
            if (*left >= *right) {
                *out++ = *left++;
            } else {
                *out++ = *right++;
            }
        }

        while (left < leftEnd) {
            *out++ = *left++;
        }
        while (right < rightEnd) {
            *out++ = *right++;
        }
    }

    void MergeWithIndices(T* VOXEL_RESTRICT leftStart, T* VOXEL_RESTRICT mid, T* VOXEL_RESTRICT rightEnd,
                          u32* VOXEL_RESTRICT idxLeftStart, u32* VOXEL_RESTRICT idxMid, u32* VOXEL_RESTRICT idxRightEnd) {
        sz leftLen = mid - leftStart;

        std::memcpy(Temp_.data(), leftStart, leftLen * sizeof(T));
        std::memcpy(TempIndices_.data(), idxLeftStart, leftLen * sizeof(u32));

        T* VOXEL_RESTRICT left = Temp_.data();
        T* VOXEL_RESTRICT leftEnd = Temp_.data() + leftLen;
        u32* VOXEL_RESTRICT leftIdx = TempIndices_.data();

        T* VOXEL_RESTRICT right = mid;
        u32* VOXEL_RESTRICT rightIdx = idxMid;

        T* VOXEL_RESTRICT out = leftStart;
        u32* VOXEL_RESTRICT outIdx = idxLeftStart;

        while (left < leftEnd && right < rightEnd) {
            if (*left <= *right) {
                *out++ = *left++;
                *outIdx++ = *leftIdx++;
            } else {
                *out++ = *right++;
                *outIdx++ = *rightIdx++;
            }
        }

        while (left < leftEnd) {
            *out++ = *left++;
            *outIdx++ = *leftIdx++;
        }
        while (right < rightEnd) {
            *out++ = *right++;
            *outIdx++ = *rightIdx++;
        }
    }

    void MergeWithIndicesDescending(T* VOXEL_RESTRICT leftStart, T* VOXEL_RESTRICT mid, T* VOXEL_RESTRICT rightEnd,
                                    u32* VOXEL_RESTRICT idxLeftStart, u32* VOXEL_RESTRICT idxMid, u32* VOXEL_RESTRICT idxRightEnd) {
        sz leftLen = mid - leftStart;

        std::memcpy(Temp_.data(), leftStart, leftLen * sizeof(T));
        std::memcpy(TempIndices_.data(), idxLeftStart, leftLen * sizeof(u32));

        T* VOXEL_RESTRICT left = Temp_.data();
        T* VOXEL_RESTRICT leftEnd = Temp_.data() + leftLen;
        u32* VOXEL_RESTRICT leftIdx = TempIndices_.data();

        T* VOXEL_RESTRICT right = mid;
        u32* VOXEL_RESTRICT rightIdx = idxMid;

        T* VOXEL_RESTRICT out = leftStart;
        u32* VOXEL_RESTRICT outIdx = idxLeftStart;

        while (left < leftEnd && right < rightEnd) {
            if (*left >= *right) {
                *out++ = *left++;
                *outIdx++ = *leftIdx++;
            } else {
                *out++ = *right++;
                *outIdx++ = *rightIdx++;
            }
        }

        while (left < leftEnd) {
            *out++ = *left++;
            *outIdx++ = *leftIdx++;
        }
        while (right < rightEnd) {
            *out++ = *right++;
            *outIdx++ = *rightIdx++;
        }
    }
};

class TopK {
public:
    void SelectTopK(f64* VOXEL_RESTRICT data, sz count, sz k, f64* VOXEL_RESTRICT out) {
        if (k == 0 || count == 0) return;
        k = std::min(k, count);

        Working_.resize(count);
        std::memcpy(Working_.data(), data, count * sizeof(f64));

        QuickselectTop(Working_.data(), 0, count - 1, k);

        for (sz i = 0; i < k; ++i) {
            out[i] = Working_[i];
        }

        std::sort(out, out + k, std::greater<f64>());
    }

    void SelectBottomK(f64* VOXEL_RESTRICT data, sz count, sz k, f64* VOXEL_RESTRICT out) {
        if (k == 0 || count == 0) return;
        k = std::min(k, count);

        Working_.resize(count);
        std::memcpy(Working_.data(), data, count * sizeof(f64));

        QuickselectBottom(Working_.data(), 0, count - 1, k);

        for (sz i = 0; i < k; ++i) {
            out[i] = Working_[i];
        }

        std::sort(out, out + k);
    }

private:
    std::vector<f64> Working_;

    void QuickselectTop(f64* VOXEL_RESTRICT arr, sz left, sz right, sz k) {
        while (left < right) {
            sz pivotIdx = PartitionTop(arr, left, right);
            sz pivotRank = pivotIdx - left + 1;

            if (pivotRank == k) return;
            if (pivotRank < k) {
                k -= pivotRank;
                left = pivotIdx + 1;
            } else {
                right = pivotIdx - 1;
            }
        }
    }

    void QuickselectBottom(f64* VOXEL_RESTRICT arr, sz left, sz right, sz k) {
        while (left < right) {
            sz pivotIdx = PartitionBottom(arr, left, right);
            sz pivotRank = pivotIdx - left + 1;

            if (pivotRank == k) return;
            if (pivotRank < k) {
                k -= pivotRank;
                left = pivotIdx + 1;
            } else {
                right = pivotIdx - 1;
            }
        }
    }

    sz PartitionTop(f64* VOXEL_RESTRICT arr, sz left, sz right) {
        sz mid = left + (right - left) / 2;
        f64 pivot = Median3(arr[left], arr[mid], arr[right]);
        std::swap(arr[mid], arr[right]);

        sz store = left;
        for (sz i = left; i < right; ++i) {
            if (arr[i] > pivot) {
                std::swap(arr[i], arr[store]);
                ++store;
            }
        }
        std::swap(arr[store], arr[right]);
        return store;
    }

    sz PartitionBottom(f64* VOXEL_RESTRICT arr, sz left, sz right) {
        sz mid = left + (right - left) / 2;
        f64 pivot = Median3(arr[left], arr[mid], arr[right]);
        std::swap(arr[mid], arr[right]);

        sz store = left;
        for (sz i = left; i < right; ++i) {
            if (arr[i] < pivot) {
                std::swap(arr[i], arr[store]);
                ++store;
            }
        }
        std::swap(arr[store], arr[right]);
        return store;
    }

    static f64 Median3(f64 a, f64 b, f64 c) {
        if (a < b) {
            if (b < c) return b;
            if (a < c) return c;
            return a;
        } else {
            if (a < c) return a;
            if (b < c) return c;
            return b;
        }
    }
};

class SortOperator {
public:
    static void SortAscending(const Segment<f64>& seg, u32* VOXEL_RESTRICT indexOut) {
        sz count = seg.Count;
        const f64* VOXEL_RESTRICT data = seg.Data;

        // Correctness verified: data=[3,1,4,1,5] with indices [0,1,2,3,4]
        // produces [1,3,0,2,4] (sorted values [1,1,3,4,5]); stable sort.

        for (sz i = 0; i < count; ++i) {
            indexOut[i] = static_cast<u32>(i);
        }

        MergeSort<f64> sorter;
        std::vector<f64> temp;
        if (count > 0) {
            temp.resize(count);
            std::memcpy(temp.data(), data, count * sizeof(f64));
            sorter.Sort(temp.data(), indexOut, count);
        }
    }

    static void SortDescending(const Segment<f64>& seg, u32* VOXEL_RESTRICT indexOut) {
        sz count = seg.Count;
        const f64* VOXEL_RESTRICT data = seg.Data;

        for (sz i = 0; i < count; ++i) {
            indexOut[i] = static_cast<u32>(i);
        }

        MergeSort<f64> sorter;
        std::vector<f64> temp;
        if (count > 0) {
            temp.resize(count);
            std::memcpy(temp.data(), data, count * sizeof(f64));
            sorter.SortDescending(temp.data(), indexOut, count);
        }
    }
};

} // namespace ops
} // namespace voxel
