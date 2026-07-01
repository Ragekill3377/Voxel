#pragma once

#include "voxel/core/types.hpp"
#include "voxel/data/segment.hpp"
#include <fstream>
#include <vector>

namespace voxel {

template<typename T>
class ChunkedReader {
    std::ifstream file_;
    sz chunk_size_;
    std::vector<T> buffer_;

public:
    ChunkedReader(const char* filepath, sz chunkSize = 1'000'000)
        : chunk_size_(chunkSize) {
        file_.open(filepath, std::ios::binary);
        if (file_.good()) buffer_.resize(chunk_size_);
    }

    bool NextChunk(Segment<T>& out) {
        if (!file_.good()) return false;
        file_.read(reinterpret_cast<char*>(buffer_.data()), chunk_size_ * sizeof(T));
        sz count = static_cast<sz>(file_.gcount()) / sizeof(T);
        if (count == 0) return false;
        out = Segment<T>(buffer_.data(), count);
        return true;
    }

    void Reset() { file_.clear(); file_.seekg(0); }
    bool IsOpen() const { return file_.is_open(); }
};

} // namespace voxel
