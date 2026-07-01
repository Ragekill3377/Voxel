#pragma once

#include "voxel/core/types.hpp"
#include "voxel/data/segment.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace voxel {

template<typename T>
class MmapSegment {
    int   fd_;
    void* mmap_addr_;
    sz    file_size_;
    sz    elem_count_;

public:
    MmapSegment() : fd_(-1), mmap_addr_(nullptr), file_size_(0), elem_count_(0) {}

    explicit MmapSegment(const char* filepath) {
        fd_ = open(filepath, O_RDONLY);
        if (fd_ < 0) { mmap_addr_ = nullptr; elem_count_ = 0; return; }
        struct stat st;
        fstat(fd_, &st);
        file_size_ = static_cast<sz>(st.st_size);
        elem_count_ = file_size_ / sizeof(T);
        mmap_addr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mmap_addr_ == MAP_FAILED) { mmap_addr_ = nullptr; elem_count_ = 0; }
    }

    ~MmapSegment() {
        if (mmap_addr_ && mmap_addr_ != MAP_FAILED) munmap(mmap_addr_, file_size_);
        if (fd_ >= 0) close(fd_);
    }

    MmapSegment(const MmapSegment&) = delete;
    MmapSegment& operator=(const MmapSegment&) = delete;

    MmapSegment(MmapSegment&& other) noexcept
        : fd_(other.fd_), mmap_addr_(other.mmap_addr_), file_size_(other.file_size_), elem_count_(other.elem_count_) {
        other.fd_ = -1; other.mmap_addr_ = nullptr; other.elem_count_ = 0;
    }

    Segment<T> view() const { return Segment<T>(static_cast<T*>(mmap_addr_), elem_count_); }
    const T* data() const { return static_cast<const T*>(mmap_addr_); }
    sz size() const { return elem_count_; }
    bool valid() const { return mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED; }
};

} // namespace voxel
