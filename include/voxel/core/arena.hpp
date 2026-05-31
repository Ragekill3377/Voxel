#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include <vector>
#include <new>
#include <mutex>
#include <atomic>
#include <ostream>

namespace voxel {

// ============================================================================
// Arena — multi-tier linear bump allocator with optional thread-safety
// ============================================================================

class Arena {
public:
    static constexpr sz kDefaultBlockSize = 65536; // 64 KiB
    static constexpr sz kAlignment       = 64;
    static constexpr sz kMaxBlockSize    = 1 << 24; // 16 MiB

    struct Metrics {
        std::atomic<u64> TotalAllocated{0};
        std::atomic<u64> PeakAllocated{0};
        std::atomic<u64> AllocationCount{0};
        std::atomic<u64> BlockCount{0};
        std::atomic<u64> ReallocationCount{0};
        std::atomic<u64> CacheLineMisaligned{0};

        void RecordAllocation(sz bytes) {
            u64 prev = TotalAllocated.fetch_add(bytes, std::memory_order_relaxed);
            u64 curr = prev + bytes;
            u64 peak = PeakAllocated.load(std::memory_order_relaxed);
            while (curr > peak && !PeakAllocated.compare_exchange_weak(peak, curr, std::memory_order_relaxed));
            AllocationCount.fetch_add(1, std::memory_order_relaxed);
        }

        void RecordBlock() { BlockCount.fetch_add(1, std::memory_order_relaxed); }
        void RecordReallocation() { ReallocationCount.fetch_add(1, std::memory_order_relaxed); }

        void Dump(std::ostream& os) const {
            os << "Arena Metrics:\n"
               << "  Total allocated:   " << TotalAllocated.load() << " bytes\n"
               << "  Peak allocated:    " << PeakAllocated.load() << " bytes\n"
               << "  Allocation count:  " << AllocationCount.load() << "\n"
               << "  Block count:       " << BlockCount.load() << "\n"
               << "  Reallocations:     " << ReallocationCount.load() << "\n"
               << "  Cache misaligned:  " << CacheLineMisaligned.load() << "\n";
        }
    };

    explicit Arena(sz blockSize = kDefaultBlockSize, bool threadSafe = false)
        : Bump_(nullptr), Limit_(nullptr), BlockSize_(blockSize), ThreadSafe_(threadSafe)
    {
        Grow(blockSize);
    }

    ~Arena() {
        for (auto& b : Blocks_)
            ::operator delete(b.Memory, std::align_val_t{kAlignment});
    }

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&)                 = delete;
    Arena& operator=(Arena&&)      = delete;

    void* Alloc(sz bytes) {
        bytes = (bytes + 7) & ~7ull;
        if (ThreadSafe_) {
            std::lock_guard<std::mutex> lock(Mutex_);
            return AllocImpl(bytes);
        }
        return AllocImpl(bytes);
    }

    template<typename T>
    T* AllocMany(sz count) {
        return static_cast<T*>(Alloc(count * sizeof(T)));
    }

    template<typename T>
    T* AllocAligned(sz count, sz alignment) {
        // alignment must be power of 2 and > kAlignment
        VOXEL_ASSERT((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
        sz totalBytes = count * sizeof(T);
        sz padded = totalBytes + alignment; // room for alignment padding
        u8* raw = static_cast<u8*>(Alloc(padded));
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
        uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<T*>(aligned);
    }

    void Reset() {
        if (ThreadSafe_) std::lock_guard<std::mutex> lock(Mutex_);
        for (auto& b : Blocks_)
            ::operator delete(b.Memory, std::align_val_t{kAlignment});
        Blocks_.clear();
        Bump_  = nullptr;
        Limit_ = nullptr;
        Grow(BlockSize_);
        Metrics_.TotalAllocated.store(0, std::memory_order_relaxed);
        Metrics_.PeakAllocated.store(0, std::memory_order_relaxed);
        Metrics_.AllocationCount.store(0, std::memory_order_relaxed);
        Metrics_.BlockCount.store(0, std::memory_order_relaxed);
        Metrics_.ReallocationCount.store(0, std::memory_order_relaxed);
        Metrics_.CacheLineMisaligned.store(0, std::memory_order_relaxed);
    }

    sz Used() const {
        sz total = 0;
        for (auto& b : Blocks_) total += b.Size;
        if (Bump_ && Limit_) total -= static_cast<sz>(Limit_ - Bump_);
        return total;
    }

    sz Capacity() const {
        sz total = 0;
        for (auto& b : Blocks_) total += b.Size;
        return total;
    }

    const Metrics& GetMetrics() const { return Metrics_; }

private:
    struct Block {
        u8*    Memory;
        size_t Size;
    };

    std::vector<Block> Blocks_;
    u8*    Bump_;
    u8*    Limit_;
    sz     BlockSize_;
    Metrics Metrics_;
    bool   ThreadSafe_;
    std::mutex Mutex_;

    void* AllocImpl(sz bytes) {
        if (!Bump_ || Bump_ + bytes > Limit_) Grow(bytes);
        void* p = Bump_;
        Bump_ += bytes;
        Metrics_.RecordAllocation(bytes);
        return p;
    }

    void Grow(sz minBytes) {
        sz sz = std::max(minBytes, BlockSize_);
        sz = std::min(sz, kMaxBlockSize);
        sz = (sz + kAlignment - 1) & ~(kAlignment - 1);
        u8* mem = static_cast<u8*>(::operator new(sz, std::align_val_t{kAlignment}));
        Blocks_.push_back({mem, sz});
        Bump_  = mem;
        Limit_ = mem + sz;
        Metrics_.RecordBlock();
    }
};

// ============================================================================
// ScratchArena — thread-local fast-path allocator, resets between queries
// ============================================================================

class ScratchArena {
public:
    static constexpr sz kScratchSize = 1 << 20; // 1 MiB

    ScratchArena() : Arena_(kScratchSize) {}

    void* Alloc(sz bytes) { return Arena_.Alloc(bytes); }

    template<typename T>
    T* AllocMany(sz count) { return Arena_.AllocMany<T>(count); }

    void Reset() { Arena_.Reset(); }

    const Arena::Metrics& GetMetrics() const { return Arena_.GetMetrics(); }

private:
    Arena Arena_;
};

// ============================================================================
// ThreadLocalArena — per-thread arena with TLS lifetime
// ============================================================================

class ThreadLocalArena {
public:
    static constexpr sz kDefaultSize = 1 << 20; // 1 MiB

    static ThreadLocalArena& Instance() {
        thread_local ThreadLocalArena instance;
        return instance;
    }

    void* Alloc(sz bytes) { return Arena_.Alloc(bytes); }
    template<typename T> T* AllocMany(sz count) { return Arena_.AllocMany<T>(count); }
    void Reset() { Arena_.Reset(); }

private:
    ThreadLocalArena() : Arena_(kDefaultSize) {}
    Arena Arena_;
};

// ============================================================================
// PageAllocator — page-granularity allocation for columnar data
// ============================================================================

class PageAllocator {
public:
    static constexpr sz kDefaultPageSize = 4096;
    static constexpr sz kHugePage2MB    = 2097152;

    explicit PageAllocator(sz pageSize = kDefaultPageSize)
        : PageSize_(pageSize) {}

    void* Allocate(sz bytes) {
        sz pages = (bytes + PageSize_ - 1) / PageSize_;
        sz total = pages * PageSize_;
        void* mem = ::operator new(total, std::align_val_t{PageSize_});
        std::memset(mem, 0, total);
        Metrics_.RecordAllocation(total);
        return mem;
    }

    void Deallocate(void* ptr, sz bytes) {
        ::operator delete(ptr, std::align_val_t{PageSize_});
    }

    const Arena::Metrics& GetMetrics() const { return Metrics_; }

private:
    sz PageSize_;
    Arena::Metrics Metrics_;
};

} // namespace voxel
