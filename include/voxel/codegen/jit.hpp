#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/bytecode/opcodes.hpp"
#include "voxel/bytecode/instruction.hpp"
#include <vector>
#include <functional>
#include <memory>
#include <cstring>

#if VOXEL_OS_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

namespace voxel {
namespace codegen {

// ============================================================================
// JitFunction — opaque handle to an executable code fragment
// ============================================================================

struct JitFunction {
    void*  CodePtr;  // pointer to executable code block
    sz     CodeSize; // size of the executable block in bytes
    using EntryFn = void (*)(void* regfile, void* segmentsBase, u64* segmentCounts);
    EntryFn Entry;

    JitFunction()
        : CodePtr(nullptr)
        , CodeSize(0)
        , Entry(nullptr)
    {}

    bool IsValid() const { return CodePtr != nullptr && Entry != nullptr; }
};

// ============================================================================
// JitCompiler — abstract interface for bytecode-to-machine-code translation
// ============================================================================

class JitCompiler {
public:
    JitCompiler()          = default;
    virtual ~JitCompiler() = default;

    JitCompiler(const JitCompiler&)            = delete;
    JitCompiler& operator=(const JitCompiler&) = delete;

    virtual bool Compile(const u32* bytecode, sz bytecodeSize, JitFunction& out) = 0;

    virtual void Release(JitFunction& func) = 0;
};

// ============================================================================
// JitMemoryManager — platform-abstracted executable memory allocation
// ============================================================================

class JitMemoryManager {
public:
    static constexpr sz kExecPageSize = 4096;

    JitMemoryManager()  = default;
    ~JitMemoryManager() = default;

    JitMemoryManager(const JitMemoryManager&)            = delete;
    JitMemoryManager& operator=(const JitMemoryManager&) = delete;

    void* Allocate(sz bytes)
    {
        sz allocSize = (bytes + kExecPageSize - 1) & ~(kExecPageSize - 1);
#if VOXEL_OS_WINDOWS
        void* ptr = VirtualAlloc(nullptr, allocSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        void* ptr = mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) ptr = nullptr;
#endif
        return ptr;
    }

    void Deallocate(void* ptr, sz bytes)
    {
        if (!ptr) return;
        sz allocSize = (bytes + kExecPageSize - 1) & ~(kExecPageSize - 1);
#if VOXEL_OS_WINDOWS
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        munmap(ptr, allocSize);
#endif
    }

    bool MakeExecutable(void* ptr, sz bytes)
    {
        if (!ptr) return false;
        sz pageSize = kExecPageSize;
        sz alignedSize = (bytes + pageSize - 1) & ~(pageSize - 1);
#if VOXEL_OS_WINDOWS
        DWORD oldProt;
        return VirtualProtect(ptr, alignedSize, PAGE_EXECUTE_READ, &oldProt) != 0;
#else
        return mprotect(ptr, alignedSize, PROT_READ | PROT_EXEC) == 0;
#endif
    }

    bool MakeWritable(void* ptr, sz bytes)
    {
        if (!ptr) return false;
        sz pageSize = kExecPageSize;
        sz alignedSize = (bytes + pageSize - 1) & ~(pageSize - 1);
#if VOXEL_OS_WINDOWS
        DWORD oldProt;
        return VirtualProtect(ptr, alignedSize, PAGE_READWRITE, &oldProt) != 0;
#else
        return mprotect(ptr, alignedSize, PROT_READ | PROT_WRITE) == 0;
#endif
    }

private:
    static constexpr sz kMaxAllocation = 64 * 1024 * 1024; // 64 MiB
};

// ============================================================================
// JitCache — simple LRU cache for compiled bytecode fragments
// ============================================================================

class JitCache {
public:
    static constexpr sz kDefaultCapacity = 256;

    struct Entry {
        u64         Hash;
        JitFunction Func;
        u64         LastAccess;
    };

    explicit JitCache(sz capacity = kDefaultCapacity)
        : Capacity_(capacity)
        , Clock_(0)
    {
        Entries_.reserve(capacity);
    }

    ~JitCache()
    {
        Entries_.clear();
    }

    JitCache(const JitCache&)            = delete;
    JitCache& operator=(const JitCache&) = delete;

    JitFunction* Lookup(u64 bytecodeHash)
    {
        ++Clock_;
        for (auto& e : Entries_) {
            if (e.Hash == bytecodeHash) {
                e.LastAccess = Clock_;
                return &e.Func;
            }
        }
        return nullptr;
    }

    void Insert(u64 bytecodeHash, JitFunction func)
    {
        ++Clock_;
        Entry e;
        e.Hash       = bytecodeHash;
        e.Func       = func;
        e.LastAccess = Clock_;
        Entries_.push_back(e);
        if (Entries_.size() > Capacity_) Evict();
    }

    void Evict()
    {
        if (Entries_.empty()) return;
        sz   lruIdx  = 0;
        u64  minTime = Entries_[0].LastAccess;
        for (sz i = 1; i < Entries_.size(); ++i) {
            if (Entries_[i].LastAccess < minTime) {
                minTime = Entries_[i].LastAccess;
                lruIdx  = i;
            }
        }
        Entries_[lruIdx] = Entries_.back();
        Entries_.pop_back();
    }

    sz Size() const     { return Entries_.size(); }
    sz Capacity() const { return Capacity_; }

    void Clear()
    {
        Entries_.clear();
        Clock_ = 0;
    }

private:
    std::vector<Entry> Entries_;
    sz                 Capacity_;
    u64                Clock_;
};

// ============================================================================
// Factory — returns the platform-appropriate JIT compiler backend
// ============================================================================

std::unique_ptr<JitCompiler> CreateJitCompiler();

} // namespace codegen
} // namespace voxel
