#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/bytecode/opcodes.hpp"

#include <ostream>
#include <iomanip>
#include <array>
#include <cstring>
#include <atomic>

#if VOXEL_ARCH_X86_64
#include <x86intrin.h>
#endif

#if VOXEL_OS_LINUX
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace voxel {

// ============================================================================
// CycleTimer — high-resolution timestamp counter
// ============================================================================

class CycleTimer {
public:
    CycleTimer() : StartCycles_(0), EndCycles_(0) {}

    u64 Start() {
        StartCycles_ = Now();
        EndCycles_ = 0;
        return StartCycles_;
    }

    u64 Stop() {
        EndCycles_ = Now();
        return EndCycles_;
    }

    u64 Elapsed() const {
        return (EndCycles_ > StartCycles_) ? (EndCycles_ - StartCycles_) : 0;
    }

    static u64 Now() {
#if VOXEL_ARCH_X86_64
        return __rdtsc();
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<u64>(ts.tv_sec) * 1000000000ULL + static_cast<u64>(ts.tv_nsec);
#endif
    }

    static f64 ToMicroseconds(u64 cycles, f64 freqGHz) {
        if (freqGHz <= 0.0) return 0.0;
        return static_cast<f64>(cycles) / (freqGHz * 1000.0);
    }

private:
    u64 StartCycles_;
    u64 EndCycles_;
};

// ============================================================================
// InstructionProfiler — per-opcode execution and cycle counts
// ============================================================================

class InstructionProfiler {
public:
    std::array<u64, 256> OpcodeExecCounts{};
    std::array<u64, 256> OpcodeCycleCounts{};
    u64 TotalInstructions = 0;
    u64 TotalCycles       = 0;

    void RecordOpcode(Opcode op, u64 startCycles, u64 endCycles) {
        u8 idx = static_cast<u8>(op);
        u64 elapsed = (endCycles > startCycles) ? (endCycles - startCycles) : 0;
        OpcodeExecCounts[idx]++;
        OpcodeCycleCounts[idx] += elapsed;
        TotalInstructions++;
        TotalCycles += elapsed;
    }

    void Reset() {
        OpcodeExecCounts.fill(0);
        OpcodeCycleCounts.fill(0);
        TotalInstructions = 0;
        TotalCycles       = 0;
    }

    void Dump(std::ostream& os) const {
        os << "=== Instruction Profiler ===\n";
        os << std::left << std::setw(18) << "Opcode"
           << std::right << std::setw(14) << "ExecCount"
           << std::setw(14) << "Cycles"
           << std::setw(14) << "AvgCycles"
           << std::setw(12) << "PctCycles" << "\n";
        os << std::string(72, '-') << "\n";

        for (u16 i = 0; i < 256; ++i) {
            if (OpcodeExecCounts[i] == 0) continue;
            u64 cnt = OpcodeExecCounts[i];
            u64 cyc = OpcodeCycleCounts[i];
            f64 avg = static_cast<f64>(cyc) / static_cast<f64>(cnt);
            f64 pct = (TotalCycles > 0) ? (100.0 * static_cast<f64>(cyc) / static_cast<f64>(TotalCycles)) : 0.0;

            os << std::left << std::setw(18) << OpcodeName(static_cast<Opcode>(i))
               << std::right << std::setw(14) << cnt
               << std::setw(14) << cyc
               << std::setw(14) << std::fixed << std::setprecision(2) << avg
               << std::setw(11) << std::fixed << std::setprecision(1) << pct << "%" << "\n";
        }

        os << std::string(72, '-') << "\n";
        os << "Total Instructions: " << TotalInstructions << "\n";
        os << "Total Cycles:       " << TotalCycles << "\n";
        if (TotalInstructions > 0) {
            os << "Avg CPI:            " << std::fixed << std::setprecision(2)
               << (static_cast<f64>(TotalCycles) / static_cast<f64>(TotalInstructions)) << "\n";
        }
    }

    void DumpTopN(std::ostream& os, sz n) const {
        struct Entry {
            u8 idx;
            u64 cycles;
        };

        std::array<Entry, 256> entries{};
        sz count = 0;
        for (u16 i = 0; i < 256; ++i) {
            if (OpcodeCycleCounts[i] > 0 && count < n) {
                entries[count++] = {static_cast<u8>(i), OpcodeCycleCounts[i]};
            }
        }

        for (sz j = 1; j < count; ++j) {
            Entry key = entries[j];
            isz i2 = static_cast<isz>(j) - 1;
            while (i2 >= 0 && entries[static_cast<sz>(i2)].cycles < key.cycles) {
                entries[static_cast<sz>(i2 + 1)] = entries[static_cast<sz>(i2)];
                i2--;
            }
            entries[static_cast<sz>(i2 + 1)] = key;
        }

        os << "=== Top " << n << " Opcodes by Cycle Count ===\n";
        os << std::left << std::setw(6) << "Rank"
           << std::setw(18) << "Opcode"
           << std::right << std::setw(14) << "Cycles"
           << std::setw(14) << "ExecCount"
           << std::setw(14) << "AvgCycles" << "\n";
        os << std::string(66, '-') << "\n";

        for (sz i = 0; i < count; ++i) {
            u8 idx = entries[i].idx;
            os << std::left << std::setw(6) << (i + 1)
               << std::setw(18) << OpcodeName(static_cast<Opcode>(idx))
               << std::right << std::setw(14) << entries[i].cycles
               << std::setw(14) << OpcodeExecCounts[idx]
               << std::setw(14) << std::fixed << std::setprecision(2)
               << (OpcodeExecCounts[idx] > 0
                   ? static_cast<f64>(entries[i].cycles) / static_cast<f64>(OpcodeExecCounts[idx])
                   : 0.0) << "\n";
        }
    }
};

// ============================================================================
// CacheMissCounter — reads hardware performance counters
// ============================================================================

class CacheMissCounter {
public:
    u64 L1Misses = 0;
    u64 L2Misses = 0;
    u64 L3Misses = 0;

#if VOXEL_OS_LINUX
    CacheMissCounter() : FdL1_(-1), FdL2_(-1), FdL3_(-1) {
        FdL1_ = PerfOpen(
            PERF_COUNT_HW_CACHE_L1D |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        FdL2_ = -1; // L2 misses not directly available via base perf, approximate
        FdL3_ = PerfOpen(
            PERF_COUNT_HW_CACHE_LL |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
    }
#else
    CacheMissCounter() {}
#endif

    void Sample() {
#if VOXEL_OS_LINUX
        ReadPerfCount(FdL1_, L1Misses);
        ReadPerfCount(FdL2_, L2Misses);
        ReadPerfCount(FdL3_, L3Misses);
#endif
    }

    void Dump(std::ostream& os) const {
        os << "=== Cache Miss Counter ===\n";
        os << "  L1 Misses: " << L1Misses << "\n";
        os << "  L2 Misses: " << L2Misses << "\n";
        os << "  L3 Misses: " << L3Misses << "\n";
    }

private:
#if VOXEL_OS_LINUX
    int FdL1_;
    int FdL2_;
    int FdL3_;

    static int PerfOpen(u64 config) {
        struct perf_event_attr pe{};
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        int fd = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        return fd;
    }

    static void ReadPerfCount(int fd, u64& out) {
        if (fd < 0) { out = 0; return; }
        struct read_format {
            u64 value;
            u64 timeEnabled;
            u64 timeRunning;
        } rf{};
        ssize_t n = read(fd, &rf, sizeof(rf));
        if (static_cast<sz>(n) == sizeof(rf)) {
            if (rf.timeRunning > 0 && rf.timeEnabled > 0) {
                out = rf.value * rf.timeEnabled / rf.timeRunning;
            } else {
                out = rf.value;
            }
        }
    }
#endif
};

// ============================================================================
// MemoryTracker — global memory allocation tracking
// ============================================================================

class MemoryTracker {
public:
    std::atomic<sz> CurrentAllocated{0};
    std::atomic<sz> PeakAllocated{0};
    std::atomic<u64> AllocationCount{0};

    void OnAlloc(sz bytes) {
        sz prev = CurrentAllocated.fetch_add(bytes, std::memory_order_relaxed);
        sz curr = prev + bytes;
        sz peak = PeakAllocated.load(std::memory_order_relaxed);
        while (curr > peak &&
               !PeakAllocated.compare_exchange_weak(peak, curr, std::memory_order_relaxed));
        AllocationCount.fetch_add(1, std::memory_order_relaxed);
    }

    void OnFree(sz bytes) {
        CurrentAllocated.fetch_sub(bytes, std::memory_order_relaxed);
    }

    void Dump(std::ostream& os) const {
        os << "=== Memory Tracker ===\n";
        os << "  Current Allocated: " << CurrentAllocated.load() << " bytes\n";
        os << "  Peak Allocated:    " << PeakAllocated.load() << " bytes\n";
        os << "  Allocation Count:  " << AllocationCount.load() << "\n";
    }
};

} // namespace voxel
