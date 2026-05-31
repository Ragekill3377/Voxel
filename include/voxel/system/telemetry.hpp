#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

#include <vector>
#include <string>
#include <mutex>
#include <ostream>
#include <iomanip>
#include <ctime>

namespace voxel {

// ============================================================================
// TelemetrySnapshot — single point-in-time metrics capture
// ============================================================================

struct TelemetrySnapshot {
    u64 TimestampNs         = 0;
    u64 InstructionsExecuted = 0;
    u64 CyclesElapsed       = 0;
    u64 BytesProcessed      = 0;
    u64 RowsProcessed       = 0;
    sz  MemoryUsed          = 0;
    u64 CacheMisses         = 0;
    f64 InstructionsPerCycle = 0.0;
    f64 BytesPerSecond      = 0.0;
    f64 RowsPerSecond       = 0.0;
    u32 ActiveThreads       = 0;
    u32 QueuedTasks         = 0;
};

// ============================================================================
// TelemetryCollector — thread-safe collection and export
// ============================================================================

class TelemetryCollector {
public:
    void Record(const TelemetrySnapshot& snap) {
        std::lock_guard<std::mutex> lock(Mutex_);
        Current_ = snap;
        History_.push_back(snap);
        if (History_.size() > kMaxHistory) {
            History_.erase(History_.begin());
        }
    }

    TelemetrySnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(Mutex_);
        return Current_;
    }

    void ExportJSON(std::ostream& os) const {
        std::lock_guard<std::mutex> lock(Mutex_);
        os << "[\n";
        for (sz i = 0; i < History_.size(); ++i) {
            const auto& s = History_[i];
            os << "  {\n";
            os << "    \"timestamp_ns\": " << s.TimestampNs << ",\n";
            os << "    \"instructions_executed\": " << s.InstructionsExecuted << ",\n";
            os << "    \"cycles_elapsed\": " << s.CyclesElapsed << ",\n";
            os << "    \"bytes_processed\": " << s.BytesProcessed << ",\n";
            os << "    \"rows_processed\": " << s.RowsProcessed << ",\n";
            os << "    \"memory_used\": " << s.MemoryUsed << ",\n";
            os << "    \"cache_misses\": " << s.CacheMisses << ",\n";
            os << "    \"instructions_per_cycle\": " << std::fixed << std::setprecision(4) << s.InstructionsPerCycle << ",\n";
            os << "    \"bytes_per_second\": " << std::fixed << std::setprecision(2) << s.BytesPerSecond << ",\n";
            os << "    \"rows_per_second\": " << std::fixed << std::setprecision(2) << s.RowsPerSecond << ",\n";
            os << "    \"active_threads\": " << s.ActiveThreads << ",\n";
            os << "    \"queued_tasks\": " << s.QueuedTasks << "\n";
            os << "  }";
            if (i + 1 < History_.size()) os << ",";
            os << "\n";
        }
        os << "]\n";
    }

    void ExportCSV(std::ostream& os) const {
        std::lock_guard<std::mutex> lock(Mutex_);
        os << "TimestampNs,InstructionsExecuted,CyclesElapsed,BytesProcessed,RowsProcessed,"
           << "MemoryUsed,CacheMisses,InstructionsPerCycle,BytesPerSecond,RowsPerSecond,"
           << "ActiveThreads,QueuedTasks\n";
        for (const auto& s : History_) {
            os << s.TimestampNs << ","
               << s.InstructionsExecuted << ","
               << s.CyclesElapsed << ","
               << s.BytesProcessed << ","
               << s.RowsProcessed << ","
               << s.MemoryUsed << ","
               << s.CacheMisses << ","
               << std::fixed << std::setprecision(4) << s.InstructionsPerCycle << ","
               << std::fixed << std::setprecision(2) << s.BytesPerSecond << ","
               << std::fixed << std::setprecision(2) << s.RowsPerSecond << ","
               << s.ActiveThreads << ","
               << s.QueuedTasks << "\n";
        }
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(Mutex_);
        Current_ = TelemetrySnapshot{};
        History_.clear();
    }

    sz HistorySize() const {
        std::lock_guard<std::mutex> lock(Mutex_);
        return History_.size();
    }

private:
    static constexpr sz kMaxHistory = 1000000;
    TelemetrySnapshot Current_;
    std::vector<TelemetrySnapshot> History_;
    mutable std::mutex Mutex_;
};

// ============================================================================
// QueryProfiler — per-query performance tracking
// ============================================================================

class QueryProfiler {
public:
    struct QueryStats {
        std::string queryName;
        u64 durationUs    = 0;
        u64 rowsProcessed = 0;
        u64 bytesProcessed = 0;
        sz  peakMemory    = 0;
        u32 threadCount   = 0;
    };

    void BeginQuery(const std::string& name) {
        Current_.queryName = name;
        Current_.durationUs = 0;
        Current_.rowsProcessed = 0;
        Current_.bytesProcessed = 0;
        Current_.peakMemory = 0;
        Current_.threadCount = 0;
        QueryStartNs_ = TimestampNs();
    }

    void EndQuery() {
        u64 now = TimestampNs();
        Current_.durationUs = (now - QueryStartNs_) / 1000;
        {
            std::lock_guard<std::mutex> lock(Mutex_);
            QueryHistory.push_back(Current_);
            if (QueryHistory.size() > kMaxHistory) {
                QueryHistory.erase(QueryHistory.begin());
            }
        }
    }

    void RecordBytes(sz bytes) {
        Current_.bytesProcessed += static_cast<u64>(bytes);
    }

    void RecordRows(sz rows) {
        Current_.rowsProcessed += static_cast<u64>(rows);
    }

    void SetPeakMemory(sz mem) {
        if (mem > Current_.peakMemory) Current_.peakMemory = mem;
    }

    void SetThreadCount(u32 count) {
        Current_.threadCount = count;
    }

    QueryStats CurrentStats() const {
        return Current_;
    }

    void ExportJSON(std::ostream& os) const {
        std::lock_guard<std::mutex> lock(Mutex_);
        os << "[\n";
        for (sz i = 0; i < QueryHistory.size(); ++i) {
            const auto& s = QueryHistory[i];
            os << "  {\n";
            os << "    \"query_name\": \"" << s.queryName << "\",\n";
            os << "    \"duration_us\": " << s.durationUs << ",\n";
            os << "    \"rows_processed\": " << s.rowsProcessed << ",\n";
            os << "    \"bytes_processed\": " << s.bytesProcessed << ",\n";
            os << "    \"peak_memory\": " << s.peakMemory << ",\n";
            os << "    \"thread_count\": " << s.threadCount << "\n";
            os << "  }";
            if (i + 1 < QueryHistory.size()) os << ",";
            os << "\n";
        }
        os << "]\n";
    }

    std::vector<QueryStats> QueryHistory;

private:
    static constexpr sz kMaxHistory = 1000000;
    QueryStats Current_;
    u64 QueryStartNs_ = 0;
    mutable std::mutex Mutex_;

    static u64 TimestampNs() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<u64>(ts.tv_sec) * 1000000000ULL + static_cast<u64>(ts.tv_nsec);
    }
};

} // namespace voxel
