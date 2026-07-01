#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>
#include <vector>
#include <functional>
#include <atomic>
#include <algorithm>

namespace voxel {

// ============================================================================
// ThreadPool — fixed-size worker pool with a single shared queue
// ============================================================================

class ThreadPool {
public:
    explicit ThreadPool(u32 numThreads = 0)
        : Stop_(false)
        , ActiveTaskCount_(0)
    {
        u32 n = (numThreads == 0) ? std::thread::hardware_concurrency() : numThreads;
        if (n < 1) n = 1;
        Workers_.reserve(n);
        for (u32 i = 0; i < n; ++i) {
            Workers_.emplace_back(&ThreadPool::WorkerLoop, this);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(QueueMutex_);
            Stop_.store(true, std::memory_order_release);
        }
        Condition_.notify_all();
        for (auto& worker : Workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename F>
    void Enqueue(F&& task) {
        {
            std::lock_guard<std::mutex> lock(QueueMutex_);
            if (Stop_.load(std::memory_order_acquire)) return;
            TaskQueue_.emplace(std::forward<F>(task));
        }
        Condition_.notify_one();
    }

    void WaitAll() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(QueueMutex_);
                if (TaskQueue_.empty() && ActiveTaskCount_.load(std::memory_order_acquire) == 0) {
                    break;
                }
            }
            std::this_thread::yield();
        }
    }

    u32 ThreadCount() const {
        return static_cast<u32>(Workers_.size());
    }

    u32 PendingTasks() const {
        std::lock_guard<std::mutex> lock(QueueMutex_);
        return static_cast<u32>(TaskQueue_.size()) + ActiveTaskCount_.load(std::memory_order_acquire);
    }

private:
    std::vector<std::thread> Workers_;
    std::queue<std::function<void()>> TaskQueue_;
    mutable std::mutex QueueMutex_;
    std::condition_variable Condition_;
    std::atomic<bool> Stop_;
    std::atomic<u32> ActiveTaskCount_;

    void WorkerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(QueueMutex_);
                Condition_.wait(lock, [this] {
                    return Stop_.load(std::memory_order_acquire) || !TaskQueue_.empty();
                });
                if (Stop_.load(std::memory_order_acquire) && TaskQueue_.empty()) {
                    return;
                }
                task = std::move(TaskQueue_.front());
                TaskQueue_.pop();
                ActiveTaskCount_.fetch_add(1, std::memory_order_release);
            }
            task();
            ActiveTaskCount_.fetch_sub(1, std::memory_order_release);
        }
    }
};

// ============================================================================
// WorkStealingThreadPool — per-worker deques with work stealing
// Workers pop from the BACK of their own deque (LIFO for cache locality).
// When idle, workers steal from the FRONT of other workers' deques (FIFO for fairness).
// ============================================================================

class WorkStealingThreadPool {
public:
    explicit WorkStealingThreadPool(u32 numThreads = 0)
        : Stop_(false)
        , ActiveTaskCount_(0)
    {
        u32 n = (numThreads == 0) ? std::thread::hardware_concurrency() : numThreads;
        if (n < 1) n = 1;
        Deques_.resize(n);
        Mutexes_ = new std::mutex[n];
        Workers_.reserve(n);
        for (u32 i = 0; i < n; ++i) {
            Workers_.emplace_back(&WorkStealingThreadPool::WorkerLoop, this, i);
        }
    }

    ~WorkStealingThreadPool() {
        {
            std::lock_guard<std::mutex> lock(GlobalMutex_);
            Stop_.store(true, std::memory_order_release);
        }
        Condition_.notify_all();
        for (auto& worker : Workers_) {
            if (worker.joinable()) worker.join();
        }
        delete[] Mutexes_;
    }

    WorkStealingThreadPool(const WorkStealingThreadPool&)            = delete;
    WorkStealingThreadPool& operator=(const WorkStealingThreadPool&) = delete;

    template<typename F>
    void Enqueue(F&& task) {
        u32 wid = NextWorker_.fetch_add(1, std::memory_order_relaxed) % static_cast<u32>(Deques_.size());
        {
            std::lock_guard<std::mutex> lock(Mutexes_[wid]);
            Deques_[wid].emplace_back(std::forward<F>(task));
        }
        ActiveTaskCount_.fetch_add(1, std::memory_order_release);
        Condition_.notify_one();
    }

    void WaitAll() {
        while (ActiveTaskCount_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

    u32 ThreadCount() const {
        return static_cast<u32>(Workers_.size());
    }

    u32 PendingTasks() const {
        return ActiveTaskCount_.load(std::memory_order_acquire);
    }

private:
    std::vector<std::thread> Workers_;
    std::vector<std::deque<std::function<void()>>> Deques_;
    std::mutex* Mutexes_;
    std::mutex GlobalMutex_;
    std::condition_variable Condition_;
    std::atomic<bool> Stop_;
    std::atomic<u32> ActiveTaskCount_;
    std::atomic<u32> NextWorker_{0};

    void WorkerLoop(u32 workerId) {
        while (true) {
            std::function<void()> task;

            {
                std::lock_guard<std::mutex> lock(Mutexes_[workerId]);
                if (!Deques_[workerId].empty()) {
                    task = std::move(Deques_[workerId].back());
                    Deques_[workerId].pop_back();
                }
            }

            if (!task) {
                u32 n = static_cast<u32>(Deques_.size());
                for (u32 i = 0; i < n; ++i) {
                    u32 victim = (workerId + i + 1) % n;
                    if (victim == workerId) continue;
                    {
                        std::lock_guard<std::mutex> lock(Mutexes_[victim]);
                        if (!Deques_[victim].empty()) {
                            task = std::move(Deques_[victim].front());
                            Deques_[victim].pop_front();
                            break;
                        }
                    }
                }
            }

            if (task) {
                task();
                ActiveTaskCount_.fetch_sub(1, std::memory_order_release);
            } else {
                if (Stop_.load(std::memory_order_acquire) &&
                    ActiveTaskCount_.load(std::memory_order_acquire) == 0) {
                    return;
                }

                std::unique_lock<std::mutex> lock(GlobalMutex_);
                Condition_.wait_for(lock, std::chrono::microseconds(100));
                if (Stop_.load(std::memory_order_acquire) &&
                    ActiveTaskCount_.load(std::memory_order_acquire) == 0) {
                    return;
                }
            }
        }
    }
};

// ============================================================================
// WorkStealingScheduler — per-worker deques with work stealing
// ============================================================================

class WorkStealingScheduler {
public:
    explicit WorkStealingScheduler(u32 numWorkers = 0)
        : Stop_(false)
        , ActiveTaskCount_(0)
    {
        u32 n = (numWorkers == 0) ? std::thread::hardware_concurrency() : numWorkers;
        if (n < 1) n = 1;
        Deques_.resize(n);
        Mutexes_ = new std::mutex[n];
        Workers_.reserve(n);
        for (u32 i = 0; i < n; ++i) {
            Workers_.emplace_back(&WorkStealingScheduler::WorkerLoop, this, i);
        }
    }

    ~WorkStealingScheduler() {
        {
            std::lock_guard<std::mutex> lock(GlobalMutex_);
            Stop_.store(true, std::memory_order_release);
        }
        Condition_.notify_all();
        for (auto& worker : Workers_) {
            if (worker.joinable()) worker.join();
        }
        delete[] Mutexes_;
    }

    WorkStealingScheduler(const WorkStealingScheduler&)            = delete;
    WorkStealingScheduler& operator=(const WorkStealingScheduler&) = delete;

    template<typename F>
    void Spawn(F&& task) {
        u32 wid = NextWorker_.fetch_add(1, std::memory_order_relaxed) % static_cast<u32>(Deques_.size());
        {
            std::lock_guard<std::mutex> lock(Mutexes_[wid]);
            Deques_[wid].push_back(std::forward<F>(task));
        }
        ActiveTaskCount_.fetch_add(1, std::memory_order_release);
        Condition_.notify_one();
    }

    void RunAndWait() {
        while (ActiveTaskCount_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

    u32 WorkerCount() const {
        return static_cast<u32>(Workers_.size());
    }

private:
    std::vector<std::thread> Workers_;
    std::vector<std::deque<std::function<void()>>> Deques_;
    std::mutex* Mutexes_;
    std::mutex GlobalMutex_;
    std::condition_variable Condition_;
    std::atomic<bool> Stop_;
    std::atomic<u32> ActiveTaskCount_;
    std::atomic<u32> NextWorker_{0};

    void WorkerLoop(u32 workerId) {
        while (true) {
            std::function<void()> task;

            {
                std::lock_guard<std::mutex> lock(Mutexes_[workerId]);
                if (!Deques_[workerId].empty()) {
                    task = std::move(Deques_[workerId].back());
                    Deques_[workerId].pop_back();
                }
            }

            if (!task) {
                bool stolen = false;
                u32 n = static_cast<u32>(Deques_.size());
                for (u32 i = 0; i < n; ++i) {
                    u32 victim = (workerId + i + 1) % n;
                    if (victim == workerId) continue;
                    {
                        std::lock_guard<std::mutex> lock(Mutexes_[victim]);
                        if (!Deques_[victim].empty()) {
                            task = std::move(Deques_[victim].front());
                            Deques_[victim].pop_front();
                            stolen = true;
                        }
                    }
                    if (stolen) break;
                }
            }

            if (task) {
                task();
                ActiveTaskCount_.fetch_sub(1, std::memory_order_release);
            } else {
                if (Stop_.load(std::memory_order_acquire) &&
                    ActiveTaskCount_.load(std::memory_order_acquire) == 0) {
                    return;
                }

                std::unique_lock<std::mutex> lock(GlobalMutex_);
                Condition_.wait_for(lock, std::chrono::microseconds(100));
                if (Stop_.load(std::memory_order_acquire) &&
                    ActiveTaskCount_.load(std::memory_order_acquire) == 0) {
                    return;
                }
            }
        }
    }
};

// ============================================================================
// ParallelFor — static work partitioning helper
// ============================================================================

class ParallelFor {
public:
    static void Exec(sz begin, sz end, std::function<void(sz, sz)> chunkFn, u32 numChunks = 0) {
        if (begin >= end) return;

        sz total = end - begin;
        u32 chunks = numChunks;

        if (chunks == 0) {
            u32 hw = std::thread::hardware_concurrency();
            if (hw < 1) hw = 1;
            chunks = hw;
        }

        u32 actualChunks = (chunks > total) ? static_cast<u32>(total) : chunks;
        if (actualChunks < 1) actualChunks = 1;

        sz chunkSize = total / actualChunks;
        sz remainder = total % actualChunks;

        std::vector<std::thread> threads;
        threads.reserve(actualChunks - 1);

        sz offset = begin;
        for (u32 i = 0; i < actualChunks; ++i) {
            sz thisSize = chunkSize + ((i < remainder) ? 1 : 0);
            sz chunkBegin = offset;
            sz chunkEnd   = offset + thisSize;

            if (i + 1 < actualChunks) {
                threads.emplace_back([chunkFn, chunkBegin, chunkEnd]() {
                    chunkFn(chunkBegin, chunkEnd);
                });
            } else {
                chunkFn(chunkBegin, chunkEnd);
            }
            offset = chunkEnd;
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }
};

} // namespace voxel
