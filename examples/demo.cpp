#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>

using namespace voxel;
using Clock = std::chrono::high_resolution_clock;

static constexpr sz N = 1'000'000;
static constexpr sz kWarmupIters = 2;
static constexpr sz kBenchIters  = 5;

struct BenchResult {
    std::string Name;
    f64         MedianTimeUs;
    f64         Throughput;
    std::string Unit;
    bool        Passed;
    std::string Method;
};

static BenchResult MakeResult(const std::string& name, f64 medianUs, f64 throughput,
                               const std::string& unit, bool passed, const std::string& method)
{
    BenchResult r;
    r.Name         = name;
    r.MedianTimeUs = medianUs;
    r.Throughput   = throughput;
    r.Unit         = unit;
    r.Passed       = passed;
    r.Method       = method;
    return r;
}

template<typename F>
static f64 MeasureMedian(F&& fn, sz warmup = kWarmupIters, sz iters = kBenchIters) {
    std::vector<f64> times(iters);
    for (sz w = 0; w < warmup; ++w) fn();
    for (sz i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[i] = static_cast<f64>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

int main()
{
    constexpr f64 kThreshold     = 500.0;
    constexpr sz  kLanesVal      = Engine<f64>::kLanes;
    std::mt19937_64 rng(42);

    std::vector<BenchResult> results;
    results.reserve(32);

    std::cout << std::setprecision(10);
    std::cout << "================================================================================\n";
    std::cout << "  VoxelVM Comprehensive Subsystem Performance Benchmark\n";
    std::cout << "================================================================================\n\n";

    // ================================================================
    // Shared test data
    // ================================================================
    std::vector<f64> f64Data(N);
    {
        std::uniform_real_distribution<f64> dist(0.0, 1000.0);
        for (sz i = 0; i < N; ++i) f64Data[i] = dist(rng);
    }

    f64 expectedF64Sum = 0.0;
    sz  expectedF64Cnt = 0;
    for (sz i = 0; i < N; ++i) {
        if (f64Data[i] > kThreshold) { expectedF64Sum += f64Data[i]; ++expectedF64Cnt; }
    }

    // ================================================================
    // 1. Interpreter Engine — f64 Filter+Sum
    // ================================================================
    {
        f64 result = 0.0;
        f64 us = MeasureMedian([&]() {
            Engine<f64> engine;
            engine.ScalarReg(0) = 0;
            engine.ScalarReg(1) = 0;
            engine.ScalarReg(2) = N;
            engine.ScalarReg(3) = std::bit_cast<u64>(kThreshold);
            engine.ScalarReg(4) = kLanesVal;
            sz segId = engine.AddSegment(f64Data.data(), N);
            std::vector<u32> code = {
                Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw,
                Instruction::VFilterGt(1, 0, 3).raw,
                Instruction::VSum(5, 1).raw,
                Instruction::Addf(0, 0, 5).raw,
                Instruction::Add(1, 1, 4).raw,
                Instruction::Cmp(1, 2).raw,
                Instruction::Jnz(-6).raw,
                Instruction::Halt().raw,
            };
            engine.LoadProgram(code);
            engine.Run();
            result = std::bit_cast<f64>(engine.ScalarReg(0));
        });
        bool pass = std::fabs(result - expectedF64Sum) < std::max(expectedF64Sum, 1.0) * 1e-12;
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "Interpreter f64 Filter+Sum", us, throughput, "M elem/s", pass,
            "VLoad+VFilterGt+VSum+Addf loop on 1M rand f64, median 5 runs"));
    }

    // ================================================================
    // 2. Interpreter Engine — i64 Filter+Sum
    // ================================================================
    {
        std::vector<i64> i64Data(N);
        std::uniform_int_distribution<i64> idist(0, 1000);
        for (sz i = 0; i < N; ++i) i64Data[i] = idist(rng);
        i64 iThreshold = 500;
        i64 expectedSum = 0;
        for (sz i = 0; i < N; ++i) if (i64Data[i] > iThreshold) expectedSum += i64Data[i];

        i64 result = 0;
        f64 us = MeasureMedian([&]() {
            Engine<i64> engine;
            engine.ScalarReg(0) = 0;
            engine.ScalarReg(1) = 0;
            engine.ScalarReg(2) = N;
            engine.ScalarReg(3) = static_cast<u64>(iThreshold);
            engine.ScalarReg(4) = kLanesVal;
            sz segId = engine.AddSegment(i64Data.data(), N);
            std::vector<u32> code = {
                Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw,
                Instruction::VFilterGt(1, 0, 3).raw,
                Instruction::VSum(5, 1).raw,
                Instruction::Add(0, 0, 5).raw,
                Instruction::Add(1, 1, 4).raw,
                Instruction::Cmp(1, 2).raw,
                Instruction::Jnz(-6).raw,
                Instruction::Halt().raw,
            };
            engine.LoadProgram(code);
            engine.Run();
            result = static_cast<i64>(engine.ScalarReg(0));
        });
        bool pass = (result == expectedSum);
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "Interpreter i64 Filter+Sum", us, throughput, "M elem/s", pass,
            "VLoad+VFilterGt+VSum+Add loop on 1M rand i64, median 5 runs"));
    }

    // ================================================================
    // 3. SIMD Scalar — filter_gt_f64 + sum_f64
    // ================================================================
    {
        sz passCount = 0;
        f64 sum = 0.0;
        std::vector<f64> filtered(N);
        f64 us = MeasureMedian([&]() {
            passCount = simd::scalar::scalar_filter_gt_f64(filtered.data(), f64Data.data(), kThreshold, N);
            sum = simd::scalar::scalar_sum_f64(filtered.data(), passCount);
        });
        bool pass = (passCount == expectedF64Cnt) &&
                    (std::fabs(sum - expectedF64Sum) < std::max(expectedF64Sum, 1.0) * 1e-12);
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "SIMD Scalar f64 filter+sum", us, throughput, "M elem/s", pass,
            "scalar_filter_gt_f64 + scalar_sum_f64 on 1M rand f64, median 5 runs"));
    }

    // ================================================================
    // 4. SIMD Scalar — add_f64 + mul_f64
    // ================================================================
    {
        std::vector<f64> a(N), b(N), result1(N), result2(N);
        for (sz i = 0; i < N; ++i) { a[i] = static_cast<f64>(i) * 0.001; b[i] = a[i] * 0.5; }
        f64 us = MeasureMedian([&]() {
            simd::scalar::scalar_add_f64(result1.data(), a.data(), b.data(), N);
            simd::scalar::scalar_mul_f64(result2.data(), a.data(), b.data(), N);
        });
        f64 expectedAdd = a[0] + b[0];
        f64 expectedMul = a[0] * b[0];
        bool pass = (std::fabs(result1[0] - expectedAdd) < 1e-12) &&
                    (std::fabs(result2[0] - expectedMul) < 1e-12);
        f64 throughput = (static_cast<f64>(N) * 2.0) / us;
        results.push_back(MakeResult(
            "SIMD Scalar f64 add+mul", us, throughput, "M elem/s", pass,
            "scalar_add_f64 + scalar_mul_f64 on 1M f64 each, median 5 runs"));
    }

    // ================================================================
    // 5. Arena Allocator
    // ================================================================
    {
        constexpr sz kArenaIter = 100;
        f64 us = MeasureMedian([&]() {
            Arena arena(1048576); // 1 MiB blocks
            for (sz j = 0; j < kArenaIter; ++j) {
                volatile i32* p = arena.AllocMany<i32>(N);
                (void)p;
                arena.Reset();
            }
        });
        sz totalBytes = static_cast<sz>(N) * sizeof(i32) * kArenaIter;
        f64 mbytes = static_cast<f64>(totalBytes) / (1024.0 * 1024.0);
        f64 throughput = (mbytes / us) * 1e6;
        results.push_back(MakeResult(
            "Arena Allocator", us, throughput, "MB/s", true,
            "Alloc+Reset 1M i32 (4MiB) x100 via Arena, median 5 runs"));
    }

    // ================================================================
    // 6. RegFile — 10M scalar set+get
    // ================================================================
    {
        constexpr sz kIter = 10'000'000;
        volatile u64 sink = 0;
        f64 us = MeasureMedian([&]() {
            RegFile regs;
            for (sz i = 0; i < kIter; ++i) {
                regs.Scalar(i & 0x3F) = i;
                sink = regs.Scalar(i & 0x3F);
            }
        });
        f64 opsPerUs = static_cast<f64>(kIter * 2) / us;
        results.push_back(MakeResult(
            "RegFile set+get", us, opsPerUs, "ops/us", (sink != 0),
            "10M Scalar reg writes + reads on 64 registers, median 5 runs"));
    }

    // ================================================================
    // 7. Instruction encode+decode
    // ================================================================
    {
        constexpr sz kIter = 10'000'000;
        volatile u32 sink = 0;
        f64 us = MeasureMedian([&]() {
            for (sz i = 0; i < kIter; ++i) {
                u32 raw = Instruction::Add(static_cast<u8>(i & 0xF),
                                           static_cast<u8>((i>>4) & 0xF),
                                           static_cast<u8>((i>>8) & 0xF)).raw;
                Instruction inst(raw);
                sink = static_cast<u32>(inst.Rd()) |
                       (static_cast<u32>(inst.Ra()) << 8) |
                       (static_cast<u32>(inst.Rb()) << 16) |
                       (static_cast<u32>(static_cast<u8>(inst.Op())) << 24);
            }
        });
        f64 opsPerUs = static_cast<f64>(kIter * 2) / us;
        results.push_back(MakeResult(
            "Instruction encode+decode", us, opsPerUs, "ops/us", (sink != 0),
            "10M Instruction::Add encode + decode (Op,Rd,Ra,Rb), median 5 runs"));
    }

    // ================================================================
    // 8. NullBitmap — create 1M bitmap, set 10% nulls, count
    // ================================================================
    {
        sz nullCount = 0;
        f64 us = MeasureMedian([&]() {
            NullBitmap bm(N);
            for (sz i = 0; i < N; i += 10) bm.SetNull(i);
            nullCount = bm.NullCount();
        });
        bool pass = (nullCount == (N + 9) / 10);
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "NullBitmap set+count", us, throughput, "M elem/s", pass,
            "Create 1M bitmap, set 10% rows null, count nulls, median 5 runs"));
    }

    // ================================================================
    // 9. DictionaryEncoding<f64>
    // ================================================================
    {
        sz dictSize = 0;
        sz memUsage = 0;
        bool decodePass = false;
        f64 us = MeasureMedian([&]() {
            encoding::DictionaryEncoding<f64> dict;
            dict.Build(f64Data.data(), N);
            dictSize = dict.Cardinality();
            memUsage = dict.MemoryUsage();
            std::vector<f64> decoded(N);
            dict.DecodeBatch(decoded.data(), 0, N);
            f64 err = 0.0;
            for (sz i = 0; i < N; ++i) err += std::fabs(decoded[i] - f64Data[i]);
            decodePass = (err < 1e-12);
        });
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "DictionaryEncoding<f64>", us, throughput, "M elem/s", decodePass,
            "Build dict on 1M rand f64, decode+verify, dictSize=" + std::to_string(dictSize) +
            " mem=" + std::to_string(memUsage / 1024) + "KiB"));
    }

    // ================================================================
    // 10. RLEEncoding<f64> — 1M f64 with 10 runs
    // ================================================================
    {
        std::vector<f64> rleData(N);
        for (sz run = 0; run < 10; ++run) {
            f64 val = static_cast<f64>(run) * 100.0;
            sz start = run * (N / 10);
            sz end = (run == 9) ? N : (run + 1) * (N / 10);
            for (sz i = start; i < end; ++i) rleData[i] = val;
        }
        sz runCount = 0;
        sz memUsage = 0;
        bool decodePass = false;
        f64 us = MeasureMedian([&]() {
            encoding::RLEEncoding<f64> rle;
            rle.Build(rleData.data(), N);
            runCount = rle.RunCount();
            memUsage = rle.MemoryUsage();
            std::vector<f64> decoded(N);
            rle.DecodeBatch(decoded.data(), 0, N);
            f64 err = 0.0;
            for (sz i = 0; i < N; ++i) err += std::fabs(decoded[i] - rleData[i]);
            decodePass = (err < 1e-12);
        });
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "RLEEncoding<f64>", us, throughput, "M elem/s", decodePass,
            "Build RLE on 1M f64 with 10 runs, decode+verify, runs=" + std::to_string(runCount) +
            " mem=" + std::to_string(memUsage) + "B"));
    }

    // ================================================================
    // 11. DeltaEncoding<i64> — 1M sequential i64
    // ================================================================
    {
        std::vector<i64> seqData(N);
        for (sz i = 0; i < N; ++i) seqData[i] = static_cast<i64>(i);
        bool decodePass = false;
        sz memUsage = 0;
        f64 us = MeasureMedian([&]() {
            encoding::DeltaEncoding<i64> delta;
            delta.Build(seqData.data(), N);
            memUsage = delta.MemoryUsage();
            std::vector<i64> decoded(N);
            delta.Decode(decoded.data(), N);
            f64 err = 0.0;
            for (sz i = 0; i < N; ++i) err += static_cast<f64>(std::abs(decoded[i] - seqData[i]));
            decodePass = (err < 1e-12);
        });
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "DeltaEncoding<i64>", us, throughput, "M elem/s", decodePass,
            "Build delta on 1M sequential i64 values, decode+verify, mem=" +
            std::to_string(memUsage) + "B"));
    }

    // ================================================================
    // 12. VectorFilter<f64> — ApplyGT + verify bitmap
    // ================================================================
    {
        ops::VectorFilter<f64>::FilterResult filterResult;
        bool verifyPass = false;
        f64 us = MeasureMedian([&]() {
            ops::VectorFilter<f64> vf;
            filterResult = vf.ApplyGT(f64Data.data(), N, kThreshold);
            verifyPass = (filterResult.PassCount == expectedF64Cnt);
        });
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "VectorFilter<f64> ApplyGT", us, throughput, "M elem/s", verifyPass,
            "ApplyGT on 1M rand f64, verified pass count " + std::to_string(filterResult.PassCount) +
            " == " + std::to_string(expectedF64Cnt)));
    }

    // ================================================================
    // 13. HashAggregator — 100K rows, 100 groups
    // ================================================================
    {
        constexpr sz kAggRows = 100'000;
        constexpr sz kGroups  = 100;
        std::vector<u32> groups(kAggRows);
        std::vector<f64> values(kAggRows);
        {
            std::uniform_int_distribution<u32> gdist(0, static_cast<u32>(kGroups) - 1);
            for (sz i = 0; i < kAggRows; ++i) {
                groups[i] = gdist(rng);
                values[i] = static_cast<f64>(groups[i]) + f64Data[i % N] * 0.001;
            }
        }
        sz groupCount = 0;
        f64 us = MeasureMedian([&]() {
            Arena arena;
            ops::HashAggregator<u32, f64> agg(arena);
            agg.Init(128);
            agg.Accumulate(groups.data(), values.data(), kAggRows);
            groupCount = agg.GroupCount();
        });
        f64 groupsPerSec = static_cast<f64>(kAggRows) / us;
        results.push_back(MakeResult(
            "HashAggregator GROUP BY", us, groupsPerSec, "M rows/s", (groupCount == kGroups),
            "Accumulate 100K rows into 100 groups via HashAggregator<u32,f64>, median 5 runs"));
    }

    // ================================================================
    // 14. SortOperator — Sort 1M indices by f64 values
    // ================================================================
    {
        std::vector<u32> indices(N);
        Segment<f64> sortSeg(f64Data.data(), N);
        bool sorted = false;
        f64 us = MeasureMedian([&]() {
            ops::SortOperator::SortAscending(sortSeg, indices.data());
            sorted = true;
            for (sz i = 1; i < N && sorted; ++i)
                if (f64Data[indices[i]] < f64Data[indices[i-1]]) sorted = false;
        });
        f64 throughput = static_cast<f64>(N) / us;
        results.push_back(MakeResult(
            "SortOperator index sort", us, throughput, "M elem/s", sorted,
            "MergeSort indices by f64 values on 1M elements, median 5 runs"));
    }

    // ================================================================
    // 15. HashJoin — Build+probe on 10K keys
    // ================================================================
    {
        constexpr sz kJoinRows = 10'000;
        std::vector<i64> buildKeys(kJoinRows);
        std::vector<i64> probeKeys(kJoinRows);
        for (sz i = 0; i < kJoinRows; ++i) {
            buildKeys[i] = static_cast<i64>(i);
            probeKeys[i] = static_cast<i64>(i);
        }
        sz matchCount = 0;
        bool matchPass = false;
        f64 us = MeasureMedian([&]() {
            Arena arena;
            ops::HashJoin<i64> join(arena);
            join.Build(buildKeys.data(), kJoinRows);
            std::vector<u32> leftMatches(kJoinRows), rightMatches(kJoinRows);
            join.Probe(probeKeys.data(), kJoinRows, leftMatches.data(), rightMatches.data(), matchCount);
            matchPass = (matchCount == kJoinRows);
        });
        f64 joinsPerSec = static_cast<f64>(kJoinRows) / us;
        results.push_back(MakeResult(
            "HashJoin build+probe", us, joinsPerSec, "M keys/s", matchPass,
            "Build hash table from 10K i64 keys, probe with same 10K keys, median 5 runs"));
    }

    // ================================================================
    // 16. ThreadPool — 100 tasks of ~100us work
    // ================================================================
    {
        constexpr sz kTasks = 100;
        constexpr sz kWorkPerTask = 5000;
        auto work = []() {
            volatile f64 v = 1.0;
            for (sz i = 0; i < kWorkPerTask; ++i) v = std::sqrt(v + 1.0);
            (void)v;
        };

        f64 seqUs = 0;
        {
            auto t0 = Clock::now();
            for (sz i = 0; i < kTasks; ++i) work();
            auto t1 = Clock::now();
            seqUs = static_cast<f64>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }

        f64 parUs = MeasureMedian([&]() {
            ThreadPool pool;
            for (sz i = 0; i < kTasks; ++i) pool.Enqueue(work);
            pool.WaitAll();
        });

        f64 speedup = seqUs / parUs;
        u32 hwThreads = std::thread::hardware_concurrency();
        results.push_back(MakeResult(
            "ThreadPool parallel speedup", parUs, speedup, "x speedup", (speedup > 1.0),
            "100 tasks of sqrt workload, sequential=" + std::to_string(static_cast<i64>(seqUs)) +
            "us, parallel=" + std::to_string(static_cast<i64>(parUs)) +
            "us, " + std::to_string(hwThreads) + " HW threads"));
    }

    // ================================================================
    // 17. MurmurHash64A — 1M strings of length 8
    // ================================================================
    {
        constexpr sz kHashN = 1'000'000;
        std::vector<std::string> strings(kHashN);
        for (sz i = 0; i < kHashN; ++i) {
            u64 val = static_cast<u64>(i) * 0x9E3779B97F4A7C15ULL;
            strings[i] = std::string(reinterpret_cast<const char*>(&val), 8);
        }
        volatile u64 hashSink = 0;
        f64 us = MeasureMedian([&]() {
            u64 h = 0;
            for (sz i = 0; i < kHashN; ++i)
                h ^= ops::MurmurHash64A(strings[i].data(), 8);
            hashSink = h;
        });
        f64 gbPerSec = (static_cast<f64>(kHashN) * 8.0) / (us * 1e3);
        results.push_back(MakeResult(
            "MurmurHash64A", us, gbPerSec, "GB/s", (hashSink != 0),
            "Hash 1M strings of length 8 via MurmurHash64A, median 5 runs"));
    }

    // ================================================================
    // 18. BitUtils — 10M popcount + clz
    // ================================================================
    {
        constexpr sz kIter = 10'000'000;
        volatile u64 sink = 0;
        f64 us = MeasureMedian([&]() {
            u64 acc = 0;
            for (sz i = 0; i < kIter; ++i) {
                u32 p = PopCount64(i);
                u64 c = CountLeadingZeros64(i);
                acc += p + c;
            }
            sink = acc;
        });
        f64 opsPerUs = static_cast<f64>(kIter * 2) / us;
        results.push_back(MakeResult(
            "BitUtils popcount+clz", us, opsPerUs, "ops/us", (sink != 0),
            "10M PopCount64 + CountLeadingZeros64 operations, median 5 runs"));
    }

    // ================================================================
    // 19. FastInvSqrt — 10M operations, verify accuracy
    // ================================================================
    {
        constexpr sz kIter = 10'000'000;
        f64 maxErr = 0.0;
        f64 us = MeasureMedian([&]() {
            f64 err = 0.0;
            for (sz i = 0; i < kIter; ++i) {
                f64 x = static_cast<f64>(i % 10000 + 1) * 0.1;
                f64 fast = FastInvSqrt(x);
                f64 exact = 1.0 / std::sqrt(x);
                f64 e = std::fabs(fast - exact) / exact;
                if (e > err) err = e;
            }
            maxErr = err;
        });
        f64 opsPerUs = static_cast<f64>(kIter) / us;
        results.push_back(MakeResult(
            "FastInvSqrt", us, opsPerUs, "ops/us", (maxErr < 0.005),
            "10M FastInvSqrt operations, max relative error = " +
            std::to_string(maxErr)));
    }

    // ================================================================
    // 20. Bytecode optimizer — filter+sum program
    // ================================================================
    {
        std::vector<u32> code = {
            Instruction::VLoad(0, 1, 0, 0).raw,
            Instruction::VFilterGt(1, 0, 3).raw,
            Instruction::VSum(5, 1).raw,
            Instruction::Addf(0, 0, 5).raw,
            Instruction::Add(1, 1, 4).raw,
            Instruction::Cmp(1, 2).raw,
            Instruction::Jnz(-6).raw,
            Instruction::Halt().raw,
        };
        sz origSize = code.size();
        sz optSize = 0;
        f64 us = MeasureMedian([&]() {
            std::vector<u32> work = code;
            opt::Optimizer opt;
            opt.AddPass(std::make_unique<opt::ConstantFolder>());
            opt.AddPass(std::make_unique<opt::DeadCodeEliminator>());
            opt.AddPass(std::make_unique<opt::PeepholeOptimizer>());
            opt.AddPass(std::make_unique<opt::NopSqueezer>());
            opt.Optimize(work);
            optSize = work.size();
        });
        f64 reductionPct = (origSize > 0) ? (1.0 - static_cast<f64>(optSize) / static_cast<f64>(origSize)) * 100.0 : 0.0;
        results.push_back(MakeResult(
            "Bytecode optimizer", us, reductionPct, "% reduction", (optSize <= origSize),
            "Optimize filter+sum program (ConstantFolder+DeadCodeElim+Peephole+NopSqueeze), "
            + std::to_string(origSize) + "->" + std::to_string(optSize) + " instrs"));
    }

    // ================================================================
    // 21. JIT Compile — filter+sum program
    // 22. JIT Execute — filter+sum on 1M f64
    // ================================================================
    {
        auto compiler = codegen::CreateJitCompiler();
        if (!compiler) {
            results.push_back(MakeResult("JIT compile time", 0, 0, "us", false, "SKIP: no JIT backend"));
            results.push_back(MakeResult("JIT execute time", 0, 0, "M elem/s", false, "SKIP: no JIT backend"));
        } else {
            // JIT compile benchmark
            f64 compileUs = 0.0;
            codegen::JitFunction func;
            bool compiled = false;
            compileUs = MeasureMedian([&]() {
                std::vector<u32> code = {
                    Instruction::VLoad(0, 1, 0, 0).raw,
                    Instruction::VFilterGt(1, 0, 3).raw,
                    Instruction::VSum(5, 1).raw,
                    Instruction::Addf(0, 0, 5).raw,
                    Instruction::Add(1, 1, 4).raw,
                    Instruction::Cmp(1, 2).raw,
                    Instruction::Jnz(-6).raw,
                    Instruction::Halt().raw,
                };
                compiled = compiler->Compile(code.data(), code.size(), func);
            });
            bool compilePass = compiled && func.IsValid();
            results.push_back(MakeResult(
                "JIT compile time", compileUs, compileUs, "us", compilePass,
                "Compile filter+sum bytecode to native x86-64/ARM64, median 5 runs"));

            // JIT execute benchmark
            if (compilePass) {
                f64 result = 0.0;
                f64 execUs = MeasureMedian([&]() {
                    RegFile regfile;
                    regfile.Scalar(0) = 0;
                    regfile.Scalar(1) = 0;
                    regfile.Scalar(2) = N;
                    regfile.Scalar(3) = std::bit_cast<u64>(kThreshold);
                    regfile.Scalar(4) = kLanesVal;
                    f64* segBase = f64Data.data();
                    u64  segCount = N;
                    func.Entry(&regfile, &segBase, &segCount);
                    result = std::bit_cast<f64>(regfile.Scalar(0));
                });
                bool execPass = std::fabs(result - expectedF64Sum) < std::max(expectedF64Sum, 1.0) * 1e-12;
                f64 throughput = static_cast<f64>(N) / execUs;
                results.push_back(MakeResult(
                    "JIT execute time", execUs, throughput, "M elem/s", execPass,
                    "Execute native-compiled filter+sum on 1M f64, median 5 runs"));
                compiler->Release(func);
            } else {
                results.push_back(MakeResult("JIT execute time", 0, 0, "M elem/s", false, "COMPILE FAIL"));
            }
        }
    }

    // ================================================================
    // SUMMARY TABLE
    // ================================================================
    std::cout << "\n================================================================================\n";
    std::cout << "  BENCHMARK SUMMARY TABLE\n";
    std::cout << "================================================================================\n";
    std::cout << std::left
              << std::setw(3)  << "#"
              << std::setw(40) << "Benchmark"
              << std::setw(14) << "Time/Thruput"
              << std::setw(12) << "Unit"
              << std::setw(8)  << "Pass"
              << "Methodology"
              << "\n";
    std::cout << std::string(3, '-') << "  "
              << std::string(40, '-') << "  "
              << std::string(14, '-') << "  "
              << std::string(12, '-') << "  "
              << std::string(8, '-')  << "  "
              << std::string(70, '-')
              << "\n";

    for (sz i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << std::left
                  << std::setw(3)  << (i + 1)
                  << std::setw(40) << r.Name
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.Throughput
                  << std::setw(12) << r.Unit
                  << std::setw(8)  << (r.Passed ? "PASS" : "FAIL")
                  << r.Method
                  << "\n";
    }

    std::cout << "================================================================================\n";
    sz passCount = 0;
    for (const auto& r : results) if (r.Passed) ++passCount;
    std::cout << "  Results: " << passCount << "/" << results.size() << " benchmarks PASSED\n";
    std::cout << "================================================================================\n";

    return 0;
}
