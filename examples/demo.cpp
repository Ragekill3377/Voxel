#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>

using namespace voxel;

int main()
{
    constexpr sz N          = 1'000'000;
    constexpr f64 kThreshold = 500.0;
    constexpr sz kLanesVal   = Engine<f64>::kLanes;

    // ================================================================
    // Generate test data
    // ================================================================
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<f64> dist(0.0, 1000.0);

    std::vector<f64> data(N);
    for (sz i = 0; i < N; ++i) data[i] = dist(rng);

    // Ground-truth computation
    f64 expectedSum = 0.0;
    f64 expectedMin = std::numeric_limits<f64>::max();
    f64 expectedMax = std::numeric_limits<f64>::lowest();
    sz  expectedCount = 0;
    for (sz i = 0; i < N; ++i) {
        if (data[i] > kThreshold) {
            expectedSum += data[i];
            expectedCount++;
            if (data[i] < expectedMin) expectedMin = data[i];
            if (data[i] > expectedMax) expectedMax = data[i];
        }
    }

    std::cout << std::setprecision(12);
    std::cout << "========================================\n";
    std::cout << " VoxelVM Production Engine Benchmark\n";
    std::cout << "========================================\n\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Threshold: " << kThreshold << "\n";
    std::cout << "Expected pass count: " << expectedCount << "\n";
    std::cout << "Expected sum: " << expectedSum << "\n";
    std::cout << "Expected min: " << expectedMin << "\n";
    std::cout << "Expected max: " << expectedMax << "\n\n";

    // ================================================================
    // 1. Bytecode Engine — Interpreted Filter+Sum Loop
    // ================================================================
    {
        Engine<f64> engine;

        engine.ScalarReg(0) = 0;                                    // accumulator
        engine.ScalarReg(1) = 0;                                    // offset counter
        engine.ScalarReg(2) = N;                                    // total count
        engine.ScalarReg(3) = std::bit_cast<u64>(kThreshold);       // threshold

        sz segId = engine.AddSegment(data.data(), N);

        std::vector<u32> rawCode;
        rawCode.reserve(64);

        rawCode.push_back(Instruction::Mov(4, static_cast<i16>(kLanesVal)).raw);             // 0: R4 = kLanes
        rawCode.push_back(Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw);          // 1: V0 = seg[R1..R1+kL]
        rawCode.push_back(Instruction::VFilter(1, 0, 3, kCmpGT).raw);                        // 2: V1 = V0 > R3
        rawCode.push_back(Instruction::VSum(5, 1).raw);                                       // 3: R5 = sum(V1)
        rawCode.push_back(Instruction::Addf(0, 0, 5).raw);                                    // 4: R0 += R5
        rawCode.push_back(Instruction::Add(1, 1, 4).raw);                                     // 5: R1 += kLanes
        rawCode.push_back(Instruction::Cmp(1, 2).raw);                                        // 6: CMP R1, R2
        rawCode.push_back(Instruction::Jnz(-6).raw);                                          // 7: if R1 != R2 goto 1
        rawCode.push_back(Instruction::Halt().raw);                                           // 8: halt

        engine.LoadProgram(rawCode);

        auto t0 = std::chrono::high_resolution_clock::now();
        engine.Run();
        auto t1 = std::chrono::high_resolution_clock::now();

        f64 result = std::bit_cast<f64>(engine.ScalarReg(0));
        f64 delta  = std::fabs(result - expectedSum);
        auto us    = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout << "[Bytecode Engine / Filter+Sum Loop]\n";
        std::cout << "  Result: " << result << "\n";
        std::cout << "  Delta:  " << delta << "\n";
        std::cout << "  Match:  " << (delta < std::max(expectedSum, 1.0) * 1e-12 ? "YES" : "NO") << "\n";
        std::cout << "  Time:   " << us << " us  (" << (static_cast<f64>(N) / us) << " M elem/s)\n\n";
    }

    // ================================================================
    // 2. SIMD Vector Filter (portable scalar path used here)
    // ================================================================
    {
        std::vector<f64> filtered(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        sz passCount = simd::scalar::scalar_filter_gt_f64(filtered.data(), data.data(), kThreshold, N);
        f64 sum = simd::scalar::scalar_sum_f64(filtered.data(), passCount);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout << "[SIMD Scalar / Filter+Sum]\n";
        std::cout << "  Pass count: " << passCount << "\n";
        std::cout << "  Sum:        " << sum << "\n";
        std::cout << "  Match:      " << (std::fabs(sum - expectedSum) < 1e-6 ? "YES" : "NO") << "\n";
        std::cout << "  Time:       " << us << " us  (" << (static_cast<f64>(N) / us) << " M elem/s)\n\n";
    }

    // ================================================================
    // 3. NullBitmap + Encoding Subsystems
    // ================================================================
    {
        NullBitmap nulls(N);
        for (sz i = 0; i < N; i += 1000) nulls.SetNull(i);
        sz nullCount = nulls.NullCount();

        // Dictionary encoding
        encoding::DictionaryEncoding<f64> dict;
        dict.Build(data.data(), N);

        // Decode and verify
        std::vector<f64> decoded(N);
        dict.DecodeBatch(decoded.data(), 0, N);

        f64 decodeErr = 0.0;
        for (sz i = 0; i < N; ++i) decodeErr += std::fabs(decoded[i] - data[i]);

        std::cout << "[NullBitmap + Dictionary Encoding]\n";
        std::cout << "  Null count:   " << nullCount << "\n";
        std::cout << "  Dict entries: " << dict.Dictionary.size() << "\n";
        std::cout << "  Dict memory:  " << (dict.MemoryUsage() / 1024.0) << " KiB\n";
        std::cout << "  Decode err:   " << decodeErr << " (should be 0)\n\n";
    }

    // ================================================================
    // 4. Sort Operator — RadixSort on indices
    // ================================================================
    {
        std::vector<u32> indices(N);
        for (sz i = 0; i < N; ++i) indices[i] = static_cast<u32>(i);

        Segment<f64> sortSeg(data.data(), N);
        auto t0 = std::chrono::high_resolution_clock::now();
        ops::SortOperator::SortAscending(sortSeg, indices.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        bool sorted = true;
        for (sz i = 1; i < N && sorted; ++i)
            if (data[indices[i]] < data[indices[i-1]]) sorted = false;

        std::cout << "[Sort / Index RadixSort]\n";
        std::cout << "  Sorted: " << (sorted ? "YES" : "NO") << "\n";
        std::cout << "  Time:   " << us << " us\n";
        std::cout << "  Min:    " << data[indices[0]] << "\n";
        std::cout << "  Max:    " << data[indices[N-1]] << "\n\n";
    }

    // ================================================================
    // 5. HashAggregator — Small-group aggregation
    // ================================================================
    {
        Arena arena;
        ops::HashAggregator<u32, f64> aggregator(arena);
        aggregator.Init(128);

        std::vector<u32> groups(1000);
        std::vector<f64> values(1000);
        for (u32 i = 0; i < 1000; ++i) { groups[i] = i % 10; values[i] = data[i]; }

        auto t0 = std::chrono::high_resolution_clock::now();
        aggregator.Accumulate(groups.data(), values.data(), 1000);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout << "[HashAggregator / GROUP BY]\n";
        std::cout << "  Groups: " << aggregator.GroupCount() << "\n";
        std::cout << "  Time:   " << us << " us\n\n";
    }

    // ================================================================
    // 6. Telemetry Dump
    // ================================================================
    {
        platform::CpuInfo cpu = platform::QueryCpuInfo();
        std::cout << "[Platform / CPU Info]\n";
        std::cout << "  Logical cores: " << cpu.LogicalCores << "\n";
        std::cout << "  L1 cache:      " << (cpu.L1CacheSize / 1024.0) << " KiB\n";
        std::cout << "  L2 cache:      " << (cpu.L2CacheSize / 1024.0) << " KiB\n";
        std::cout << "  L3 cache:      " << (cpu.L3CacheSize / (1024.0*1024.0)) << " MiB\n";
        std::cout << "  AVX2:          " << (cpu.HasAVX2 ? "YES" : "NO") << "\n";
        std::cout << "  AVX-512:       " << (cpu.HasAVX512F ? "YES" : "NO") << "\n";
        std::cout << "  FMA:           " << (cpu.HasFMA ? "YES" : "NO") << "\n";
        std::cout << "  POPCNT:        " << (cpu.HasPOPCNT ? "YES" : "NO") << "\n\n";
    }

    std::cout << "========================================\n";
    std::cout << " All benchmarks complete.\n";
    std::cout << "========================================\n";

    return 0;
}
