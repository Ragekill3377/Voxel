#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <sched.h>
using namespace voxel;

int main() {
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    constexpr sz N = 1'000'000;
    constexpr f64 kThreshold = 500.0;
    constexpr u64 kSeed = 42;
    constexpr int kWarmup = 2, kRuns = 3;

    std::mt19937_64 rng(kSeed);
    std::uniform_real_distribution<f64> dist(0.0, 1000.0);
    std::vector<f64> data(N);
    for (sz i = 0; i < N; ++i) data[i] = dist(rng);

    f64 expected = 0;
    for (sz i = 0; i < N; ++i)
        if (data[i] > kThreshold) expected += data[i];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Dataset: " << N << " f64, threshold " << kThreshold
              << ", expected sum " << expected << "\n";
    std::cout << "GCC " << __VERSION__ << ", -O3 -march=native, pinned core 0\n";
    std::cout << "Warmup: " << kWarmup << ", measured: " << kRuns << "\n\n";

    const sz kLanes = Engine<f64>::kLanes;

    // Interpreter
    {
        Engine<f64> e;
        sz segId = e.AddSegment(data.data(), N);
        std::vector<u32> code = {
            Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw,
            Instruction::VFilterGt(1, 0, 3).raw,
            Instruction::VSum(5, 1).raw,
            Instruction::Addf(0, 0, 5).raw,
            Instruction::Add(1, 1, static_cast<u16>(kLanes)).raw,
            Instruction::Cmp(1, 2).raw,
            Instruction::Jnz(-6).raw, Instruction::Halt().raw,
        };
        std::vector<f64> times; f64 result = 0;
        for (int i = 0; i < kWarmup + kRuns; ++i) {
            e.ScalarReg(0) = 0; e.ScalarReg(1) = 0;
            e.ScalarReg(2) = N; e.ScalarReg(3) = std::bit_cast<u64>(kThreshold);
            e.LoadProgram(code);
            auto t0 = std::chrono::high_resolution_clock::now();
            e.Run();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= kWarmup)
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
            result = std::bit_cast<f64>(e.ScalarReg(0));
        }
        std::sort(times.begin(), times.end());
        f64 med = times[times.size()/2];
        std::cout << "Interpreter: " << med << " us, " << (N/(med/1e6)/1e6)
                  << " M/s, sum=" << result
                  << " (" << (std::fabs(result-expected) < expected * 1e-12 ? "OK":"FAIL") << ")\n";
    }

    // JIT
    {
        Engine<f64> e;
        sz segId = e.AddSegment(data.data(), N);
        std::vector<u32> code = {
            Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw,
            Instruction::VFilterGt(1, 0, 3).raw,
            Instruction::VSum(5, 1).raw,
            Instruction::Addf(0, 0, 5).raw,
            Instruction::Add(1, 1, static_cast<u16>(kLanes)).raw,
            Instruction::Cmp(1, 2).raw,
            Instruction::Jnz(-6).raw, Instruction::Halt().raw,
        };
        auto compiler = codegen::CreateJitCompiler();
        if (!compiler) { std::cout << "JIT: SKIP\n"; return 0; }
        codegen::JitFunction func;
        if (!compiler->Compile(code.data(), code.size(), func) || !func.IsValid())
            { std::cout << "JIT: COMPILE FAIL\n"; return 1; }
        std::cout << "JIT code: " << func.CodeSize << " bytes\n";

        std::vector<f64> times; f64 result = 0;
        for (int i = 0; i < kWarmup + kRuns; ++i) {
            RegFile rf; rf.Scalar(0)=0; rf.Scalar(1)=0;
            rf.Scalar(2)=N; rf.Scalar(3)=std::bit_cast<u64>(kThreshold);
            f64* sb = data.data(); u64 sc = N;
            auto t0 = std::chrono::high_resolution_clock::now();
            func.Entry(&rf, &sb, &sc);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= kWarmup)
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
            result = std::bit_cast<f64>(rf.Scalar(0));
        }
        std::sort(times.begin(), times.end());
        f64 med = times[times.size()/2];
        std::cout << "JIT:         " << med << " us, " << (N/(med/1e6)/1e6)
                  << " M/s, sum=" << result
                  << " (" << (std::fabs(result-expected) < expected * 1e-12 ? "OK":"FAIL") << ")\n";
        compiler->Release(func);
    }

    // Native
    {
        std::vector<f64> times; f64 result = 0;
        for (int i = 0; i < kWarmup + kRuns; ++i) {
            f64 sum = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (sz j = 0; j < N; ++j)
                if (data[j] > kThreshold) sum += data[j];
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= kWarmup)
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
            result = sum;
        }
        std::sort(times.begin(), times.end());
        f64 med = times[times.size()/2];
        std::cout << "Native C++:  " << med << " us, " << (N/(med/1e6)/1e6)
                  << " M/s, sum=" << result
                  << " (" << (std::fabs(result-expected) < expected * 1e-12 ? "OK":"FAIL") << ")\n";
    }

    std::cout << "\n=== Reproduce ===\n";
    std::cout << "g++ -std=c++20 -O3 -march=native -I include examples/validate.cpp -o build/validate\n";
    std::cout << "taskset -c 0 ./build/validate\n";
    return 0;
}
