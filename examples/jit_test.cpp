#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <bit>
#include <chrono>
#include <cstdint>

using namespace voxel;

int main()
{
    constexpr sz N          = 1'000'000;
    constexpr f64 kThreshold = 500.0;
    constexpr sz kLanesVal   = Engine<f64>::kLanes;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<f64> dist(0.0, 1000.0);
    std::vector<f64> data(N);
    for (sz i = 0; i < N; ++i) data[i] = dist(rng);

    f64 expectedSum = 0.0;
    for (sz i = 0; i < N; ++i)
        if (data[i] > kThreshold) expectedSum += data[i];

    std::cout << std::setprecision(12);
    std::cout << "Ground truth: " << expectedSum << "\n\n";

    // ================================================================
    // 1. Interpreter
    // ================================================================
    {
        Engine<f64> engine;
        engine.ScalarReg(0) = 0;
        engine.ScalarReg(1) = 0;
        engine.ScalarReg(2) = N;
        engine.ScalarReg(3) = std::bit_cast<u64>(kThreshold);
        engine.ScalarReg(4) = kLanesVal;
        sz segId = engine.AddSegment(data.data(), N);

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

        auto t0 = std::chrono::high_resolution_clock::now();
        engine.Run();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        f64 result = std::bit_cast<f64>(engine.ScalarReg(0));
        f64 delta  = std::fabs(result - expectedSum);
        bool pass  = delta < expectedSum * 1e-12;

        std::cout << "Interpreter: " << result << "  " << (pass ? "OK" : "FAIL")
                  << "  " << us << " us  " << (N / (us / 1e6) / 1e6) << " M elem/s\n";
    }

    // ================================================================
    // 2. JIT
    // ================================================================
    {
        auto compiler = codegen::CreateJitCompiler();
        if (!compiler) {
            std::cout << "JIT: SKIP (no backend)\n";
        } else {
            sz segId = 0;
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

            auto tc0 = std::chrono::high_resolution_clock::now();
            codegen::JitFunction func;
            bool compiled = compiler->Compile(code.data(), code.size(), func);
            auto tc1 = std::chrono::high_resolution_clock::now();
            auto compileUs = std::chrono::duration_cast<std::chrono::microseconds>(tc1 - tc0).count();

            if (!compiled || !func.IsValid()) {
                std::cout << "JIT: COMPILE FAIL\n";
            } else {
                RegFile regfile;
                regfile.Scalar(0) = 0;
                regfile.Scalar(1) = 0;
                regfile.Scalar(2) = N;
                regfile.Scalar(3) = std::bit_cast<u64>(kThreshold);
                regfile.Scalar(4) = kLanesVal;

                f64* segBase = data.data();
                u64  segCount = N;

                auto te0 = std::chrono::high_resolution_clock::now();
                func.Entry(&regfile, &segBase, &segCount);
                auto te1 = std::chrono::high_resolution_clock::now();
                auto execUs = std::chrono::duration_cast<std::chrono::microseconds>(te1 - te0).count();

                f64 result = std::bit_cast<f64>(regfile.Scalar(0));
                f64 delta  = std::fabs(result - expectedSum);
                bool pass  = delta < expectedSum * 1e-12;

                std::cout << "JIT compile:  " << compileUs << " us  (" << func.CodeSize << " bytes)\n";
                std::cout << "JIT execute:  " << execUs << " us  "
                          << (N / (execUs / 1e6) / 1e6) << " M elem/s\n";
                std::cout << "JIT total:    " << (compileUs + execUs) << " us\n";
                std::cout << "JIT result:   " << result << "  " << (pass ? "OK" : "FAIL") << "\n";

                compiler->Release(func);
            }
        }
    }

    // ================================================================
    // 3. Native C++
    // ================================================================
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        f64 sum = 0.0;
        for (sz i = 0; i < N; ++i)
            if (data[i] > kThreshold) sum += data[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        bool pass = std::fabs(sum - expectedSum) < expectedSum * 1e-12;
        std::cout << "\nNative C++:   " << sum << "  " << (pass ? "OK" : "FAIL")
                  << "  " << us << " us  " << (N / (us / 1e6) / 1e6) << " M elem/s\n";
    }

    return 0;
}
