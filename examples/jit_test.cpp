#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <bit>
#include <cstdint>

using namespace voxel;

int main()
{
    constexpr sz kLanes = Engine<f64>::kLanes;  // 4 doubles per 256-bit vector

    // Data: [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
    f64 data[] = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};
    sz  dataCount = sizeof(data) / sizeof(data[0]);

    f64 threshold = 4.0;

    // ================================================================
    // 1. Interpreter run
    // ================================================================
    Engine<f64> engine;
    engine.ScalarReg(0) = 0;                                      // R0 = accumulator
    engine.ScalarReg(1) = 0;                                      // R1 = offset index
    engine.ScalarReg(2) = dataCount;                               // R2 = total count
    engine.ScalarReg(3) = std::bit_cast<u64>(threshold);          // R3 = threshold
    engine.ScalarReg(4) = kLanes;                                  // R4 = kLanes

    sz segId = engine.AddSegment(data, dataCount);

    std::vector<u32> code;
    code.reserve(16);

    //  0: VLOAD V0, seg[R1..R1+4]
    code.push_back(Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw);
    //  1: VFILTER_GT V1, V0, R3
    code.push_back(Instruction::VFilterGt(1, 0, 3).raw);
    //  2: VSUM R5, V1
    code.push_back(Instruction::VSum(5, 1).raw);
    //  3: ADDF R0, R0, R5
    code.push_back(Instruction::Addf(0, 0, 5).raw);
    //  4: ADD R1, R1, 4
    code.push_back(Instruction::Add(1, 1, 4).raw);
    //  5: CMP R1, R2
    code.push_back(Instruction::Cmp(1, 2).raw);
    //  6: JNZ -6 = jump to pc 0
    code.push_back(Instruction::Jnz(-6).raw);
    //  7: HALT
    code.push_back(Instruction::Halt().raw);

    engine.LoadProgram(code);
    engine.Run();

    f64 interpResult = std::bit_cast<f64>(engine.ScalarReg(0));
    std::cout << std::setprecision(12);
    std::cout << "Interpreter result: " << interpResult << "\n";

    // ================================================================
    // 2. JIT compilation
    // ================================================================
    auto compiler = codegen::CreateJitCompiler();
    if (!compiler) {
        std::cout << "JIT SKIP (no backend available)\n";
        return 0;
    }

    codegen::JitFunction func;
    bool compiled = compiler->Compile(code.data(), code.size(), func);
    if (!compiled || !func.IsValid()) {
        std::cout << "JIT COMPILE FAIL\n";
        return 1;
    }

    std::cout << "JIT compiled: " << func.CodeSize << " bytes at " << func.CodePtr << "\n";

    // ================================================================
    // 3. JIT execution
    // ================================================================
    RegFile regfile;
    regfile.Scalar(0) = 0;                                      // R0 = accumulator
    regfile.Scalar(1) = 0;                                      // R1 = offset index
    regfile.Scalar(2) = dataCount;                               // R2 = total count
    regfile.Scalar(3) = std::bit_cast<u64>(threshold);          // R3 = threshold
    regfile.Scalar(4) = kLanes;                                  // R4 = kLanes

    // segmentsBase is a pointer to the segment data array (f64*[1])
    f64* segBase = data;
    u64  segCount = dataCount;

    func.Entry(&regfile, &segBase, &segCount);

    f64 jitResult = std::bit_cast<f64>(regfile.Scalar(0));
    std::cout << "JIT result:        " << jitResult << "\n";

    // ================================================================
    // 4. Verification
    // ================================================================
    f64 expectedSum = 0.0;
    for (sz i = 0; i < dataCount; ++i) {
        if (data[i] > threshold) expectedSum += data[i];
    }
    std::cout << "Expected sum:      " << expectedSum << "\n";

    bool interpPass = std::fabs(interpResult - expectedSum) < 1e-12;
    bool jitPass    = std::fabs(jitResult - expectedSum)    < 1e-12;

    std::cout << "Interpreter: " << (interpPass ? "PASS" : "FAIL") << "\n";
    std::cout << "JIT:         " << (jitPass    ? "PASS" : "FAIL") << "\n";

    compiler->Release(func);

    return (interpPass && jitPass) ? 0 : 1;
}
