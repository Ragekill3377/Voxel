#include "voxel/voxel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <atomic>

using namespace voxel;

// Test macros: each TEST block has a boolean _ok flag.
// CHECK gates on _ok so subsequent checks are skipped after first failure.
// No goto/break needed avoids crossing try boundaries.
#define TEST(name) \
    std::cout << "  " << name << "... "; \
    { bool _ok = true; try {

#define CHECK(cond) \
    if (_ok && !(cond)) { std::cout << "FAIL (" << __LINE__ << ")\n"; _ok = false; } else (void)0

#define ENDTEST \
    } catch(...) { std::cout << "EXCEPTION\n"; _ok = false; } \
    if (_ok) { std::cout << "PASS\n"; passed++; } else { failed++; } }

int main()
{
    int passed = 0;
    int failed = 0;

    std::cout << "=== VoxelVM Comprehensive Test Suite ===\n\n";

    // ========================================================================
    // 1. Types & TypeTraits
    // ========================================================================
    TEST("Types & TypeTraits")
        CHECK(TypeTraits<f64>::kTypeId == DataType::Float64);
        CHECK(TypeTraits<f64>::kSize == 8);
        CHECK(TypeTraits<f64>::kIsFloating == true);
        CHECK(TypeTraits<f64>::kIsIntegral == false);

        CHECK(TypeTraits<i32>::kTypeId == DataType::Int32);
        CHECK(TypeTraits<i32>::kSize == 4);
        CHECK(TypeTraits<u64>::kTypeId == DataType::Uint64);

        CHECK(TypeSize(DataType::Int32) == 4);
        CHECK(TypeSize(DataType::Float64) == 8);
        CHECK(TypeSize(DataType::Int8) == 1);
        CHECK(TypeSize(DataType::Null) == 0);

        CHECK(IsFloatingType(DataType::Float32));
        CHECK(IsFloatingType(DataType::Float64));
        CHECK(!IsFloatingType(DataType::Int32));
        CHECK(IsNumericType(DataType::Int64));
        CHECK(IsNumericType(DataType::Bool));
        CHECK(IsIntegralType(DataType::Uint32));

        CHECK(PromoteType(DataType::Int32, DataType::Float64) == DataType::Float64);
        CHECK(PromoteType(DataType::Int32, DataType::Int32) == DataType::Int32);
    ENDTEST

    // ========================================================================
    // 2. Arena Allocator
    // ========================================================================
    TEST("Arena Allocator")
        Arena arena(65536);
        sz capBefore = arena.Capacity();
        CHECK(capBefore >= 65536);

        int* ptr = arena.AllocMany<int>(1000);
        CHECK(ptr != nullptr);

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        CHECK((addr & 63) == 0);

        for (int i = 0; i < 1000; ++i) ptr[i] = i * 7;
        for (int i = 0; i < 1000; ++i) CHECK(ptr[i] == i * 7);

        sz used = arena.Used();
        CHECK(used > 0);

        sz cap = arena.Capacity();
        CHECK(cap >= used);

        arena.Reset();
        sz usedAfter = arena.Used();
        CHECK(usedAfter == 0);
    ENDTEST

    // ========================================================================
    // 3. RegFile
    // ========================================================================
    TEST("RegFile")
        RegFile rf;

        rf.Scalar(0) = 42;
        CHECK(rf.Scalar(0) == 42);
        CHECK(rf.ScalarAs<i64>(0) == 42);

        double pi = 3.14;
        u64 piBits = std::bit_cast<u64>(pi);
        rf.SetScalarT<f64>(1, pi);
        CHECK(rf.Scalar(1) == piBits);
        f64 recovered = rf.ScalarAs<f64>(1);
        CHECK(std::abs(recovered - 3.14) < 1e-12);

        rf.ClearVector(0);
        f64* vec = rf.VecLanes<f64>(0);
        vec[0] = 1.0; vec[1] = 2.0; vec[2] = 3.0; vec[3] = 4.0;
        const f64* rvec = rf.VecLanes<f64>(0);
        CHECK(rvec[0] == 1.0);
        CHECK(rvec[1] == 2.0);
        CHECK(rvec[2] == 3.0);
        CHECK(rvec[3] == 4.0);

        rf.SetFlag(RegFile::Flag::Zero);
        CHECK(rf.Test(RegFile::Flag::Zero));
        rf.ClearAllFlags();
        CHECK(!rf.Test(RegFile::Flag::Zero));

        rf.Reset();
        CHECK(rf.Scalar(0) == 0);
    ENDTEST

    // ========================================================================
    // 4. Instruction Encoding
    // ========================================================================
    TEST("Instruction Encoding")
        Instruction vload = Instruction::VLoad(0, 1, 3, 4);
        CHECK(vload.Op() == Opcode::VLOAD);
        CHECK(vload.Rd() == 0);
        CHECK(vload.Ra() == 1);
        u16 imm = vload.Imm12();
        u8 segId = (imm >> 8) & 0xF;
        u8 count = imm & 0xFF;
        CHECK(segId == 3);
        CHECK(count == 4);

        Instruction mov = Instruction::Mov(5, -42);
        CHECK(mov.Op() == Opcode::MOV);
        CHECK(mov.Rd() == 5);
        CHECK(mov.Simm12() == -42);

        Instruction jmp = Instruction::Jmp(-10);
        CHECK(jmp.Op() == Opcode::JMP);
        CHECK(jmp.Simm12() == -10);

        Instruction add = Instruction::Add(2, 3, 4);
        CHECK(add.Op() == Opcode::ADD);
        CHECK(add.Rd() == 2);
        CHECK(add.Ra() == 3);
        CHECK(add.Rb() == 4);

        Instruction halt = Instruction::Halt();
        CHECK(halt.Op() == Opcode::HALT);

        Instruction nop = Instruction::Nop();
        CHECK(nop.Op() == Opcode::NOP);

        Instruction trap = Instruction::Trap();
        CHECK(trap.Op() == Opcode::TRAP);

        Instruction addf = Instruction::Addf(1, 2, 3);
        CHECK(addf.Op() == Opcode::ADDF);
        CHECK(addf.Rd() == 1);
        CHECK(addf.Ra() == 2);
        CHECK(addf.Rb() == 3);

        Instruction cmpf = Instruction::Cmpf(1, 2);
        CHECK(cmpf.Op() == Opcode::CMPF);

        Instruction vsum = Instruction::VSum(3, 1);
        CHECK(vsum.Op() == Opcode::VSUM);
        CHECK(vsum.Rd() == 3);
        CHECK(vsum.Ra() == 1);
    ENDTEST

    // ========================================================================
    // 5. Assembler
    // ========================================================================
    TEST("Assembler")
        Assembler as;
        CHECK(as.Offset() == 0);

        as.Mov(0, 0);
        CHECK(as.Offset() == 1);

        as.VLoad(0, 1, 0, 0);
        CHECK(as.Offset() == 2);

        as.VFilterGt(1, 0, 3);
        CHECK(as.Offset() == 3);

        as.VSum(5, 1);
        CHECK(as.Offset() == 4);

        as.Addf(0, 0, 5);
        CHECK(as.Offset() == 5);

        as.Halt();
        CHECK(as.Offset() == 6);

        bool finalized = as.Finalize();
        CHECK(finalized);

        const auto& code = as.GetCode();
        CHECK(code.size() == 6);

        Instruction i0(code[0]);
        CHECK(i0.Op() == Opcode::MOV);
        Instruction i1(code[1]);
        CHECK(i1.Op() == Opcode::VLOAD);
        Instruction iLast(code[5]);
        CHECK(iLast.Op() == Opcode::HALT);
    ENDTEST

    // ========================================================================
    // 6. Verifier
    // ========================================================================
    TEST("Verifier")
        Verifier vrf;

        {
            std::vector<u32> valid = {
                Instruction::Mov(0, 0).raw,
                Instruction::Halt().raw,
            };
            auto r = vrf.Verify(valid.data(), valid.size(), 16);
            CHECK(r.Ok());
        }

        {
            std::vector<u32> badBranch = {
                Instruction::Mov(0, 0).raw,
                Instruction::Jmp(-10).raw,
                Instruction::Halt().raw,
            };
            auto r = vrf.Verify(badBranch.data(), badBranch.size(), 16);
            CHECK(!r.Ok());
            CHECK(r.error == VerifyError::BranchOutOfBounds);
        }

        {
            std::vector<u32> noHalt = {
                Instruction::Mov(0, 0).raw,
            };
            auto r = vrf.Verify(noHalt.data(), noHalt.size(), 16);
            CHECK(r.Ok());
        }
    ENDTEST

    // ========================================================================
    // 7. Engine / f64
    // ========================================================================
    TEST("Engine / f64")
        std::vector<f64> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        f64 threshold = 4.0;
        f64 expectedSum = 5.0 + 6.0 + 7.0 + 8.0;

        Engine<f64> engine;
        engine.ScalarReg(0) = 0;
        engine.ScalarReg(1) = 0;
        engine.ScalarReg(2) = 8;
        engine.ScalarReg(3) = std::bit_cast<u64>(threshold);

        sz segId = engine.AddSegment(data.data(), data.size());

        constexpr i16 kL = static_cast<i16>(Engine<f64>::kLanes);
        std::vector<u32> code = {
            Instruction::Mov(4, kL).raw,
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

        f64 result = std::bit_cast<f64>(engine.ScalarReg(0));
        CHECK(std::abs(result - expectedSum) < 1e-10);
    ENDTEST

    // ========================================================================
    // 8. Engine / i64
    // ========================================================================
    TEST("Engine / i64")
        std::vector<i64> data2 = {10, 20, 30, 40};
        i64 threshold2 = 15;
        i64 expectedSum2 = 20 + 30 + 40;

        Engine<i64> engine2;
        engine2.ScalarReg(0) = 0;
        engine2.ScalarReg(1) = 0;
        engine2.ScalarReg(2) = 4;
        engine2.ScalarReg(3) = static_cast<u64>(threshold2);

        sz segId2 = engine2.AddSegment(data2.data(), data2.size());

        constexpr i16 kL2 = static_cast<i16>(Engine<i64>::kLanes);
        std::vector<u32> code2 = {
            Instruction::Mov(4, kL2).raw,
            Instruction::VLoad(0, 1, static_cast<u8>(segId2), 0).raw,
            Instruction::VFilterGt(1, 0, 3).raw,
            Instruction::VSum(5, 1).raw,
            Instruction::Add(0, 0, 5).raw,
            Instruction::Add(1, 1, 4).raw,
            Instruction::Cmp(1, 2).raw,
            Instruction::Jnz(-6).raw,
            Instruction::Halt().raw,
        };

        engine2.LoadProgram(code2);
        engine2.Run();

        i64 result2 = static_cast<i64>(engine2.ScalarReg(0));
        CHECK(result2 == expectedSum2);
    ENDTEST

    // ========================================================================
    // 9. SIMD Scalar / f64
    // ========================================================================
    TEST("SIMD Scalar / f64")
        f64 simd_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        f64 ssum = simd::scalar::scalar_sum_f64(simd_data, 8);
        CHECK(std::abs(ssum - 36.0) < 1e-12);

        f64 filtered[8] = {};
        sz passCount = simd::scalar::scalar_filter_gt_f64(filtered, simd_data, 4.0, 8);
        CHECK(passCount == 4);
        CHECK(std::abs(filtered[0] - 5.0) < 1e-12);
        CHECK(std::abs(filtered[3] - 8.0) < 1e-12);

        f64 a[4] = {1.0, 2.0, 3.0, 4.0};
        f64 b[4] = {5.0, 6.0, 7.0, 8.0};
        f64 dst[4] = {};
        simd::scalar::scalar_add_f64(dst, a, b, 4);
        CHECK(std::abs(dst[0] - 6.0) < 1e-12);
        CHECK(std::abs(dst[1] - 8.0) < 1e-12);
        CHECK(std::abs(dst[2] - 10.0) < 1e-12);
        CHECK(std::abs(dst[3] - 12.0) < 1e-12);
    ENDTEST

    // ========================================================================
    // 10. SIMD Scalar / f32
    // ========================================================================
    TEST("SIMD Scalar / f32")
        f32 f32data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

        f32 fsum = simd::scalar::scalar_sum_f32(f32data, 8);
        CHECK(std::abs(fsum - 36.0f) < 1e-5f);
    ENDTEST

    // ========================================================================
    // 11. NullBitmap
    // ========================================================================
    TEST("NullBitmap")
        NullBitmap bm(72);
        CHECK(bm.NullCount() == 0);
        CHECK(bm.AllValid());

        for (sz i = 8; i < 72; i += 8) bm.SetNull(i);
        CHECK(bm.NullCount() == 8);
        CHECK(bm.IsNull(8) == true);
        CHECK(bm.IsNull(0) == false);
        CHECK(bm.IsValid(0) == true);
    ENDTEST

    // ========================================================================
    // 12. DictionaryEncoding<f64>
    // ========================================================================
    TEST("DictionaryEncoding<f64>")
        f64 dictdata[5] = {1.0, 2.0, 1.0, 2.0, 3.0};
        encoding::DictionaryEncoding<f64> dict;
        dict.Build(dictdata, 5);

        CHECK(dict.Cardinality() == 3);
        CHECK(dict.RowCount() == 5);

        f64 decoded[5] = {};
        dict.DecodeBatch(decoded, 0, 5);
        for (sz i = 0; i < 5; ++i)
            CHECK(std::abs(decoded[i] - dictdata[i]) < 1e-12);
    ENDTEST

    // ========================================================================
    // 13. RLEEncoding<f64>
    // ========================================================================
    TEST("RLEEncoding<f64>")
        f64 rledata[6] = {1.0, 1.0, 1.0, 2.0, 2.0, 3.0};
        encoding::RLEEncoding<f64> rle;
        rle.Build(rledata, 6);

        CHECK(rle.RunCount() == 3);
        CHECK(rle.RowCount() == 6);

        f64 rledecoded[6] = {};
        rle.DecodeBatch(rledecoded, 0, 6);
        for (sz i = 0; i < 6; ++i)
            CHECK(std::abs(rledecoded[i] - rledata[i]) < 1e-12);
    ENDTEST

    // ========================================================================
    // 14. VectorFilter
    // ========================================================================
    TEST("VectorFilter")
        f64 vfdata[8] = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};
        ops::VectorFilter<f64> filter;
        auto result = filter.ApplyGT(vfdata, 8, 4.0);

        CHECK(result.PassCount == 4);

        CHECK(!result.Bitmap.empty());
        u64 word = result.Bitmap[0];
        bool bit2 = (word >> 2) & 1;
        bool bit3 = (word >> 3) & 1;
        bool bit6 = (word >> 6) & 1;
        bool bit7 = (word >> 7) & 1;
        CHECK(bit2 == true);
        CHECK(bit3 == true);
        CHECK(bit6 == true);
        CHECK(bit7 == true);
    ENDTEST

    // ========================================================================
    // 15. HashAggregator
    // ========================================================================
    TEST("HashAggregator")
        Arena aggarena;
        ops::HashAggregator<i32, i64> agg(aggarena);
        agg.Init(16);

        i32 keys[5]   = {0, 1, 0, 1, 0};
        i64 values[5] = {10, 20, 30, 40, 50};
        agg.Accumulate(keys, values, 5);

        CHECK(agg.GroupCount() == 2);

        auto* s0 = agg.Lookup(0);
        CHECK(s0 != nullptr);
        CHECK(s0->Sum == 90);
        CHECK(s0->Count == 3);

        auto* s1 = agg.Lookup(1);
        CHECK(s1 != nullptr);
        CHECK(s1->Sum == 60);
        CHECK(s1->Count == 2);
    ENDTEST

    // ========================================================================
    // 16. SortOperator
    // ========================================================================
    TEST("SortOperator")
        f64 sortdata[8] = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
        Segment<f64> seg(sortdata, 8);
        std::vector<u32> indices(8);
        ops::SortOperator::SortAscending(seg, indices.data());

        f64 expected_sorted[8] = {1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0};
        for (sz i = 0; i < 8; ++i)
            CHECK(std::abs(sortdata[indices[i]] - expected_sorted[i]) < 1e-12);
    ENDTEST

    // ========================================================================
    // 17. HashJoin
    // ========================================================================
    TEST("HashJoin")
        Arena joarena;
        ops::HashJoin<i32> join(joarena);

        i32 buildKeys[4] = {1, 2, 3, 4};
        join.Build(buildKeys, 4);

        i32 probeKeys[3] = {2, 4, 6};
        u32 leftMatches[3];
        u32 rightMatches[3];
        sz matchCount = 0;
        join.Probe(probeKeys, 3, leftMatches, rightMatches, matchCount);

        CHECK(matchCount == 2);

        bool found2 = false, found4 = false;
        for (sz i = 0; i < matchCount; ++i) {
            if (probeKeys[leftMatches[i]] == 2 && rightMatches[i] == 1) found2 = true;
            if (probeKeys[leftMatches[i]] == 4 && rightMatches[i] == 3) found4 = true;
        }
        CHECK(found2);
        CHECK(found4);
    ENDTEST

    // ========================================================================
    // 18. ThreadPool
    // ========================================================================
    TEST("ThreadPool")
        ThreadPool pool(2);
        CHECK(pool.ThreadCount() == 2);

        std::atomic<int> counter{0};
        for (int i = 0; i < 4; ++i) {
            pool.Enqueue([&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }

        pool.WaitAll();
        CHECK(counter.load(std::memory_order_relaxed) == 4);
    ENDTEST

    // ========================================================================
    // 19. MurmurHash64A
    // ========================================================================
    TEST("MurmurHash64A")
        const char* hashstr = "hello";
        u64 h1 = ops::MurmurHash64A(hashstr, std::strlen(hashstr), 0);
        u64 h2 = ops::MurmurHash64A(hashstr, std::strlen(hashstr), 0);

        CHECK(h1 != 0);
        CHECK(h1 == h2);
    ENDTEST

    // ========================================================================
    // 20. BitUtils
    // ========================================================================
    TEST("BitUtils")
        CHECK(AlignUp(42, 64) == 64);
        CHECK(AlignUp(64, 64) == 64);
        CHECK(AlignUp(65, 64) == 128);

        CHECK(PopCount64(0xFFULL) == 8);
        CHECK(PopCount64(0x0ULL) == 0);
        CHECK(PopCount64(0x1010101010101010ULL) == 8);

        CHECK(CountLeadingZeros64(1) == 63);
        CHECK(CountLeadingZeros64(0x8000000000000000ULL) == 0);

        CHECK(IsPow2(64) == true);
        CHECK(IsPow2(1) == true);
        CHECK(IsPow2(63) == false);
        CHECK(IsPow2(0) == false);
    ENDTEST

    // ========================================================================
    // 21. DataBlock
    // ========================================================================
    TEST("DataBlock")
        DataBlock<f64> block(1024);
        CHECK(block.Capacity == 1024);
        CHECK(block.Count == 0);

        f64 blockvals[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        block.Append(blockvals, 8);
        CHECK(block.Count == 8);
        CHECK(std::abs(block[0] - 1.0) < 1e-12);
        CHECK(std::abs(block[7] - 8.0) < 1e-12);

        block.Clear();
        CHECK(block.Count == 0);
    ENDTEST

    // ========================================================================
    // 22. Profiling
    // ========================================================================
    TEST("Profiling")
        InstructionProfiler prof;
        prof.RecordOpcode(Opcode::MOV, 100, 200);
        prof.RecordOpcode(Opcode::ADD, 200, 300);
        prof.RecordOpcode(Opcode::HALT, 300, 400);

        CHECK(prof.TotalInstructions == 3);
        CHECK(prof.TotalCycles == 300);
    ENDTEST

    std::cout << "\n========================================\n";
    std::cout << passed << "/" << (passed + failed) << " tests passed\n";
    std::cout << "========================================\n";

    return failed > 0 ? 1 : 0;
}
