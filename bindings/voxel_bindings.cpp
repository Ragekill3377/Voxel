#include "voxel/voxel.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace voxel;

// ============================================================================
// Helper: numpy array -> Segment
// ============================================================================
template<typename T>
Segment<T> NumpyToSegment(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    return Segment<T>(static_cast<T*>(buf.ptr), static_cast<sz>(buf.size));
}

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(voxel_py, m) {
    m.doc() = "VoxelVM Python Bindings - High-performance columnar analytics engine";

    // ---- Core Types ----
    py::enum_<DataType>(m, "DataType")
        .value("Null", DataType::Null).value("Bool", DataType::Bool)
        .value("Int8", DataType::Int8).value("Int16", DataType::Int16)
        .value("Int32", DataType::Int32).value("Int64", DataType::Int64)
        .value("Uint8", DataType::Uint8).value("Uint16", DataType::Uint16)
        .value("Uint32", DataType::Uint32).value("Uint64", DataType::Uint64)
        .value("Float32", DataType::Float32).value("Float64", DataType::Float64)
        .value("Decimal128", DataType::Decimal128)
        .value("String", DataType::String).value("Binary", DataType::Binary)
        .value("Date", DataType::Date).value("Timestamp", DataType::Timestamp)
        .export_values();

    // ---- Opcode enum ----
    py::enum_<Opcode>(m, "Opcode")
        .value("NOP", Opcode::NOP).value("HALT", Opcode::HALT)
        .value("MOV", Opcode::MOV).value("ADD", Opcode::ADD)
        .value("SUB", Opcode::SUB).value("MUL", Opcode::MUL)
        .value("CMP", Opcode::CMP).value("JMP", Opcode::JMP)
        .value("JZ", Opcode::JZ).value("JNZ", Opcode::JNZ)
        .value("VLOAD", Opcode::VLOAD).value("VSTORE", Opcode::VSTORE)
        .value("VADD", Opcode::VADD).value("VFILTER_GT", Opcode::VFILTER_GT)
        .value("VSUM", Opcode::VSUM).value("ADDF", Opcode::ADDF)
        .export_values();

    // ---- Filter Comparison Constants ----
    m.attr("CMP_EQ") = kCmpEQ; m.attr("CMP_NE") = kCmpNE;
    m.attr("CMP_LT") = kCmpLT; m.attr("CMP_LE") = kCmpLE;
    m.attr("CMP_GT") = kCmpGT; m.attr("CMP_GE") = kCmpGE;

    // ---- Instruction ----
    py::class_<Instruction>(m, "Instruction")
        .def(py::init<>())
        .def_property("raw", [](const Instruction& i) -> u32 { return i.raw; },
                             [](Instruction& i, u32 v) { i.raw = v; })
        .def("op", &Instruction::Op)
        .def("rd", &Instruction::Rd).def("ra", &Instruction::Ra).def("rb", &Instruction::Rb)
        .def("imm12", &Instruction::Imm12)
        .def_static("nop", &Instruction::Nop)
        .def_static("halt", &Instruction::Halt)
        .def_static("mov", &Instruction::Mov)
        .def_static("add", &Instruction::Add)
        .def_static("sub", &Instruction::Sub)
        .def_static("mul", &Instruction::Mul)
        .def_static("addf", &Instruction::Addf)
        .def_static("cmp", &Instruction::Cmp)
        .def_static("jmp", &Instruction::Jmp)
        .def_static("jz", &Instruction::Jz)
        .def_static("jnz", &Instruction::Jnz)
        .def_static("vload", &Instruction::VLoad)
        .def_static("vstore", &Instruction::VStore)
        .def_static("vadd", &Instruction::VAdd)
        .def_static("vsub", &Instruction::VSub)
        .def_static("vmul", &Instruction::VMul)
        .def_static("vfilter", &Instruction::VFilter)
        .def_static("vfilter_gt", &Instruction::VFilterGt)
        .def_static("vfilter_ge", &Instruction::VFilterGe)
        .def_static("vsum", &Instruction::VSum)
        .def_static("vmin", &Instruction::VMin)
        .def_static("vmax", &Instruction::VMax)
        .def_static("vcount", &Instruction::VCount)
        .def_static("movr", &Instruction::Movr)
        .def_static("jle", &Instruction::Jle)
        .def_static("jge", &Instruction::Jge);

    // ---- Segment (f64) ----
    py::class_<Segment<f64>>(m, "SegmentF64")
        .def(py::init<>())
        .def(py::init<f64*, sz>())
        .def("__getitem__", [](Segment<f64>& s, sz i) -> f64 { return s[i]; })
        .def("__len__", [](Segment<f64>& s) -> sz { return s.Size(); })
        .def_property_readonly("size", [](Segment<f64>& s) -> sz { return s.Size(); });

    // ---- Engine<f64> ----
    py::class_<Engine<f64>>(m, "EngineF64")
        .def(py::init<>())
        .def("add_segment", [](Engine<f64>& e, py::array_t<f64> arr) {
            auto buf = arr.request();
            return e.AddSegment(static_cast<f64*>(buf.ptr), static_cast<sz>(buf.size));
        })
        .def("load_program", [](Engine<f64>& e, std::vector<u32> code) {
            e.LoadProgram(code);
        })
        .def("run", &Engine<f64>::Run)
        .def("run_parallel", [](Engine<f64>& e, sz segId, u32 threads, u8 resultReg) {
            e.RunParallel(segId, threads, resultReg);
        }, py::arg("seg_id"), py::arg("threads") = 0, py::arg("result_reg") = 0)
        .def("get_scalar", [](Engine<f64>& e, sz i) { return e.ScalarReg(i); })
        .def("set_scalar", [](Engine<f64>& e, sz i, u64 v) { e.ScalarReg(i) = v; })
        .def("get_scalar_f64", [](Engine<f64>& e, sz i) -> f64 {
            return std::bit_cast<f64>(e.ScalarReg(i));
        })
        .def("set_scalar_f64", [](Engine<f64>& e, sz i, f64 v) {
            e.ScalarReg(i) = std::bit_cast<u64>(v);
        })
        .def_property_readonly_static("k_lanes", [](py::object) { return Engine<f64>::kLanes; });

    // ---- Engine<i64> ----
    py::class_<Engine<i64>>(m, "EngineI64")
        .def(py::init<>())
        .def("add_segment", [](Engine<i64>& e, py::array_t<i64> arr) {
            auto buf = arr.request();
            return e.AddSegment(static_cast<i64*>(buf.ptr), static_cast<sz>(buf.size));
        })
        .def("load_program", [](Engine<i64>& e, std::vector<u32> code) {
            e.LoadProgram(code);
        })
        .def("run", &Engine<i64>::Run)
        .def("get_scalar", [](Engine<i64>& e, sz i) { return e.ScalarReg(i); })
        .def("set_scalar", [](Engine<i64>& e, sz i, u64 v) { e.ScalarReg(i) = v; })
        .def("get_scalar_i64", [](Engine<i64>& e, sz i) -> i64 {
            return static_cast<i64>(e.ScalarReg(i));
        })
        .def_property_readonly_static("k_lanes", [](py::object) { return Engine<i64>::kLanes; });

    // ---- NullBitmap ----
    py::class_<NullBitmap>(m, "NullBitmap")
        .def(py::init<sz>())
        .def("is_null", &NullBitmap::IsNull)
        .def("set_null", &NullBitmap::SetNull)
        .def("set_valid", &NullBitmap::SetValid)
        .def("null_count", &NullBitmap::NullCount)
        .def("valid_count", &NullBitmap::ValidCount)
        .def("set_all_valid", &NullBitmap::SetAllValid)
        .def("set_all_null", &NullBitmap::SetAllNull);

    // ---- DictionaryEncoding<f64> ----
    py::class_<encoding::DictionaryEncoding<f64>>(m, "DictionaryEncodingF64")
        .def(py::init<>())
        .def("build", [](encoding::DictionaryEncoding<f64>& d, py::array_t<f64> arr) {
            auto buf = arr.request();
            d.Build(static_cast<f64*>(buf.ptr), buf.size);
        })
        .def_property_readonly("dictionary", [](encoding::DictionaryEncoding<f64>& d) {
            return py::array_t<f64>(d.Dictionary.size(), d.Dictionary.data());
        })
        .def_property_readonly("memory_usage", &encoding::DictionaryEncoding<f64>::MemoryUsage);

    // ---- RLEEncoding<f64> ----
    py::class_<encoding::RLEEncoding<f64>>(m, "RLEEncodingF64")
        .def(py::init<>())
        .def("build", [](encoding::RLEEncoding<f64>& r, py::array_t<f64> arr) {
            auto buf = arr.request();
            r.Build(static_cast<f64*>(buf.ptr), buf.size);
        })
        .def_property_readonly("memory_usage", &encoding::RLEEncoding<f64>::MemoryUsage);

    // ---- VectorFilter<f64> ----
    py::class_<ops::VectorFilter<f64>>(m, "VectorFilterF64")
        .def(py::init<>())
        .def("apply_gt", [](ops::VectorFilter<f64>& f, py::array_t<f64> arr, f64 threshold) {
            auto buf = arr.request();
            auto result = f.ApplyGT(static_cast<f64*>(buf.ptr), buf.size, threshold);
            return py::make_tuple(
                py::array_t<u64>(result.Bitmap.size(), result.Bitmap.data()),
                result.PassCount
            );
        });

    // ---- HashAggregator ----
    py::class_<ops::HashAggregator<u32, f64>>(m, "HashAggregator")
        .def(py::init<Arena&>())
        .def("init", &ops::HashAggregator<u32, f64>::Init)
        .def("accumulate", [](ops::HashAggregator<u32, f64>& a,
                               py::array_t<u32> keys, py::array_t<f64> values) {
            auto kb = keys.request(); auto vb = values.request();
            sz count = std::min(kb.size, vb.size);
            a.Accumulate(static_cast<u32*>(kb.ptr), static_cast<f64*>(vb.ptr), count);
        })
        .def("group_count", &ops::HashAggregator<u32, f64>::GroupCount);

    // ---- SortOperator ----
    m.def("sort_ascending", [](py::array_t<f64> arr) {
        auto buf = arr.request();
        sz count = buf.size;
        std::vector<u32> indices(count);
        for (sz i = 0; i < count; ++i) indices[i] = static_cast<u32>(i);
        Segment<f64> seg(static_cast<f64*>(buf.ptr), count);
        ops::SortOperator::SortAscending(seg, indices.data());
        return py::array_t<u32>(count, indices.data());
    });

    // ---- TopK ----
    m.def("topk_select", [](py::array_t<f64> arr, sz k, bool largest) {
        auto buf = arr.request();
        std::vector<f64> out(k);
        ops::TopKSelect(static_cast<f64*>(buf.ptr), buf.size, k, out.data(), largest);
        return py::array_t<f64>(k, out.data());
    });

    // ---- HashJoin ----
    py::class_<ops::HashJoin<u32>>(m, "HashJoin")
        .def(py::init<Arena&>())
        .def("build", [](ops::HashJoin<u32>& j, py::array_t<u32> keys) {
            auto buf = keys.request();
            j.Build(static_cast<u32*>(buf.ptr), buf.size);
        })
        .def("contains", &ops::HashJoin<u32>::Contains)
        .def("probe", [](ops::HashJoin<u32>& j, py::array_t<u32> probeKeys) {
            auto buf = probeKeys.request();
            sz count = buf.size;
            std::vector<u32> leftOut(count), rightOut(count);
            sz matches = 0;
            j.Probe(static_cast<u32*>(buf.ptr), count, leftOut.data(), rightOut.data(), matches);
            leftOut.resize(matches); rightOut.resize(matches);
            return py::make_tuple(
                py::array_t<u32>(matches, leftOut.data()),
                py::array_t<u32>(matches, rightOut.data()),
                matches
            );
        });

    // ---- ThreadPool ----
    py::class_<ThreadPool>(m, "ThreadPool")
        .def(py::init<u32>(), py::arg("num_threads") = 0)
        .def("enqueue", [](ThreadPool& tp, std::function<void()> task) {
            tp.Enqueue(std::move(task));
        })
        .def("wait_all", &ThreadPool::WaitAll)
        .def_property_readonly("thread_count", &ThreadPool::ThreadCount);

    // ---- Profiling ----
    py::class_<InstructionProfiler>(m, "InstructionProfiler")
        .def(py::init<>())
        .def("record", [](InstructionProfiler& p, u8 op, u64 cycles) {
            p.RecordOpcode(static_cast<Opcode>(op), cycles, cycles + 1);
        })
        .def("dump", [](InstructionProfiler& p) {
            std::ostringstream os;
            p.Dump(os);
            return os.str();
        });

    // ---- Platform Info ----
    m.def("cpu_info", []() {
        auto cpu = platform::QueryCpuInfo();
        return py::dict(
            py::arg("logical_cores") = cpu.LogicalCores,
            py::arg("physical_cores") = cpu.PhysicalCores,
            py::arg("l1_cache") = cpu.L1CacheSize,
            py::arg("l2_cache") = cpu.L2CacheSize,
            py::arg("l3_cache") = cpu.L3CacheSize,
            py::arg("has_avx2") = cpu.HasAVX2,
            py::arg("has_avx512") = cpu.HasAVX512F,
            py::arg("has_fma") = cpu.HasFMA
        );
    });

    // ---- JIT ----
    m.def("jit_compile", [](std::vector<u32> code) {
        auto compiler = codegen::CreateJitCompiler();
        if (!compiler) return py::dict(py::arg("compiled") = false);
        codegen::JitFunction func;
        bool ok = compiler->Compile(code.data(), code.size(), func);
        if (!ok || !func.IsValid()) return py::dict(py::arg("compiled") = false);
        return py::dict(
            py::arg("compiled") = true,
            py::arg("code_size") = func.CodeSize,
            py::arg("code_ptr") = reinterpret_cast<uintptr_t>(func.CodePtr)
        );
    });

    // ---- Version ----
    m.attr("__version__") = "1.0.0";
}
