#include "voxel/voxel.hpp"
#include "bindings/voxel_c.h"
#include <cstring>

using namespace voxel;

static thread_local char g_lastError[256] = "";

extern "C" {

// ================================================================
// Engine (f64)
// ================================================================
void* voxel_engine_create_f64(void) { return new Engine<f64>(); }

void voxel_engine_destroy(void* e) { delete static_cast<Engine<f64>*>(e); }

size_t voxel_engine_add_segment(void* e, const double* data, size_t count) {
    return static_cast<Engine<f64>*>(e)->AddSegment(const_cast<double*>(data), count);
}

void voxel_engine_load_program(void* e, const uint32_t* code, size_t len) {
    std::vector<u32> v(code, code + len);
    static_cast<Engine<f64>*>(e)->LoadProgram(v);
}

void voxel_engine_run(void* e) { static_cast<Engine<f64>*>(e)->Run(); }

void voxel_engine_run_parallel(void* e, size_t seg_id, uint32_t threads, uint8_t result_reg) {
    static_cast<Engine<f64>*>(e)->RunParallel(seg_id, threads, result_reg);
}

void voxel_engine_set_scalar(void* e, size_t reg, uint64_t value) {
    static_cast<Engine<f64>*>(e)->ScalarReg(reg) = value;
}

uint64_t voxel_engine_get_scalar(void* e, size_t reg) {
    return static_cast<Engine<f64>*>(e)->ScalarReg(reg);
}

void voxel_engine_set_scalar_f64(void* e, size_t reg, double value) {
    static_cast<Engine<f64>*>(e)->ScalarReg(reg) = std::bit_cast<u64>(value);
}

double voxel_engine_get_scalar_f64(void* e, size_t reg) {
    return std::bit_cast<f64>(static_cast<Engine<f64>*>(e)->ScalarReg(reg));
}

size_t voxel_engine_k_lanes(void) { return Engine<f64>::kLanes; }

// ================================================================
// Engine (i64)
// ================================================================
void* voxel_engine_create_i64(void) { return new Engine<i64>(); }

void voxel_engine_destroy_i64(void* e) { delete static_cast<Engine<i64>*>(e); }

size_t voxel_engine_add_segment_i64(void* e, const int64_t* data, size_t count) {
    return static_cast<Engine<i64>*>(e)->AddSegment(const_cast<int64_t*>(data), count);
}

void voxel_engine_load_program_i64(void* e, const uint32_t* code, size_t len) {
    std::vector<u32> v(code, code + len);
    static_cast<Engine<i64>*>(e)->LoadProgram(v);
}

void voxel_engine_run_i64(void* e) { static_cast<Engine<i64>*>(e)->Run(); }

void voxel_engine_set_scalar_i64(void* e, size_t reg, uint64_t value) {
    static_cast<Engine<i64>*>(e)->ScalarReg(reg) = value;
}

uint64_t voxel_engine_get_scalar_i64(void* e, size_t reg) {
    return static_cast<Engine<i64>*>(e)->ScalarReg(reg);
}

// ================================================================
// JIT
// ================================================================
double voxel_jit_run(const uint32_t* code, size_t code_len,
                     const double* data, size_t data_len, double threshold) {
    auto compiler = codegen::CreateJitCompiler();
    if (!compiler) return 0.0;
    codegen::JitFunction func;
    std::vector<u32> v(code, code + code_len);
    if (!compiler->Compile(v.data(), v.size(), func) || !func.IsValid()) {
        compiler->Release(func);
        return 0.0;
    }
    RegFile rf;
    rf.Scalar(0) = 0;
    rf.Scalar(1) = 0;
    rf.Scalar(2) = data_len;
    rf.Scalar(3) = std::bit_cast<u64>(threshold);
    f64* seg = const_cast<f64*>(data);
    u64  cnt = data_len;
    func.Entry(&rf, &seg, &cnt);
    f64 result = std::bit_cast<f64>(rf.Scalar(0));
    compiler->Release(func);
    return result;
}

// ================================================================
// Instruction encoding
// ================================================================
static u32 raw(const Instruction& i) { return i.raw; }
uint32_t voxel_instr_vload(uint8_t vd, uint8_t ra, uint8_t seg_id, uint8_t count)
    { return raw(Instruction::VLoad(vd, ra, seg_id, count)); }
uint32_t voxel_instr_vfilter_gt(uint8_t vd, uint8_t va, uint8_t rb)
    { return raw(Instruction::VFilterGt(vd, va, rb)); }
uint32_t voxel_instr_vfilter_ge(uint8_t vd, uint8_t va, uint8_t rb)
    { return raw(Instruction::VFilterGe(vd, va, rb)); }
uint32_t voxel_instr_vfilter_lt(uint8_t vd, uint8_t va, uint8_t rb)
    { return raw(Instruction::VFilterLt(vd, va, rb)); }
uint32_t voxel_instr_vfilter_le(uint8_t vd, uint8_t va, uint8_t rb)
    { return raw(Instruction::VFilterLe(vd, va, rb)); }
uint32_t voxel_instr_vsum(uint8_t rd, uint8_t va)
    { return raw(Instruction::VSum(rd, va)); }
uint32_t voxel_instr_vred_min(uint8_t rd, uint8_t va)
    { return raw(Instruction::VRedMin(rd, va)); }
uint32_t voxel_instr_vred_max(uint8_t rd, uint8_t va)
    { return raw(Instruction::VRedMax(rd, va)); }
uint32_t voxel_instr_vcount(uint8_t rd, uint8_t va)
    { return raw(Instruction::VCount(rd, va)); }
uint32_t voxel_instr_addf(uint8_t rd, uint8_t ra, uint8_t rb)
    { return raw(Instruction::Addf(rd, ra, rb)); }
uint32_t voxel_instr_add(uint8_t rd, uint8_t ra, uint8_t rb)
    { return raw(Instruction::Add(rd, ra, rb)); }
uint32_t voxel_instr_sub(uint8_t rd, uint8_t ra, uint8_t rb)
    { return raw(Instruction::Sub(rd, ra, rb)); }
uint32_t voxel_instr_mul(uint8_t rd, uint8_t ra, uint8_t rb)
    { return raw(Instruction::Mul(rd, ra, rb)); }
uint32_t voxel_instr_mov(uint8_t rd, int16_t imm)
    { return raw(Instruction::Mov(rd, imm)); }
uint32_t voxel_instr_movr(uint8_t rd, uint8_t ra)
    { return raw(Instruction::Movr(rd, ra)); }
uint32_t voxel_instr_cmp(uint8_t ra, uint8_t rb)
    { return raw(Instruction::Cmp(ra, rb)); }
uint32_t voxel_instr_jnz(int16_t offset)
    { return raw(Instruction::Jnz(offset)); }
uint32_t voxel_instr_jz(int16_t offset)
    { return raw(Instruction::Jz(offset)); }
uint32_t voxel_instr_jmp(int16_t offset)
    { return raw(Instruction::Jmp(offset)); }
uint32_t voxel_instr_halt(void)
    { return raw(Instruction::Halt()); }

// ================================================================
// NullBitmap
// ================================================================
void* voxel_null_create(size_t count) { return new NullBitmap(count); }
void  voxel_null_destroy(void* bm) { delete static_cast<NullBitmap*>(bm); }
int   voxel_null_is_null(const void* bm, size_t row)
    { return static_cast<const NullBitmap*>(bm)->IsNull(row) ? 1 : 0; }
void  voxel_null_set_null(void* bm, size_t row)
    { static_cast<NullBitmap*>(bm)->SetNull(row); }
void  voxel_null_set_valid(void* bm, size_t row)
    { static_cast<NullBitmap*>(bm)->SetValid(row); }
size_t voxel_null_null_count(const void* bm)
    { return static_cast<const NullBitmap*>(bm)->NullCount(); }
size_t voxel_null_valid_count(const void* bm)
    { return static_cast<const NullBitmap*>(bm)->ValidCount(); }

// ================================================================
// HashAggregator
// ================================================================
void* voxel_agg_create(void) {
    auto* arena = new Arena();
    return new ops::HashAggregator<u32, f64>(*arena);
}
void  voxel_agg_destroy(void* a) { delete static_cast<ops::HashAggregator<u32, f64>*>(a); }
void  voxel_agg_init(void* a, size_t groups)
    { static_cast<ops::HashAggregator<u32, f64>*>(a)->Init(groups); }
void  voxel_agg_accumulate(void* a, const uint32_t* keys, const double* values, size_t count)
    { static_cast<ops::HashAggregator<u32, f64>*>(a)->Accumulate(keys, values, count); }
size_t voxel_agg_group_count(const void* a)
    { return static_cast<const ops::HashAggregator<u32, f64>*>(a)->GroupCount(); }

// ================================================================
// Sort & TopK
// ================================================================
void voxel_sort_ascending(const double* data, size_t count, uint32_t* out) {
    std::vector<u32> idx(count); for (size_t i = 0; i < count; ++i) idx[i] = static_cast<u32>(i);
    Segment<f64> seg(const_cast<f64*>(data), count);
    ops::SortOperator::SortAscending(seg, idx.data());
    std::memcpy(out, idx.data(), count * sizeof(uint32_t));
}
void voxel_sort_descending(const double* data, size_t count, uint32_t* out) {
    std::vector<u32> idx(count); for (size_t i = 0; i < count; ++i) idx[i] = static_cast<u32>(i);
    Segment<f64> seg(const_cast<f64*>(data), count);
    ops::SortOperator::SortDescending(seg, idx.data());
    std::memcpy(out, idx.data(), count * sizeof(uint32_t));
}
void voxel_topk(const double* data, size_t count, size_t k, double* out, int largest) {
    ops::TopKSelect(data, count, k, out, largest != 0);
}

// ================================================================
// DictionaryEncoding
// ================================================================
void* voxel_dict_create(void) { return new encoding::DictionaryEncoding<f64>(); }
void  voxel_dict_destroy(void* d) { delete static_cast<encoding::DictionaryEncoding<f64>*>(d); }
void  voxel_dict_build(void* d, const double* data, size_t count)
    { static_cast<encoding::DictionaryEncoding<f64>*>(d)->Build(data, count); }
size_t voxel_dict_cardinality(const void* d)
    { return static_cast<const encoding::DictionaryEncoding<f64>*>(d)->Dictionary.size(); }
void   voxel_dict_decode_batch(void* d, double* out, size_t offset, size_t count)
    { static_cast<encoding::DictionaryEncoding<f64>*>(d)->DecodeBatch(out, offset, count); }

// ================================================================
// ThreadPool
// ================================================================
void* voxel_pool_create(uint32_t threads) { return new ThreadPool(threads); }
void  voxel_pool_destroy(void* pool) { delete static_cast<ThreadPool*>(pool); }
void  voxel_pool_wait_all(void* pool) { static_cast<ThreadPool*>(pool)->WaitAll(); }
size_t voxel_pool_thread_count(const void* pool)
    { return static_cast<const ThreadPool*>(pool)->ThreadCount(); }

// ================================================================
// Error handling
// ================================================================
const char* voxel_last_error(void) { return g_lastError; }

} // extern "C"
