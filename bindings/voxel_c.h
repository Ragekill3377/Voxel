#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// Engine (f64)
// ================================================================
void*     voxel_engine_create_f64(void);
void      voxel_engine_destroy(void* engine);
size_t    voxel_engine_add_segment(void* engine, const double* data, size_t count);
void      voxel_engine_load_program(void* engine, const uint32_t* code, size_t len);
void      voxel_engine_run(void* engine);
void      voxel_engine_run_parallel(void* engine, size_t seg_id, uint32_t threads, uint8_t result_reg);
void      voxel_engine_set_scalar(void* engine, size_t reg, uint64_t value);
uint64_t  voxel_engine_get_scalar(void* engine, size_t reg);
void      voxel_engine_set_scalar_f64(void* engine, size_t reg, double value);
double    voxel_engine_get_scalar_f64(void* engine, size_t reg);
size_t    voxel_engine_k_lanes(void);

// ================================================================
// Engine (i64)
// ================================================================
void*     voxel_engine_create_i64(void);
void      voxel_engine_destroy_i64(void* engine);
size_t    voxel_engine_add_segment_i64(void* engine, const int64_t* data, size_t count);
void      voxel_engine_load_program_i64(void* engine, const uint32_t* code, size_t len);
void      voxel_engine_run_i64(void* engine);
void      voxel_engine_set_scalar_i64(void* engine, size_t reg, uint64_t value);
uint64_t  voxel_engine_get_scalar_i64(void* engine, size_t reg);

// ================================================================
// JIT
// ================================================================
double    voxel_jit_run(const uint32_t* code, size_t code_len,
                        const double* data, size_t data_len, double threshold);

// ================================================================
// Instruction encoding
// ================================================================
uint32_t  voxel_instr_vload(uint8_t vd, uint8_t ra, uint8_t seg_id, uint8_t count);
uint32_t  voxel_instr_vfilter_gt(uint8_t vd, uint8_t va, uint8_t rb);
uint32_t  voxel_instr_vfilter_ge(uint8_t vd, uint8_t va, uint8_t rb);
uint32_t  voxel_instr_vfilter_lt(uint8_t vd, uint8_t va, uint8_t rb);
uint32_t  voxel_instr_vfilter_le(uint8_t vd, uint8_t va, uint8_t rb);
uint32_t  voxel_instr_vsum(uint8_t rd, uint8_t va);
uint32_t  voxel_instr_vred_min(uint8_t rd, uint8_t va);
uint32_t  voxel_instr_vred_max(uint8_t rd, uint8_t va);
uint32_t  voxel_instr_vcount(uint8_t rd, uint8_t va);
uint32_t  voxel_instr_addf(uint8_t rd, uint8_t ra, uint8_t rb);
uint32_t  voxel_instr_add(uint8_t rd, uint8_t ra, uint8_t rb);
uint32_t  voxel_instr_sub(uint8_t rd, uint8_t ra, uint8_t rb);
uint32_t  voxel_instr_mul(uint8_t rd, uint8_t ra, uint8_t rb);
uint32_t  voxel_instr_mov(uint8_t rd, int16_t imm);
uint32_t  voxel_instr_movr(uint8_t rd, uint8_t ra);
uint32_t  voxel_instr_cmp(uint8_t ra, uint8_t rb);
uint32_t  voxel_instr_jnz(int16_t offset);
uint32_t  voxel_instr_jz(int16_t offset);
uint32_t  voxel_instr_jmp(int16_t offset);
uint32_t  voxel_instr_halt(void);

// ================================================================
// NullBitmap
// ================================================================
void*     voxel_null_create(size_t count);
void      voxel_null_destroy(void* bitmap);
int       voxel_null_is_null(const void* bitmap, size_t row);
void      voxel_null_set_null(void* bitmap, size_t row);
void      voxel_null_set_valid(void* bitmap, size_t row);
size_t    voxel_null_null_count(const void* bitmap);
size_t    voxel_null_valid_count(const void* bitmap);

// ================================================================
// HashAggregator
// ================================================================
void*     voxel_agg_create(void);
void      voxel_agg_destroy(void* agg);
void      voxel_agg_init(void* agg, size_t expected_groups);
void      voxel_agg_accumulate(void* agg, const uint32_t* keys, const double* values, size_t count);
size_t    voxel_agg_group_count(const void* agg);

// ================================================================
// Sort & TopK
// ================================================================
void      voxel_sort_ascending(const double* data, size_t count, uint32_t* indices_out);
void      voxel_sort_descending(const double* data, size_t count, uint32_t* indices_out);
void      voxel_topk(const double* data, size_t count, size_t k, double* out, int largest);

// ================================================================
// DictionaryEncoding
// ================================================================
void*     voxel_dict_create(void);
void      voxel_dict_destroy(void* dict);
void      voxel_dict_build(void* dict, const double* data, size_t count);
size_t    voxel_dict_cardinality(const void* dict);
void      voxel_dict_decode_batch(void* dict, double* out, size_t offset, size_t count);

// ================================================================
// ThreadPool
// ================================================================
void*     voxel_pool_create(uint32_t threads);
void      voxel_pool_destroy(void* pool);
void      voxel_pool_wait_all(void* pool);
size_t    voxel_pool_thread_count(const void* pool);

// ================================================================
// Error handling
// ================================================================
const char* voxel_last_error(void);

#ifdef __cplusplus
}
#endif
