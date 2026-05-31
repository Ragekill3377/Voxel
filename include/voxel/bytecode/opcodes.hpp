#pragma once

#include "voxel/core/types.hpp"

namespace voxel {

// ============================================================================
// Opcode — 8-bit instruction identifiers organized by functional category
// ============================================================================

enum class Opcode : u8 {
    // --- Control (0x00-0x0F) ---
    NOP          = 0x00,
    HALT         = 0x01,
    TRAP         = 0x02,
    BREAK        = 0x03,
    YIELD        = 0x04,
    BARRIER      = 0x05,
    PREFETCH     = 0x06,
    FLUSH_CACHE  = 0x07,
    SYNC         = 0x08,
    MEMFENCE     = 0x09,

    // --- Scalar Move (0x10-0x1F) ---
    MOV          = 0x10,
    MOVR         = 0x11,
    ADDI         = 0x12,
    SUBI         = 0x13,
    MULI         = 0x14,
    ANDI         = 0x15,
    ORI          = 0x16,
    XORI         = 0x17,
    SHLI         = 0x18,
    SHRI         = 0x19,
    SAR_I        = 0x1A,
    MOVZ         = 0x1B,
    MOVN         = 0x1C,
    MOVK         = 0x1D,
    LEA          = 0x1E,

    // --- Scalar Arithmetic (0x20-0x2F) ---
    ADD          = 0x20,
    SUB          = 0x21,
    MUL          = 0x22,
    DIV          = 0x23,
    MOD          = 0x24,
    NEG          = 0x25,
    ABS          = 0x26,
    MIN          = 0x27,
    MAX          = 0x28,
    AVG          = 0x29,
    ADDF         = 0x2A,
    SUBF         = 0x2B,
    MULF         = 0x2C,
    DIVF         = 0x2D,
    NEGF         = 0x2E,
    ABSF         = 0x2F,

    // --- Scalar Bitwise (0x30-0x3F) ---
    AND          = 0x30,
    OR           = 0x31,
    XOR          = 0x32,
    NOT          = 0x33,
    SHL          = 0x34,
    SHR          = 0x35,
    SAR          = 0x36,
    ROL          = 0x37,
    ROR          = 0x38,
    POPCNT       = 0x39,
    CLZ          = 0x3A,
    CTZ          = 0x3B,
    BSWAP        = 0x3C,
    BEXTR        = 0x3D,
    BZHI         = 0x3E,
    PDEP         = 0x3F,

    // --- Comparison (0x40-0x4F) ---
    CMP          = 0x40,
    CMPF         = 0x41,
    CMPU         = 0x42,
    TST          = 0x43,
    TSTF         = 0x44,
    ISNULL       = 0x45,
    ISNOTNULL    = 0x46,
    SELECT       = 0x47,
    SELECTV      = 0x48,

    // --- Branching (0x50-0x5F) ---
    JMP          = 0x50,
    JZ           = 0x51,
    JNZ          = 0x52,
    JS           = 0x53,
    JNS          = 0x54,
    JO           = 0x55,
    JNO          = 0x56,
    JC           = 0x57,
    JNC          = 0x58,
    JL           = 0x59,
    JLE          = 0x5A,
    JG           = 0x5B,
    JGE          = 0x5C,
    CALL         = 0x5D,
    RET          = 0x5E,
    TABLE_JMP    = 0x5F,

    // --- Type Conversion (0x60-0x6F) ---
    CVT_I8       = 0x60,
    CVT_I16      = 0x61,
    CVT_I32      = 0x62,
    CVT_I64      = 0x63,
    CVT_F32      = 0x64,
    CVT_F64      = 0x65,
    CVT_U8       = 0x66,
    CVT_U16      = 0x67,
    CVT_U32      = 0x68,
    CVT_U64      = 0x69,
    BITCAST      = 0x6A,
    REINTERPRET  = 0x6B,
    TRUNC        = 0x6C,
    ROUND        = 0x6D,
    CEIL         = 0x6E,
    FLOOR        = 0x6F,

    // --- Vector I/O (0x70-0x7F) ---
    VLOAD        = 0x70,
    VSTORE       = 0x71,
    VGATHER      = 0x72,
    VSCATTER     = 0x73,
    VLOAD_STRIDED = 0x74,
    VSTORE_STRIDED = 0x75,
    VLOAD_MASKED  = 0x76,
    VSTORE_MASKED = 0x77,
    VSPLAT       = 0x78,
    VEXTRACT     = 0x79,
    VINSERT      = 0x7A,
    VPERMUTE     = 0x7B,
    VSHUFFLE     = 0x7C,
    VREVERSE     = 0x7D,
    VROTATE      = 0x7E,
    VSLIDE       = 0x7F,

    // --- Vector Arithmetic (0x80-0x8F) ---
    VADD         = 0x80,
    VSUB         = 0x81,
    VMUL         = 0x82,
    VDIV         = 0x83,
    VMOD         = 0x84,
    VNEG         = 0x85,
    VABS         = 0x86,
    VMIN         = 0x87,
    VMAX         = 0x88,
    VAVG         = 0x89,
    VFMA         = 0x8A,
    VFMS         = 0x8B,
    VSQRT        = 0x8C,
    VRSQRT       = 0x8D,
    VRCP         = 0x8E,
    VPOW         = 0x8F,

    // --- Vector Scalar Arithmetic (0x90-0x9F) ---
    VSADD        = 0x90,
    VSSUB        = 0x91,
    VSMUL        = 0x92,
    VSDIV        = 0x93,
    VSMOD        = 0x94,
    VSMIN        = 0x95,
    VSMAX        = 0x96,

    // --- Vector Comparison (0xA0-0xAF) ---
    VCMPEQ       = 0xA0,
    VCMPNE       = 0xA1,
    VCMPLT       = 0xA2,
    VCMPLE       = 0xA3,
    VCMPGT       = 0xA4,
    VCMPGE       = 0xA5,
    VCMPNULL     = 0xA6,
    VCMPNOTNULL  = 0xA7,

    // --- Vector Logical (0xB0-0xBF) ---
    VAND         = 0xB0,
    VOR          = 0xB1,
    VXOR         = 0xB2,
    VNOT         = 0xB3,
    VANDN        = 0xB4,
    VSHL         = 0xB5,
    VSHR         = 0xB6,
    VSAR         = 0xB7,

    // --- Vector Filter (0xC0-0xCF) ---
    VFILTER      = 0xC0,
    VFILTER_EQ   = 0xC1,
    VFILTER_NE   = 0xC2,
    VFILTER_LT   = 0xC3,
    VFILTER_LE   = 0xC4,
    VFILTER_GT   = 0xC5,
    VFILTER_GE   = 0xC6,
    VBLEND       = 0xC7,
    VMASK_STORE  = 0xC8,
    VMASK_LOAD   = 0xC9,

    // --- Vector Reduction (0xD0-0xDF) ---
    VSUM         = 0xD0,
    VPROD        = 0xD1,
    VMEAN        = 0xD2,
    VSTDDEV      = 0xD3,
    VVARIANCE    = 0xD4,
    VRED_MIN     = 0xD5,
    VRED_MAX     = 0xD6,
    VCOUNT       = 0xD7,
    VANY         = 0xD8,
    VALL         = 0xD9,
    VFIRST       = 0xDA,
    VLAST        = 0xDB,
    VNTH         = 0xDC,

    // --- Aggregate Operators (0xE0-0xEF) ---
    AGG_COUNT     = 0xE0,
    AGG_SUM       = 0xE1,
    AGG_AVG       = 0xE2,
    AGG_MIN       = 0xE3,
    AGG_MAX       = 0xE4,
    AGG_FIRST     = 0xE5,
    AGG_LAST      = 0xE6,
    AGG_STDDEV    = 0xE7,
    AGG_VARIANCE  = 0xE8,
    AGG_COUNT_DISTINCT = 0xE9,
    AGG_SUM_DISTINCT   = 0xEA,
    AGG_MEDIAN    = 0xEB,
    AGG_MODE      = 0xEC,
    AGG_PERCENTILE = 0xED,
    HASH_INIT     = 0xEE,
    HASH_PROBE    = 0xEF,

    // --- Hash / Sort / Join (0xF0-0xFF) ---
    HASH_BUILD     = 0xF0,
    HASH_LOOKUP    = 0xF1,
    SORT_ASC       = 0xF2,
    SORT_DESC      = 0xF3,
    SORT_TOPK      = 0xF4,
    SORT_BOTTOMK   = 0xF5,
    JOIN_HASH      = 0xF6,
    JOIN_MERGE     = 0xF7,
    JOIN_NESTED    = 0xF8,
    JOIN_ANTI      = 0xF9,
    JOIN_SEMI      = 0xFA,
    WINDOW_ROW     = 0xFB,
    WINDOW_RANGE   = 0xFC,
    PARTITION_HASH = 0xFD,
    SERIALIZE      = 0xFE,
    DESERIALIZE    = 0xFF,
};

inline constexpr bool IsScalarOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0x10 && v <= 0x6F;
}

inline constexpr bool IsVectorOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0x70 && v <= 0xDF;
}

inline constexpr bool IsAggregateOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0xE0 && v <= 0xEF;
}

inline constexpr bool IsHashSortJoinOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0xF0;
}

inline constexpr bool IsControlFlowOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v <= 0x0F || (v >= 0x50 && v <= 0x5F);
}

inline constexpr bool ModifiesPc(Opcode op) {
    u8 v = static_cast<u8>(op);
    return (v >= 0x50 && v <= 0x5F) || v == 0x03;
}

inline constexpr const char* OpcodeName(Opcode op) {
    switch (op) {
    case Opcode::NOP: return "nop"; case Opcode::HALT: return "halt";
    case Opcode::TRAP: return "trap"; case Opcode::BREAK: return "break";
    case Opcode::YIELD: return "yield"; case Opcode::BARRIER: return "barrier";
    case Opcode::PREFETCH: return "prefetch"; case Opcode::FLUSH_CACHE: return "flush_cache";
    case Opcode::SYNC: return "sync"; case Opcode::MEMFENCE: return "memfence";
    case Opcode::MOV: return "mov"; case Opcode::MOVR: return "movr";
    case Opcode::ADDI: return "addi"; case Opcode::SUBI: return "subi";
    case Opcode::MULI: return "muli"; case Opcode::ANDI: return "andi";
    case Opcode::ORI: return "ori"; case Opcode::XORI: return "xori";
    case Opcode::SHLI: return "shli"; case Opcode::SHRI: return "shri";
    case Opcode::SAR_I: return "sar_i"; case Opcode::MOVZ: return "movz";
    case Opcode::MOVN: return "movn"; case Opcode::MOVK: return "movk";
    case Opcode::LEA: return "lea";
    case Opcode::ADD: return "add"; case Opcode::SUB: return "sub";
    case Opcode::MUL: return "mul"; case Opcode::DIV: return "div";
    case Opcode::MOD: return "mod"; case Opcode::NEG: return "neg";
    case Opcode::ABS: return "abs"; case Opcode::MIN: return "min";
    case Opcode::MAX: return "max"; case Opcode::AVG: return "avg";
    case Opcode::ADDF: return "addf"; case Opcode::SUBF: return "subf";
    case Opcode::MULF: return "mulf"; case Opcode::DIVF: return "divf";
    case Opcode::NEGF: return "negf"; case Opcode::ABSF: return "absf";
    case Opcode::AND: return "and"; case Opcode::OR: return "or";
    case Opcode::XOR: return "xor"; case Opcode::NOT: return "not";
    case Opcode::SHL: return "shl"; case Opcode::SHR: return "shr";
    case Opcode::SAR: return "sar"; case Opcode::ROL: return "rol";
    case Opcode::ROR: return "ror"; case Opcode::POPCNT: return "popcnt";
    case Opcode::CLZ: return "clz"; case Opcode::CTZ: return "ctz";
    case Opcode::BSWAP: return "bswap"; case Opcode::BEXTR: return "bextr";
    case Opcode::BZHI: return "bzhi"; case Opcode::PDEP: return "pdep";
    case Opcode::CMP: return "cmp"; case Opcode::CMPF: return "cmpf";
    case Opcode::CMPU: return "cmpu"; case Opcode::TST: return "tst";
    case Opcode::TSTF: return "tstf"; case Opcode::ISNULL: return "isnull";
    case Opcode::ISNOTNULL: return "isnotnull"; case Opcode::SELECT: return "select";
    case Opcode::SELECTV: return "selectv";
    case Opcode::JMP: return "jmp"; case Opcode::JZ: return "jz";
    case Opcode::JNZ: return "jnz"; case Opcode::JS: return "js";
    case Opcode::JNS: return "jns"; case Opcode::JO: return "jo";
    case Opcode::JNO: return "jno"; case Opcode::JC: return "jc";
    case Opcode::JNC: return "jnc"; case Opcode::JL: return "jl";
    case Opcode::JLE: return "jle"; case Opcode::JG: return "jg";
    case Opcode::JGE: return "jge"; case Opcode::CALL: return "call";
    case Opcode::RET: return "ret"; case Opcode::TABLE_JMP: return "table_jmp";
    case Opcode::CVT_I8: return "cvt_i8"; case Opcode::CVT_I16: return "cvt_i16";
    case Opcode::CVT_I32: return "cvt_i32"; case Opcode::CVT_I64: return "cvt_i64";
    case Opcode::CVT_F32: return "cvt_f32"; case Opcode::CVT_F64: return "cvt_f64";
    case Opcode::CVT_U8: return "cvt_u8"; case Opcode::CVT_U16: return "cvt_u16";
    case Opcode::CVT_U32: return "cvt_u32"; case Opcode::CVT_U64: return "cvt_u64";
    case Opcode::BITCAST: return "bitcast"; case Opcode::REINTERPRET: return "reinterpret";
    case Opcode::TRUNC: return "trunc"; case Opcode::ROUND: return "round";
    case Opcode::CEIL: return "ceil"; case Opcode::FLOOR: return "floor";
    case Opcode::VLOAD: return "vload"; case Opcode::VSTORE: return "vstore";
    case Opcode::VGATHER: return "vgather"; case Opcode::VSCATTER: return "vscatter";
    case Opcode::VLOAD_STRIDED: return "vload_strided"; case Opcode::VSTORE_STRIDED: return "vstore_strided";
    case Opcode::VLOAD_MASKED: return "vload_masked"; case Opcode::VSTORE_MASKED: return "vstore_masked";
    case Opcode::VSPLAT: return "vsplat"; case Opcode::VEXTRACT: return "vextract";
    case Opcode::VINSERT: return "vinsert"; case Opcode::VPERMUTE: return "vpermute";
    case Opcode::VSHUFFLE: return "vshuffle"; case Opcode::VREVERSE: return "vreverse";
    case Opcode::VROTATE: return "vrotate"; case Opcode::VSLIDE: return "vslide";
    case Opcode::VADD: return "vadd"; case Opcode::VSUB: return "vsub";
    case Opcode::VMUL: return "vmul"; case Opcode::VDIV: return "vdiv";
    case Opcode::VMOD: return "vmod"; case Opcode::VNEG: return "vneg";
    case Opcode::VABS: return "vabs"; case Opcode::VMIN: return "vmin";
    case Opcode::VMAX: return "vmax"; case Opcode::VAVG: return "vavg";
    case Opcode::VFMA: return "vfma"; case Opcode::VFMS: return "vfms";
    case Opcode::VSQRT: return "vsqrt"; case Opcode::VRSQRT: return "vrsqrt";
    case Opcode::VRCP: return "vrcp"; case Opcode::VPOW: return "vpow";
    case Opcode::VSADD: return "vsadd"; case Opcode::VSSUB: return "vssub";
    case Opcode::VSMUL: return "vsmul"; case Opcode::VSDIV: return "vsdiv";
    case Opcode::VSMOD: return "vsmod"; case Opcode::VSMIN: return "vsmin";
    case Opcode::VSMAX: return "vsmax";
    case Opcode::VCMPEQ: return "vcmpeq"; case Opcode::VCMPNE: return "vcmpne";
    case Opcode::VCMPLT: return "vcmplt"; case Opcode::VCMPLE: return "vcmple";
    case Opcode::VCMPGT: return "vcmpgt"; case Opcode::VCMPGE: return "vcmpge";
    case Opcode::VCMPNULL: return "vcmpnull"; case Opcode::VCMPNOTNULL: return "vcmpnotnull";
    case Opcode::VAND: return "vand"; case Opcode::VOR: return "vor";
    case Opcode::VXOR: return "vxor"; case Opcode::VNOT: return "vnot";
    case Opcode::VANDN: return "vandn"; case Opcode::VSHL: return "vshl";
    case Opcode::VSHR: return "vshr"; case Opcode::VSAR: return "vsar";
    case Opcode::VFILTER: return "vfilter"; case Opcode::VFILTER_EQ: return "vfilter_eq";
    case Opcode::VFILTER_NE: return "vfilter_ne"; case Opcode::VFILTER_LT: return "vfilter_lt";
    case Opcode::VFILTER_LE: return "vfilter_le"; case Opcode::VFILTER_GT: return "vfilter_gt";
    case Opcode::VFILTER_GE: return "vfilter_ge"; case Opcode::VBLEND: return "vblend";
    case Opcode::VMASK_STORE: return "vmask_store"; case Opcode::VMASK_LOAD: return "vmask_load";
    case Opcode::VSUM: return "vsum"; case Opcode::VPROD: return "vprod";
    case Opcode::VMEAN: return "vmean"; case Opcode::VSTDDEV: return "vstddev";
    case Opcode::VVARIANCE: return "vvariance"; case Opcode::VRED_MIN: return "vred_min";
    case Opcode::VRED_MAX: return "vred_max"; case Opcode::VCOUNT: return "vcount";
    case Opcode::VANY: return "vany"; case Opcode::VALL: return "vall";
    case Opcode::VFIRST: return "vfirst"; case Opcode::VLAST: return "vlast";
    case Opcode::VNTH: return "vnth";
    case Opcode::AGG_COUNT: return "agg_count"; case Opcode::AGG_SUM: return "agg_sum";
    case Opcode::AGG_AVG: return "agg_avg"; case Opcode::AGG_MIN: return "agg_min";
    case Opcode::AGG_MAX: return "agg_max"; case Opcode::AGG_FIRST: return "agg_first";
    case Opcode::AGG_LAST: return "agg_last"; case Opcode::AGG_STDDEV: return "agg_stddev";
    case Opcode::AGG_VARIANCE: return "agg_variance";
    case Opcode::AGG_COUNT_DISTINCT: return "agg_count_distinct";
    case Opcode::AGG_SUM_DISTINCT: return "agg_sum_distinct";
    case Opcode::AGG_MEDIAN: return "agg_median"; case Opcode::AGG_MODE: return "agg_mode";
    case Opcode::AGG_PERCENTILE: return "agg_percentile";
    case Opcode::HASH_INIT: return "hash_init"; case Opcode::HASH_PROBE: return "hash_probe";
    case Opcode::HASH_BUILD: return "hash_build"; case Opcode::HASH_LOOKUP: return "hash_lookup";
    case Opcode::SORT_ASC: return "sort_asc"; case Opcode::SORT_DESC: return "sort_desc";
    case Opcode::SORT_TOPK: return "sort_topk"; case Opcode::SORT_BOTTOMK: return "sort_bottomk";
    case Opcode::JOIN_HASH: return "join_hash"; case Opcode::JOIN_MERGE: return "join_merge";
    case Opcode::JOIN_NESTED: return "join_nested"; case Opcode::JOIN_ANTI: return "join_anti";
    case Opcode::JOIN_SEMI: return "join_semi"; case Opcode::WINDOW_ROW: return "window_row";
    case Opcode::WINDOW_RANGE: return "window_range";
    case Opcode::PARTITION_HASH: return "partition_hash";
    case Opcode::SERIALIZE: return "serialize"; case Opcode::DESERIALIZE: return "deserialize";
    default: return "unknown";
    }
}

inline constexpr u8 kCmpEQ = 0;
inline constexpr u8 kCmpNE = 1;
inline constexpr u8 kCmpLT = 2;
inline constexpr u8 kCmpLE = 3;
inline constexpr u8 kCmpGT = 4;
inline constexpr u8 kCmpGE = 5;

} // namespace voxel
