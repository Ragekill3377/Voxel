#pragma once

#include "voxel/bytecode/opcodes.hpp"
#include "voxel/core/types.hpp"

namespace voxel {

struct Instruction {
    u32 raw;

    Instruction()
        : raw(0)
    {}

    Instruction(u32 r)
        : raw(r)
    {}

    Opcode Op() const {
        return static_cast<Opcode>(raw & 0xFF);
    }

    u8 Rd() const {
        return (raw >> 8) & 0xF;
    }

    u8 Ra() const {
        return (raw >> 12) & 0xF;
    }

    u8 Rb() const {
        return (raw >> 16) & 0xF;
    }

    u16 Imm12() const {
        return (raw >> 20) & 0xFFF;
    }

    u8 SegId() const {
        return (raw >> 28) & 0xF;
    }

    u8 VecCount() const {
        return (raw >> 20) & 0xFF;
    }

    u8 CmpMode() const {
        return (raw >> 20) & 0x7;
    }

    i16 Simm12() const {
        u16 v = (raw >> 20) & 0xFFF;
        if (v & 0x800) {
            return static_cast<i16>(v | 0xF000);
        } else {
            return static_cast<i16>(v);
        }
    }

    u32 ExtendedImm32() const {
        return 0;
    }

    static u32 Encode(Opcode op, u8 rd, u8 ra, u8 rb, u16 imm) {
        u32 opBits  = static_cast<u32>(static_cast<u8>(op));
        u32 rdBits  = (static_cast<u32>(rd)  & 0xF) << 8;
        u32 raBits  = (static_cast<u32>(ra)  & 0xF) << 12;
        u32 rbBits  = (static_cast<u32>(rb)  & 0xF) << 16;
        u32 immBits = (static_cast<u32>(imm) & 0xFFF) << 20;
        return opBits | rdBits | raBits | rbBits | immBits;
    }

    static u32 Encode(Opcode op, u16 imm) {
        return Encode(op, 0, 0, 0, imm);
    }

    static u32 Encode(Opcode op) {
        return Encode(op, 0, 0, 0, 0);
    }

    // ====================================================================
    // Control — 0x00..0x0F
    // ====================================================================

    static Instruction Nop() {
        return Instruction{Encode(Opcode::NOP)};
    }

    static Instruction Halt() {
        return Instruction{Encode(Opcode::HALT)};
    }

    static Instruction Trap() {
        return Instruction{Encode(Opcode::TRAP)};
    }

    static Instruction Break() {
        return Instruction{Encode(Opcode::BREAK)};
    }

    static Instruction Yield() {
        return Instruction{Encode(Opcode::YIELD)};
    }

    static Instruction Barrier() {
        return Instruction{Encode(Opcode::BARRIER)};
    }

    static Instruction Prefetch() {
        return Instruction{Encode(Opcode::PREFETCH)};
    }

    static Instruction FlushCache() {
        return Instruction{Encode(Opcode::FLUSH_CACHE)};
    }

    static Instruction Sync() {
        return Instruction{Encode(Opcode::SYNC)};
    }

    static Instruction Memfence() {
        return Instruction{Encode(Opcode::MEMFENCE)};
    }

    // ====================================================================
    // Scalar Move — 0x10..0x1F
    // ====================================================================

    static Instruction Mov(u8 rd, i16 imm) {
        return Instruction{Encode(Opcode::MOV, rd, 0, 0, static_cast<u16>(imm))};
    }

    static Instruction Movr(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::MOVR, rd, ra, 0, 0)};
    }

    static Instruction Addi(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::ADDI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Subi(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::SUBI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Muli(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::MULI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Andi(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::ANDI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Ori(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::ORI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Xori(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::XORI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Shli(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::SHLI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Shri(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::SHRI, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction SarI(u8 rd, u8 ra, i16 imm) {
        return Instruction{Encode(Opcode::SAR_I, rd, ra, 0, static_cast<u16>(imm))};
    }

    static Instruction Movz(u8 rd, i16 imm) {
        return Instruction{Encode(Opcode::MOVZ, rd, 0, 0, static_cast<u16>(imm))};
    }

    static Instruction Movn(u8 rd, i16 imm) {
        return Instruction{Encode(Opcode::MOVN, rd, 0, 0, static_cast<u16>(imm))};
    }

    static Instruction Movk(u8 rd, i16 imm) {
        return Instruction{Encode(Opcode::MOVK, rd, 0, 0, static_cast<u16>(imm))};
    }

    static Instruction Lea(u8 rd, u8 ra, i16 off) {
        return Instruction{Encode(Opcode::LEA, rd, ra, 0, static_cast<u16>(off))};
    }

    // ====================================================================
    // Scalar Arithmetic — 0x20..0x2F
    // ====================================================================

    static Instruction Add(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::ADD, rd, ra, rb, 0)};
    }

    static Instruction Sub(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SUB, rd, ra, rb, 0)};
    }

    static Instruction Mul(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::MUL, rd, ra, rb, 0)};
    }

    static Instruction Div(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::DIV, rd, ra, rb, 0)};
    }

    static Instruction Mod(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::MOD, rd, ra, rb, 0)};
    }

    static Instruction Neg(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::NEG, rd, ra, 0, 0)};
    }

    static Instruction Abs(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::ABS, rd, ra, 0, 0)};
    }

    static Instruction Min(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::MIN, rd, ra, rb, 0)};
    }

    static Instruction Max(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::MAX, rd, ra, rb, 0)};
    }

    static Instruction Avg(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::AVG, rd, ra, rb, 0)};
    }

    static Instruction Addf(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::ADDF, rd, ra, rb, 0)};
    }

    static Instruction Subf(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SUBF, rd, ra, rb, 0)};
    }

    static Instruction Mulf(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::MULF, rd, ra, rb, 0)};
    }

    static Instruction Divf(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::DIVF, rd, ra, rb, 0)};
    }

    static Instruction Negf(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::NEGF, rd, ra, 0, 0)};
    }

    static Instruction Absf(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::ABSF, rd, ra, 0, 0)};
    }

    // ====================================================================
    // Scalar Bitwise — 0x30..0x3F
    // ====================================================================

    static Instruction And(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::AND, rd, ra, rb, 0)};
    }

    static Instruction Or(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::OR, rd, ra, rb, 0)};
    }

    static Instruction Xor(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::XOR, rd, ra, rb, 0)};
    }

    static Instruction Not(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::NOT, rd, ra, 0, 0)};
    }

    static Instruction Shl(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SHL, rd, ra, rb, 0)};
    }

    static Instruction Shr(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SHR, rd, ra, rb, 0)};
    }

    static Instruction Sar(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SAR, rd, ra, rb, 0)};
    }

    static Instruction Rol(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::ROL, rd, ra, rb, 0)};
    }

    static Instruction Ror(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::ROR, rd, ra, rb, 0)};
    }

    static Instruction Popcnt(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::POPCNT, rd, ra, 0, 0)};
    }

    static Instruction Clz(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CLZ, rd, ra, 0, 0)};
    }

    static Instruction Ctz(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CTZ, rd, ra, 0, 0)};
    }

    static Instruction Bswap(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::BSWAP, rd, ra, 0, 0)};
    }

    static Instruction Bextr(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::BEXTR, rd, ra, rb, 0)};
    }

    static Instruction Bzhi(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::BZHI, rd, ra, rb, 0)};
    }

    static Instruction Pdep(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::PDEP, rd, ra, rb, 0)};
    }

    // ====================================================================
    // Comparison — 0x40..0x4F
    // ====================================================================

    static Instruction Cmp(u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::CMP, 0, ra, rb, 0)};
    }

    static Instruction Cmpf(u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::CMPF, 0, ra, rb, 0)};
    }

    static Instruction Cmpu(u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::CMPU, 0, ra, rb, 0)};
    }

    static Instruction Tst(u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::TST, 0, ra, rb, 0)};
    }

    static Instruction Tstf(u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::TSTF, 0, ra, rb, 0)};
    }

    static Instruction Isnull(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::ISNULL, rd, ra, 0, 0)};
    }

    static Instruction Isnotnull(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::ISNOTNULL, rd, ra, 0, 0)};
    }

    static Instruction Select(u8 rd, u8 ra, u8 rb) {
        return Instruction{Encode(Opcode::SELECT, rd, ra, rb, 0)};
    }

    static Instruction Selectv(u8 rd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::SELECTV, rd, va, vb, 0)};
    }

    // ====================================================================
    // Branching — 0x50..0x5F
    // ====================================================================

    static Instruction Jmp(i16 offset) {
        return Instruction{Encode(Opcode::JMP, static_cast<u16>(offset))};
    }

    static Instruction Jz(i16 offset) {
        return Instruction{Encode(Opcode::JZ, static_cast<u16>(offset))};
    }

    static Instruction Jnz(i16 offset) {
        return Instruction{Encode(Opcode::JNZ, static_cast<u16>(offset))};
    }

    static Instruction Js(i16 offset) {
        return Instruction{Encode(Opcode::JS, static_cast<u16>(offset))};
    }

    static Instruction Jns(i16 offset) {
        return Instruction{Encode(Opcode::JNS, static_cast<u16>(offset))};
    }

    static Instruction Jo(i16 offset) {
        return Instruction{Encode(Opcode::JO, static_cast<u16>(offset))};
    }

    static Instruction Jno(i16 offset) {
        return Instruction{Encode(Opcode::JNO, static_cast<u16>(offset))};
    }

    static Instruction Jc(i16 offset) {
        return Instruction{Encode(Opcode::JC, static_cast<u16>(offset))};
    }

    static Instruction Jnc(i16 offset) {
        return Instruction{Encode(Opcode::JNC, static_cast<u16>(offset))};
    }

    static Instruction Jl(i16 offset) {
        return Instruction{Encode(Opcode::JL, static_cast<u16>(offset))};
    }

    static Instruction Jle(i16 offset) {
        return Instruction{Encode(Opcode::JLE, static_cast<u16>(offset))};
    }

    static Instruction Jg(i16 offset) {
        return Instruction{Encode(Opcode::JG, static_cast<u16>(offset))};
    }

    static Instruction Jge(i16 offset) {
        return Instruction{Encode(Opcode::JGE, static_cast<u16>(offset))};
    }

    static Instruction Call(i16 offset) {
        return Instruction{Encode(Opcode::CALL, static_cast<u16>(offset))};
    }

    static Instruction Ret() {
        return Instruction{Encode(Opcode::RET)};
    }

    static Instruction TableJmp(u8 ra) {
        return Instruction{Encode(Opcode::TABLE_JMP, 0, ra, 0, 0)};
    }

    // ====================================================================
    // Type Conversion — 0x60..0x6F
    // ====================================================================

    static Instruction CvtI8(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_I8, rd, ra, 0, 0)};
    }

    static Instruction CvtI16(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_I16, rd, ra, 0, 0)};
    }

    static Instruction CvtI32(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_I32, rd, ra, 0, 0)};
    }

    static Instruction CvtI64(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_I64, rd, ra, 0, 0)};
    }

    static Instruction CvtF32(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_F32, rd, ra, 0, 0)};
    }

    static Instruction CvtF64(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_F64, rd, ra, 0, 0)};
    }

    static Instruction CvtU8(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_U8, rd, ra, 0, 0)};
    }

    static Instruction CvtU16(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_U16, rd, ra, 0, 0)};
    }

    static Instruction CvtU32(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_U32, rd, ra, 0, 0)};
    }

    static Instruction CvtU64(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CVT_U64, rd, ra, 0, 0)};
    }

    static Instruction Bitcast(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::BITCAST, rd, ra, 0, 0)};
    }

    static Instruction Reinterpret(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::REINTERPRET, rd, ra, 0, 0)};
    }

    static Instruction Trunc(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::TRUNC, rd, ra, 0, 0)};
    }

    static Instruction Round(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::ROUND, rd, ra, 0, 0)};
    }

    static Instruction Ceil(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::CEIL, rd, ra, 0, 0)};
    }

    static Instruction Floor(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::FLOOR, rd, ra, 0, 0)};
    }

    // ====================================================================
    // Vector I/O — 0x70..0x7F
    // ====================================================================

    static Instruction VLoad(u8 vd, u8 ra, u8 segId, u8 count) {
        u16 imm = (static_cast<u16>(segId & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VLOAD, vd, ra, 0, imm)};
    }

    static Instruction VStore(u8 vs, u8 ra, u8 segId, u8 count) {
        u16 imm = (static_cast<u16>(segId & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VSTORE, vs, ra, 0, imm)};
    }

    static Instruction VGather(u8 vd, u8 ra, u8 segId, u8 count) {
        u16 imm = (static_cast<u16>(segId & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VGATHER, vd, ra, 0, imm)};
    }

    static Instruction VScatter(u8 vs, u8 ra, u8 segId, u8 count) {
        u16 imm = (static_cast<u16>(segId & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VSCATTER, vs, ra, 0, imm)};
    }

    static Instruction VLoadStrided(u8 vd, u8 ra, u8 stride, u8 count) {
        u16 imm = (static_cast<u16>(stride & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VLOAD_STRIDED, vd, ra, 0, imm)};
    }

    static Instruction VStoreStrided(u8 vs, u8 ra, u8 stride, u8 count) {
        u16 imm = (static_cast<u16>(stride & 0xF) << 8) | (count & 0xFF);
        return Instruction{Encode(Opcode::VSTORE_STRIDED, vs, ra, 0, imm)};
    }

    static Instruction VLoadMasked(u8 vd, u8 ra, u8 maskReg) {
        return Instruction{Encode(Opcode::VLOAD_MASKED, vd, ra, maskReg, 0)};
    }

    static Instruction VStoreMasked(u8 vs, u8 ra, u8 maskReg) {
        return Instruction{Encode(Opcode::VSTORE_MASKED, vs, ra, maskReg, 0)};
    }

    static Instruction VSplat(u8 vd, u8 ra) {
        return Instruction{Encode(Opcode::VSPLAT, vd, ra, 0, 0)};
    }

    static Instruction VExtract(u8 rd, u8 va, u8 lane) {
        return Instruction{Encode(Opcode::VEXTRACT, rd, va, lane, 0)};
    }

    static Instruction VInsert(u8 vd, u8 va, u8 rb, u8 lane) {
        return Instruction{Encode(Opcode::VINSERT, vd, va, rb, lane & 0xF)};
    }

    static Instruction VPermute(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VPERMUTE, vd, va, vb, 0)};
    }

    static Instruction VShuffle(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VSHUFFLE, vd, va, vb, 0)};
    }

    static Instruction VReverse(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VREVERSE, vd, va, 0, 0)};
    }

    static Instruction VRotate(u8 vd, u8 va, u8 amount) {
        return Instruction{Encode(Opcode::VROTATE, vd, va, 0, amount & 0xFF)};
    }

    static Instruction VSlide(u8 vd, u8 va, u8 amount) {
        return Instruction{Encode(Opcode::VSLIDE, vd, va, 0, amount & 0xFF)};
    }

    // ====================================================================
    // Vector Arithmetic — 0x80..0x8F
    // ====================================================================

    static Instruction VAdd(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VADD, vd, va, vb, 0)};
    }

    static Instruction VSub(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VSUB, vd, va, vb, 0)};
    }

    static Instruction VMul(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VMUL, vd, va, vb, 0)};
    }

    static Instruction VDiv(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VDIV, vd, va, vb, 0)};
    }

    static Instruction VMod(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VMOD, vd, va, vb, 0)};
    }

    static Instruction VNeg(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VNEG, vd, va, 0, 0)};
    }

    static Instruction VAbs(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VABS, vd, va, 0, 0)};
    }

    static Instruction VMin(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VMIN, vd, va, vb, 0)};
    }

    static Instruction VMax(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VMAX, vd, va, vb, 0)};
    }

    static Instruction VAvg(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VAVG, vd, va, vb, 0)};
    }

    static Instruction VFma(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VFMA, vd, va, vb, 0)};
    }

    static Instruction VFms(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VFMS, vd, va, vb, 0)};
    }

    static Instruction VSqrt(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VSQRT, vd, va, 0, 0)};
    }

    static Instruction VRSqrt(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VRSQRT, vd, va, 0, 0)};
    }

    static Instruction VRcp(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VRCP, vd, va, 0, 0)};
    }

    static Instruction VPow(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VPOW, vd, va, vb, 0)};
    }

    // ====================================================================
    // Vector Scalar Arithmetic — 0x90..0x9F
    // ====================================================================

    static Instruction VSadd(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSADD, vd, va, rb, 0)};
    }

    static Instruction VSsub(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSSUB, vd, va, rb, 0)};
    }

    static Instruction VSmul(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSMUL, vd, va, rb, 0)};
    }

    static Instruction VSdiv(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSDIV, vd, va, rb, 0)};
    }

    static Instruction VSmod(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSMOD, vd, va, rb, 0)};
    }

    static Instruction VSmin(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSMIN, vd, va, rb, 0)};
    }

    static Instruction VSmax(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VSMAX, vd, va, rb, 0)};
    }

    // ====================================================================
    // Vector Comparison — 0xA0..0xAF
    // ====================================================================

    static Instruction VCmpeq(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPEQ, vd, va, vb, 0)};
    }

    static Instruction VCmpne(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPNE, vd, va, vb, 0)};
    }

    static Instruction VCmplt(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPLT, vd, va, vb, 0)};
    }

    static Instruction VCmple(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPLE, vd, va, vb, 0)};
    }

    static Instruction VCmpgt(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPGT, vd, va, vb, 0)};
    }

    static Instruction VCmpge(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VCMPGE, vd, va, vb, 0)};
    }

    static Instruction VCmpnull(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VCMPNULL, vd, va, 0, 0)};
    }

    static Instruction VCmpnotnull(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VCMPNOTNULL, vd, va, 0, 0)};
    }

    // ====================================================================
    // Vector Logical — 0xB0..0xBF
    // ====================================================================

    static Instruction VAnd(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VAND, vd, va, vb, 0)};
    }

    static Instruction VOr(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VOR, vd, va, vb, 0)};
    }

    static Instruction VXor(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VXOR, vd, va, vb, 0)};
    }

    static Instruction VNot(u8 vd, u8 va) {
        return Instruction{Encode(Opcode::VNOT, vd, va, 0, 0)};
    }

    static Instruction VAndn(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VANDN, vd, va, vb, 0)};
    }

    static Instruction VShl(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VSHL, vd, va, vb, 0)};
    }

    static Instruction VShr(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VSHR, vd, va, vb, 0)};
    }

    static Instruction VSar(u8 vd, u8 va, u8 vb) {
        return Instruction{Encode(Opcode::VSAR, vd, va, vb, 0)};
    }

    // ====================================================================
    // Vector Filter — 0xC0..0xCF
    // ====================================================================

    static Instruction VFilter(u8 vd, u8 va, u8 rb, u8 mode) {
        return Instruction{Encode(Opcode::VFILTER, vd, va, rb, mode & 0x7)};
    }

    static Instruction VFilterEq(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_EQ, vd, va, rb, 0)};
    }

    static Instruction VFilterNe(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_NE, vd, va, rb, 0)};
    }

    static Instruction VFilterLt(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_LT, vd, va, rb, 0)};
    }

    static Instruction VFilterLe(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_LE, vd, va, rb, 0)};
    }

    static Instruction VFilterGt(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_GT, vd, va, rb, 0)};
    }

    static Instruction VFilterGe(u8 vd, u8 va, u8 rb) {
        return Instruction{Encode(Opcode::VFILTER_GE, vd, va, rb, 0)};
    }

    static Instruction VBlend(u8 vd, u8 va, u8 vb, u8 maskReg) {
        return Instruction{Encode(Opcode::VBLEND, vd, va, vb, maskReg & 0x7)};
    }

    static Instruction VMaskStore(u8 vs, u8 ra, u8 maskReg, u8 segId) {
        u16 imm = static_cast<u16>(segId & 0xF) << 8;
        return Instruction{Encode(Opcode::VMASK_STORE, vs, ra, maskReg, imm)};
    }

    static Instruction VMaskLoad(u8 vd, u8 ra, u8 maskReg, u8 segId) {
        u16 imm = static_cast<u16>(segId & 0xF) << 8;
        return Instruction{Encode(Opcode::VMASK_LOAD, vd, ra, maskReg, imm)};
    }

    // ====================================================================
    // Vector Reduction — 0xD0..0xDF
    // ====================================================================

    static Instruction VSum(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VSUM, rd, va, 0, 0)};
    }

    static Instruction VProd(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VPROD, rd, va, 0, 0)};
    }

    static Instruction VMean(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VMEAN, rd, va, 0, 0)};
    }

    static Instruction VStddev(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VSTDDEV, rd, va, 0, 0)};
    }

    static Instruction VVariance(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VVARIANCE, rd, va, 0, 0)};
    }

    static Instruction VRedMin(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VRED_MIN, rd, va, 0, 0)};
    }

    static Instruction VRedMax(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VRED_MAX, rd, va, 0, 0)};
    }

    static Instruction VCount(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VCOUNT, rd, va, 0, 0)};
    }

    static Instruction VAny(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VANY, rd, va, 0, 0)};
    }

    static Instruction VAll(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VALL, rd, va, 0, 0)};
    }

    static Instruction VFirst(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VFIRST, rd, va, 0, 0)};
    }

    static Instruction VLast(u8 rd, u8 va) {
        return Instruction{Encode(Opcode::VLAST, rd, va, 0, 0)};
    }

    static Instruction VNth(u8 rd, u8 va, u8 n) {
        return Instruction{Encode(Opcode::VNTH, rd, va, 0, n & 0xFFF)};
    }

    // --- Window-Streaming Reduction (0xDD-0xDF) ---
    // Encodes: [op:8][vd:4][ra:4][rb:4][window:8][segId:4]
    // WDELTA: rb is the carry scalar register (holds prev value for diff).
    // WINDOW_SUM/WINDOW_MEAN: imm is the window size (1-255).
    static Instruction WDelta(u8 vd, u8 ra, u8 segId, u8 carryReg) {
        return Instruction{Encode(Opcode::WDELTA, vd, ra, carryReg,
            static_cast<u16>((segId & 0xF) << 8))};
    }
    static Instruction WindowSum(u8 vd, u8 ra, u8 segId, u8 window) {
        return Instruction{Encode(Opcode::WINDOW_SUM, vd, ra, 0,
            static_cast<u16>((window & 0xFF) | ((segId & 0xF) << 8)))};
    }
    static Instruction WindowMean(u8 vd, u8 ra, u8 segId, u8 window) {
        return Instruction{Encode(Opcode::WINDOW_MEAN, vd, ra, 0,
            static_cast<u16>((window & 0xFF) | ((segId & 0xF) << 8)))};
    }

    // ====================================================================
    // Aggregate Operators — 0xE0..0xEF
    // ====================================================================

    static Instruction AggCount(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_COUNT, rd, segId, 0, 0)};
    }

    static Instruction AggSum(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_SUM, rd, segId, 0, 0)};
    }

    static Instruction AggAvg(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_AVG, rd, segId, 0, 0)};
    }

    static Instruction AggMin(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_MIN, rd, segId, 0, 0)};
    }

    static Instruction AggMax(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_MAX, rd, segId, 0, 0)};
    }

    static Instruction AggFirst(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_FIRST, rd, segId, 0, 0)};
    }

    static Instruction AggLast(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_LAST, rd, segId, 0, 0)};
    }

    static Instruction AggStddev(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_STDDEV, rd, segId, 0, 0)};
    }

    static Instruction AggVariance(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_VARIANCE, rd, segId, 0, 0)};
    }

    static Instruction AggCountDistinct(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_COUNT_DISTINCT, rd, segId, 0, 0)};
    }

    static Instruction AggSumDistinct(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_SUM_DISTINCT, rd, segId, 0, 0)};
    }

    static Instruction AggMedian(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_MEDIAN, rd, segId, 0, 0)};
    }

    static Instruction AggMode(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_MODE, rd, segId, 0, 0)};
    }

    static Instruction AggPercentile(u8 rd, u8 segId) {
        return Instruction{Encode(Opcode::AGG_PERCENTILE, rd, segId, 0, 0)};
    }

    static Instruction HashInit(u8 rd, u8 keyReg) {
        return Instruction{Encode(Opcode::HASH_INIT, rd, keyReg, 0, 0)};
    }

    static Instruction HashProbe(u8 rd, u8 keyReg, u8 hashReg) {
        return Instruction{Encode(Opcode::HASH_PROBE, rd, keyReg, hashReg, 0)};
    }

    // ====================================================================
    // Hash / Sort / Join — 0xF0..0xFF
    // ====================================================================

    static Instruction HashBuild(u8 hashReg, u8 segId, u8 resultSegId) {
        return Instruction{Encode(Opcode::HASH_BUILD, hashReg, segId, resultSegId, 0)};
    }

    static Instruction HashLookup(u8 hashReg, u8 keyReg, u8 resultSegId) {
        return Instruction{Encode(Opcode::HASH_LOOKUP, hashReg, keyReg, resultSegId, 0)};
    }

    static Instruction SortAsc(u8 segId, u8 keyColReg) {
        return Instruction{Encode(Opcode::SORT_ASC, segId, keyColReg, 0, 0)};
    }

    static Instruction SortDesc(u8 segId, u8 keyColReg) {
        return Instruction{Encode(Opcode::SORT_DESC, segId, keyColReg, 0, 0)};
    }

    static Instruction SortTopk(u8 segId, u8 k, u8 keyColReg) {
        return Instruction{Encode(Opcode::SORT_TOPK, segId, keyColReg, k, 0)};
    }

    static Instruction SortBottomk(u8 segId, u8 k, u8 keyColReg) {
        return Instruction{Encode(Opcode::SORT_BOTTOMK, segId, keyColReg, k, 0)};
    }

    static Instruction JoinHash(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        return Instruction{Encode(Opcode::JOIN_HASH, leftSegId, rightSegId, resultSegId, 0)};
    }

    static Instruction JoinMerge(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        return Instruction{Encode(Opcode::JOIN_MERGE, leftSegId, rightSegId, resultSegId, 0)};
    }

    static Instruction JoinNested(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        return Instruction{Encode(Opcode::JOIN_NESTED, leftSegId, rightSegId, resultSegId, 0)};
    }

    static Instruction JoinAnti(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        return Instruction{Encode(Opcode::JOIN_ANTI, leftSegId, rightSegId, resultSegId, 0)};
    }

    static Instruction JoinSemi(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        return Instruction{Encode(Opcode::JOIN_SEMI, leftSegId, rightSegId, resultSegId, 0)};
    }

    static Instruction WindowRow(u8 segId, u8 startOff, u8 endOff, u8 keyColReg) {
        u16 imm = (static_cast<u16>(startOff & 0xF) << 8) | (endOff & 0xF);
        return Instruction{Encode(Opcode::WINDOW_ROW, segId, keyColReg, 0, imm)};
    }

    static Instruction WindowRange(u8 segId, u8 rangeVal, u8 keyColReg) {
        return Instruction{Encode(Opcode::WINDOW_RANGE, segId, keyColReg, 0, rangeVal & 0xFF)};
    }

    static Instruction PartitionHash(u8 segId, u8 numParts) {
        return Instruction{Encode(Opcode::PARTITION_HASH, segId, numParts, 0, 0)};
    }

    static Instruction Serialize(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::SERIALIZE, rd, ra, 0, 0)};
    }

    static Instruction Deserialize(u8 rd, u8 ra) {
        return Instruction{Encode(Opcode::DESERIALIZE, rd, ra, 0, 0)};
    }
};

struct ExtendedInstruction : Instruction {
    u32 ext1;
    u32 ext2;

    ExtendedInstruction()
        : Instruction(), ext1(0), ext2(0)
    {}

    ExtendedInstruction(u32 base, u32 e1, u32 e2)
        : Instruction(base), ext1(e1), ext2(e2)
    {}

    u32 ExtendedImm32() const {
        return ext1;
    }

    u64 ExtendedImm64() const {
        return (static_cast<u64>(ext2) << 32) | static_cast<u64>(ext1);
    }

    static ExtendedInstruction Mov64(u8 rd, u64 imm) {
        u32 base = Encode(Opcode::MOV, rd, 0, 0, 0);
        u32 lo   = static_cast<u32>(imm & 0xFFFFFFFFull);
        u32 hi   = static_cast<u32>((imm >> 32) & 0xFFFFFFFFull);
        return ExtendedInstruction{base, lo, hi};
    }
};

} // namespace voxel
