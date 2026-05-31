#pragma once

#include "voxel/bytecode/instruction.hpp"
#include "voxel/core/types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace voxel {

class Assembler {
public:
    struct Relocation {
        sz offset;
        std::string label;
        i32 addend;

        Relocation(sz off, std::string lbl, i32 add = 0)
            : offset(off), label(std::move(lbl)), addend(add) {}
    };

    Assembler()
        : Code_(), SymTable_(), Relocs_() {}

    void Emit(Instruction inst) {
        Code_.push_back(inst.raw);
    }

    void Emit(u32 raw) {
        Code_.push_back(raw);
    }

    void Emit(Opcode op, u8 rd, u8 ra, u8 rb, u16 imm) {
        Emit(Instruction::Encode(op, rd, ra, rb, imm));
    }

    sz Offset() const { return Code_.size(); }

    void Clear() {
        Code_.clear();
        SymTable_.clear();
        Relocs_.clear();
    }

    const std::vector<u32>& GetCode() const { return Code_; }
    std::vector<u32>&       GetCode()       { return Code_; }

    void Label(const std::string& name) { Bind(name); }

    void Bind(const std::string& name) {
        SymTable_[name] = Code_.size();
    }

    sz SymbolOffset(const std::string& name) const {
        auto it = SymTable_.find(name);
        return (it != SymTable_.end()) ? it->second : 0;
    }

    bool HasSymbol(const std::string& name) const {
        return SymTable_.find(name) != SymTable_.end();
    }

    // ========================================================================
    // Control — 0x00..0x0F
    // ========================================================================

    void Halt()     { Emit(Instruction::Halt()); }
    void Nop()      { Emit(Instruction::Nop()); }
    void Trap()     { Emit(Instruction::Trap()); }
    void Break()    { Emit(Instruction::Break()); }
    void Yield()    { Emit(Instruction::Yield()); }
    void Barrier()  { Emit(Instruction::Barrier()); }
    void Prefetch() { Emit(Instruction::Prefetch()); }
    void FlushCache() { Emit(Instruction::FlushCache()); }
    void Sync()     { Emit(Instruction::Sync()); }
    void Memfence() { Emit(Instruction::Memfence()); }

    // ========================================================================
    // Scalar Move — 0x10..0x1F
    // ========================================================================

    void Mov(u8 rd, i16 imm) {
        Emit(Instruction::Mov(rd, imm));
    }

    void Movr(u8 rd, u8 ra) {
        Emit(Instruction::Movr(rd, ra));
    }

    void Addi(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Addi(rd, ra, imm));
    }

    void Subi(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Subi(rd, ra, imm));
    }

    void Muli(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Muli(rd, ra, imm));
    }

    void Andi(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Andi(rd, ra, imm));
    }

    void Ori(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Ori(rd, ra, imm));
    }

    void Xori(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Xori(rd, ra, imm));
    }

    void Shli(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Shli(rd, ra, imm));
    }

    void Shri(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::Shri(rd, ra, imm));
    }

    void SarI(u8 rd, u8 ra, i16 imm) {
        Emit(Instruction::SarI(rd, ra, imm));
    }

    void Movz(u8 rd, i16 imm) {
        Emit(Instruction::Movz(rd, imm));
    }

    void Movn(u8 rd, i16 imm) {
        Emit(Instruction::Movn(rd, imm));
    }

    void Movk(u8 rd, i16 imm) {
        Emit(Instruction::Movk(rd, imm));
    }

    void Lea(u8 rd, u8 ra, i16 off) {
        Emit(Instruction::Lea(rd, ra, off));
    }

    // ========================================================================
    // Scalar Arithmetic — 0x20..0x2F
    // ========================================================================

    void Add(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Add(rd, ra, rb));
    }

    void Sub(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Sub(rd, ra, rb));
    }

    void Mul(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Mul(rd, ra, rb));
    }

    void Div(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Div(rd, ra, rb));
    }

    void Mod(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Mod(rd, ra, rb));
    }

    void Neg(u8 rd, u8 ra) {
        Emit(Instruction::Neg(rd, ra));
    }

    void Abs(u8 rd, u8 ra) {
        Emit(Instruction::Abs(rd, ra));
    }

    void Min(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Min(rd, ra, rb));
    }

    void Max(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Max(rd, ra, rb));
    }

    void Avg(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Avg(rd, ra, rb));
    }

    void Addf(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Addf(rd, ra, rb));
    }

    void Subf(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Subf(rd, ra, rb));
    }

    void Mulf(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Mulf(rd, ra, rb));
    }

    void Divf(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Divf(rd, ra, rb));
    }

    void Negf(u8 rd, u8 ra) {
        Emit(Instruction::Negf(rd, ra));
    }

    void Absf(u8 rd, u8 ra) {
        Emit(Instruction::Absf(rd, ra));
    }

    // ========================================================================
    // Scalar Bitwise — 0x30..0x3F
    // ========================================================================

    void And(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::And(rd, ra, rb));
    }

    void Or(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Or(rd, ra, rb));
    }

    void Xor(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Xor(rd, ra, rb));
    }

    void Not(u8 rd, u8 ra) {
        Emit(Instruction::Not(rd, ra));
    }

    void Shl(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Shl(rd, ra, rb));
    }

    void Shr(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Shr(rd, ra, rb));
    }

    void Sar(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Sar(rd, ra, rb));
    }

    void Rol(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Rol(rd, ra, rb));
    }

    void Ror(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Ror(rd, ra, rb));
    }

    void Popcnt(u8 rd, u8 ra) {
        Emit(Instruction::Popcnt(rd, ra));
    }

    void Clz(u8 rd, u8 ra) {
        Emit(Instruction::Clz(rd, ra));
    }

    void Ctz(u8 rd, u8 ra) {
        Emit(Instruction::Ctz(rd, ra));
    }

    void Bswap(u8 rd, u8 ra) {
        Emit(Instruction::Bswap(rd, ra));
    }

    void Bextr(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Bextr(rd, ra, rb));
    }

    void Bzhi(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Bzhi(rd, ra, rb));
    }

    void Pdep(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Pdep(rd, ra, rb));
    }

    // ========================================================================
    // Comparison — 0x40..0x4F
    // ========================================================================

    void Cmp(u8 ra, u8 rb) {
        Emit(Instruction::Cmp(ra, rb));
    }

    void Cmpf(u8 ra, u8 rb) {
        Emit(Instruction::Cmpf(ra, rb));
    }

    void Cmpu(u8 ra, u8 rb) {
        Emit(Instruction::Cmpu(ra, rb));
    }

    void Tst(u8 ra, u8 rb) {
        Emit(Instruction::Tst(ra, rb));
    }

    void Tstf(u8 ra, u8 rb) {
        Emit(Instruction::Tstf(ra, rb));
    }

    void Isnull(u8 rd, u8 ra) {
        Emit(Instruction::Isnull(rd, ra));
    }

    void Isnotnull(u8 rd, u8 ra) {
        Emit(Instruction::Isnotnull(rd, ra));
    }

    void Select(u8 rd, u8 ra, u8 rb) {
        Emit(Instruction::Select(rd, ra, rb));
    }

    void Selectv(u8 rd, u8 va, u8 vb) {
        Emit(Instruction::Selectv(rd, va, vb));
    }

    // ========================================================================
    // Branching — 0x50..0x5F (immediate offset)
    // ========================================================================

    void Jmp(i16 offset)  { Emit(Instruction::Jmp(offset)); }
    void Jz(i16 offset)   { Emit(Instruction::Jz(offset)); }
    void Jnz(i16 offset)  { Emit(Instruction::Jnz(offset)); }
    void Js(i16 offset)   { Emit(Instruction::Js(offset)); }
    void Jns(i16 offset)  { Emit(Instruction::Jns(offset)); }
    void Jo(i16 offset)   { Emit(Instruction::Jo(offset)); }
    void Jno(i16 offset)  { Emit(Instruction::Jno(offset)); }
    void Jc(i16 offset)   { Emit(Instruction::Jc(offset)); }
    void Jnc(i16 offset)  { Emit(Instruction::Jnc(offset)); }
    void Jl(i16 offset)   { Emit(Instruction::Jl(offset)); }
    void Jle(i16 offset)  { Emit(Instruction::Jle(offset)); }
    void Jg(i16 offset)   { Emit(Instruction::Jg(offset)); }
    void Jge(i16 offset)  { Emit(Instruction::Jge(offset)); }
    void Call(i16 offset) { Emit(Instruction::Call(offset)); }
    void Ret()            { Emit(Instruction::Ret()); }
    void TableJmp(u8 ra)  { Emit(Instruction::TableJmp(ra)); }

    // ========================================================================
    // Branching — label-based (records relocation, emits placeholder)
    // ========================================================================

    void Jmp(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jmp(0)); }
    void Jz(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jz(0)); }
    void Jnz(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jnz(0)); }
    void Js(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Js(0)); }
    void Jns(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jns(0)); }
    void Jo(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jo(0)); }
    void Jno(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jno(0)); }
    void Jc(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jc(0)); }
    void Jnc(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jnc(0)); }
    void Jl(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jl(0)); }
    void Jle(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jle(0)); }
    void Jg(const std::string& label)   { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jg(0)); }
    void Jge(const std::string& label)  { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Jge(0)); }
    void Call(const std::string& label) { Relocs_.emplace_back(Code_.size(), label, 0); Emit(Instruction::Call(0)); }

    // ========================================================================
    // Type Conversion — 0x60..0x6F
    // ========================================================================

    void CvtI8(u8 rd, u8 ra)      { Emit(Instruction::CvtI8(rd, ra)); }
    void CvtI16(u8 rd, u8 ra)     { Emit(Instruction::CvtI16(rd, ra)); }
    void CvtI32(u8 rd, u8 ra)     { Emit(Instruction::CvtI32(rd, ra)); }
    void CvtI64(u8 rd, u8 ra)     { Emit(Instruction::CvtI64(rd, ra)); }
    void CvtF32(u8 rd, u8 ra)     { Emit(Instruction::CvtF32(rd, ra)); }
    void CvtF64(u8 rd, u8 ra)     { Emit(Instruction::CvtF64(rd, ra)); }
    void CvtU8(u8 rd, u8 ra)      { Emit(Instruction::CvtU8(rd, ra)); }
    void CvtU16(u8 rd, u8 ra)     { Emit(Instruction::CvtU16(rd, ra)); }
    void CvtU32(u8 rd, u8 ra)     { Emit(Instruction::CvtU32(rd, ra)); }
    void CvtU64(u8 rd, u8 ra)     { Emit(Instruction::CvtU64(rd, ra)); }
    void Bitcast(u8 rd, u8 ra)     { Emit(Instruction::Bitcast(rd, ra)); }
    void Reinterpret(u8 rd, u8 ra) { Emit(Instruction::Reinterpret(rd, ra)); }
    void Trunc(u8 rd, u8 ra)       { Emit(Instruction::Trunc(rd, ra)); }
    void Round(u8 rd, u8 ra)       { Emit(Instruction::Round(rd, ra)); }
    void Ceil(u8 rd, u8 ra)        { Emit(Instruction::Ceil(rd, ra)); }
    void Floor(u8 rd, u8 ra)       { Emit(Instruction::Floor(rd, ra)); }

    // ========================================================================
    // Vector I/O — 0x70..0x7F
    // ========================================================================

    void VLoad(u8 vd, u8 ra, u8 segId, u8 count) {
        Emit(Instruction::VLoad(vd, ra, segId, count));
    }

    void VStore(u8 vs, u8 ra, u8 segId, u8 count) {
        Emit(Instruction::VStore(vs, ra, segId, count));
    }

    void VGather(u8 vd, u8 ra, u8 segId, u8 count) {
        Emit(Instruction::VGather(vd, ra, segId, count));
    }

    void VScatter(u8 vs, u8 ra, u8 segId, u8 count) {
        Emit(Instruction::VScatter(vs, ra, segId, count));
    }

    void VLoadStrided(u8 vd, u8 ra, u8 stride, u8 count) {
        Emit(Instruction::VLoadStrided(vd, ra, stride, count));
    }

    void VStoreStrided(u8 vs, u8 ra, u8 stride, u8 count) {
        Emit(Instruction::VStoreStrided(vs, ra, stride, count));
    }

    void VLoadMasked(u8 vd, u8 ra, u8 maskReg) {
        Emit(Instruction::VLoadMasked(vd, ra, maskReg));
    }

    void VStoreMasked(u8 vs, u8 ra, u8 maskReg) {
        Emit(Instruction::VStoreMasked(vs, ra, maskReg));
    }

    void VSplat(u8 vd, u8 ra) {
        Emit(Instruction::VSplat(vd, ra));
    }

    void VExtract(u8 rd, u8 va, u8 lane) {
        Emit(Instruction::VExtract(rd, va, lane));
    }

    void VInsert(u8 vd, u8 va, u8 rb, u8 lane) {
        Emit(Instruction::VInsert(vd, va, rb, lane));
    }

    void VPermute(u8 vd, u8 va, u8 vb) {
        Emit(Instruction::VPermute(vd, va, vb));
    }

    void VShuffle(u8 vd, u8 va, u8 vb) {
        Emit(Instruction::VShuffle(vd, va, vb));
    }

    void VReverse(u8 vd, u8 va) {
        Emit(Instruction::VReverse(vd, va));
    }

    void VRotate(u8 vd, u8 va, u8 amount) {
        Emit(Instruction::VRotate(vd, va, amount));
    }

    void VSlide(u8 vd, u8 va, u8 amount) {
        Emit(Instruction::VSlide(vd, va, amount));
    }

    // ========================================================================
    // Vector Arithmetic — 0x80..0x8F
    // ========================================================================

    void VAdd(u8 vd, u8 va, u8 vb) { Emit(Instruction::VAdd(vd, va, vb)); }
    void VSub(u8 vd, u8 va, u8 vb) { Emit(Instruction::VSub(vd, va, vb)); }
    void VMul(u8 vd, u8 va, u8 vb) { Emit(Instruction::VMul(vd, va, vb)); }
    void VDiv(u8 vd, u8 va, u8 vb) { Emit(Instruction::VDiv(vd, va, vb)); }
    void VMod(u8 vd, u8 va, u8 vb) { Emit(Instruction::VMod(vd, va, vb)); }
    void VNeg(u8 vd, u8 va)        { Emit(Instruction::VNeg(vd, va)); }
    void VAbs(u8 vd, u8 va)        { Emit(Instruction::VAbs(vd, va)); }
    void VMin(u8 vd, u8 va, u8 vb) { Emit(Instruction::VMin(vd, va, vb)); }
    void VMax(u8 vd, u8 va, u8 vb) { Emit(Instruction::VMax(vd, va, vb)); }
    void VAvg(u8 vd, u8 va, u8 vb) { Emit(Instruction::VAvg(vd, va, vb)); }
    void VFma(u8 vd, u8 va, u8 vb) { Emit(Instruction::VFma(vd, va, vb)); }
    void VFms(u8 vd, u8 va, u8 vb) { Emit(Instruction::VFms(vd, va, vb)); }
    void VSqrt(u8 vd, u8 va)       { Emit(Instruction::VSqrt(vd, va)); }
    void VRSqrt(u8 vd, u8 va)      { Emit(Instruction::VRSqrt(vd, va)); }
    void VRcp(u8 vd, u8 va)        { Emit(Instruction::VRcp(vd, va)); }
    void VPow(u8 vd, u8 va, u8 vb) { Emit(Instruction::VPow(vd, va, vb)); }

    // ========================================================================
    // Vector Scalar Arithmetic — 0x90..0x9F
    // ========================================================================

    void VSadd(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSadd(vd, va, rb)); }
    void VSsub(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSsub(vd, va, rb)); }
    void VSmul(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSmul(vd, va, rb)); }
    void VSdiv(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSdiv(vd, va, rb)); }
    void VSmod(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSmod(vd, va, rb)); }
    void VSmin(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSmin(vd, va, rb)); }
    void VSmax(u8 vd, u8 va, u8 rb) { Emit(Instruction::VSmax(vd, va, rb)); }

    // ========================================================================
    // Vector Comparison — 0xA0..0xAF
    // ========================================================================

    void VCmpeq(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmpeq(vd, va, vb)); }
    void VCmpne(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmpne(vd, va, vb)); }
    void VCmplt(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmplt(vd, va, vb)); }
    void VCmple(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmple(vd, va, vb)); }
    void VCmpgt(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmpgt(vd, va, vb)); }
    void VCmpge(u8 vd, u8 va, u8 vb)      { Emit(Instruction::VCmpge(vd, va, vb)); }
    void VCmpnull(u8 vd, u8 va)           { Emit(Instruction::VCmpnull(vd, va)); }
    void VCmpnotnull(u8 vd, u8 va)        { Emit(Instruction::VCmpnotnull(vd, va)); }

    // ========================================================================
    // Vector Logical — 0xB0..0xBF
    // ========================================================================

    void VAnd(u8 vd, u8 va, u8 vb)  { Emit(Instruction::VAnd(vd, va, vb)); }
    void VOr(u8 vd, u8 va, u8 vb)   { Emit(Instruction::VOr(vd, va, vb)); }
    void VXor(u8 vd, u8 va, u8 vb)  { Emit(Instruction::VXor(vd, va, vb)); }
    void VNot(u8 vd, u8 va)         { Emit(Instruction::VNot(vd, va)); }
    void VAndn(u8 vd, u8 va, u8 vb) { Emit(Instruction::VAndn(vd, va, vb)); }
    void VShl(u8 vd, u8 va, u8 vb)  { Emit(Instruction::VShl(vd, va, vb)); }
    void VShr(u8 vd, u8 va, u8 vb)  { Emit(Instruction::VShr(vd, va, vb)); }
    void VSar(u8 vd, u8 va, u8 vb)  { Emit(Instruction::VSar(vd, va, vb)); }

    // ========================================================================
    // Vector Filter — 0xC0..0xCF
    // ========================================================================

    void VFilter(u8 vd, u8 va, u8 rb, u8 mode) {
        Emit(Instruction::VFilter(vd, va, rb, mode));
    }

    void VFilterEq(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterEq(vd, va, rb)); }
    void VFilterNe(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterNe(vd, va, rb)); }
    void VFilterLt(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterLt(vd, va, rb)); }
    void VFilterLe(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterLe(vd, va, rb)); }
    void VFilterGt(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterGt(vd, va, rb)); }
    void VFilterGe(u8 vd, u8 va, u8 rb)  { Emit(Instruction::VFilterGe(vd, va, rb)); }

    void VBlend(u8 vd, u8 va, u8 vb, u8 maskReg) {
        Emit(Instruction::VBlend(vd, va, vb, maskReg));
    }

    void VMaskStore(u8 vs, u8 ra, u8 maskReg, u8 segId) {
        Emit(Instruction::VMaskStore(vs, ra, maskReg, segId));
    }

    void VMaskLoad(u8 vd, u8 ra, u8 maskReg, u8 segId) {
        Emit(Instruction::VMaskLoad(vd, ra, maskReg, segId));
    }

    // ========================================================================
    // Vector Reduction — 0xD0..0xDF
    // ========================================================================

    void VSum(u8 rd, u8 va)       { Emit(Instruction::VSum(rd, va)); }
    void VProd(u8 rd, u8 va)      { Emit(Instruction::VProd(rd, va)); }
    void VMean(u8 rd, u8 va)      { Emit(Instruction::VMean(rd, va)); }
    void VStddev(u8 rd, u8 va)    { Emit(Instruction::VStddev(rd, va)); }
    void VVariance(u8 rd, u8 va)  { Emit(Instruction::VVariance(rd, va)); }
    void VRedMin(u8 rd, u8 va)    { Emit(Instruction::VRedMin(rd, va)); }
    void VRedMax(u8 rd, u8 va)    { Emit(Instruction::VRedMax(rd, va)); }
    void VCount(u8 rd, u8 va)     { Emit(Instruction::VCount(rd, va)); }
    void VAny(u8 rd, u8 va)       { Emit(Instruction::VAny(rd, va)); }
    void VAll(u8 rd, u8 va)       { Emit(Instruction::VAll(rd, va)); }
    void VFirst(u8 rd, u8 va)     { Emit(Instruction::VFirst(rd, va)); }
    void VLast(u8 rd, u8 va)      { Emit(Instruction::VLast(rd, va)); }
    void VNth(u8 rd, u8 va, u8 n) { Emit(Instruction::VNth(rd, va, n)); }

    // ========================================================================
    // Aggregate Operators — 0xE0..0xEF
    // ========================================================================

    void AggCount(u8 rd, u8 segId)          { Emit(Instruction::AggCount(rd, segId)); }
    void AggSum(u8 rd, u8 segId)            { Emit(Instruction::AggSum(rd, segId)); }
    void AggAvg(u8 rd, u8 segId)            { Emit(Instruction::AggAvg(rd, segId)); }
    void AggMin(u8 rd, u8 segId)            { Emit(Instruction::AggMin(rd, segId)); }
    void AggMax(u8 rd, u8 segId)            { Emit(Instruction::AggMax(rd, segId)); }
    void AggFirst(u8 rd, u8 segId)          { Emit(Instruction::AggFirst(rd, segId)); }
    void AggLast(u8 rd, u8 segId)           { Emit(Instruction::AggLast(rd, segId)); }
    void AggStddev(u8 rd, u8 segId)         { Emit(Instruction::AggStddev(rd, segId)); }
    void AggVariance(u8 rd, u8 segId)       { Emit(Instruction::AggVariance(rd, segId)); }
    void AggCountDistinct(u8 rd, u8 segId)  { Emit(Instruction::AggCountDistinct(rd, segId)); }
    void AggSumDistinct(u8 rd, u8 segId)    { Emit(Instruction::AggSumDistinct(rd, segId)); }
    void AggMedian(u8 rd, u8 segId)         { Emit(Instruction::AggMedian(rd, segId)); }
    void AggMode(u8 rd, u8 segId)           { Emit(Instruction::AggMode(rd, segId)); }
    void AggPercentile(u8 rd, u8 segId)     { Emit(Instruction::AggPercentile(rd, segId)); }

    void HashInit(u8 rd, u8 keyReg) {
        Emit(Instruction::HashInit(rd, keyReg));
    }

    void HashProbe(u8 rd, u8 keyReg, u8 hashReg) {
        Emit(Instruction::HashProbe(rd, keyReg, hashReg));
    }

    // ========================================================================
    // Hash / Sort / Join — 0xF0..0xFF
    // ========================================================================

    void HashBuild(u8 hashReg, u8 segId, u8 resultSegId) {
        Emit(Instruction::HashBuild(hashReg, segId, resultSegId));
    }

    void HashLookup(u8 hashReg, u8 keyReg, u8 resultSegId) {
        Emit(Instruction::HashLookup(hashReg, keyReg, resultSegId));
    }

    void SortAsc(u8 segId, u8 keyColReg) {
        Emit(Instruction::SortAsc(segId, keyColReg));
    }

    void SortDesc(u8 segId, u8 keyColReg) {
        Emit(Instruction::SortDesc(segId, keyColReg));
    }

    void SortTopk(u8 segId, u8 k, u8 keyColReg) {
        Emit(Instruction::SortTopk(segId, k, keyColReg));
    }

    void SortBottomk(u8 segId, u8 k, u8 keyColReg) {
        Emit(Instruction::SortBottomk(segId, k, keyColReg));
    }

    void JoinHash(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        Emit(Instruction::JoinHash(leftSegId, rightSegId, resultSegId));
    }

    void JoinMerge(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        Emit(Instruction::JoinMerge(leftSegId, rightSegId, resultSegId));
    }

    void JoinNested(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        Emit(Instruction::JoinNested(leftSegId, rightSegId, resultSegId));
    }

    void JoinAnti(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        Emit(Instruction::JoinAnti(leftSegId, rightSegId, resultSegId));
    }

    void JoinSemi(u8 leftSegId, u8 rightSegId, u8 resultSegId) {
        Emit(Instruction::JoinSemi(leftSegId, rightSegId, resultSegId));
    }

    void WindowRow(u8 segId, u8 startOff, u8 endOff, u8 keyColReg) {
        Emit(Instruction::WindowRow(segId, startOff, endOff, keyColReg));
    }

    void WindowRange(u8 segId, u8 rangeVal, u8 keyColReg) {
        Emit(Instruction::WindowRange(segId, rangeVal, keyColReg));
    }

    void PartitionHash(u8 segId, u8 numParts) {
        Emit(Instruction::PartitionHash(segId, numParts));
    }

    void Serialize(u8 rd, u8 ra) {
        Emit(Instruction::Serialize(rd, ra));
    }

    void Deserialize(u8 rd, u8 ra) {
        Emit(Instruction::Deserialize(rd, ra));
    }

    // ========================================================================
    // Multi-word encoding support
    // ========================================================================

    void EmitExtended(const ExtendedInstruction& ext) {
        Emit(ext.raw);
        Emit(ext.ext1);
        Emit(ext.ext2);
    }

    void Mov64(u8 rd, u64 imm) {
        EmitExtended(ExtendedInstruction::Mov64(rd, imm));
    }

    // ========================================================================
    // Relocation finalization
    // ========================================================================

    bool Finalize() {
        for (auto& reloc : Relocs_) {
            auto it = SymTable_.find(reloc.label);
            if (it == SymTable_.end()) {
                return false;
            }

            i64 target  = static_cast<i64>(it->second) + reloc.addend;
            i64 current = static_cast<i64>(reloc.offset);
            i64 branchOffset = target - current - 1;

            if (branchOffset < -2048 || branchOffset > 2047) {
                return false;
            }

            u32 old = Code_[reloc.offset];
            u32 mask = 0x000FFFFFu;
            u32 shift = static_cast<u32>(static_cast<u16>(branchOffset)) & 0xFFF;
            Code_[reloc.offset] = (old & mask) | (shift << 20);
        }
        Relocs_.clear();
        return true;
    }

    // ========================================================================
    // High-level helpers for common patterns
    // ========================================================================

    void MovImm(u8 rd, i16 imm)   { Mov(rd, imm); }
    void LoadImm(u8 rd, i16 imm)  { Mov(rd, imm); }
    void ClearReg(u8 rd)          { Mov(rd, 0); }
    void CopyReg(u8 rd, u8 ra)    { Movr(rd, ra); }
    void BranchIfZero(const std::string& label)   { Jz(label); }
    void BranchIfNotZero(const std::string& label) { Jnz(label); }
    void UnconditionalBranch(const std::string& label) { Jmp(label); }

private:
    std::vector<u32> Code_;
    std::unordered_map<std::string, sz> SymTable_;
    std::vector<Relocation> Relocs_;
};

} // namespace voxel
