#pragma once

#include "voxel/bytecode/opcodes.hpp"
#include "voxel/bytecode/instruction.hpp"
#include "voxel/core/types.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <ostream>
#include <cstring>
#include <chrono>

namespace voxel {
namespace opt {

using Clock = std::chrono::high_resolution_clock;

inline bool IsBranchOp(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0x50 && v <= 0x5E;
}

inline bool IsBranchOrRet(Opcode op) {
    u8 v = static_cast<u8>(op);
    return (v >= 0x50 && v <= 0x5E) || v == 0x5E;
}

inline bool IsUnconditionalJump(Opcode op) {
    return op == Opcode::JMP;
}

inline bool IsConditionalBranch(Opcode op) {
    u8 v = static_cast<u8>(op);
    return v >= 0x51 && v <= 0x5C;
}

inline bool Terminates(Opcode op) {
    return op == Opcode::HALT || op == Opcode::RET || op == Opcode::TRAP;
}

inline i16 ExtractSimm12(u32 raw) {
    u16 imm12 = (raw >> 20) & 0xFFF;
    if (imm12 & 0x800)
        return static_cast<i16>(imm12 | 0xF000);
    return static_cast<i16>(imm12);
}

inline u32 SetSimm12(u32 raw, i16 val) {
    raw &= ~(0xFFFul << 20);
    raw |= (static_cast<u32>(static_cast<u16>(val)) & 0xFFFu) << 20;
    return raw;
}

inline Opcode ExtractOp(u32 raw) {
    return static_cast<Opcode>(raw & 0xFF);
}

inline u8 ExtractRd(u32 raw) {
    return (raw >> 8) & 0xF;
}

inline u8 ExtractRa(u32 raw) {
    return (raw >> 12) & 0xF;
}

inline u8 ExtractRb(u32 raw) {
    return (raw >> 16) & 0xF;
}

inline sz BranchTarget(sz pc, u32 raw) {
    i16 off = ExtractSimm12(raw);
    return static_cast<sz>(static_cast<isz>(pc) + off);
}

inline u32 MakeNop() {
    return static_cast<u32>(Opcode::NOP);
}

inline bool IsNop(u32 raw) {
    return ExtractOp(raw) == Opcode::NOP;
}

inline bool IsScalarArithImm(Opcode op) {
    u8 v = static_cast<u8>(op);
    return (v >= 0x12 && v <= 0x19) || v == 0x1A || v == 0x1E;
}

inline bool IsScalarArithReg(Opcode op) {
    u8 v = static_cast<u8>(op);
    return (v >= 0x20 && v <= 0x2D) || (v >= 0x30 && v <= 0x38);
}

inline u32 EncodeMov(u8 rd, i16 imm) {
    return Instruction::Encode(Opcode::MOV, rd, 0, 0, static_cast<u16>(imm));
}

inline u32 EncodeAddi(u8 rd, u8 ra, i16 imm) {
    return Instruction::Encode(Opcode::ADDI, rd, ra, 0, static_cast<u16>(imm));
}

inline u32 EncodeSubi(u8 rd, u8 ra, i16 imm) {
    return Instruction::Encode(Opcode::SUBI, rd, ra, 0, static_cast<u16>(imm));
}

inline u32 EncodeMuli(u8 rd, u8 ra, i16 imm) {
    return Instruction::Encode(Opcode::MULI, rd, ra, 0, static_cast<u16>(imm));
}

struct OptimizationPass {
    virtual ~OptimizationPass() = default;
    virtual const char* Name() const = 0;
    virtual bool Run(std::vector<u32>& code, sz& pc) = 0; // returns true if any changes made
};

// ============================================================================
// ConstantFolder — evaluates compile-time constant expressions
// ============================================================================

class ConstantFolder : public OptimizationPass {
public:
    const char* Name() const override { return "ConstantFolder"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        ValueMap constants;
        sz codeSize = code.size();

        for (sz i = 0; i < codeSize; ++i) {
            u32 raw = code[i];
            Opcode op = ExtractOp(raw);
            u8 rd = ExtractRd(raw);
            u8 ra = ExtractRa(raw);
            u8 rb = ExtractRb(raw);
            i16 simm12 = ExtractSimm12(raw);

            if (IsBranchOrRet(op) || Terminates(op) || op == Opcode::TRAP || op == Opcode::CALL || op == Opcode::TABLE_JMP) {
                constants.clear();
                continue;
            }

            bool writesDest = (op != Opcode::CMP && op != Opcode::CMPF &&
                               op != Opcode::CMPU && op != Opcode::TST && op != Opcode::TSTF &&
                               op != Opcode::JMP);

            if (writesDest) {
                constants.erase(rd);
            }

            switch (op) {
            case Opcode::MOV: {
                constants[rd] = static_cast<i64>(simm12);
                break;
            }
            case Opcode::MOVR: {
                if (ra != rd) {
                    auto it = constants.find(ra);
                    if (it != constants.end()) {
                        constants[rd] = it->second;
                    }
                }
                break;
            }
            case Opcode::ADDI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    i64 result = it->second + static_cast<i64>(simm12);
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                }
                break;
            }
            case Opcode::SUBI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    i64 result = it->second - static_cast<i64>(simm12);
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                }
                break;
            }
            case Opcode::MULI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    i64 result = it->second * static_cast<i64>(simm12);
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                }
                break;
            }
            case Opcode::ANDI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second & static_cast<i64>(simm12);
                }
                break;
            }
            case Opcode::ORI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second | static_cast<i64>(simm12);
                }
                break;
            }
            case Opcode::XORI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second ^ static_cast<i64>(simm12);
                }
                break;
            }
            case Opcode::SHLI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second << (simm12 & 0x3F);
                }
                break;
            }
            case Opcode::SHRI: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    i64 val = it->second;
                    if (val >= 0) constants[rd] = static_cast<i64>(static_cast<u64>(val) >> (simm12 & 0x3F));
                    else constants[rd] = val >> (simm12 & 0x3F);
                }
                break;
            }
            case Opcode::LEA: {
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second + static_cast<i64>(simm12);
                }
                break;
            }
            case Opcode::ADD: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    i64 result = itA->second + itB->second;
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::SUB: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    i64 result = itA->second - itB->second;
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::MUL: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    i64 result = itA->second * itB->second;
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::DIV: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end() && itB->second != 0) {
                    i64 result = itA->second / itB->second;
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::MOD: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end() && itB->second != 0) {
                    i64 result = itA->second % itB->second;
                    if (FitsInI12(result)) {
                        code[i] = EncodeMov(rd, static_cast<i16>(result));
                        constants[rd] = result;
                        changed = true;
                        ChangedCount_++;
                    } else {
                        constants[rd] = result;
                    }
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::AND: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    constants[rd] = itA->second & itB->second;
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::OR: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    constants[rd] = itA->second | itB->second;
                } else {
                    constants.erase(rd);
                }
                break;
            }
            case Opcode::XOR: {
                auto itA = constants.find(ra);
                auto itB = constants.find(rb);
                if (itA != constants.end() && itB != constants.end()) {
                    constants[rd] = itA->second ^ itB->second;
                } else {
                    constants.erase(rd);
                }
                break;
            }
            default:
                break;
            }

            // Track dead MOVs: if MOV Rd, imm is followed by another MOV/MOVR to Rd
            // without Rd being read, mark it as dead. We do a second pass for that.
        }

        // Second pass: remove dead MOVs (MOV Rd, X followed by MOV/MOVR/ADDI that overwrites
        // Rd without Rd being read in between)
        for (sz i = 1; i < codeSize; ++i) {
            u32 raw = code[i];
            u8 rd = ExtractRd(raw);
            Opcode preOp = ExtractOp(code[i - 1]);
            u8 preRd = ExtractRd(code[i - 1]);

            if (preOp == Opcode::MOV && preRd == rd) {
                bool rdReadBefore = false;
                for (sz j = i - 1; (isz)j >= 0; --j) {
                    if (j == i - 1) continue;
                    u32 chkRaw = code[j];
                    Opcode chkOp = ExtractOp(chkRaw);
                    u8 chkRa = ExtractRa(chkRaw);
                    u8 chkRb = ExtractRb(chkRaw);
                    if (ExtractRd(chkRaw) == rd && (chkOp == Opcode::MOV || chkOp == Opcode::MOVR)) {
                        rdReadBefore = false;
                        break;
                    }
                    if (chkRa == rd || chkRb == rd) {
                        rdReadBefore = true;
                        break;
                    }
                    if (IsBranchOrRet(chkOp) || Terminates(chkOp)) break;
                }
                if (!rdReadBefore) {
                    code[i - 1] = MakeNop();
                    changed = true;
                    ChangedCount_++;
                }
            }
        }

        return changed;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    using ValueMap = std::unordered_map<u8, i64>;
    sz ChangedCount_ = 0;

    static bool FitsInI12(i64 val) {
        return val >= -2048 && val <= 2047;
    }
};

// ============================================================================
// DeadCodeEliminator — removes instructions whose result is never used
// ============================================================================

class DeadCodeEliminator : public OptimizationPass {
public:
    const char* Name() const override { return "DeadCodeEliminator"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        sz codeSize = code.size();
        (void)pc;

        for (sz i = 0; i < codeSize; ++i) {
            u32 raw = code[i];
            Opcode op = ExtractOp(raw);
            u8 rd = ExtractRd(raw);

            if (IsNop(code[i])) continue;
            if (IsBranchOrRet(op)) continue;
            if (Terminates(op)) continue;
            if (op == Opcode::TRAP || op == Opcode::BREAK) continue;

            if (!WritesDestReg(op)) continue;

            sz nextWr = codeSize;
            for (sz j = i + 1; j < codeSize; ++j) {
                u32 fwd = code[j];
                if (IsNop(fwd)) continue;
                u8 fRd = ExtractRd(fwd);
                Opcode fOp = ExtractOp(fwd);

                if (WritesDestReg(fOp) && fRd == rd) { nextWr = j; break; }

                u8 fRa = ExtractRa(fwd);
                u8 fRb = ExtractRb(fwd);
                if (fRa == rd || fRb == rd) { nextWr = codeSize + 1; break; }

                if (IsBranchOrRet(fOp) || Terminates(fOp)) break;
            }

            if (nextWr < codeSize && nextWr != codeSize + 1) {
                code[i] = MakeNop();
                changed = true;
                ChangedCount_++;
            }
        }

        return changed;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    sz ChangedCount_ = 0;

    static bool WritesDestReg(Opcode op) {
        u8 v = static_cast<u8>(op);
        return v != 0x00 && v != 0x01 && v != 0x02 && v != 0x03
            && v != 0x04 && v != 0x05 && v != 0x06 && v != 0x07
            && v != 0x08 && v != 0x09
            && op != Opcode::CMP && op != Opcode::CMPF
            && op != Opcode::CMPU && op != Opcode::TST && op != Opcode::TSTF
            && op != Opcode::JMP && op != Opcode::RET && op != Opcode::TABLE_JMP
            && op != Opcode::TRAP && op != Opcode::BREAK
            && op != Opcode::YIELD && op != Opcode::BARRIER
            && op != Opcode::PREFETCH && op != Opcode::FLUSH_CACHE
            && op != Opcode::SYNC && op != Opcode::MEMFENCE;
    }
};

// ============================================================================
// NopSqueezer — removes NOPs and adjusts branch targets
// ============================================================================

class NopSqueezer : public OptimizationPass {
public:
    const char* Name() const override { return "NopSqueezer"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        sz codeSize = code.size();
        bool hasNops = false;
        for (sz i = 0; i < codeSize; ++i) {
            if (IsNop(code[i])) { hasNops = true; break; }
        }
        if (!hasNops) return false;

        // Build old->new PC mapping
        std::vector<isz> newPcFromOld(codeSize, -1);
        sz idx = 0;
        for (sz i = 0; i < codeSize; ++i) {
            if (!IsNop(code[i])) {
                newPcFromOld[i] = static_cast<isz>(idx);
                ++idx;
            }
        }

        for (sz i = 0; i < codeSize; ++i) {
            u32 raw = code[i];
            Opcode op = ExtractOp(raw);
            if (IsNop(raw)) continue;

            if (IsBranchOrRet(op) && op != Opcode::RET && op != Opcode::TABLE_JMP) {
                i16 oldOff = ExtractSimm12(raw);
                sz oldTarget = static_cast<sz>(static_cast<isz>(i) + oldOff);
                if (oldTarget < codeSize) {
                    // Find the nearest non-NOP before/at oldTarget
                    isz adjustedTarget = newPcFromOld[oldTarget];
                    if (adjustedTarget == -1) {
                        // Target was a NOP, find next non-NOP
                        for (sz t = oldTarget + 1; t < codeSize; ++t) {
                            if (newPcFromOld[t] != -1) {
                                adjustedTarget = newPcFromOld[t];
                                break;
                            }
                        }
                        if (adjustedTarget == -1) adjustedTarget = static_cast<isz>(idx);
                    }
                    isz curNewPc = newPcFromOld[i];
                    i16 newOff = static_cast<i16>(adjustedTarget - curNewPc);
                    code[i] = SetSimm12(raw, newOff);
                }
            }
        }

        // Now remove NOPs
        sz writePos = 0;
        for (sz i = 0; i < codeSize; ++i) {
            if (!IsNop(code[i])) {
                code[writePos++] = code[i];
            }
        }
        sz removed = codeSize - writePos;
        code.resize(writePos);

        if (pc >= writePos) pc = writePos > 0 ? writePos - 1 : 0;

        ChangedCount_ += removed;
        return removed > 0;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    sz ChangedCount_ = 0;
};

// ============================================================================
// LoopInvariantHoister — moves loop-invariant instructions out of loops
// ============================================================================

class LoopInvariantHoister : public OptimizationPass {
public:
    const char* Name() const override { return "LoopInvariantHoister"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        sz codeSize = code.size();
        (void)pc;

        // Detect loops: scan for JMP backwards (negative offset)
        // That's the loop backedge. Loop header = target of backedge.
        struct Loop {
            sz header;
            sz latch;  // PC of backedge JMP
            sz end;    // exclusive end of loop body
        };
        std::vector<Loop> loops;

        for (sz i = 0; i < codeSize; ++i) {
            Opcode op = ExtractOp(code[i]);
            if (op == Opcode::JMP) {
                i16 off = ExtractSimm12(code[i]);
                if (off < 0) {
                    sz target = static_cast<sz>(static_cast<isz>(i) + off);
                    if (target < i) {
                        // backwards jump found; header = target, latch = i
                        Loop l;
                        l.header = target;
                        l.latch = i;
                        l.end = i + 1;
                        loops.push_back(l);
                    }
                }
            }
        }

        for (const auto& loop : loops) {
            // Identify registers that are invariant in the loop
            // A register is invariant if:
            //   - It is only written by MOV or MOVR inside the loop (constant or copied from outside)
            //   - Or it's written by arithmetic using only invariant operands

            std::unordered_set<u8> writtenInLoop;
            std::unordered_set<u8> invariant;
            std::vector<sz> invariantPcs;

            // First pass: collect all written registers
            for (sz i = loop.header; i < loop.end; ++i) {
                u32 raw = code[i];
                if (IsNop(raw)) continue;
                Opcode op = ExtractOp(raw);
                u8 rd = ExtractRd(raw);
                bool writes = (op != Opcode::CMP && op != Opcode::CMPF &&
                               op != Opcode::CMPU && op != Opcode::TST && op != Opcode::TSTF &&
                               op != Opcode::JMP && op != Opcode::JZ && op != Opcode::JNZ &&
                               op != Opcode::JS && op != Opcode::JNS && op != Opcode::JO &&
                               op != Opcode::JNO && op != Opcode::JC && op != Opcode::JNC &&
                               op != Opcode::JL && op != Opcode::JLE && op != Opcode::JG &&
                               op != Opcode::JGE && op != Opcode::CALL && op != Opcode::RET &&
                               op != Opcode::TABLE_JMP && op != Opcode::HALT && op != Opcode::TRAP &&
                               op != Opcode::BREAK && op != Opcode::YIELD && op != Opcode::BARRIER &&
                               op != Opcode::PREFETCH && op != Opcode::FLUSH_CACHE &&
                               op != Opcode::SYNC && op != Opcode::MEMFENCE);
                if (writes) writtenInLoop.insert(rd);
            }

            // Check each instruction in the loop for invariants
            // An instruction is invariant if all its source operands are invariant
            std::unordered_set<u8> definedOutside;
            for (u8 r = 0; r < 16; ++r)
                if (writtenInLoop.find(r) == writtenInLoop.end())
                    definedOutside.insert(r);

            bool anyInvariant = true;
            std::unordered_set<u8> currentInvariant = definedOutside;
            std::vector<sz> invariantInsts;

            while (anyInvariant) {
                anyInvariant = false;
                for (sz i = loop.header; i < loop.end; ++i) {
                    u32 raw = code[i];
                    if (IsNop(raw)) continue;
                    Opcode op = ExtractOp(raw);
                    u8 rd = ExtractRd(raw);
                    u8 ra = ExtractRa(raw);
                    u8 rb = ExtractRb(raw);

                    if (currentInvariant.find(rd) != currentInvariant.end()) continue;

                    bool allInvariant = true;

                    switch (op) {
                    case Opcode::MOV:
                        allInvariant = true;
                        break;
                    case Opcode::MOVR:
                    case Opcode::VSPLAT: {
                        if (currentInvariant.find(ra) == currentInvariant.end())
                            allInvariant = false;
                        break;
                    }
                    case Opcode::NEG: case Opcode::ABS: case Opcode::NEGF: case Opcode::ABSF:
                    case Opcode::NOT: case Opcode::POPCNT: case Opcode::CLZ: case Opcode::CTZ:
                    case Opcode::BSWAP:
                    case Opcode::BITCAST: case Opcode::REINTERPRET:
                    case Opcode::TRUNC: case Opcode::ROUND: case Opcode::CEIL: case Opcode::FLOOR: {
                        if (currentInvariant.find(ra) == currentInvariant.end())
                            allInvariant = false;
                        break;
                    }
                    case Opcode::ADDI: case Opcode::SUBI: case Opcode::MULI:
                    case Opcode::ANDI: case Opcode::ORI: case Opcode::XORI:
                    case Opcode::SHLI: case Opcode::SHRI: case Opcode::SAR_I:
                    case Opcode::LEA: {
                        if (currentInvariant.find(ra) == currentInvariant.end())
                            allInvariant = false;
                        break;
                    }
                    case Opcode::ADD: case Opcode::SUB: case Opcode::MUL:
                    case Opcode::DIV: case Opcode::MOD:
                    case Opcode::AND: case Opcode::OR: case Opcode::XOR:
                    case Opcode::SHL: case Opcode::SHR: case Opcode::SAR:
                    case Opcode::ROL: case Opcode::ROR:
                    case Opcode::ADDF: case Opcode::SUBF: case Opcode::MULF: case Opcode::DIVF:
                    case Opcode::BEXTR: case Opcode::BZHI: case Opcode::PDEP: {
                        if (currentInvariant.find(ra) == currentInvariant.end() ||
                            currentInvariant.find(rb) == currentInvariant.end())
                            allInvariant = false;
                        break;
                    }
                    default:
                        allInvariant = false;
                        break;
                    }

                    if (allInvariant && !IsBranchOrRet(op) && !Terminates(op)) {
                        currentInvariant.insert(rd);
                        invariantInsts.push_back(i);
                        anyInvariant = true;
                    }
                }
            }

            // Hoist invariant instructions
            if (!invariantInsts.empty() && loop.header > 0) {
                sz insertPos = loop.header;
                for (isz ii = static_cast<isz>(invariantInsts.size()) - 1; ii >= 0; --ii) {
                    sz instPos = invariantInsts[static_cast<sz>(ii)];
                    u32 inst = code[instPos];
                    code.insert(code.begin() + static_cast<isz>(insertPos), inst);
                    code[instPos + 1] = MakeNop();
                    ChangedCount_++;
                    changed = true;
                }
            }
        }

        return changed;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    sz ChangedCount_ = 0;
};

// ============================================================================
// PeepholeOptimizer — instruction pattern replacement
// ============================================================================

class PeepholeOptimizer : public OptimizationPass {
public:
    const char* Name() const override { return "PeepholeOptimizer"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        sz codeSize = code.size();
        (void)pc;

        for (isz i = 0; i + 1 < static_cast<isz>(codeSize); ++i) {
            sz i0 = static_cast<sz>(i);
            sz i1 = static_cast<sz>(i + 1);

            if (IsNop(code[i0]) || IsNop(code[i1])) continue;

            Opcode op0 = ExtractOp(code[i0]);
            Opcode op1 = ExtractOp(code[i1]);
            u8 rd0 = ExtractRd(code[i0]);
            u8 ra0 = ExtractRa(code[i0]);
            u8 rb0 = ExtractRb(code[i0]);
            u8 rd1 = ExtractRd(code[i1]);
            u8 ra1 = ExtractRa(code[i1]);
            u8 rb1 = ExtractRb(code[i1]);
            i16 imm0 = ExtractSimm12(code[i0]);

            // Pattern 1: MOV R0, 0; ADD R0, R0, Rx -> MOVR R0, Rx
            if (op0 == Opcode::MOV && imm0 == 0 &&
                op1 == Opcode::ADD && rd0 == rd1 && rd1 == ra1) {
                code[i1] = Instruction::Encode(Opcode::MOVR, rd1, rb1, 0, 0);
                code[i0] = MakeNop();
                changed = true;
                ChangedCount_++;
                continue;
            }

            // Pattern 1b: MOV R0, 0; ADDF R0, R0, Rx -> MOVR R0, Rx
            if (op0 == Opcode::MOV && imm0 == 0 &&
                op1 == Opcode::ADDF && rd0 == rd1 && rd1 == ra1) {
                code[i1] = Instruction::Encode(Opcode::MOVR, rd1, rb1, 0, 0);
                code[i0] = MakeNop();
                changed = true;
                ChangedCount_++;
                continue;
            }

            // Pattern 2: CMP Ra, Rb; JZ target; JMP next -> JNZ target (invert, remove JMP)
            if (i + 2 < static_cast<isz>(codeSize)) {
                sz i2 = static_cast<sz>(i + 2);
                if (!IsNop(code[i2])) {
                    Opcode op2 = ExtractOp(code[i2]);
                    if (op0 == Opcode::CMP && op1 == Opcode::JZ && op2 == Opcode::JMP) {
                        i16 jzOff = ExtractSimm12(code[i1]);
                        i16 jmpOff = ExtractSimm12(code[i2]);
                        sz jzTarget = static_cast<sz>(static_cast<isz>(i1) + jzOff);

                        // JMP should target the instruction right after itself
                        if (jmpOff == 1) {
                            // Replace JZ with JNZ to jzTarget, remove JMP
                            i16 newOff = static_cast<i16>(static_cast<isz>(jzTarget) - static_cast<isz>(i1));
                            code[i1] = Instruction::Encode(Opcode::JNZ, static_cast<u16>(newOff));
                            code[i2] = MakeNop();
                            changed = true;
                            ChangedCount_++;
                            continue;
                        }
                    }
                }
            }

            // Pattern 3: ADD R0, R0, 0 -> NOP (no-op arithmetic)
            if (op0 == Opcode::ADD && ra0 == rd0 && rb0 == 0) {
                // Check if imm is zero... actually ADD uses ra,rb registers not immediate.
                // ADD R0, R0, 0 means rb=0 register, not immediate.
                // If R0(register 0) has zero, that's not a compile-time noop.
                // But ADDI R0, R0, 0 IS provably a no-op.
                // The spec says "ADD R0, R0, 0 -> NOP" but the actual encoding has no way
                // to have an immediate 0 except through ADDI. Let me check: ADD has Rd=rd, Ra=ra, Rb=rb.
                // Rb can be a register holding zero. We can't know at compile time.
                // So we'll handle ADDI variant.
            }
            if (op0 == Opcode::ADDI && ra0 == rd0 && imm0 == 0) {
                code[i0] = MakeNop();
                changed = true;
                ChangedCount_++;
                continue;
            }
            if (op0 == Opcode::SUBI && ra0 == rd0 && imm0 == 0) {
                code[i0] = MakeNop();
                changed = true;
                ChangedCount_++;
                continue;
            }

            // Pattern 4: MOVR Rd, Ra; MOVR Re, Rd -> MOVR Re, Ra; MOVR Rd, Ra
            if (op0 == Opcode::MOVR && op1 == Opcode::MOVR &&
                rd1 != rd0 && ra1 == rd0) {
                code[i1] = Instruction::Encode(Opcode::MOVR, rd1, ra0, 0, 0);
                changed = true;
                ChangedCount_++;
                continue;
            }
        }

        return changed;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    sz ChangedCount_ = 0;
};

// ============================================================================
// Optimizer — manages all passes and runs them to a fixed point
// ============================================================================

class Optimizer {
public:
    void AddPass(std::unique_ptr<OptimizationPass> pass) {
        Passes_.push_back(std::move(pass));
    }

    bool Optimize(std::vector<u32>& code) {
        bool anyChange = false;
        sz iteration = 0;
        constexpr sz kMaxIterations = 32;
        sz pc = 0;

        bool changed = true;
        while (changed && iteration < kMaxIterations) {
            changed = false;
            ++iteration;

            for (auto& pass : Passes_) {
                auto t0 = Clock::now();
                bool passChanged = pass->Run(code, pc);
                auto t1 = Clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

                if (passChanged) {
                    changed = true;
                    anyChange = true;
                    Stats_[pass->Name()].Changes++;
                    Stats_[pass->Name()].RemovedCount += GetRemovedCount(pass.get());
                }
                Stats_[pass->Name()].TotalTime += elapsed.count();
                Stats_[pass->Name()].Passes++;
            }
        }

        return anyChange;
    }

    void DumpStats(std::ostream& os) const {
        os << "=== Optimizer Statistics ===\n";
        for (const auto& pass : Passes_) {
            auto it = Stats_.find(pass->Name());
            if (it != Stats_.end()) {
                os << "  " << pass->Name() << ":\n"
                   << "    Passes:    " << it->second.Passes << "\n"
                   << "    Changes:   " << it->second.Changes << "\n"
                   << "    Removed:   " << it->second.RemovedCount << "\n"
                   << "    Time (us): " << it->second.TotalTime << "\n";
            }
        }
    }

private:
    struct PassStats {
        sz Passes   = 0;
        sz Changes  = 0;
        sz RemovedCount = 0;
        i64 TotalTime = 0;
    };

    std::vector<std::unique_ptr<OptimizationPass>> Passes_;
    mutable std::unordered_map<std::string, PassStats> Stats_;

    static sz GetRemovedCount(const OptimizationPass* pass) {
        if (auto* cf = dynamic_cast<const ConstantFolder*>(pass))
            return cf->ChangedCount();
        if (auto* dc = dynamic_cast<const DeadCodeEliminator*>(pass))
            return dc->ChangedCount();
        if (auto* ns = dynamic_cast<const NopSqueezer*>(pass))
            return ns->ChangedCount();
        if (auto* li = dynamic_cast<const LoopInvariantHoister*>(pass))
            return li->ChangedCount();
        if (auto* ph = dynamic_cast<const PeepholeOptimizer*>(pass))
            return ph->ChangedCount();
        return 0;
    }
};

} // namespace opt
} // namespace voxel
