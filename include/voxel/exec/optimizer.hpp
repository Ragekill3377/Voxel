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
#include <sstream>
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
                if (simm12 == 0) {
                    if (rd == ra) {
                        code[i] = MakeNop();
                        changed = true;
                        ChangedCount_++;
                    }
                    auto it = constants.find(ra);
                    if (it != constants.end()) constants[rd] = it->second;
                    break;
                }
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
                if (simm12 == 0) {
                    if (rd == ra) {
                        code[i] = MakeNop();
                        changed = true;
                        ChangedCount_++;
                    }
                    auto it = constants.find(ra);
                    if (it != constants.end()) constants[rd] = it->second;
                    break;
                }
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
                if (simm12 == 0) {
                    code[i] = EncodeMov(rd, 0);
                    constants[rd] = 0;
                    changed = true;
                    ChangedCount_++;
                    break;
                }
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
                if (simm12 == 0) {
                    code[i] = EncodeMov(rd, 0);
                    constants[rd] = 0;
                    changed = true;
                    ChangedCount_++;
                    break;
                }
                auto it = constants.find(ra);
                if (it != constants.end()) {
                    constants[rd] = it->second & static_cast<i64>(simm12);
                }
                break;
            }
            case Opcode::ORI: {
                if (simm12 == -1) {
                    code[i] = EncodeMov(rd, -1);
                    constants[rd] = -1;
                    changed = true;
                    ChangedCount_++;
                    break;
                }
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
                if (ra == rb) {
                    code[i] = EncodeMov(rd, 0);
                    constants[rd] = 0;
                    changed = true;
                    ChangedCount_++;
                    break;
                }
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
// FilterMerge — combines adjacent filter operations on the same column
// ============================================================================

class FilterMerge : public OptimizationPass {
public:
    const char* Name() const override { return "FilterMerge"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        (void)pc;
        sz codeSize = code.size();

        for (sz i = 0; i + 1 < codeSize; ++i) {
            u32 raw0 = code[i];
            u32 raw1 = code[i + 1];
            if (IsNop(raw0) || IsNop(raw1)) continue;

            Opcode op0 = ExtractOp(raw0);
            Opcode op1 = ExtractOp(raw1);

            u8 v0 = static_cast<u8>(op0);
            u8 v1 = static_cast<u8>(op1);
            if (v0 < 0xC0 || v0 > 0xC6 || v1 < 0xC0 || v1 > 0xC6) continue;

            u8 seg0 = ExtractRa(raw0);
            u8 seg1 = ExtractRa(raw1);
            if (seg0 != seg1) continue;

            u8 thresh0 = ExtractRb(raw0);
            u8 thresh1 = ExtractRb(raw1);

            if (op0 == op1) {
                switch (op0) {
                case Opcode::VFILTER_GT:
                case Opcode::VFILTER_GE:
                    if (thresh1 >= thresh0) {
                        code[i] = MakeNop();
                    } else {
                        code[i + 1] = MakeNop();
                    }
                    changed = true;
                    ChangedCount_++;
                    break;
                case Opcode::VFILTER_LT:
                case Opcode::VFILTER_LE:
                    if (thresh1 <= thresh0) {
                        code[i] = MakeNop();
                    } else {
                        code[i + 1] = MakeNop();
                    }
                    changed = true;
                    ChangedCount_++;
                    break;
                case Opcode::VFILTER_EQ:
                case Opcode::VFILTER_NE:
                    if (thresh0 == thresh1) {
                        code[i] = MakeNop();
                        changed = true;
                        ChangedCount_++;
                    }
                    break;
                default:
                    break;
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
// PredicatePushdown — moves filters before sorts, aggs, and joins
// ============================================================================

class PredicatePushdown : public OptimizationPass {
public:
    const char* Name() const override { return "PredicatePushdown"; }

    bool Run(std::vector<u32>& code, sz& pc) override {
        bool changed = false;
        (void)pc;
        sz codeSize = code.size();

        for (sz i = 0; i < codeSize; ++i) {
            u32 raw = code[i];
            if (IsNop(raw)) continue;
            Opcode op = ExtractOp(raw);

            u8 v = static_cast<u8>(op);
            if (v < 0xC0 || v > 0xC6) continue;

            u8 filterSeg = ExtractRa(raw);
            u8 filterRd = ExtractRd(raw);
            u8 filterRb = ExtractRb(raw);

            for (isz j = static_cast<isz>(i) - 1; j >= 0; --j) {
                sz prev = static_cast<sz>(j);
                u32 prevRaw = code[prev];
                if (IsNop(prevRaw)) continue;
                Opcode prevOp = ExtractOp(prevRaw);

                if (IsBranchOrRet(prevOp) || Terminates(prevOp)) break;

                if (!HasMatchingSegment(prevOp, prevRaw, filterSeg)) continue;

                u8 expRa = ExtractRa(prevRaw);
                u8 expRb = ExtractRb(prevRaw);

                if (expRa == filterRb || expRb == filterRb) continue;
                if (expRa == filterRd || expRb == filterRd) continue;

                bool hasDependency = false;
                for (sz k = prev + 1; k < i; ++k) {
                    u32 kRaw = code[k];
                    if (IsNop(kRaw)) continue;
                    Opcode kOp = ExtractOp(kRaw);
                    u8 kRd = ExtractRd(kRaw);
                    u8 kRa = ExtractRa(kRaw);
                    u8 kRb = ExtractRb(kRaw);

                    if (kRd != 0 && kRd == filterRb && kOp != Opcode::CMP
                        && kOp != Opcode::CMPF && kOp != Opcode::CMPU
                        && kOp != Opcode::TST && kOp != Opcode::TSTF
                        && !IsBranchOrRet(kOp) && !Terminates(kOp)) {
                        hasDependency = true;
                        break;
                    }
                    if (kRa == filterRd || kRb == filterRd) {
                        hasDependency = true;
                        break;
                    }
                }

                if (hasDependency) continue;

                u32 filterInst = code[i];
                for (sz k = i; k > prev; --k) {
                    code[k] = code[k - 1];
                }
                code[prev] = filterInst;
                changed = true;
                ChangedCount_++;
                break;
            }
        }

        return changed;
    }

    sz ChangedCount() const { return ChangedCount_; }

private:
    sz ChangedCount_ = 0;

    static bool HasMatchingSegment(Opcode op, u32 raw, u8 filterSeg) {
        switch (op) {
        case Opcode::SORT_ASC:
        case Opcode::SORT_DESC:
        case Opcode::SORT_TOPK:
        case Opcode::SORT_BOTTOMK:
            return ExtractRd(raw) == filterSeg;
        case Opcode::AGG_COUNT:
        case Opcode::AGG_SUM:
        case Opcode::AGG_AVG:
        case Opcode::AGG_MIN:
        case Opcode::AGG_MAX:
        case Opcode::AGG_FIRST:
        case Opcode::AGG_LAST:
        case Opcode::AGG_STDDEV:
        case Opcode::AGG_VARIANCE:
        case Opcode::AGG_COUNT_DISTINCT:
        case Opcode::AGG_SUM_DISTINCT:
        case Opcode::AGG_MEDIAN:
        case Opcode::AGG_MODE:
        case Opcode::AGG_PERCENTILE:
            return ExtractRa(raw) == filterSeg;
        case Opcode::JOIN_HASH:
        case Opcode::JOIN_MERGE:
        case Opcode::JOIN_NESTED:
        case Opcode::JOIN_ANTI:
        case Opcode::JOIN_SEMI:
            return ExtractRd(raw) == filterSeg || ExtractRa(raw) == filterSeg;
        default:
            return false;
        }
    }
};

// ============================================================================
// Optimizer — manages all passes and runs them to a fixed point
// ============================================================================

class Optimizer {
public:
    Optimizer() {
        AddPass(std::make_unique<ConstantFolder>());
        AddPass(std::make_unique<FilterMerge>());
        AddPass(std::make_unique<PredicatePushdown>());
        AddPass(std::make_unique<DeadCodeEliminator>());
        AddPass(std::make_unique<LoopInvariantHoister>());
        AddPass(std::make_unique<PeepholeOptimizer>());
        AddPass(std::make_unique<NopSqueezer>());
    }

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
        if (auto* fm = dynamic_cast<const FilterMerge*>(pass))
            return fm->ChangedCount();
        if (auto* pp = dynamic_cast<const PredicatePushdown*>(pass))
            return pp->ChangedCount();
        return 0;
    }
};

// ============================================================================
// ExplainPlan — prints execution plan with human-readable annotations
// ============================================================================

class ExplainPlan {
public:
    static void Print(std::ostream& os, const std::vector<u32>& bytecode) {
        for (sz i = 0; i < bytecode.size(); ++i) {
            os << i << ": ";
            PrintOne(os, bytecode[i], i, static_cast<isz>(bytecode.size()));
            os << "\n";
        }
    }

    static void PrintTree(std::ostream& os, const std::vector<u32>& bytecode) {
        auto root = BuildTree(bytecode.data(), bytecode.size());
        PrintTreeRecursive(os, root, "", true);
    }

    struct OperatorNode {
        std::string opName;
        sz bytecodeOffset;
        std::vector<OperatorNode> children;
    };

    static OperatorNode BuildTree(const u32* code, sz count) {
        OperatorNode root;
        root.opName = "Program";
        root.bytecodeOffset = 0;

        struct LoopInfo {
            sz header;
            sz latch;
            sz end;
        };
        std::vector<LoopInfo> loops;
        std::vector<sz> leaders;
        if (count > 0) leaders.push_back(0);

        for (sz i = 0; i < count; ++i) {
            u32 raw = code[i];
            Opcode op = ExtractOp(raw);
            i16 off = ExtractSimm12(raw);

            if (i + 1 < count) leaders.push_back(i + 1);

            if (IsBranchOrRet(op) && op != Opcode::RET) {
                sz target = static_cast<sz>(static_cast<isz>(i) + off);
                if (target < count) leaders.push_back(target);
                if (off < 0 && target < i &&
                    (op == Opcode::JMP || IsConditionalBranch(op))) {
                    LoopInfo l;
                    l.header = target;
                    l.latch = i;
                    l.end = i + 1;
                    loops.push_back(l);
                }
            }

            if (Terminates(op)) {
                if (i + 1 < count) leaders.push_back(i + 1);
            }
        }

        std::sort(leaders.begin(), leaders.end());
        leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());

        // Merge adjacent leadership points to form basic blocks
        std::vector<std::pair<sz, sz>> blocks;
        for (sz li = 0; li < leaders.size(); ++li) {
            sz start = leaders[li];
            if (start >= count) continue;
            sz end = count;
            for (sz lj = li + 1; lj < leaders.size(); ++lj) {
                if (leaders[lj] > start) { end = leaders[lj]; break; }
            }
            blocks.push_back({start, end});
        }

        // Determine loop body range for each header
        std::vector<sz> processedBlock(count, 0);

        for (sz bi = 0; bi < blocks.size(); ++bi) {
            sz start = blocks[bi].first;
            sz end = blocks[bi].second;

            if (processedBlock[start]) continue;

            bool isLoopHeader = false;
            sz loopEnd = end;
            for (const auto& l : loops) {
                if (start == l.header) {
                    isLoopHeader = true;
                    loopEnd = l.end;
                    break;
                }
            }

            if (isLoopHeader) {
                OperatorNode loopNode;
                loopNode.opName = "Loop (pc " + std::to_string(start) + "-" + std::to_string(loopEnd - 1) + ")";
                loopNode.bytecodeOffset = start;

                for (sz pc = start; pc < loopEnd; ++pc) {
                    Opcode op = ExtractOp(code[pc]);
                    if (op == Opcode::NOP) continue;
                    OperatorNode in;
                    in.opName = FormatInstruction(code[pc]);
                    in.bytecodeOffset = pc;
                    loopNode.children.push_back(in);
                }
                root.children.push_back(loopNode);

                for (sz pc = start; pc < loopEnd; ++pc)
                    processedBlock[pc] = 1;
            }
        }

        for (sz bi = 0; bi < blocks.size(); ++bi) {
            sz start = blocks[bi].first;
            sz end = blocks[bi].second;

            if (processedBlock[start]) continue;

            bool inLoop = false;
            for (const auto& l : loops) {
                if (start >= l.header && start < l.end) { inLoop = true; break; }
            }
            if (inLoop) continue;

            for (sz pc = start; pc < end; ++pc) {
                Opcode op = ExtractOp(code[pc]);
                if (op == Opcode::NOP) continue;
                OperatorNode in;
                in.opName = FormatInstruction(code[pc]);
                in.bytecodeOffset = pc;
                root.children.push_back(in);
            }
        }

        return root;
    }

private:
    static void PrintTreeRecursive(std::ostream& os, const OperatorNode& node,
                                    const std::string& prefix, bool isLast) {
        os << node.opName << "\n";
        for (sz i = 0; i < node.children.size(); ++i) {
            bool last = (i == node.children.size() - 1);
            os << prefix << (isLast ? "    " : "|   ") << (last ? "`-- " : "|-- ");
            PrintTreeRecursive(os, node.children[i],
                prefix + (isLast ? "    " : "|   "), last);
        }
    }

    static void PrintOne(std::ostream& os, u32 raw, sz pc, isz total) {
        Opcode op = ExtractOp(raw);
        u8 rd = ExtractRd(raw);
        u8 ra = ExtractRa(raw);
        u8 rb = ExtractRb(raw);
        i16 simm12 = ExtractSimm12(raw);
        u16 imm12 = static_cast<u16>(raw >> 20) & 0xFFF;
        u8 count8 = static_cast<u8>(imm12 & 0xFF);
        u8 segId = static_cast<u8>((imm12 >> 8) & 0xF);

        (void)total;

        std::string comment;

        switch (op) {
        case Opcode::VLOAD: {
            os << "VLOAD V" << static_cast<int>(rd)
               << ", segment[" << static_cast<int>(segId)
               << "], offset=R" << static_cast<int>(ra);
            comment = "load " + std::to_string(count8 > 0 ? static_cast<int>(count8) : 4) + " doubles from segment " + std::to_string(static_cast<int>(segId));
            break;
        }
        case Opcode::VSTORE: {
            os << "VSTORE V" << static_cast<int>(rd)
               << ", segment[" << static_cast<int>(segId)
               << "], offset=R" << static_cast<int>(ra);
            comment = "store vector to segment " + std::to_string(static_cast<int>(segId));
            break;
        }
        case Opcode::VLOAD_STRIDED: {
            os << "VLOAD_STRIDED V" << static_cast<int>(rd)
               << ", offset=R" << static_cast<int>(ra)
               << ", stride=" << static_cast<int>(segId);
            comment = "strided load " + std::to_string(count8) + " elements";
            break;
        }
        case Opcode::VGATHER: {
            os << "VGATHER V" << static_cast<int>(rd)
               << ", index=R" << static_cast<int>(ra)
               << ", seg=" << static_cast<int>(segId);
            comment = "gather from segment";
            break;
        }
        case Opcode::VSCATTER: {
            os << "VSCATTER V" << static_cast<int>(rd)
               << ", index=R" << static_cast<int>(ra)
               << ", seg=" << static_cast<int>(segId);
            comment = "scatter to segment";
            break;
        }
        case Opcode::VSPLAT: {
            os << "VSPLAT V" << static_cast<int>(rd) << ", R" << static_cast<int>(ra);
            comment = "broadcast scalar to all vector lanes";
            break;
        }
        case Opcode::VEXTRACT: {
            os << "VEXTRACT R" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", lane=" << static_cast<int>(rb);
            comment = "extract lane from vector";
            break;
        }
        case Opcode::VINSERT: {
            os << "VINSERT V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb) << ", lane=" << static_cast<int>(imm12 & 0xF);
            comment = "insert scalar into vector lane";
            break;
        }
        case Opcode::VFILTER: {
            os << "VFILTER V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter by mode " + std::to_string(imm12 & 0x7);
            break;
        }
        case Opcode::VFILTER_GT: {
            os << "VFILTER_GT V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values > threshold";
            break;
        }
        case Opcode::VFILTER_GE: {
            os << "VFILTER_GE V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values >= threshold";
            break;
        }
        case Opcode::VFILTER_LT: {
            os << "VFILTER_LT V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values < threshold";
            break;
        }
        case Opcode::VFILTER_LE: {
            os << "VFILTER_LE V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values <= threshold";
            break;
        }
        case Opcode::VFILTER_EQ: {
            os << "VFILTER_EQ V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values == threshold";
            break;
        }
        case Opcode::VFILTER_NE: {
            os << "VFILTER_NE V" << static_cast<int>(rd)
               << ", V" << static_cast<int>(ra)
               << ", R" << static_cast<int>(rb);
            comment = "filter values != threshold";
            break;
        }
        case Opcode::VSUM: {
            os << "VSUM R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "horizontal sum of vector into scalar";
            break;
        }
        case Opcode::VPROD: {
            os << "VPROD R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "horizontal product of vector into scalar";
            break;
        }
        case Opcode::VMEAN: {
            os << "VMEAN R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "mean of vector elements";
            break;
        }
        case Opcode::VRED_MIN: {
            os << "VRED_MIN R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "find minimum value in vector";
            break;
        }
        case Opcode::VRED_MAX: {
            os << "VRED_MAX R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "find maximum value in vector";
            break;
        }
        case Opcode::VCOUNT: {
            os << "VCOUNT R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            comment = "count non-null elements in vector";
            break;
        }
        case Opcode::VADD: {
            os << "VADD V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra) << ", V" << static_cast<int>(rb);
            comment = "element-wise vector add";
            break;
        }
        case Opcode::VSUB: {
            os << "VSUB V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra) << ", V" << static_cast<int>(rb);
            comment = "element-wise vector sub";
            break;
        }
        case Opcode::VMUL: {
            os << "VMUL V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra) << ", V" << static_cast<int>(rb);
            comment = "element-wise vector mul";
            break;
        }
        case Opcode::MOV: {
            os << "MOV R" << static_cast<int>(rd) << ", " << static_cast<int>(simm12);
            comment = "move immediate";
            break;
        }
        case Opcode::MOVR: {
            os << "MOVR R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra);
            comment = "copy register";
            break;
        }
        case Opcode::ADDI: {
            os << "ADDI R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", " << static_cast<int>(simm12);
            comment = "add immediate";
            break;
        }
        case Opcode::SUBI: {
            os << "SUBI R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", " << static_cast<int>(simm12);
            comment = "subtract immediate";
            break;
        }
        case Opcode::MULI: {
            os << "MULI R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", " << static_cast<int>(simm12);
            comment = "multiply immediate";
            break;
        }
        case Opcode::ADD: {
            os << "ADD R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "integer add";
            break;
        }
        case Opcode::SUB: {
            os << "SUB R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "integer sub";
            break;
        }
        case Opcode::MUL: {
            os << "MUL R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "integer mul";
            break;
        }
        case Opcode::DIV: {
            os << "DIV R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "integer div";
            break;
        }
        case Opcode::MOD: {
            os << "MOD R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "integer mod";
            break;
        }
        case Opcode::ADDF: {
            os << "ADDF R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "floating-point add";
            break;
        }
        case Opcode::SUBF: {
            os << "SUBF R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "floating-point sub";
            break;
        }
        case Opcode::MULF: {
            os << "MULF R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "floating-point mul";
            break;
        }
        case Opcode::DIVF: {
            os << "DIVF R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "floating-point div";
            break;
        }
        case Opcode::CMP: {
            os << "CMP R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "compare scalar integers (set flags)";
            break;
        }
        case Opcode::CMPF: {
            os << "CMPF R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "compare scalar floats (set flags)";
            break;
        }
        case Opcode::CMPU: {
            os << "CMPU R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            comment = "compare unsigned scalars (set flags)";
            break;
        }
        case Opcode::JMP: {
            sz target = static_cast<sz>(static_cast<isz>(pc) + simm12);
            os << "JMP " << static_cast<int>(simm12);
            comment = "unconditional jump to pc=" + std::to_string(target);
            break;
        }
        case Opcode::JZ: {
            sz target = static_cast<sz>(static_cast<isz>(pc) + simm12);
            os << "JZ " << static_cast<int>(simm12);
            comment = "jump if zero to pc=" + std::to_string(target);
            break;
        }
        case Opcode::JNZ: {
            sz target = static_cast<sz>(static_cast<isz>(pc) + simm12);
            os << "JNZ " << static_cast<int>(simm12);
            if (simm12 < 0)
                comment = "loop back to instruction " + std::to_string(target);
            else
                comment = "jump if not zero to pc=" + std::to_string(target);
            break;
        }
        case Opcode::HALT: {
            os << "HALT";
            comment = "stop execution";
            break;
        }
        case Opcode::RET: {
            os << "RET";
            comment = "return from procedure";
            break;
        }
        case Opcode::TRAP: {
            os << "TRAP";
            comment = "trap to debugger";
            break;
        }
        case Opcode::BREAK: {
            os << "BREAK";
            comment = "breakpoint";
            break;
        }
        case Opcode::YIELD: {
            os << "YIELD";
            comment = "yield execution";
            break;
        }
        case Opcode::NOP: {
            os << "NOP";
            comment = "no operation";
            break;
        }
        case Opcode::CALL: {
            sz target = static_cast<sz>(static_cast<isz>(pc) + simm12);
            os << "CALL pc+" << static_cast<int>(simm12);
            comment = "call procedure at pc=" + std::to_string(target);
            break;
        }
        case Opcode::MOVZ: {
            os << "MOVZ R" << static_cast<int>(rd) << ", " << static_cast<int>(simm12);
            comment = "conditional move if zero";
            break;
        }
        case Opcode::MOVN: {
            os << "MOVN R" << static_cast<int>(rd) << ", " << static_cast<int>(simm12);
            comment = "conditional move if not zero";
            break;
        }
        case Opcode::MOVK: {
            os << "MOVK R" << static_cast<int>(rd) << ", " << static_cast<int>(simm12) << " (shift=" << static_cast<int>(rd) << ")";
            comment = "move keeping 16-bit immediate";
            break;
        }
        case Opcode::LEA: {
            os << "LEA R" << static_cast<int>(rd) << ", [R" << static_cast<int>(ra) << " + " << static_cast<int>(simm12) << "]";
            comment = "load effective address";
            break;
        }
        case Opcode::SORT_ASC: {
            os << "SORT_ASC seg=" << static_cast<int>(rd) << ", col=R" << static_cast<int>(ra);
            comment = "sort segment ascending";
            break;
        }
        case Opcode::SORT_DESC: {
            os << "SORT_DESC seg=" << static_cast<int>(rd) << ", col=R" << static_cast<int>(ra);
            comment = "sort segment descending";
            break;
        }
        case Opcode::SORT_TOPK: {
            os << "SORT_TOPK seg=" << static_cast<int>(rd) << ", k=" << static_cast<int>(rb) << ", col=R" << static_cast<int>(ra);
            comment = "select top K elements";
            break;
        }
        case Opcode::SORT_BOTTOMK: {
            os << "SORT_BOTTOMK seg=" << static_cast<int>(rd) << ", k=" << static_cast<int>(rb) << ", col=R" << static_cast<int>(ra);
            comment = "select bottom K elements";
            break;
        }
        case Opcode::AGG_COUNT: case Opcode::AGG_SUM: case Opcode::AGG_AVG:
        case Opcode::AGG_MIN: case Opcode::AGG_MAX: {
            os << OpcodeName(op) << " R" << static_cast<int>(rd) << ", seg=" << static_cast<int>(ra);
            comment = "aggregate on segment";
            break;
        }
        default: {
            os << OpcodeName(op);
            if (rd != 0 || ra != 0 || rb != 0) {
                os << " R" << static_cast<int>(rd)
                   << ", R" << static_cast<int>(ra)
                   << ", R" << static_cast<int>(rb);
            }
            comment = "";
            break;
        }
        }

        if (!comment.empty()) {
            os << " → " << comment;
        }
    }

    static std::string FormatInstruction(u32 raw) {
        std::ostringstream oss;
        Opcode op = ExtractOp(raw);
        u8 rd = ExtractRd(raw);
        u8 ra = ExtractRa(raw);
        u8 rb = ExtractRb(raw);
        i16 simm12 = ExtractSimm12(raw);
        u16 imm12 = static_cast<u16>(raw >> 20) & 0xFFF;
        u8 count8 = static_cast<u8>(imm12 & 0xFF);
        u8 segId = static_cast<u8>((imm12 >> 8) & 0xF);

        switch (op) {
        case Opcode::VLOAD:
            oss << "VLOAD V" << static_cast<int>(rd) << " seg[" << static_cast<int>(segId)
                << "] off=R" << static_cast<int>(ra) << " cnt=" << static_cast<int>(count8);
            break;
        case Opcode::VFILTER_GT:
            oss << "VFILTER_GT V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::VFILTER_GE:
            oss << "VFILTER_GE V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::VFILTER_LT:
            oss << "VFILTER_LT V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::VFILTER_LE:
            oss << "VFILTER_LE V" << static_cast<int>(rd) << ", V" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::VSUM:
            oss << "VSUM R" << static_cast<int>(rd) << ", V" << static_cast<int>(ra);
            break;
        case Opcode::ADDF:
            oss << "ADDF R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::ADD:
            oss << "ADD R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra)
                << ", R" << static_cast<int>(rb);
            break;
        case Opcode::ADDI:
            oss << "ADDI R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra)
                << ", " << static_cast<int>(simm12);
            break;
        case Opcode::MOV:
            oss << "MOV R" << static_cast<int>(rd) << ", " << static_cast<int>(simm12);
            break;
        case Opcode::MOVR:
            oss << "MOVR R" << static_cast<int>(rd) << ", R" << static_cast<int>(ra);
            break;
        case Opcode::CMP:
            oss << "CMP R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            break;
        case Opcode::CMPF:
            oss << "CMPF R" << static_cast<int>(ra) << ", R" << static_cast<int>(rb);
            break;
        case Opcode::JMP:
            oss << "JMP " << static_cast<int>(simm12);
            break;
        case Opcode::JNZ:
            oss << "JNZ " << static_cast<int>(simm12);
            break;
        case Opcode::JZ:
            oss << "JZ " << static_cast<int>(simm12);
            break;
        case Opcode::HALT:
            oss << "HALT";
            break;
        case Opcode::RET:
            oss << "RET";
            break;
        case Opcode::NOP:
            oss << "NOP";
            break;
        case Opcode::CALL:
            oss << "CALL " << static_cast<int>(simm12);
            break;
        default:
            oss << OpcodeName(op);
            if (rd != 0 || ra != 0 || rb != 0) {
                oss << " R" << static_cast<int>(rd)
                    << ", R" << static_cast<int>(ra)
                    << ", R" << static_cast<int>(rb);
            }
            break;
        }
        return oss.str();
    }
};

} // namespace opt
} // namespace voxel
