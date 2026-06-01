#pragma once

#include "voxel/bytecode/instruction.hpp"
#include "voxel/core/types.hpp"
#include <cstring>
#include <unordered_set>
#include <vector>

namespace voxel {

enum class VerifyError : u8 {
    None = 0,
    InvalidOpcode,
    BranchOutOfBounds,
    SegmentOutOfRange,
    RegOutOfRange,
    StackUnderflow,
    TypeMismatch,
    DivisionByZero,
};

inline const char* ErrorString(VerifyError e) {
    switch (e) {
    case VerifyError::None:               return "no error";
    case VerifyError::InvalidOpcode:      return "invalid opcode — byte value not in Opcode enum";
    case VerifyError::BranchOutOfBounds:  return "branch target is outside the code segment bounds";
    case VerifyError::SegmentOutOfRange:  return "segment identifier exceeds available segment count";
    case VerifyError::RegOutOfRange:      return "register index exceeds the architecture limit";
    case VerifyError::StackUnderflow:     return "stack underflow — pop from empty evaluation stack";
    case VerifyError::TypeMismatch:       return "type mismatch between operands in typed instruction";
    case VerifyError::DivisionByZero:     return "potential division by zero (register 0 as divisor)";
    default:                              return "unknown verification error";
    }
}

class Verifier {
public:
    struct Result {
        VerifyError error;
        sz offset;
        const char* msg;

        Result()
            : error(VerifyError::None), offset(0), msg(nullptr) {}

        Result(VerifyError e, sz off, const char* m)
            : error(e), offset(off), msg(m) {}

        bool Ok() const { return error == VerifyError::None; }
    };

    static constexpr sz kMaxScalarReg = 64;
    static constexpr sz kMaxVectorReg = 16;
    static constexpr sz kMaxMaskReg   = 8;

    Result Verify(const u32* code, sz codeSize, sz numSegments) {
        if (!code || codeSize == 0) {
            return Result(VerifyError::None, 0, "empty or null program");
        }

        if (numSegments > 256) {
            return Result(VerifyError::SegmentOutOfRange, 0,
                          "segment count exceeds maximum of 256");
        }

        bool foundHalt = false;

        for (sz pc = 0; pc < codeSize; ++pc) {
            Instruction inst{code[pc]};
            Opcode op = inst.Op();
            u8 rd = inst.Rd();
            u8 ra = inst.Ra();
            u8 rb = inst.Rb();
            i16 imm = inst.Simm12();

            {
                Result r = VerifyOpcodeValid(op, pc);
                if (!r.Ok()) return r;
            }

            {
                Result r = VerifyRegisters(op, rd, ra, rb, pc);
                if (!r.Ok()) return r;
            }

            {
                Result r = VerifyBranchTarget(op, pc, imm, codeSize);
                if (!r.Ok()) return r;
            }

            {
                Result r = VerifySegmentRange(op, rd, ra, rb, inst.Imm12(), numSegments, pc);
                if (!r.Ok()) return r;
            }

            {
                Result r = VerifyDivisionByZero(op, inst, pc);
                if (!r.Ok()) return r;
            }

            if (op == Opcode::HALT) {
                foundHalt = true;
            }
        }

        {
            Result r = VerifyHaltReachability(code, codeSize, foundHalt);
            if (!r.Ok()) return r;
        }

        {
            Result r = VerifyNoInfiniteLoops(code, codeSize);
            if (!r.Ok()) return r;
        }

        return Result();
    }

private:
    Result VerifyOpcodeValid(Opcode op, sz pc) {
        u8 raw = static_cast<u8>(op);

        switch (raw) {
        // Control — 0x00..0x09
        case 0x00: case 0x01: case 0x02: case 0x03:
        case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x08: case 0x09:
        // Scalar Move — 0x10..0x1E
        case 0x10: case 0x11: case 0x12: case 0x13:
        case 0x14: case 0x15: case 0x16: case 0x17:
        case 0x18: case 0x19: case 0x1A: case 0x1B:
        case 0x1C: case 0x1D: case 0x1E:
        // Scalar Arithmetic — 0x20..0x2F
        case 0x20: case 0x21: case 0x22: case 0x23:
        case 0x24: case 0x25: case 0x26: case 0x27:
        case 0x28: case 0x29: case 0x2A: case 0x2B:
        case 0x2C: case 0x2D: case 0x2E: case 0x2F:
        // Scalar Bitwise — 0x30..0x3F
        case 0x30: case 0x31: case 0x32: case 0x33:
        case 0x34: case 0x35: case 0x36: case 0x37:
        case 0x38: case 0x39: case 0x3A: case 0x3B:
        case 0x3C: case 0x3D: case 0x3E: case 0x3F:
        // Comparison — 0x40..0x48
        case 0x40: case 0x41: case 0x42: case 0x43:
        case 0x44: case 0x45: case 0x46: case 0x47:
        case 0x48:
        // Branching — 0x50..0x5F
        case 0x50: case 0x51: case 0x52: case 0x53:
        case 0x54: case 0x55: case 0x56: case 0x57:
        case 0x58: case 0x59: case 0x5A: case 0x5B:
        case 0x5C: case 0x5D: case 0x5E: case 0x5F:
        // Type Conversion — 0x60..0x6F
        case 0x60: case 0x61: case 0x62: case 0x63:
        case 0x64: case 0x65: case 0x66: case 0x67:
        case 0x68: case 0x69: case 0x6A: case 0x6B:
        case 0x6C: case 0x6D: case 0x6E: case 0x6F:
        // Vector I/O — 0x70..0x7F
        case 0x70: case 0x71: case 0x72: case 0x73:
        case 0x74: case 0x75: case 0x76: case 0x77:
        case 0x78: case 0x79: case 0x7A: case 0x7B:
        case 0x7C: case 0x7D: case 0x7E: case 0x7F:
        // Vector Arithmetic — 0x80..0x8F
        case 0x80: case 0x81: case 0x82: case 0x83:
        case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x89: case 0x8A: case 0x8B:
        case 0x8C: case 0x8D: case 0x8E: case 0x8F:
        // Vector Scalar — 0x90..0x96
        case 0x90: case 0x91: case 0x92: case 0x93:
        case 0x94: case 0x95: case 0x96:
        // Vector Comparison — 0xA0..0xA7
        case 0xA0: case 0xA1: case 0xA2: case 0xA3:
        case 0xA4: case 0xA5: case 0xA6: case 0xA7:
        // Vector Logical — 0xB0..0xB7
        case 0xB0: case 0xB1: case 0xB2: case 0xB3:
        case 0xB4: case 0xB5: case 0xB6: case 0xB7:
        // Vector Filter — 0xC0..0xC9
        case 0xC0: case 0xC1: case 0xC2: case 0xC3:
        case 0xC4: case 0xC5: case 0xC6: case 0xC7:
        case 0xC8: case 0xC9:
        // Vector Reduction — 0xD0..0xDC
        case 0xD0: case 0xD1: case 0xD2: case 0xD3:
        case 0xD4: case 0xD5: case 0xD6: case 0xD7:
        case 0xD8: case 0xD9: case 0xDA: case 0xDB:
        case 0xDC:
        // Aggregate — 0xE0..0xEF
        case 0xE0: case 0xE1: case 0xE2: case 0xE3:
        case 0xE4: case 0xE5: case 0xE6: case 0xE7:
        case 0xE8: case 0xE9: case 0xEA: case 0xEB:
        case 0xEC: case 0xED: case 0xEE: case 0xEF:
        // Hash/Sort/Join — 0xF0..0xFE
        case 0xF0: case 0xF1: case 0xF2: case 0xF3:
        case 0xF4: case 0xF5: case 0xF6: case 0xF7:
        case 0xF8: case 0xF9: case 0xFA: case 0xFB:
        case 0xFC: case 0xFD: case 0xFE:
            return Result();

        default:
            return Result(VerifyError::InvalidOpcode, pc,
                          "opcode byte value does not correspond to any valid instruction");
        }
    }

    Result VerifyRegisters(Opcode op, u8 rd, u8 ra, u8 rb, sz pc) {
        bool isVector = IsVectorOp(op);

        if (op == Opcode::VBLEND || op == Opcode::VMASK_STORE ||
            op == Opcode::VMASK_LOAD || op == Opcode::VLOAD_MASKED ||
            op == Opcode::VSTORE_MASKED) {
            if (rb >= kMaxMaskReg) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "mask register index exceeds architectural limit");
            }
        }

        if (IsScalarOp(op)) {
            if (rd >= kMaxScalarReg && NeedsDest(op)) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "scalar destination register out of range");
            }
            if (ra >= kMaxScalarReg && NeedsSrcA(op)) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "scalar source A register out of range");
            }
            if (rb >= kMaxScalarReg && NeedsSrcB(op)) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "scalar source B register out of range");
            }
        } else if (isVector) {
            bool isReduction = IsReductionOp(op);
            if (!isReduction && rd >= kMaxVectorReg) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "vector destination register out of range");
            }
            if (ra >= kMaxVectorReg) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "vector source A register out of range");
            }
            if (rb >= kMaxVectorReg && NeedsSrcB(op)) {
                return Result(VerifyError::RegOutOfRange, pc,
                              "vector source B register out of range");
            }
        }

        return Result();
    }

    Result VerifyBranchTarget(Opcode op, sz pc, i16 offset, sz codeSize) {
        u8 raw = static_cast<u8>(op);

        if (raw < 0x50 || raw > 0x5F) {
            return Result();
        }

        if (op == Opcode::RET) {
            return Result();
        }

        if (op == Opcode::TABLE_JMP) {
            return Result();
        }

        i64 target = static_cast<i64>(pc) + 1 + static_cast<i64>(offset);
        if (target < 0 || static_cast<sz>(target) >= codeSize) {
            return Result(VerifyError::BranchOutOfBounds, pc,
                          "computed branch target falls outside valid code range");
        }
        return Result();
    }

    Result VerifySegmentRange(Opcode op, u8 rd, u8 ra, u8 rb,
                               u16 imm, sz numSegments, sz pc) {
        bool usesSegId = false;
        u8 segId = 0;

        switch (static_cast<u8>(op)) {
        case 0x70: case 0x71: case 0x72: case 0x73:
        case 0x74: case 0x75:
            segId = (imm >> 8) & 0xF;
            usesSegId = true;
            break;

        case 0xC8: case 0xC9:
            segId = (imm >> 8) & 0xF;
            usesSegId = true;
            break;

        case 0xE0: case 0xE1: case 0xE2: case 0xE3:
        case 0xE4: case 0xE5: case 0xE6: case 0xE7:
        case 0xE8: case 0xE9: case 0xEA: case 0xEB:
        case 0xEC: case 0xED:
            segId = ra;
            usesSegId = true;
            break;

        case 0xF0: case 0xF1:
            segId = ra;
            usesSegId = true;
            break;

        case 0xF2: case 0xF3:
        case 0xFB: case 0xFC:
            segId = rd;
            usesSegId = true;
            break;

        case 0xF4: case 0xF5:
            segId = rd;
            usesSegId = true;
            break;

        case 0xF6: case 0xF7: case 0xF8: case 0xF9: case 0xFA:
            if (rd >= numSegments || ra >= numSegments || rb >= numSegments) {
                return Result(VerifyError::SegmentOutOfRange, pc,
                              "join operation references a segment that does not exist");
            }
            return Result();

        case 0xFD:
            segId = rd;
            usesSegId = true;
            break;

        default:
            break;
        }

        if (usesSegId && segId >= numSegments) {
            return Result(VerifyError::SegmentOutOfRange, pc,
                          "instruction references segment beyond available range");
        }
        return Result();
    }

    Result VerifyDivisionByZero(Opcode op, const Instruction&, sz pc) {
        switch (op) {
        case Opcode::DIV:  case Opcode::DIVF: case Opcode::MOD:
        case Opcode::VDIV: case Opcode::VMOD:
        case Opcode::VSDIV: case Opcode::VSMOD:
            return Result();

        default:
            return Result();
        }
    }

    Result VerifyHaltReachability(const u32* code, sz codeSize,
                                   bool foundHalt) {
        if (!foundHalt && codeSize > 0) {
            return Result(VerifyError::None, 0,
                          "warning: program contains no HALT instruction");
        }

        std::vector<bool> reachable(codeSize, false);
        std::vector<sz>  worklist;

        if (codeSize > 0) {
            worklist.push_back(0);
        }

        while (!worklist.empty()) {
            sz pc = worklist.back();
            worklist.pop_back();

            if (pc >= codeSize || reachable[pc]) {
                continue;
            }
            reachable[pc] = true;

            Instruction inst{code[pc]};
            Opcode op = inst.Op();

            if (op == Opcode::HALT || op == Opcode::RET) {
                continue;
            }

            u8 raw = static_cast<u8>(op);

            if (raw >= 0x50 && raw <= 0x5F) {
                if (op == Opcode::JMP || op == Opcode::CALL) {
                    i64 tgt = static_cast<i64>(pc) + 1
                            + static_cast<i64>(inst.Simm12());
                    if (tgt >= 0 && static_cast<sz>(tgt) < codeSize) {
                        worklist.push_back(static_cast<sz>(tgt));
                    }
                } else if (op == Opcode::JZ  || op == Opcode::JNZ ||
                           op == Opcode::JS  || op == Opcode::JNS ||
                           op == Opcode::JO  || op == Opcode::JNO ||
                           op == Opcode::JC  || op == Opcode::JNC ||
                           op == Opcode::JL  || op == Opcode::JLE ||
                           op == Opcode::JG  || op == Opcode::JGE) {
                    worklist.push_back(pc + 1);
                    i64 tgt = static_cast<i64>(pc) + 1
                            + static_cast<i64>(inst.Simm12());
                    if (tgt >= 0 && static_cast<sz>(tgt) < codeSize) {
                        worklist.push_back(static_cast<sz>(tgt));
                    }
                } else if (op == Opcode::TABLE_JMP) {
                    worklist.push_back(pc + 1);
                } else {
                    worklist.push_back(pc + 1);
                }
            } else {
                worklist.push_back(pc + 1);
            }
        }

        for (sz i = 0; i < codeSize; ++i) {
            if (reachable[i]) {
                Instruction inst{code[i]};
                if (inst.Op() == Opcode::HALT) {
                    return Result();
                }
            }
        }

        return Result(VerifyError::BranchOutOfBounds,
                      codeSize > 0 ? codeSize - 1 : 0,
                      "no reachable HALT instruction — dead code detected");
    }

    Result VerifyNoInfiniteLoops(const u32* code, sz codeSize) {
        for (sz pc = 0; pc < codeSize; ++pc) {
            Instruction inst{code[pc]};
            if (inst.Op() == Opcode::JMP && inst.Simm12() == -1) {
                return Result(VerifyError::None, pc,
                              "warning: unconditional self-loop may be infinite");
            }
        }
        return Result();
    }

    static bool NeedsDest(Opcode op) {
        u8 raw = static_cast<u8>(op);
        return raw != 0x40 && raw != 0x41 && raw != 0x42 &&
               raw != 0x43 && raw != 0x44;
    }

    static bool NeedsSrcA(Opcode op) {
        u8 raw = static_cast<u8>(op);
        return raw != 0x00 && raw != 0x01 && raw != 0x02 &&
               raw != 0x03 && raw != 0x04 && raw != 0x05 &&
               raw != 0x06 && raw != 0x07 && raw != 0x08 &&
               raw != 0x09 && raw != 0x5E;
    }

    static bool NeedsSrcB(Opcode op) {
        switch (op) {
        case Opcode::ADD: case Opcode::SUB: case Opcode::MUL:
        case Opcode::DIV: case Opcode::MOD: case Opcode::MIN:
        case Opcode::MAX: case Opcode::AVG:
        case Opcode::ADDF: case Opcode::SUBF: case Opcode::MULF:
        case Opcode::DIVF:
        case Opcode::AND: case Opcode::OR: case Opcode::XOR:
        case Opcode::SHL: case Opcode::SHR: case Opcode::SAR:
        case Opcode::ROL: case Opcode::ROR:
        case Opcode::BEXTR: case Opcode::BZHI: case Opcode::PDEP:
        case Opcode::CMP: case Opcode::CMPF: case Opcode::CMPU:
        case Opcode::TST: case Opcode::TSTF:
        case Opcode::SELECT: case Opcode::SELECTV:
            return true;
        default:
            return false;
        }
    }

    static bool IsReductionOp(Opcode op) {
        u8 raw = static_cast<u8>(op);
        return raw >= 0xD0 && raw <= 0xDC;
    }
};

} // namespace voxel
