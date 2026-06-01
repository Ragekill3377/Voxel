#pragma once

#include "voxel/codegen/jit.hpp"
#include <vector>
#include <cstring>
#include <cstdint>

namespace voxel {
namespace codegen {
namespace arm64 {

// ============================================================================
// Register enumerations
// ============================================================================

enum class Reg64 : u8 {
    X0 = 0,  X1 = 1,  X2 = 2,  X3 = 3,
    X4 = 4,  X5 = 5,  X6 = 6,  X7 = 7,
    X8 = 8,  X9 = 9,  X10 = 10, X11 = 11,
    X12 = 12, X13 = 13, X14 = 14, X15 = 15,
    X16 = 16, X17 = 17, X18 = 18, X19 = 19,
    X20 = 20, X21 = 21, X22 = 22, X23 = 23,
    X24 = 24, X25 = 25, X26 = 26, X27 = 27,
    X28 = 28, X29 = 29, X30 = 30,
};

constexpr u8 XZR = 31;
constexpr u8 SP  = 31;
constexpr u8 LR  = 30; // X30 alias

enum class Reg32 : u8 {
    W0 = 0,  W1 = 1,  W2 = 2,  W3 = 3,
    W4 = 4,  W5 = 5,  W6 = 6,  W7 = 7,
    W8 = 8,  W9 = 9,  W10 = 10, W11 = 11,
    W12 = 12, W13 = 13, W14 = 14, W15 = 15,
    W16 = 16, W17 = 17, W18 = 18, W19 = 19,
    W20 = 20, W21 = 21, W22 = 22, W23 = 23,
    W24 = 24, W25 = 25, W26 = 26, W27 = 27,
    W28 = 28, W29 = 29, W30 = 30,
};
constexpr u8 WZR = 31;

enum class VReg : u8 {
    V0 = 0,   V1 = 1,   V2 = 2,   V3 = 3,
    V4 = 4,   V5 = 5,   V6 = 6,   V7 = 7,
    V8 = 8,   V9 = 9,   V10 = 10, V11 = 11,
    V12 = 12, V13 = 13, V14 = 14, V15 = 15,
    V16 = 16, V17 = 17, V18 = 18, V19 = 19,
    V20 = 20, V21 = 21, V22 = 22, V23 = 23,
    V24 = 24, V25 = 25, V26 = 26, V27 = 27,
    V28 = 28, V29 = 29, V30 = 30, V31 = 31,
};

enum class QReg : u8 {
    Q0 = 0,   Q1 = 1,   Q2 = 2,   Q3 = 3,
    Q4 = 4,   Q5 = 5,   Q6 = 6,   Q7 = 7,
    Q8 = 8,   Q9 = 9,   Q10 = 10, Q11 = 11,
    Q12 = 12, Q13 = 13, Q14 = 14, Q15 = 15,
    Q16 = 16, Q17 = 17, Q18 = 18, Q19 = 19,
    Q20 = 20, Q21 = 21, Q22 = 22, Q23 = 23,
    Q24 = 24, Q25 = 25, Q26 = 26, Q27 = 27,
    Q28 = 28, Q29 = 29, Q30 = 30, Q31 = 31,
};

// ============================================================================
// Branch conditions
// ============================================================================

enum class Cond : u8 {
    EQ = 0,  NE = 1,  CS = 2,  CC = 3,
    MI = 4,  PL = 5,  VS = 6,  VC = 7,
    HI = 8,  LS = 9,  GE = 10, LT = 11,
    GT = 12, LE = 13, AL = 14, NV = 15,
};

// ============================================================================
// ARM64Assembler — machine code emitter for AArch64
// ============================================================================

class ARM64Assembler {
public:
    std::vector<u32>& Code;
    sz                CurrentOffset; // in u32 units

    explicit ARM64Assembler(std::vector<u32>& code)
        : Code(code)
        , CurrentOffset(0)
    {}

    ARM64Assembler(const ARM64Assembler&)            = delete;
    ARM64Assembler& operator=(const ARM64Assembler&) = delete;

    void Emit(u32 inst)
    {
        Code.push_back(inst);
        ++CurrentOffset;
    }

    // ----------------------------------------------------------------
    // ADD (shifted register)
    // sf=1, op=S=0, sh=00, N=0
    // encoding: sf|0|0|01011|shift|0|Rm|imm6|Rn|Rd
    // ADD Xd, Xn, Xm: 0x8B000000 | (Rm<<16) | (Rn<<5) | Rd
    // ----------------------------------------------------------------

    u32 AddRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x8B000000u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 AddWRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x0B000000u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 SubRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0xCB000000u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 SubWRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x4B000000u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 MulRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x9B007C00u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 MulWRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x1B007C00u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 SDivRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x9AC00C00u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 UDivRegRegReg(u8 rd, u8 rn, u8 rm)
    {
        return 0x9AC00800u | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    // ----------------------------------------------------------------
    // ADD (immediate)
    // sf=1, op=S=0, sh=0
    // encoding: sf|op|S|100010|sh|imm12|Rn|Rd
    // ADD Xd, Xn, #imm12: 0x91000000 | (imm12<<10) | (Rn<<5) | Rd
    // ----------------------------------------------------------------

    u32 AddImm(u8 rd, u8 rn, u16 imm12)
    {
        return 0x91000000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 SubImm(u8 rd, u8 rn, u16 imm12)
    {
        return 0xD1000000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 AddWImm(u8 rd, u8 rn, u16 imm12)
    {
        return 0x11000000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    u32 SubWImm(u8 rd, u8 rn, u16 imm12)
    {
        return 0x51000000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    // ----------------------------------------------------------------
    // CMP (SUBS XZR, Rn, Rm/N#imm)
    // ----------------------------------------------------------------

    u32 CmpRegReg(u8 rn, u8 rm)
    {
        return 0xEB00001Fu | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5);
    }

    u32 CmpImm(u8 rn, u16 imm12)
    {
        return 0xF100001Fu | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5);
    }

    u32 CmpWRegReg(u8 rn, u8 rm)
    {
        return 0x6B00001Fu | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5);
    }

    // ----------------------------------------------------------------
    // MOV via ORR with XZR
    // ORR Xd, XZR, Xm: 0xAA0003E0 | (Xm<<16) | Xd
    // ----------------------------------------------------------------

    u32 MovRegReg(u8 rd, u8 rn)
    {
        return 0xAA0003E0u | ((rn & 0x1F) << 16) | (rd & 0x1F);
    }

    // ----------------------------------------------------------------
    // MOVZ/MOVK
    // MOVZ Xd, #imm16, lsl #shift
    // sf=1, opc=10, hw=shift/16
    // encoding: sf|10|100101|hw|imm16|Rd
    // = 0xD2800000 | (hw<<21) | (imm16<<5) | Rd
    // ----------------------------------------------------------------

    u32 MovzRegImm(u8 rd, u16 imm16, u8 shift = 0)
    {
        u8 hw = shift / 16;
        return 0xD2800000u | ((hw & 3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F);
    }

    u32 MovkRegImm(u8 rd, u16 imm16, u8 shift)
    {
        u8 hw = shift / 16;
        return 0xF2800000u | ((hw & 3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F);
    }

    u32 MovnRegImm(u8 rd, u16 imm16, u8 shift = 0)
    {
        u8 hw = shift / 16;
        return 0x92800000u | ((hw & 3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F);
    }

    // ----------------------------------------------------------------
    // LDR / STR (unsigned offset, scaled by element size)
    // ----------------------------------------------------------------

    u32 LdrRegOffset(u8 rt, u8 rn, i16 offset)
    {
        if (offset >= 0) {
            u32 uoff = static_cast<u32>(offset) & 0xFFF;
            return 0xF9400000u | (uoff << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
        }
        return LdrPreIndex(rt, rn, offset);
    }

    u32 StrRegOffset(u8 rt, u8 rn, i16 offset)
    {
        if (offset >= 0) {
            u32 uoff = static_cast<u32>(offset) & 0xFFF;
            return 0xF9000000u | (uoff << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
        }
        return StrPreIndex(rt, rn, offset);
    }

    u32 LdrWOffset(u8 rt, u8 rn, i16 offset)
    {
        u32 uoff = (static_cast<u32>(offset) & 0xFFF) >> 0; // scaled by 4
        return 0xB9400000u | (uoff << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    u32 StrWOffset(u8 rt, u8 rn, i16 offset)
    {
        u32 uoff = (static_cast<u32>(offset) & 0xFFF) >> 0;
        return 0xB9000000u | (uoff << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    // Pre-/post-index LDR: LDR Xt, [Xn, #simm9]!
    u32 LdrPreIndex(u8 rt, u8 rn, i16 imm9)
    {
        u32 uimm = static_cast<u32>(imm9) & 0x1FF;
        return 0xF8400C00u | (uimm << 12) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    u32 LdrPostIndex(u8 rt, u8 rn, i16 imm9)
    {
        u32 uimm = static_cast<u32>(imm9) & 0x1FF;
        return 0xF8400400u | (uimm << 12) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    u32 StrPreIndex(u8 rt, u8 rn, i16 imm9)
    {
        u32 uimm = static_cast<u32>(imm9) & 0x1FF;
        return 0xF8000C00u | (uimm << 12) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    u32 StrPostIndex(u8 rt, u8 rn, i16 imm9)
    {
        u32 uimm = static_cast<u32>(imm9) & 0x1FF;
        return 0xF8000400u | (uimm << 12) | ((rn & 0x1F) << 5) | (rt & 0x1F);
    }

    // ----------------------------------------------------------------
    // LDP / STP (load/store pair)
    // LDP Xt1, Xt2, [Xn, #imm7]: imm7 is signed, scaled by 8
    // ----------------------------------------------------------------

    u32 LdpRegOffset(u8 rt1, u8 rt2, u8 rn, i16 offset)
    {
        u32 uimm = (static_cast<u32>(offset) >> 3) & 0x7F;
        return 0xA9400000u | (uimm << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt1 & 0x1F);
    }

    u32 StpRegOffset(u8 rt1, u8 rt2, u8 rn, i16 offset)
    {
        u32 uimm = (static_cast<u32>(offset) >> 3) & 0x7F;
        return 0xA9000000u | (uimm << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt1 & 0x1F);
    }

    // STP (pre-index): STP Xt1, Xt2, [Xn, #imm7]!
    u32 StpPreIndex(u8 rt1, u8 rt2, u8 rn, i16 imm)
    {
        u32 uimm = (static_cast<u32>(imm) >> 3) & 0x7F;
        return 0xA9800000u | (uimm << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt1 & 0x1F);
    }

    // LDP (post-index): LDP Xt1, Xt2, [Xn], #imm7
    u32 LdpPostIndex(u8 rt1, u8 rt2, u8 rn, i16 imm)
    {
        u32 uimm = (static_cast<u32>(imm) >> 3) & 0x7F;
        return 0xA8C00000u | (uimm << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt1 & 0x1F);
    }

    // ----------------------------------------------------------------
    // SIMD/NEON — LDR/STR for Q (128-bit) registers
    // LDR Qd, [Xn, #imm12]: scaled by 16
    // ----------------------------------------------------------------

    u32 LdrQOffset(u8 qt, u8 xn, i16 offset)
    {
        u32 uoff = (static_cast<u32>(offset) >> 4) & 0xFFF;
        return 0x3DC00000u | (uoff << 10) | ((xn & 0x1F) << 5) | (qt & 0x1F);
    }

    u32 StrQOffset(u8 qt, u8 xn, i16 offset)
    {
        u32 uoff = (static_cast<u32>(offset) >> 4) & 0xFFF;
        return 0x3D800000u | (uoff << 10) | ((xn & 0x1F) << 5) | (qt & 0x1F);
    }

    u32 LdpQOffset(u8 qt1, u8 qt2, u8 xn, i16 offset)
    {
        u32 uimm = (static_cast<u32>(offset) >> 4) & 0x7F;
        return 0xAD400000u | (uimm << 15) | ((qt2 & 0x1F) << 10) | ((xn & 0x1F) << 5) | (qt1 & 0x1F);
    }

    u32 StpQOffset(u8 qt1, u8 qt2, u8 xn, i16 offset)
    {
        u32 uimm = (static_cast<u32>(offset) >> 4) & 0x7F;
        return 0xAD000000u | (uimm << 15) | ((qt2 & 0x1F) << 10) | ((xn & 0x1F) << 5) | (qt1 & 0x1F);
    }

    // ----------------------------------------------------------------
    // SIMD/NEON — .2D (f64×2) floating-point arithmetic
    // ----------------------------------------------------------------

    u32 FaddV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E60D400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FsubV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EE0D400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FmulV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EE0FC00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FdivV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E60FC00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FminV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EE0F400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FmaxV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E60F400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FnegV2D(u8 vd, u8 vn)
    {
        return 0x6EF17C00u | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FabsV2D(u8 vd, u8 vn)
    {
        return 0x6EF07C00u | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // FADDP (horizontal add pair): Vd.2D, Vn.2D, Vm.2D
    u32 FaddpV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E60D400u | 0x00200000u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // SIMD .4S (f32×4) floating-point arithmetic
    // ----------------------------------------------------------------

    u32 FaddV4S(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E20D400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FsubV4S(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EA0D400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FmulV4S(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EA0FC00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FdivV4S(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E20FC00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // SIMD floating-point comparisons
    // ----------------------------------------------------------------

    u32 FcmeqV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EE0E400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FcmgtV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EE0E400u | 0x00200000u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 FcmgeV2D(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E60E400u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // BSL (bitwise select / blend)
    // BSL Vd.16B, Vn.16B, Vm.16B
    // encoding: 0x4E601C00 | ...
    // ----------------------------------------------------------------

    u32 BslV16B(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E601C00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // BIT (bitwise insert if true): Vd.16B, Vn.16B, Vm.16B
    u32 BitV16B(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EA01C00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // AND/ORR/EOR (vector): Vd.16B, Vn.16B, Vm.16B
    u32 AndV16B(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E201C00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 OrrV16B(u8 vd, u8 vn, u8 vm)
    {
        return 0x4EA01C00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    u32 EorV16B(u8 vd, u8 vn, u8 vm)
    {
        return 0x4E601C00u | ((vm & 0x1F) << 16) | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // NOT (vector): Vd.16B, Vn.16B
    // ----------------------------------------------------------------

    u32 NotV16B(u8 vd, u8 vn)
    {
        return 0x4E205800u | ((vn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // DUP (element): DUP Vd.2D, Xn
    // ----------------------------------------------------------------

    u32 DupV2D(u8 vd, u8 xn)
    {
        return 0x4E080C00u | ((xn & 0x1F) << 5) | (vd & 0x1F);
    }

    // ----------------------------------------------------------------
    // Branches
    // B offset: 0|00101|imm26
    // B.cond offset: 01010100|imm19|0|cond
    // RET: 110101100101111100000011111000?? actually RET Xn = 0xD65F0000 | (Xn<<5)
    // ----------------------------------------------------------------

    u32 BImm26(i32 offset)
    {
        u32 imm = static_cast<u32>(offset >> 2) & 0x3FFFFFFu;
        return 0x14000000u | imm;
    }

    u32 BcondImm19(u8 cond, i32 offset)
    {
        u32 imm = static_cast<u32>(offset >> 2) & 0x7FFFFu;
        return 0x54000000u | (imm << 5) | (cond & 0xF);
    }

    u32 BlImm26(i32 offset)
    {
        u32 imm = static_cast<u32>(offset >> 2) & 0x3FFFFFFu;
        return 0x94000000u | imm;
    }

    u32 Ret()
    {
        return 0xD65F03C0u;
    }

    u32 Blr(u8 xn)
    {
        return 0xD63F0000u | ((xn & 0x1F) << 5);
    }

    u32 Br(u8 xn)
    {
        return 0xD61F0000u | ((xn & 0x1F) << 5);
    }

    sz Label()
    {
        return CurrentOffset;
    }

    void PatchBranch(sz branchSlot, i32 targetOffset)
    {
        i32 byteOffset = static_cast<i32>(branchSlot) * 4;
        i32 targetByte  = targetOffset * 4;
        i32 rel         = targetByte - byteOffset;
        u32& inst       = Code[branchSlot];

        u32 op = inst >> 26;
        if (op == 0x05) { // B
            inst = BImm26(rel);
        } else if (op == 0x15) { // B.cond
            u8 cond = inst & 0xF;
            inst = BcondImm19(cond, rel);
        }
    }

    // ----------------------------------------------------------------
    // NOP
    // ----------------------------------------------------------------

    u32 Nop()
    {
        return 0xD503201Fu;
    }

    // ----------------------------------------------------------------
    // UBFM / SBFM (bitfield move) — for extracting bit ranges
    // ----------------------------------------------------------------

    u32 Ubfm(u8 rd, u8 rn, u8 immr, u8 imms)
    {
        u32 sf = 1; // 64-bit
        return (sf << 31) | (0b11010110 << 22) | ((immr & 0x3F) << 16) | ((imms & 0x3F) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    // LSL (alias for UBFM)
    u32 LslImm(u8 rd, u8 rn, u8 shift)
    {
        if (shift == 0) return MovRegReg(rd, rn);
        return Ubfm(rd, rn, static_cast<u8>((-shift) & 0x3F), static_cast<u8>(63 - shift));
    }

    // LSR (alias for UBFM)
    u32 LsrImm(u8 rd, u8 rn, u8 shift)
    {
        return Ubfm(rd, rn, shift, 63);
    }

    // ASR (alias for SBFM)
    u32 AsrImm(u8 rd, u8 rn, u8 shift)
    {
        return 0x93800000u | 0x00400000u | ((shift & 0x3F) << 16) | (63 << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }

    // ----------------------------------------------------------------
    // CSEL (conditional select): CSEL Xd, Xn, Xm, cond
    // ----------------------------------------------------------------

    u32 Csel(u8 rd, u8 rn, u8 rm, u8 cond)
    {
        return 0x9A800000u | ((rm & 0x1F) << 16) | ((cond & 0xF) << 12) | ((rn & 0x1F) << 5) | (rd & 0x1F);
    }
};

// ============================================================================
// ARM64Compiler — translates VoxelVM bytecode to AArch64 machine code
// ============================================================================

class ARM64Compiler : public JitCompiler {
public:
    static constexpr sz kScalarRegOffset = 0;
    static constexpr sz kVectorRegOffset = 128;
    static constexpr sz kVectorStride    = 32;
    static constexpr sz kFlagsOffset     = 384;

    ARM64Compiler() = default;
    ~ARM64Compiler() override = default;

    bool Compile(const u32* bytecode, sz bytecodeSize, JitFunction& out) override
    {
        std::vector<u32> code;
        code.reserve(bytecodeSize * 16);
        ARM64Assembler a(code);

        // Register allocation:
        // X19 = regfile base pointer (callee-saved)
        // X20 = segments pointer (callee-saved)
        // X21 = ctx pointer (callee-saved)
        // X22-X25 = scratch for scalar loads/stores
        // Q16-Q23 = vector registers (callee-saved lower 64 bits)
        // Q0-Q15 = scratch vector registers

        auto loadScalar = [&](u8 vreg, u8 hostreg) {
            i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
            if (off < 4096) {
                a.Emit(a.LdrRegOffset(hostreg, 19, static_cast<i16>(off)));
            } else {
                a.Emit(a.MovzRegImm(22, static_cast<u16>(off & 0xFFFF), 0));
                a.Emit(a.MovkRegImm(22, static_cast<u16>((off >> 16) & 0xFFFF), 16));
                a.Emit(a.AddRegRegReg(22, 22, 19));
                a.Emit(a.LdrRegOffset(hostreg, 22, 0));
            }
        };

        auto storeScalar = [&](u8 vreg, u8 hostreg) {
            i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
            if (off < 4096) {
                a.Emit(a.StrRegOffset(hostreg, 19, static_cast<i16>(off)));
            } else {
                a.Emit(a.MovzRegImm(22, static_cast<u16>(off & 0xFFFF), 0));
                a.Emit(a.MovkRegImm(22, static_cast<u16>((off >> 16) & 0xFFFF), 16));
                a.Emit(a.AddRegRegReg(22, 22, 19));
                a.Emit(a.StrRegOffset(hostreg, 22, 0));
            }
        };

        auto loadVector = [&](u8 vreg, u8 qreg) {
            i32 off = static_cast<i32>(kVectorRegOffset + static_cast<sz>(vreg) * kVectorStride);
            if (off < 4096 && (off & 0xF) == 0) {
                a.Emit(a.LdrQOffset(qreg, 19, static_cast<i16>(off)));
            } else {
                i32 alignedOff = off & ~0xF;
                i32 remainder = off - alignedOff;
                a.Emit(a.MovzRegImm(22, static_cast<u16>(alignedOff & 0xFFFF), 0));
                if ((alignedOff >> 16) & 0xFFFF) a.Emit(a.MovkRegImm(22, static_cast<u16>((alignedOff >> 16) & 0xFFFF), 16));
                a.Emit(a.AddRegRegReg(22, 22, 19));
                a.Emit(a.LdrQOffset(qreg, 22, static_cast<i16>(remainder)));
            }
        };

        auto storeVector = [&](u8 vreg, u8 qreg) {
            i32 off = static_cast<i32>(kVectorRegOffset + static_cast<sz>(vreg) * kVectorStride);
            if (off < 4096 && (off & 0xF) == 0) {
                a.Emit(a.StrQOffset(qreg, 19, static_cast<i16>(off)));
            } else {
                i32 alignedOff = off & ~0xF;
                i32 remainder = off - alignedOff;
                a.Emit(a.MovzRegImm(22, static_cast<u16>(alignedOff & 0xFFFF), 0));
                if ((alignedOff >> 16) & 0xFFFF) a.Emit(a.MovkRegImm(22, static_cast<u16>((alignedOff >> 16) & 0xFFFF), 16));
                a.Emit(a.AddRegRegReg(22, 22, 19));
                a.Emit(a.StrQOffset(qreg, 22, static_cast<i16>(remainder)));
            }
        };

        auto immAsRel = [](u16 imm12) -> i32 {
            if (imm12 & 0x800) return static_cast<i32>(static_cast<i16>(imm12 | 0xF000));
            return static_cast<i32>(imm12);
        };

        // ---------------------------------------------------------------
        // Prologue
        // ---------------------------------------------------------------
        // Save frame: X0=ctx, X1=regfile, X2=segments
        a.Emit(a.StpPreIndex(29, 30, SP, -16));
        a.Emit(a.MovRegReg(29, SP));
        a.Emit(a.StpRegOffset(19, 20, SP, -16));
        a.Emit(a.StpRegOffset(21, 22, SP, -32));
        a.Emit(a.StpRegOffset(23, 24, SP, -48));
        a.Emit(a.StpRegOffset(25, 26, SP, -64));
        a.Emit(a.StpRegOffset(27, 28, SP, -80));
        a.Emit(a.SubImm(SP, SP, 96));

        // Save input parameters into callee-saved regs
        a.Emit(a.MovRegReg(19, 1)); // X19 = regfile (X1)
        a.Emit(a.MovRegReg(20, 2)); // X20 = segments (X2)
        a.Emit(a.MovRegReg(21, 0)); // X21 = ctx (X0)

        // ---------------------------------------------------------------
        // Translate bytecode
        // ---------------------------------------------------------------
        struct BranchPatch {
            sz codeSlot;
            sz bcTarget;
        };
        std::vector<BranchPatch> patches;

        sz bcIdx = 0;
        while (bcIdx < bytecodeSize) {
            u32 raw = bytecode[bcIdx];
            u8  op  = static_cast<u8>(raw & 0xFF);
            u8  rdd = static_cast<u8>((raw >> 8) & 0xF);
            u8  ra  = static_cast<u8>((raw >> 12) & 0xF);
            u8  rb  = static_cast<u8>((raw >> 16) & 0xF);
            u16 imm12 = static_cast<u16>((raw >> 20) & 0xFFF);

            sz instrSlot = a.CurrentOffset;

            switch (static_cast<Opcode>(op)) {

            case Opcode::NOP:
                a.Emit(a.Nop());
                ++bcIdx;
                break;

            case Opcode::HALT:
                goto epilogue;

            case Opcode::MOV:
                {
                    i64 val = immAsRel(imm12);
                    u16 lo  = static_cast<u16>(val & 0xFFFF);
                    u16 hi  = static_cast<u16>((val >> 16) & 0xFFFF);
                    u16 e32 = static_cast<u16>((val >> 32) & 0xFFFF);
                    u16 e48 = static_cast<u16>((val >> 48) & 0xFFFF);
                    a.Emit(a.MovzRegImm(22, lo, 0));
                    if (hi)  a.Emit(a.MovkRegImm(22, hi, 16));
                    if (e32) a.Emit(a.MovkRegImm(22, e32, 32));
                    if (e48) a.Emit(a.MovkRegImm(22, e48, 48));
                    storeScalar(rdd, 22);
                    ++bcIdx;
                }
                break;

            case Opcode::MOVR:
                loadScalar(ra, 22);
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::ADDI:
                loadScalar(ra, 22);
                {
                    i32 imm = immAsRel(imm12);
                    if (imm >= 0 && imm < 4096) {
                        a.Emit(a.AddImm(22, 22, static_cast<u16>(imm)));
                    } else {
                        u16 lo = static_cast<u16>(imm & 0xFFFF);
                        a.Emit(a.MovzRegImm(23, lo, 0));
                        if ((imm >> 16) & 0xFFFF) a.Emit(a.MovkRegImm(23, static_cast<u16>((imm >> 16) & 0xFFFF), 16));
                        a.Emit(a.AddRegRegReg(22, 22, 23));
                    }
                    storeScalar(rdd, 22);
                }
                ++bcIdx;
                break;

            case Opcode::SUBI:
                loadScalar(ra, 22);
                {
                    i32 imm = immAsRel(imm12);
                    if (imm >= 0 && imm < 4096) {
                        a.Emit(a.SubImm(22, 22, static_cast<u16>(imm)));
                    } else {
                        u16 lo = static_cast<u16>(imm & 0xFFFF);
                        a.Emit(a.MovzRegImm(23, lo, 0));
                        if ((imm >> 16) & 0xFFFF) a.Emit(a.MovkRegImm(23, static_cast<u16>((imm >> 16) & 0xFFFF), 16));
                        a.Emit(a.SubRegRegReg(22, 22, 23));
                    }
                    storeScalar(rdd, 22);
                }
                ++bcIdx;
                break;

            case Opcode::ADD:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.AddRegRegReg(22, 22, 23));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::SUB:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.SubRegRegReg(22, 22, 23));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::MUL:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.MulRegRegReg(22, 22, 23));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::DIV:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.SDivRegRegReg(22, 22, 23));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::MOD:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.SDivRegRegReg(24, 22, 23));
                a.Emit(a.MulRegRegReg(24, 24, 23));
                a.Emit(a.SubRegRegReg(22, 22, 24));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::NEG:
                loadScalar(ra, 22);
                a.Emit(a.SubRegRegReg(22, XZR, 22));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::AND:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.AndV16B(0, 0, 0)); // placeholder — use scalar AND
                {
                    // Use scalar AND on X registers via vector or via AND (shifted register)
                    a.Emit(0x8A000000u | ((23 & 0x1F) << 16) | ((22 & 0x1F) << 5) | (22 & 0x1F));
                }
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::OR:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(0xAA000000u | ((23 & 0x1F) << 16) | ((22 & 0x1F) << 5) | (22 & 0x1F));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::XOR:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(0xCA000000u | ((23 & 0x1F) << 16) | ((22 & 0x1F) << 5) | (22 & 0x1F));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::NOT:
                loadScalar(ra, 22);
                a.Emit(0xAA2003E0u | ((22 & 0x1F) << 16) | (22 & 0x1F));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::CMP:
                loadScalar(ra, 22);
                loadScalar(rb, 23);
                a.Emit(a.CmpRegReg(22, 23));
                ++bcIdx;
                break;

            case Opcode::JMP:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BImm26(0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JZ:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::EQ), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JNZ:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::NE), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JL:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::LT), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JLE:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::LE), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JG:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::GT), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::JGE:
                {
                    i32 target = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.Emit(a.BcondImm19(static_cast<u8>(Cond::GE), 0));
                    patches.push_back({a.CurrentOffset - 1, static_cast<sz>(target)});
                    ++bcIdx;
                }
                break;

            case Opcode::RET:
                goto epilogue;

            // -----------------------------------------------------------
            // Vector I/O
            // -----------------------------------------------------------

            case Opcode::VLOAD:
                {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.Emit(a.LdrRegOffset(22, 20, static_cast<i16>(static_cast<sz>(segId) * 8)));
                    a.Emit(a.LdrRegOffset(23, 19, static_cast<i16>(static_cast<sz>(kScalarRegOffset + ra * 8))));
                    a.Emit(a.AddRegRegReg(22, 22, 23));
                    a.Emit(a.LdrQOffset(static_cast<u8>(16 + (rdd & 7)), 22, 0));
                    if (count > 4) {
                        a.Emit(a.LdrQOffset(static_cast<u8>(16 + ((rdd + 1) & 7)), 22, 16));
                    }
                    ++bcIdx;
                }
                break;

            case Opcode::VSTORE:
                {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.Emit(a.LdrRegOffset(22, 20, static_cast<i16>(static_cast<sz>(segId) * 8)));
                    a.Emit(a.LdrRegOffset(23, 19, static_cast<i16>(static_cast<sz>(kScalarRegOffset + ra * 8))));
                    a.Emit(a.AddRegRegReg(22, 22, 23));
                    a.Emit(a.StrQOffset(static_cast<u8>(16 + (rdd & 7)), 22, 0));
                    if (count > 4) {
                        a.Emit(a.StrQOffset(static_cast<u8>(16 + ((rdd + 1) & 7)), 22, 16));
                    }
                    ++bcIdx;
                }
                break;

            // -----------------------------------------------------------
            // Vector Arithmetic
            // -----------------------------------------------------------

            case Opcode::VADD:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FaddV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VSUB:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FsubV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VMUL:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FmulV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VDIV:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FdivV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VMIN:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FminV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VMAX:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.FmaxV2D(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            // -----------------------------------------------------------
            // Vector Filter
            // -----------------------------------------------------------

            case Opcode::VFILTER:
                {
                    u8 mode = imm12 & 0x7;
                    loadVector(ra, 0);
                    loadScalar(rb, 22);
                    a.Emit(a.DupV2D(1, 22));

                    switch (mode) {
                    case 0: a.Emit(a.FcmeqV2D(2, 0, 1)); break;
                    case 1: // NE via !EQ
                        a.Emit(a.FcmeqV2D(2, 0, 1));
                        a.Emit(a.NotV16B(2, 2));
                        break;
                    case 2: a.Emit(a.FcmgtV2D(2, 1, 0)); break;
                    case 3: a.Emit(a.FcmgeV2D(2, 0, 1)); break;
                    case 4: a.Emit(a.FcmgtV2D(2, 0, 1)); break;
                    case 5: a.Emit(a.FcmgeV2D(2, 0, 1)); break;
                    default: a.Emit(a.FcmeqV2D(2, 0, 1)); break;
                    }
                    a.Emit(a.BslV16B(0, 0, 2));
                    storeVector(rdd, 0);
                    ++bcIdx;
                }
                break;

            case Opcode::VFILTER_EQ:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmeqV2D(2, 0, 1));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VFILTER_NE:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmeqV2D(2, 0, 1));
                a.Emit(a.NotV16B(2, 2));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VFILTER_LT:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmgtV2D(2, 1, 0));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VFILTER_LE:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmgeV2D(2, 0, 1));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VFILTER_GT:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmgtV2D(2, 0, 1));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VFILTER_GE:
                loadVector(ra, 0);
                loadScalar(rb, 22);
                a.Emit(a.DupV2D(1, 22));
                a.Emit(a.FcmgeV2D(2, 0, 1));
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VBLEND:
                loadVector(ra, 0);
                loadVector(rb, 1);
                loadVector(0, 2); // mask in Q2
                a.Emit(a.BslV16B(0, 0, 2));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            // -----------------------------------------------------------
            // Vector Reduction
            // -----------------------------------------------------------

            case Opcode::VSUM:
                loadVector(ra, 0);
                a.Emit(a.FaddpV2D(1, 0, 0));
                a.Emit(a.MovRegReg(22, XZR));
                a.Emit(a.LdrRegOffset(22, 22, 0));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::VRED_MIN:
                loadVector(ra, 0);
                a.Emit(a.FminV2D(1, 0, 0));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            case Opcode::VRED_MAX:
                loadVector(ra, 0);
                a.Emit(a.FmaxV2D(1, 0, 0));
                storeScalar(rdd, 22);
                ++bcIdx;
                break;

            // -----------------------------------------------------------
            // Vector Logical
            // -----------------------------------------------------------

            case Opcode::VAND:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.AndV16B(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VOR:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.OrrV16B(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VXOR:
                loadVector(ra, 0);
                loadVector(rb, 1);
                a.Emit(a.EorV16B(0, 0, 1));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            case Opcode::VNOT:
                loadVector(ra, 0);
                a.Emit(a.NotV16B(0, 0));
                storeVector(rdd, 0);
                ++bcIdx;
                break;

            // -----------------------------------------------------------
            // Default
            // -----------------------------------------------------------

            default:
                ++bcIdx;
                break;
            }
        }

        // ---------------------------------------------------------------
        // Epilogue
        // ---------------------------------------------------------------
    epilogue:
        a.Emit(a.AddImm(SP, SP, 96));
        a.Emit(a.LdpRegOffset(27, 28, SP, -80));
        a.Emit(a.LdpRegOffset(25, 26, SP, -64));
        a.Emit(a.LdpRegOffset(23, 24, SP, -48));
        a.Emit(a.LdpRegOffset(21, 22, SP, -32));
        a.Emit(a.LdpRegOffset(19, 20, SP, -16));
        a.Emit(a.MovRegReg(SP, 29));
        a.Emit(a.LdpPostIndex(29, 30, SP, 16));
        a.Emit(a.Ret());

        // ---------------------------------------------------------------
        // Back-patch branches
        // ---------------------------------------------------------------
        for (auto& p : patches) {
            a.PatchBranch(p.codeSlot, static_cast<i32>(p.bcTarget));
        }

        // ---------------------------------------------------------------
        // Copy to executable memory
        // ---------------------------------------------------------------
        JitMemoryManager mm;
        sz codeBytes = code.size() * 4;
        void* execMem = mm.Allocate(codeBytes);
        if (!execMem) return false;

        std::memcpy(execMem, code.data(), codeBytes);
        mm.MakeExecutable(execMem, codeBytes);

        out.CodePtr  = execMem;
        out.CodeSize = codeBytes;
        out.Entry    = reinterpret_cast<void(*)(void*, void*, void*)>(execMem);
        return true;
    }

    void Release(JitFunction& func) override
    {
        if (func.CodePtr) {
            JitMemoryManager mm;
            mm.Deallocate(func.CodePtr, func.CodeSize);
            func.CodePtr  = nullptr;
            func.CodeSize = 0;
            func.Entry    = nullptr;
        }
    }
};

} // namespace arm64
} // namespace codegen
} // namespace voxel
