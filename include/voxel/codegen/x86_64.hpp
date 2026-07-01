#pragma once

#include "voxel/codegen/jit.hpp"
#include <vector>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <bitset>

namespace voxel {
namespace codegen {
namespace x64 {

// ============================================================================
// Register enumerations
// ============================================================================

enum class Reg64 : u8 {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8  = 8, R9  = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

enum class Reg32 : u8 {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3,
    ESP = 4, EBP = 5, ESI = 6, EDI = 7,
    R8D = 8, R9D = 9, R10D = 10, R11D = 11,
    R12D = 12, R13D = 13, R14D = 14, R15D = 15,
};

enum class Reg16 : u8 {
    AX = 0, CX = 1, DX = 2, BX = 3,
    SP = 4, BP = 5, SI = 6, DI = 7,
    R8W = 8, R9W = 9, R10W = 10, R11W = 11,
    R12W = 12, R13W = 13, R14W = 14, R15W = 15,
};

enum class Reg8 : u8 {
    AL = 0,  CL = 1,  DL = 2,  BL = 3,
    AH = 4,  CH = 5,  DH = 6,  BH = 7,
    SPL = 4, BPL = 5, SIL = 6, DIL = 7,
    R8B = 8, R9B = 9, R10B = 10, R11B = 11,
    R12B = 12, R13B = 13, R14B = 14, R15B = 15,
};

enum class Xmm : u8 {
    XMM0 = 0,  XMM1 = 1,  XMM2 = 2,  XMM3 = 3,
    XMM4 = 4,  XMM5 = 5,  XMM6 = 6,  XMM7 = 7,
    XMM8 = 8,  XMM9 = 9,  XMM10 = 10, XMM11 = 11,
    XMM12 = 12, XMM13 = 13, XMM14 = 14, XMM15 = 15,
};

enum class Ymm : u8 {
    YMM0 = 0,  YMM1 = 1,  YMM2 = 2,  YMM3 = 3,
    YMM4 = 4,  YMM5 = 5,  YMM6 = 6,  YMM7 = 7,
    YMM8 = 8,  YMM9 = 9,  YMM10 = 10, YMM11 = 11,
    YMM12 = 12, YMM13 = 13, YMM14 = 14, YMM15 = 15,
};

enum class Zmm : u8 {
    ZMM0 = 0,  ZMM1 = 1,  ZMM2 = 2,  ZMM3 = 3,
    ZMM4 = 4,  ZMM5 = 5,  ZMM6 = 6,  ZMM7 = 7,
    ZMM8 = 8,  ZMM9 = 9,  ZMM10 = 10, ZMM11 = 11,
    ZMM12 = 12, ZMM13 = 13, ZMM14 = 14, ZMM15 = 15,
    ZMM16 = 16, ZMM17 = 17, ZMM18 = 18, ZMM19 = 19,
    ZMM20 = 20, ZMM21 = 21, ZMM22 = 22, ZMM23 = 23,
    ZMM24 = 24, ZMM25 = 25, ZMM26 = 26, ZMM27 = 27,
    ZMM28 = 28, ZMM29 = 29, ZMM30 = 30, ZMM31 = 31,
};

// ============================================================================
// X64Assembler — machine code emitter for x86-64
// ============================================================================

class X64Assembler {
public:
    std::vector<u8>& Code;
    sz               CurrentOffset;

    explicit X64Assembler(std::vector<u8>& code)
        : Code(code)
        , CurrentOffset(0)
    {}

    X64Assembler(const X64Assembler&)            = delete;
    X64Assembler& operator=(const X64Assembler&) = delete;

    // ----------------------------------------------------------------
    // Low-level byte emission
    // ----------------------------------------------------------------

    void EmitByte(u8 b)
    {
        Code.push_back(b);
        ++CurrentOffset;
    }

    void EmitWord(u16 v)
    {
        Code.push_back(static_cast<u8>(v & 0xFF));
        Code.push_back(static_cast<u8>((v >> 8) & 0xFF));
        CurrentOffset += 2;
    }

    void EmitDWord(u32 v)
    {
        EmitByte(static_cast<u8>(v & 0xFF));
        EmitByte(static_cast<u8>((v >> 8) & 0xFF));
        EmitByte(static_cast<u8>((v >> 16) & 0xFF));
        EmitByte(static_cast<u8>((v >> 24) & 0xFF));
    }

    void EmitQWord(u64 v)
    {
        EmitDWord(static_cast<u32>(v & 0xFFFFFFFFull));
        EmitDWord(static_cast<u32>((v >> 32) & 0xFFFFFFFFull));
    }

    void EmitBytes(const u8* data, sz len)
    {
        for (sz i = 0; i < len; ++i) EmitByte(data[i]);
    }

    // ----------------------------------------------------------------
    // REX prefix emission
    // ----------------------------------------------------------------

    static constexpr u8 REX_W  = 0x48;
    static constexpr u8 REX_R  = 0x44;
    static constexpr u8 REX_X  = 0x42;
    static constexpr u8 REX_B  = 0x41;

    void EmitREX(bool w, bool r, bool x, bool b)
    {
        u8 v = 0x40 | (w ? 8 : 0) | (r ? 4 : 0) | (x ? 2 : 0) | (b ? 1 : 0);
        if (v != 0x40) EmitByte(v);
    }

    // ----------------------------------------------------------------
    // ModRM and SIB byte emission
    // ----------------------------------------------------------------

    static constexpr u8 ModRM(u8 mod, u8 reg, u8 rm)
    {
        return static_cast<u8>(((mod & 3) << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    void EmitModRM(u8 mod, u8 reg, u8 rm) { EmitByte(ModRM(mod, reg, rm)); }

    static constexpr u8 SIB(u8 scale, u8 index, u8 base)
    {
        return static_cast<u8>(((scale & 3) << 6) | ((index & 7) << 3) | (base & 7));
    }

    void EmitSIB(u8 scale, u8 index, u8 base) { EmitByte(SIB(scale, index, base)); }

    // ----------------------------------------------------------------
    // VEX prefix emission
    // ----------------------------------------------------------------

    void EmitVEX2(u8 pp, u8 l, u8 vvvvReg, bool r)
    {
        u8 vvvv = static_cast<u8>((~static_cast<u8>(vvvvReg)) & 0xF);
        EmitByte(0xC5);
        EmitByte(static_cast<u8>((r ? 0x80 : 0x00) | (vvvv << 3) | ((l & 1) << 2) | (pp & 3)));
    }

    void EmitVEX3(u8 pp, u8 l, u8 vvvvReg, bool rExt, bool xExt, bool bExt, u8 mmmmm, bool w = true)
    {
        u8 vvvv = static_cast<u8>((~static_cast<u8>(vvvvReg)) & 0xF);
        EmitByte(0xC4);
        EmitByte(static_cast<u8>((rExt ? 0x80 : 0x00) | (xExt ? 0x40 : 0x00) | (bExt ? 0x20 : 0x00) | (mmmmm & 0x1F)));
        EmitByte(static_cast<u8>((w ? 0x80 : 0x00) | (vvvv << 3) | ((l & 1) << 2) | (pp & 3)));
    }

    void EmitEVEX(u8 pp, u8 ll, u8 vvvvReg, bool rExt, bool xExt, bool bExt, bool rp, u8 mmmm, bool z, u8 aaa, u8 vp)
    {
        u8 vvvv = static_cast<u8>((~static_cast<u8>(vvvvReg)) & 0xF);
        EmitByte(0x62);
        EmitByte(static_cast<u8>((rExt ? 0x00 : 0x80) | (xExt ? 0x00 : 0x40) | (bExt ? 0x00 : 0x20) | (rp ? 0x00 : 0x10) | (mmmm & 3)));
        EmitByte(static_cast<u8>(0x80 | (vvvv << 3) | 0x04 | (pp & 3)));
        EmitByte(static_cast<u8>((z ? 0x80 : 0x00) | ((ll & 3) << 5) | 0x10 | ((vp & 0xF) << 3) | (aaa & 7)));
    }

    // ----------------------------------------------------------------
    // MOV — move instructions
    // ----------------------------------------------------------------

    void MovRegImm(u8 reg64, u64 imm)
    {
        EmitREX(true, false, false, (reg64 >= 8));
        EmitByte(static_cast<u8>(0xB8 + (reg64 & 7)));
        EmitQWord(imm);
    }

    void MovRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (dst >= 8), false, (src >= 8));
        EmitByte(0x8B);
        EmitModRM(3, dst & 7, src & 7);
    }

    void MovRegReg32(u8 dst, u8 src)
    {
        if (dst >= 8 || src >= 8) EmitREX(false, (dst >= 8), false, (src >= 8));
        EmitByte(0x8B);
        EmitModRM(3, dst & 7, src & 7);
    }

    void MovRegMem(u8 reg, u8 base, i32 disp)
    {
        EmitREX(true, (reg >= 8), false, (base >= 8));
        EmitByte(0x8B);
        if (disp == 0 && (base & 7) != 5) {
            EmitModRM(0, reg & 7, base & 7);
        } else if (disp >= -128 && disp <= 127) {
            EmitModRM(1, reg & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, reg & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void MovMemReg(u8 base, i32 disp, u8 reg)
    {
        EmitREX(true, (reg >= 8), false, (base >= 8));
        EmitByte(0x89);
        if (disp == 0 && (base & 7) != 5) {
            EmitModRM(0, reg & 7, base & 7);
        } else if (disp >= -128 && disp <= 127) {
            EmitModRM(1, reg & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, reg & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void MovRegMemIndex(u8 reg, u8 base, u8 index, u8 scale, i32 disp)
    {
        EmitREX(true, (reg >= 8), (index >= 8), (base >= 8));
        EmitByte(0x8B);
        if (disp == 0 && (base & 7) != 5) {
            EmitModRM(0, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
        } else if (disp >= -128 && disp <= 127) {
            EmitModRM(1, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void MovMemRegIndex(u8 base, u8 index, u8 scale, i32 disp, u8 reg)
    {
        EmitREX(true, (reg >= 8), (index >= 8), (base >= 8));
        EmitByte(0x89);
        if (disp == 0 && (base & 7) != 5) {
            EmitModRM(0, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
        } else if (disp >= -128 && disp <= 127) {
            EmitModRM(1, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, reg & 7, 4);
            EmitSIB(scale, index & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void LeaRegMem(u8 reg, u8 base, i32 disp)
    {
        EmitREX(true, (reg >= 8), false, (base >= 8));
        EmitByte(0x8D);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, reg & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, reg & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    // ----------------------------------------------------------------
    // SSE MOV instructions
    // ----------------------------------------------------------------

    void MovapsXmmXmm(u8 dst, u8 src)
    {
        if (dst >= 8 || src >= 8) EmitREX(false, (dst >= 8), false, (src >= 8));
        EmitByte(0x0F);
        EmitByte(0x28);
        EmitModRM(3, dst & 7, src & 7);
    }

    void MovapsXmmMem(u8 xmm, u8 base, i32 disp)
    {
        bool ext = (xmm >= 8) || (base >= 8);
        if (ext) EmitREX(false, (xmm >= 8), false, (base >= 8));
        EmitByte(0x0F);
        EmitByte(0x28);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, xmm & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, xmm & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void MovapsMemXmm(u8 base, i32 disp, u8 xmm)
    {
        bool ext = (xmm >= 8) || (base >= 8);
        if (ext) EmitREX(false, (xmm >= 8), false, (base >= 8));
        EmitByte(0x0F);
        EmitByte(0x29);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, xmm & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, xmm & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    // ----------------------------------------------------------------
    // AVX MOV instructions
    // ----------------------------------------------------------------

    void VmovapdYmmYmm(u8 dst, u8 src)
    {
        if (dst >= 8 || src >= 8) {
            EmitVEX3(1, 1, 0, (dst < 8), true, (src < 8), 1);
        } else {
            EmitVEX2(1, 1, 0, true);
        }
        EmitByte(0x29);
        EmitModRM(3, dst & 7, src & 7);
    }

    void VmovupdYmmMem(u8 ymm, u8 base, i32 disp)
    {
        bool re = (ymm < 8);
        bool be = (base < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, 0, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, 0, true);
        }
        EmitByte(0x10);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, ymm & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, ymm & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void VmovupdYmmMemSib(u8 ymm, u8 base, u8 index, u8 scale)
    {
        EmitVEX2(1, 1, 0, (ymm < 8));
        EmitByte(0x10);
        EmitModRM(0, ymm & 7, 4);
        EmitSIB(scale & 3, index & 7, base & 7);
    }

    void VmovupdYmmMemSibDisp(u8 ymm, u8 base, u8 index, u8 scale, i8 disp)
    {
        EmitVEX2(1, 1, 0, (ymm < 8));
        EmitByte(0x10);
        EmitModRM(1, ymm & 7, 4);
        EmitSIB(scale & 3, index & 7, base & 7);
        EmitByte(static_cast<u8>(disp));
    }

    void VmovupdMemYmm(u8 base, i32 disp, u8 ymm)
    {
        bool re = (ymm < 8);
        bool be = (base < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, 0, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, 0, true);
        }
        EmitByte(0x11);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, ymm & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, ymm & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    // ----------------------------------------------------------------
    // Scalar arithmetic — ADD, SUB, IMul, IDiv
    // ----------------------------------------------------------------

    void AddRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (src >= 8), false, (dst >= 8));
        EmitByte(0x01);
        EmitModRM(3, src & 7, dst & 7);
    }

    void SubRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (src >= 8), false, (dst >= 8));
        EmitByte(0x29);
        EmitModRM(3, src & 7, dst & 7);
    }

    void ImulRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (dst >= 8), false, (src >= 8));
        EmitByte(0x0F);
        EmitByte(0xAF);
        EmitModRM(3, dst & 7, src & 7);
    }

    void IdivReg(u8 divisor)
    {
        EmitREX(true, false, false, false);
        EmitByte(0x99); // CQO — sign-extend RAX into RDX:RAX
        EmitREX(true, false, false, (divisor >= 8));
        EmitByte(0xF7);
        EmitModRM(3, 7, divisor & 7);
    }

    void AddRegImm(u8 reg, i32 imm)
    {
        if (imm >= -128 && imm <= 127) {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x83);
            EmitModRM(3, 0, reg & 7);
            EmitByte(static_cast<u8>(imm & 0xFF));
        } else {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x81);
            EmitModRM(3, 0, reg & 7);
            EmitDWord(static_cast<u32>(imm));
        }
    }

    void SubRegImm(u8 reg, i32 imm)
    {
        if (imm >= -128 && imm <= 127) {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x83);
            EmitModRM(3, 5, reg & 7);
            EmitByte(static_cast<u8>(imm & 0xFF));
        } else {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x81);
            EmitModRM(3, 5, reg & 7);
            EmitDWord(static_cast<u32>(imm));
        }
    }

    void CmpRegReg(u8 a, u8 b)
    {
        EmitREX(true, (b >= 8), false, (a >= 8));
        EmitByte(0x39);
        EmitModRM(3, b & 7, a & 7);
    }

    void CmpRegImm(u8 reg, i32 imm)
    {
        if (imm >= -128 && imm <= 127) {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x83);
            EmitModRM(3, 7, reg & 7);
            EmitByte(static_cast<u8>(imm & 0xFF));
        } else {
            EmitREX(true, false, false, (reg >= 8));
            EmitByte(0x81);
            EmitModRM(3, 7, reg & 7);
            EmitDWord(static_cast<u32>(imm));
        }
    }

    void NEGReg(u8 reg)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xF7);
        EmitModRM(3, 3, reg & 7);
    }

    void ANDRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (src >= 8), false, (dst >= 8));
        EmitByte(0x21);
        EmitModRM(3, src & 7, dst & 7);
    }

    void ORRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (src >= 8), false, (dst >= 8));
        EmitByte(0x09);
        EmitModRM(3, src & 7, dst & 7);
    }

    void XORRegReg(u8 dst, u8 src)
    {
        EmitREX(true, (src >= 8), false, (dst >= 8));
        EmitByte(0x31);
        EmitModRM(3, src & 7, dst & 7);
    }

    void SHLRegCL(u8 reg)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xD3);
        EmitModRM(3, 4, reg & 7);
    }

    void SHRRegCL(u8 reg)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xD3);
        EmitModRM(3, 5, reg & 7);
    }

    void SARRegCL(u8 reg)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xD3);
        EmitModRM(3, 7, reg & 7);
    }

    void SHLRegImm(u8 reg, u8 imm)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xC1);
        EmitModRM(3, 4, reg & 7);
        EmitByte(imm);
    }

    void SHRRegImm(u8 reg, u8 imm)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xC1);
        EmitModRM(3, 5, reg & 7);
        EmitByte(imm);
    }

    void SARRegImm(u8 reg, u8 imm)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xC1);
        EmitModRM(3, 7, reg & 7);
        EmitByte(imm);
    }

    void NOTReg(u8 reg)
    {
        EmitREX(true, false, false, (reg >= 8));
        EmitByte(0xF7);
        EmitModRM(3, 2, reg & 7);
    }

    // ----------------------------------------------------------------
    // CVT instructions
    // ----------------------------------------------------------------

    void Cqo()
    {
        EmitREX(true, false, false, false);
        EmitByte(0x99);
    }

    void Cdqe()
    {
        EmitREX(true, false, false, false);
        EmitByte(0x98);
    }

    // ----------------------------------------------------------------
    // SETcc — set byte on condition
    // ----------------------------------------------------------------

    void SetZ(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x94);
        EmitModRM(3, 0, reg8 & 7);
    }

    void SetNZ(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x95);
        EmitModRM(3, 0, reg8 & 7);
    }

    void SetL(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x9C);
        EmitModRM(3, 0, reg8 & 7);
    }

    void SetLE(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x9E);
        EmitModRM(3, 0, reg8 & 7);
    }

    void SetG(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x9F);
        EmitModRM(3, 0, reg8 & 7);
    }

    void SetGE(u8 reg8)
    {
        if (reg8 >= 4 && reg8 < 8) EmitREX(false, false, false, false);
        else if (reg8 >= 8) EmitREX(false, false, false, true);
        EmitByte(0x0F);
        EmitByte(0x9D);
        EmitModRM(3, 0, reg8 & 7);
    }

    // ----------------------------------------------------------------
    // SIMD AVX/AVX2 — Floating-point double (PD) arithmetic
    // ----------------------------------------------------------------

    void Vaddpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x58);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vaddpd128(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 0, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 0, a, re);
        }
        EmitByte(0x58);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vsubpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x5C);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vmulpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x59);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vdivpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x5E);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vminpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x5D);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vmaxpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x5F);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vcmppd(u8 dst, u8 a, u8 b, u8 predicate)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xC2);
        EmitModRM(3, dst & 7, b & 7);
        EmitByte(predicate);
    }

    void Vblendvpd(u8 dst, u8 a, u8 b, u8 mask)
    {
        EmitVEX3(0x03, 1, mask, (dst < 8), true, (b < 8), 0x03);
        EmitByte(0x4B);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vhaddpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x7C);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vpermilpd(u8 dst, u8 src, u8 imm8)
    {
        EmitVEX3(0x03, 1, 0, (dst < 8), true, (src < 8), 0x03);
        EmitByte(0x05);
        EmitModRM(3, dst & 7, src & 7);
        EmitByte(imm8);
    }

    void Vbroadcastsd(u8 ymm, u8 base)
    {
        EmitVEX3(1, 1, 0, (ymm < 8), true, (base < 8), 0x02, false);
        EmitByte(0x19);
        EmitModRM(0, ymm & 7, base & 7);
    }

    // ----------------------------------------------------------------
    // SSE/AVX scalar double
    // ----------------------------------------------------------------

    void MovqXmmReg(u8 xmm, u8 reg64)
    {
        EmitByte(0x66);
        EmitREX(true, false, false, (reg64 >= 8));
        EmitByte(0x0F);
        EmitByte(0x6E);
        EmitModRM(3, xmm & 7, reg64 & 7);
    }

    void MovqRegXmm(u8 reg64, u8 xmm)
    {
        EmitByte(0x66);
        EmitREX(true, false, false, (reg64 >= 8));
        EmitByte(0x0F);
        EmitByte(0x7E);
        EmitModRM(3, xmm & 7, reg64 & 7);
    }

    void Vaddsd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(3, 0, a, re, true, be, 1);
        } else {
            EmitVEX2(3, 0, a, re);
        }
        EmitByte(0x58);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vmulsd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(3, 0, a, re, true, be, 1);
        } else {
            EmitVEX2(3, 0, a, re);
        }
        EmitByte(0x59);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vminsd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(3, 0, a, re, true, be, 1);
        } else {
            EmitVEX2(3, 0, a, re);
        }
        EmitByte(0x5D);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vmaxsd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(3, 0, a, re, true, be, 1);
        } else {
            EmitVEX2(3, 0, a, re);
        }
        EmitByte(0x5F);
        EmitModRM(3, dst & 7, b & 7);
    }

    void MovsdXmmMem(u8 xmm, u8 base, i32 disp)
    {
        bool ext = (xmm >= 8) || (base >= 8);
        if (ext) EmitREX(false, (xmm >= 8), false, (base >= 8));
        EmitByte(0xF2);
        EmitByte(0x0F);
        EmitByte(0x10);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, xmm & 7, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, xmm & 7, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    void Vperm2f128(u8 dst, u8 a, u8 b, u8 imm8)
    {
        EmitVEX3(1, 1, a, (dst < 8), true, (b < 8), 0x03, false);
        EmitByte(0x06);
        EmitModRM(3, dst & 7, b & 7);
        EmitByte(imm8);
    }

    void Vextractf128(u8 xdst, u8 ysrc, u8 imm8)
    {
        // VEX.256.66.0F3A.W0 19 /r ib
        // ModRM.reg = ysrc (source ymm, read), ModRM.rm = xdst (dest xmm/m128, written)
        // R-ext tracks the reg operand (ysrc); B-ext tracks the rm operand (xdst).
        EmitVEX3(1, 1, 0, (ysrc < 8), true, (xdst < 8), 0x03, false);
        EmitByte(0x19);
        EmitModRM(3, ysrc & 7, xdst & 7);
        EmitByte(imm8);
    }

    void Vmovhlps(u8 dst, u8 a, u8 b)
    {
        // VEX.NDS.128.0F.WIG 12 /r
        // ModRM.reg = dst, vvvv = a (src1), ModRM.rm = b (src2)
        // No mandatory prefix (pp=0), 128-bit (l=0), map 0F (mmmmm=0x01), W ignored.
        // R-ext tracks the reg operand (dst); B-ext tracks the rm operand (b).
        EmitVEX3(0, 0, a, (dst < 8), true, (b < 8), 0x01, false);
        EmitByte(0x12);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vxorpd(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0x57);
        EmitModRM(3, dst & 7, b & 7);
    }

    // ----------------------------------------------------------------
    // AVX gather
    // ----------------------------------------------------------------

    // KNOWN-BROKEN: Vgatherdpd has three confirmed VSIB encoding bugs and a
    // signature that cannot express a correct gather. Verified against llvm-mc
    // ground truth (vgatherdpd ymm0,[rax+xmm1*8],ymm2 = c4 e2 ed 92 04 c8):
    //   1. ModRM.reg uses indexYmm; must be dst.
    //   2. VEX.R tracks (indexYmm<8); must track (dst<8) since R extends ModRM.reg.
    //   3. VEX.X hardcoded true; must track (indexYmm<8) to extend the SIB vector index.
    //   4. vvvv carries dst; gather's middle operand is the write-mask (~mask).
    // A correct fix requires a new signature with a mask parameter:
    //   Vgatherdpd(u8 dst, u8 base, u8 indexYmm, u8 scale, u8 mask)
    // Deferred: zero call sites in the current emission loop (hot kernel uses
    // contiguous vmovupd, not gather). Revisit if a kernel needs gather, and
    // verify with the 32-case VSIB matrix (dst x index x base x mask x scale)
    // byte-exact against llvm-mc before use.
    void Vgatherdpd(u8 dst, u8 base, u8 indexYmm, u8 scale)
    {
        EmitVEX3(1, 1, dst, (indexYmm < 8), true, (base < 8), 0x02);
        EmitByte(0x92);
        EmitModRM(0, indexYmm & 7, base & 7);
        EmitSIB(scale & 3, indexYmm & 7, base & 7);
    }

    // ----------------------------------------------------------------
    // AVX2 Integer SIMD
    // ----------------------------------------------------------------

    void Vpaddq(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xD4);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vpsubq(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xFB);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vpand(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xDB);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vpor(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xEB);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vpxor(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(1, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(1, 1, a, re);
        }
        EmitByte(0xEF);
        EmitModRM(3, dst & 7, b & 7);
    }

    // ----------------------------------------------------------------
    // AVX/SSE PS (single-precision float) variants
    // ----------------------------------------------------------------

    void Vaddps(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(0, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(0, 1, a, re);
        }
        EmitByte(0x58);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vsubps(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(0, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(0, 1, a, re);
        }
        EmitByte(0x5C);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vmulps(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(0, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(0, 1, a, re);
        }
        EmitByte(0x59);
        EmitModRM(3, dst & 7, b & 7);
    }

    void Vdivps(u8 dst, u8 a, u8 b)
    {
        bool re = (dst < 8), be = (b < 8);
        if (!re || !be) {
            EmitVEX3(0, 1, a, re, true, be, 1);
        } else {
            EmitVEX2(0, 1, a, re);
        }
        EmitByte(0x5E);
        EmitModRM(3, dst & 7, b & 7);
    }

    // ----------------------------------------------------------------
    // Conditional branches — rel32
    // ----------------------------------------------------------------

    sz Label()
    {
        return CurrentOffset;
    }

    void JmpRel32(i32 offset)
    {
        EmitByte(0xE9);
        EmitDWord(static_cast<u32>(offset));
    }

    void JzRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x84);
        EmitDWord(static_cast<u32>(offset));
    }

    void JnzRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x85);
        EmitDWord(static_cast<u32>(offset));
    }

    void JlRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x8C);
        EmitDWord(static_cast<u32>(offset));
    }

    void JgRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x8F);
        EmitDWord(static_cast<u32>(offset));
    }

    void JleRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x8E);
        EmitDWord(static_cast<u32>(offset));
    }

    void JgeRel32(i32 offset)
    {
        EmitByte(0x0F);
        EmitByte(0x8D);
        EmitDWord(static_cast<u32>(offset));
    }

    void PatchBranch(sz branchOffset, i32 targetOffset)
    {
        i32 rel = targetOffset - static_cast<i32>(branchOffset) - 4;
        u8* dst = &Code[branchOffset];
        dst[0] = static_cast<u8>(rel & 0xFF);
        dst[1] = static_cast<u8>((rel >> 8) & 0xFF);
        dst[2] = static_cast<u8>((rel >> 16) & 0xFF);
        dst[3] = static_cast<u8>((rel >> 24) & 0xFF);
    }

    void CallRel32(i32 offset)
    {
        EmitByte(0xE8);
        EmitDWord(static_cast<u32>(offset));
    }

    // ----------------------------------------------------------------
    // Stack frame — PUSH, POP, SUB/ADD RSP, RET
    // ----------------------------------------------------------------

    void PushReg(u8 reg)
    {
        if (reg >= 8) {
            EmitByte(0x41);
            EmitByte(static_cast<u8>(0x50 + (reg & 7)));
        } else {
            EmitByte(static_cast<u8>(0x50 + reg));
        }
    }

    void PopReg(u8 reg)
    {
        if (reg >= 8) {
            EmitByte(0x41);
            EmitByte(static_cast<u8>(0x58 + (reg & 7)));
        } else {
            EmitByte(static_cast<u8>(0x58 + reg));
        }
    }

    void SubRspImm(i32 imm)
    {
        if (imm >= -128 && imm <= 127) {
            EmitREX(true, false, false, false);
            EmitByte(0x83);
            EmitModRM(3, 5, 4);
            EmitByte(static_cast<u8>(imm & 0xFF));
        } else {
            EmitREX(true, false, false, false);
            EmitByte(0x81);
            EmitModRM(3, 5, 4);
            EmitDWord(static_cast<u32>(imm));
        }
    }

    void AddRspImm(i32 imm)
    {
        if (imm >= -128 && imm <= 127) {
            EmitREX(true, false, false, false);
            EmitByte(0x83);
            EmitModRM(3, 0, 4);
            EmitByte(static_cast<u8>(imm & 0xFF));
        } else {
            EmitREX(true, false, false, false);
            EmitByte(0x81);
            EmitModRM(3, 0, 4);
            EmitDWord(static_cast<u32>(imm));
        }
    }

    void Ret()
    {
        EmitByte(0xC3);
    }

    void Nop()
    {
        EmitByte(0x90);
    }

    void PrefetchT0(u8 base, i32 disp)
    {
        EmitREX(false, false, false, (base >= 8));
        EmitByte(0x0F);
        EmitByte(0x18);
        if (disp >= -128 && disp <= 127) {
            EmitModRM(1, 1, base & 7);
            EmitByte(static_cast<u8>(disp & 0xFF));
        } else {
            EmitModRM(2, 1, base & 7);
            EmitDWord(static_cast<u32>(disp));
        }
    }

    // ----------------------------------------------------------------
    // Test instruction
    // ----------------------------------------------------------------

    void TestRegReg(u8 a, u8 b)
    {
        EmitREX(true, (b >= 8), false, (a >= 8));
        EmitByte(0x85);
        EmitModRM(3, b & 7, a & 7);
    }

    // ----------------------------------------------------------------
    // XCHG — exchange register with RAX
    // ----------------------------------------------------------------

    void XchgRaxReg(u8 reg)
    {
        EmitREX(true, (reg >= 8), false, false);
        EmitByte(static_cast<u8>(0x90 + (reg & 7)));
    }
};

// ============================================================================
// X64Compiler — translates VoxelVM bytecode to x86-64 machine code
// ============================================================================

class X64Compiler : public JitCompiler {
public:
    static constexpr sz kScalarRegOffset    = 0;
    static constexpr sz kVectorRegOffset    = 512;  // 64 scalars * 8 bytes
    static constexpr sz kVectorStride       = 32;
    static constexpr sz kFlagsOffset        = 1088; // after vectors + masks + scratch
    static constexpr sz kSegmentsPtrOffset  = 0;

    X64Compiler() = default;
    ~X64Compiler() override = default;

    bool Compile(const u32* bytecode, sz bytecodeSize, JitFunction& out) override
    {
        (void)bytecodeSize;
        std::vector<u8> code;
        code.reserve(bytecodeSize * 64);
        X64Assembler a(code);
        a.CurrentOffset = 0;

        // ------------------------------------------------------------
        // Prologue
        // ------------------------------------------------------------
        // Input: RDI = regfile*, RSI = segmentsBase**, RDX = segmentCounts*
        // R14 = regfile, R13 = segmentsBase, R12 = segmentCounts
        a.PushReg(static_cast<u8>(Reg64::RBP));
        a.MovRegReg(static_cast<u8>(Reg64::RBP), static_cast<u8>(Reg64::RSP));

        // Allocate shadow space + callee-save area
        a.SubRspImm(96);

        // Save callee-saved registers we'll use: RBX, R12-R15
        a.MovMemReg(static_cast<u8>(Reg64::RBP), -8,  static_cast<u8>(Reg64::RBX));
        a.MovMemReg(static_cast<u8>(Reg64::RBP), -16, static_cast<u8>(Reg64::R12));
        a.MovMemReg(static_cast<u8>(Reg64::RBP), -24, static_cast<u8>(Reg64::R13));
        a.MovMemReg(static_cast<u8>(Reg64::RBP), -32, static_cast<u8>(Reg64::R14));
        a.MovMemReg(static_cast<u8>(Reg64::RBP), -40, static_cast<u8>(Reg64::R15));

        // R14 = regfile (from RDI, 1st arg)
        a.MovRegReg(static_cast<u8>(Reg64::R14), static_cast<u8>(Reg64::RDI));
        // R13 = segmentsBase (from RSI, 2nd arg)
        a.MovRegReg(static_cast<u8>(Reg64::R13), static_cast<u8>(Reg64::RSI));
        // R12 = segmentCounts (from RDX, 3rd arg)
        a.MovRegReg(static_cast<u8>(Reg64::R12), static_cast<u8>(Reg64::RDX));

        // ------------------------------------------------------------
        // Branch patch support
        // ------------------------------------------------------------
        struct BranchPatch {
            sz codeOffset;
            sz bcIndex;
            BranchPatch(sz co, sz bi) : codeOffset(co), bcIndex(bi) {}
        };
        std::vector<BranchPatch> patches;
        std::vector<sz> haltPatches;

        std::unordered_map<sz, sz> bcToCode2;

        // ------------------------------------------------------------
        // Basic-block structures
        // ------------------------------------------------------------
        struct Block {
            sz start;
            sz end;
            sz branchTarget;
            sz fallthrough;
            bool isBackward;
            bool isConditional;
            std::bitset<16> liveIn;
            std::bitset<16> liveOut;
        };

        static constexpr u8 REG_NONE = 0xFF;
        static constexpr u8 kAllocGprs[] = {
            static_cast<u8>(Reg64::RBX),
            static_cast<u8>(Reg64::R8),
            static_cast<u8>(Reg64::R9),
            static_cast<u8>(Reg64::R10),
            static_cast<u8>(Reg64::R11),
            static_cast<u8>(Reg64::RDI),
            static_cast<u8>(Reg64::RSI),
        };
        static constexpr sz kNumAllocGprs = sizeof(kAllocGprs) / sizeof(kAllocGprs[0]);

        auto immAsRel = [](u16 imm12) -> i32 {
            if (imm12 & 0x800) return static_cast<i32>(static_cast<i16>(imm12 | 0xF000));
            return static_cast<i32>(imm12);
        };

        static auto isBranchOp = [](u8 op) -> bool {
            return op == 0x50 || op == 0x51 || op == 0x52 ||
                   op == 0x59 || op == 0x5A || op == 0x5B || op == 0x5C;
        };

        static auto isTerminalOp = [](u8 op) -> bool {
            return isBranchOp(op) || op == 0x01 || op == 0x5E; // HALT, RET
        };

        // ------------------------------------------------------------
        // Step 1: Basic-block detection
        // ------------------------------------------------------------
        std::unordered_set<sz> blockStarts;
        blockStarts.insert(0);

        for (sz i = 0; i < bytecodeSize; ++i) {
            u32 raw = bytecode[i];
            u8 op = static_cast<u8>(raw & 0xFF);
            u16 imm12 = static_cast<u16>((raw >> 20) & 0xFFF);

            if (isBranchOp(op)) {
                i32 rel = immAsRel(imm12);
                sz target = static_cast<sz>(static_cast<isz>(i) + rel);
                if (target < bytecodeSize) blockStarts.insert(target);
                if (i + 1 < bytecodeSize) blockStarts.insert(i + 1);
            } else if (op == 0x01 || op == 0x5E) {
                if (i + 1 < bytecodeSize) blockStarts.insert(i + 1);
            }
        }

        std::vector<Block> blocks;
        sz currentStart = 0;
        for (sz i = 0; i < bytecodeSize; ++i) {
            bool endBlock = false;
            u32 raw = bytecode[i];
            u8 op = static_cast<u8>(raw & 0xFF);

            if (isTerminalOp(op)) {
                endBlock = true;
            } else if (i + 1 < bytecodeSize && blockStarts.count(i + 1)) {
                endBlock = true;
            }

            if (endBlock) {
                Block b;
                b.start         = currentStart;
                b.end           = i;
                b.branchTarget  = static_cast<sz>(-1);
                b.fallthrough   = static_cast<sz>(-1);
                b.isBackward    = false;
                b.isConditional = false;

                op = static_cast<u8>(bytecode[i] & 0xFF);
                u16 imm12 = static_cast<u16>((bytecode[i] >> 20) & 0xFFF);

                if (op == 0x50) {
                    i32 rel = immAsRel(imm12);
                    sz target = static_cast<sz>(static_cast<isz>(i) + rel);
                    if (target < bytecodeSize) b.branchTarget = target;
                } else if (isBranchOp(op)) {
                    i32 rel = immAsRel(imm12);
                    sz target = static_cast<sz>(static_cast<isz>(i) + rel);
                    if (target < bytecodeSize) b.branchTarget = target;
                    b.isConditional = true;
                    if (i + 1 < bytecodeSize) b.fallthrough = i + 1;
                }
                if (b.branchTarget != static_cast<sz>(-1) && b.branchTarget <= i)
                    b.isBackward = true;

                blocks.push_back(b);
                currentStart = i + 1;
            }
        }
        if (currentStart < bytecodeSize) {
            Block b;
            b.start         = currentStart;
            b.end           = bytecodeSize - 1;
            b.branchTarget  = static_cast<sz>(-1);
            b.fallthrough   = static_cast<sz>(-1);
            b.isBackward    = false;
            b.isConditional = false;
            blocks.push_back(b);
        }

        // Build successor/predecessor maps
        std::vector<std::vector<sz>> successors(blocks.size());
        std::vector<std::vector<sz>> predecessors(blocks.size());
        std::unordered_map<sz, sz> bcToBlockId;

        for (sz bid = 0; bid < blocks.size(); ++bid) {
            bcToBlockId[blocks[bid].start] = bid;
        }
        for (sz bid = 0; bid < blocks.size(); ++bid) {
            const auto& b = blocks[bid];
            if (b.branchTarget != static_cast<sz>(-1)) {
                auto it = bcToBlockId.find(b.branchTarget);
                if (it != bcToBlockId.end()) {
                    successors[bid].push_back(it->second);
                    predecessors[it->second].push_back(bid);
                }
            }
            if (b.isConditional && b.fallthrough != static_cast<sz>(-1)) {
                auto it = bcToBlockId.find(b.fallthrough);
                if (it != bcToBlockId.end()) {
                    successors[bid].push_back(it->second);
                    predecessors[it->second].push_back(bid);
                }
            }
        }

        // ------------------------------------------------------------
        // Step 2: Liveness analysis (iterative dataflow)
        // ------------------------------------------------------------
        bool changed = true;
        while (changed) {
            changed = false;
            for (isz bidx = static_cast<isz>(blocks.size()) - 1; bidx >= 0; --bidx) {
                sz bid = static_cast<sz>(bidx);
                auto& b = blocks[bid];

                // recompute liveOut from successors
                std::bitset<16> newLiveOut;
                for (sz sid : successors[bid])
                    newLiveOut |= blocks[sid].liveIn;
                if (newLiveOut != b.liveOut) {
                    b.liveOut = newLiveOut;
                    changed = true;
                }

                // backward scan through block to compute liveIn
                std::bitset<16> live = b.liveOut;
                for (sz i = b.end; i >= b.start && i != static_cast<sz>(-1); --i) {
                    u32 raw = bytecode[i];
                    u8 op = static_cast<u8>(raw & 0xFF);
                    u8 rd = static_cast<u8>((raw >> 8) & 0xF);
                    u8 ra = static_cast<u8>((raw >> 12) & 0xF);
                    u8 rb = static_cast<u8>((raw >> 16) & 0xF);

                    bool scalarDef = false;
                    scalarDef = (op == 0x10 || op == 0x11 || op == 0x12 ||
                                 op == 0x20 || op == 0x21 || op == 0x22 ||
                                 op == 0x2A || op == 0xD0);

                    if (scalarDef && rd < 16) live.reset(rd);

                    bool raIsScalarUse = (op == 0x11 || op == 0x12 || op == 0x20 ||
                                          op == 0x21 || op == 0x22 || op == 0x2A ||
                                          op == 0x40 || op == 0x41 || op == 0x70 ||
                                          op == 0x71 || op == 0x78 || op == 0xD0);
                    if (raIsScalarUse && ra < 16) live.set(ra);

                    bool rbIsScalarUse = (op == 0x20 || op == 0x21 || op == 0x22 ||
                                          op == 0x2A || op == 0x40 || op == 0x41 ||
                                          op == 0xC5 || op == 0xD0); // D0=VSUM uses ra only
                    // VSUM: rd=scalar, ra=vector (not scalar use for ra, but rd is scalar def)
                    // Actually VSUM(D0): rd=scalar dest, ra=vector source
                    // VFILTER_GT(C5): rd=vector, ra=vector, rb=scalar threshold
                    if (rbIsScalarUse && rb < 16) live.set(rb);
                    // Also VLOAD/VSTORE: ra is scalar index
                    if ((op == 0x70 || op == 0x71) && ra < 16) live.set(ra);
                }

                if (live != b.liveIn) {
                    b.liveIn = live;
                    changed = true;
                }
            }
        }

        // ------------------------------------------------------------
        // Step 3: Per-block register allocation
        // ------------------------------------------------------------
        struct BlockAlloc {
            u8 gprForReg[16];
            BlockAlloc() { for (int i = 0; i < 16; ++i) gprForReg[i] = REG_NONE; }
        };
        std::vector<BlockAlloc> blockAllocs(blocks.size());

        for (sz bid = 0; bid < blocks.size(); ++bid) {
            auto& alloc = blockAllocs[bid];
            const auto& b = blocks[bid];

            std::bitset<16> need = b.liveIn | b.liveOut;
            // Also add scalar temps (written within block)
            for (sz i = b.start; i <= b.end; ++i) {
                u32 raw = bytecode[i];
                u8 op = static_cast<u8>(raw & 0xFF);
                u8 rd = static_cast<u8>((raw >> 8) & 0xF);
                if ((op == 0x10 || op == 0x11 || op == 0x12 ||
                     op == 0x20 || op == 0x21 || op == 0x22 ||
                     op == 0x2A || op == 0xD0) && rd < 16)
                    need.set(rd);
            }

            int gprIdx = 0;
            // Assign liveIn first
            for (int r = 0; r < 16; ++r) {
                if (b.liveIn.test(r) && gprIdx < static_cast<int>(kNumAllocGprs)) {
                    alloc.gprForReg[r] = kAllocGprs[gprIdx++];
                }
            }
            // Assign remaining liveOut
            for (int r = 0; r < 16; ++r) {
                if (b.liveOut.test(r) && !b.liveIn.test(r) && alloc.gprForReg[r] == REG_NONE && gprIdx < static_cast<int>(kNumAllocGprs)) {
                    alloc.gprForReg[r] = kAllocGprs[gprIdx++];
                }
            }
            // Assign other temps
            for (int r = 0; r < 16; ++r) {
                if (need.test(r) && alloc.gprForReg[r] == REG_NONE && gprIdx < static_cast<int>(kNumAllocGprs)) {
                    alloc.gprForReg[r] = kAllocGprs[gprIdx++];
                }
            }
        }

        // ------------------------------------------------------------
        // Step 4: Code emission
        // ------------------------------------------------------------
        constexpr u8 VEC_NONE = 0xFF;
        u8 ymmHolds = VEC_NONE;

        for (sz bid = 0; bid < blocks.size(); ++bid) {
            const auto& b     = blocks[bid];
            const auto& alloc = blockAllocs[bid];

            auto getScalar = [&](u8 vreg) -> u8 {
                if (vreg < 16 && alloc.gprForReg[vreg] != REG_NONE)
                    return alloc.gprForReg[vreg];
                i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
                a.MovRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off);
                return static_cast<u8>(Reg64::RAX);
            };

            auto putScalar = [&](u8 vreg, u8 srcGpr) {
                if (vreg < 16 && alloc.gprForReg[vreg] != REG_NONE) {
                    if (alloc.gprForReg[vreg] != srcGpr)
                        a.MovRegReg(alloc.gprForReg[vreg], srcGpr);
                } else {
                    i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
                    a.MovMemReg(static_cast<u8>(Reg64::R14), off, srcGpr);
                }
            };

            bool isLoopHeader = b.isBackward;
            // For loop headers: load liveIn FIRST (prologue falls through here),
            // then record bcToCode2 (backward branches jump here, skipping load).
            if (isLoopHeader) {
                for (int r = 0; r < 16; ++r) {
                    if (b.liveIn.test(r) && alloc.gprForReg[r] != REG_NONE) {
                        i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(r) * 8);
                        a.MovRegMem(alloc.gprForReg[r], static_cast<u8>(Reg64::R14), off);
                    }
                }
            }

            // Record block start for branch targets
            bcToCode2[b.start] = a.CurrentOffset;

            if (!isLoopHeader) ymmHolds = VEC_NONE;

            // For non-loop-headers: load liveIn AFTER bcToCode2 (branches enter here with load)
            if (!isLoopHeader) {
                for (int r = 0; r < 16; ++r) {
                    if (b.liveIn.test(r) && alloc.gprForReg[r] != REG_NONE) {
                        i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(r) * 8);
                        a.MovRegMem(alloc.gprForReg[r], static_cast<u8>(Reg64::R14), off);
                    }
                }
            }

            // ================================================================
            // Fusion detection: VLOAD→VFILTER_GT→VSUM→ADDF → vectorized kernel
            // ================================================================
            bool fused = false;
            {
                sz blen = b.end - b.start;
                if (blen >= 7) {
                    u32 r0=bytecode[b.start], r1=bytecode[b.start+1], r2=bytecode[b.start+2];
                    u32 r3=bytecode[b.start+3], r4=bytecode[b.start+4], r5=bytecode[b.start+5];
                    u8 o0=r0&0xFF, o1=r1&0xFF, o2=r2&0xFF, o3=r3&0xFF, o4=r4&0xFF, o5=r5&0xFF;
                    u8 v0rd=(r0>>8)&0xF, v0ra=(r0>>12)&0xF;
                    u8 v1rd=(r1>>8)&0xF, v1ra=(r1>>12)&0xF, v1rb=(r1>>16)&0xF;
                    u8 v2rd=(r2>>8)&0xF, v2ra=(r2>>12)&0xF;
                    u8 v3rd=(r3>>8)&0xF, v3rb=(r3>>16)&0xF;
                    u8 v4ra=(r4>>12)&0xF;
                    u8 v5ra=(r5>>12)&0xF;

                    bool isFilterSum = (o0==static_cast<u8>(Opcode::VLOAD) &&
                                        o1==static_cast<u8>(Opcode::VFILTER_GT) &&
                                        o2==static_cast<u8>(Opcode::VSUM) &&
                                        o3==static_cast<u8>(Opcode::ADDF) &&
                                        o4==static_cast<u8>(Opcode::ADD) &&
                                        o5==static_cast<u8>(Opcode::CMP) &&
                                        v1ra==v0rd && v2ra==v1rd && v3rb==v2rd && v4ra==v0ra &&
                                        v5ra==v0ra && b.isBackward);

                    if (isFilterSum) {
                        u8 segId = static_cast<u8>((r0 >> 28) & 0xF);
                        u8 offReg = v0ra, threshReg = v1rb, accReg = v3rd;

                        // Pre-loop: load segment base, broadcast threshold, zero accumulators
                        a.MovRegMem(static_cast<u8>(Reg64::R15), static_cast<u8>(Reg64::R13),
                                    static_cast<i32>(static_cast<sz>(segId) * 8));
                        i32 thOff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(threshReg) * 8);
                        a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), thOff);
                        a.Vbroadcastsd(4, static_cast<u8>(Reg64::RAX));
                        a.Vxorpd(5, 4, 4);
                        a.Vxorpd(6, 4, 4);

                        // Pre-load offset into R8 as byte offset, pre-compute iteration count into R9
                        u8 hOff = alloc.gprForReg[offReg];
                        if (hOff != REG_NONE) {
                            if (hOff != static_cast<u8>(Reg64::R8))
                                a.MovRegReg(static_cast<u8>(Reg64::R8), hOff);
                        } else {
                            i32 ooff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(offReg) * 8);
                            a.MovRegMem(static_cast<u8>(Reg64::R8), static_cast<u8>(Reg64::R14), ooff);
                        }
                        // R8 = element offset, now convert to byte offset and iteration count
                        a.MovRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R8));
                        a.SHLRegImm(static_cast<u8>(Reg64::R8), 3);  // offset *= 8 for SIB

                        // Compute iteration count: (total - start + 7) / 8 for 2-way unrolled
                        u8 hR2 = alloc.gprForReg[2];
                        if (hR2 != REG_NONE) {
                            a.MovRegReg(static_cast<u8>(Reg64::RCX), hR2);
                        } else {
                            i32 r2off = static_cast<i32>(kScalarRegOffset + 16);
                            a.MovRegMem(static_cast<u8>(Reg64::RCX), static_cast<u8>(Reg64::R14), r2off);
                        }
                        a.SubRegReg(static_cast<u8>(Reg64::RCX), static_cast<u8>(Reg64::RAX));
                        a.AddRegImm(static_cast<u8>(Reg64::RCX), 7);
                        a.SHRRegImm(static_cast<u8>(Reg64::RCX), 3);  // rcx = iterations

                        bcToCode2[b.start] = a.CurrentOffset;

                        // Stage 1+2+3: SIB addressing, countdown loop, 2-way unrolled
                        // Loop: 8 elements/iter, dual accumulators ymm5+ymm6
                        a.VmovupdYmmMemSib(0, static_cast<u8>(Reg64::R15),
                                            static_cast<u8>(Reg64::R8), 3);
                        a.Vcmppd(2, 0, 4, 14);
                        a.Vpand(0, 0, 2);
                        a.Vaddpd(5, 5, 0);

                        a.VmovupdYmmMemSibDisp(1, static_cast<u8>(Reg64::R15),
                                                static_cast<u8>(Reg64::R8), 3, 32);
                        a.Vcmppd(2, 1, 4, 14);
                        a.Vpand(1, 1, 2);
                        a.Vaddpd(6, 6, 1);

                        a.AddRegImm(static_cast<u8>(Reg64::R8), 64);
                        a.SubRegImm(static_cast<u8>(Reg64::RCX), 1);
                        a.JnzRel32(0);
                        patches.push_back(BranchPatch{a.CurrentOffset - 4, b.start});

                        // Post-loop: combine accumulators, horizontal reduce, flush liveOut
                        a.Vaddpd(5, 5, 6);
                        for (int r = 0; r < 16; ++r) {
                            if (b.liveOut.test(r)) {
                                u8 h = alloc.gprForReg[r];
                                if (h != REG_NONE) {
                                    i32 poff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(static_cast<sz>(r)) * 8);
                                    a.MovMemReg(static_cast<u8>(Reg64::R14), poff, h);
                                }
                            }
                        }
                        a.Vextractf128(1, 5, 1);
                        a.Vmovhlps(2, 5, 5);
                        a.Vaddsd(5, 5, 2);
                        a.Vmovhlps(2, 1, 1);
                        a.Vaddsd(1, 1, 2);
                        a.Vaddsd(5, 5, 1);
                        a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 5);
                        if (alloc.gprForReg[accReg] != REG_NONE)
                            a.MovRegReg(alloc.gprForReg[accReg], static_cast<u8>(Reg64::RAX));
                        else {
                            i32 aoff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(accReg) * 8);
                            a.MovMemReg(static_cast<u8>(Reg64::R14), aoff, static_cast<u8>(Reg64::RAX));
                        }

                        fused = true;
                    }
                }
            }

            if (fused) continue;

            // Emit instructions in the block
            for (sz i = b.start; i <= b.end; ++i) {
                u32 raw = bytecode[i];
                u8 op  = static_cast<u8>(raw & 0xFF);
                u8 rd  = static_cast<u8>((raw >> 8) & 0xF);
                u8 ra  = static_cast<u8>((raw >> 12) & 0xF);
                u8 rb  = static_cast<u8>((raw >> 16) & 0xF);
                u16 imm12 = static_cast<u16>((raw >> 20) & 0xFFF);

                switch (static_cast<Opcode>(op)) {

                case Opcode::NOP:
                    a.Nop();
                    break;

                case Opcode::MOV: {
                    i64 immVal = static_cast<i64>(immAsRel(imm12));
                    a.MovRegImm(static_cast<u8>(Reg64::RAX), static_cast<u64>(immVal));
                    putScalar(rd, static_cast<u8>(Reg64::RAX));
                    break;
                }

                case Opcode::MOVR: {
                    u8 g = getScalar(ra);
                    putScalar(rd, g);
                    break;
                }

                case Opcode::ADDI: {
                    u8 g = getScalar(ra);
                    if (g != static_cast<u8>(Reg64::RAX)) a.MovRegReg(static_cast<u8>(Reg64::RAX), g);
                    a.AddRegImm(static_cast<u8>(Reg64::RAX), immAsRel(imm12));
                    putScalar(rd, static_cast<u8>(Reg64::RAX));
                    break;
                }

                case Opcode::ADD: {
                    u8 ga = getScalar(ra);
                    u8 gb = getScalar(rb);
                    a.AddRegReg(ga, gb);
                    putScalar(rd, ga);
                    break;
                }

                case Opcode::SUB: {
                    u8 ga = getScalar(ra);
                    u8 gb = getScalar(rb);
                    a.SubRegReg(ga, gb);
                    putScalar(rd, ga);
                    break;
                }

                case Opcode::MUL: {
                    u8 ga = getScalar(ra);
                    u8 gb = getScalar(rb);
                    a.ImulRegReg(ga, gb);
                    putScalar(rd, ga);
                    break;
                }

                case Opcode::ADDF: {
                    u8 ga = getScalar(ra);
                    u8 gb = getScalar(rb);
                    a.MovqXmmReg(0, ga);
                    a.MovqXmmReg(1, gb);
                    a.Vaddsd(0, 0, 1);
                    a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                    putScalar(rd, static_cast<u8>(Reg64::RAX));
                    break;
                }

                case Opcode::CMP: {
                    u8 ga = getScalar(ra);
                    u8 gb = getScalar(rb);
                    a.CmpRegReg(ga, gb);
                    break;
                }

                case Opcode::VLOAD: {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.MovRegMem(static_cast<u8>(Reg64::R15),
                                static_cast<u8>(Reg64::R13),
                                static_cast<i32>(static_cast<sz>(segId) * 8));
                    u8 g = getScalar(ra);
                    if (g != static_cast<u8>(Reg64::RAX)) a.MovRegReg(static_cast<u8>(Reg64::RAX), g);
                    a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    a.AddRegReg(static_cast<u8>(Reg64::R15), static_cast<u8>(Reg64::RAX));
                    a.PrefetchT0(static_cast<u8>(Reg64::R15), 64);
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R15), 0);
                    ymmHolds = rd;
                    if (count > 4) {
                        a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R15), 32);
                        a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                        static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd + 1) * kVectorStride), 1);
                    }
                    break;
                }

                case Opcode::VSTORE: {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.MovRegMem(static_cast<u8>(Reg64::R15),
                                static_cast<u8>(Reg64::R13),
                                static_cast<i32>(static_cast<sz>(segId) * 8));
                    u8 g = getScalar(ra);
                    if (g != static_cast<u8>(Reg64::RAX)) a.MovRegReg(static_cast<u8>(Reg64::RAX), g);
                    a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    a.AddRegReg(static_cast<u8>(Reg64::R15), static_cast<u8>(Reg64::RAX));
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride));
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R15), 0, 0);
                    if (count > 4) {
                        a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                        static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd + 1) * kVectorStride));
                        a.VmovupdMemYmm(static_cast<u8>(Reg64::R15), 32, 1);
                    }
                    break;
                }

                case Opcode::VSPLAT: {
                    u8 g = getScalar(ra);
                    a.MovMemReg(static_cast<u8>(Reg64::RBP), -48, g);
                    a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RBP), -48);
                    a.Vbroadcastsd(0, static_cast<u8>(Reg64::RAX));
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VFILTER_GT: {
                    if (ymmHolds != ra) {
                        a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                        static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    }
                    i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(rb) * 8);
                    a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off);
                    a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                    a.Vcmppd(2, 0, 1, 14);
                    a.Vpand(0, 0, 2);
                    ymmHolds = rd;
                    break;
                }

                case Opcode::VSUM: {
                    if (ymmHolds != ra) {
                        a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                        static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    }
                    a.Vextractf128(1, 0, 1);
                    a.Vmovhlps(2, 0, 0);
                    a.Vaddsd(0, 0, 2);
                    a.Vmovhlps(2, 1, 1);
                    a.Vaddsd(1, 1, 2);
                    a.Vaddsd(0, 0, 1);
                    a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                    putScalar(rd, static_cast<u8>(Reg64::RAX));
                    ymmHolds = VEC_NONE;
                    break;
                }

                case Opcode::VADD: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vaddpd(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VSUB: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vsubpd(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VMUL: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vmulpd(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VMIN: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vminpd(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VMAX: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vmaxpd(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VAND: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vpand(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VOR: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vpor(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                case Opcode::VXOR: {
                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(ra) * kVectorStride));
                    a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rb) * kVectorStride));
                    a.Vpxor(0, 0, 1);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R14),
                                    static_cast<i32>(kVectorRegOffset + static_cast<sz>(rd) * kVectorStride), 0);
                    break;
                }

                default:
                    break;
                }
            }

            // Store liveOut values at block end (before branch/halt)
            for (int r = 0; r < 16; ++r) {
                if (b.liveOut.test(r) && alloc.gprForReg[r] != REG_NONE) {
                    i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(r) * 8);
                    a.MovMemReg(static_cast<u8>(Reg64::R14), off, alloc.gprForReg[r]);
                }
            }

            // Handle the terminal instruction of the block
            u8 termOp = static_cast<u8>(bytecode[b.end] & 0xFF);
            u16 termImm12 = static_cast<u16>((bytecode[b.end] >> 20) & 0xFFF);

            if (termOp == 0x50) {
                // JMP
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JmpRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x51) {
                // JZ
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JzRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
                // Write stop byte (JMP to fallthrough 2 bytes, will get patched)
            } else if (termOp == 0x52) {
                // JNZ
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JnzRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x59) {
                // JL
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JlRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x5A) {
                // JLE
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JleRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x5B) {
                // JG
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JgRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x5C) {
                // JGE
                i32 rel = immAsRel(termImm12);
                sz target = static_cast<sz>(static_cast<isz>(b.end) + rel);
                a.JgeRel32(0);
                patches.push_back(BranchPatch(a.CurrentOffset - 4, target));
            } else if (termOp == 0x01 || termOp == 0x5E) {
                // HALT or RET — jump to epilogue
                a.JmpRel32(0);
                haltPatches.push_back(a.CurrentOffset - 4);
            }
        }

        // ------------------------------------------------------------
        // doneTranslating: resolve patches, epilogue, JIT allocation
        // ------------------------------------------------------------
        sz epilogueStart = a.CurrentOffset;
        for (auto& p : patches) {
            auto it = bcToCode2.find(p.bcIndex);
            if (it != bcToCode2.end()) {
                a.PatchBranch(p.codeOffset, static_cast<i32>(it->second));
            }
        }
        for (sz off : haltPatches) {
            a.PatchBranch(off, static_cast<i32>(epilogueStart));
        }

        // ------------------------------------------------------------
        // Epilogue
        // ------------------------------------------------------------
        a.MovRegMem(static_cast<u8>(Reg64::RBX),
                    static_cast<u8>(Reg64::RBP), -8);
        a.MovRegMem(static_cast<u8>(Reg64::R12),
                    static_cast<u8>(Reg64::RBP), -16);
        a.MovRegMem(static_cast<u8>(Reg64::R13),
                    static_cast<u8>(Reg64::RBP), -24);
        a.MovRegMem(static_cast<u8>(Reg64::R14),
                    static_cast<u8>(Reg64::RBP), -32);
        a.MovRegMem(static_cast<u8>(Reg64::R15),
                    static_cast<u8>(Reg64::RBP), -40);
        a.MovRegReg(static_cast<u8>(Reg64::RSP), static_cast<u8>(Reg64::RBP));
        a.PopReg(static_cast<u8>(Reg64::RBP));
        a.Ret();

        // ------------------------------------------------------------
        // Allocate executable memory and copy code
        // ------------------------------------------------------------
        JitMemoryManager mm;
        sz codeLength = code.size();
        void* execMem = mm.Allocate(codeLength);
        if (!execMem) return false;

        std::memcpy(execMem, code.data(), codeLength);
        mm.MakeExecutable(execMem, codeLength);

        out.CodePtr  = execMem;
        out.CodeSize = codeLength;
        out.Entry    = reinterpret_cast<JitFunction::EntryFn>(execMem);
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

} // namespace x64
} // namespace codegen
} // namespace voxel
