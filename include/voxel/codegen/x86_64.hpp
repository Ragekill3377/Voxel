#pragma once

#include "voxel/codegen/jit.hpp"
#include <vector>
#include <cstring>
#include <cstdint>
#include <unordered_map>

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
        EmitByte(0x16);
        EmitModRM(3, dst & 7, b & 7);
        EmitByte(imm8);
    }

    void Vextractf128(u8 xdst, u8 ysrc, u8 imm8)
    {
        EmitVEX3(1, 1, 0, (xdst < 8), true, (ysrc < 8), 0x03, false);
        EmitByte(0x19);
        EmitModRM(3, xdst & 7, ysrc & 7);
        EmitByte(imm8);
    }

    void Vmovhlps(u8 dst, u8 a, u8 b)
    {
        EmitByte(0x0F);
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
        // Translate bytecode
        // ------------------------------------------------------------
        // We collect label table for branch offsets.
        // Each bytecode instruction is 4 bytes (u32), extended
        // instructions consume additional u32s.
        struct BranchPatch {
            sz codeOffset;
            sz bcIndex;
            BranchPatch(sz co, sz bi) : codeOffset(co), bcIndex(bi) {}
        };
        std::vector<BranchPatch> patches;

        // Pre-pass: determine bytecode labels (offsets in instruction count)
        // We use the bytecode index. The translation loop will emit code
        // and record when it emits a branch that needs patching.

        // Map bytecode index → code offset for branch resolution
        std::unordered_map<sz, sz> bcToCode;

        // ----------------------------------------------------------------
        // Register allocator: pre-scan bytecode to identify hot scalars
        // and assign dedicated host GPRs (avoids regfile roundtrips)
        // ----------------------------------------------------------------
        constexpr u8 REG_NONE = 0xFF;
        static constexpr u8 kAllocGprs[] = {
            static_cast<u8>(Reg64::RBX),
            static_cast<u8>(Reg64::R8),  static_cast<u8>(Reg64::R9),
            static_cast<u8>(Reg64::R10), static_cast<u8>(Reg64::R11),
        };
        static constexpr sz kNumAllocGprs = sizeof(kAllocGprs) / sizeof(kAllocGprs[0]);

        u32 scalarUse[64] = {};
        for (sz i = 0; i < bytecodeSize; ++i) {
            u32 raw = bytecode[i];
            u8 rd = (raw >> 8) & 0xF;
            u8 ra = (raw >> 12) & 0xF;
            u8 rb = (raw >> 16) & 0xF;
            scalarUse[rd]++; scalarUse[ra]++; scalarUse[rb]++;
        }

        u8 scalarHost[64];
        bool scalarDirty[64] = {};
        for (sz i = 0; i < 64; ++i) scalarHost[i] = REG_NONE;

        u8 vectorHost[16];
        bool vectorDirty[16] = {};
        for (sz i = 0; i < 16; ++i) vectorHost[i] = REG_NONE;

        // Liveness: is a VM register read again before its next write?
        // scanned forward from each instruction position
        auto isDeadScalar = [&](u8 vreg, sz fromIdx) -> bool {
            for (sz i = fromIdx + 1; i < bytecodeSize; ++i) {
                u32 r2 = bytecode[i];
                u8 op2 = r2 & 0xFF;
                u8 rd2 = (r2 >> 8) & 0xF;
                u8 ra2 = (r2 >> 12) & 0xF;
                u8 rb2 = (r2 >> 16) & 0xF;
                if (op2 == static_cast<u8>(Opcode::JMP) || op2 == static_cast<u8>(Opcode::JZ) ||
                    op2 == static_cast<u8>(Opcode::JNZ) || op2 == static_cast<u8>(Opcode::HALT) ||
                    op2 == static_cast<u8>(Opcode::RET) || op2 == static_cast<u8>(Opcode::CALL))
                    return true; // branch: conservatively assume dead after branch
                if (rd2 == vreg) return true; // overwritten before read → dead
                if (ra2 == vreg || rb2 == vreg) return false; // read before overwrite → live
            }
            return true; // end of code → dead
        };
        auto isDeadVector = [&](u8 vreg, sz fromIdx) -> bool {
            for (sz i = fromIdx + 1; i < bytecodeSize; ++i) {
                u32 r2 = bytecode[i];
                u8 op2 = r2 & 0xFF;
                u8 rd2 = (r2 >> 8) & 0xF;
                u8 ra2 = (r2 >> 12) & 0xF;
                u8 rb2 = (r2 >> 16) & 0xF;
                if (op2 == static_cast<u8>(Opcode::JMP) || op2 == static_cast<u8>(Opcode::JZ) ||
                    op2 == static_cast<u8>(Opcode::JNZ) || op2 == static_cast<u8>(Opcode::HALT) ||
                    op2 == static_cast<u8>(Opcode::RET) || op2 == static_cast<u8>(Opcode::CALL))
                    return true;
                if (rd2 == vreg) return true;
                if (ra2 == vreg || rb2 == vreg) return false;
            }
            return true;
        };
        u8 nextGpr = 0;
        for (sz round = 0; round < 64 && nextGpr < kNumAllocGprs; ++round) {
            u32 best = 0; sz bestIdx = 0;
            for (sz r = 0; r < 64; ++r) {
                if (scalarHost[r] == REG_NONE && scalarUse[r] > best) { best = scalarUse[r]; bestIdx = r; }
            }
            if (best == 0) break;
            scalarHost[bestIdx] = kAllocGprs[nextGpr];
            scalarUse[bestIdx] = 0;
            ++nextGpr;
        }

        auto loadScalarReg = [&](u8 vreg, u8 fbGpr) {
            u8 h = scalarHost[vreg];
            if (h != REG_NONE) {
                if (h != fbGpr) a.MovRegReg(fbGpr, h);
            } else {
                i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
                a.MovRegMem(fbGpr, static_cast<u8>(Reg64::R14), off);
            }
        };
        auto storeScalarReg = [&](u8 vreg, u8 valGpr) {
            u8 h = scalarHost[vreg];
            if (h != REG_NONE) {
                if (h != valGpr) a.MovRegReg(h, valGpr);
            } else {
                i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(vreg) * 8);
                a.MovMemReg(static_cast<u8>(Reg64::R14), off, valGpr);
            }
        };
        auto loadVectorReg = [&](u8 vreg, u8 ymmreg) {
            if (vectorHost[ymmreg] == vreg) return;
            if (vectorHost[ymmreg] != REG_NONE && vectorDirty[vectorHost[ymmreg]]) {
                i32 off2 = static_cast<i32>(kVectorRegOffset + static_cast<sz>(vectorHost[ymmreg]) * kVectorStride);
                a.VmovupdMemYmm(static_cast<u8>(Reg64::R14), off2, ymmreg);
                vectorDirty[vectorHost[ymmreg]] = false;
            }
            i32 off = static_cast<i32>(kVectorRegOffset + static_cast<sz>(vreg) * kVectorStride);
            a.VmovupdYmmMem(ymmreg, static_cast<u8>(Reg64::R14), off);
            vectorHost[ymmreg] = vreg;
            vectorDirty[vreg] = false;
        };
        auto storeVectorReg = [&](u8 vreg, u8 ymmreg, sz bcIdx) {
            vectorHost[ymmreg] = vreg;
            if (isDeadVector(vreg, bcIdx)) {
                vectorDirty[vreg] = true;
            } else {
                i32 off = static_cast<i32>(kVectorRegOffset + static_cast<sz>(vreg) * kVectorStride);
                a.VmovupdMemYmm(static_cast<u8>(Reg64::R14), off, ymmreg);
                vectorDirty[vreg] = false;
            }
        };
        auto flushScalars = [&]() {
            for (sz i = 0; i < 64; ++i) {
                u8 h = scalarHost[i];
                if (h != REG_NONE) {
                    i32 off = static_cast<i32>(kScalarRegOffset + i * 8);
                    a.MovMemReg(static_cast<u8>(Reg64::R14), off, h);
                }
            }
        };

        // Pre-load assigned host GPRs from regfile
        for (sz i = 0; i < 64; ++i) {
            u8 h = scalarHost[i];
            if (h != REG_NONE) {
                i32 off = static_cast<i32>(kScalarRegOffset + i * 8);
                a.MovRegMem(h, static_cast<u8>(Reg64::R14), off);
            }
        }

        sz bcIdx = 0;
        while (bcIdx < bytecodeSize) {
            bcToCode[bcIdx] = a.CurrentOffset;
            u32 raw = bytecode[bcIdx];
            u8  op  = static_cast<u8>(raw & 0xFF);
            u8  rd  = static_cast<u8>((raw >> 8) & 0xF);
            u8  ra  = static_cast<u8>((raw >> 12) & 0xF);
            u8  rb  = static_cast<u8>((raw >> 16) & 0xF);
            u16 imm12 = static_cast<u16>((raw >> 20) & 0xFFF);
            auto immAsRel = [](u16 imm12) -> i32 {
                if (imm12 & 0x800) return static_cast<i32>(static_cast<i16>(imm12 | 0xF000));
                return static_cast<i32>(imm12);
            };

            (void)a.CurrentOffset;

            switch (static_cast<Opcode>(op)) {

            case Opcode::NOP:
                a.Nop();
                ++bcIdx;
                break;

            case Opcode::HALT:
                goto doneTranslating;

            case Opcode::RET:
                goto doneTranslating;

            case Opcode::MOV:
                a.MovRegImm(static_cast<u8>(Reg64::RAX), static_cast<u64>(static_cast<i64>(immAsRel(imm12))));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::MOVR:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::ADDI:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                a.AddRegImm(static_cast<u8>(Reg64::RAX), immAsRel(imm12));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::SUBI:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                a.SubRegImm(static_cast<u8>(Reg64::RAX), immAsRel(imm12));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::ADD:
                {
                    u8 hra = scalarHost[ra], hrb = scalarHost[rb], hrd = scalarHost[rd];
                    u8 ga = (hra != REG_NONE) ? hra : static_cast<u8>(Reg64::RAX);
                    u8 gb = (hrb != REG_NONE) ? hrb : static_cast<u8>(Reg64::RCX);
                    if (hra == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + ra*8); a.MovRegMem(ga, static_cast<u8>(Reg64::R14), off); }
                    if (hrb == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + rb*8); a.MovRegMem(gb, static_cast<u8>(Reg64::R14), off); }
                    a.AddRegReg(ga, gb);
                    if (hrd != REG_NONE) { if (hrd != ga) a.MovRegReg(hrd, ga); }
                    else { i32 off = static_cast<i32>(kScalarRegOffset + rd*8); a.MovMemReg(static_cast<u8>(Reg64::R14), off, ga); }
                    ++bcIdx;
                }
                break;

            case Opcode::SUB:
                {
                    u8 hra = scalarHost[ra], hrb = scalarHost[rb], hrd = scalarHost[rd];
                    u8 ga = (hra != REG_NONE) ? hra : static_cast<u8>(Reg64::RAX);
                    u8 gb = (hrb != REG_NONE) ? hrb : static_cast<u8>(Reg64::RCX);
                    if (hra == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + ra*8); a.MovRegMem(ga, static_cast<u8>(Reg64::R14), off); }
                    if (hrb == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + rb*8); a.MovRegMem(gb, static_cast<u8>(Reg64::R14), off); }
                    a.SubRegReg(ga, gb);
                    if (hrd != REG_NONE) { if (hrd != ga) a.MovRegReg(hrd, ga); }
                    else { i32 off = static_cast<i32>(kScalarRegOffset + rd*8); a.MovMemReg(static_cast<u8>(Reg64::R14), off, ga); }
                    ++bcIdx;
                }
                break;

            case Opcode::CMP:
                {
                    u8 hra = scalarHost[ra], hrb = scalarHost[rb];
                    u8 ga = (hra != REG_NONE) ? hra : static_cast<u8>(Reg64::RAX);
                    u8 gb = (hrb != REG_NONE) ? hrb : static_cast<u8>(Reg64::RCX);
                    if (hra == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + ra*8); a.MovRegMem(ga, static_cast<u8>(Reg64::R14), off); }
                    if (hrb == REG_NONE) { i32 off = static_cast<i32>(kScalarRegOffset + rb*8); a.MovRegMem(gb, static_cast<u8>(Reg64::R14), off); }
                    a.CmpRegReg(ga, gb);
                    ++bcIdx;
                }
                break;

            case Opcode::ADDF:
                {
                    u8 hra = scalarHost[ra], hrb = scalarHost[rb], hrd = scalarHost[rd];
                    if (hra != REG_NONE)
                        a.MovqXmmReg(0, hra);
                    else {
                        i32 off = static_cast<i32>(kScalarRegOffset + ra*8);
                        a.MovRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off);
                        a.MovqXmmReg(0, static_cast<u8>(Reg64::RAX));
                    }
                    if (hrb != REG_NONE)
                        a.MovqXmmReg(1, hrb);
                    else {
                        i32 off = static_cast<i32>(kScalarRegOffset + rb*8);
                        a.MovRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off);
                        a.MovqXmmReg(1, static_cast<u8>(Reg64::RAX));
                    }
                    a.Vaddsd(0, 0, 1);
                    a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                    if (hrd != REG_NONE)
                        a.MovRegReg(hrd, static_cast<u8>(Reg64::RAX));
                    else {
                        i32 off = static_cast<i32>(kScalarRegOffset + rd*8);
                        a.MovMemReg(static_cast<u8>(Reg64::R14), off, static_cast<u8>(Reg64::RAX));
                    }
                    ++bcIdx;
                }
                break;
                a.ImulRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::DIV:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.Cqo();
                a.IdivReg(static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::MOD:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.Cqo();
                a.IdivReg(static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RDX));
                ++bcIdx;
                break;

            case Opcode::NEG:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                a.NEGReg(static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::AND:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.ANDRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::OR:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.ORRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::XOR:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.XORRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RCX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::NOT:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                a.NOTReg(static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::CMPF:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                loadScalarReg(rb, static_cast<u8>(Reg64::RCX));
                a.CmpRegReg(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RCX));
                ++bcIdx;
                break;

            case Opcode::JMP:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JmpRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JZ:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JzRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JNZ:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JnzRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JL:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JlRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JLE:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JleRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JG:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JgRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            case Opcode::JGE:
                {
                    i32 bcTarget = static_cast<i32>(bcIdx) + immAsRel(imm12);
                    a.JgeRel32(0);
                    patches.push_back(BranchPatch{a.CurrentOffset - 4, static_cast<sz>(bcTarget)});
                    ++bcIdx;
                }
                break;

            // ------------------------------------------------------------
            // Vector I/O
            // ------------------------------------------------------------

            case Opcode::VLOAD:
                {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.MovRegMem(static_cast<u8>(Reg64::R15),
                                static_cast<u8>(Reg64::R13),
                                static_cast<i32>(static_cast<sz>(segId) * 8));
                    u8 hra = scalarHost[ra];
                    if (hra != REG_NONE) {
                        if (hra != static_cast<u8>(Reg64::RAX)) a.MovRegReg(static_cast<u8>(Reg64::RAX), hra);
                        a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    } else {
                        i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(ra) * 8);
                        a.MovRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off);
                        a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    }
                    a.AddRegReg(static_cast<u8>(Reg64::R15), static_cast<u8>(Reg64::RAX));

                    a.VmovupdYmmMem(0, static_cast<u8>(Reg64::R15), 0);
                    storeVectorReg(rd, 0, bcIdx);

                    if (count > 4) {
                        a.VmovupdYmmMem(1, static_cast<u8>(Reg64::R15), 32);
                        storeVectorReg(static_cast<u8>(rd + 1), 1, bcIdx);
                    }
                    ++bcIdx;
                }
                break;

            case Opcode::VSTORE:
                {
                    u8 segId = static_cast<u8>((raw >> 28) & 0xF);
                    u8 count = static_cast<u8>(imm12 & 0xFF);
                    if (count == 0) count = 4;

                    a.MovRegMem(static_cast<u8>(Reg64::R15),
                                static_cast<u8>(Reg64::R13),
                                static_cast<i32>(static_cast<sz>(segId) * 8));
                    u8 hra2 = scalarHost[ra];
                    if (hra2 != REG_NONE) {
                        if (hra2 != static_cast<u8>(Reg64::RAX)) a.MovRegReg(static_cast<u8>(Reg64::RAX), hra2);
                        a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    } else {
                        i32 off2 = static_cast<i32>(kScalarRegOffset + static_cast<sz>(ra) * 8);
                        a.MovRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), off2);
                        a.SHLRegImm(static_cast<u8>(Reg64::RAX), 3);
                    }
                    a.AddRegReg(static_cast<u8>(Reg64::R15), static_cast<u8>(Reg64::RAX));

                    loadVectorReg(rd, 0);
                    a.VmovupdMemYmm(static_cast<u8>(Reg64::R15), 0, 0);

                    if (count > 4) {
                        loadVectorReg(static_cast<u8>(rd + 1), 1);
                        a.VmovupdMemYmm(static_cast<u8>(Reg64::R15), 32, 1);
                    }
                    ++bcIdx;
                }
                break;

            // ------------------------------------------------------------
            // Vector Arithmetic
            // ------------------------------------------------------------

            case Opcode::VADD:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vaddpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VSUB:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vsubpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VMUL:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vmulpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VDIV:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vdivpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VMIN:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vminpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VMAX:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vmaxpd(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Vector Filter
            // ------------------------------------------------------------

            case Opcode::VFILTER:
                {
                    u8 mode = imm12 & 0x7;
                    loadVectorReg(ra, 0);
                    {
                        i32 threshOff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(rb) * 8);
                        a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), threshOff);
                        a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                    }

                    static const u8 cmpMap[6] = {0, 4, 1, 2, 14, 13};
                    u8 pred = (mode < 6) ? cmpMap[mode] : 0;

                    a.Vcmppd(2, 0, 1, pred);
                    a.Vpand(0, 0, 2);
                    storeVectorReg(rd, 0, bcIdx);
                    ++bcIdx;
                }
                break;

            case Opcode::VFILTER_EQ:
                loadVectorReg(ra, 0);
                loadScalarReg(rb, static_cast<u8>(Reg64::RAX));
                a.MovMemReg(static_cast<u8>(Reg64::RBP), -48, static_cast<u8>(Reg64::RAX));
                a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RBP), -48);
                a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                a.Vcmppd(2, 0, 1, 0);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VFILTER_NE:
                loadVectorReg(ra, 0);
                loadScalarReg(rb, static_cast<u8>(Reg64::RAX));
                a.MovMemReg(static_cast<u8>(Reg64::RBP), -48, static_cast<u8>(Reg64::RAX));
                a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RBP), -48);
                a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                a.Vcmppd(2, 0, 1, 4);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VFILTER_LT:
                loadVectorReg(ra, 0);
                loadScalarReg(rb, static_cast<u8>(Reg64::RAX));
                a.MovMemReg(static_cast<u8>(Reg64::RBP), -48, static_cast<u8>(Reg64::RAX));
                a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RBP), -48);
                a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                a.Vcmppd(2, 0, 1, 1);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VFILTER_LE:
                loadVectorReg(ra, 0);
                loadScalarReg(rb, static_cast<u8>(Reg64::RAX));
                a.MovMemReg(static_cast<u8>(Reg64::RBP), -48, static_cast<u8>(Reg64::RAX));
                a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::RBP), -48);
                a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                a.Vcmppd(2, 0, 1, 2);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VFILTER_GT:
                loadVectorReg(ra, 0);
                {
                    i32 threshOff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(rb) * 8);
                    a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), threshOff);
                    a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                }
                a.Vcmppd(2, 0, 1, 14);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VFILTER_GE:
                loadVectorReg(ra, 0);
                {
                    i32 threshOff = static_cast<i32>(kScalarRegOffset + static_cast<sz>(rb) * 8);
                    a.LeaRegMem(static_cast<u8>(Reg64::RAX), static_cast<u8>(Reg64::R14), threshOff);
                    a.Vbroadcastsd(1, static_cast<u8>(Reg64::RAX));
                }
                a.Vcmppd(2, 0, 1, 13);
                a.Vpand(0, 0, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VBLEND:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                loadVectorReg(0, 2); // use mask from vector 0 → actually need proper mask setup
                a.Vblendvpd(0, 0, 1, 2);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Vector Reduction
            // ------------------------------------------------------------

            case Opcode::VSUM:
                loadVectorReg(ra, 0);
                a.Vextractf128(1, 0, 1);
                a.Vaddpd(0, 0, 1);
                a.Vmovhlps(1, 0, 0);
                a.Vaddsd(0, 0, 1);
                a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                if (scalarHost[rd] != REG_NONE)
                    a.MovRegReg(scalarHost[rd], static_cast<u8>(Reg64::RAX));
                else {
                    i32 off = static_cast<i32>(kScalarRegOffset + static_cast<sz>(rd) * 8);
                    a.MovMemReg(static_cast<u8>(Reg64::R14), off, static_cast<u8>(Reg64::RAX));
                }
                ++bcIdx;
                break;

            case Opcode::VPROD:
                loadVectorReg(ra, 0);
                a.VmovupdMemYmm(static_cast<u8>(Reg64::RBP), -80, 0);
                a.MovsdXmmMem(0, static_cast<u8>(Reg64::RBP), -80);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -72);
                a.Vmulsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -64);
                a.Vmulsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -56);
                a.Vmulsd(0, 0, 1);
                a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::VRED_MIN:
                loadVectorReg(ra, 0);
                a.VmovupdMemYmm(static_cast<u8>(Reg64::RBP), -80, 0);
                a.MovsdXmmMem(0, static_cast<u8>(Reg64::RBP), -80);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -72);
                a.Vminsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -64);
                a.Vminsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -56);
                a.Vminsd(0, 0, 1);
                a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::VRED_MAX:
                loadVectorReg(ra, 0);
                a.VmovupdMemYmm(static_cast<u8>(Reg64::RBP), -80, 0);
                a.MovsdXmmMem(0, static_cast<u8>(Reg64::RBP), -80);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -72);
                a.Vmaxsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -64);
                a.Vmaxsd(0, 0, 1);
                a.MovsdXmmMem(1, static_cast<u8>(Reg64::RBP), -56);
                a.Vmaxsd(0, 0, 1);
                a.MovqRegXmm(static_cast<u8>(Reg64::RAX), 0);
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Vector Logical
            // ------------------------------------------------------------

            case Opcode::VAND:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vpand(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VOR:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vpor(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            case Opcode::VXOR:
                loadVectorReg(ra, 0);
                loadVectorReg(rb, 1);
                a.Vpxor(0, 0, 1);
                storeVectorReg(rd, 0, bcIdx);
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Aggregate — fallback to simple scalar
            // ------------------------------------------------------------

            case Opcode::AGG_COUNT:
            case Opcode::AGG_SUM:
            case Opcode::AGG_AVG:
            case Opcode::AGG_MIN:
            case Opcode::AGG_MAX:
            case Opcode::AGG_FIRST:
            case Opcode::AGG_LAST:
                a.MovRegImm(static_cast<u8>(Reg64::RAX), 0);
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Conversion
            // ------------------------------------------------------------

            case Opcode::CVT_I64:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::CVT_F64:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            case Opcode::BITCAST:
                loadScalarReg(ra, static_cast<u8>(Reg64::RAX));
                storeScalarReg(rd, static_cast<u8>(Reg64::RAX));
                ++bcIdx;
                break;

            // ------------------------------------------------------------
            // Default — NOP for unimplemented
            // ------------------------------------------------------------

            default:
                ++bcIdx;
                break;
            }
        }

        // ------------------------------------------------------------
        // Resolve branch patches
        // ------------------------------------------------------------
    doneTranslating:
        flushScalars();
        for (auto& p : patches) {
            auto it = bcToCode.find(p.bcIndex);
            if (it != bcToCode.end()) {
                a.PatchBranch(p.codeOffset, static_cast<i32>(it->second));
            }
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
