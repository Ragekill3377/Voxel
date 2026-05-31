#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/registers.hpp"
#include "voxel/bytecode/opcodes.hpp"

#include <ostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cctype>

namespace voxel {

// ============================================================================
// HexDump — formatted hex dump with ASCII representation
// ============================================================================

class HexDump {
public:
    static void Dump(std::ostream& os, const void* data, sz bytes, sz bytesPerLine = 16) {
        const u8* p = static_cast<const u8*>(data);
        for (sz offset = 0; offset < bytes; offset += bytesPerLine) {
            os << std::hex << std::setfill('0') << std::setw(8) << offset << ": ";

            sz lineBytes = (offset + bytesPerLine <= bytes) ? bytesPerLine : (bytes - offset);
            for (sz i = 0; i < bytesPerLine; ++i) {
                if (i < lineBytes) {
                    os << std::setw(2) << static_cast<u32>(p[offset + i]) << " ";
                } else {
                    os << "   ";
                }
            }

            os << " ";
            for (sz i = 0; i < lineBytes; ++i) {
                u8 c = p[offset + i];
                os << (std::isprint(static_cast<int>(c)) ? static_cast<char>(c) : '.');
            }
            os << "\n";
        }
        os << std::dec << std::setfill(' ');
    }

    static void DumpMemory(std::ostream& os, const u8* mem, sz offset, sz length) {
        Dump(os, mem + offset, length);
    }

    static void DumpSegment(std::ostream& os, const f64* data, sz count, sz maxRows = 256) {
        sz rows = (count < maxRows) ? count : maxRows;
        os << "=== Segment dump (" << rows << " of " << count << " rows) ===\n";
        os << std::setprecision(6) << std::fixed;
        for (sz i = 0; i < rows; ++i) {
            os << "  [" << std::setw(6) << i << "]: " << data[i] << "\n";
        }
        if (rows < count) {
            os << "  ... (" << (count - rows) << " more rows)\n";
        }
    }
};

// ============================================================================
// StateDumper — register file, bytecode, and stack visualization
// ============================================================================

class StateDumper {
public:
    static void DumpRegFile(std::ostream& os, const RegFile& regs) {
        regs.DumpAll(os);
    }

    static void DumpBytecode(std::ostream& os, const u32* code, sz count, sz highlightPc = ~0ULL) {
        os << "=== Bytecode Dump (" << count << " instructions) ===\n";
        os << std::left << std::setw(2) << " "
           << std::setw(8) << "PC"
           << std::setw(18) << "Opcode"
           << std::setw(4) << "RD"
           << std::setw(4) << "RA"
           << std::setw(4) << "RB"
           << std::setw(8) << "Imm12"
           << "\n";
        os << std::string(48, '-') << "\n";

        for (sz pc = 0; pc < count; ++pc) {
            u32 raw = code[pc];
            Opcode op = static_cast<Opcode>(raw & 0xFF);
            u8 rd = (raw >> 8) & 0xF;
            u8 ra = (raw >> 12) & 0xF;
            u8 rb = (raw >> 16) & 0xF;
            u16 imm = (raw >> 20) & 0xFFF;
            u8 segId = (raw >> 28) & 0xF;

            os << (pc == highlightPc ? ">" : " ")
               << std::right << std::setw(8) << pc
               << std::left << std::setw(18) << OpcodeName(op)
               << std::right << std::setw(4) << static_cast<u32>(rd)
               << std::setw(4) << static_cast<u32>(ra)
               << std::setw(4) << static_cast<u32>(rb);

            if (static_cast<u8>(op) >= 0x70 && static_cast<u8>(op) <= 0x7F) {
                os << std::setw(4) << "seg" << segId << "/cnt" << static_cast<u32>(raw >> 20 & 0xFF);
            } else {
                os << std::setw(8) << std::hex << "0x" << imm << std::dec;
            }
            os << "\n";
        }
    }

    static void DumpStack(std::ostream& os, const u64* stack, sz depth) {
        os << "=== Stack Dump (" << depth << " entries) ===\n";
        os << "  Offset   Hex Value             Integer          Float\n";
        os << std::string(68, '-') << "\n";
        for (sz i = 0; i < depth; ++i) {
            u64 val = stack[i];
            os << "  " << std::setw(6) << i
               << "  0x" << std::hex << std::setfill('0') << std::setw(16) << val << std::dec << std::setfill(' ')
               << "  " << std::setw(18) << static_cast<i64>(val)
               << "  " << std::fixed << std::setprecision(6) << std::bit_cast<f64>(val) << "\n";
        }
    }

    static void DumpSegments(std::ostream& os, const f64* const* segData, const sz* segSizes, sz numSegs) {
        os << "=== All Segments ===\n";
        for (sz i = 0; i < numSegs; ++i) {
            os << "Segment[" << i << "]: " << segSizes[i] << " rows\n";
            HexDump::DumpSegment(os, segData[i], segSizes[i], 32);
            os << "\n";
        }
    }
};

// ============================================================================
// CrashRecovery — save and load VM state for post-mortem debugging
// ============================================================================

class CrashRecovery {
public:
    static void SaveState(const std::string& filename, const RegFile& regs, const u32* code, sz codeSize, sz pc) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return;

        u64 magic = kFileMagic;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        u64 version = kFileVersion;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        u64 storedPc = static_cast<u64>(pc);
        file.write(reinterpret_cast<const char*>(&storedPc), sizeof(storedPc));

        u64 storedCodeSize = static_cast<u64>(codeSize);
        file.write(reinterpret_cast<const char*>(&storedCodeSize), sizeof(storedCodeSize));

        file.write(reinterpret_cast<const char*>(code), static_cast<std::streamsize>(codeSize * sizeof(u32)));

        for (sz i = 0; i < RegFile::kScalarCount; ++i) {
            u64 sv = regs.Scalar(i);
            file.write(reinterpret_cast<const char*>(&sv), sizeof(sv));
        }

        for (sz i = 0; i < RegFile::kVectorCount; ++i) {
            const u8* vd = regs.VecRaw(i);
            file.write(reinterpret_cast<const char*>(vd), static_cast<std::streamsize>(RegFile::kVecWidth));
        }

        for (sz i = 0; i < RegFile::kMaskCount; ++i) {
            u32 mv = regs.Mask(i);
            file.write(reinterpret_cast<const char*>(&mv), sizeof(mv));
        }

        u32 flags = regs.RawFlags();
        file.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    }

    static bool LoadState(const std::string& filename, RegFile& regs, std::vector<u32>& code, sz& pc) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        u64 magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != kFileMagic) return false;

        u64 version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kFileVersion) return false;

        u64 storedPc;
        file.read(reinterpret_cast<char*>(&storedPc), sizeof(storedPc));
        pc = static_cast<sz>(storedPc);

        u64 storedCodeSize;
        file.read(reinterpret_cast<char*>(&storedCodeSize), sizeof(storedCodeSize));

        code.resize(static_cast<sz>(storedCodeSize));
        file.read(reinterpret_cast<char*>(code.data()),
                  static_cast<std::streamsize>(storedCodeSize * sizeof(u32)));

        regs.Reset();

        u64 sv;
        for (sz i = 0; i < RegFile::kScalarCount; ++i) {
            file.read(reinterpret_cast<char*>(&sv), sizeof(sv));
            regs.Scalar(i) = sv;
        }

        for (sz i = 0; i < RegFile::kVectorCount; ++i) {
            u8* vd = regs.VecRaw(i);
            file.read(reinterpret_cast<char*>(vd), static_cast<std::streamsize>(RegFile::kVecWidth));
        }

        for (sz i = 0; i < RegFile::kMaskCount; ++i) {
            u32 mv;
            file.read(reinterpret_cast<char*>(&mv), sizeof(mv));
            regs.Mask(i) = mv;
        }

        u32 savedFlags;
        file.read(reinterpret_cast<char*>(&savedFlags), sizeof(savedFlags));
        (void)savedFlags;

        return true;
    }

private:
    static constexpr u64 kFileMagic   = 0x564F58454C504D50ULL;
    static constexpr u64 kFileVersion = 1;
};

} // namespace voxel
