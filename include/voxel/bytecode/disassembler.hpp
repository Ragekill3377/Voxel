#pragma once

#include "voxel/bytecode/instruction.hpp"
#include "voxel/core/types.hpp"
#include <ostream>
#include <iomanip>
#include <string>

namespace voxel {

class Disassembler {
public:
    void Disassemble(std::ostream& os, const u32* code, sz count) {
        for (sz pc = 0; pc < count; ++pc) {
            DisassembleInstruction(os, code, pc);
        }
    }

    void DisassembleInstruction(std::ostream& os, const u32* code, sz pc) {
        Instruction inst{code[pc]};
        Opcode op = inst.Op();

        os << std::hex << std::setfill('0')
           << "0x" << std::setw(8) << (pc * 4)
           << " [" << std::dec << std::setw(4) << std::setfill(' ') << pc << "] "
           << std::hex << std::setfill('0') << std::setw(8) << inst.raw
           << "  ";

        WriteMnemonic(os, op);

        switch (op) {
        // Control — no operands
        case Opcode::NOP:
        case Opcode::HALT:
        case Opcode::TRAP:
        case Opcode::BREAK:
        case Opcode::YIELD:
        case Opcode::BARRIER:
        case Opcode::PREFETCH:
        case Opcode::FLUSH_CACHE:
        case Opcode::SYNC:
        case Opcode::MEMFENCE:
        case Opcode::RET:
            break;

        // Scalar Move
        case Opcode::MOV:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", " << inst.Simm12();
            break;
        case Opcode::MOVR:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::ADDI:
        case Opcode::SUBI:
        case Opcode::MULI:
        case Opcode::ANDI:
        case Opcode::ORI:
        case Opcode::XORI:
        case Opcode::SHLI:
        case Opcode::SHRI:
        case Opcode::SAR_I:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", " << inst.Simm12();
            break;
        case Opcode::MOVZ:
        case Opcode::MOVN:
        case Opcode::MOVK:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", " << inst.Simm12();
            break;
        case Opcode::LEA:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", " << inst.Simm12();
            break;

        // Scalar Arithmetic (rd, ra, rb)
        case Opcode::ADD:
        case Opcode::SUB:
        case Opcode::MUL:
        case Opcode::DIV:
        case Opcode::MOD:
        case Opcode::MIN:
        case Opcode::MAX:
        case Opcode::AVG:
        case Opcode::ADDF:
        case Opcode::SUBF:
        case Opcode::MULF:
        case Opcode::DIVF:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;
        case Opcode::NEG:
        case Opcode::ABS:
        case Opcode::NEGF:
        case Opcode::ABSF:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;

        // Scalar Bitwise
        case Opcode::AND:
        case Opcode::OR:
        case Opcode::XOR:
        case Opcode::SHL:
        case Opcode::SHR:
        case Opcode::SAR:
        case Opcode::ROL:
        case Opcode::ROR:
        case Opcode::BEXTR:
        case Opcode::BZHI:
        case Opcode::PDEP:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;
        case Opcode::NOT:
        case Opcode::POPCNT:
        case Opcode::CLZ:
        case Opcode::CTZ:
        case Opcode::BSWAP:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;

        // Comparison
        case Opcode::CMP:
        case Opcode::CMPF:
        case Opcode::CMPU:
        case Opcode::TST:
        case Opcode::TSTF:
            os << " R" << std::dec << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;
        case Opcode::ISNULL:
        case Opcode::ISNOTNULL:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::SELECT:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;
        case Opcode::SELECTV:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb());
            break;

        // Branching
        case Opcode::JMP:
        case Opcode::JZ:
        case Opcode::JNZ:
        case Opcode::JS:
        case Opcode::JNS:
        case Opcode::JO:
        case Opcode::JNO:
        case Opcode::JC:
        case Opcode::JNC:
        case Opcode::JL:
        case Opcode::JLE:
        case Opcode::JG:
        case Opcode::JGE:
        case Opcode::CALL:
        {
            i16 offset = inst.Simm12();
            sz target = pc + 1 + static_cast<i64>(offset);
            os << " 0x" << std::hex << std::setw(8) << (target * 4)
               << " [" << std::dec << target << "]";
            break;
        }
        case Opcode::TABLE_JMP:
            os << " R" << std::dec << unsigned(inst.Ra());
            break;

        // Type Conversion
        case Opcode::CVT_I8:
        case Opcode::CVT_I16:
        case Opcode::CVT_I32:
        case Opcode::CVT_I64:
        case Opcode::CVT_F32:
        case Opcode::CVT_F64:
        case Opcode::CVT_U8:
        case Opcode::CVT_U16:
        case Opcode::CVT_U32:
        case Opcode::CVT_U64:
        case Opcode::BITCAST:
        case Opcode::REINTERPRET:
        case Opcode::TRUNC:
        case Opcode::ROUND:
        case Opcode::CEIL:
        case Opcode::FLOOR:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;

        // Vector I/O
        case Opcode::VLOAD:
        case Opcode::VSTORE:
        case Opcode::VGATHER:
        case Opcode::VSCATTER:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", seg:" << unsigned(inst.SegId())
               << ", cnt:" << unsigned(inst.VecCount());
            break;
        case Opcode::VLOAD_STRIDED:
        case Opcode::VSTORE_STRIDED:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", str:" << unsigned(inst.SegId())
               << ", cnt:" << unsigned(inst.VecCount());
            break;
        case Opcode::VLOAD_MASKED:
        case Opcode::VSTORE_MASKED:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", M" << unsigned(inst.Rb());
            break;
        case Opcode::VSPLAT:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::VEXTRACT:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", " << unsigned(inst.Rb());
            break;
        case Opcode::VINSERT:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb())
               << ", " << unsigned(inst.Imm12() & 0xF);
            break;
        case Opcode::VPERMUTE:
        case Opcode::VSHUFFLE:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb());
            break;
        case Opcode::VREVERSE:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra());
            break;
        case Opcode::VROTATE:
        case Opcode::VSLIDE:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", " << unsigned(inst.VecCount());
            break;

        // Vector Arithmetic
        case Opcode::VADD:
        case Opcode::VSUB:
        case Opcode::VMUL:
        case Opcode::VDIV:
        case Opcode::VMOD:
        case Opcode::VMIN:
        case Opcode::VMAX:
        case Opcode::VAVG:
        case Opcode::VFMA:
        case Opcode::VFMS:
        case Opcode::VPOW:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb());
            break;
        case Opcode::VNEG:
        case Opcode::VABS:
        case Opcode::VSQRT:
        case Opcode::VRSQRT:
        case Opcode::VRCP:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra());
            break;

        // Vector Scalar Arithmetic
        case Opcode::VSADD:
        case Opcode::VSSUB:
        case Opcode::VSMUL:
        case Opcode::VSDIV:
        case Opcode::VSMOD:
        case Opcode::VSMIN:
        case Opcode::VSMAX:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;

        // Vector Comparison
        case Opcode::VCMPEQ:
        case Opcode::VCMPNE:
        case Opcode::VCMPLT:
        case Opcode::VCMPLE:
        case Opcode::VCMPGT:
        case Opcode::VCMPGE:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb());
            break;
        case Opcode::VCMPNULL:
        case Opcode::VCMPNOTNULL:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra());
            break;

        // Vector Logical
        case Opcode::VAND:
        case Opcode::VOR:
        case Opcode::VXOR:
        case Opcode::VANDN:
        case Opcode::VSHL:
        case Opcode::VSHR:
        case Opcode::VSAR:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb());
            break;
        case Opcode::VNOT:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra());
            break;

        // Vector Filter
        case Opcode::VFILTER:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb())
               << ", mode:" << unsigned(inst.CmpMode());
            break;
        case Opcode::VFILTER_EQ:
        case Opcode::VFILTER_NE:
        case Opcode::VFILTER_LT:
        case Opcode::VFILTER_LE:
        case Opcode::VFILTER_GT:
        case Opcode::VFILTER_GE:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;
        case Opcode::VBLEND:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", V" << unsigned(inst.Rb())
               << ", M" << unsigned(inst.Imm12() & 0x7);
            break;
        case Opcode::VMASK_STORE:
        case Opcode::VMASK_LOAD:
            os << " V" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", M" << unsigned(inst.Rb())
               << ", seg:" << unsigned(inst.SegId());
            break;

        // Vector Reduction
        case Opcode::VSUM:
        case Opcode::VPROD:
        case Opcode::VMEAN:
        case Opcode::VSTDDEV:
        case Opcode::VVARIANCE:
        case Opcode::VRED_MIN:
        case Opcode::VRED_MAX:
        case Opcode::VCOUNT:
        case Opcode::VANY:
        case Opcode::VALL:
        case Opcode::VFIRST:
        case Opcode::VLAST:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra());
            break;
        case Opcode::VNTH:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", V" << unsigned(inst.Ra())
               << ", " << unsigned(inst.Imm12());
            break;

        // Aggregate
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
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", seg:" << unsigned(inst.Ra());
            break;
        case Opcode::HASH_INIT:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::HASH_PROBE:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", R" << unsigned(inst.Rb());
            break;

        // Hash / Sort / Join
        case Opcode::HASH_BUILD:
            os << " H" << std::dec << unsigned(inst.Rd())
               << ", seg:" << unsigned(inst.Ra())
               << ", seg:" << unsigned(inst.Rb());
            break;
        case Opcode::HASH_LOOKUP:
            os << " H" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra())
               << ", seg:" << unsigned(inst.Rb());
            break;
        case Opcode::SORT_ASC:
        case Opcode::SORT_DESC:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::SORT_TOPK:
        case Opcode::SORT_BOTTOMK:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", k:" << unsigned(inst.Rb())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::JOIN_HASH:
        case Opcode::JOIN_MERGE:
        case Opcode::JOIN_NESTED:
        case Opcode::JOIN_ANTI:
        case Opcode::JOIN_SEMI:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", seg:" << unsigned(inst.Ra())
               << ", seg:" << unsigned(inst.Rb());
            break;
        case Opcode::WINDOW_ROW:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", st:" << unsigned(inst.SegId())
               << ", en:" << unsigned(inst.Imm12() & 0xF)
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::WINDOW_RANGE:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", rng:" << unsigned(inst.VecCount())
               << ", R" << unsigned(inst.Ra());
            break;
        case Opcode::PARTITION_HASH:
            os << " seg:" << std::dec << unsigned(inst.Rd())
               << ", parts:" << unsigned(inst.Ra());
            break;
        case Opcode::SERIALIZE:
        case Opcode::DESERIALIZE:
            os << " R" << std::dec << unsigned(inst.Rd())
               << ", R" << unsigned(inst.Ra());
            break;

        default:
            os << " ???";
            break;
        }

        os << "\n";
    }

private:
    void WriteMnemonic(std::ostream& os, Opcode op) {
        const char* name = OpcodeName(op);
        os << std::left << std::setw(20) << std::setfill(' ') << name << std::right;
    }
};

} // namespace voxel
