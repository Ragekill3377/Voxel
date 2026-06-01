#pragma once

// ============================================================================
// VoxelVM (Voxel) — Production-Grade Virtual Machine Bytecode Engine
// Single-header C++20 library for high-throughput columnar data analytics.
//
// Include via: #include "voxel/voxel.hpp"
// Link with:    target_link_libraries(your_target voxel)
//
// Architecture overview:
//   core/       Fundamental types, arena allocator, register file, platform detection
//   bytecode/   Opcodes (200+), instruction encoding, assembler, verifier, disassembler
//   exec/       Execution engine with jump-table dispatch for all opcodes
//   codegen/    JIT compilation to x86-64 and ARM64 native code
//   simd/       AVX2, AVX-512, NEON, and scalar SIMD abstraction layers
//   data/       Columnar segments, encodings (dict/RLE/delta/bitpacked), nulls, blocks
//   ops/        Query operators: filter, aggregate, hash, sort, join
//   system/     Profiling, telemetry, error handling, state debugging
//   util/       Bit manipulation, hashing, math, string, thread pool
// ============================================================================

// Core
#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/arena.hpp"
#include "voxel/core/registers.hpp"

// Bytecode
#include "voxel/bytecode/opcodes.hpp"
#include "voxel/bytecode/instruction.hpp"
#include "voxel/bytecode/assembler.hpp"
#include "voxel/bytecode/verifier.hpp"
#include "voxel/bytecode/disassembler.hpp"

// Execution
#include "voxel/exec/engine.hpp"
#include "voxel/exec/optimizer.hpp"

// JIT Codegen (platform-specific)
#include "voxel/codegen/jit.hpp"
#if VOXEL_ARCH_X86_64
#include "voxel/codegen/x86_64.hpp"
#elif VOXEL_ARCH_ARM64
#include "voxel/codegen/arm64.hpp"
#endif

// SIMD (platform-specific; all provide same interface)
#include "voxel/simd/scalar.hpp"   // always available portable fallback
#if VOXEL_ARCH_X86_64
#include "voxel/simd/x86.hpp"
#if VOXEL_SIMD_AVX512
#include "voxel/simd/avx512.hpp"
#endif
#elif VOXEL_ARCH_ARM64
#include "voxel/simd/neon.hpp"
#endif

// Data layer
#include "voxel/data/segment.hpp"
#include "voxel/data/encoding.hpp"
#include "voxel/data/nulls.hpp"
#include "voxel/data/block.hpp"

// Query operators
#include "voxel/ops/filter.hpp"
#include "voxel/ops/aggregate.hpp"
#include "voxel/ops/hash.hpp"
#include "voxel/ops/sort.hpp"
#include "voxel/ops/join.hpp"

// System
#include "voxel/system/profile.hpp"
#include "voxel/system/telemetry.hpp"
#include "voxel/system/error.hpp"
#include "voxel/system/debug.hpp"

// Utilities
#include "voxel/util/bit.hpp"
#include "voxel/util/hash.hpp"
#include "voxel/util/math.hpp"
#include "voxel/util/string.hpp"
#include "voxel/util/thread.hpp"

namespace voxel {
namespace codegen {

inline std::unique_ptr<JitCompiler> CreateJitCompiler()
{
#if VOXEL_ARCH_X86_64
    return std::make_unique<x64::X64Compiler>();
#elif VOXEL_ARCH_ARM64
    return std::make_unique<arm64::ARM64Compiler>();
#else
    return nullptr;
#endif
}

} // namespace codegen
} // namespace voxel
