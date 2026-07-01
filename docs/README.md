# VoxelVM Documentation

## Architecture

VoxelVM is a header-only C++20 bytecode virtual machine for columnar data analytics with Python bindings, a SQL frontend, and a JIT backend. It compiles query plans into a dense 32-bit instruction stream and executes them via a jump-table dispatch interpreter, with an optional JIT backend that emits native AVX2 code.

```
┌─────────────────────────────────────────────────────────────┐
│                      VoxelVM Architecture                    │
├──────────┬─────────────┬──────────┬───────────┬─────────────┤
│ bytecode │    exec     │ codegen  │   simd    │    data     │
│  opcodes │   engine    │   jit    │   avx2    │  segments   │
│  instr   │  dispatch   │  x86-64  │  avx512   │  encoding   │
│assembler │  optimizer  │  arm64   │   neon    │   nulls     │
│ verifier │  streaming  │  cache   │  scalar   │   blocks    │
├──────────┴─────────────┴──────────┴───────────┴─────────────┤
│    ops: filter │ aggregate │ sort │ join │ hash              │
│   system: telemetry │ error │ debug │ profile               │
│    util: bit │ math │ hash │ string │ thread                │
├─────────────────────────────────────────────────────────────┤
│  Python: bindings (pybind11) │ query builder │ SQL (pglast) │
│  Market: VWAP │ RSI │ MACD │ Bollinger │ Sharpe │ OHLC     │
│  Data: mmap segments │ chunked reader │ streaming engine    │
└─────────────────────────────────────────────────────────────┘
```

Module layout under `include/voxel/`:

| Directory   | Purpose                                              |
|-------------|------------------------------------------------------|
| `core/`     | Types, register file, arena allocator, platform detection, chunked reader |
| `bytecode/` | 208 opcodes, `Instruction` encoder, assembler/linker, verifier, disassembler |
| `exec/`     | `Engine<T>` interpreter, optimizer passes, streaming engine |
| `codegen/`  | `JitCompiler` interface, x86-64 backend, executable memory manager, LRU JIT cache |
| `simd/`     | Per-platform SIMD abstraction: AVX2, AVX-512, NEON, scalar fallback |
| `data/`     | `Segment<T>`, mmap segment, encodings (dictionary, RLE, delta, bit-packed), null bitmap |
| `ops/`      | Vector filter, hash aggregation (Robin Hood), sort, hash/merge/anti/semi join |
| `system/`   | Instruction profiler, cycle timer, telemetry, error handling |
| `util/`     | Hash functions, thread pool, work-stealing scheduler, bit utilities |

Python modules under `bindings/`:

| File | Purpose |
|---|---|
| `voxel_bindings.cpp` | pybind11 C++ bindings for Engine, Segment, Instruction, NullBitmap, encodings, ops |
| `voxel_query.py` | High-level query builder: `from_numpy(data).filter_gt(t).sum().run()` |
| `voxel_sql.py` | PostgreSQL-compatible SQL via libpg_query (pglast) |
| `voxel_market.py` | Market data analytics: VWAP, RSI, MACD, Bollinger, Sharpe, OHLC |
| `voxel_data.py` | Large dataset API: memory-mapped files, chunked streaming |
| `voxel_stream.py` | Real-time streaming data feed processing |

## Getting Started

### Requirements

C++20. GCC 12+, Clang 16+, or MSVC 2022+. x86-64 hosts need SSE4.2; AVX2 and AVX-512 are detected automatically. ARM64 hosts need NEON.

### Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Include

```cpp
#include "voxel/voxel.hpp"
```

### CMake Integration

```cmake
add_subdirectory(path/to/voxelvm)
target_link_libraries(your_target voxel)
```

The `voxel` target is `INTERFACE` — it adds `include/` to the include path and sets `-std=c++20`.

## API Reference

### Engine\<T\>

Defined in `<voxel/exec/engine.hpp>`. Monomorphized per element type `T`. Owns the register file, a vector of `Segment<T>` views, and a bytecode buffer.

```cpp
Engine<f64> engine;

// Attach external data as a segment, returns segment ID
sz segId = engine.AddSegment(dataPtr, rowCount);

// Load bytecode program
engine.LoadProgram(code);  // code is std::vector<u32>

// Run to completion (halts on HALT opcode or error flag)
engine.Run();

// Access scalar registers (read/write refs to u64)
u64& acc = engine.ScalarReg(0);       // R0
u64  val = engine.ScalarReg(5);       // read R5

// Access full register file
RegFile& regs = engine.Registers();

// Reset engine state (clears regs, segments, code)
engine.Reset();

// Access segments by ID
Segment<f64>& seg = engine.GetSegment(segId);
sz count = seg.Count;
```

### Instruction

Defined in `<voxel/bytecode/instruction.hpp>`. 32-bit packed representation: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`. All factory methods are `static`.

```cpp
// Encode an instruction word
u32 raw = Instruction::Add(0, 1, 2).raw;   // R0 = R1 + R2
```

#### Control (0x00–0x0F)

| Factory                    | Encoded opcode         |
|----------------------------|------------------------|
| `Instruction::Nop()`       | `NOP` (0x00)           |
| `Instruction::Halt()`      | `HALT` (0x01)          |
| `Instruction::Trap()`      | `TRAP` (0x02)          |
| `Instruction::Break()`     | `BREAK` (0x03)         |
| `Instruction::Yield()`     | `YIELD` (0x04)         |
| `Instruction::Barrier()`   | `BARRIER` (0x05)       |
| `Instruction::Prefetch()`  | `PREFETCH` (0x06)      |
| `Instruction::FlushCache()`| `FLUSH_CACHE` (0x07)   |
| `Instruction::Sync()`      | `SYNC` (0x08)          |
| `Instruction::Memfence()`  | `MEMFENCE` (0x09)      |

#### Scalar Move / Immediate (0x10–0x1F)

| Factory                                   | Opcode    |
|-------------------------------------------|-----------|
| `Instruction::Mov(rd, imm)`               | `MOV`     |
| `Instruction::Movr(rd, ra)`               | `MOVR`    |
| `Instruction::Addi(rd, ra, imm)`          | `ADDI`    |
| `Instruction::Subi(rd, ra, imm)`          | `SUBI`    |
| `Instruction::Muli(rd, ra, imm)`          | `MULI`    |
| `Instruction::Andi(rd, ra, imm)`          | `ANDI`    |
| `Instruction::Ori(rd, ra, imm)`           | `ORI`     |
| `Instruction::Xori(rd, ra, imm)`          | `XORI`    |
| `Instruction::Shli(rd, ra, imm)`          | `SHLI`    |
| `Instruction::Shri(rd, ra, imm)`          | `SHRI`    |
| `Instruction::SarI(rd, ra, imm)`          | `SAR_I`   |
| `Instruction::Movz(rd, imm)`              | `MOVZ`    |
| `Instruction::Movn(rd, imm)`              | `MOVN`    |
| `Instruction::Movk(rd, imm)`              | `MOVK`    |
| `Instruction::Lea(rd, ra, off)`           | `LEA`     |

#### Scalar Arithmetic (0x20–0x2F)

| Factory                               | Opcode  | Domain |
|---------------------------------------|---------|--------|
| `Instruction::Add(rd, ra, rb)`        | `ADD`   | Int    |
| `Instruction::Sub(rd, ra, rb)`        | `SUB`   | Int    |
| `Instruction::Mul(rd, ra, rb)`        | `MUL`   | Int    |
| `Instruction::Div(rd, ra, rb)`        | `DIV`   | Int    |
| `Instruction::Mod(rd, ra, rb)`        | `MOD`   | Int    |
| `Instruction::Neg(rd, ra)`            | `NEG`   | Int    |
| `Instruction::Abs(rd, ra)`            | `ABS`   | Int    |
| `Instruction::Min(rd, ra, rb)`        | `MIN`   | Int    |
| `Instruction::Max(rd, ra, rb)`        | `MAX`   | Int    |
| `Instruction::Avg(rd, ra, rb)`        | `AVG`   | Int    |
| `Instruction::Addf(rd, ra, rb)`       | `ADDF`  | Float  |
| `Instruction::Subf(rd, ra, rb)`       | `SUBF`  | Float  |
| `Instruction::Mulf(rd, ra, rb)`       | `MULF`  | Float  |
| `Instruction::Divf(rd, ra, rb)`       | `DIVF`  | Float  |
| `Instruction::Negf(rd, ra)`           | `NEGF`  | Float  |
| `Instruction::Absf(rd, ra)`           | `ABSF`  | Float  |

#### Scalar Bitwise (0x30–0x3F)

| Factory                               | Opcode    |
|---------------------------------------|-----------|
| `Instruction::And(rd, ra, rb)`        | `AND`     |
| `Instruction::Or(rd, ra, rb)`         | `OR`      |
| `Instruction::Xor(rd, ra, rb)`        | `XOR`     |
| `Instruction::Not(rd, ra)`            | `NOT`     |
| `Instruction::Shl(rd, ra, rb)`        | `SHL`     |
| `Instruction::Shr(rd, ra, rb)`        | `SHR`     |
| `Instruction::Sar(rd, ra, rb)`        | `SAR`     |
| `Instruction::Rol(rd, ra, rb)`        | `ROL`     |
| `Instruction::Ror(rd, ra, rb)`        | `ROR`     |
| `Instruction::Popcnt(rd, ra)`         | `POPCNT`  |
| `Instruction::Clz(rd, ra)`            | `CLZ`     |
| `Instruction::Ctz(rd, ra)`            | `CTZ`     |
| `Instruction::Bswap(rd, ra)`          | `BSWAP`   |
| `Instruction::Bextr(rd, ra, rb)`      | `BEXTR`   |
| `Instruction::Bzhi(rd, ra, rb)`       | `BZHI`    |
| `Instruction::Pdep(rd, ra, rb)`       | `PDEP`    |

#### Comparison (0x40–0x4F)

| Factory                               | Opcode      |
|---------------------------------------|-------------|
| `Instruction::Cmp(ra, rb)`            | `CMP`       |
| `Instruction::Cmpf(ra, rb)`           | `CMPF`      |
| `Instruction::Cmpu(ra, rb)`           | `CMPU`      |
| `Instruction::Tst(ra, rb)`            | `TST`       |
| `Instruction::Tstf(ra, rb)`           | `TSTF`      |
| `Instruction::Isnull(rd, ra)`         | `ISNULL`    |
| `Instruction::Isnotnull(rd, ra)`      | `ISNOTNULL` |
| `Instruction::Select(rd, ra, rb)`     | `SELECT`    |
| `Instruction::Selectv(rd, va, vb)`    | `SELECTV`   |

#### Branching (0x50–0x5F)

| Factory                            | Opcode      | Condition          |
|------------------------------------|-------------|--------------------|
| `Instruction::Jmp(offset)`         | `JMP`       | Unconditional      |
| `Instruction::Jz(offset)`          | `JZ`        | Zero flag set      |
| `Instruction::Jnz(offset)`         | `JNZ`       | Zero flag clear    |
| `Instruction::Js(offset)`          | `JS`        | Sign flag set      |
| `Instruction::Jns(offset)`         | `JNS`       | Sign flag clear    |
| `Instruction::Jo(offset)`          | `JO`        | Overflow flag set  |
| `Instruction::Jno(offset)`         | `JNO`       | Overflow flag clear|
| `Instruction::Jc(offset)`          | `JC`        | Carry flag set     |
| `Instruction::Jnc(offset)`         | `JNC`       | Carry flag clear   |
| `Instruction::Jl(offset)`          | `JL`        | Less than          |
| `Instruction::Jle(offset)`         | `JLE`       | Less or equal      |
| `Instruction::Jg(offset)`          | `JG`        | Greater than       |
| `Instruction::Jge(offset)`         | `JGE`       | Greater or equal   |
| `Instruction::Call(offset)`        | `CALL`      | Subroutine call    |
| `Instruction::Ret()`               | `RET`       | Subroutine return  |
| `Instruction::TableJmp(ra)`        | `TABLE_JMP` | Indirect jump      |

Offsets are signed 12-bit instruction offsets (not byte offsets). `Jnz(-6)` jumps back 6 instructions.

#### Type Conversion (0x60–0x6F)

| Factory                            | Opcode       |
|------------------------------------|--------------|
| `Instruction::CvtI8(rd, ra)`       | `CVT_I8`     |
| `Instruction::CvtI16(rd, ra)`      | `CVT_I16`    |
| `Instruction::CvtI32(rd, ra)`      | `CVT_I32`    |
| `Instruction::CvtI64(rd, ra)`      | `CVT_I64`    |
| `Instruction::CvtF32(rd, ra)`      | `CVT_F32`    |
| `Instruction::CvtF64(rd, ra)`      | `CVT_F64`    |
| `Instruction::CvtU8(rd, ra)`       | `CVT_U8`     |
| `Instruction::CvtU16(rd, ra)`      | `CVT_U16`    |
| `Instruction::CvtU32(rd, ra)`      | `CVT_U32`    |
| `Instruction::CvtU64(rd, ra)`      | `CVT_U64`    |
| `Instruction::Bitcast(rd, ra)`     | `BITCAST`    |
| `Instruction::Reinterpret(rd, ra)` | `REINTERPRET`|
| `Instruction::Trunc(rd, ra)`       | `TRUNC`      |
| `Instruction::Round(rd, ra)`       | `ROUND`      |
| `Instruction::Ceil(rd, ra)`        | `CEIL`       |
| `Instruction::Floor(rd, ra)`       | `FLOOR`      |

#### Vector I/O (0x70–0x7F)

| Factory                                          | Opcode          |
|--------------------------------------------------|-----------------|
| `Instruction::VLoad(vd, ra, segId, count)`       | `VLOAD`         |
| `Instruction::VStore(vs, ra, segId, count)`      | `VSTORE`        |
| `Instruction::VGather(vd, ra, segId, count)`     | `VGATHER`       |
| `Instruction::VScatter(vs, ra, segId, count)`    | `VSCATTER`      |
| `Instruction::VLoadStrided(vd, ra, stride, cnt)` | `VLOAD_STRIDED`  |
| `Instruction::VStoreStrided(vs, ra, stride, cnt)`| `VSTORE_STRIDED` |
| `Instruction::VLoadMasked(vd, ra, maskReg)`      | `VLOAD_MASKED`  |
| `Instruction::VStoreMasked(vs, ra, maskReg)`     | `VSTORE_MASKED` |
| `Instruction::VSplat(vd, ra)`                    | `VSPLAT`        |
| `Instruction::VExtract(rd, va, lane)`            | `VEXTRACT`      |
| `Instruction::VInsert(vd, va, rb, lane)`         | `VINSERT`       |
| `Instruction::VPermute(vd, va, vb)`              | `VPERMUTE`      |
| `Instruction::VShuffle(vd, va, vb)`              | `VSHUFFLE`      |
| `Instruction::VReverse(vd, va)`                  | `VREVERSE`      |
| `Instruction::VRotate(vd, va, amount)`           | `VROTATE`       |
| `Instruction::VSlide(vd, va, amount)`            | `VSLIDE`        |

`VLoad` reads `count` elements from segment `segId` starting at the offset in `Ra`. If `count` is 0, the full vector width is loaded.

#### Vector Arithmetic (0x80–0x8F)

| Factory                            | Opcode  |
|------------------------------------|---------|
| `Instruction::VAdd(vd, va, vb)`    | `VADD`  |
| `Instruction::VSub(vd, va, vb)`    | `VSUB`  |
| `Instruction::VMul(vd, va, vb)`    | `VMUL`  |
| `Instruction::VDiv(vd, va, vb)`    | `VDIV`  |
| `Instruction::VMod(vd, va, vb)`    | `VMOD`  |
| `Instruction::VNeg(vd, va)`        | `VNEG`  |
| `Instruction::VAbs(vd, va)`        | `VABS`  |
| `Instruction::VMin(vd, va, vb)`    | `VMIN`  |
| `Instruction::VMax(vd, va, vb)`    | `VMAX`  |
| `Instruction::VAvg(vd, va, vb)`    | `VAVG`  |
| `Instruction::VFma(vd, va, vb)`    | `VFMA`  |
| `Instruction::VFms(vd, va, vb)`    | `VFMS`  |
| `Instruction::VSqrt(vd, va)`       | `VSQRT` |
| `Instruction::VRSqrt(vd, va)`      | `VRSQRT`|
| `Instruction::VRcp(vd, va)`        | `VRCP`  |
| `Instruction::VPow(vd, va, vb)`    | `VPOW`  |

#### Vector-Scalar Arithmetic (0x90–0x9F)

Each broadcasts scalar `rb` across all lanes of `va`. `vd = va {op} rb`.

| Factory                            | Opcode  |
|------------------------------------|---------|
| `Instruction::VSadd(vd, va, rb)`   | `VSADD` |
| `Instruction::VSsub(vd, va, rb)`   | `VSSUB` |
| `Instruction::VSmul(vd, va, rb)`   | `VSMUL` |
| `Instruction::VSdiv(vd, va, rb)`   | `VSDIV` |
| `Instruction::VSmod(vd, va, rb)`   | `VSMOD` |
| `Instruction::VSmin(vd, va, rb)`   | `VSMIN` |
| `Instruction::VSmax(vd, va, rb)`   | `VSMAX` |

#### Vector Comparison (0xA0–0xAF)

Lane-wise comparison. Result lanes are all-bits-set on true, zero on false. Mask register updated.

| Factory                                | Opcode       |
|----------------------------------------|--------------|
| `Instruction::VCmpeq(vd, va, vb)`      | `VCMPEQ`     |
| `Instruction::VCmpne(vd, va, vb)`      | `VCMPNE`     |
| `Instruction::VCmplt(vd, va, vb)`      | `VCMPLT`     |
| `Instruction::VCmple(vd, va, vb)`      | `VCMPLE`     |
| `Instruction::VCmpgt(vd, va, vb)`      | `VCMPGT`     |
| `Instruction::VCmpge(vd, va, vb)`      | `VCMPGE`     |
| `Instruction::VCmpnull(vd, va)`        | `VCMPNULL`   |
| `Instruction::VCmpnotnull(vd, va)`     | `VCMPNOTNULL`|

#### Vector Logical (0xB0–0xBF)

Bitwise operations on the full 256-bit vector register.

| Factory                            | Opcode  |
|------------------------------------|---------|
| `Instruction::VAnd(vd, va, vb)`    | `VAND`  |
| `Instruction::VOr(vd, va, vb)`     | `VOR`   |
| `Instruction::VXor(vd, va, vb)`    | `VXOR`  |
| `Instruction::VNot(vd, va)`        | `VNOT`  |
| `Instruction::VAndn(vd, va, vb)`   | `VANDN` |
| `Instruction::VShl(vd, va, vb)`    | `VSHL`  |
| `Instruction::VShr(vd, va, vb)`    | `VSHR`  |
| `Instruction::VSar(vd, va, vb)`    | `VSAR`  |

#### Vector Filter (0xC0–0xCF)

Each compares vector `va` lanes against scalar `rb` threshold. Passing lanes retain their value; failing lanes become zero. Mask register updated.

| Factory                               | Opcode        |
|---------------------------------------|---------------|
| `Instruction::VFilter(vd, va, rb, mode)`| `VFILTER`   |
| `Instruction::VFilterEq(vd, va, rb)`  | `VFILTER_EQ`  |
| `Instruction::VFilterNe(vd, va, rb)`  | `VFILTER_NE`  |
| `Instruction::VFilterLt(vd, va, rb)`  | `VFILTER_LT`  |
| `Instruction::VFilterLe(vd, va, rb)`  | `VFILTER_LE`  |
| `Instruction::VFilterGt(vd, va, rb)`  | `VFILTER_GT`  |
| `Instruction::VFilterGe(vd, va, rb)`  | `VFILTER_GE`  |
| `Instruction::VBlend(vd, va, vb, mk)` | `VBLEND`      |
| `Instruction::VMaskStore(vs, ra, mk, seg)` | `VMASK_STORE` |
| `Instruction::VMaskLoad(vd, ra, mk, seg)`  | `VMASK_LOAD`  |

Filter modes: 0 = EQ, 1 = NE, 2 = LT, 3 = LE, 4 = GT, 5 = GE.

#### Vector Reduction (0xD0–0xDF)

Reduces all lanes of vector `va` into scalar register `rd`.

| Factory                            | Opcode     |
|------------------------------------|------------|
| `Instruction::VSum(rd, va)`        | `VSUM`     |
| `Instruction::VProd(rd, va)`       | `VPROD`    |
| `Instruction::VMean(rd, va)`       | `VMEAN`    |
| `Instruction::VStddev(rd, va)`     | `VSTDDEV`  |
| `Instruction::VVariance(rd, va)`   | `VVARIANCE`|
| `Instruction::VRedMin(rd, va)`     | `VRED_MIN` |
| `Instruction::VRedMax(rd, va)`     | `VRED_MAX` |
| `Instruction::VCount(rd, va)`      | `VCOUNT`   |
| `Instruction::VAny(rd, va)`        | `VANY`     |
| `Instruction::VAll(rd, va)`        | `VALL`     |
| `Instruction::VFirst(rd, va)`      | `VFIRST`   |
| `Instruction::VLast(rd, va)`       | `VLAST`    |
| `Instruction::VNth(rd, va, n)`     | `VNTH`     |

#### Aggregate Operators (0xE0–0xEF)

Scan segment `segId` from offset `rb` for up to `imm` rows and write result to `rd`.

| Factory                                     | Opcode              |
|---------------------------------------------|---------------------|
| `Instruction::AggCount(rd, segId)`          | `AGG_COUNT`         |
| `Instruction::AggSum(rd, segId)`            | `AGG_SUM`           |
| `Instruction::AggAvg(rd, segId)`            | `AGG_AVG`           |
| `Instruction::AggMin(rd, segId)`            | `AGG_MIN`           |
| `Instruction::AggMax(rd, segId)`            | `AGG_MAX`           |
| `Instruction::AggFirst(rd, segId)`          | `AGG_FIRST`         |
| `Instruction::AggLast(rd, segId)`           | `AGG_LAST`          |
| `Instruction::AggStddev(rd, segId)`         | `AGG_STDDEV`        |
| `Instruction::AggVariance(rd, segId)`       | `AGG_VARIANCE`      |
| `Instruction::AggCountDistinct(rd, segId)`  | `AGG_COUNT_DISTINCT`|
| `Instruction::AggSumDistinct(rd, segId)`    | `AGG_SUM_DISTINCT`  |
| `Instruction::AggMedian(rd, segId)`         | `AGG_MEDIAN`        |
| `Instruction::AggMode(rd, segId)`           | `AGG_MODE`          |
| `Instruction::AggPercentile(rd, segId)`     | `AGG_PERCENTILE`    |
| `Instruction::HashInit(rd, keyReg)`         | `HASH_INIT`         |
| `Instruction::HashProbe(rd, keyReg, hashReg)`| `HASH_PROBE`       |

#### Hash / Sort / Join (0xF0–0xFF)

| Factory                                               | Opcode          |
|-------------------------------------------------------|-----------------|
| `Instruction::HashBuild(hashReg, segId, resultSegId)` | `HASH_BUILD`    |
| `Instruction::HashLookup(hashReg, keyReg, resultSeg)` | `HASH_LOOKUP`   |
| `Instruction::SortAsc(segId, keyColReg)`              | `SORT_ASC`      |
| `Instruction::SortDesc(segId, keyColReg)`             | `SORT_DESC`     |
| `Instruction::SortTopk(segId, k, keyColReg)`          | `SORT_TOPK`     |
| `Instruction::SortBottomk(segId, k, keyColReg)`       | `SORT_BOTTOMK`  |
| `Instruction::JoinHash(leftSeg, rightSeg, resultSeg)` | `JOIN_HASH`     |
| `Instruction::JoinMerge(leftSeg, rightSeg, resultSeg)`| `JOIN_MERGE`    |
| `Instruction::JoinNested(leftSeg, rightSeg, resultSeg)`| `JOIN_NESTED`  |
| `Instruction::JoinAnti(leftSeg, rightSeg, resultSeg)` | `JOIN_ANTI`     |
| `Instruction::JoinSemi(leftSeg, rightSeg, resultSeg)` | `JOIN_SEMI`     |
| `Instruction::WindowRow(segId, startOff, endOff, kr)` | `WINDOW_ROW`    |
| `Instruction::WindowRange(segId, rangeVal, kr)`       | `WINDOW_RANGE`  |
| `Instruction::PartitionHash(segId, numParts)`         | `PARTITION_HASH`|
| `Instruction::Serialize(rd, ra)`                      | `SERIALIZE`     |
| `Instruction::Deserialize(rd, ra)`                    | `DESERIALIZE`   |

### RegFile

Defined in `<voxel/core/registers.hpp>`. 64 64-bit scalar registers, 16 256-bit vector registers, 8 mask registers, condition flags.

```cpp
RegFile regs;

regs.Scalar(0) = 42;                   // Set R0 to 42
u64 val = regs.Scalar(5);              // Read R5

f64* vec = regs.VecLanes<f64>(0);     // V0 as f64 lanes (4 lanes)
vec[0] = 1.0; vec[1] = 2.0;           // Write lanes

u32 mask = 0b1010;                     // Bit-lane mask
regs.Mask(0) = mask;                   // Set mask register M0

regs.SetFlag(RegFile::Flag::Zero);     // Set condition flag
bool isZero = regs.Test(RegFile::Flag::Zero);

regs.Reset();                          // Clear all regs, masks, flags

Constants:
  kScalarCount = 64    // 64-bit scalar registers R0..R63
  kVectorCount = 16    // 256-bit vector registers V0..V15
  kMaskCount   = 8     // 32-bit mask registers M0..M7
  kVecWidth    = 32    // bytes (256 bits)
```

### Segment\<T\>

Defined in `<voxel/data/segment.hpp>`. Zero-copy typed column view.

```cpp
// Wrap existing data (no copy)
Segment<f64> seg(externalArray, rowCount);

// Owned segment allocated from an arena
Arena arena;
Segment<f64> owned(arena, 1024);   // 1024 zero-initialized f64s

// Access
seg.Data[5] = 3.14;               // direct element access
f64 val = seg[5];                 // operator[]
sz count = seg.Count;             // number of rows

// Iteration (returns T*, range-for compatible)
for (f64 v : seg) { ... }

// Slice a sub-view (no copy)
Segment<f64> sub = seg.Slice(100, 50);

// Clone into arena
Segment<f64> clone = seg.Clone(arena);

// Statistics
f64 sum = seg.Sum();
f64 min = seg.Min();
f64 max = seg.Max();
f64 mean = seg.Mean();
```

### MutableSegment\<T\>

Growable segment backed by an arena.

```cpp
MutableSegment<f64> mut(arena, 128);  // initial capacity 128
mut.Append(1.0);
mut.Append(3.5);
mut.Append(valuesArec, 50);
mut.Sort();
mut.SortDescending();
mut.Unique();
mut.Clear();
```

### Arena

Defined in `<voxel/core/arena.hpp>`. Linear bump allocator with 64-byte alignment.

```cpp
Arena arena(65536);               // 64 KiB block size

void* p = arena.Alloc(128);       // bump-allocate 128 bytes (8-byte aligned)
T* arr = arena.AllocMany<T>(100);  // allocate 100 T elements
T* aln = arena.AllocAligned<T>(1, 64); // allocate 1 T at 64-byte boundary

arena.Reset();                    // free all blocks, reset bump pointer
sz used = arena.Used();           // total bytes actually allocated within blocks

arena.GetMetrics().Dump(std::cout); // print allocation statistics
```

Also available: `ScratchArena` (thread-local, 1 MiB, auto-reset) and `PageAllocator` (page-granularity with 4 KiB or 2 MiB pages).

### NullBitmap

Defined in `<voxel/data/nulls.hpp>`. Bitmap with 1 = valid, 0 = null. Uses 64-bit words internally.

```cpp
NullBitmap bm(1000);                // 1000 rows, all initially valid
bm.SetNull(42);                     // mark row 42 as null
bm.SetValid(42);                    // mark row 42 as valid
bool isNull = bm.IsNull(42);        // check
sz nulls    = bm.NullCount();       // count null rows
sz valids   = bm.ValidCount();      // count valid rows
bm.SetAllNull();                    // mark all rows null
bm.SetAllValid();                   // mark all rows valid
bm.Resize(2000);                    // resize (preserves existing bits)
bm.Print(std::cout);                // print as '1'/'0' string
```

### DictionaryEncoding\<T\>

Defined in `<voxel/data/encoding.hpp>`. Builds sorted unique-value table, stores indices.

```cpp
encoding::DictionaryEncoding<f64> dict;
dict.Build(data, rowCount);                     // build from raw data
f64 value = dict.Decode(500);                    // decode single row
dict.DecodeBatch(outputArray, 0, 1000);          // decode range
sz cardinality = dict.Cardinality();             // unique values
sz bytes = dict.MemoryUsage();                   // encoded size
dict.Clear();                                    // free memory
```

### RLEEncoding\<T\>

Run-length encoding.

```cpp
encoding::RLEEncoding<f64> rle;
rle.Build(data, rowCount);
f64 value = rle.Decode(500);
rle.DecodeBatch(output, 0, count);
sz runs = rle.RunCount();
rle.Clear();
```

Also available: `DeltaEncoding<T>` (integer differences), `BitPackedEncoding` (minimum-bit-width packing for u32), `FrameOfReferenceEncoding` (float scaling).

### VectorFilter\<T\>

Defined in `<voxel/ops/filter.hpp>`. Produces bitmap of row matches using 64-element chunks.

```cpp
ops::VectorFilter<f64> filter;

auto result = filter.ApplyGT(data, rowCount, 500.0);  // rows > 500
// result.Bitmap   — std::vector<u64> of bitmask words
// result.PassCount — number of matching rows

filter.ApplyGE(data, count, threshold);
filter.ApplyLT(data, count, threshold);
filter.ApplyLE(data, count, threshold);
filter.ApplyEQ(data, count, threshold);
filter.ApplyNE(data, count, threshold);
filter.ApplyBetween(data, count, lo, hi);
```

### HashAggregator

Defined in `<voxel/ops/aggregate.hpp>`. Open-addressing hash table for GROUP BY aggregation.

```cpp
Arena arena;
ops::HashAggregator<u32, f64> agg(arena);
agg.Init(1024);                              // expected number of groups
agg.Accumulate(keys, values, rowCount);      // batch accumulate
agg.AccumulateOne(key, value);               // single-row accumulate
sz groups = agg.GroupCount();                // unique groups found

agg.Finalize(outKeys, outSums, outCounts, maxGroups); // stream results
agg.Clear();
```

Also available: `StreamingAggregator<T>` (single-group streaming), `GroupedAggregator<TKey,TValue>` (pre-sorted grouped), `HybridAggregator<TKey,TValue>` (hash/sort hybrid).

### SortOperator

Defined in `<voxel/ops/sort.hpp>`. Index-based sort.

```cpp
Segment<f64> seg(data, count);
std::vector<u32> indices(count);

ops::SortOperator::SortAscending(seg, indices.data());    // stable indexed sort
ops::SortOperator::SortDescending(seg, indices.data());

// indices[0] now points to the smallest value in seg
f64 minVal = seg[indices[0]];
```

Also available: `RadixSort<T>` (LSD radix for integers), `MergeSort<T>` (bottom-up for floats), `TopK` (quickselect).

### HashJoin

Defined in `<voxel/ops/join.hpp>`. Build-probe hash join.

```cpp
Arena arena;
ops::HashJoin<f64> join(arena);
join.Build(buildKeys, buildCount);              // build hash table
u32 matchingLeft[1000], matchingRight[1000];
sz matchCount;
join.Probe(probeKeys, probeCount,
           matchingLeft, matchingRight, matchCount);
bool contains = join.Contains(someKey);
u32 rowId = join.Lookup(someKey);
join.Clear();
```

Also available: `MergeJoin<TKey>`, `AntiJoin<TKey>`, `SemiJoin<TKey>`, `NestedLoopJoin`.

### ThreadPool

Defined in `<voxel/util/thread.hpp>`. Fixed-size worker pool with shared queue.

```cpp
ThreadPool pool(4);               // 4 workers; 0 = hardware concurrency
pool.Enqueue([]() {
    // do work
});
pool.WaitAll();                   // block until all tasks complete
u32 pending = pool.PendingTasks();
```

Also available: `WorkStealingScheduler` (per-worker deques with stealing), `ParallelFor::Exec(begin, end, chunkFn)`.

### InstructionProfiler

Defined in `<voxel/system/profile.hpp>`. Per-opcode execution counts and cycle counts via `__rdtsc`.

```cpp
InstructionProfiler profiler;

// In your dispatch loop:
u64 t0 = CycleTimer::Now();
// ... execute instruction ...
u64 t1 = CycleTimer::Now();
profiler.RecordOpcode(op, t0, t1);

profiler.Dump(std::cout);         // full table, 256 rows
profiler.DumpTopN(std::cout, 10); // top 10 by cycle count
profiler.Reset();
```

Also available: `CycleTimer` (start/stop/elapsed cycles), `CacheMissCounter` (hardware L1/L3 miss counts via perf), `MemoryTracker` (global allocation tracking).

## Python Bindings

### Installation

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Module is at build/voxel_py.cpython-*.so
```

Requires `pip install pglast` for SQL support.

### Low-Level API

```python
import voxel_py as vx
import numpy as np

engine = vx.EngineF64()
engine.add_segment(np.array([1.0, 2.0, 3.0, 4.0]))
engine.set_scalar_f64(3, 2.5)
code = [vx.Instruction.vload(0, 1, 0, 0).raw, vx.Instruction.halt().raw]
engine.load_program(code); engine.run()
```

### High-Level Query Builder

```python
import voxel_query as vq
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = vq.from_numpy(data).filter_gt(4.0).sum().run()   # 26.0
result = vq.from_numpy(data).sum().run()                    # 36.0
top3   = vq.from_numpy(data).topk(3, largest=True).run()   # [8,7,6]
```

### SQL Frontend (PostgreSQL-compatible)

```python
import voxel_sql as vsql
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

result = vsql.execute("SELECT SUM(price) FROM data WHERE price > 4",
                       data={"data": data})                # 26.0

result = vsql.execute("SELECT AVG(price) FROM data WHERE price > 4",
                       data={"data": data})                # 6.5

# GROUP BY support
result = vsql.execute("SELECT SUM(price) FROM t WHERE price > 500 GROUP BY sym",
                       data={"t": prices, "sym": symbols})

# From CSV files
result = vsql.execute("SELECT SUM(col) FROM 'trades.csv' WHERE col > 500")
```

Supports: SELECT, FROM, WHERE (>, >=, <, <=, =), SUM, COUNT, AVG, GROUP BY, ORDER BY, LIMIT.

### Market Data Analytics

```python
import voxel_market as vm
md = vm.MarketData(prices=prices, volumes=volumes)

md.vwap()              # Volume-weighted average price
md.volatility(20)       # Annualized historical volatility
md.rsi(14)              # Relative Strength Index
md.macd(12, 26, 9)      # MACD + signal + histogram
md.bollinger_bands(20, 2.0)  # Upper/middle/lower bands
md.sharpe_ratio()       # Risk-adjusted return
md.max_drawdown()       # Peak-to-trough decline
md.resample_ohlc(100)   # Tick-to-bar resampling
```

### Large Data Processing

```python
import voxel_data as vd

# Memory-mapped file (100GB file on 8GB RAM — OS handles paging)
data = vd.from_file("trades.bin")
total = data.sum()
above = data.count_above(threshold=500)

# Chunked streaming
total = data.chunked_sum(chunk_size=10_000_000)
```

### Real-Time Streaming

```python
import voxel_stream as vs

def tick_generator():
    while True:
        yield get_next_tick_batch()

result = vs.stream_filter_sum(tick_generator(), threshold=500, batch_size=1024)
```

## Bytecode Reference

All 208 opcodes. Each is 8 bits embedded in a 32-bit instruction word: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`.

### Control (0x00–0x0F)

| Opcode        | Hex  | Description                     |
|---------------|------|---------------------------------|
| `NOP`         | 0x00 | No operation                    |
| `HALT`        | 0x01 | Stop execution                  |
| `TRAP`        | 0x02 | Trap with error flag            |
| `BREAK`       | 0x03 | Breakpoint                      |
| `YIELD`       | 0x04 | Yield to scheduler              |
| `BARRIER`     | 0x05 | Memory barrier                  |
| `PREFETCH`    | 0x06 | Prefetch segment data           |
| `FLUSH_CACHE` | 0x07 | Flush CPU cache                 |
| `SYNC`        | 0x08 | Thread sync barrier             |
| `MEMFENCE`    | 0x09 | Full memory fence               |

### Scalar Move / Immediate (0x10–0x1F)

| Opcode | Hex  | Description              |
|--------|------|--------------------------|
| `MOV`  | 0x10 | Rd = sext(imm12)         |
| `MOVR` | 0x11 | Rd = Ra (copy)           |
| `ADDI` | 0x12 | Rd = Ra + sext(imm12)    |
| `SUBI` | 0x13 | Rd = Ra - sext(imm12)    |
| `MULI` | 0x14 | Rd = Ra * sext(imm12)    |
| `ANDI` | 0x15 | Rd = Ra & imm12          |
| `ORI`  | 0x16 | Rd = Ra \| imm12         |
| `XORI` | 0x17 | Rd = Ra ^ imm12          |
| `SHLI` | 0x18 | Rd = Ra << (imm12 & 0x3F)|
| `SHRI` | 0x19 | Rd = Ra >> (imm12 & 0x3F)|
| `SAR_I`| 0x1A | Rd = Ra >>a (imm12 & 0x3F)|
| `MOVZ` | 0x1B | Conditional move (Zero)  |
| `MOVN` | 0x1C | Conditional move (!Zero) |
| `MOVK` | 0x1D | Move 16-bit field        |
| `LEA`  | 0x1E | Rd = Ra + sext(imm12)    |

### Scalar Arithmetic (0x20–0x2F)

| Opcode | Hex  | Description          |
|--------|------|----------------------|
| `ADD`  | 0x20 | Rd = Ra + Rb (i64)   |
| `SUB`  | 0x21 | Rd = Ra - Rb (i64)   |
| `MUL`  | 0x22 | Rd = Ra * Rb (i64)   |
| `DIV`  | 0x23 | Rd = Ra / Rb (i64)   |
| `MOD`  | 0x24 | Rd = Ra % Rb (i64)   |
| `NEG`  | 0x25 | Rd = -Ra (i64)       |
| `ABS`  | 0x26 | Rd = \|Ra\| (i64)    |
| `MIN`  | 0x27 | Rd = min(Ra,Rb) (i64)|
| `MAX`  | 0x28 | Rd = max(Ra,Rb) (i64)|
| `AVG`  | 0x29 | Rd = (Ra+Rb)/2 (i64) |
| `ADDF` | 0x2A | Rd = Ra + Rb (f64)   |
| `SUBF` | 0x2B | Rd = Ra - Rb (f64)   |
| `MULF` | 0x2C | Rd = Ra * Rb (f64)   |
| `DIVF` | 0x2D | Rd = Ra / Rb (f64)   |
| `NEGF` | 0x2E | Rd = -Ra (f64)       |
| `ABSF` | 0x2F | Rd = \|Ra\| (f64)    |

### Scalar Bitwise (0x30–0x3F)

| Opcode  | Hex  | Description              |
|---------|------|--------------------------|
| `AND`   | 0x30 | Rd = Ra & Rb             |
| `OR`    | 0x31 | Rd = Ra \| Rb            |
| `XOR`   | 0x32 | Rd = Ra ^ Rb             |
| `NOT`   | 0x33 | Rd = ~Ra                 |
| `SHL`   | 0x34 | Rd = Ra << Rb            |
| `SHR`   | 0x35 | Rd = Ra >> Rb            |
| `SAR`   | 0x36 | Rd = Ra >>a Rb           |
| `ROL`   | 0x37 | Rd = Ra rotl Rb          |
| `ROR`   | 0x38 | Rd = Ra rotr Rb          |
| `POPCNT`| 0x39 | Rd = popcount(Ra)        |
| `CLZ`   | 0x3A | Rd = count-leading-zeros |
| `CTZ`   | 0x3B | Rd = count-trailing-zeros|
| `BSWAP` | 0x3C | Rd = byte-swap(Ra)       |
| `BEXTR` | 0x3D | Rd = bit-extract(Ra, Rb) |
| `BZHI`  | 0x3E | Rd = zero-high-bits(Ra,Rb)|
| `PDEP`  | 0x3F | Rd = parallel-deposit(Ra,Rb)|

### Comparison (0x40–0x4F)

| Opcode     | Hex  | Description                 |
|------------|------|-----------------------------|
| `CMP`      | 0x40 | Compare Ra, Rb (i64), sets flags |
| `CMPF`     | 0x41 | Compare Ra, Rb (f64), sets flags |
| `CMPU`     | 0x42 | Compare Ra, Rb (u64), sets flags |
| `TST`      | 0x43 | AND Ra, Rb, set flags on result |
| `TSTF`     | 0x44 | Compare Ra, Rb (f64), set numeric flags |
| `ISNULL`   | 0x45 | Rd = (Ra == 0) ? 1 : 0      |
| `ISNOTNULL`| 0x46 | Rd = (Ra != 0) ? 1 : 0      |
| `SELECT`   | 0x47 | Rd = Zero ? Ra : Rb          |
| `SELECTV`  | 0x48 | Rd = mask[lane] ? Rb : Ra    |

### Branching (0x50–0x5F)

| Opcode     | Hex  | Condition            |
|------------|------|----------------------|
| `JMP`      | 0x50 | Unconditional        |
| `JZ`       | 0x51 | Zero flag            |
| `JNZ`      | 0x52 | Not zero             |
| `JS`       | 0x53 | Sign flag            |
| `JNS`      | 0x54 | Not sign             |
| `JO`       | 0x55 | Overflow             |
| `JNO`      | 0x56 | Not overflow         |
| `JC`       | 0x57 | Carry                |
| `JNC`      | 0x58 | Not carry            |
| `JL`       | 0x59 | Less than            |
| `JLE`      | 0x5A | Less or equal        |
| `JG`       | 0x5B | Greater than         |
| `JGE`      | 0x5C | Greater or equal     |
| `CALL`     | 0x5D | Subroutine call      |
| `RET`      | 0x5E | Return from call     |
| `TABLE_JMP`| 0x5F | PC = baseRa + indexRb|

### Type Conversion (0x60–0x6F)

| Opcode      | Hex  | Description           |
|-------------|------|-----------------------|
| `CVT_I8`    | 0x60 | Rd = (i8)Ra          |
| `CVT_I16`   | 0x61 | Rd = (i16)Ra         |
| `CVT_I32`   | 0x62 | Rd = (i32)Ra         |
| `CVT_I64`   | 0x63 | Rd = (i64)Ra         |
| `CVT_F32`   | 0x64 | Rd = (f32)Ra         |
| `CVT_F64`   | 0x65 | Rd = (f64)Ra         |
| `CVT_U8`    | 0x66 | Rd = (u8)Ra          |
| `CVT_U16`   | 0x67 | Rd = (u16)Ra         |
| `CVT_U32`   | 0x68 | Rd = (u32)Ra         |
| `CVT_U64`   | 0x69 | Rd = (u64)Ra         |
| `BITCAST`   | 0x6A | Rd = Ra (raw u64 copy)|
| `REINTERPRET`| 0x6B| Rd = T(Ra) via reinterpret|
| `TRUNC`     | 0x6C | Rd = trunc(Ra)       |
| `ROUND`     | 0x6D | Rd = round(Ra)       |
| `CEIL`      | 0x6E | Rd = ceil(Ra)        |
| `FLOOR`     | 0x6F | Rd = floor(Ra)       |

### Vector I/O (0x70–0x7F)

| Opcode          | Hex  | Description                      |
|-----------------|------|----------------------------------|
| `VLOAD`         | 0x70 | Vd[i] = seg[Ra + i]              |
| `VSTORE`        | 0x71 | seg[Ra + i] = Vs[i]              |
| `VGATHER`       | 0x72 | Vd[i] = seg[indices[i]]          |
| `VSCATTER`      | 0x73 | seg[indices[i]] = Vs[i]          |
| `VLOAD_STRIDED` | 0x74 | Vd[i] = seg[Ra + i * Rb]         |
| `VSTORE_STRIDED`| 0x75 | seg[Ra + i * Rb] = Vs[i]         |
| `VLOAD_MASKED`  | 0x76 | Vd[i] = mask[i] ? seg[Ra+i] : Vd[i]|
| `VSTORE_MASKED` | 0x77 | mask[i] ? seg[Ra+i] = Vs[i]      |
| `VSPLAT`        | 0x78 | Vd[i] = Ra (broadcast scalar)    |
| `VEXTRACT`      | 0x79 | Rd = Va[lane]                    |
| `VINSERT`       | 0x7A | Vd[lane] = Rb                    |
| `VPERMUTE`      | 0x7B | Vd[i] = Va[idxVec[i]]            |
| `VSHUFFLE`      | 0x7C | Vd = shuffle(Va, Vb, mode)       |
| `VREVERSE`      | 0x7D | Vd[i] = Va[N-1-i]                |
| `VROTATE`       | 0x7E | Vd[i] = Va[(i+sh) % N]           |
| `VSLIDE`        | 0x7F | Vd[i] = Va[i+sh]                 |

### Vector Arithmetic (0x80–0x8F)

| Opcode | Hex  | Description          |
|--------|------|----------------------|
| `VADD` | 0x80 | Vd = Va + Vb         |
| `VSUB` | 0x81 | Vd = Va - Vb         |
| `VMUL` | 0x82 | Vd = Va * Vb         |
| `VDIV` | 0x83 | Vd = Va / Vb         |
| `VMOD` | 0x84 | Vd = Va % Vb         |
| `VNEG` | 0x85 | Vd = -Va             |
| `VABS` | 0x86 | Vd = \|Va\|          |
| `VMIN` | 0x87 | Vd = min(Va, Vb)     |
| `VMAX` | 0x88 | Vd = max(Va, Vb)     |
| `VAVG` | 0x89 | Vd = (Va + Vb) / 2   |
| `VFMA` | 0x8A | Vd = Va * Vb + c     |
| `VFMS` | 0x8B | Vd = Va * Vb - c     |
| `VSQRT`| 0x8C | Vd = sqrt(Va)        |
| `VRSQRT`| 0x8D| Vd = 1 / sqrt(Va)    |
| `VRCP` | 0x8E | Vd = 1 / Va          |
| `VPOW` | 0x8F | Vd = Va ^ Vb         |

### Vector-Scalar Arithmetic (0x90–0x9F)

| Opcode | Hex  | Description          |
|--------|------|----------------------|
| `VSADD`| 0x90 | Vd = Va + Rb         |
| `VSSUB`| 0x91 | Vd = Va - Rb         |
| `VSMUL`| 0x92 | Vd = Va * Rb         |
| `VSDIV`| 0x93 | Vd = Va / Rb         |
| `VSMOD`| 0x94 | Vd = Va % Rb         |
| `VSMIN`| 0x95 | Vd = min(Va, Rb)     |
| `VSMAX`| 0x96 | Vd = max(Va, Rb)     |

### Vector Comparison (0xA0–0xAF)

| Opcode       | Hex  | Description                |
|--------------|------|----------------------------|
| `VCMPEQ`     | 0xA0 | Vd = Va == Vb (all-bits)   |
| `VCMPNE`     | 0xA1 | Vd = Va != Vb              |
| `VCMPLT`     | 0xA2 | Vd = Va <  Vb              |
| `VCMPLE`     | 0xA3 | Vd = Va <= Vb              |
| `VCMPGT`     | 0xA4 | Vd = Va >  Vb              |
| `VCMPGE`     | 0xA5 | Vd = Va >= Vb              |
| `VCMPNULL`   | 0xA6 | Vd = Va is null            |
| `VCMPNOTNULL`| 0xA7 | Vd = Va is not null        |

### Vector Logical (0xB0–0xBF)

| Opcode | Hex  | Description          |
|--------|------|----------------------|
| `VAND` | 0xB0 | Vd = Va & Vb (bitwise)|
| `VOR`  | 0xB1 | Vd = Va \| Vb        |
| `VXOR` | 0xB2 | Vd = Va ^ Vb         |
| `VNOT` | 0xB3 | Vd = ~Va             |
| `VANDN`| 0xB4 | Vd = ~Va & Vb        |
| `VSHL` | 0xB5 | Vd = Va << Vb        |
| `VSHR` | 0xB6 | Vd = Va >> Vb        |
| `VSAR` | 0xB7 | Vd = Va >>a Vb       |

### Vector Filter (0xC0–0xCF)

| Opcode       | Hex  | Description                |
|--------------|------|----------------------------|
| `VFILTER`    | 0xC0 | Vd = Va[pred] (mode in imm)|
| `VFILTER_EQ` | 0xC1 | Vd = Va if Va == Rb        |
| `VFILTER_NE` | 0xC2 | Vd = Va if Va != Rb        |
| `VFILTER_LT` | 0xC3 | Vd = Va if Va <  Rb        |
| `VFILTER_LE` | 0xC4 | Vd = Va if Va <= Rb        |
| `VFILTER_GT` | 0xC5 | Vd = Va if Va >  Rb        |
| `VFILTER_GE` | 0xC6 | Vd = Va if Va >= Rb        |
| `VBLEND`     | 0xC7 | Vd = mask ? Vb : Va        |
| `VMASK_STORE`| 0xC8 | seg[Ra+i] = Vs[i] if mask[i]|
| `VMASK_LOAD` | 0xC9 | Vd[i] = seg[Ra+i] if mask[i]|

### Vector Reduction (0xD0–0xDF)

| Opcode     | Hex  | Description              |
|------------|------|--------------------------|
| `VSUM`     | 0xD0 | Rd = sum(Va)             |
| `VPROD`    | 0xD1 | Rd = product(Va)         |
| `VMEAN`    | 0xD2 | Rd = mean(Va)            |
| `VSTDDEV`  | 0xD3 | Rd = stddev(Va)          |
| `VVARIANCE`| 0xD4 | Rd = variance(Va)        |
| `VRED_MIN` | 0xD5 | Rd = min(Va)             |
| `VRED_MAX` | 0xD6 | Rd = max(Va)             |
| `VCOUNT`   | 0xD7 | Rd = count non-zero(Va)  |
| `VANY`     | 0xD8 | Rd = any non-zero(Va)    |
| `VALL`     | 0xD9 | Rd = all non-zero(Va)    |
| `VFIRST`   | 0xDA | Rd = Va[0]               |
| `VLAST`    | 0xDB | Rd = Va[N-1]             |
| `VNTH`     | 0xDC | Rd = Va[n]               |

### Aggregate (0xE0–0xEF)

| Opcode               | Hex  | Description                   |
|----------------------|------|-------------------------------|
| `AGG_COUNT`          | 0xE0 | Count non-zero in seg[ra..]   |
| `AGG_SUM`            | 0xE1 | Sum seg[ra..]                 |
| `AGG_AVG`            | 0xE2 | Average seg[ra..]             |
| `AGG_MIN`            | 0xE3 | Min seg[ra..]                 |
| `AGG_MAX`            | 0xE4 | Max seg[ra..]                 |
| `AGG_FIRST`          | 0xE5 | First seg[ra]                 |
| `AGG_LAST`           | 0xE6 | Last seg[end-1]               |
| `AGG_STDDEV`         | 0xE7 | StdDev (Welford)              |
| `AGG_VARIANCE`       | 0xE8 | Variance (Welford)            |
| `AGG_COUNT_DISTINCT` | 0xE9 | Count distinct                |
| `AGG_SUM_DISTINCT`   | 0xEA | Sum distinct                  |
| `AGG_MEDIAN`         | 0xEB | Median                        |
| `AGG_MODE`           | 0xEC | Mode (most frequent)          |
| `AGG_PERCENTILE`     | 0xED | Percentile (imm12 = pct*100)  |
| `HASH_INIT`          | 0xEE | Initialize hash table         |
| `HASH_PROBE`         | 0xEF | Probe hash table by key       |

### Hash / Sort / Join (0xF0–0xFF)

| Opcode          | Hex  | Description                    |
|-----------------|------|--------------------------------|
| `HASH_BUILD`    | 0xF0 | Build hash table from segment  |
| `HASH_LOOKUP`   | 0xF1 | Lookup key in hash table       |
| `SORT_ASC`      | 0xF2 | Sort segment ascending         |
| `SORT_DESC`     | 0xF3 | Sort segment descending        |
| `SORT_TOPK`     | 0xF4 | Partial sort, top K            |
| `SORT_BOTTOMK`  | 0xF5 | Partial sort, bottom K         |
| `JOIN_HASH`     | 0xF6 | Hash join two segments         |
| `JOIN_MERGE`    | 0xF7 | Merge join two segments        |
| `JOIN_NESTED`   | 0xF8 | Nested-loop join               |
| `JOIN_ANTI`     | 0xF9 | Anti join (left unmatched)     |
| `JOIN_SEMI`     | 0xFA | Semi join (left matched)       |
| `WINDOW_ROW`    | 0xFB | Row-based window function      |
| `WINDOW_RANGE`  | 0xFC | Range-based window function    |
| `PARTITION_HASH`| 0xFD | Hash-partition a segment       |
| `SERIALIZE`     | 0xFE | Copy segment to arena buffer   |
| `DESERIALIZE`   | 0xFF | Copy from buffer to segment    |

## Example Programs

### Filter + Sum via Interpreter

Bytecode engine that filters rows where `value > threshold` and sums the results. Uses 8-instruction loop: VLoad a batch, VFilterGt against threshold, VSum reduction, Addf to accumulator, Add offset, Cmp limit, Jnz back.

```cpp
#include "voxel/voxel.hpp"
#include <vector>
#include <bit>
using namespace voxel;

int main() {
    constexpr sz N = 1'000'000;
    constexpr f64 kThreshold = 500.0;
    constexpr sz kLanes = Engine<f64>::kLanes;

    std::vector<f64> data(N);
    for (sz i = 0; i < N; ++i) data[i] = static_cast<f64>(i % 1000);

    Engine<f64> engine;
    engine.ScalarReg(0) = 0;                                    // R0 = accumulator
    engine.ScalarReg(1) = 0;                                    // R1 = offset cursor
    engine.ScalarReg(2) = N;                                    // R2 = total count
    engine.ScalarReg(3) = std::bit_cast<u64>(kThreshold);       // R3 = threshold
    engine.ScalarReg(4) = kLanes;                                // R4 = batch size
    sz segId = engine.AddSegment(data.data(), N);

    std::vector<u32> code = {
        Instruction::VLoad(0, 1, static_cast<u8>(segId), 0).raw,   // V0 = seg[R1..]
        Instruction::VFilterGt(1, 0, 3).raw,                        // V1 = V0 > R3
        Instruction::VSum(5, 1).raw,                                 // R5 = sum(V1)
        Instruction::Addf(0, 0, 5).raw,                              // R0 += R5
        Instruction::Add(1, 1, 4).raw,                               // R1 += kLanes
        Instruction::Cmp(1, 2).raw,                                  // CMP R1,R2; set flags
        Instruction::Jnz(-6).raw,                                    // if R1!=R2 goto VLoad
        Instruction::Halt().raw,
    };

    engine.LoadProgram(code);
    engine.Run();

    f64 result = std::bit_cast<f64>(engine.ScalarReg(0));
    // Use result...
    return 0;
}
```

### Sort via SortOperator

Index-based merge sort on a columnar segment, producing a permutation vector.

```cpp
Segment<f64> seg(data, count);
std::vector<u32> indices(count);

ops::SortOperator::SortAscending(seg, indices.data());

// Original data at seg[indices[0]] = smallest, seg[indices[count-1]] = largest
// Stable sort preserves input order for equal keys
```

### Hash Aggregation

Group-by with sum/min/max/count per key using open-addressing hash table.

```cpp
Arena arena;
ops::HashAggregator<u32, f64> agg(arena);
agg.Init(128);                                    // expected group count

std::vector<u32> keys(1000);
std::vector<f64> values(1000);
for (u32 i = 0; i < 1000; ++i) {
    keys[i] = i % 10;                            // 10 groups
    values[i] = data[i];
}

agg.Accumulate(keys.data(), values.data(), 1000);
sz groupCount = agg.GroupCount();                // 10

std::vector<u32> outKeys(groupCount);
std::vector<f64> outSums(groupCount);
std::vector<sz> outCounts(groupCount);
agg.Finalize(outKeys.data(), outSums.data(), outCounts.data(), groupCount);
```

### JIT Compilation and Execution

Compile bytecode to native x86-64 machine code and execute.

```cpp
auto compiler = codegen::CreateJitCompiler();
if (!compiler) return;  // no backend available for this platform

std::vector<u32> code = {
    Instruction::VLoad(0, 1, segId, 0).raw,
    Instruction::VFilterGt(1, 0, 3).raw,
    Instruction::VSum(5, 1).raw,
    Instruction::Addf(0, 0, 5).raw,
    Instruction::Add(1, 1, 4).raw,
    Instruction::Cmp(1, 2).raw,
    Instruction::Jnz(-6).raw,
    Instruction::Halt().raw,
};

codegen::JitFunction func;
bool ok = compiler->Compile(code.data(), code.size(), func);
if (!ok || !func.IsValid()) return;

// Prepare execution context
RegFile regfile;
regfile.Scalar(0) = 0;                             // accumulator
regfile.Scalar(1) = 0;                             // offset
regfile.Scalar(2) = N;                             // total
regfile.Scalar(3) = std::bit_cast<u64>(threshold);  // threshold
regfile.Scalar(4) = kLanes;

f64* segBase = data.data();
u64  segCount = N;

// Execute native code
func.Entry(&regfile, &segBase, &segCount);

f64 result = std::bit_cast<f64>(regfile.Scalar(0));
compiler->Release(func);
```

### NullBitmap and Dictionary Encoding

```cpp
// Null tracking
NullBitmap nulls(1'000'000);
for (sz i = 0; i < 1'000'000; i += 1000)
    nulls.SetNull(i);                              // 0.1% nulls
sz nullCount = nulls.NullCount();

// Dictionary compression
encoding::DictionaryEncoding<f64> dict;
dict.Build(data, 1'000'000);                       // build dictionary
sz cardinality = dict.Cardinality();               // unique values
sz encBytes = dict.MemoryUsage();                  // encoded size

std::vector<f64> decoded(1'000'000);
dict.DecodeBatch(decoded.data(), 0, 1'000'000);    // decode all rows
```

### Arena-Based Memory

```cpp
Arena arena;

// Bump-allocate arrays
i64* colA = arena.AllocMany<i64>(100'000);
f64* colB = arena.AllocMany<f64>(100'000);

// Fixed-capacity segment
Segment<f64> seg(arena, 1'000'000);

// Growable segment
MutableSegment<f64> mut(arena, 1024);
for (sz i = 0; i < 1000; ++i) mut.Append(data[i]);

// Reset between queries
arena.Reset();
```

## Performance Guide

### End-to-end filter and sum (1M f64 rows)

| Method | Throughput (M elem/s) | vs Native |
|--------|----------------------|-----------|
| Native C++ (scalar SIMD) | 185 | 1.0x |
| VoxelVM interpreter | 53 | 3.5x |
| VoxelVM JIT (fusion kernel) | 33 | 5.6x |

The interpreter uses a 256-entry function-pointer dispatch table with inline handlers. The JIT uses a vector accumulation fusion kernel: it detects VLOAD→VFILTER_GT→VSUM→ADDF patterns and emits a fused loop that keeps values in ymm registers with the accumulator as 4 parallel partial sums, reduced only at loop exit. The JIT uses SIB-based addressing (`vmovupd [r15+r8*8]`), countdown loops, and 4-way unrolling on the Intel i3 test machine. On modern CPUs (Zen 3+, Ice Lake+) with wider execution ports, unrolling yields 2-3x higher throughput.

The JIT allocates memory as RW, writes native code, then `mprotect` to RX before execution, avoiding W^X restrictions that block certain AVX instructions on hardened kernels.

### Dispatch method comparison

| Dispatch method | Throughput (M elem/s) |
|-----------------|----------------------|
| Switch statement | 34 |
| Function pointer table | 53 (+56%) |

### Subsystem benchmarks (Intel i3-4130T, GCC 15.2, -O3 -march=native)

| Subsystem | Throughput | Unit |
|-----------|------------|------|
| VectorFilter f64 ApplyGT | 670 | M elem/s |
| NullBitmap set+count (10% null) | 9,259 | M elem/s |
| Arena Allocator (100 x 1M i32) | 13,154 | GB/s |
| HashAggregator GROUP BY (100K rows, 100 groups) | 86 | M rows/s |
| HashJoin build+probe (10K keys) | 80 | M keys/s |
| SortOperator index sort (1M f64) | 6.1 | M elem/s |
| DictionaryEncoding f64 build+decode (1M) | 1.2 | M elem/s |
| RLEEncoding f64 build (1M, 10 runs) | 251 | M elem/s |
| ThreadPool parallel speedup (4 threads) | 1.75x | speedup |
| JIT compile (filter+sum bytecode) | 46 | us |
| JIT execute (1M f64 native code) | 33 | M elem/s |

## Contributing

Submit issues and pull requests at the project repository.

### Code Style

- PascalCase for types: `Engine`, `RegFile`, `Segment`, `HashAggregator`
- camelCase for variables and methods: `engine.Run()`, `regs.Scalar(0)`, `seg.Count`
- `k` prefix for compile-time constants: `kLanes`, `kVerticalCount`
- `VOXEL_` prefix for macros: `VOXEL_ALWAYS_INLINE`, `VOXEL_RESTRICT`, `VOXEL_ASSERT`
- Namespace `voxel` for core types, `voxel::ops` for operators, `voxel::encoding` for encodings, `voxel::codegen` for JIT

### Build Verification

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/voxel-test        # 22-subsystem validation suite
./build/voxel-demo        # performance benchmarks
./build/voxel-jit-test    # JIT vs interpreter correctness
ctest --test-dir build    # run all three
```

All code compiles under `-Wall -Wextra -Werror`. Ensure new changes pass this before submitting.

### Adding an Opcode

1. Add the enum value in `voxel/bytecode/opcodes.hpp` (pick an unused slot in the correct range)
2. Add the `OpcodeName` case in the same file
3. Add the static factory method in `voxel/bytecode/instruction.hpp`
4. Add the dispatch handler in `voxel/exec/engine.hpp` (static method + dispatch table entry)
5. Add the JIT translation in the platform-specific codegen file
6. Add a test case in `examples/test_suite.cpp`

### Adding an Operator

Place the header in `include/voxel/ops/`, include it from `voxel/voxel.hpp`, and follow the existing patterns: arena-backed allocators, `VOXEL_RESTRICT` pointer parameters, `VOXEL_HOT` marking on inner loops, and a public `Reset()`/`Clear()` method.
