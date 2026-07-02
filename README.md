# VoxelVM

A bytecode virtual machine engine for analytical queries on columnar data.

## Problem

Columnar analytics engines spend most of their cycles inside tight loops: load a batch of values, compare against a threshold, accumulate the ones that pass, advance the cursor, repeat. Writing these loops by hand in C++ is fast but rigid. Writing them in SQL or a dataframe library gives flexibility at the cost of dispatch overhead. A bytecode VM sits in between: the query is compiled once into a dense instruction stream, then the VM runs it with minimal per-row branching.

Most bytecode VMs are either toys that handle six opcodes and two types, or full JVM/CLR-class runtimes that include garbage collection, class loading, and a large baseline. VoxelVM targets the middle ground: it handles 208 opcodes across f64, f32, i64, i32, u64, and u32, monomorphizes the entire engine per element type, and distributes as a single header tree under one `#include`.

## Quick start

```sh
git clone https://github.com/Ragekill3377/Voxel
cd Voxel
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/voxel-demo       # 25 subsystem benchmarks with ground-truth checks
./build/voxel-test        # validation suite
./build/voxel-jit-test    # JIT correctness: interpreter vs native
ctest --test-dir build    # all three
```

## Architecture

```
+-----------------------------------------------------------+
|                    VoxelVM Architecture                    |
+------------+------------+------------+------------+--------+
|  bytecode  |    exec    |  codegen   |    simd    |  data  |
|  opcodes   |   engine   |    jit     |    avx2    |segment |
|   instr    |  dispatch  |   x86-64   |   avx512   |encode  |
| assembler  |  runtime   |   arm64    |    neon    | nulls  |
|  verifier  |  profile   |   cache    |   scalar   | blocks |
+------------+------------+------------+------------+--------+
|  ops: filter | aggregate | sort | join | hash           |
|  system: telemetry | error | debug | profile           |
|  util: bit | math | hash | string | thread             |
+-----------------------------------------------------------+
```

### Core

The type system defines 30 runtime data type identifiers plus a `TypeTraits<T>` template that maps each C++ primitive to its size, alignment, signedness, and runtime ID. `ScalarValue` is a 16-byte type-erased container used at decode boundaries.

The register file has 64 64-bit scalar registers, 16 256-bit vector registers, 8 mask registers, and a flags word (Zero, Carry, Sign, Overflow, NaN, DivByZero). Vector registers are typed at access time via `VecLanes<T>(idx)`, which returns a `T*` into the aligned backing store. The engine template parameter `T` determines the interpretation.

The arena allocator is a bump-pointer with 64 KiB default block size and 64-byte alignment. It tracks total allocated, peak, allocation count, and block count via atomic counters.

### Bytecode

Instructions are 32-bit words: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`. For vector loads and stores, the immediate is subdivided: bits 11:8 encode the segment ID, bits 7:0 encode the element count (0 means "fill the vector register").

The opcode space has 208 entries: control, scalar move, scalar arithmetic, scalar bitwise, comparison, branch, type conversion, vector I/O, vector arithmetic, vector-scalar, vector compare, vector logical, vector filter, vector reduction, aggregate, and hash/sort/join.

The assembler is a programmatic builder with a label/relocation system. Named labels like `Assembler::Bind("loop")` and `Assembler::Jnz("loop")` are resolved during `Finalize()`. The verifier checks opcode validity, branch target bounds, register range, segment range, and halt reachability via DFS. The disassembler pretty-prints every instruction with hex addresses and decoded operands.

### Execution engine

`Engine<T>` owns a `RegFile`, an arena, a vector of `Segment<T>` views, and a bytecode vector. The dispatch mechanism uses a 256-entry static table of function pointers, one per opcode:

```cpp
static void HandleVADD_f64(Engine* e, u32 raw) {
    u8 rd = Rd(raw), ra = Ra(raw), rb = Rb(raw);
    f64* dst = e->Regs_.VecLanes<f64>(rd);
    const f64* a = e->Regs_.VecLanes<f64>(ra);
    const f64* b = e->Regs_.VecLanes<f64>(rb);
    for (sz i = 0; i < 4; ++i) dst[i] = a[i] + b[i];
    e->PC_++;
}
```

Each handler is marked `VOXEL_ALWAYS_INLINE`. The `Run()` method fetches a raw instruction word, masks out the opcode byte, indexes the table, and calls the handler. This is faster than a large switch statement because each handler is a leaf function the compiler can optimize independently, and the indirect branch target is fully predictable after the first iteration of a loop.

Vector filters are handled per-comparison-mode. `VFILTER_GT`, `VFILTER_GE`, `VFILTER_LT`, etc. each have their own dispatch slot, eliminating the inner `switch(mode)` that a unified VFILTER would require.

### JIT code generation

The x86-64 backend emits actual machine code bytes, handling REX, VEX2, VEX3, ModR/M, and SIB encoding. `X64Compiler::Compile()` walks the bytecode instruction by instruction and emits native code. A VLOAD becomes `VMOVUPD ymm0, [r8 + rax]` (segment base in r8, offset in rax). A VFILTER_GT becomes `VCMPPD ymm2, ymm0, ymm1, 14` (ordered greater-than) followed by `VBLENDVPD`. Branches are resolved to relative offsets with backpatching.

The `JitMemoryManager` allocates executable pages via `mmap` with `PROT_READ | PROT_WRITE | PROT_EXEC` on Linux, or `VirtualAlloc` with `PAGE_EXECUTE_READWRITE` on Windows. After code generation, the page is marked read-execute only. `JitCache` is an LRU cache keyed by bytecode content hash.

### SIMD layer

Four backends, selected at compile time by `#if` on `__AVX512F__`, `__AVX2__`, `__ARM_NEON`, or the portable fallback. Each provides `Vec<T>` specializations with `Load`, `Store`, `Set1`, `Zero`, element-wise arithmetic, comparison, blend, gather/scatter, and horizontal reduction. The x86 backend uses `__m256d`/`__m256`/`__m256i` for AVX2 and `__m512d`/`__m512`/`__m512i` for AVX-512. The scalar fallback uses plain arrays and counted loops; the compiler auto-vectorizes these on platforms without SIMD intrinsics.

Bulk free functions like `simd::scalar::scalar_filter_gt_f64` process arrays in vector-sized chunks with a scalar tail.

### Data layer

`Segment<T>` is a zero-copy typed view over external memory. It holds `T* Data`, `sz Count`, and optional ownership tracking. Its iterators are raw `T*`, so range-for loops compile to direct pointer arithmetic.

Five columnar encodings: `DictionaryEncoding<T>` (sorted unique-value table), `RLEEncoding<T>` ((value, run-length) pairs), `DeltaEncoding<T>` (difference from previous row), `BitPackedEncoding` (minimum bit width integer packing), and `FrameOfReferenceEncoding` (floating-point to integer scaling).

`NullBitmap` packs one bit per row into `u64` words. `NullableSegment<T>` pairs a Segment with a NullBitmap and provides a sentinel value for null propagation during vectorized operations. `DataBlock<T>` is a fixed-capacity chunk for columnar batch processing.

### Operators

`VectorFilter<T>` processes data in 256-element chunks, writes a 64-bit word per chunk into a bitmap, and counts passes via hardware popcount. `SelectionVector` converts bitmaps to index lists.

`HashAggregator` uses open addressing with linear probing on a power-of-two table. It accumulates sum, min, max, count, and a Welford online variance state per group.

The sort module provides LSD radix sort for integers (8 counting passes, one per byte) and bottom-up merge sort for floating point. `TopK` uses median-of-three quickselect for partial sorting.

The join module implements hash join (build hash table on smaller side, probe with larger), merge join, anti join, semi join, and nested loop join.

### System and utilities

Profiling tracks per-opcode execution counts and cycle counts using `__rdtsc`. Telemetry exports to JSON and CSV. Error handling uses a thread-local handler chain. Debug utilities include hex dumps with ASCII sidebars and register file snapshots.

Utilities include bit manipulation (popcount, count-leading-zeros, bit reverse), three hash functions (MurmurHash64A, XXHash64, FarmHash64), fast inverse square root, a stack-allocated string builder, a thread pool with work stealing, and a parallel-for executor.

## Instruction set

208 opcodes organized as follows. Every opcode is handled by a dedicated dispatch handler.

| Range | Count | Category | Representative instructions |
|-------|-------|----------|-----------------------------|
| 0x00 | 10 | Control | NOP, HALT, TRAP, BARRIER |
| 0x10 | 15 | Scalar Move | MOV, ADDI, SUBI, LEA |
| 0x20 | 16 | Scalar Arithmetic | ADD, SUB, MUL, DIV, MOD, ADDF..ABSF |
| 0x30 | 16 | Scalar Bitwise | AND, OR, XOR, NOT, POPCNT, CLZ, CTZ |
| 0x40 | 9 | Comparison | CMP, CMPF, CMPU, ISNULL, SELECT |
| 0x50 | 16 | Branching | JMP, JZ, JNZ, JL, JG, CALL, RET |
| 0x60 | 16 | Type Conversion | CVT_I64, CVT_F64, BITCAST, ROUND, CEIL |
| 0x70 | 16 | Vector I/O | VLOAD, VSTORE, VGATHER, VSPLAT, VPERMUTE |
| 0x80 | 16 | Vector Arithmetic | VADD, VSUB, VMUL, VDIV, VFMA, VSQRT |
| 0x90 | 7 | Vector-Scalar | VSADD, VSSUB, VSMUL, VSDIV |
| 0xA0 | 8 | Vector Compare | VCMPEQ..VCMPGE |
| 0xB0 | 8 | Vector Logical | VAND, VOR, VXOR, VNOT |
| 0xC0 | 10 | Vector Filter | VFILTER_EQ..VFILTER_GE, VBLEND |
| 0xD0 | 13 | Vector Reduction | VSUM, VMEAN, VSTDDEV, VCOUNT, VANY |
| 0xE0 | 16 | Aggregate | AGG_COUNT/SUM/AVG/MIN/MAX/STDDEV |
| 0xF0 | 16 | Hash/Sort/Join | HASH_BUILD, SORT_ASC, JOIN_HASH |

## Performance

### Methodology

All measurements taken on an Intel Core i3-4130T (Haswell, 2 cores, 4 threads, AVX2) at 2.90 GHz, GCC 15.2.0, `-O3 -march=native`, Linux 6.x. The filter+sum workload uses 1,000,000 random f64 values drawn uniformly from [0, 1000) with a threshold of 500.0. Each benchmark runs 5 iterations after 2 warmup rounds; reported results are the median. Ground truth sum is 375,538,740.218 across 500,720 passing rows.

### End-to-end filter and sum (1M rows)

| Method | Time (us) | Throughput (M elem/s) | Overhead vs native |
|--------|-----------|----------------------|---------------------|
| Native C++ (scalar SIMD) | 5,400 | 185 | 1.0x |
| VoxelVM interpreter | 18,748 | 53 | 3.5x |
| VoxelVM JIT (fusion kernel) | 700 | **1,400** | **0.13x** |

The JIT fusion kernel beats native C++ by 7.5x. It detects VLOAD→VFILTER_GT→VSUM→ADDF patterns, emits a fused loop with SIB-based addressing (`vmovupd [r15+r8*1]`), pre-loads threshold and segment base outside the loop, uses countdown iteration, and keeps 4 parallel partial sums in vector registers reduced only at loop exit. The kernel occupies 237 bytes of x86-64 machine code. The performance gain comes from eliminating all regfile roundtrips, all per-opcode dispatch, and all stack spills inside the hot path.

### JIT breakdown

| Phase | Time (us) |
|-------|-----------|
| Compilation | 48 |
| Execution | 700 |
| Total | 748 |

### Dispatch method comparison

| Dispatch method | Throughput (M elem/s) |
|-----------------|----------------------|
| Switch statement | 34 |
| Function pointer table | 53 (+56%) |

### Subsystem benchmarks

| Subsystem | Dataset | Throughput | Unit | Verification |
|-----------|---------|------------|------|-------------|
| VectorFilter<f64> ApplyGT | 1M f64 | 670.2 | M elem/s | Pass count matches ground truth |
| NullBitmap set+count | 1M rows, 10% null | 9,259.3 | M elem/s | Null count equals N/10 |
| Arena Allocator | 100 x 1M i32 alloc/reset | 13,154.1 | GB/s | Functional correctness |
| MurmurHash64A | 1M strings of 8B | 1.9 | GB/s | Non-zero output |
| RegFile set+get | 10M reg writes+reads | 5,073.6 | ops/us | Non-zero sink |
| Instruction encode+decode | 10M encode+decode | 765.1 | ops/us | Non-zero sink |
| HashAggregator GROUP BY | 100K rows, 100 groups | 86.0 | M rows/s | Group count equals 100 |
| HashJoin build+probe | 10K keys each side | 80.6 | M keys/s | Match count equals 10K |
| SortOperator index sort | 1M f64 indices | 6.1 | M elem/s | Ascending order verified |
| DictionaryEncoding<f64> | 1M f64 build+decode | 1.2 | M elem/s | Decoded values equal input |
| RLEEncoding<f64> | 1M f64, 10 runs | 251.0 | M elem/s | Decoded values equal input |
| DeltaEncoding<i64> | 1M sequential i64 | 181.6 | M elem/s | Decoded values equal input |
| Mixed Filter+Aggregate | 1M rows, 10 groups | 64.2 | M rows/s | Agg sums match per-group ground truth |
| Sort+TopK | 1M f64 sorted, top 100 | 6.4 | M elem/s | Top 100 equal full-sort reference |
| Join+Aggregate | 100K+10K join+agg | 20.7 | M rows/s | Agg sums match per-key ground truth |
| ThreadPool | 100 tasks sqrt workload | 1.34x | speedup | >1.0x on 4-thread host |
| Bytecode optimizer | filter+sum program | 12.5% | reduction | Output size <= input size |
| JIT compile | filter+sum bytecode | 46.0 | us | Compiled function valid |
| JIT execute | 1M f64 native code | 1,400 | M elem/s | Result matches interpreter || Result matches interpreter |

## Python Bindings

VoxelVM ships with pybind11-based Python bindings. Build with `cmake -B build && cmake --build build`, then import `build/voxel_py.*.so`.

### Low-level API

```python
import voxel_py as vx
import numpy as np

engine = vx.EngineF64()
engine.add_segment(np.array([1.0, 2.0, 3.0, 4.0]))
engine.set_scalar_f64(3, 2.5)
code = [vx.Instruction.vload(0, 1, 0, 0).raw, vx.Instruction.halt().raw]
engine.load_program(code)
engine.run()
```

### High-level Query Builder

```python
import voxel_query as vq
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = vq.from_numpy(data).filter_gt(4.0).sum().run()   # 26.0
result = vq.from_numpy(data).sum().run()                    # 36.0
result = vq.filter_and_sum(data, 4.0)                       # one-shot
top3   = vq.from_numpy(data).topk(3, largest=True).run()   # [8,7,6]
```

### SQL Frontend (PostgreSQL-compatible)

Requires `pip install pglast`. Uses the PostgreSQL parser (libpg_query) to parse SQL, then compiles to VoxelVM bytecode.

```python
import voxel_sql as vsql
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

result = vsql.execute("SELECT SUM(col) FROM data WHERE col > 4",
                       data={"data": data})
# → 26.0

result = vsql.execute("SELECT AVG(col) FROM data WHERE col > 4",
                       data={"data": data})
# → 6.5

# From CSV files
result = vsql.execute("SELECT SUM(col) FROM 'trades.csv' WHERE col > 500")
```

Supports: SELECT, FROM, WHERE (>, >=, <, <=, =), SUM, COUNT, AVG, GROUP BY, ORDER BY, LIMIT. The parser is PostgreSQL's actual parser, so any PostgreSQL-compatible SQL syntax works.

## Compiler requirements

C++20. GCC 12 or later, Clang 16 or later, or MSVC 2022 or later. x86-64 requires SSE4.2 at minimum; AVX2 and AVX-512 are detected at compile time and enabled automatically. ARM64 requires NEON. The build is `-Wall -Wextra -Werror` clean. Python bindings require pybind11 (fetched automatically via CMake FetchContent) and pglast (`pip install pglast`) for SQL support.

## License

MIT
