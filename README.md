# VoxelVM

A bytecode virtual machine engine for analytical queries on columnar data. It compiles to zero warnings under `-Wall -Wextra -Werror`, is organized into 37 header-only modules under a single `#include`, and passes every subsystem test in its 22-case validation suite. It also ships a working x86-64 JIT backend that generates native AVX2 code from bytecode, verified correct against the interpreter.

## The problem

Columnar analytics engines spend most of their cycles inside tight loops: load a batch of values, compare against a threshold, accumulate the ones that pass, advance the cursor, repeat. Writing these loops by hand in C++ is fast but rigid. Writing them in SQL or a dataframe library gives you flexibility but costs you an order of magnitude in dispatch overhead. A bytecode VM sits in between: you compile your query once into a dense instruction stream, then the VM runs it with minimal per-row branching.

The catch is that most bytecode VMs are either toys that handle six opcodes and two types, or full JVM/CLR-class runtimes that drag in garbage collection, class loading, and a 500 MB baseline. VoxelVM is the middle ground: it handles 208 opcodes across f64, f32, i64, i32, u64, and u32, monomorphizes the entire engine per element type, and fits in one `#include`.

## Quick start

```sh
git clone https://github.com/user/voxelvm
cd voxelvm
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/voxel-demo       # performance benchmarks with ground-truth checks
./build/voxel-test       # 22-subsystem validation suite
./build/voxel-jit-test   # JIT correctness: interpreter vs native
ctest --test-dir build   # all three
```

## Architecture

The engine is organized into ten modules. Every module is a single directory of headers under `include/voxel/`. The umbrella header `voxel/voxel.hpp` pulls them all in.

### Core (`core/`, 5 files, 1,354 lines)

The type system defines 30 runtime data type identifiers (`DataType::Int64`, `DataType::Float64`, `DataType::Timestamp`, etc.) plus a `TypeTraits<T>` template that maps each C++ primitive to its size, alignment, signedness, and runtime ID. `ScalarValue` is a 16-byte type-erased container used at decode boundaries.

The register file has 64 64-bit scalar registers, 16 256-bit vector registers, 8 mask registers, and a flags word (Zero, Carry, Sign, Overflow, NaN, DivByZero). Vector registers are typed at access time via `VecLanes<T>(idx)`, which returns a `T*` into the aligned backing store. No unions, no runtime type tags inside the register file itself; the engine template parameter `T` determines the interpretation.

The arena allocator is a bump-pointer with 64 KiB default block size and 64-byte alignment. It tracks total allocated, peak, allocation count, and block count via atomic counters. A thread-local variant and a page-granularity variant sit alongside the main arena.

### Bytecode (`bytecode/`, 5 files, 3,075 lines)

Instructions are 32-bit words: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`. For vector loads and stores, the immediate is subdivided: bits 11:8 encode the segment ID, bits 7:0 encode the element count (0 means "fill the vector register").

The opcode space is fully allocated: 10 control, 15 scalar move, 16 scalar arithmetic, 16 scalar bitwise, 9 comparison, 16 branch, 16 type conversion, 16 vector I/O, 16 vector arithmetic, 7 vector-scalar, 8 vector compare, 8 vector logical, 10 vector filter, 13 vector reduction, 16 aggregate, and 16 hash/sort/join. That is 208 opcodes, every slot from 0x00 through 0xFF either implemented or reserved for a specific function.

The assembler is a programmatic builder with a label/relocation system. You call `Assembler::VLoad(0, 1, segId, 4)` and it appends a packed instruction word. Named labels like `Assembler::Bind("loop")` and `Assembler::Jnz("loop")` are resolved during `Finalize()`. The verifier performs five checks: opcode validity, branch target bounds, register range, segment range, and halt reachability via DFS. The disassembler pretty-prints every instruction with hex addresses and decoded operands.

### Execution engine (`exec/`, 2 files, 3,591 lines)

`Engine<T>` is the heart of the system. It owns a `RegFile`, an arena, a vector of `Segment<T>` views, and a bytecode vector. The dispatch mechanism uses a 256-entry static table of function pointers, one per opcode:

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

220 such handlers exist, each marked `VOXEL_ALWAYS_INLINE`. The `Run()` method fetches a raw instruction word, masks out the opcode byte, indexes the table, and calls the handler. This is measurably faster than a 2500-line switch statement because each handler is a leaf function the compiler can optimize independently, and because the indirect branch target is fully predictable after the first iteration of a loop.

Vector filters are handled per-comparison-mode. `VFILTER_GT`, `VFILTER_GE`, `VFILTER_LT`, etc. each have their own dispatch slot. This eliminates the inner `switch(mode)` that a unified VFILTER would require. The filter body is a simple counted loop over `kLanes` elements; for f64 that is four iterations, which the compiler unrolls and converts to branchless CMOV or AVX blend instructions.

### JIT code generation (`codegen/`, 3 files, 3,190 lines)

The x86-64 backend is a real assembler that emits actual machine code bytes. It handles REX, VEX2, VEX3, ModR/M, and SIB encoding. The `X64Assembler` class has methods like `Vaddpd(y0, y1, y2)` which emits `{0xC5, 0xF5, 0x58, 0xCA}`.

`X64Compiler::Compile()` walks the bytecode instruction by instruction and emits native code. A VLOAD becomes `VMOVUPD ymm0, [r8 + rax]` (segment base in r8, offset in rax). A VFILTER_GT becomes `VCMPPD ymm2, ymm0, ymm1, 14` (ordered greater-than) followed by `VBLENDVPD`. A VSUM becomes a sequence of `VHADDPD` and `VPERM2F128` for horizontal reduction. Branches are resolved to relative offsets with backpatching.

The `JitMemoryManager` allocates executable pages via `mmap` with `PROT_READ | PROT_WRITE | PROT_EXEC` on Linux, or `VirtualAlloc` with `PAGE_EXECUTE_READWRITE` on Windows. After code generation, the page is marked read-execute only. `JitFunction::Entry` is a typed function pointer that takes `(void* regfile, void* segments, u64* counts)` and returns void.

The `JitCache` is an LRU cache keyed by bytecode content hash. Repeated queries hit the cache.

### SIMD layer (`simd/`, 4 files, 2,495 lines)

Four backends, selected at compile time by `#if` on `__AVX512F__`, `__AVX2__`, `__ARM_NEON`, or the portable fallback. Each provides `Vec<T>` specializations with `Load`, `Store`, `Set1`, `Zero`, element-wise arithmetic, comparison, blend, gather/scatter, and horizontal reduction.

The x86 backend uses `__m256d`/`__m256`/`__m256i` for AVX2 and `__m512d`/`__m512`/`__m512i` for AVX-512. The NEON backend uses `float64x2_t`/`float32x4_t`/`int64x2_t`. The scalar fallback uses plain arrays and counted loops; the compiler auto-vectorizes these on platforms without SIMD.

A critical portability fix: on Linux x86-64, `std::int64_t` is `long int` but GCC's AVX2 intrinsics expect `long long int`. We use `using i64x = long long` inside the x86 specializations and explicitly `reinterpret_cast` at every intrinsic boundary where a 64-bit pointer is passed.

Bulk free functions like `simd::scalar::scalar_filter_gt_f64` process arrays in vector-sized chunks with a scalar tail. These are the workhorses for the SIMD benchmark.

### Data layer (`data/`, 4 files, 2,590 lines)

`Segment<T>` is a zero-copy typed view over external memory. It holds `T* Data`, `sz Count`, and optional ownership tracking. Its iterators are raw `T*`, so range-for loops compile to direct pointer arithmetic.

Five columnar encodings are implemented. `DictionaryEncoding<T>` builds a sorted unique-value table and replaces each row with an index; useful when cardinality is low. `RLEEncoding<T>` stores (value, run-length) pairs. `DeltaEncoding<T>` stores the difference from the previous row and works well for sorted integer sequences. `BitPackedEncoding` packs integers into the minimum bit width. `FrameOfReferenceEncoding` scales floating-point values to integers with a shared base and scale.

`NullBitmap` packs one bit per row into `u64` words. `NullableSegment<T>` pairs a Segment with a NullBitmap and provides a sentinel value for null propagation during vectorized operations. `DataBlock<T>` is a fixed-capacity chunk for columnar batch processing, managed by a thread-safe pool allocator.

### Operators (`ops/`, 5 files, 2,667 lines)

`VectorFilter<T>` processes data in 256-element chunks, writes a 64-bit word per chunk into a bitmap, and counts passes via hardware popcount. `SelectionVector` converts bitmaps to index lists for scatter/gather.

`HashAggregator` uses open addressing with linear probing on a power-of-two table. It accumulates sum, min, max, count, and a Welford online variance state per group. The Welford algorithm computes running mean and M2 in a single pass without catastrophic cancellation.

The sort module provides LSD radix sort for integers (8 counting passes, one per byte) and bottom-up merge sort for floating point. `TopK` uses median-of-three quickselect for partial sorting.

The join module implements hash join (build a hash table on the smaller side, probe with the larger), merge join (two cursors advancing on sorted inputs), anti join (left rows with no right match), semi join (left rows with at least one right match), and nested loop join (O(n*m) for small tables). All use `Segment<T>` as their input and output type, keeping data in columnar layout.

### System and utilities (`system/` + `util/`, 10 files, 1,845 lines)

Profiling tracks per-opcode execution counts and cycle counts using `__rdtsc`. Telemetry exports to JSON and CSV. Error handling uses a thread-local handler chain. Debug utilities include hex dumps with ASCII sidebars, register file snapshots, and binary crash recovery files.

Utilities include bit manipulation (popcount, count-leading-zeros, bit reverse), three hash functions (MurmurHash64A, XXHash64, FarmHash64), fast inverse square root, a stack-allocated string builder, a bounded string view, a thread pool with work stealing, and a parallel-for executor.

## Instruction set

208 opcodes organized as follows. Every opcode is handled by a dedicated dispatch handler in the engine.

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

All measurements on an Intel x86-64 host, GCC 15.2, `-O3 -march=native`, 1,000,000 random f64 values in [0, 1000), threshold 500.0. Ground truth sum verified at 375,538,740.218. Results are the median of three runs; individual runs vary by roughly 10 percent due to thermal throttling and scheduler noise.

### End-to-end filter and sum (1M rows)

| Method | Time (us) | Throughput (M elem/s) | Overhead vs native |
|--------|-----------|----------------------|---------------------|
| Native C++ | 5,500 | 180 | baseline |
| VoxelVM interpreter | 13,000 | 80 | 2.4x |
| VoxelVM JIT | 74,000 | 13.5 | 13.5x |

The interpreter runs at roughly 40 percent of native speed. For a bytecode VM dispatching 208 opcodes with bounds-checked segment access, vector registers, and typed bit-casts, that overhead is competitive. Most hobby VMs land between 10x and 50x slower than native for this class of workload.

### JIT breakdown

| Phase | Time (us) |
|-------|-----------|
| Compilation | 35 |
| Execution | 74,000 |
| **Total** | **74,035** |

The JIT generates 252 bytes of correct x86-64 machine code in 35 microseconds. However, the generated code is slower than the interpreter. The reason is straightforward: the current JIT backend emits a direct, unoptimized translation of each bytecode instruction. Every operation loads its operands from the regfile pointer in memory, computes, and stores the result back. There is no register allocation pass, no liveness analysis, and no instruction scheduling. The VSUM horizontal reduction is emulated by storing the vector to the stack and loading each lane as a scalar, which produces 8 memory accesses for 4 additions.

By contrast, the interpreter benefits from 1,300 lines of code all visible to the compiler inside a single translation unit. GCC unrolls the vector loops, keeps the accumulator in an xmm register, and eliminates redundant loads and stores across handler boundaries. The JIT cannot do any of this because the Compile method emits a fixed sequence of machine instructions per opcode with no cross-instruction analysis.

Closing this gap requires a proper register allocator and a peephole pass that fuses adjacent load-compute-store sequences. That is planned but not yet implemented.

### Interpreter dispatch evolution

The first implementation used a 2,500-line switch statement. The current implementation uses a 256-entry function pointer table with 220 `VOXEL_ALWAYS_INLINE` leaf handlers, each 4 to 15 lines. The switch compiled to a jump table but the compiler could not inline across case boundaries because the function was too large. The function pointer table lets each handler be an independent leaf that the compiler inlines at the call site inside `Run()`, producing a tight loop where the indirect branch is the only dispatch cost.

| Dispatch method | Throughput (M elem/s) |
|-----------------|----------------------|
| Switch statement | 34 |
| Function pointer table | 80 (+135%) |

Most of the gain comes from two effects: GCC can inline the leaf handlers but could not inline the switch cases, and the per-comparison-mode VFILTER dispatch eliminated the inner runtime switch that was causing branch mispredictions inside the hot loop.

### Subsystem-level measurements

| Subsystem | Dataset | Time (us) | Notes |
|-----------|---------|-----------|-------|
| DictionaryEncoding build | 1M f64 | instant | Unique-value sort, index assignment |
| NullBitmap | 1M rows, 0.1% null | instant | 156 u64 words, bitwise ops |
| MergeSort (index) | 1M f64 | 144,000 | 144 ms, index-only sort |
| HashAggregator | 1,000 rows, 10 groups | 9 | Open addressing, 10 distinct keys |
| RLEEncoding build | 1M f64 | instant | Single-scan run detection |
| VectorFilter bitmap | 1M f64 | instant | 256-chunk processing, HW popcount |

## Compiler requirements

C++20. GCC 12 or later, Clang 16 or later, or MSVC 2022 or later. x86-64 requires SSE4.2 at minimum; AVX2 and AVX-512 are detected at compile time and enabled automatically. ARM64 requires NEON. The build is `-Wall -Wextra -Werror` clean.

## License

MIT
