# Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VoxelVM Architecture                       │
├──────────┬──────────────┬───────────┬───────────┬──────────────┤
│ bytecode │     exec     │  codegen  │   simd    │     data     │
│  opcodes │    engine    │    jit    │   avx2    │   segments   │
│   instr  │   dispatch   │  x86-64   │  avx512   │   encoding   │
│ assembler│   optimizer  │   arm64   │   neon    │    nulls     │
│ verifier │  streaming   │   cache   │  scalar   │    blocks    │
├──────────┴──────────────┴───────────┴───────────┴──────────────┤
│    ops: filter │ aggregate │ sort │ join │ hash                 │
│   system: telemetry │ error │ debug │ profile                  │
│    util: bit │ math │ hash │ string │ thread                   │
├────────────────────────────────────────────────────────────────┤
│ Python: bindings (pybind11) │ query builder │ SQL (pglast)     │
│ Market: VWAP │ RSI │ MACD │ Bollinger │ Sharpe │ OHLC         │
│ Data: mmap segments │ chunked reader │ streaming engine        │
└────────────────────────────────────────────────────────────────┘
```

## Core

### Type System

30 runtime data type identifiers (`DataType::Int64`, `DataType::Float64`, `DataType::Timestamp`, etc.) plus a `TypeTraits<T>` template that maps each C++ primitive to its size, alignment, signedness, and runtime ID. `ScalarValue` is a 16-byte type-erased container for decode boundaries.

### Register File

64 64-bit scalar registers, 16 256-bit vector registers, 8 mask registers, and a flags word (Zero, Carry, Sign, Overflow, NaN, DivByZero). Vector registers are typed at access time via `VecLanes<T>(idx)`, which returns a `T*` into the aligned backing store. The engine template parameter `T` determines the interpretation on monomorphized access.

### Arena Allocator

Bump-pointer with 64 KiB default block size and 64-byte alignment. Tracks total allocated, peak, allocation count, and block count via atomic counters. Provides `SoftReset()` for arena reuse across queries without freeing pages. `ThreadLocalArena` variant for per-thread allocation. `PageAllocator` variant for page-granularity (4 KiB or 2 MiB).

## Bytecode

### Instruction Format

32-bit packed word: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`. For vector loads and stores, the immediate is subdivided: bits 11:8 encode the segment ID, bits 7:0 encode the element count (0 means "fill the vector register"). Branch offsets are 12-bit signed instruction offsets.

### Opcode Space

208 opcodes across 16 categories. See [Bytecode.md](Bytecode.md) for the full reference.

### Toolchain

- **Assembler**: Programmatic builder with label/relocation system. `Assembler::Bind("loop")` and `Assembler::Jnz("loop")` resolved during `Finalize()`.
- **Verifier**: Checks opcode validity, branch target bounds, register range, segment range, and halt reachability via DFS.
- **Disassembler**: Pretty-prints every instruction with hex addresses and decoded operands.
- **Optimizer**: 7-pass pipeline (ConstantFolder, FilterMerge, PredicatePushdown, DeadCodeEliminator, LoopInvariantHoister, PeepholeOptimizer, NopSqueezer).

## Execution Engine

`Engine<T>` monomorphizes per element type. It owns a `RegFile`, an arena, a vector of `Segment<T>` views, and a bytecode buffer. The dispatch mechanism uses a 256-entry static table of function pointers, one per opcode. Each handler is a `VOXEL_ALWAYS_INLINE` leaf function. The `Run()` method fetches a raw instruction word, masks out the opcode byte, indexes the table, and calls the handler.

At `LoadProgram()` time, the engine scans for known patterns. When the filter+sum loop (`VLOAD → VFILTER_GT → VSUM → ADDF → ADD → CMP → JNZ`) is detected, `Run()` bypasses the dispatch table and executes a specialized C++ fast path. The fast path keeps the threshold, accumulator, offset, and count in CPU registers, eliminates all regfile memory access, and runs a tight loop that GCC auto-vectorizes to SIMD. This pushes interpreter throughput from 53 M elem/s to 375 M elem/s on the same hardware.

Vector filters are handled per-comparison-mode: `VFILTER_GT`, `VFILTER_GE`, `VFILTER_LT`, etc. each have their own dispatch slot, eliminating the inner `switch(mode)`.

The engine supports `RunParallel(segId, numThreads, resultReg)` for data-parallel execution across CPU cores using the built-in `ThreadPool`.

## JIT Code Generation

See [JIT.md](JIT.md) for the full JIT story.

In brief: The x86-64 backend emits actual machine code bytes handling REX, VEX2, VEX3, ModR/M, and SIB encoding. The basic-block compiler performs liveness analysis and register allocation. The fusion kernel detects filter+sum patterns and generates a fused loop that beats native C++ by 10x.

## SIMD Layer

Four backends selected at compile time: AVX-512 (`__AVX512F__`), AVX2 (`__AVX2__`), NEON (`__ARM_NEON`), and a portable scalar fallback. Each provides `Vec<T>` specializations with `Load`, `Store`, `Set1`, `Zero`, element-wise arithmetic, comparison, blend, gather/scatter, and horizontal reduction. Bulk free functions like `simd::scalar::scalar_filter_gt_f64` process arrays in vector-sized chunks with scalar tails.

## Data Layer

### Segment<T>

Zero-copy typed view over external memory. Holds `T* Data`, `sz Count`, and optional ownership tracking. Iterators are raw `T*` — range-for loops compile to direct pointer arithmetic. `MmapSegment<T>` wraps a file descriptor for zero-copy file access. `MutableSegment<T>` provides growable append.

### Columnar Encodings

- `DictionaryEncoding<T>`: Sorted unique-value table with index substitution
- `RLEEncoding<T>`: (Value, run-length) pairs for repeated sequences
- `DeltaEncoding<T>`: Difference-from-previous for sorted integer columns
- `BitPackedEncoding`: Minimum-bit-width integer packing
- `FrameOfReferenceEncoding`: Floating-point to scaled-integer conversion

### NullBitmap

One bit per row packed into `u64` words. `NullableSegment<T>` pairs a Segment with a NullBitmap for null propagation during vectorized operations. `DataBlock<T>` is a fixed-capacity chunk for columnar batch processing managed by a thread-safe pool.

## Operators

| Operator | Description |
|----------|-------------|
| `VectorFilter<T>` | 256-element chunk filtering, bitmap generation, hardware popcount |
| `HashAggregator` | Robin Hood open-addressing hash table, Welford online variance |
| `SortOperator` | LSD radix sort (integers), merge sort (floats), TopK quickselect |
| `HashJoin` | Build-probe hash join with open addressing |
| `MergeJoin` | Two-cursor merge on sorted inputs |
| `AntiJoin`, `SemiJoin` | Set-based join variants |
| `NestedLoopJoin` | O(n*m) join for small tables or non-equi predicates |

## System

- **Profiling**: Per-opcode execution counts and cycle counts via `__rdtsc`, cache miss counters via `perf_event`
- **Telemetry**: JSON and CSV export of execution snapshots
- **Error handling**: Thread-local error handler chain, halts on division by zero or out-of-bounds access
- **Debug**: Hex dumps with ASCII sidebars, register file snapshots, bytecode disassembly

## Utilities

- **Bit**: PopCount, count-leading-zeros, count-trailing-zeros, bit reverse, next power of two
- **Hash**: MurmurHash64A, XXHash64, FarmHash64
- **Math**: Fast inverse square root, lerp, clamp, floor/ceil division
- **String**: Non-owning `StringView`, stack-allocated `StringBuilder`
- **Thread**: `ThreadPool` with shared queue, `WorkStealingScheduler` with per-worker deques, `ParallelFor` executor
