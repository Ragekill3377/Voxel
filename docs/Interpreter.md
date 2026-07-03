# Interpreter

VoxelVM's interpreter executes bytecode via a static dispatch table. At LoadProgram time, the engine scans the instruction stream for known patterns. When the classic filter+sum loop is detected (`VLOAD → VFILTER_GT → VSUM → ADDF → ADD → CMP → JNZ`), the interpreter skips the dispatch loop entirely and runs a specialized C++ fast path that GCC auto-vectorizes into SIMD.

## Two-Tier Execution

| Tier | Description | Throughput (f64, Haswell i3) |
|------|-------------|------------------------------|
| Tier 0 | Dispatch-table interpreter with inline handlers | 53 M elem/s |
| Tier 1 | Pattern-matched fast path bypassing regfile and dispatch | 375 M elem/s |

Tier 0 is the general-purpose interpreter. Every instruction goes through `Run()`'s fetch-decode-execute loop: read a 32-bit encoded word, mask out the opcode byte, index the 256-entry static function pointer table, call the handler. Handlers are `VOXEL_ALWAYS_INLINE` + `VOXEL_FLATTEN`, so GCC inlines the entire dispatch loop into one function. The cost is per-iteration regfile materialization: on the filter+sum pattern, 35+ scalar and vector register reads/writes through the `RegFile` class per 4-element iteration. At 53 M/s on a 2.90 GHz Haswell, that is roughly 55 CPU cycles per element, dominated by memory latency on the regfile backing store.

Tier 1 detects the filter+sum bytecode pattern at LoadProgram time and replaces the dispatch loop with a direct C++ implementation:

```cpp
while (offset + kLanes <= total) {
    T v0[kLanes]; for (sz i = 0; i < kLanes; ++i) v0[i] = data[offset + i];
    for (sz i = 0; i < kLanes; ++i) v0[i] = (v0[i] > thresh) ? v0[i] : T{};
    T sum = T{}; for (sz i = 0; i < kLanes; ++i) sum += v0[i];
    acc += sum;
    offset += kLanes;
}
```

This eliminates all dispatch overhead, all regfile memory access, and all per-instruction decode. The threshold, accumulator, offset, and total count remain in CPU registers. GCC auto-vectorizes the loads, comparisons, and accumulates into ymm-wide SIMD instructions. The result is 375 M elem/s — a 7x improvement over Tier 0, and 2.3x faster than the scalar C++ baseline on the same hardware.

## Why Faster Than Scalar C++

The "native" C++ benchmark uses a scalar loop: `for (i=0; i<n; ++i) if (data[i] > t) sum += data[i]`. GCC struggles to auto-vectorize this because the conditional accumulation creates a loop-carried dependency. The Tier 1 fast path works around this by summing in groups of 4 (explicitly breaking the dependency chain), which GCC recognizes as vectorizable. The compiler sees a flat loop with known bounds (4 iterations for f64 on AVX2) and unrolls + auto-vectorizes it directly to native SIMD instructions. The JIT takes this one step further by emitting the exact AVX2 opcodes — 2-way unrolled, SIB-addressed, zero branches — matching what a human would write with `_mm256` intrinsics.

## Pattern Matching

Pattern detection runs once at `LoadProgram()`. The `DetectFastPath()` method checks the first 7 instructions against the filter+sum pattern. More pattern templates (window-sum, time-window, filter+count) can be added following the same mechanism: detect the bytecode sequence, implement the equivalent C++ loop, route `Run()` accordingly.

```
[0] VLOAD       rd=0, ra=1(must match [4]+[5] offset reg), segId, count
[1] VFILTER_GT  rd=1, ra=0, rb=3 (threshold reg)
[2] VSUM        rd=1, ra=1
[3] ADDF        rd=0, rb=1 (accumulate into acc reg)
[4] ADD         ra=1 (must match [0] offset reg)
[5] CMP         ra=1 (must match [0] offset reg), rb=2 (total count reg)
[6] JNZ         -6 (backward to instruction 0)
```

The constraints are strict: VLOAD's offset register must appear as the target of both ADD and CMP; VFILTER_GT's source must match VLOAD's destination; VSUM's source must match VFILTER_GT's destination; ADDF's source must match VSUM's destination; JNZ must target instruction 0. When all constraints hold, `HasFastPath_` is set and `Run()` delegates to `RunFastPath()`.

## Fallback

Programs that do not match the filter+sum pattern execute through the dispatch table as before. All 208 opcodes are supported. The fast path is an optimization, not a replacement. Adding more patterns (filter+count, filter+min/max, filter with mask register, multi-segment joins) follows the same approach: detect the bytecode sequence, implement the equivalent C++ loop, route `Run()` accordingly.

## Window Family (static API)

For programs that don't need bytecode interpretation, the engine exposes six static methods that compute full-array windowed results in one call:

| Method | Output | Benchmark (1M, w=20) |
|--------|--------|----------------------|
| `WindowSum` | sliding sum | 12 ms |
| `WindowMean` | sliding mean | 15 ms |
| `WindowDelta` | adjacent diff | 2 ms |
| `WindowStdDev` | sliding stddev | 39 ms |
| `WindowVariance` | sliding variance | 37 ms |
| `WindowQuantile` | sliding quantile | 1000 ms (w=100) |

All six are available as `EngineF64` methods in Python (`engine.window_mean(data, out, window)`) and as `Engine<T>::WindowMean(...)` in C++. They operate on raw pointers — no heap allocation, no bytecode dispatch, GCC auto-vectorizes.

The bytecode opcodes (WDELTA, WINDOW_SUM, WINDOW_MEAN, WSTD, WVARIANCE_W, WQUANTILE) implement the same algorithms as per-call handlers in the dispatch table. The static methods are the recommended API for single-pass computation; the bytecode ops exist for programs that embed window ops within larger query plans.

## Performance Tier Roadmap

| Tier | Mechanism | Target | Status |
|------|-----------|--------|--------|
| Tier 0 | Dispatch table | 53 M/s | Implemented |
| Tier 1 | Pattern-matched fast path | 375 M/s | Implemented |
| Tier 2 | Register caching | 500-600 M/s | Planned |
| Tier 3 | Vector register caching | 600-800 M/s | Planned |
| Tier 4 | Computed-goto dispatch (GCC) | +20% vs Tier 0 | Planned |

Tier 2 keeps hot scalar registers (R0-R4) in C++ locals inside `Run()`, flushing only at branches. Tier 3 extends that to vector registers (V0-V1) for the VLOAD→VFILTER chain. Tier 4 replaces the function pointer table with a computed-goto (labels-as-values) dispatcher for GCC, eliminating the indirect branch penalty on Tier 0 paths.

## JIT Escape Hatch

The Tier 1 fast path is a stop on the way to the JIT fusion kernel (1,400-1,700 M elem/s). It provides a 7x interpreter speedup without runtime code generation or memory protection syscalls — useful on embedded systems, locked-down environments, or architectures without a JIT backend. When the JIT is available, the fusion kernel takes priority because it generates exactly the instruction sequence the CPU wants down to the SIB byte.
