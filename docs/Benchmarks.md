# Benchmarks

All measurements taken on an Intel Core i3-4130T (Haswell, 2 cores, 4 threads, AVX2) at 2.90 GHz, GCC 15.2.0, `-O3 -march=native`, Linux 6.x. CPU pinned to core 0 via `taskset -c 0` or `sched_setaffinity`.

## Methodology

The filter+sum workload uses 1,000,000 random f64 values drawn uniformly from [0, 1000) with a threshold of 500.0, seed 42. Ground truth: 500,720 rows pass, expected sum = 375,538,740.218. Each benchmark runs 2 warmup iterations followed by 3 measured iterations. Reported results are the median of the measured runs. All three execution paths produce identical results.

Reproduce:
```sh
g++ -std=c++20 -O3 -march=native -I include examples/validate.cpp -o build/validate
taskset -c 0 ./build/validate
```

## End-to-end Filter and Sum

| Method | Time (us) | Throughput (M elem/s) | vs Native |
|--------|-----------|----------------------|-----------|
| Native C++ (scalar SIMD) | 5,312 | 188 | 1.0x |
| VoxelVM interpreter | 11,939 | 84 | 2.2x slower |
| **VoxelVM JIT (fusion)** | **872** | **1,147** | **6.1x faster** |

Warm runs (subsequent invocations after caches are hot) reach 600 us / 1,500+ M elem/s.

## JIT Breakdown

| Phase | Time (us) |
|-------|-----------|
| Compilation | 48 |
| Execution (warm) | 600-700 |
| Total (first call) | ~750 |

Compilation produces 237 bytes of x86-64 machine code. The JIT compiles once; execution is then a direct function call through a typed pointer.

## Dispatch Method Comparison

| Dispatch | Throughput (M elem/s) |
|----------|----------------------|
| Switch statement (original) | 34 |
| Function pointer table | 84 (+147%) |

## Single-Column Filter+Sum: Cross-Engine Comparison

| Engine | ~M elem/s | Type |
|--------|----------|------|
| **VoxelVM JIT** | **1,147-1,500** | Bytecode JIT (i3-4130T) |
| ClickHouse | 300-800 | Columnar SIMD |
| DuckDB | 200-500 | Vectorized C++ |
| NumPy (compiled) | 200-500 | C vectorized loops |
| Polars (streaming) | 100-300 | Rust/Arrow |
| Pandas | 5-10 | Python |

Not a full database benchmark — only the core scan primitive. DuckDB and ClickHouse win on complex queries, joins, optimizer quality, and ecosystem maturity. VoxelVM wins on raw single-column scan speed because the JIT generates exactly the instruction sequence the CPU wants with zero abstraction overhead.

## Subsystem Benchmarks

| Subsystem | Dataset | Throughput | Unit |
|-----------|---------|------------|------|
| VectorFilter<f64> ApplyGT | 1M f64 | 670 | M elem/s |
| NullBitmap set+count | 1M rows, 10% null | 9,259 | M elem/s |
| Arena Allocator | 100 x 1M i32 alloc/reset | 13,154 | GB/s |
| MurmurHash64A | 1M strings of 8B | 1.9 | GB/s |
| RegFile set+get | 10M reg writes+reads | 5,074 | ops/us |
| Instruction encode+decode | 10M encode+decode | 765 | ops/us |
| HashAggregator GROUP BY | 100K rows, 100 groups | 86 | M rows/s |
| HashJoin build+probe | 10K keys each side | 81 | M keys/s |
| SortOperator index sort | 1M f64 indices | 6.1 | M elem/s |
| DictionaryEncoding<f64> | 1M f64 build+decode | 1.2 | M elem/s |
| RLEEncoding<f64> | 1M f64, 10 runs | 251 | M elem/s |
| DeltaEncoding<i64> | 1M sequential i64 | 182 | M elem/s |
| Mixed Filter+Aggregate | 1M rows, 10 groups | 64 | M rows/s |
| Sort+TopK | 1M f64, top 100 | 6.4 | M elem/s |
| Join+Aggregate | 100K+10K join+agg | 21 | M rows/s |
| ThreadPool | 100 tasks sqrt workload | 1.34x | speedup |
| Bytecode optimizer | filter+sum program | 12.5% | reduction |
| JIT compile | filter+sum bytecode | 46 | us |
| JIT execute | 1M f64 native code | 1,400 | M elem/s |

## Projected Performance on Modern Hardware

The JIT fusion kernel's instruction mix (3 FP ops per 4-element iteration on port 5) is the bottleneck on Haswell. On CPUs with wider execution ports:

| CPU | FP pipes | Projected |
|-----|----------|-----------|
| Haswell i3 (2013) | 2 (1 shared) | 1,500 M/s (measured) |
| Skylake (2015) | 3 | ~2,500 M/s |
| Zen 3 (2020) | 4 (2 FP) | ~4,000 M/s |
| Zen 5 / Ice Lake | 5+ | ~6,000-8,000 M/s |

The kernel is already memory-bandwidth-bound on modern CPUs. On a Ryzen 9950X with DDR5-6000, the projected ceiling is 6-8 billion elem/s single-core before hitting L3 bandwidth limits.
