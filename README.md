# VoxelVM

A bytecode virtual machine engine for analytical queries on columnar data.

## Problem

Columnar analytics engines spend their cycles inside tight loops: load a batch of values, compare against a threshold, accumulate the ones that pass, advance the cursor, repeat. Writing these loops by hand in C++ is fast but rigid. Writing them in SQL or a dataframe library gives flexibility at the cost of dispatch overhead. A bytecode VM sits in between: the query is compiled once into a dense instruction stream, then the VM runs it with minimal per-row branching.

Most bytecode VMs are either toys that handle six opcodes and two types, or full JVM/CLR-class runtimes that include garbage collection, class loading, and a large baseline. VoxelVM targets the middle ground: it handles 208 opcodes across f64, f32, i64, i32, u64, and u32, monomorphizes the entire engine per element type, and distributes as a single header tree under one `#include`.

The JIT fusion kernel detects filter+sum patterns and emits native x86-64 code that runs at 1,600+ M elem/s — matching hand-tuned AVX2 intrinsics, 10x faster than scalar C++. The interpreter's pattern-matched fast path reaches 375 M elem/s without runtime code generation, a 7x improvement over the dispatch-table baseline.

## Quick Start

```sh
git clone https://github.com/Ragekill3377/Voxel
cd Voxel
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/voxel-demo        # 30 subsystem benchmarks with ground-truth checks
./build/voxel-test         # 22-subsystem validation suite
./build/voxel-jit-test     # JIT correctness: interpreter vs native
ctest --test-dir build     # all three
```

Validate independently:

```sh
g++ -std=c++20 -O3 -march=native -I include examples/validate.cpp -o build/validate
taskset -c 0 ./build/validate
```

## Python

```python
# Low-level engine
import voxel_py as vx
engine = vx.EngineF64()
engine.run()

# JIT: one-shot filter+sum at native speed
import numpy as np
data = np.random.uniform(0, 1000, 1_000_000).astype(np.float64)
result = vx.jit_run(code, data, threshold=500.0, total_count=len(data))
# ~1,500 M elem/s, correct result verified

# Query builder
import voxel_query as vq
result = vq.from_numpy(data).filter_gt(500.0).sum().run()

# SQL (PostgreSQL-compatible via pglast)
import voxel_sql as vsql
result = vsql.execute("SELECT SUM(price) FROM trades WHERE price > 500",
                       data={"trades": data})

# Market data
import voxel_market as vm
md = vm.MarketData(prices=arr, volumes=arr)
print(md.vwap(), md.rsi(14), md.sharpe_ratio())
```

## Documentation

| File | Topic |
|------|-------|
| [docs/Architecture.md](docs/Architecture.md) | Module layout, engine design, memory topology |
| [docs/JIT.md](docs/JIT.md) | JIT compiler, fusion kernel, performance |
| [docs/Interpreter.md](docs/Interpreter.md) | Dispatch, fast-path pattern matching, performance tiers |
| [docs/Bytecode.md](docs/Bytecode.md) | Instruction encoding, full opcode reference |
| [docs/Benchmarks.md](docs/Benchmarks.md) | Methodology, end-to-end comparisons, subsystem table |

## Compiler Requirements

C++20. GCC 12+, Clang 16+, or MSVC 2022+. x86-64 requires SSE4.2; AVX2 and AVX-512 detected automatically. ARM64 requires NEON. Builds `-Wall -Wextra -Werror` clean. Python bindings require pybind11 (auto-fetched via CMake) and `pip install pglast` for SQL.

## License

MIT
