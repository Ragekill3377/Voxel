# Building VoxelVM

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| C++ compiler | GCC 12+, Clang 16+, MSVC 2022+ | C++20 required |
| CMake | 3.20+ | |
| Python (optional) | 3.8+ | For Python bindings |
| Go (optional) | 1.21+ | For Go bindings |
| Java (optional) | 21+ | For Panama FFM bindings |

## Quick Build (C++ engine only)

```sh
git clone https://github.com/Ragekill3377/Voxel
cd Voxel
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
./build/voxel-test        # 22-subsystem validation
./build/voxel-jit-test    # JIT correctness: interpreter vs native
./build/voxel-demo        # 30-subsystem benchmarks
ctest --test-dir build    # all three
```

## Python Bindings

Python bindings are built automatically when CMake detects Python.

```sh
pip install numpy pglast
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Import
PYTHONPATH=build python3 -c "import voxel_py as vx; print(vx.__version__)"
```

The module is at `build/voxel_py.cpython-*.so`. pybind11 is auto-fetched via CMake FetchContent.

Dependencies: `numpy` (data arrays), `pglast` (SQL parsing, optional). Install both with `pip install numpy pglast`.

## C ABI Shared Library

The C ABI layer (`libvoxel_c.so` / `.dylib` / `.dll`) is the bridge for Go and Java.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxel_c

# Verify
g++ -std=c++20 -I include examples/c_test.cpp -L build -lvoxel_c -o build/c_test
LD_LIBRARY_PATH=build ./build/c_test
```

The library exports ~60 flat C functions (`voxel_engine_create_f64`, `voxel_jit_run`, etc.) using opaque `void*` handles.

## Go Bindings

Requires the C ABI library built first.

```sh
# Build the C library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxel_c

# Build and run Go
cd bindings/go
CGO_CFLAGS="-I$(pwd)/../.." \
CGO_LDFLAGS="-L$(pwd)/../../build -lvoxel_c -lstdc++ -lm -Wl,-rpath,$(pwd)/../../build" \
go run ./example/main.go
```

For a system-wide install, copy `libvoxel_c.so` to `/usr/local/lib` and run `ldconfig`. Then Go needs only `CGO_LDFLAGS="-lvoxel_c -lstdc++ -lm"`.

The Go package is at `bindings/go/voxel.go`. Import as:

```go
import v "voxelvm"

engine := v.NewEngine()
engine.AddSegment(data)
engine.Run()
result := v.JITRun(code, data, 500.0)
```

## Java Bindings (Panama FFM)

Requires Java 21 or later. Requires the C ABI library built first.

```sh
# Build the C library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxel_c

# Compile Java
javac --enable-preview --release 21 -d classes bindings/java/VoxelVM.java

# Run
java --enable-preview --enable-native-access=ALL-UNNAMED \
     -cp classes com.voxelvm.VoxelVM
```

The library path is resolved in this order:
1. System property `-Dvoxel.lib=/absolute/path/to/libvoxel_c.so`
2. `System.mapLibraryName("voxel_c")` (resolves to `libvoxel_c.so` / `.dylib` / `.dll` by OS)
3. `./build/` relative to working directory

For a system-wide install, copy the library to a directory in `java.library.path` and skip step 1.

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `VOXEL_BUILD_EXAMPLES` | ON | Build C++ example programs |
| `VOXEL_BUILD_PYTHON` | ON | Build Python pybind11 module |
| `VOXEL_BUILD_C_ABI` | ON | Build C ABI shared library |
| `VOXEL_BUILD_SHARED` | OFF | Build as shared library (header-only by default) |

Example:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVOXEL_BUILD_PYTHON=OFF
```

## Platform-Specific Notes

### Linux

All features work. The JIT uses `mmap`/`mprotect` for executable memory.

### macOS (Intel)

All features work. Same JIT path as Linux.

### macOS (Apple Silicon M1/M2/M3)

The JIT uses `pthread_jit_write_protect_cb` to toggle between write and execute permissions on JIT memory. This is handled automatically in `JitMemoryManager`. No additional configuration needed.

### Windows

The C++ engine compiles with MSVC. The JIT uses `VirtualAlloc`/`VirtualProtect`. Python bindings require MSVC-compatible Python (not tested). Go cgo requires MinGW or MSYS2. Java Panama FFM works with JDK 21+ on Windows.

## Troubleshooting

**Go: "ld: library not found for -lvoxel_c"**
Set `CGO_LDFLAGS` to include `-L/path/to/build` and `-Wl,-rpath,/path/to/build`.

**Java: "UnsatisfiedLinkError: unable to load library"**
Set `-Dvoxel.lib=/full/path/to/libvoxel_c.so` or copy the library to a system path.

**Python: "ModuleNotFoundError: No module named 'voxel_py'"**
Set `PYTHONPATH=build` or copy the `.so` file to your Python path.

**JIT crashes on macOS ARM64**
Ensure the binary is built with `pthread_jit_write_protect_cb` support (automatic with VOXEL_OS_MACOS && VOXEL_ARCH_ARM64). If running as a sandboxed app, add `com.apple.security.cs.allow-jit` entitlement.
