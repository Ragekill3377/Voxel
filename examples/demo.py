"""
VoxelVM Python Example
Demonstrates filter+sum, hash aggregation, sort, and JIT compilation.
"""
import numpy as np
import voxel_py as vx
import time

# ================================================================
# 1. Filter+Sum via bytecode engine
# ================================================================
print("=== Filter+Sum (Bytecode Engine) ===")
N = 1_000_000
data = np.random.uniform(0, 1000, N).astype(np.float64)
threshold = 500.0

# Ground truth
expected = data[data > threshold].sum()
print(f"Ground truth: {expected:.3f}")

engine = vx.EngineF64()
seg_id = engine.add_segment(data)

# Set scalar registers
engine.set_scalar_f64(0, 0.0)      # accumulator
engine.set_scalar_f64(1, 0.0)      # offset
engine.set_scalar_f64(2, float(N)) # total count
engine.set_scalar_f64(3, threshold) # threshold
k_lanes = engine.k_lanes

# Build bytecode
code = [
    vx.Instruction.mov(4, int(k_lanes)).raw,
    vx.Instruction.vload(0, 1, seg_id, 0).raw,
    vx.Instruction.vfilter_gt(1, 0, 3).raw,
    vx.Instruction.vsum(5, 1).raw,
    vx.Instruction.addf(0, 0, 5).raw,
    vx.Instruction.add(1, 1, 4).raw,
    vx.Instruction.cmp(1, 2).raw,
    vx.Instruction.jnz(-6).raw,
    vx.Instruction.halt().raw,
]

engine.load_program(code)
t0 = time.perf_counter()
engine.run()
t1 = time.perf_counter()
result = engine.get_scalar_f64(0)
print(f"VM result:    {result:.3f}")
print(f"Match:        {abs(result - expected) < expected * 1e-12}")
print(f"Time:         {(t1-t0)*1e6:.0f} us")

# ================================================================
# 2. Hash Aggregation (GROUP BY)
# ================================================================
print("\n=== Hash Aggregation ===")
groups = np.array([i % 10 for i in range(1000)], dtype=np.uint32)
values = data[:1000]

arena = vx.Arena()
agg = vx.HashAggregator(arena)
agg.init(128)
agg.accumulate(groups, values)

print(f"Groups: {agg.group_count}")

# ================================================================
# 3. Sort
# ================================================================
print("\n=== Sort ===")
small = data[:10].copy()
indices = vx.sort_ascending(small)
print(f"Original:  {small}")
print(f"Sorted indices: {indices}")
print(f"Sorted values:  {small[indices]}")

# ================================================================
# 4. TopK
# ================================================================
print("\n=== TopK ===")
top1000 = vx.topk_select(data, 1000, largest=True)
print(f"Top 1000: min={top1000.min():.1f}, max={top1000.max():.1f}")

# ================================================================
# 5. Dictionary Encoding
# ================================================================
print("\n=== Dictionary Encoding ===")
small_data = np.array([1.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float64)
d = vx.DictionaryEncodingF64()
d.build(small_data)
print(f"Dict size:    {len(d.dictionary)} entries")
print(f"Dict memory:  {d.memory_usage} bytes")

# ================================================================
# 6. CPU Info
# ================================================================
print("\n=== Platform ===")
info = vx.cpu_info()
for k, v in info.items():
    print(f"  {k}: {v}")

# ================================================================
# 7. JIT Compilation
# ================================================================
print("\n=== JIT ===")
jit_result = vx.jit_compile(code)
if jit_result["compiled"]:
    print(f"Compiled: {jit_result['code_size']} bytes")
else:
    print("JIT not available on this platform")

print(f"\nVoxelVM v{vx.__version__} - All examples complete.")
