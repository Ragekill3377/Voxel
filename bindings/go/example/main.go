package main

/*
#cgo CFLAGS: -I${SRCDIR}/../..
#cgo LDFLAGS: -L${SRCDIR}/../../build -lvoxel_c -lstdc++ -lm -Wl,-rpath,${SRCDIR}/../../build
*/
import "C"

import (
	"fmt"
	"math"
	"sort"
	"time"
	v "voxelvm"
)

func main() {
	fmt.Println("=== Go JIT Benchmark ===")

	N := 1_000_000
	threshold := 500.0
	kL := v.KLanes()
	lane := uint8(kL)

	// Generate data
	data := make([]float64, N)
	rng := uint64(42)
	for i := range data {
		rng = rng*6364136223846793005 + 1
		data[i] = float64(rng%1_000_000) / 1000.0
	}

	var expected float64
	for _, dv := range data {
		if dv > threshold {
			expected += dv
		}
	}
	fmt.Printf("Dataset: %d f64, threshold %.0f, expected sum %.3f\n", N, threshold, expected)

	code := []uint32{
		v.IVLoad(0, 1, 0, 0), v.IVFilterGt(1, 0, 3), v.IVSum(5, 1),
		v.IAddf(0, 0, 5), v.IAdd(1, 1, lane), v.ICmp(1, 2), v.IJnz(-6), v.IHalt(),
	}

	// Warmup
	_ = v.JITRun(code, data, threshold)

	// Measured runs
	var times []float64
	for i := 0; i < 3; i++ {
		t0 := time.Now()
		result := v.JITRun(code, data, threshold)
		elapsed := time.Since(t0)
		us := float64(elapsed.Microseconds())
		times = append(times, us)
		mps := float64(N) / (us / 1e6) / 1e6
		ok := math.Abs(result-expected) < 1e-6
		fmt.Printf("  Go JIT run %d: %.0f us, %.0f M/s, %.3f (exp %.3f) %v\n", i+1, us, mps, result, expected, ok)
	}
	sort.Float64s(times)
	med := times[1]
	fmt.Printf("  Go JIT median: %.0f us, %.0f M/s\n", med, float64(N)/(med/1e6)/1e6)

	// Sort benchmark
	fmt.Println("\n=== Go Sort Benchmark ===")
	times = nil
	for i := 0; i < 3; i++ {
		t0 := time.Now()
		v.SortAscending(data)
		elapsed := time.Since(t0)
		times = append(times, float64(elapsed.Microseconds()))
	}
	sort.Float64s(times)
	fmt.Printf("  SortAscending 1M f64: %.0f us\n", times[1])

	// HashAgg benchmark
	fmt.Println("\n=== Go HashAgg Benchmark ===")
	keys := make([]uint32, N)
	vals := make([]float64, N)
	for i := 0; i < N; i++ {
		keys[i] = uint32(i % 100)
		vals[i] = data[i]
	}
	times = nil
	for i := 0; i < 3; i++ {
		agg := v.NewHashAggregator()
		agg.Init(100)
		t0 := time.Now()
		agg.Accumulate(keys, vals)
		elapsed := time.Since(t0)
		times = append(times, float64(elapsed.Microseconds()))
		agg.Destroy()
	}
	sort.Float64s(times)
	fmt.Printf("  HashAgg 1M rows, 100 groups: %.0f us, %d groups\n", times[1], 100)

	fmt.Println("\nAll Go benchmarks complete!")
}
