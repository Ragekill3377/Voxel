package main

/*
#cgo CFLAGS: -I${SRCDIR}/../..
#cgo LDFLAGS: -L${SRCDIR}/../../build -lvoxel_c -lstdc++ -lm -Wl,-rpath,${SRCDIR}/../../build
*/
import "C"

import (
	"fmt"
	"math"
	v "voxelvm"
)

func main() {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	N := len(data)
	kL := v.KLanes()
	fmt.Printf("kLanes=%d, N=%d\n", kL, N)
	lane := uint8(kL)
	lane16 := int16(kL)

	code := []uint32{
		v.IMov(4, lane16),
		v.IVLoad(0, 1, 0, 0),
		v.IVFilterGt(1, 0, 3),
		v.IVSum(5, 1),
		v.IAddf(0, 0, 5),
		v.IAdd(1, 1, lane),
		v.ICmp(1, 2),
		v.IJnz(-6),
		v.IHalt(),
	}

	eng := v.NewEngine()
	eng.AddSegment(data)
	eng.SetScalarF64(0, 0)
	eng.SetScalarF64(1, 0)
	eng.SetScalar(2, uint64(N))
	eng.SetScalarF64(3, 4.0)
	eng.LoadProgram(code)
	eng.Run()
	fmt.Printf("Interpreter: %.1f (expected 26.0) %s\n", eng.GetScalarF64(0), ifeq(eng.GetScalarF64(0), 26.0))

	jitR := v.JITRun(code, data, 4.0)
	fmt.Printf("JIT:         %.1f (expected 26.0) %s\n", jitR, ifeq(jitR, 26.0))

	idx := v.SortAscending([]float64{3, 1, 4, 1, 5})
	fmt.Printf("Sort:        %v %s\n", idx, ifeq(float64(idx[0]), 1.0))

	top := v.TopK([]float64{3, 1, 4, 1, 5, 9, 2, 6}, 3)
	fmt.Printf("TopK:        %v (expected [9 6 5]) %s\n", top, ifeq(top[0], 9.0))

	agg := v.NewHashAggregator()
	agg.Init(10)
	agg.Accumulate([]uint32{0, 1, 0, 1, 0}, []float64{10, 20, 30, 40, 50})
	fmt.Printf("HashAgg:     %d groups (expected 2) %s\n", agg.GroupCount(), ifeq(float64(agg.GroupCount()), 2.0))

	nb := v.NewNullBitmap(64)
	nb.SetNull(10)
	nb.SetNull(20)
	fmt.Printf("NullBitmap:  %d nulls (expected 2) %s\n", nb.NullCount(), ifeq(float64(nb.NullCount()), 2.0))

	// Test JIT on 1M elements
	bigData := make([]float64, 1_000_000)
	rng := uint64(42)
	for i := range bigData {
		rng = rng*6364136223846793005 + 1
		bigData[i] = float64(rng%1_000_000) / 1000.0
	}
	var expected float64
	for _, dv := range bigData {
		if dv > 500.0 {
			expected += dv
		}
	}
	bigCode := []uint32{
		v.IVLoad(0, 1, 0, 0), v.IVFilterGt(1, 0, 3), v.IVSum(5, 1),
		v.IAddf(0, 0, 5), v.IAdd(1, 1, lane), v.ICmp(1, 2), v.IJnz(-6), v.IHalt(),
	}
	jitBig := v.JITRun(bigCode, bigData, 500.0)
	fmt.Printf("\nBig JIT:     %.3f (expected %.3f)\n", jitBig, expected)

	fmt.Println("\nAll Go tests complete!")
}

func ifeq(a, b float64) string {
	if math.Abs(a-b) < 1e-6 {
		return "OK"
	}
	return "FAIL"
}
