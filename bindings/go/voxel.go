package voxel

/*
#cgo CFLAGS: -I${SRCDIR}/../..
// Link: user must set CGO_LDFLAGS="-L/path/to/build" or install libvoxel_c to system path
// The library name is platform-independent (libvoxel_c.so / .dylib / .dll)
#cgo LDFLAGS: -lvoxel_c -lstdc++ -lm
#include "bindings/voxel_c.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

type Engine struct{ ptr unsafe.Pointer }

func NewEngine() *Engine {
	e := &Engine{ptr: C.voxel_engine_create_f64()}
	runtime.SetFinalizer(e, func(e *Engine) { C.voxel_engine_destroy(e.ptr) })
	return e
}
func (e *Engine) Destroy() { if e.ptr != nil { C.voxel_engine_destroy(e.ptr); e.ptr = nil } }
func (e *Engine) AddSegment(data []float64) int {
	if len(data) == 0 { return 0 }
	return int(C.voxel_engine_add_segment(e.ptr, (*C.double)(&data[0]), C.size_t(len(data))))
}
func (e *Engine) LoadProgram(code []uint32) {
	if len(code) == 0 { return }
	C.voxel_engine_load_program(e.ptr, (*C.uint32_t)(&code[0]), C.size_t(len(code)))
}
func (e *Engine) Run()                        { C.voxel_engine_run(e.ptr) }
func (e *Engine) RunParallel(s int, t uint32, r uint8) { C.voxel_engine_run_parallel(e.ptr, C.size_t(s), C.uint32_t(t), C.uint8_t(r)) }
func (e *Engine) SetScalar(r int, v uint64)   { C.voxel_engine_set_scalar(e.ptr, C.size_t(r), C.uint64_t(v)) }
func (e *Engine) GetScalar(r int) uint64      { return uint64(C.voxel_engine_get_scalar(e.ptr, C.size_t(r))) }
func (e *Engine) SetScalarF64(r int, v float64) { C.voxel_engine_set_scalar_f64(e.ptr, C.size_t(r), C.double(v)) }
func (e *Engine) GetScalarF64(r int) float64  { return float64(C.voxel_engine_get_scalar_f64(e.ptr, C.size_t(r))) }
func KLanes() int                              { return int(C.voxel_engine_k_lanes()) }

func JITRun(code []uint32, data []float64, threshold float64) float64 {
	return float64(C.voxel_jit_run((*C.uint32_t)(&code[0]), C.size_t(len(code)), (*C.double)(&data[0]), C.size_t(len(data)), C.double(threshold)))
}

type NullBitmap struct{ ptr unsafe.Pointer }
func NewNullBitmap(count int) *NullBitmap {
	b := &NullBitmap{ptr: C.voxel_null_create(C.size_t(count))}
	runtime.SetFinalizer(b, func(b *NullBitmap) { C.voxel_null_destroy(b.ptr) })
	return b
}
func (b *NullBitmap) Destroy()  { if b.ptr != nil { C.voxel_null_destroy(b.ptr); b.ptr = nil } }
func (b *NullBitmap) IsNull(r int) bool  { return C.voxel_null_is_null(b.ptr, C.size_t(r)) != 0 }
func (b *NullBitmap) SetNull(r int)      { C.voxel_null_set_null(b.ptr, C.size_t(r)) }
func (b *NullBitmap) SetValid(r int)     { C.voxel_null_set_valid(b.ptr, C.size_t(r)) }
func (b *NullBitmap) NullCount() int     { return int(C.voxel_null_null_count(b.ptr)) }
func (b *NullBitmap) ValidCount() int    { return int(C.voxel_null_valid_count(b.ptr)) }

type HashAggregator struct{ ptr unsafe.Pointer }
func NewHashAggregator() *HashAggregator {
	a := &HashAggregator{ptr: C.voxel_agg_create()}
	runtime.SetFinalizer(a, func(a *HashAggregator) { C.voxel_agg_destroy(a.ptr) })
	return a
}
func (a *HashAggregator) Destroy() { if a.ptr != nil { C.voxel_agg_destroy(a.ptr); a.ptr = nil } }
func (a *HashAggregator) Init(g int) { C.voxel_agg_init(a.ptr, C.size_t(g)) }
func (a *HashAggregator) Accumulate(keys []uint32, values []float64) {
	n := len(keys); if len(values) < n { n = len(values) }
	C.voxel_agg_accumulate(a.ptr, (*C.uint32_t)(&keys[0]), (*C.double)(&values[0]), C.size_t(n))
}
func (a *HashAggregator) GroupCount() int { return int(C.voxel_agg_group_count(a.ptr)) }

func SortAscending(data []float64) []uint32 {
	n := len(data); out := make([]uint32, n)
	C.voxel_sort_ascending((*C.double)(&data[0]), C.size_t(n), (*C.uint32_t)(&out[0]))
	return out
}
func SortDescending(data []float64) []uint32 {
	n := len(data); out := make([]uint32, n)
	C.voxel_sort_descending((*C.double)(&data[0]), C.size_t(n), (*C.uint32_t)(&out[0]))
	return out
}
func TopK(data []float64, k int) []float64 {
	out := make([]float64, k)
	C.voxel_topk((*C.double)(&data[0]), C.size_t(len(data)), C.size_t(k), (*C.double)(&out[0]), 1)
	return out
}
func BottomK(data []float64, k int) []float64 {
	out := make([]float64, k)
	C.voxel_topk((*C.double)(&data[0]), C.size_t(len(data)), C.size_t(k), (*C.double)(&out[0]), 0)
	return out
}

type ThreadPool struct{ ptr unsafe.Pointer }
func NewThreadPool(t uint32) *ThreadPool {
	p := &ThreadPool{ptr: C.voxel_pool_create(C.uint32_t(t))}
	runtime.SetFinalizer(p, func(p *ThreadPool) { C.voxel_pool_destroy(p.ptr) })
	return p
}
func (p *ThreadPool) Destroy() { if p.ptr != nil { C.voxel_pool_destroy(p.ptr); p.ptr = nil } }
func (p *ThreadPool) WaitAll()            { C.voxel_pool_wait_all(p.ptr) }
func (p *ThreadPool) ThreadCount() int    { return int(C.voxel_pool_thread_count(p.ptr)) }

// Instruction factories
func IVLoad(vd, ra, seg, cnt uint8) uint32   { return uint32(C.voxel_instr_vload(C.uint8_t(vd),C.uint8_t(ra),C.uint8_t(seg),C.uint8_t(cnt))) }
func IVFilterGt(vd,va,rb uint8) uint32        { return uint32(C.voxel_instr_vfilter_gt(C.uint8_t(vd),C.uint8_t(va),C.uint8_t(rb))) }
func IVFilterGe(vd,va,rb uint8) uint32        { return uint32(C.voxel_instr_vfilter_ge(C.uint8_t(vd),C.uint8_t(va),C.uint8_t(rb))) }
func IVFilterLt(vd,va,rb uint8) uint32        { return uint32(C.voxel_instr_vfilter_lt(C.uint8_t(vd),C.uint8_t(va),C.uint8_t(rb))) }
func IVFilterLe(vd,va,rb uint8) uint32        { return uint32(C.voxel_instr_vfilter_le(C.uint8_t(vd),C.uint8_t(va),C.uint8_t(rb))) }
func IVSum(rd,va uint8) uint32                { return uint32(C.voxel_instr_vsum(C.uint8_t(rd),C.uint8_t(va))) }
func IVRedMin(rd,va uint8) uint32             { return uint32(C.voxel_instr_vred_min(C.uint8_t(rd),C.uint8_t(va))) }
func IVRedMax(rd,va uint8) uint32             { return uint32(C.voxel_instr_vred_max(C.uint8_t(rd),C.uint8_t(va))) }
func IVCount(rd,va uint8) uint32              { return uint32(C.voxel_instr_vcount(C.uint8_t(rd),C.uint8_t(va))) }
func IAddf(rd,ra,rb uint8) uint32             { return uint32(C.voxel_instr_addf(C.uint8_t(rd),C.uint8_t(ra),C.uint8_t(rb))) }
func IAdd(rd,ra,rb uint8) uint32              { return uint32(C.voxel_instr_add(C.uint8_t(rd),C.uint8_t(ra),C.uint8_t(rb))) }
func ISub(rd,ra,rb uint8) uint32              { return uint32(C.voxel_instr_sub(C.uint8_t(rd),C.uint8_t(ra),C.uint8_t(rb))) }
func IMul(rd,ra,rb uint8) uint32              { return uint32(C.voxel_instr_mul(C.uint8_t(rd),C.uint8_t(ra),C.uint8_t(rb))) }
func IMov(rd uint8, imm int16) uint32         { return uint32(C.voxel_instr_mov(C.uint8_t(rd),C.int16_t(imm))) }
func IMovr(rd,ra uint8) uint32                { return uint32(C.voxel_instr_movr(C.uint8_t(rd),C.uint8_t(ra))) }
func ICmp(ra,rb uint8) uint32                 { return uint32(C.voxel_instr_cmp(C.uint8_t(ra),C.uint8_t(rb))) }
func IJnz(off int16) uint32                   { return uint32(C.voxel_instr_jnz(C.int16_t(off))) }
func IJz(off int16) uint32                    { return uint32(C.voxel_instr_jz(C.int16_t(off))) }
func IJmp(off int16) uint32                   { return uint32(C.voxel_instr_jmp(C.int16_t(off))) }
func IHalt() uint32                           { return uint32(C.voxel_instr_halt()) }
