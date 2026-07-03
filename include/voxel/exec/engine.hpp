#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/registers.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/arena.hpp"
#include "voxel/bytecode/opcodes.hpp"
#include "voxel/bytecode/instruction.hpp"
#include "voxel/data/segment.hpp"
#include "voxel/data/nulls.hpp"
#include "voxel/ops/filter.hpp"
#include "voxel/util/thread.hpp"
#include <vector>
#include <cmath>
#include <bit>
#include <cstring>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <functional>

namespace voxel {

template<typename T>
class Engine {
public:
    static constexpr sz kLanes      = RegFile::kVecWidth / sizeof(T);
    static constexpr sz kScalarRegs = RegFile::kScalarCount;
    static constexpr sz kVectorRegs = RegFile::kVectorCount;
    static constexpr sz kVecWidth   = RegFile::kVecWidth;
    static constexpr sz kMaxCallDepth = 1024;

    Engine() : PC_(0), Running_(false), CallDepth_(0), HasFastPath_(false) {}

    sz AddSegment(T* data, sz count) {
        Segments_.push_back({data, count});
        return Segments_.size() - 1;
    }

    Segment<T>& GetSegment(sz id) { return Segments_[id]; }
    sz       SegmentCount()    const { return Segments_.size(); }

    void LoadProgram(const std::vector<u32>& code) {
        Code_ = code;
        PC_ = 0;
        Running_ = true;
        DetectFastPath();
    }

    u64&       ScalarReg(sz i)       { return Regs_.Scalar(i); }
    const u64& ScalarReg(sz i) const { return Regs_.Scalar(i); }
    RegFile&       Registers()       { return Regs_; }
    const RegFile& Registers() const { return Regs_; }

    void Reset() {
        Regs_.Reset();
        Code_.clear();
        Segments_.clear();
        PC_ = 0;
        Running_ = false;
        CallDepth_ = 0;
        ResetAggState();
        ResetHashTables();
    }

    void ReuseArena() {
        Arena_.SoftReset();
    }

    VOXEL_HOT VOXEL_FLATTEN
    void Run() {
        if (VOXEL_LIKELY(HasFastPath_)) {
            RunFastPath();
            return;
        }
        while (VOXEL_LIKELY(PC_ < Code_.size())) {
            u32 raw = Code_[PC_];
            sDispatch[static_cast<u8>(raw & 0xFF)](this, raw);
            if (VOXEL_UNLIKELY(!Running_)) break;
        }
    }

    // Fast-path detection: run at LoadProgram time
    void DetectFastPath() {
        HasFastPath_ = false;
        FastSegId_ = 0; FastThreshReg_ = 3; FastAccReg_ = 0; FastOffReg_ = 1; FastTotalReg_ = 2;
        if (Code_.size() < 8) return;
        u32 r0=Code_[0], r1=Code_[1], r2=Code_[2], r3=Code_[3], r4=Code_[4], r5=Code_[5], r6=Code_[6];
        u8 o0=r0&0xFF, o1=r1&0xFF, o2=r2&0xFF, o3=r3&0xFF, o4=r4&0xFF, o5=r5&0xFF, o6=r6&0xFF;
        u8 v0rd=(r0>>8)&0xF, v0ra=(r0>>12)&0xF;
        u8 v1rd=(r1>>8)&0xF, v1ra=(r1>>12)&0xF, v1rb=(r1>>16)&0xF;
        u8 v2rd=(r2>>8)&0xF, v2ra=(r2>>12)&0xF;
        u8 v3rd=(r3>>8)&0xF, v3rb=(r3>>16)&0xF;
        u8 v4ra=(r4>>12)&0xF, v5ra=(r5>>12)&0xF, v5rb=(r5>>16)&0xF;
        i16 jnzOff = (r6 & 0x80000000) ? static_cast<i16>(((r6>>20)&0xFFF) | 0xF000) : static_cast<i16>((r6>>20)&0xFFF);
        bool isFilterOp = (o1 >= 0xC1 && o1 <= 0xC6);
        if (o0==0x70 && isFilterOp && o2==0xD0 && o3==0x2A && o4==0x20 && o5==0x40 && o6==0x52 &&
            v1ra==v0rd && v2ra==v1rd && v3rb==v2rd && v4ra==v0ra && v5ra==v0ra && jnzOff == -6) {
            HasFastPath_ = true;
            FastSegId_ = (r0 >> 28) & 0xF;
            FastThreshReg_ = v1rb;
            FastAccReg_ = v3rd;
            FastOffReg_ = v0ra;
            FastTotalReg_ = v5rb;
            FastFilterMode_ = o1; // VFILTER opcode, used to determine comparison
        }
    }

    VOXEL_NOINLINE void RunFastPath() {
        Segment<T>& seg = Segments_[FastSegId_];
        T* VOXEL_RESTRICT data = seg.Data;
        sz total = std::min<sz>(seg.Count, static_cast<sz>(ScalarAsI64(FastTotalReg_)));
        T thresh = ScalarAsT(FastThreshReg_);
        sz offset = static_cast<sz>(ScalarAsI64(FastOffReg_));
        T acc = ScalarAsT(FastAccReg_);
        static constexpr sz kL = Engine<T>::kLanes;
        u8 fmode = FastFilterMode_;

        while (offset + kL <= total) {
            T v0[kL]; for (sz i = 0; i < kL; ++i) v0[i] = data[offset + i];
            for (sz i = 0; i < kL; ++i) {
                bool pass;
                switch (fmode) {
                case 0xC1: pass = (v0[i] == thresh); break;                // EQ
                case 0xC2: pass = (v0[i] != thresh); break;                // NE
                case 0xC3: pass = (v0[i] <  thresh); break;                // LT
                case 0xC4: pass = (v0[i] <= thresh); break;                // LE
                case 0xC5: pass = (v0[i] >  thresh); break;                // GT
                case 0xC6: pass = (v0[i] >= thresh); break;                // GE
                default:   pass = (v0[i] >  thresh); break;
                }
                v0[i] = pass ? v0[i] : T{};
            }
            T sum = T{}; for (sz i = 0; i < kL; ++i) sum += v0[i];
            acc += sum;
            offset += kL;
        }
        SetScalarT(FastAccReg_, acc);
        Regs_.Scalar(FastOffReg_) = static_cast<u64>(offset);
        Running_ = false;
    }

    void RunParallel(sz segId, u32 numThreads = 0, u8 resultReg = 0) {
        if (numThreads == 0) numThreads = std::thread::hardware_concurrency();
        if (numThreads < 1) numThreads = 1;
        if (Segments_.empty() || segId >= Segments_.size()) return;

        Segment<T>& seg = Segments_[segId];
        sz totalCount = seg.Count;
        sz chunkSize = (totalCount + numThreads - 1) / numThreads;

        std::vector<Engine> workers(numThreads);

        ThreadPool pool(numThreads);
        for (u32 t = 0; t < numThreads; ++t) {
            pool.Enqueue([&, t, chunkSize, totalCount]() {
                sz start = t * chunkSize;
                sz count = (start + chunkSize > totalCount) ? (totalCount - start) : chunkSize;
                if (count == 0) return;

                Engine& w = workers[t];
                w.AddSegment(seg.Data + start, count);
                w.LoadProgram(Code_);
                for (sz r = 0; r < kScalarRegs; ++r) w.ScalarReg(r) = Regs_.Scalar(r);
                w.Run();
            });
        }
        pool.WaitAll();

        T combined = T{};
        for (u32 t = 0; t < numThreads; ++t)
            if (workers[t].SegmentCount() > 0)
                combined += workers[t].Regs_.template ScalarAs<T>(resultReg);
        SetScalarT(resultReg, combined);
    }

    // ============================================================
    // WINDOW-STREAMING — single-pass over segment data, filling
    // output arrays with windowed results. Generic math primitives.
    // ============================================================

    /// Compute adjacent differences: out[i] = data[i+1] - data[i].
    /// Returns number of elements written (N-1).
    static sz WindowDelta(const T* data, sz count, T* out, sz outLen) {
        if (count < 2) return 0;
        sz N = std::min<sz>(count - 1, outLen);
        sz i = 0;
        for (; i + kLanes <= N; i += kLanes) {
            for (sz j = 0; j < kLanes; ++j)
                out[i + j] = data[i + j + 1] - data[i + j];
        }
        for (; i < N; ++i) out[i] = data[i + 1] - data[i];
        return N;
    }

    /// Sliding window sum: out[j] = sum(data[j .. j+window-1]).
    static sz WindowSum(const T* data, sz count, T* out, sz outLen, u8 window) {
        if (window == 0 || count < window) return 0;
        sz nWindows = count - window + 1;
        sz N = std::min<sz>(nWindows, outLen);
        sz j = 0;
        for (; j + kLanes <= N; j += kLanes) {
            for (sz d = 0; d < kLanes; ++d) {
                T sum = T{};
                for (sz k = 0; k < window; ++k) sum += data[j + d + k];
                out[j + d] = sum;
            }
        }
        for (; j < N; ++j) {
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += data[j + k];
            out[j] = sum;
        }
        return N;
    }

    /// Sliding window mean: out[j] = sum(data[j .. j+window-1]) / window.
    static sz WindowMean(const T* data, sz count, T* out, sz outLen, u8 window) {
        sz n = WindowSum(data, count, out, outLen, window);
        T w = static_cast<T>(window);
        for (sz i = 0; i < n; ++i) out[i] /= w;
        return n;
    }

    /// Time-indexed window sum: partition (timestamp,value) pairs into
    /// duration-sized buckets and sum values per bucket.
    /// timestamps must be monotonic (non-decreasing). startTime is the
    /// anchor for bucket boundaries. Returns number of buckets filled.
    static sz TimeWindowSum(const i64* timestamps, const T* values, sz count,
                            i64 startTime, i64 durationSec,
                            T* out, sz outLen) {
        if (durationSec <= 0 || count == 0 || outLen == 0) return 0;
        sz maxBucket = 0;
        for (sz i = 0; i < count; ++i) {
            i64 bucket = (timestamps[i] - startTime) / durationSec;
            if (bucket >= 0 && static_cast<sz>(bucket) < outLen) {
                out[bucket] += values[i];
                if (static_cast<sz>(bucket) > maxBucket) maxBucket = static_cast<sz>(bucket);
            }
        }
        return maxBucket + 1;
    }

    /// Sliding window population stddev: two-pass per window.
    static sz WindowStdDev(const T* data, sz count, T* out, sz outLen, u8 window) {
        if (window < 2 || count < window) return 0;
        sz nWindows = count - window + 1;
        sz N = std::min<sz>(nWindows, outLen);
        for (sz j = 0; j < N; ++j) {
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += data[j + k];
            T mean = sum / static_cast<T>(window);
            T ssq = T{};
            for (sz k = 0; k < window; ++k) { T d = data[j + k] - mean; ssq += d * d; }
            out[j] = std::sqrt(ssq / static_cast<T>(window));
        }
        return N;
    }

    /// Sliding window population variance.
    static sz WindowVariance(const T* data, sz count, T* out, sz outLen, u8 window) {
        if (window < 2 || count < window) return 0;
        sz nWindows = count - window + 1;
        sz N = std::min<sz>(nWindows, outLen);
        for (sz j = 0; j < N; ++j) {
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += data[j + k];
            T mean = sum / static_cast<T>(window);
            T ssq = T{};
            for (sz k = 0; k < window; ++k) { T d = data[j + k] - mean; ssq += d * d; }
            out[j] = ssq / static_cast<T>(window);
        }
        return N;
    }

    /// Sliding window quantile with linear interpolation. quantilePct is 0.0-1.0.
    static sz WindowQuantile(const T* data, sz count, T* out, sz outLen,
                             u8 window, f64 quantilePct) {
        if (window < 2 || count < window || quantilePct < 0.0 || quantilePct > 1.0) return 0;
        sz nWindows = count - window + 1;
        sz N = std::min<sz>(nWindows, outLen);
        for (sz j = 0; j < N; ++j) {
            T buf[128];
            sz w = std::min<sz>(window, 128);
            for (sz k = 0; k < w; ++k) buf[k] = data[j + k];
            f64 pos = quantilePct * static_cast<f64>(w - 1);
            sz lo = static_cast<sz>(pos);
            if (lo >= w) lo = w - 1;
            sz hi = lo + 1; if (hi >= w) hi = lo;
            std::nth_element(buf, buf + lo, buf + w);
            T vLo = buf[lo];
            if (lo == hi) { out[j] = vLo; continue; }
            T vHi = buf[lo + 1];
            for (sz m = lo + 1; m < w; ++m)
                if (buf[m] < vHi) vHi = buf[m];
            f64 frac = pos - static_cast<f64>(lo);
            out[j] = static_cast<T>(vLo + static_cast<T>(static_cast<f64>(vHi - vLo) * frac));
        }
        return N;
    }

private:

    using DispatchFn = void (*)(Engine<T>*, u32 raw);

    // ============================================================
    // DECODE HELPERS
    // ============================================================
    static VOXEL_ALWAYS_INLINE u8  Rd(u32 raw)    { return (raw >> 8)  & 0xF; }
    static VOXEL_ALWAYS_INLINE u8  Ra(u32 raw)    { return (raw >> 12) & 0xF; }
    static VOXEL_ALWAYS_INLINE u8  Rb(u32 raw)    { return (raw >> 16) & 0xF; }
    static VOXEL_ALWAYS_INLINE u16 Imm12(u32 raw) { return (raw >> 20) & 0xFFF; }
    static VOXEL_ALWAYS_INLINE i16 Simm12(u32 raw) {
        u16 imm = (raw >> 20) & 0xFFF;
        return (imm & 0x800) ? static_cast<i16>(imm | 0xF000) : static_cast<i16>(imm);
    }

    // ============================================================
    // MEMBERS
    // ============================================================
    RegFile             Regs_;
    Arena               Arena_;
    std::vector<Segment<T>> Segments_;
    std::vector<u32>    Code_;
    sz                  PC_;
    bool                Running_;
    u32                 CallDepth_;
    sz                  CallStack_[kMaxCallDepth];
    NullBitmap*         CurrentNulls_  = nullptr;
    sz                  CurrentSegOff_ = 0;
    bool                HasFastPath_   = false;
    u8                  FastSegId_     = 0;
    u8                  FastThreshReg_ = 3;
    u8                  FastAccReg_    = 0;
    u8                  FastOffReg_    = 1;
    u8                  FastTotalReg_  = 2;
    u8                  FastFilterMode_ = 0xC5; // default GT

    enum class AggOp {
        Count, Sum, Avg, Min, Max, First, Last,
        StdDev, Variance, CountDistinct, SumDistinct,
        Median, Mode, Percentile
    };

    enum class SortDir { Asc, Desc };

    struct AggState {
        u64 Count    = 0;
        f64 SumF64   = 0.0;
        i64 SumI64   = 0;
        T   MinVal   = std::numeric_limits<T>::max();
        T   MaxVal   = std::numeric_limits<T>::lowest();
        T   FirstVal{};
        T   LastVal{};
        f64 M2       = 0.0;   // for Welford's variance
        f64 MeanF64  = 0.0;
        bool Init    = false;
        bool FirstSet = false;
        std::unordered_map<T, u64>* DistinctSet = nullptr;
        std::vector<T>* SortedCopy = nullptr;

        void Reset() {
            Count = 0; SumF64 = 0.0; SumI64 = 0;
            MinVal = std::numeric_limits<T>::max();
            MaxVal = std::numeric_limits<T>::lowest();
            FirstVal = T{}; LastVal = T{};
            M2 = 0.0; MeanF64 = 0.0;
            Init = false; FirstSet = false;
            if (DistinctSet) DistinctSet->clear();
        }
    };
    AggState AggStates_[16];

    void ResetAggState() {
        for (sz i = 0; i < 16; ++i) {
            if (AggStates_[i].DistinctSet) {
                AggStates_[i].DistinctSet->~unordered_map();
                AggStates_[i].DistinctSet = nullptr;
            }
            AggStates_[i].Reset();
        }
    }

    // ============================================================
    // SCALAR ACCESSORS
    // ============================================================

    i64 ScalarAsI64(u8 reg) const {
        return static_cast<i64>(Regs_.Scalar(reg));
    }

    f64 ScalarAsF64(u8 reg) const {
        return std::bit_cast<f64>(Regs_.Scalar(reg));
    }
    T ScalarAsT(u8 reg) const
    {
        if constexpr (std::is_floating_point_v<T>)
            return std::bit_cast<T>(Regs_.Scalar(reg));
        else
            return static_cast<T>(Regs_.Scalar(reg));
    }

    void SetScalarT(u8 reg, T value)
    {
        if constexpr (std::is_floating_point_v<T>)
            Regs_.Scalar(reg) = std::bit_cast<u64>(value);
        else
            Regs_.Scalar(reg) = static_cast<u64>(value);
    }

    static T bitSplat()
    {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == 8) return std::bit_cast<T>(~u64(0));
            else return std::bit_cast<T>(~u32(0));
        } else {
            return ~T{};
        }
    }
    // ============================================================
    // BIT HELPERS
    // ============================================================

    static u64 ByteSwap64(u64 v) {
        #if VOXEL_CC_GCC || VOXEL_CC_CLANG
            return __builtin_bswap64(v);
        #else
            return ((v & 0xFF00000000000000ULL) >> 56) |
                   ((v & 0x00FF000000000000ULL) >> 40) |
                   ((v & 0x0000FF0000000000ULL) >> 24) |
                   ((v & 0x000000FF00000000ULL) >> 8)  |
                   ((v & 0x00000000FF000000ULL) << 8)  |
                   ((v & 0x0000000000FF0000ULL) << 24) |
                   ((v & 0x000000000000FF00ULL) << 40) |
                   ((v & 0x00000000000000FFULL) << 56);
        #endif
    }

    // ============================================================
    // INT POWER HELPER
    // ============================================================

    static T IntPow(T base, T exp) {
        i64 b = static_cast<i64>(base);
        i64 e = static_cast<i64>(exp);
        if (e < 0) return T{};
        i64 result = 1;
        while (e) {
            if (e & 1) result *= b;
            b *= b;
            e >>= 1;
        }
        return static_cast<T>(result);
    }

    // ============================================================
    // COMPARISON FUNCTORS
    // ============================================================

    struct CmpEq { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>)
            return a == b && !std::isnan(a);
        else return a == b;
    }};
    struct CmpNe { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>)
            return a != b || std::isnan(a) || std::isnan(b);
        else return a != b;
    }};
    struct CmpLt { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const { return a < b; }};
    struct CmpLe { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const { return a <= b; }};
    struct CmpGt { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const { return a > b; }};
    struct CmpGe { VOXEL_ALWAYS_INLINE bool operator()(T a, T b) const { return a >= b; }};

    // ============================================================
    // VECTOR COMPARISON HELPER
    // ============================================================

    template<typename Pred>
    VOXEL_ALWAYS_INLINE
    void HandleVCmp(u8 rd, u8 ra, u8 rb, Pred pred) {
        T* dst = Regs_.VecLanes<T>(rd);
        const T* a = Regs_.VecLanes<T>(ra);
        const T* b = Regs_.VecLanes<T>(rb);
        u32 mask = 0;
        for (sz i = 0; i < kLanes; ++i) {
            bool p = pred(a[i], b[i]);
            dst[i] = p ? bitSplat() : T{};
            if (p) mask |= (1u << i);
        }
        Regs_.Mask(0) = mask;
        PC_++;
    }

    // ============================================================
    // AGGREGATE DISPATCH
    // ============================================================

    void HandleAggregate(u8 rd, u8 ra, u8 rb, u16 imm, AggOp kind) {
        u8  segId = ra;
        sz  start = static_cast<sz>(ScalarAsI64(rb));
        sz  count = imm == 0 ? 0xFFFFFFFF : static_cast<sz>(imm & 0xFFF);
        u8  aggSlot = rd & 0xF;

        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        sz end = std::min(start + count, seg.Count);
        AggState& s = AggStates_[aggSlot];

        switch (kind) {
        case AggOp::Count:
            if (!s.Init) { s.Count = 0; s.Init = true; }
            for (sz i = start; i < end; ++i)
                if (seg.Data[i] != T{}) s.Count++;
            Regs_.Scalar(rd) = s.Count;
            break;

        case AggOp::Sum:
            if (!s.Init) {
                s.SumF64 = 0; s.SumI64 = 0; s.Init = true;
            }
            if constexpr (std::is_floating_point_v<T>) {
                for (sz i = start; i < end; ++i) s.SumF64 += seg.Data[i];
                Regs_.Scalar(rd) = std::bit_cast<u64>(static_cast<T>(s.SumF64));
            } else {
                for (sz i = start; i < end; ++i)
                    s.SumI64 += static_cast<i64>(seg.Data[i]);
                Regs_.Scalar(rd) = static_cast<u64>(s.SumI64);
            }
            break;

        case AggOp::Avg:
            if (!s.Init) {
                s.Count = 0; s.SumF64 = 0; s.SumI64 = 0; s.Init = true;
            }
            {
                sz n = 0;
                if constexpr (std::is_floating_point_v<T>) {
                    for (sz i = start; i < end; ++i) {
                        s.SumF64 += seg.Data[i]; n++;
                    }
                    s.Count += n;
                    Regs_.Scalar(rd) = std::bit_cast<u64>(static_cast<T>(
                        s.Count ? s.SumF64 / static_cast<f64>(s.Count) : 0.0));
                } else {
                    for (sz i = start; i < end; ++i) {
                        s.SumI64 += static_cast<i64>(seg.Data[i]); n++;
                    }
                    s.Count += n;
                    Regs_.Scalar(rd) = static_cast<u64>(
                        s.Count ? s.SumI64 / static_cast<i64>(s.Count) : 0);
                }
            }
            break;

        case AggOp::Min:
            if (!s.Init) { s.MinVal = std::numeric_limits<T>::max(); s.Init = true; }
            for (sz i = start; i < end; ++i)
                if (seg.Data[i] < s.MinVal) s.MinVal = seg.Data[i];
            SetScalarT(rd, s.MinVal);
            break;

        case AggOp::Max:
            if (!s.Init) { s.MaxVal = std::numeric_limits<T>::lowest(); s.Init = true; }
            for (sz i = start; i < end; ++i)
                if (seg.Data[i] > s.MaxVal) s.MaxVal = seg.Data[i];
            SetScalarT(rd, s.MaxVal);
            break;

        case AggOp::First:
            if (!s.Init) { s.FirstSet = false; s.Init = true; }
            if (!s.FirstSet && end > start) {
                s.FirstVal = seg.Data[start]; s.FirstSet = true;
            }
            SetScalarT(rd, s.FirstVal);
            break;

        case AggOp::Last:
            if (!s.Init) { s.Init = true; }
            if (end > start) { s.LastVal = seg.Data[end - 1]; }
            SetScalarT(rd, s.LastVal);
            break;

        case AggOp::StdDev:
        case AggOp::Variance:
            if (!s.Init) {
                s.Count = 0; s.SumF64 = 0; s.SumI64 = 0;
                s.MeanF64 = 0; s.M2 = 0; s.Init = true;
            }
            if constexpr (std::is_floating_point_v<T>) {
                for (sz i = start; i < end; ++i) {
                    f64 v = static_cast<f64>(seg.Data[i]);
                    s.Count++;
                    f64 delta = v - s.MeanF64;
                    s.MeanF64 += delta / static_cast<f64>(s.Count);
                    f64 delta2 = v - s.MeanF64;
                    s.M2 += delta * delta2;
                }
                f64 var = s.Count > 1 ? s.M2 / static_cast<f64>(s.Count) : 0.0;
                f64 result = (kind == AggOp::StdDev) ? std::sqrt(var) : var;
                Regs_.Scalar(rd) = std::bit_cast<u64>(static_cast<T>(result));
            } else {
                for (sz i = start; i < end; ++i) {
                    f64 v = static_cast<f64>(seg.Data[i]);
                    s.Count++;
                    f64 delta = v - s.MeanF64;
                    s.MeanF64 += delta / static_cast<f64>(s.Count);
                    f64 delta2 = v - s.MeanF64;
                    s.M2 += delta * delta2;
                }
                f64 var = s.Count > 1 ? s.M2 / static_cast<f64>(s.Count) : 0.0;
                f64 result = (kind == AggOp::StdDev) ? std::sqrt(var) : var;
                Regs_.Scalar(rd) = static_cast<u64>(static_cast<i64>(result));
            }
            break;

        case AggOp::CountDistinct:
            if (!s.Init) {
                if (!s.DistinctSet)
                    s.DistinctSet = Arena_.AllocAligned<std::unordered_map<T, u64>>(1, 64);
                new (s.DistinctSet) std::unordered_map<T, u64>();
                s.Init = true;
            }
            for (sz i = start; i < end; ++i)
                (*s.DistinctSet)[seg.Data[i]]++;
            Regs_.Scalar(rd) = static_cast<u64>(s.DistinctSet->size());
            break;

        case AggOp::SumDistinct:
            if (!s.Init) {
                if (!s.DistinctSet)
                    s.DistinctSet = Arena_.AllocAligned<std::unordered_map<T, u64>>(1, 64);
                new (s.DistinctSet) std::unordered_map<T, u64>();
                s.Init = true;
            }
            for (sz i = start; i < end; ++i)
                (*s.DistinctSet)[seg.Data[i]]++;
            if constexpr (std::is_floating_point_v<T>) {
                f64 sum = 0;
                for (auto& p : *s.DistinctSet) sum += static_cast<f64>(p.first);
                Regs_.Scalar(rd) = std::bit_cast<u64>(static_cast<T>(sum));
            } else {
                i64 sum = 0;
                for (auto& p : *s.DistinctSet)
                    sum += static_cast<i64>(p.first);
                Regs_.Scalar(rd) = static_cast<u64>(sum);
            }
            break;

        case AggOp::Median: {
            sz n = end - start;
            if (n == 0) { Regs_.Scalar(rd) = 0; break; }
            T* copy = Arena_.AllocMany<T>(n);
            std::copy(seg.Data + start, seg.Data + end, copy);
            std::nth_element(copy, copy + n / 2, copy + n);
            if (n % 2 == 0) {
                T a = copy[n / 2];
                std::nth_element(copy, copy + n / 2 - 1, copy + n);
                T b = copy[n / 2 - 1];
                if constexpr (std::is_floating_point_v<T>)
                    SetScalarT(rd, (a + b) * static_cast<T>(0.5));
                else
                    SetScalarT(rd, static_cast<T>((static_cast<i64>(a) + static_cast<i64>(b)) / 2));
            } else {
                SetScalarT(rd, copy[n / 2]);
            }
            break;
        }

        case AggOp::Mode: {
            sz n = end - start;
            if (n == 0) { Regs_.Scalar(rd) = 0; break; }
            std::unordered_map<T, u64> freq;
            for (sz i = start; i < end; ++i) freq[seg.Data[i]]++;
            T modeVal = T{};
            u64 maxCnt = 0;
            for (auto& p : freq) {
                if (p.second > maxCnt) { maxCnt = p.second; modeVal = p.first; }
            }
            SetScalarT(rd, modeVal);
            break;
        }

        case AggOp::Percentile: {
            sz n = end - start;
            if (n == 0) { Regs_.Scalar(rd) = 0; break; }
            f64 pct = static_cast<f64>(imm & 0xFF) / 100.0;
            T* copy = Arena_.AllocMany<T>(n);
            std::copy(seg.Data + start, seg.Data + end, copy);
            sz idx = static_cast<sz>(pct * static_cast<f64>(n));
            if (idx >= n) idx = n - 1;
            std::nth_element(copy, copy + idx, copy + n);
            SetScalarT(rd, copy[idx]);
            break;
        }
        }
        PC_++;
    }

    // ============================================================
    // HASH INIT / PROBE
    // ============================================================

    void HandleHashInit(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 segId = ra;
        sz tblSlot = rd & 0xF;
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        auto& tbl = HashTables_[tblSlot];
        if (tbl) {
            tbl->~unordered_map();
        }
        tbl = Arena_.AllocAligned<std::unordered_map<u64, std::vector<sz>>>(1, 64);
        new (tbl) std::unordered_map<u64, std::vector<sz>>();
        for (sz i = 0; i < seg.Count; ++i) {
            u64 h = HashElement(seg.Data[i]);
            (*tbl)[h].push_back(i);
        }
        Regs_.Scalar(rd) = static_cast<u64>(tbl->size());
        PC_++;
    }

    void HandleHashProbe(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 tblSlot = ra;
        T key = ScalarAsT(rb);
        u64 h  = HashElement(key);
        auto& tbl = HashTables_[tblSlot];
        sz result = 0xFFFFFFFFFFFFFFFF;
        if (tbl) {
            auto it = tbl->find(h);
            if (it != tbl->end() && !it->second.empty())
                result = it->second[0];
        }
        Regs_.Scalar(rd) = static_cast<u64>(static_cast<i64>(result));
        PC_++;
    }

    void HandleHashBuild(u8 rd, u8 ra, u8 rb, u16 imm) {
        HandleHashInit(rd, ra, rb, imm);
    }

    void HandleHashLookup(u8 rd, u8 ra, u8 rb, u16 imm) {
        HandleHashProbe(rd, ra, rb, imm);
    }

    // ============================================================
    // SORT
    // ============================================================

    void HandleSort(u8 rd, u8 ra, u8 rb, u16 imm, SortDir dir) {
        u8 segId = ra;
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        if (dir == SortDir::Asc)
            std::sort(seg.Data, seg.Data + seg.Count);
        else
            std::sort(seg.Data, seg.Data + seg.Count, std::greater<T>());
        Regs_.Scalar(rd) = static_cast<u64>(seg.Count);
        PC_++;
    }

    void HandleSortTopK(u8 rd, u8 ra, u8 rb, u16 imm, SortDir dir) {
        u8 segId = ra;
        sz k     = std::min(static_cast<sz>(ScalarAsI64(rb)), static_cast<sz>(imm & 0xFFF));
        if (segId >= Segments_.size() || k == 0) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        if (dir == SortDir::Asc) {
            std::partial_sort(seg.Data, seg.Data + k,
                              seg.Data + seg.Count);
        } else {
            std::partial_sort(seg.Data, seg.Data + k,
                              seg.Data + seg.Count,
                              std::greater<T>());
        }
        Regs_.Scalar(rd) = static_cast<u64>(k);
        PC_++;
    }

    // ============================================================
    // JOIN
    // ============================================================

    void HandleJoinHash(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 leftId  = ra;
        u8 rightId = rb;
        u8 resultId = (imm >> 8) & 0xF;
        (void)resultId;
        if (leftId >= Segments_.size() || rightId >= Segments_.size()) { PC_++; return; }
        Segment<T>& left  = Segments_[leftId];
        Segment<T>& right = Segments_[rightId];

        std::unordered_map<T, std::vector<sz>> hash;
        for (sz i = 0; i < right.Count; ++i)
            hash[right.Data[i]].push_back(i);

        sz matched = 0;
        for (sz i = 0; i < left.Count; ++i) {
            auto it = hash.find(left.Data[i]);
            if (it != hash.end()) matched += it->second.size();
        }
        Regs_.Scalar(rd) = static_cast<u64>(matched);
        PC_++;
    }

    void HandleJoinMerge(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 leftId  = ra;
        u8 rightId = rb;
        if (leftId >= Segments_.size() || rightId >= Segments_.size()) { PC_++; return; }
        Segment<T>& left  = Segments_[leftId];
        Segment<T>& right = Segments_[rightId];

        T* lCopy = Arena_.AllocMany<T>(left.Count);
        T* rCopy = Arena_.AllocMany<T>(right.Count);
        std::copy(left.Data, left.Data + left.Count, lCopy);
        std::copy(right.Data, right.Data + right.Count, rCopy);
        std::sort(lCopy, lCopy + left.Count);
        std::sort(rCopy, rCopy + right.Count);

        sz i = 0, j = 0, matches = 0;
        while (i < left.Count && j < right.Count) {
            if (lCopy[i] == rCopy[j]) { matches++; i++; j++; }
            else if (lCopy[i] < rCopy[j]) i++;
            else j++;
        }
        Regs_.Scalar(rd) = static_cast<u64>(matches);
        PC_++;
    }

    void HandleJoinNested(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 leftId  = ra;
        u8 rightId = rb;
        if (leftId >= Segments_.size() || rightId >= Segments_.size()) { PC_++; return; }
        Segment<T>& left  = Segments_[leftId];
        Segment<T>& right = Segments_[rightId];

        sz matches = 0;
        for (sz i = 0; i < left.Count; ++i)
            for (sz j = 0; j < right.Count; ++j)
                if (left.Data[i] == right.Data[j]) matches++;
        Regs_.Scalar(rd) = static_cast<u64>(matches);
        PC_++;
    }

    void HandleJoinAntiSemi(u8 rd, u8 ra, u8 rb, u16 imm, bool anti) {
        u8 leftId  = ra;
        u8 rightId = rb;
        if (leftId >= Segments_.size() || rightId >= Segments_.size()) { PC_++; return; }
        Segment<T>& left  = Segments_[leftId];
        Segment<T>& right = Segments_[rightId];

        std::unordered_map<T, bool> rSet;
        for (sz i = 0; i < right.Count; ++i)
            rSet[right.Data[i]] = true;

        sz result = 0;
        for (sz i = 0; i < left.Count; ++i) {
            bool found = rSet.count(left.Data[i]) > 0;
            if (anti ? !found : found) result++;
        }
        Regs_.Scalar(rd) = static_cast<u64>(result);
        PC_++;
    }

    // ============================================================
    // WINDOW FUNCTIONS
    // ============================================================

    void HandleWindowRow(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 segId   = ra;
        sz rowIdx  = static_cast<sz>(ScalarAsI64(rb));
        sz winSize = static_cast<sz>(imm & 0xFF);
        u8 winFunc = static_cast<u8>((imm >> 8) & 0xF);
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        if (rowIdx >= seg.Count) { Regs_.Scalar(rd) = 0; PC_++; return; }

        sz start = rowIdx >= winSize ? rowIdx - winSize : 0;
        sz end   = std::min(rowIdx + winSize + 1, seg.Count);
        T result{};

        switch (winFunc) {
        case 0: { // ROW_NUMBER
            result = static_cast<T>(rowIdx + 1);
            break;
        }
        case 1: { // SUM
            if constexpr (std::is_floating_point_v<T>) {
                for (sz i = start; i < end; ++i) result += seg.Data[i];
            } else {
                i64 sum = 0;
                for (sz i = start; i < end; ++i) sum += static_cast<i64>(seg.Data[i]);
                result = static_cast<T>(sum);
            }
            break;
        }
        case 2: { // AVG
            if constexpr (std::is_floating_point_v<T>) {
                T s{};
                for (sz i = start; i < end; ++i) s += seg.Data[i];
                result = s / static_cast<T>(end - start);
            } else {
                i64 sum = 0;
                for (sz i = start; i < end; ++i) sum += static_cast<i64>(seg.Data[i]);
                result = static_cast<T>(sum / static_cast<i64>(end - start));
            }
            break;
        }
        case 3: { // MIN
            result = seg.Data[start];
            for (sz i = start + 1; i < end; ++i)
                if (seg.Data[i] < result) result = seg.Data[i];
            break;
        }
        case 4: { // MAX
            result = seg.Data[start];
            for (sz i = start + 1; i < end; ++i)
                if (seg.Data[i] > result) result = seg.Data[i];
            break;
        }
        case 5: { // RANK
            sz rank = 1;
            T val = seg.Data[rowIdx];
            for (sz i = start; i < end; ++i)
                if (seg.Data[i] < val) rank++;
            result = static_cast<T>(rank);
            break;
        }
        default:
            break;
        }

        SetScalarT(rd, result);
        PC_++;
    }

    void HandleWindowRange(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 segId   = ra;
        sz rowIdx  = static_cast<sz>(ScalarAsI64(rb));
        T rangeVal = ScalarAsT(imm & 0xF);
        u8 winFunc = static_cast<u8>((imm >> 8) & 0xF);
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        if (rowIdx >= seg.Count) { Regs_.Scalar(rd) = 0; PC_++; return; }

        T center = seg.Data[rowIdx];
        T result{};
        sz count = 0;

        switch (winFunc) {
        case 0: { // COUNT
            for (sz i = 0; i < seg.Count; ++i) {
                T diff = seg.Data[i] > center
                    ? seg.Data[i] - center : center - seg.Data[i];
                if (!(diff > rangeVal)) count++;
            }
            result = static_cast<T>(count);
            break;
        }
        case 1: { // SUM
            if constexpr (std::is_floating_point_v<T>) {
                for (sz i = 0; i < seg.Count; ++i) {
                    T diff = seg.Data[i] > center
                        ? seg.Data[i] - center : center - seg.Data[i];
                    if (!(diff > rangeVal)) result += seg.Data[i];
                }
            } else {
                i64 sum = 0;
                for (sz i = 0; i < seg.Count; ++i) {
                    T diff = seg.Data[i] > center
                        ? seg.Data[i] - center : center - seg.Data[i];
                    if (!(diff > rangeVal)) sum += static_cast<i64>(seg.Data[i]);
                }
                result = static_cast<T>(sum);
            }
            break;
        }
        case 2: { // AVG
            if constexpr (std::is_floating_point_v<T>) {
                for (sz i = 0; i < seg.Count; ++i) {
                    T diff = seg.Data[i] > center
                        ? seg.Data[i] - center : center - seg.Data[i];
                    if (!(diff > rangeVal)) { result += seg.Data[i]; count++; }
                }
                result = count ? result / static_cast<T>(count) : T{};
            } else {
                i64 sum = 0;
                for (sz i = 0; i < seg.Count; ++i) {
                    T diff = seg.Data[i] > center
                        ? seg.Data[i] - center : center - seg.Data[i];
                    if (!(diff > rangeVal)) { sum += static_cast<i64>(seg.Data[i]); count++; }
                }
                result = count ? static_cast<T>(sum / static_cast<i64>(count)) : T{};
            }
            break;
        }
        default:
            break;
        }

        SetScalarT(rd, result);
        PC_++;
    }

    // ============================================================
    // HASH PARTITION
    // ============================================================

    void HandlePartitionHash(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 segId = ra;
        u8 numPartitions = static_cast<u8>(imm & 0xFF);
        if (numPartitions == 0) numPartitions = 16;
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        Regs_.Scalar(rd) = static_cast<u64>(seg.Count);
        PC_++;
    }

    // ============================================================
    // SERIALIZATION
    // ============================================================

    void HandleSerialize(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 segId = ra;
        if (segId >= Segments_.size()) { PC_++; return; }
        Segment<T>& seg = Segments_[segId];
        sz size = seg.Count * sizeof(T);
        u8* buf = static_cast<u8*>(Arena_.Alloc(size));
        std::memcpy(buf, seg.Data, size);
        sz bufId = AddSegment(reinterpret_cast<T*>(buf), seg.Count);
        Regs_.Scalar(rd) = static_cast<u64>(bufId);
        PC_++;
    }

    void HandleDeserialize(u8 rd, u8 ra, u8 rb, u16 imm) {
        u8 srcId = ra;
        if (srcId >= Segments_.size()) { PC_++; return; }
        Segment<T>& src = Segments_[srcId];
        u8 dstId = (imm >> 8) & 0xF;
        if (dstId >= Segments_.size()) { PC_++; return; }
        Segment<T>& dst = Segments_[dstId];
        sz n = std::min(src.Count, dst.Count);
        std::memcpy(dst.Data, src.Data, n * sizeof(T));
        Regs_.Scalar(rd) = static_cast<u64>(n);
        PC_++;
    }

    // ============================================================
    // HASH ELEMENT HELPER
    // ============================================================

    static u64 HashElement(T value) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(value)) return 0;
            u64 raw = std::bit_cast<u64>(static_cast<f64>(value));
            return HashU64(raw);
        } else if constexpr (sizeof(T) <= 8) {
            u64 raw = static_cast<u64>(value);
            return HashU64(raw);
        } else {
            return 0;
        }
    }

    static u64 HashU64(u64 x) {
        x ^= x >> 33;
        x *= 0xFF51AFD7ED558CCDULL;
        x ^= x >> 33;
        x *= 0xC4CEB9FE1A85EC53ULL;
        x ^= x >> 33;
        return x;
    }

    // ============================================================
    // HASH TABLE STORAGE (per hash-table slot)
    // ============================================================
    std::unordered_map<u64, std::vector<sz>>* HashTables_[16]{};

    void ResetHashTables() {
        for (sz i = 0; i < 16; ++i) {
            if (HashTables_[i]) {
                HashTables_[i]->~unordered_map();
                HashTables_[i] = nullptr;
            }
        }
    }
    // ============================================================
    // VECTOR FILTER HELPER (BRANCHLESS)
    // ============================================================

    template<typename Pred>
    VOXEL_ALWAYS_INLINE
    static void VFilterImpl(Engine<T>* e, u8 vd, u8 va, u8 rb, u16 imm12, Pred pred) {
        T* dst = e->Regs_.VecLanes<T>(vd);
        const T* src = e->Regs_.VecLanes<T>(va);
        T thresh = e->ScalarAsT(rb);
        u32 mask = 0;

        if (e->CurrentNulls_ && IsChunkAllNull(e->CurrentNulls_, e->CurrentSegOff_, e->CurrentSegOff_ + e->kLanes)) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = T{};
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                bool pass = pred(src[i], thresh);
                dst[i] = pass ? src[i] : T{};
                if (pass) mask |= (1u << i);
            }
        }
        e->Regs_.Mask(imm12 & 0x7) = mask;
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — CONTROL (0x00-0x0F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleNOP(Engine<T>* e, u32) { e->PC_++; }

    VOXEL_ALWAYS_INLINE
    static void HandleHALT(Engine<T>* e, u32) { e->Running_ = false; }

    VOXEL_ALWAYS_INLINE
    static void HandleTRAP(Engine<T>* e, u32) {
        e->Regs_.SetFlag(RegFile::Flag::InvalidOp);
        e->Running_ = false;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleBREAK(Engine<T>* e, u32) {
        e->Regs_.SetFlag(RegFile::Flag::InvalidOp);
        e->Running_ = false;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleYIELD(Engine<T>* e, u32) { e->PC_++; }

    VOXEL_ALWAYS_INLINE
    static void HandleBARRIER(Engine<T>* e, u32) { e->PC_++; }

    VOXEL_ALWAYS_INLINE
    static void HandlePREFETCH(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        if (segId < e->Segments_.size()) {
            sz off = static_cast<sz>(e->ScalarAsI64(ra));
            if (off < e->Segments_[segId].Count) {
                VOXEL_PREFETCH(&e->Segments_[segId].Data[off]);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleFLUSH_CACHE(Engine<T>* e, u32) { e->PC_++; }

    VOXEL_ALWAYS_INLINE
    static void HandleSYNC(Engine<T>* e, u32) { e->PC_++; }

    VOXEL_ALWAYS_INLINE
    static void HandleMEMFENCE(Engine<T>* e, u32) { e->PC_++; }

    // ================================================================
    // DISPATCH HANDLERS — SCALAR MOVE / IMMEDIATE (0x10-0x1F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleMOV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); i16 simm12 = Simm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(static_cast<i64>(simm12));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMOVR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleADDI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); i16 simm12 = Simm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) + simm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSUBI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); i16 simm12 = Simm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) - simm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMULI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); i16 simm12 = Simm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) * simm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleANDI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) & static_cast<u64>(imm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleORI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) | static_cast<u64>(imm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleXORI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) ^ static_cast<u64>(imm12);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSHLI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) << (imm12 & 0x3F);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSHRI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) >> (imm12 & 0x3F);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSAR_I(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) >> (imm12 & 0x3F));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMOVZ(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Test(RegFile::Flag::Zero)
            ? e->Regs_.Scalar(ra) : e->Regs_.Scalar(rd);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMOVN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = !e->Regs_.Test(RegFile::Flag::Zero)
            ? e->Regs_.Scalar(ra) : e->Regs_.Scalar(rd);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMOVK(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u16 imm12 = Imm12(raw);
        u8 pos = static_cast<u8>((imm12 >> 8) & 0x3) * 16;
        u64 mask = ~(0xFFFFULL << pos);
        e->Regs_.Scalar(rd) = (e->Regs_.Scalar(rd) & mask)
            | ((static_cast<u64>(imm12 & 0xFFFF) << pos));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleLEA(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); i16 simm12 = Simm12(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) + simm12);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — SCALAR ARITHMETIC INTEGER (0x20-0x29)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleADD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 result = e->ScalarAsI64(ra) + e->ScalarAsI64(rb);
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSUB(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 result = e->ScalarAsI64(ra) - e->ScalarAsI64(rb);
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMUL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 result = e->ScalarAsI64(ra) * e->ScalarAsI64(rb);
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleDIV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 divisor = e->ScalarAsI64(rb);
        if (VOXEL_UNLIKELY(divisor == 0)) {
            e->Regs_.SetFlag(RegFile::Flag::DivByZero);
            e->Running_ = false; return;
        }
        i64 result = e->ScalarAsI64(ra) / divisor;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMOD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 divisor = e->ScalarAsI64(rb);
        if (VOXEL_UNLIKELY(divisor == 0)) {
            e->Regs_.SetFlag(RegFile::Flag::DivByZero);
            e->Running_ = false; return;
        }
        i64 result = e->ScalarAsI64(ra) % divisor;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleNEG(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        i64 result = -e->ScalarAsI64(ra);
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleABS(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        i64 v = e->ScalarAsI64(ra);
        i64 result = v < 0 ? -v : v;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMIN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 a = e->ScalarAsI64(ra), b = e->ScalarAsI64(rb);
        i64 result = a < b ? a : b;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMAX(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 a = e->ScalarAsI64(ra), b = e->ScalarAsI64(rb);
        i64 result = a > b ? a : b;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleAVG(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 a = e->ScalarAsI64(ra), b = e->ScalarAsI64(rb);
        i64 result = (a >> 1) + (b >> 1) + ((a & 1) + (b & 1)) / 2;
        e->Regs_.Scalar(rd) = static_cast<u64>(result);
        e->Regs_.UpdateArithFlags(result);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — SCALAR ARITHMETIC FLOAT (0x2A-0x2F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleADDF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 result = e->ScalarAsF64(ra) + e->ScalarAsF64(rb);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSUBF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 result = e->ScalarAsF64(ra) - e->ScalarAsF64(rb);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleMULF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 result = e->ScalarAsF64(ra) * e->ScalarAsF64(rb);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleDIVF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 divisor = e->ScalarAsF64(rb);
        if (VOXEL_UNLIKELY(divisor == 0.0)) {
            e->Regs_.SetFlag(RegFile::Flag::DivByZero);
            e->Running_ = false; return;
        }
        f64 result = e->ScalarAsF64(ra) / divisor;
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleNEGF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        f64 result = -e->ScalarAsF64(ra);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleABSF(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        f64 result = std::abs(e->ScalarAsF64(ra));
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(result);
        e->Regs_.UpdateFloatFlags(result);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — SCALAR BITWISE (0x30-0x3F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleAND(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) & e->Regs_.Scalar(rb);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleOR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) | e->Regs_.Scalar(rb);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleXOR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) ^ e->Regs_.Scalar(rb);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleNOT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = ~e->Regs_.Scalar(ra);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSHL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) << (e->Regs_.Scalar(rb) & 0x3F);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSHR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) >> (e->Regs_.Scalar(rb) & 0x3F);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSAR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(e->Regs_.Scalar(ra)) >> (e->Regs_.Scalar(rb) & 0x3F));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleROL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 v = e->Regs_.Scalar(ra);
        u8 sh = e->Regs_.Scalar(rb) & 0x3F;
        e->Regs_.Scalar(rd) = (v << sh) | (v >> (64 - sh));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleROR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 v = e->Regs_.Scalar(ra);
        u8 sh = e->Regs_.Scalar(rb) & 0x3F;
        e->Regs_.Scalar(rd) = (v >> sh) | (v << (64 - sh));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandlePOPCNT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            VOXEL_POPCOUNT(e->Regs_.Scalar(ra)));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCLZ(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            VOXEL_CLZ(e->Regs_.Scalar(ra)));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCTZ(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            VOXEL_CTZ(e->Regs_.Scalar(ra)));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleBSWAP(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->ByteSwap64(e->Regs_.Scalar(ra));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleBEXTR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 src = e->Regs_.Scalar(ra);
        u64 ctl = e->Regs_.Scalar(rb);
        u8 start = ctl & 0xFF;
        u8 len   = (ctl >> 8) & 0xFF;
        if (len == 0) { e->Regs_.Scalar(rd) = 0; }
        else {
            u64 mask = (len >= 64) ? ~0ULL : ((1ULL << len) - 1);
            e->Regs_.Scalar(rd) = (src >> start) & mask;
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleBZHI(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 src = e->Regs_.Scalar(ra);
        u8 pos  = e->Regs_.Scalar(rb) & 0x3F;
        if (pos >= 63) { e->Regs_.Scalar(rd) = src; }
        else { e->Regs_.Scalar(rd) = src & ((1ULL << pos) - 1); }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandlePDEP(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 val = e->Regs_.Scalar(ra);
        u64 mask = e->Regs_.Scalar(rb);
        u64 result = 0;
        u64 bit = 1;
        while (mask) {
            u64 lsb = mask & (~mask + 1);
            if (val & 1) result |= lsb;
            val >>= 1;
            mask ^= lsb;
            bit <<= 1;
        }
        e->Regs_.Scalar(rd) = result;
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — COMPARISON (0x40-0x4F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleCMP(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        i64 diff = e->ScalarAsI64(ra) - e->ScalarAsI64(rb);
        e->Regs_.UpdateArithFlags(diff);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCMPF(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 a = e->ScalarAsF64(ra), b = e->ScalarAsF64(rb);
        e->Regs_.ClearAllFlags();
        if (std::isnan(a) || std::isnan(b)) {
            e->Regs_.SetFlag(RegFile::Flag::NaN);
        } else if (a > b) {
            e->Regs_.SetFlag(RegFile::Flag::Sign);
        } else if (a < b) {
        } else {
            e->Regs_.SetFlag(RegFile::Flag::Zero);
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCMPU(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 a = e->Regs_.Scalar(ra), b = e->Regs_.Scalar(rb);
        e->Regs_.ClearAllFlags();
        if (a > b)      e->Regs_.SetFlag(RegFile::Flag::Sign);
        else if (a < b) e->Regs_.ClearFlag(RegFile::Flag::Sign);
        else            e->Regs_.SetFlag(RegFile::Flag::Zero);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleTST(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64 result = e->Regs_.Scalar(ra) & e->Regs_.Scalar(rb);
        e->Regs_.UpdateArithFlags(static_cast<i64>(result));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleTSTF(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        f64 a = e->ScalarAsF64(ra), b = e->ScalarAsF64(rb);
        u64 result = (a == b) ? 0 : ((a < b) ? 1 : 0xFF);
        e->Regs_.UpdateArithFlags(static_cast<i64>(result));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleISNULL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = (e->Regs_.Scalar(ra) == 0) ? 1 : 0;
        e->Regs_.UpdateArithFlags(e->Regs_.Scalar(rd) ? 1 : 0);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleISNOTNULL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = (e->Regs_.Scalar(ra) != 0) ? 1 : 0;
        e->Regs_.UpdateArithFlags(e->Regs_.Scalar(rd) ? 1 : 0);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSELECT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Test(RegFile::Flag::Zero)
            ? e->Regs_.Scalar(rb) : e->Regs_.Scalar(ra);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleSELECTV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        u32 mask = e->Regs_.Mask(imm12 & 0x7);
        u8 lane = imm12 & (e->kLanes - 1);
        e->Regs_.Scalar(rd) = (mask & (1u << lane))
            ? e->Regs_.Scalar(rb) : e->Regs_.Scalar(ra);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — BRANCHING (0x50-0x5F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleJMP(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        e->PC_ += simm12;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJZ(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        e->Regs_.Test(RegFile::Flag::Zero)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJNZ(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        !e->Regs_.Test(RegFile::Flag::Zero)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJS(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        e->Regs_.Test(RegFile::Flag::Sign)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJNS(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        !e->Regs_.Test(RegFile::Flag::Sign)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJO(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        e->Regs_.Test(RegFile::Flag::Overflow)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJNO(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        !e->Regs_.Test(RegFile::Flag::Overflow)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJC(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        e->Regs_.Test(RegFile::Flag::Carry)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJNC(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        !e->Regs_.Test(RegFile::Flag::Carry)
            ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJG(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        bool z = e->Regs_.Test(RegFile::Flag::Zero);
        bool s = e->Regs_.Test(RegFile::Flag::Sign);
        (!z && !s) ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJGE(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        bool z = e->Regs_.Test(RegFile::Flag::Zero);
        bool s = e->Regs_.Test(RegFile::Flag::Sign);
        (z || !s) ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJL(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        bool s = e->Regs_.Test(RegFile::Flag::Sign);
        s ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleJLE(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        bool z = e->Regs_.Test(RegFile::Flag::Zero);
        bool s = e->Regs_.Test(RegFile::Flag::Sign);
        (z || s) ? (e->PC_ += simm12) : e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCALL(Engine<T>* e, u32 raw) {
        i16 simm12 = Simm12(raw);
        if (VOXEL_UNLIKELY(e->CallDepth_ >= e->kMaxCallDepth)) {
            e->Regs_.SetFlag(RegFile::Flag::InvalidOp);
            e->Running_ = false;
            return;
        }
        e->CallStack_[e->CallDepth_++] = e->PC_ + 1;
        e->PC_ += simm12;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleRET(Engine<T>* e, u32) {
        if (VOXEL_UNLIKELY(e->CallDepth_ == 0)) {
            e->Regs_.SetFlag(RegFile::Flag::InvalidOp);
            e->Running_ = false;
            return;
        }
        e->PC_ = e->CallStack_[--e->CallDepth_];
    }

    VOXEL_ALWAYS_INLINE
    static void HandleTABLE_JMP(Engine<T>* e, u32 raw) {
        u8 ra = Ra(raw); u8 rb = Rb(raw);
        u8 baseReg = ra;
        i64 index  = e->ScalarAsI64(rb);
        sz target  = static_cast<sz>(e->Regs_.Scalar(baseReg)) + static_cast<sz>(index);
        if (target < e->Code_.size()) { e->PC_ = target; }
        else { e->Regs_.SetFlag(RegFile::Flag::InvalidOp); e->Running_ = false; return; }
    }

    // ================================================================
    // DISPATCH HANDLERS — TYPE CONVERSION (0x60-0x6F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_I8(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        i64 v = e->ScalarAsI64(ra);
        e->Regs_.Scalar(rd) = static_cast<u64>(static_cast<i8>(v & 0xFF));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_I16(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        i64 v = e->ScalarAsI64(ra);
        e->Regs_.Scalar(rd) = static_cast<u64>(static_cast<i16>(v & 0xFFFF));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_I32(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        i64 v = e->ScalarAsI64(ra);
        e->Regs_.Scalar(rd) = static_cast<u64>(static_cast<i32>(v & 0xFFFFFFFF));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_I64(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(e->ScalarAsI64(ra));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_F32(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        f32 v = static_cast<f32>(e->ScalarAsF64(ra));
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(static_cast<f64>(v));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_F64(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        f64 v = e->ScalarAsF64(ra);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(v);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_U8(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) & 0xFF;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_U16(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) & 0xFFFF;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_U32(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra) & 0xFFFFFFFF;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCVT_U64(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleBITCAST(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = e->Regs_.Scalar(ra);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleREINTERPRET(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.SetScalarT<T>(rd, e->ScalarAsT(ra));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleTRUNC(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = static_cast<u64>(
            static_cast<i64>(std::trunc(e->ScalarAsF64(ra))));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleROUND(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(
            std::round(e->ScalarAsF64(ra)));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleCEIL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(
            std::ceil(e->ScalarAsF64(ra)));
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleFLOOR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        e->Regs_.Scalar(rd) = std::bit_cast<u64>(
            std::floor(e->ScalarAsF64(ra)));
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR I/O (0x70-0x7F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVLOAD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            e->CurrentNulls_  = seg.Nulls;
            e->CurrentSegOff_ = offset;
            for (sz i = 0; i < e->kLanes; ++i) {
                sz idx = offset + i;
                dst[i] = (i < count && idx < seg.Count)
                    ? seg.Data[idx] : T{};
            }
        } else {
            e->CurrentNulls_ = nullptr;
            for (sz i = 0; i < e->kLanes; ++i) dst[i] = T{};
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSTORE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        const T* src = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < count && offset + i < seg.Count; ++i)
                seg.Data[offset + i] = src[i];
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVGATHER(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const i32* indices = reinterpret_cast<const i32*>(
            e->Regs_.VecRaw(ra));
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes && i < count; ++i) {
                sz idx = static_cast<sz>(indices[i]);
                dst[i] = (idx < seg.Count) ? seg.Data[idx] : T{};
            }
        } else {
            for (sz i = 0; i < e->kLanes; ++i) dst[i] = T{};
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSCATTER(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        const T* src = e->Regs_.VecLanes<T>(rd);
        const i32* indices = reinterpret_cast<const i32*>(
            e->Regs_.VecRaw(ra));
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes && i < count; ++i) {
                sz idx = static_cast<sz>(indices[i]);
                if (idx < seg.Count) seg.Data[idx] = src[i];
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVLOAD_STRIDED(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        i64 stride = e->ScalarAsI64(rb);
        T* dst = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes && i < count; ++i) {
                i64 idx = static_cast<i64>(offset) + static_cast<i64>(i) * stride;
                dst[i] = (idx >= 0 && static_cast<sz>(idx) < seg.Count)
                    ? seg.Data[idx] : T{};
            }
        } else { for (sz i = 0; i < e->kLanes; ++i) dst[i] = T{}; }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSTORE_STRIDED(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 count = imm12 & 0xFF;
        if (count == 0) count = static_cast<u8>(e->kLanes);
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        i64 stride = e->ScalarAsI64(rb);
        const T* src = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < count; ++i) {
                i64 idx = static_cast<i64>(offset) + static_cast<i64>(i) * stride;
                if (idx >= 0 && static_cast<sz>(idx) < seg.Count)
                    seg.Data[idx] = src[i];
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVLOAD_MASKED(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId = (imm12 >> 8) & 0xF;
        u8 maskIdx = imm12 & 0x7;
        sz offset  = static_cast<sz>(e->ScalarAsI64(ra));
        u32 mask   = e->Regs_.Mask(maskIdx);
        T* dst = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes; ++i) {
                sz idx = offset + i;
                if (mask & (1u << i)) {
                    dst[i] = (idx < seg.Count) ? seg.Data[idx] : T{};
                }
            }
        } else { e->Regs_.ClearVector(rd); }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSTORE_MASKED(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId   = (imm12 >> 8) & 0xF;
        u8 maskIdx = imm12 & 0x7;
        sz offset  = static_cast<sz>(e->ScalarAsI64(ra));
        u32 mask   = e->Regs_.Mask(maskIdx);
        const T* src = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes; ++i) {
                sz idx = offset + i;
                if ((mask & (1u << i)) && idx < seg.Count)
                    seg.Data[idx] = src[i];
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSPLAT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T val = e->ScalarAsT(ra);
        T* dst = e->Regs_.VecLanes<T>(rd);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = val;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVEXTRACT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 lane = imm12 & (e->kLanes - 1);
        const T* src = e->Regs_.VecLanes<T>(ra);
        e->SetScalarT(rd, src[lane]);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVINSERT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        u8 lane = imm12 & (e->kLanes - 1);
        T val  = e->ScalarAsT(rb);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* src = e->Regs_.VecLanes<T>(ra);
        for (sz i = 0; i < e->kLanes; ++i)
            dst[i] = (i == lane) ? val : src[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVPERMUTE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a   = e->Regs_.VecLanes<T>(ra);
        const u8* idx = reinterpret_cast<const u8*>(
            e->Regs_.VecRaw(rb));
        for (sz i = 0; i < e->kLanes; ++i)
            dst[i] = a[idx[i] % e->kLanes];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSHUFFLE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        u8 mode = imm12 & 0x3;
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        sz half = e->kLanes / 2;
        switch (mode) {
        case 0:
            for (sz i = 0; i < half; ++i) {
                dst[i] = a[i*2]; dst[i+half] = b[i*2];
            }
            break;
        case 1:
            for (sz i = 0; i < half; ++i) {
                dst[i] = a[i*2+1]; dst[i+half] = b[i*2+1];
            }
            break;
        case 2:
            for (sz i = 0; i < half; ++i) {
                dst[i] = a[i]; dst[i+half] = b[i];
            }
            break;
        case 3:
            for (sz i = 0; i < half; ++i) {
                dst[i] = a[i+half]; dst[i+half] = b[i+half];
            }
            break;
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVREVERSE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* src = e->Regs_.VecLanes<T>(ra);
        for (sz i = 0; i < e->kLanes; ++i)
            dst[i] = src[e->kLanes - 1 - i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVROTATE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 shift = imm12 & (e->kLanes - 1);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T temp[e->kLanes];
        for (sz i = 0; i < e->kLanes; ++i)
            temp[(i + shift) % e->kLanes] = src[i];
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = temp[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSLIDE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        sz shift = static_cast<sz>(e->ScalarAsI64(rb));
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* src = e->Regs_.VecLanes<T>(ra);
        for (sz i = 0; i < e->kLanes; ++i) {
            sz si = i + shift;
            dst[i] = (si < e->kLanes) ? src[si] : T{};
        }
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR ARITHMETIC (0x80-0x8F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVADD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] + b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSUB(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] - b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMUL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] * b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVDIV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i) {
                if (VOXEL_UNLIKELY(b[i] == T{}))
                    dst[i] = std::numeric_limits<T>::quiet_NaN();
                else dst[i] = a[i] / b[i];
            }
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                if (VOXEL_UNLIKELY(b[i] == T{})) {
                    e->Regs_.SetFlag(RegFile::Flag::DivByZero);
                    dst[i] = std::numeric_limits<T>::max();
                } else dst[i] = static_cast<T>(
                    static_cast<i64>(a[i]) / static_cast<i64>(b[i]));
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMOD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::fmod(static_cast<f64>(a[i]),
                                   static_cast<f64>(b[i]));
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                if (VOXEL_UNLIKELY(b[i] == T{})) {
                    e->Regs_.SetFlag(RegFile::Flag::DivByZero);
                    dst[i] = T{};
                } else dst[i] = static_cast<T>(
                    static_cast<i64>(a[i]) % static_cast<i64>(b[i]));
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVNEG(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = -a[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVABS(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::abs(a[i]);
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] < T{} ? static_cast<T>(-static_cast<i64>(a[i])) : a[i];
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMIN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = (a[i] < b[i] || std::isnan(a[i])) ? a[i] : b[i];
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] < b[i] ? a[i] : b[i];
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMAX(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = (a[i] > b[i] || std::isnan(b[i])) ? a[i] : b[i];
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] > b[i] ? a[i] : b[i];
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVAVG(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = (a[i] + b[i]) * static_cast<T>(0.5);
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                i64 ai = static_cast<i64>(a[i]);
                i64 bi = static_cast<i64>(b[i]);
                dst[i] = static_cast<T>((ai >> 1) + (bi >> 1) + ((ai & 1) + (bi & 1)) / 2);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFMA(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        T c = e->ScalarAsT(imm12 & 0xF);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::fma(static_cast<f64>(a[i]),
                                  static_cast<f64>(b[i]),
                                  static_cast<f64>(c));
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] * b[i] + c;
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFMS(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        T c = e->ScalarAsT(imm12 & 0xF);
        for (sz i = 0; i < e->kLanes; ++i)
            dst[i] = a[i] * b[i] - c;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSQRT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::sqrt(a[i]);
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                f64 v = static_cast<f64>(a[i]);
                dst[i] = static_cast<T>(v < 0 ? T{} : std::sqrt(v));
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVRSQRT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = static_cast<T>(T{1} / std::sqrt(a[i]));
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                f64 v = static_cast<f64>(a[i]);
                dst[i] = static_cast<T>(v <= 0 ? T{} : T{1} / std::sqrt(v));
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVRCP(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = T{1} / a[i];
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] == T{} ? T{}
                    : static_cast<T>(1.0 / static_cast<f64>(a[i]));
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVPOW(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::pow(a[i], b[i]);
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = e->IntPow(a[i], b[i]);
        }
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR-SCALAR ARITHMETIC (0x90-0x9F)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVSADD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] + s;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSSUB(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] - s;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSMUL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] * s;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSDIV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        if constexpr (std::is_floating_point_v<T>) {
            if (VOXEL_UNLIKELY(s == T{})) {
                e->Regs_.SetFlag(RegFile::Flag::DivByZero);
                for (sz i = 0; i < e->kLanes; ++i)
                    dst[i] = std::numeric_limits<T>::quiet_NaN();
            } else {
                for (sz i = 0; i < e->kLanes; ++i) dst[i] = a[i] / s;
            }
        } else {
            if (VOXEL_UNLIKELY(s == T{})) {
                e->Regs_.SetFlag(RegFile::Flag::DivByZero);
                for (sz i = 0; i < e->kLanes; ++i) dst[i] = T{};
            } else {
                i64 si = static_cast<i64>(s);
                for (sz i = 0; i < e->kLanes; ++i)
                    dst[i] = static_cast<T>(static_cast<i64>(a[i]) / si);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSMOD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::fmod(static_cast<f64>(a[i]),
                                   static_cast<f64>(s));
        } else {
            if (VOXEL_UNLIKELY(s == T{})) {
                e->Regs_.SetFlag(RegFile::Flag::DivByZero);
                for (sz i = 0; i < e->kLanes; ++i) dst[i] = T{};
            } else {
                i64 si = static_cast<i64>(s);
                for (sz i = 0; i < e->kLanes; ++i)
                    dst[i] = static_cast<T>(static_cast<i64>(a[i]) % si);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSMIN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = (a[i] < s || std::isnan(a[i])) ? a[i] : s;
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] < s ? a[i] : s;
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSMAX(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        T s = e->ScalarAsT(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = (a[i] > s || std::isnan(s)) ? a[i] : s;
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = a[i] > s ? a[i] : s;
        }
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — WINDOW STREAMING MATH (0x97-0x99)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleWSTD(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 window = imm12 & 0xFF; u8 segId = (imm12 >> 8) & 0xF;
        if (window < 2 || segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        for (sz j = 0; j < e->kLanes; ++j) {
            sz start = offset + j;
            if (start + window > seg.Count) { dst[j] = T{}; continue; }
            // Two-pass: mean then stddev
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += seg.Data[start + k];
            T mean = sum / static_cast<T>(window);
            T ssq = T{};
            for (sz k = 0; k < window; ++k) {
                T d = seg.Data[start + k] - mean;
                ssq += d * d;
            }
            dst[j] = std::sqrt(ssq / static_cast<T>(window));
        }
        sz advanced = std::min<sz>(e->kLanes, seg.Count > offset ? seg.Count - offset : 0);
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + advanced);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleWVARIANCE_W(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 window = imm12 & 0xFF; u8 segId = (imm12 >> 8) & 0xF;
        if (window < 2 || segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        for (sz j = 0; j < e->kLanes; ++j) {
            sz start = offset + j;
            if (start + window > seg.Count) { dst[j] = T{}; continue; }
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += seg.Data[start + k];
            T mean = sum / static_cast<T>(window);
            T ssq = T{};
            for (sz k = 0; k < window; ++k) {
                T d = seg.Data[start + k] - mean;
                ssq += d * d;
            }
            dst[j] = ssq / static_cast<T>(window);
        }
        sz advanced = std::min<sz>(e->kLanes, seg.Count > offset ? seg.Count - offset : 0);
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + advanced);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleWQUANTILE(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 window = imm12 & 0x1F; if (window == 0) window = 16;
        u8 quantile = (imm12 >> 5) & 0x7; // 0-7 scale, maps to 0%/14%/29%/43%/57%/71%/86%/100%
        u8 segId = (imm12 >> 8) & 0xF;
        if (window < 2 || segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        for (sz j = 0; j < e->kLanes; ++j) {
            sz start = offset + j;
            if (start + window > seg.Count) { dst[j] = T{}; continue; }
            T buf[32]; // max window = 32
            for (sz k = 0; k < window; ++k) buf[k] = seg.Data[start + k];
            sz idx = (quantile * window) / 7; // map 0-7 to 0..window-1
            if (idx >= window) idx = window - 1;
            std::nth_element(buf, buf + idx, buf + window);
            dst[j] = buf[idx];
        }
        sz advanced = std::min<sz>(e->kLanes, seg.Count > offset ? seg.Count - offset : 0);
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + advanced);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR COMPARISON (0xA0-0xAF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPEQ(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpEq{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPNE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpNe{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPLT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpLt{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPLE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpLe{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPGT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpGt{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPGE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        e->HandleVCmp(rd, ra, rb, typename Engine<T>::CmpGe{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPNULL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        u32 mask = 0;
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i) {
                bool isNull = std::isnan(a[i]);
                dst[i] = isNull ? e->bitSplat() : T{};
                if (isNull) mask |= (1u << i);
            }
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                bool isNull = (a[i] == T{});
                dst[i] = isNull ? e->bitSplat() : T{};
                if (isNull) mask |= (1u << i);
            }
        }
        e->Regs_.Mask(imm12 & 0x7) = mask;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCMPNOTNULL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        u32 mask = 0;
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i) {
                bool isNotNull = !std::isnan(a[i]);
                dst[i] = isNotNull ? e->bitSplat() : T{};
                if (isNotNull) mask |= (1u << i);
            }
        } else {
            for (sz i = 0; i < e->kLanes; ++i) {
                bool isNotNull = (a[i] != T{});
                dst[i] = isNotNull ? e->bitSplat() : T{};
                if (isNotNull) mask |= (1u << i);
            }
        }
        e->Regs_.Mask(imm12 & 0x7) = mask;
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR LOGICAL (0xB0-0xBF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVAND(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64* dst = reinterpret_cast<u64*>(e->Regs_.VecLanes<T>(rd));
        const u64* a = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(ra));
        const u64* b = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(rb));
        for (sz i = 0; i < e->kVecWidth / 8; ++i) dst[i] = a[i] & b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVOR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64* dst = reinterpret_cast<u64*>(e->Regs_.VecLanes<T>(rd));
        const u64* a = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(ra));
        const u64* b = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(rb));
        for (sz i = 0; i < e->kVecWidth / 8; ++i) dst[i] = a[i] | b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVXOR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64* dst = reinterpret_cast<u64*>(e->Regs_.VecLanes<T>(rd));
        const u64* a = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(ra));
        const u64* b = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(rb));
        for (sz i = 0; i < e->kVecWidth / 8; ++i) dst[i] = a[i] ^ b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVNOT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        u64* dst = reinterpret_cast<u64*>(e->Regs_.VecLanes<T>(rd));
        const u64* a = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(ra));
        for (sz i = 0; i < e->kVecWidth / 8; ++i) dst[i] = ~a[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVANDN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        u64* dst = reinterpret_cast<u64*>(e->Regs_.VecLanes<T>(rd));
        const u64* a = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(ra));
        const u64* b = reinterpret_cast<const u64*>(e->Regs_.VecLanes<T>(rb));
        for (sz i = 0; i < e->kVecWidth / 8; ++i) dst[i] = (~a[i]) & b[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSHL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::ldexp(a[i], static_cast<int>(b[i]));
        } else {
            u8 bitWidth = static_cast<u8>(sizeof(T) * 8);
            for (sz i = 0; i < e->kLanes; ++i) {
                using UT = typename TypeTraits<T>::UnsignedType;
                u8 sh = static_cast<u8>(static_cast<u64>(b[i]) & (bitWidth - 1));
                dst[i] = static_cast<T>(static_cast<UT>(a[i]) << sh);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSHR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::ldexp(a[i], -static_cast<int>(b[i]));
        } else {
            u8 bitWidth = static_cast<u8>(sizeof(T) * 8);
            for (sz i = 0; i < e->kLanes; ++i) {
                using UT = typename TypeTraits<T>::UnsignedType;
                u8 sh = static_cast<u8>(static_cast<u64>(b[i]) & (bitWidth - 1));
                dst[i] = static_cast<T>(static_cast<UT>(a[i]) >> sh);
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSAR(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                dst[i] = std::ldexp(a[i], -static_cast<int>(b[i]));
        } else {
            u8 bitWidth = static_cast<u8>(sizeof(T) * 8);
            for (sz i = 0; i < e->kLanes; ++i) {
                using ST = typename TypeTraits<T>::SignedType;
                u8 sh = static_cast<u8>(static_cast<u64>(b[i]) & (bitWidth - 1));
                dst[i] = static_cast<T>(static_cast<ST>(a[i]) >> sh);
            }
        }
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR FILTER (0xC0-0xCF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T thresh = e->ScalarAsT(rb);
        u8 mode = imm12 & 0x7;
        u32 mask = 0;
        for (sz i = 0; i < e->kLanes; ++i) {
            bool pass;
            switch (mode) {
            case 0: pass = src[i] == thresh; break;
            case 1: pass = src[i] != thresh; break;
            case 2: pass = src[i] <  thresh; break;
            case 3: pass = src[i] <= thresh; break;
            case 4: pass = src[i] >  thresh; break;
            case 5: pass = src[i] >= thresh; break;
            default: pass = false; break;
            }
            dst[i] = pass ? src[i] : T{};
            if (pass) mask |= (1u << i);
        }
        e->Regs_.Mask(imm12 & 0x7) = mask;
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_EQ(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::equal_to<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_NE(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::not_equal_to<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_LT(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::less<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_LE(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::less_equal<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_GT(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::greater<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFILTER_GE(Engine<T>* e, u32 raw) {
        VFilterImpl(e, Rd(raw), Ra(raw), Rb(raw), Imm12(raw), std::greater_equal<>{});
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVBLEND(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw); u16 imm12 = Imm12(raw);
        T* dst = e->Regs_.VecLanes<T>(rd);
        const T* a = e->Regs_.VecLanes<T>(ra);
        const T* b = e->Regs_.VecLanes<T>(rb);
        u32 mask = e->Regs_.Mask(imm12 & 0x7);
        for (sz i = 0; i < e->kLanes; ++i)
            dst[i] = (mask & (1u << i)) ? b[i] : a[i];
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMASK_STORE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId   = (imm12 >> 8) & 0xF;
        u8 maskIdx = imm12 & 0x7;
        sz offset  = static_cast<sz>(e->ScalarAsI64(ra));
        u32 mask   = e->Regs_.Mask(maskIdx);
        const T* src = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes; ++i) {
                sz idx = offset + i;
                if ((mask & (1u << i)) && idx < seg.Count)
                    seg.Data[idx] = src[i];
            }
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMASK_LOAD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 segId   = (imm12 >> 8) & 0xF;
        u8 maskIdx = imm12 & 0x7;
        sz offset  = static_cast<sz>(e->ScalarAsI64(ra));
        u32 mask   = e->Regs_.Mask(maskIdx);
        T* dst = e->Regs_.VecLanes<T>(rd);
        if (VOXEL_LIKELY(segId < e->Segments_.size())) {
            Segment<T>& seg = e->Segments_[segId];
            for (sz i = 0; i < e->kLanes; ++i) {
                sz idx = offset + i;
                dst[i] = ((mask & (1u << i)) && idx < seg.Count)
                    ? seg.Data[idx] : dst[i];
            }
        }
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — VECTOR REDUCTION (0xD0-0xDF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleVSUM(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T sum = T{};
        for (sz i = 0; i < e->kLanes; ++i) sum += src[i];
        e->SetScalarT(rd, sum);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVPROD(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T prod = T{1};
        for (sz i = 0; i < e->kLanes; ++i) prod *= src[i];
        e->SetScalarT(rd, prod);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVMEAN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            T sum = T{};
            for (sz i = 0; i < e->kLanes; ++i) sum += src[i];
            e->SetScalarT(rd, sum / static_cast<T>(e->kLanes));
        } else {
            i64 sum = 0;
            for (sz i = 0; i < e->kLanes; ++i)
                sum += static_cast<i64>(src[i]);
            e->SetScalarT(rd, static_cast<T>(sum / static_cast<i64>(e->kLanes)));
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVSTDDEV(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            T sum = T{}, sumSq = T{};
            for (sz i = 0; i < e->kLanes; ++i) {
                sum   += src[i];
                sumSq += src[i] * src[i];
            }
            T mean   = sum / static_cast<T>(e->kLanes);
            T var    = (sumSq / static_cast<T>(e->kLanes)) - (mean * mean);
            e->SetScalarT(rd, std::sqrt(std::max(T{}, var)));
        } else {
            f64 sum = 0, sumSq = 0;
            for (sz i = 0; i < e->kLanes; ++i) {
                f64 v = static_cast<f64>(src[i]);
                sum   += v;
                sumSq += v * v;
            }
            f64 mean = sum / static_cast<f64>(e->kLanes);
            f64 var  = (sumSq / static_cast<f64>(e->kLanes)) - (mean * mean);
            e->SetScalarT(rd, static_cast<T>(std::sqrt(std::max(0.0, var))));
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVVARIANCE(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        if constexpr (std::is_floating_point_v<T>) {
            T sum = T{}, sumSq = T{};
            for (sz i = 0; i < e->kLanes; ++i) {
                sum   += src[i];
                sumSq += src[i] * src[i];
            }
            T mean = sum / static_cast<T>(e->kLanes);
            T var  = (sumSq / static_cast<T>(e->kLanes)) - (mean * mean);
            e->SetScalarT(rd, std::max(T{}, var));
        } else {
            f64 sum = 0, sumSq = 0;
            for (sz i = 0; i < e->kLanes; ++i) {
                f64 v = static_cast<f64>(src[i]);
                sum   += v;
                sumSq += v * v;
            }
            f64 mean = sum / static_cast<f64>(e->kLanes);
            f64 var  = (sumSq / static_cast<f64>(e->kLanes)) - (mean * mean);
            e->SetScalarT(rd, static_cast<T>(std::max(0.0, var)));
        }
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVRED_MIN(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T m = src[0];
        for (sz i = 1; i < e->kLanes; ++i)
            if (src[i] < m ||
                (std::is_floating_point_v<T> && std::isnan(src[i])))
                m = src[i];
        e->SetScalarT(rd, m);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVRED_MAX(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        T m = src[0];
        for (sz i = 1; i < e->kLanes; ++i)
            if (src[i] > m ||
                (std::is_floating_point_v<T> && std::isnan(m)))
                m = src[i];
        e->SetScalarT(rd, m);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVCOUNT(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        sz cnt = 0;
        if constexpr (std::is_floating_point_v<T>) {
            for (sz i = 0; i < e->kLanes; ++i)
                if (!std::isnan(src[i]) && src[i] != T{}) cnt++;
        } else {
            for (sz i = 0; i < e->kLanes; ++i)
                if (src[i] != T{}) cnt++;
        }
        e->Regs_.Scalar(rd) = static_cast<u64>(cnt);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVANY(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        bool any = false;
        for (sz i = 0; i < e->kLanes; ++i)
            if (src[i] != T{}) { any = true; break; }
        e->Regs_.Scalar(rd) = any ? 1 : 0;
        e->Regs_.UpdateArithFlags(any ? 1 : 0);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVALL(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        bool all = true;
        for (sz i = 0; i < e->kLanes; ++i)
            if (src[i] == T{}) { all = false; break; }
        e->Regs_.Scalar(rd) = all ? 1 : 0;
        e->Regs_.UpdateArithFlags(all ? 1 : 0);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVFIRST(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        e->SetScalarT(rd, src[0]);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVLAST(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        e->SetScalarT(rd, src[e->kLanes > 0 ? e->kLanes - 1 : 0]);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleVNTH(Engine<T>* e, u32 raw) {
        u8 rd = Rd(raw); u8 ra = Ra(raw); u8 rb = Rb(raw);
        const T* src = e->Regs_.VecLanes<T>(ra);
        sz idx = static_cast<sz>(e->ScalarAsI64(rb));
        if (idx < e->kLanes) e->SetScalarT(rd, src[idx]);
        else e->SetScalarT(rd, T{});
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — WINDOW-STREAMING REDUCTION (0xDD-0xDF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleWDELTA(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u8 segId = (Imm12(raw) >> 8) & 0xF;
        (void)Rb(raw); // carryReg: reserved for cross-chunk carry
        if (segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        sz count = e->kLanes;
        if (offset + count >= seg.Count) count = seg.Count > offset + 1 ? seg.Count - offset - 1 : 0;
        for (sz i = 0; i < count; ++i) {
            sz idx = offset + i;
            dst[i] = seg.Data[idx + 1] - seg.Data[idx];
        }
        for (sz i = count; i < e->kLanes; ++i) dst[i] = T{};
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + count);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleWINDOW_SUM(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 window = imm12 & 0xFF; u8 segId = (imm12 >> 8) & 0xF;
        if (window == 0 || segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        for (sz j = 0; j < e->kLanes; ++j) {
            sz start = offset + j;
            if (start + window > seg.Count) { dst[j] = T{}; continue; }
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += seg.Data[start + k];
            dst[j] = sum;
        }
        sz advanced = std::min<sz>(e->kLanes, seg.Count > offset ? seg.Count - offset : 0);
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + advanced);
        e->PC_++;
    }

    VOXEL_ALWAYS_INLINE
    static void HandleWINDOW_MEAN(Engine<T>* e, u32 raw) {
        u8 vd = Rd(raw); u8 ra = Ra(raw); u16 imm12 = Imm12(raw);
        u8 window = imm12 & 0xFF; u8 segId = (imm12 >> 8) & 0xF;
        if (window == 0 || segId >= e->Segments_.size()) { e->PC_++; return; }
        Segment<T>& seg = e->Segments_[segId];
        sz offset = static_cast<sz>(e->ScalarAsI64(ra));
        T* dst = e->Regs_.VecLanes<T>(vd);
        T winv = T(1) / static_cast<T>(window);
        for (sz j = 0; j < e->kLanes; ++j) {
            sz start = offset + j;
            if (start + window > seg.Count) { dst[j] = T{}; continue; }
            T sum = T{};
            for (sz k = 0; k < window; ++k) sum += seg.Data[start + k];
            dst[j] = sum * winv;
        }
        sz advanced = std::min<sz>(e->kLanes, seg.Count > offset ? seg.Count - offset : 0);
        e->Regs_.Scalar(ra) = static_cast<u64>(offset + advanced);
        e->PC_++;
    }

    // ================================================================
    // DISPATCH HANDLERS — AGGREGATE (0xE0-0xEF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleAGG_COUNT(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Count);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_SUM(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Sum);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_AVG(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Avg);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_MIN(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Min);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_MAX(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Max);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_FIRST(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::First);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_LAST(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Last);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_STDDEV(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::StdDev);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_VARIANCE(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Variance);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_COUNT_DISTINCT(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::CountDistinct);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_SUM_DISTINCT(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::SumDistinct);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_MEDIAN(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Median);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_MODE(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Mode);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleAGG_PERCENTILE(Engine<T>* e, u32 raw) {
        e->HandleAggregate(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::AggOp::Percentile);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleHASH_INIT(Engine<T>* e, u32 raw) {
        e->HandleHashInit(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleHASH_PROBE(Engine<T>* e, u32 raw) {
        e->HandleHashProbe(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }

    // ================================================================
    // DISPATCH HANDLERS — HASH / SORT / JOIN (0xF0-0xFF)
    // ================================================================

    VOXEL_ALWAYS_INLINE
    static void HandleHASH_BUILD(Engine<T>* e, u32 raw) {
        e->HandleHashBuild(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleHASH_LOOKUP(Engine<T>* e, u32 raw) {
        e->HandleHashLookup(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleSORT_ASC(Engine<T>* e, u32 raw) {
        e->HandleSort(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::SortDir::Asc);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleSORT_DESC(Engine<T>* e, u32 raw) {
        e->HandleSort(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::SortDir::Desc);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleSORT_TOPK(Engine<T>* e, u32 raw) {
        e->HandleSortTopK(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::SortDir::Desc);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleSORT_BOTTOMK(Engine<T>* e, u32 raw) {
        e->HandleSortTopK(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), Engine<T>::SortDir::Asc);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleJOIN_HASH(Engine<T>* e, u32 raw) {
        e->HandleJoinHash(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleJOIN_MERGE(Engine<T>* e, u32 raw) {
        e->HandleJoinMerge(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleJOIN_NESTED(Engine<T>* e, u32 raw) {
        e->HandleJoinNested(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleJOIN_ANTI(Engine<T>* e, u32 raw) {
        e->HandleJoinAntiSemi(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), true);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleJOIN_SEMI(Engine<T>* e, u32 raw) {
        e->HandleJoinAntiSemi(Rd(raw), Ra(raw), Rb(raw), Imm12(raw), false);
    }
    VOXEL_ALWAYS_INLINE
    static void HandleWINDOW_ROW(Engine<T>* e, u32 raw) {
        e->HandleWindowRow(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleWINDOW_RANGE(Engine<T>* e, u32 raw) {
        e->HandleWindowRange(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandlePARTITION_HASH(Engine<T>* e, u32 raw) {
        e->HandlePartitionHash(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleSERIALIZE(Engine<T>* e, u32 raw) {
        e->HandleSerialize(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }
    VOXEL_ALWAYS_INLINE
    static void HandleDESERIALIZE(Engine<T>* e, u32 raw) {
        e->HandleDeserialize(Rd(raw), Ra(raw), Rb(raw), Imm12(raw));
    }

    VOXEL_ALWAYS_INLINE
    static void HandleINVALID(Engine<T>* e, u32) {
        e->Regs_.SetFlag(RegFile::Flag::InvalidOp);
        e->Running_ = false;
    }

    static inline constexpr DispatchFn sDispatch[256] = {
        /* 0x00-0x0F: Control */
        &HandleNOP, &HandleHALT, &HandleTRAP, &HandleBREAK,
        &HandleYIELD, &HandleBARRIER, &HandlePREFETCH, &HandleFLUSH_CACHE,
        &HandleSYNC, &HandleMEMFENCE, &HandleINVALID, &HandleINVALID,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0x10-0x1F: Scalar Move */
        &HandleMOV, &HandleMOVR, &HandleADDI, &HandleSUBI,
        &HandleMULI, &HandleANDI, &HandleORI, &HandleXORI,
        &HandleSHLI, &HandleSHRI, &HandleSAR_I, &HandleMOVZ,
        &HandleMOVN, &HandleMOVK, &HandleLEA, &HandleINVALID,
        /* 0x20-0x2F: Scalar Arithmetic */
        &HandleADD, &HandleSUB, &HandleMUL, &HandleDIV,
        &HandleMOD, &HandleNEG, &HandleABS, &HandleMIN,
        &HandleMAX, &HandleAVG, &HandleADDF, &HandleSUBF,
        &HandleMULF, &HandleDIVF, &HandleNEGF, &HandleABSF,
        /* 0x30-0x3F: Scalar Bitwise */
        &HandleAND, &HandleOR, &HandleXOR, &HandleNOT,
        &HandleSHL, &HandleSHR, &HandleSAR, &HandleROL,
        &HandleROR, &HandlePOPCNT, &HandleCLZ, &HandleCTZ,
        &HandleBSWAP, &HandleBEXTR, &HandleBZHI, &HandlePDEP,
        /* 0x40-0x4F: Comparison */
        &HandleCMP, &HandleCMPF, &HandleCMPU, &HandleTST,
        &HandleTSTF, &HandleISNULL, &HandleISNOTNULL, &HandleSELECT,
        &HandleSELECTV, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0x50-0x5F: Branching */
        &HandleJMP, &HandleJZ, &HandleJNZ, &HandleJS,
        &HandleJNS, &HandleJO, &HandleJNO, &HandleJC,
        &HandleJNC, &HandleJL, &HandleJLE, &HandleJG,
        &HandleJGE, &HandleCALL, &HandleRET, &HandleTABLE_JMP,
        /* 0x60-0x6F: Type Conversion */
        &HandleCVT_I8, &HandleCVT_I16, &HandleCVT_I32, &HandleCVT_I64,
        &HandleCVT_F32, &HandleCVT_F64, &HandleCVT_U8, &HandleCVT_U16,
        &HandleCVT_U32, &HandleCVT_U64, &HandleBITCAST, &HandleREINTERPRET,
        &HandleTRUNC, &HandleROUND, &HandleCEIL, &HandleFLOOR,
        /* 0x70-0x7F: Vector I/O */
        &HandleVLOAD, &HandleVSTORE, &HandleVGATHER, &HandleVSCATTER,
        &HandleVLOAD_STRIDED, &HandleVSTORE_STRIDED, &HandleVLOAD_MASKED, &HandleVSTORE_MASKED,
        &HandleVSPLAT, &HandleVEXTRACT, &HandleVINSERT, &HandleVPERMUTE,
        &HandleVSHUFFLE, &HandleVREVERSE, &HandleVROTATE, &HandleVSLIDE,
        /* 0x80-0x8F: Vector Arithmetic */
        &HandleVADD, &HandleVSUB, &HandleVMUL, &HandleVDIV,
        &HandleVMOD, &HandleVNEG, &HandleVABS, &HandleVMIN,
        &HandleVMAX, &HandleVAVG, &HandleVFMA, &HandleVFMS,
        &HandleVSQRT, &HandleVRSQRT, &HandleVRCP, &HandleVPOW,
        /* 0x90-0x9F: Vector-Scalar Arithmetic */
        &HandleVSADD, &HandleVSSUB, &HandleVSMUL, &HandleVSDIV,
         &HandleVSMOD, &HandleVSMIN, &HandleVSMAX, &HandleWSTD,
         &HandleWVARIANCE_W, &HandleWQUANTILE, &HandleINVALID, &HandleINVALID,
         &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0xA0-0xAF: Vector Comparison */
        &HandleVCMPEQ, &HandleVCMPNE, &HandleVCMPLT, &HandleVCMPLE,
        &HandleVCMPGT, &HandleVCMPGE, &HandleVCMPNULL, &HandleVCMPNOTNULL,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0xB0-0xBF: Vector Logical */
        &HandleVAND, &HandleVOR, &HandleVXOR, &HandleVNOT,
        &HandleVANDN, &HandleVSHL, &HandleVSHR, &HandleVSAR,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0xC0-0xCF: Vector Filter */
        &HandleVFILTER, &HandleVFILTER_EQ, &HandleVFILTER_NE, &HandleVFILTER_LT,
        &HandleVFILTER_LE, &HandleVFILTER_GT, &HandleVFILTER_GE, &HandleVBLEND,
        &HandleVMASK_STORE, &HandleVMASK_LOAD, &HandleINVALID, &HandleINVALID,
        &HandleINVALID, &HandleINVALID, &HandleINVALID, &HandleINVALID,
        /* 0xD0-0xDF: Vector Reduction */
        &HandleVSUM, &HandleVPROD, &HandleVMEAN, &HandleVSTDDEV,
        &HandleVVARIANCE, &HandleVRED_MIN, &HandleVRED_MAX, &HandleVCOUNT,
        &HandleVANY, &HandleVALL, &HandleVFIRST, &HandleVLAST,
         &HandleVNTH, &HandleWDELTA, &HandleWINDOW_SUM, &HandleWINDOW_MEAN,
        /* 0xE0-0xEF: Aggregate */
        &HandleAGG_COUNT, &HandleAGG_SUM, &HandleAGG_AVG, &HandleAGG_MIN,
        &HandleAGG_MAX, &HandleAGG_FIRST, &HandleAGG_LAST, &HandleAGG_STDDEV,
        &HandleAGG_VARIANCE, &HandleAGG_COUNT_DISTINCT, &HandleAGG_SUM_DISTINCT, &HandleAGG_MEDIAN,
        &HandleAGG_MODE, &HandleAGG_PERCENTILE, &HandleHASH_INIT, &HandleHASH_PROBE,
        /* 0xF0-0xFF: Hash / Sort / Join */
        &HandleHASH_BUILD, &HandleHASH_LOOKUP, &HandleSORT_ASC, &HandleSORT_DESC,
        &HandleSORT_TOPK, &HandleSORT_BOTTOMK, &HandleJOIN_HASH, &HandleJOIN_MERGE,
        &HandleJOIN_NESTED, &HandleJOIN_ANTI, &HandleJOIN_SEMI, &HandleWINDOW_ROW,
        &HandleWINDOW_RANGE, &HandlePARTITION_HASH, &HandleSERIALIZE, &HandleDESERIALIZE,
    };

};

} // namespace voxel
