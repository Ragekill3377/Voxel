#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/core/arena.hpp"
#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <type_traits>
#include <cassert>
#include <string>
#include <map>
#include <bitset>
#include <unordered_set>

namespace voxel {
namespace encoding {

// ============================================================================
// EncodingKind — identifies the compression strategy
// ============================================================================

enum class EncodingKind : u8 {
    Plain           = 0,
    Dictionary      = 1,
    RLE             = 2,
    Delta           = 3,
    BitPacked       = 4,
    FSST            = 5,
    FrameOfReference = 6,
};

inline constexpr const char* EncodingKindName(EncodingKind k) {
    switch (k) {
    case EncodingKind::Plain:            return "Plain";
    case EncodingKind::Dictionary:       return "Dictionary";
    case EncodingKind::RLE:              return "RLE";
    case EncodingKind::Delta:            return "Delta";
    case EncodingKind::BitPacked:        return "BitPacked";
    case EncodingKind::FSST:             return "FSST";
    case EncodingKind::FrameOfReference: return "FrameOfReference";
    default: return "Unknown";
    }
}

// ============================================================================
// EncodingStats — per-encoding compression statistics
// ============================================================================

struct EncodingStats {
    sz RawBytes     = 0;
    sz EncodedBytes = 0;
    sz ElementCount = 0;
    f64 CompressionRatio = 1.0;
    EncodingKind Kind    = EncodingKind::Plain;

    void Compute(sz rawSz, sz encodedSz, sz count) {
        RawBytes     = rawSz;
        EncodedBytes = encodedSz;
        ElementCount = count;
        CompressionRatio = (encodedSz > 0) ? static_cast<f64>(rawSz) / static_cast<f64>(encodedSz) : 1.0;
    }

    void Print(std::ostream& os) const {
        os << EncodingKindName(Kind) << ": " << RawBytes << "B -> "
           << EncodedBytes << "B (" << (CompressionRatio * 100.0) << "%) "
           << ElementCount << " elements";
    }
};

// ============================================================================
// IEncoding — type-erased base for all encodings
// ============================================================================

class IEncoding {
public:
    virtual ~IEncoding() = default;
    virtual EncodingKind Kind() const = 0;
    virtual void DecodeBatch(void* out, sz offset, sz count) const = 0;
    virtual sz   MemoryUsage() const = 0;
    virtual void Clear() = 0;
    virtual sz   RowCount() const = 0;
    virtual DataType ElementType() const = 0;
    virtual EncodingStats GetStats() const {
        EncodingStats s;
        s.Kind     = Kind();
        s.EncodedBytes = MemoryUsage();
        s.ElementCount = RowCount();
        s.RawBytes = s.ElementCount * TypeSize(ElementType());
        s.Compute(s.RawBytes, s.EncodedBytes, s.ElementCount);
        return s;
    }
};

// ============================================================================
// DictionaryEncoding — sorts unique values, stores indices
// ============================================================================

template<typename T>
class DictionaryEncoding final : public IEncoding {
public:
    std::vector<T>   Dictionary;
    std::vector<u32> Indices;

    EncodingKind Kind() const override { return EncodingKind::Dictionary; }
    DataType ElementType() const override { return TypeTraits<T>::kTypeId; }
    sz RowCount() const override { return Indices.size(); }

    void Build(const T* data, sz count) {
        Clear();

        if (count == 0) return;

        std::unordered_map<T, u32> valueToIndex;
        for (sz i = 0; i < count; ++i) {
            auto it = valueToIndex.find(data[i]);
            if (it == valueToIndex.end()) {
                u32 nextIdx = static_cast<u32>(Dictionary.size());
                Dictionary.push_back(data[i]);
                valueToIndex[data[i]] = nextIdx;
                Indices.push_back(nextIdx);
            } else {
                Indices.push_back(it->second);
            }
        }

        if (!Dictionary.empty()) {
            std::vector<std::pair<T, u32>> sortedPairs;
            sortedPairs.reserve(Dictionary.size());
            for (sz i = 0; i < Dictionary.size(); ++i) {
                sortedPairs.emplace_back(Dictionary[i], static_cast<u32>(i));
            }
            std::sort(sortedPairs.begin(), sortedPairs.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            std::vector<u32> oldToNew(Dictionary.size());
            for (u32 i = 0; i < static_cast<u32>(sortedPairs.size()); ++i) {
                Dictionary[i] = sortedPairs[i].first;
                oldToNew[sortedPairs[i].second] = i;
            }

            for (u32& idx : Indices) {
                idx = oldToNew[idx];
            }
        }
    }

    T Decode(sz row) const {
        return Dictionary[Indices[row]];
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        T* dst = static_cast<T*>(out);
        for (sz i = 0; i < count && (offset + i) < Indices.size(); ++i) {
            dst[i] = Dictionary[Indices[offset + i]];
        }
    }

    void DecodeBatch(T* out, sz offset, sz count) const {
        for (sz i = 0; i < count && (offset + i) < Indices.size(); ++i) {
            out[i] = Dictionary[Indices[offset + i]];
        }
    }

    sz MemoryUsage() const override {
        return Dictionary.size() * sizeof(T) + Indices.size() * sizeof(u32);
    }

    sz Cardinality() const { return Dictionary.size(); }

    void Clear() override {
        Dictionary.clear();
        Dictionary.shrink_to_fit();
        Indices.clear();
        Indices.shrink_to_fit();
    }
};

// ============================================================================
// RLEEncoding — run-length encoding for repeated values
// ============================================================================

template<typename T>
class RLEEncoding final : public IEncoding {
public:
    std::vector<T>   Values;
    std::vector<u32> RunLengths;

    EncodingKind Kind() const override { return EncodingKind::RLE; }
    DataType ElementType() const override { return TypeTraits<T>::kTypeId; }

    sz RowCount() const override {
        sz total = 0;
        for (u32 rl : RunLengths) total += rl;
        return total;
    }

    void Build(const T* data, sz count) {
        Clear();
        if (count == 0) return;

        T current    = data[0];
        u32 runLen   = 1;
        for (sz i = 1; i < count; ++i) {
            if (data[i] == current) {
                ++runLen;
            } else {
                Values.push_back(current);
                RunLengths.push_back(runLen);
                current = data[i];
                runLen  = 1;
            }
        }
        Values.push_back(current);
        RunLengths.push_back(runLen);
    }

    T Decode(sz row) const {
        u32 cumulative = 0;
        for (sz i = 0; i < RunLengths.size(); ++i) {
            cumulative += RunLengths[i];
            if (row < static_cast<sz>(cumulative)) return Values[i];
        }
        return T{};
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        T* dst = static_cast<T*>(out);

        u32 cum = 0;
        sz runIdx = 0;

        while (runIdx < RunLengths.size()) {
            cum += RunLengths[runIdx];
            if (static_cast<sz>(offset) < cum) break;
            ++runIdx;
        }

        sz row = offset;
        for (sz i = 0; i < count; ++i) {
            if (runIdx >= RunLengths.size()) {
                dst[i] = T{};
                continue;
            }
            dst[i] = Values[runIdx];
            ++row;
            if (static_cast<sz>(row) >= cum) {
                ++runIdx;
                if (runIdx < RunLengths.size())
                    cum += RunLengths[runIdx];
            }
        }
    }

    void DecodeBatch(T* out, sz offset, sz count) const {
        DecodeBatch(static_cast<void*>(out), offset, count);
    }

    sz MemoryUsage() const override {
        return Values.size() * sizeof(T) + RunLengths.size() * sizeof(u32);
    }

    sz RunCount() const { return Values.size(); }

    void Clear() override {
        Values.clear();
        Values.shrink_to_fit();
        RunLengths.clear();
        RunLengths.shrink_to_fit();
    }
};

// ============================================================================
// DeltaEncoding — stores successive differences
// ============================================================================

template<typename T>
    requires (std::is_integral_v<T>)
class DeltaEncoding final : public IEncoding {
public:
    T              BaseValue = T{};
    std::vector<T> Deltas;

    EncodingKind Kind() const override { return EncodingKind::Delta; }
    DataType ElementType() const override { return TypeTraits<T>::kTypeId; }
    sz RowCount() const override { return Deltas.size(); }

    void Build(const T* data, sz count) {
        Clear();
        if (count == 0) return;

        BaseValue = T{};
        Deltas.reserve(count);

        T prev = T{};
        for (sz i = 0; i < count; ++i) {
            T delta = data[i] - prev;
            Deltas.push_back(delta);
            prev = data[i];
        }
    }

    void Decode(T* out, sz count) const {
        T running = BaseValue;
        for (sz i = 0; i < count && i < Deltas.size(); ++i) {
            running += Deltas[i];
            out[i] = running;
        }
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        T* dst   = static_cast<T*>(out);
        T running = BaseValue;

        for (sz i = 0; i < offset && i < Deltas.size(); ++i) {
            running += Deltas[i];
        }

        for (sz i = 0; i < count && (offset + i) < Deltas.size(); ++i) {
            running += Deltas[offset + i];
            dst[i] = running;
        }
    }

    sz MemoryUsage() const override {
        return sizeof(T) + Deltas.size() * sizeof(T);
    }

    T MaxDelta() const {
        if (Deltas.empty()) return T{};
        T maxDV = Deltas[0];
        for (sz i = 1; i < Deltas.size(); ++i)
            if (Deltas[i] > maxDV) maxDV = Deltas[i];
        return maxDV;
    }

    T MinDelta() const {
        if (Deltas.empty()) return T{};
        T minDV = Deltas[0];
        for (sz i = 1; i < Deltas.size(); ++i)
            if (Deltas[i] < minDV) minDV = Deltas[i];
        return minDV;
    }

    void Clear() override {
        BaseValue = T{};
        Deltas.clear();
        Deltas.shrink_to_fit();
    }
};

// ============================================================================
// BitPackedEncoding — packs integers into minimal-width bit fields
// ============================================================================

class BitPackedEncoding final : public IEncoding {
public:
    u32              BitWidth   = 0;
    std::vector<u64> PackedData;
    sz               ValueCount = 0;

    EncodingKind Kind() const override { return EncodingKind::BitPacked; }
    DataType ElementType() const override { return DataType::Uint32; }
    sz RowCount() const override { return ValueCount; }

    void Build(const u32* data, sz count) {
        Clear();
        if (count == 0) return;

        u32 maxVal = 0;
        for (sz i = 0; i < count; ++i) {
            if (data[i] > maxVal) maxVal = data[i];
        }

        BitWidth = 0;
        u32 temp = maxVal;
        while (temp > 0) {
            ++BitWidth;
            temp >>= 1;
        }
        if (BitWidth == 0) BitWidth = 1;

        sz totalBits = count * BitWidth;
        sz numWords  = (totalBits + 63) / 64;
        PackedData.assign(numWords, 0);
        ValueCount = count;

        for (sz i = 0; i < count; ++i) {
            Pack(data[i], i);
        }
    }

    u32 Decode(sz row) const {
        u64 bitPos = row * BitWidth;
        u64 wordLo = PackedData[bitPos / 64];
        u32 shift  = bitPos % 64;
        u64 mask   = (1ULL << BitWidth) - 1;

        u64 result = wordLo >> shift;
        if (shift + BitWidth > 64 && (bitPos / 64 + 1) < PackedData.size()) {
            u64 wordHi = PackedData[bitPos / 64 + 1];
            result |= wordHi << (64 - shift);
        }
        return static_cast<u32>(result & mask);
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        u32* dst = static_cast<u32*>(out);
        for (sz i = 0; i < count && (offset + i) < ValueCount; ++i) {
            dst[i] = Decode(offset + i);
        }
    }

    void DecodeBatch(u32* out, sz offset, sz count) const {
        for (sz i = 0; i < count && (offset + i) < ValueCount; ++i) {
            out[i] = Decode(offset + i);
        }
    }

    sz MemoryUsage() const override {
        return sizeof(BitWidth) + sizeof(ValueCount) + PackedData.size() * sizeof(u64);
    }

    sz TotalBits() const { return ValueCount * BitWidth; }
    sz PackedWords() const { return PackedData.size(); }

    void Clear() override {
        BitWidth = 0;
        PackedData.clear();
        PackedData.shrink_to_fit();
        ValueCount = 0;
    }

private:
    void Pack(u32 value, sz idx) {
        u64 bitPos = idx * BitWidth;
        u64 val    = static_cast<u64>(value) & ((1ULL << BitWidth) - 1);
        sz wordIdx = bitPos / 64;
        u32 shift  = bitPos % 64;

        PackedData[wordIdx] |= val << shift;
        if (shift + BitWidth > 64 && wordIdx + 1 < PackedData.size()) {
            PackedData[wordIdx + 1] |= val >> (64 - shift);
        }
    }
};

// ============================================================================
// FrameOfReferenceEncoding — stores scaled offset from baseline as integer
// ============================================================================

class FrameOfReferenceEncoding final : public IEncoding {
public:
    f64              BaseValue = 0.0;
    f64              Scale     = 1.0;
    std::vector<i32> EncodedValues;

    EncodingKind Kind() const override { return EncodingKind::FrameOfReference; }
    DataType ElementType() const override { return DataType::Float64; }
    sz RowCount() const override { return EncodedValues.size(); }

    void Build(const f64* data, sz count) {
        Clear();
        if (count == 0) return;

        f64 minVal = data[0];
        f64 maxVal = data[0];
        for (sz i = 1; i < count; ++i) {
            if (data[i] < minVal) minVal = data[i];
            if (data[i] > maxVal) maxVal = data[i];
        }

        BaseValue = minVal;

        f64 range = maxVal - minVal;
        if (range <= 0.0) {
            Scale = 1.0;
        } else {
            Scale = range / static_cast<f64>(INT32_MAX);
        }

        EncodedValues.reserve(count);
        for (sz i = 0; i < count; ++i) {
            f64 normalized = (data[i] - BaseValue) / Scale;
            EncodedValues.push_back(static_cast<i32>(std::llround(normalized)));
        }
    }

    f64 Decode(sz row) const {
        i32 enc = EncodedValues[row];
        return BaseValue + static_cast<f64>(enc) * Scale;
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        f64* dst = static_cast<f64*>(out);
        for (sz i = 0; i < count && (offset + i) < EncodedValues.size(); ++i) {
            dst[i] = Decode(offset + i);
        }
    }

    void DecodeBatch(f64* out, sz offset, sz count) const {
        for (sz i = 0; i < count && (offset + i) < EncodedValues.size(); ++i) {
            out[i] = Decode(offset + i);
        }
    }

    sz MemoryUsage() const override {
        return sizeof(BaseValue) + sizeof(Scale) + EncodedValues.size() * sizeof(i32);
    }

    void Clear() override {
        BaseValue = 0.0;
        Scale     = 1.0;
        EncodedValues.clear();
        EncodedValues.shrink_to_fit();
    }
};

// ============================================================================
// FSSTEncoding — Fast Static Symbol Table for string compression
// ============================================================================

class FSSTEncoding final : public IEncoding {
public:
    static constexpr u32 kSymbolCount = 255;
    static constexpr u32 kMaxSymbolLen = 8;

    struct Symbol {
        u8  Len = 0;
        u8  Code = 0;
        u8  Data[kMaxSymbolLen] = {};
    };

    std::vector<Symbol> SymbolTable;
    std::vector<u8>     CompressedData;
    std::vector<u32>    Offsets;
    sz                  DecompressedLen = 0;

    EncodingKind Kind() const override { return EncodingKind::FSST; }
    DataType ElementType() const override { return DataType::String; }
    sz RowCount() const override { return Offsets.empty() ? 0 : Offsets.size() - 1; }

    void Build(const char* const* strings, const sz* lengths, sz count) {
        Clear();
        if (count == 0) return;

        std::vector<u8> freq(256, 0);
        for (sz i = 0; i < count; ++i) {
            for (sz j = 0; j < lengths[i]; ++j)
                freq[static_cast<u8>(strings[i][j])]++;
        }

        SymbolTable.resize(kSymbolCount);
        u8 code = 0;
        for (u32 c = 0; c < 256 && code < 255; ++c) {
            if (!ShouldInclude(c, freq[c])) continue;
            SymbolTable[code].Code = code;
            SymbolTable[code].Len  = 1;
            SymbolTable[code].Data[0] = static_cast<u8>(c);
            ++code;
        }

        Offsets.reserve(count + 1);
        Offsets.push_back(0);

        sz compressedPos = 0;
        for (sz i = 0; i < count; ++i) {
            sz len = lengths[i];
            sz cmpSz = (len + 3) / 4 * 3 + 1;
            if (compressedPos + cmpSz > CompressedData.size()) {
                CompressedData.resize((compressedPos + cmpSz) * 2);
            }
            for (sz j = 0; j < len; ++j) {
                Symbol* sym = FindSymbol(strings[i][j]);
                if (sym) {
                    CompressedData[compressedPos++] = sym->Code;
                } else {
                    CompressedData[compressedPos++] = 0;
                    CompressedData[compressedPos++] = static_cast<u8>(strings[i][j]);
                }
            }
            Offsets.push_back(compressedPos);
        }
    }

    sz MemoryUsage() const override {
        return SymbolTable.size() * sizeof(Symbol) + CompressedData.size() + Offsets.size() * sizeof(u32);
    }

    void DecodeBatch(void* out, sz offset, sz count) const override {
        char** dst = static_cast<char**>(out);
        for (sz i = 0; i < count && (offset + i) < RowCount(); ++i) {
            sz row = offset + i;
            sz start = Offsets[row];
            sz end   = Offsets[row + 1];
            sz outLen = end - start;
            if (dst[i]) {
                for (sz j = 0; j < outLen; ++j) {
                    u8 c = CompressedData[start + j];
                    dst[i][j] = (c < kSymbolCount && SymbolTable[c].Len > 0)
                        ? SymbolTable[c].Data[0]
                        : static_cast<char>(CompressedData[start + j + 1]);
                    if (c == 0) ++j;
                }
            }
        }
    }

    void Clear() override {
        SymbolTable.clear();
        CompressedData.clear();
        CompressedData.shrink_to_fit();
        Offsets.clear();
        Offsets.shrink_to_fit();
        DecompressedLen = 0;
    }

private:
    bool ShouldInclude(u32 c, u8 freq) const {
        return freq > 0 && c != 0;
    }

    Symbol* FindSymbol(char c) {
        u8 uc = static_cast<u8>(c);
        for (auto& s : SymbolTable) {
            if (s.Len > 0 && s.Data[0] == uc) return &s;
        }
        return nullptr;
    }
};

// ============================================================================
// EncodingFactory — dynamically creates encodings based on runtime type + kind
// ============================================================================

class EncodingFactory {
public:
    static std::unique_ptr<IEncoding> Create(EncodingKind kind, DataType elementType, Arena& arena) {
        (void)arena;

        switch (kind) {
        case EncodingKind::Plain:
            return nullptr;

        case EncodingKind::Dictionary: {
            switch (elementType) {
            case DataType::Int8:     return std::make_unique<DictionaryEncoding<i8>>();
            case DataType::Int16:    return std::make_unique<DictionaryEncoding<i16>>();
            case DataType::Int32:    return std::make_unique<DictionaryEncoding<i32>>();
            case DataType::Int64:    return std::make_unique<DictionaryEncoding<i64>>();
            case DataType::Uint8:    return std::make_unique<DictionaryEncoding<u8>>();
            case DataType::Uint16:   return std::make_unique<DictionaryEncoding<u16>>();
            case DataType::Uint32:   return std::make_unique<DictionaryEncoding<u32>>();
            case DataType::Uint64:   return std::make_unique<DictionaryEncoding<u64>>();
            case DataType::Float32:  return std::make_unique<DictionaryEncoding<f32>>();
            case DataType::Float64:  return std::make_unique<DictionaryEncoding<f64>>();
            default: return nullptr;
            }
        }

        case EncodingKind::RLE: {
            switch (elementType) {
            case DataType::Int8:     return std::make_unique<RLEEncoding<i8>>();
            case DataType::Int16:    return std::make_unique<RLEEncoding<i16>>();
            case DataType::Int32:    return std::make_unique<RLEEncoding<i32>>();
            case DataType::Int64:    return std::make_unique<RLEEncoding<i64>>();
            case DataType::Uint8:    return std::make_unique<RLEEncoding<u8>>();
            case DataType::Uint16:   return std::make_unique<RLEEncoding<u16>>();
            case DataType::Uint32:   return std::make_unique<RLEEncoding<u32>>();
            case DataType::Uint64:   return std::make_unique<RLEEncoding<u64>>();
            case DataType::Float32:  return std::make_unique<RLEEncoding<f32>>();
            case DataType::Float64:  return std::make_unique<RLEEncoding<f64>>();
            default: return nullptr;
            }
        }

        case EncodingKind::Delta: {
            switch (elementType) {
            case DataType::Int8:     return std::make_unique<DeltaEncoding<i8>>();
            case DataType::Int16:    return std::make_unique<DeltaEncoding<i16>>();
            case DataType::Int32:    return std::make_unique<DeltaEncoding<i32>>();
            case DataType::Int64:    return std::make_unique<DeltaEncoding<i64>>();
            case DataType::Uint8:    return std::make_unique<DeltaEncoding<u8>>();
            case DataType::Uint16:   return std::make_unique<DeltaEncoding<u16>>();
            case DataType::Uint32:   return std::make_unique<DeltaEncoding<u32>>();
            case DataType::Uint64:   return std::make_unique<DeltaEncoding<u64>>();
            default: return nullptr;
            }
        }

        case EncodingKind::BitPacked:
            return std::make_unique<BitPackedEncoding>();

        case EncodingKind::FrameOfReference:
            return std::make_unique<FrameOfReferenceEncoding>();

        case EncodingKind::FSST:
            return std::make_unique<FSSTEncoding>();

        default:
            return nullptr;
        }
    }

    static EncodingKind SuggestEncoding(DataType elementType, sz cardinality, sz rowCount, bool isSorted) {
        if (cardinality == 0) return EncodingKind::Plain;

        f64 ratio = static_cast<f64>(cardinality) / static_cast<f64>(rowCount);

        if (ratio <= 0.01) return EncodingKind::Dictionary;
        if (isSorted && IsIntegralType(elementType)) return EncodingKind::Delta;
        if (ratio <= 0.3) return EncodingKind::Dictionary;
        if (elementType == DataType::Uint32 || elementType == DataType::Uint64)
            return EncodingKind::BitPacked;
        if (elementType == DataType::Float64 || elementType == DataType::Float32)
            return EncodingKind::FrameOfReference;

        return EncodingKind::Plain;
    }

    static bool IsEncodingApplicable(EncodingKind kind, DataType elementType) {
        switch (kind) {
        case EncodingKind::Dictionary: return IsNumericType(elementType) || elementType == DataType::String;
        case EncodingKind::RLE:        return IsNumericType(elementType);
        case EncodingKind::Delta:      return IsIntegralType(elementType) && elementType != DataType::Bool;
        case EncodingKind::BitPacked:  return elementType == DataType::Uint32;
        case EncodingKind::FrameOfReference: return elementType == DataType::Float64;
        case EncodingKind::FSST:       return elementType == DataType::String;
        case EncodingKind::Plain:      return true;
        default: return false;
        }
    }
};

// ============================================================================
// EncodingBuilder — builds multiple encodings and picks the best one
// ============================================================================

class EncodingBuilder {
public:
    template<typename T>
    static std::unique_ptr<IEncoding> BuildBest(const T* data, sz count, Arena& arena) {
        (void)arena;

        if (count == 0) return nullptr;

        DataType dt = TypeTraits<T>::kTypeId;

        sz cardinality = ComputeCardinality(data, count);
        bool isSorted   = IsSortedAscending(data, count);
        EncodingKind suggested = EncodingFactory::SuggestEncoding(dt, cardinality, count, isSorted);

        std::unique_ptr<IEncoding> best;
        f64 bestRatio = 0.0;

        for (u8 k = 0; k <= 6; ++k) {
            EncodingKind kind = static_cast<EncodingKind>(k);
            if (kind == EncodingKind::Plain) continue;
            if (!EncodingFactory::IsEncodingApplicable(kind, dt)) continue;

            auto enc = TryBuild(data, count, kind, arena);
            if (!enc) continue;

            EncodingStats stats = enc->GetStats();
            if (stats.CompressionRatio > bestRatio) {
                bestRatio = stats.CompressionRatio;
                best = std::move(enc);
            }
        }

        return best;
    }

private:
    template<typename T>
    static sz ComputeCardinality(const T* data, sz count) {
        if (count == 0) return 0;
        std::unordered_set<T> seen;
        for (sz i = 0; i < count && seen.size() < 10000; ++i)
            seen.insert(data[i]);
        return static_cast<sz>(seen.size());
    }

    template<typename T>
    static bool IsSortedAscending(const T* data, sz count) {
        if (count <= 1) return true;
        for (sz i = 1; i < count; ++i)
            if (data[i] < data[i - 1]) return false;
        return true;
    }

    template<typename T>
    static std::unique_ptr<IEncoding> TryBuild(const T* data, sz count, EncodingKind kind, Arena& arena) {
        switch (kind) {
        case EncodingKind::Dictionary: {
            if constexpr (std::is_arithmetic_v<T>) {
                auto enc = std::make_unique<DictionaryEncoding<T>>();
                enc->Build(data, count);
                return enc;
            }
            break;
        }
        case EncodingKind::RLE: {
            if constexpr (std::is_arithmetic_v<T>) {
                auto enc = std::make_unique<RLEEncoding<T>>();
                enc->Build(data, count);
                return enc;
            }
            break;
        }
        case EncodingKind::Delta: {
            if constexpr (std::is_integral_v<T>) {
                auto enc = std::make_unique<DeltaEncoding<T>>();
                enc->Build(data, count);
                return enc;
            }
            break;
        }
        default:
            break;
        }
        (void)arena;
        return nullptr;
    }
};

} // namespace encoding
} // namespace voxel
