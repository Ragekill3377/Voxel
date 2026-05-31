#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <limits>
#include <bit>
#include <string_view>
#include <array>

namespace voxel {

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

using sz  = std::size_t;
using isz = std::ptrdiff_t;

// ============================================================================
// DataType — runtime type identifier for polymorphic dispatch
// ============================================================================

enum class DataType : u32 {
    Null     = 0,
    Bool     = 1,
    Int8     = 2,
    Int16    = 3,
    Int32    = 4,
    Int64    = 5,
    Uint8    = 6,
    Uint16   = 7,
    Uint32   = 8,
    Uint64   = 9,
    Float32  = 10,
    Float64  = 11,
    Decimal32  = 12,
    Decimal64  = 13,
    Decimal128 = 14,
    String     = 20,
    Binary     = 21,
    FixedString = 22,
    Date       = 30,
    Time       = 31,
    Timestamp  = 32,
    Interval   = 33,
    Duration   = 34,
    List       = 50,
    Map        = 51,
    Struct_    = 52,
    Union_     = 53,
};

inline constexpr bool IsIntegralType(DataType t) {
    return t >= DataType::Bool && t <= DataType::Uint64 && t != DataType::Null;
}

inline constexpr bool IsFloatingType(DataType t) {
    return t >= DataType::Float32 && t <= DataType::Float64;
}

inline constexpr bool IsNumericType(DataType t) {
    return (t >= DataType::Int8 && t <= DataType::Decimal128) || t == DataType::Bool;
}

inline constexpr bool IsDecimalType(DataType t) {
    return t >= DataType::Decimal32 && t <= DataType::Decimal128;
}

inline constexpr bool IsTemporalType(DataType t) {
    return t >= DataType::Date && t <= DataType::Duration;
}

inline constexpr bool IsStringLikeType(DataType t) {
    return t >= DataType::String && t <= DataType::FixedString;
}

inline constexpr bool IsNestedType(DataType t) {
    return t >= DataType::List && t <= DataType::Union_;
}

inline constexpr u32 TypeSize(DataType t) {
    switch (t) {
    case DataType::Bool:     return 1;
    case DataType::Int8:     return 1;
    case DataType::Int16:    return 2;
    case DataType::Int32:    return 4;
    case DataType::Int64:    return 8;
    case DataType::Uint8:    return 1;
    case DataType::Uint16:   return 2;
    case DataType::Uint32:   return 4;
    case DataType::Uint64:   return 8;
    case DataType::Float32:  return 4;
    case DataType::Float64:  return 8;
    case DataType::Decimal32:  return 4;
    case DataType::Decimal64:  return 8;
    case DataType::Decimal128: return 16;
    case DataType::Date:     return 4;
    case DataType::Time:     return 8;
    case DataType::Timestamp: return 8;
    case DataType::Interval: return 16;
    case DataType::Duration: return 8;
    default: return 0;
    }
}

inline constexpr u32 TypeAlignment(DataType t) {
    u32 s = TypeSize(t);
    return (s >= 16) ? 16 : (s >= 8 ? 8 : (s >= 4 ? 4 : (s >= 2 ? 2 : 1)));
}

inline constexpr const char* TypeName(DataType t) {
    switch (t) {
    case DataType::Null:     return "null";
    case DataType::Bool:     return "bool";
    case DataType::Int8:     return "i8";
    case DataType::Int16:    return "i16";
    case DataType::Int32:    return "i32";
    case DataType::Int64:    return "i64";
    case DataType::Uint8:    return "u8";
    case DataType::Uint16:   return "u16";
    case DataType::Uint32:   return "u32";
    case DataType::Uint64:   return "u64";
    case DataType::Float32:  return "f32";
    case DataType::Float64:  return "f64";
    case DataType::Decimal32:  return "dec32";
    case DataType::Decimal64:  return "dec64";
    case DataType::Decimal128: return "dec128";
    case DataType::String:     return "string";
    case DataType::Binary:     return "binary";
    case DataType::FixedString: return "fixed_string";
    case DataType::Date:       return "date";
    case DataType::Time:       return "time";
    case DataType::Timestamp:  return "timestamp";
    case DataType::Interval:   return "interval";
    case DataType::Duration:   return "duration";
    case DataType::List:       return "list";
    case DataType::Map:        return "map";
    case DataType::Struct_:    return "struct";
    case DataType::Union_:     return "union";
    default: return "unknown";
    }
}

// ============================================================================
// TypeTraits — compile-time type properties
// ============================================================================

template<typename T> struct TypeTraits {};

template<> struct TypeTraits<bool> {
    static constexpr DataType kTypeId = DataType::Bool;
    static constexpr sz kSize = 1;
    static constexpr sz kAlign = 1;
    static constexpr bool kIsNumeric = true;
    static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false;
    static constexpr bool kIsSigned = false;
    using UnsignedType = u8;
    using SignedType = i8;
};

template<> struct TypeTraits<i8> {
    static constexpr DataType kTypeId = DataType::Int8;
    static constexpr sz kSize = 1; static constexpr sz kAlign = 1;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = true;
    using UnsignedType = u8; using SignedType = i8;
};

template<> struct TypeTraits<i16> {
    static constexpr DataType kTypeId = DataType::Int16;
    static constexpr sz kSize = 2; static constexpr sz kAlign = 2;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = true;
    using UnsignedType = u16; using SignedType = i16;
};

template<> struct TypeTraits<i32> {
    static constexpr DataType kTypeId = DataType::Int32;
    static constexpr sz kSize = 4; static constexpr sz kAlign = 4;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = true;
    using UnsignedType = u32; using SignedType = i32;
};

template<> struct TypeTraits<i64> {
    static constexpr DataType kTypeId = DataType::Int64;
    static constexpr sz kSize = 8; static constexpr sz kAlign = 8;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = true;
    using UnsignedType = u64; using SignedType = i64;
};

template<> struct TypeTraits<u8> {
    static constexpr DataType kTypeId = DataType::Uint8;
    static constexpr sz kSize = 1; static constexpr sz kAlign = 1;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = false;
    using UnsignedType = u8; using SignedType = i8;
};

template<> struct TypeTraits<u16> {
    static constexpr DataType kTypeId = DataType::Uint16;
    static constexpr sz kSize = 2; static constexpr sz kAlign = 2;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = false;
    using UnsignedType = u16; using SignedType = i16;
};

template<> struct TypeTraits<u32> {
    static constexpr DataType kTypeId = DataType::Uint32;
    static constexpr sz kSize = 4; static constexpr sz kAlign = 4;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = false;
    using UnsignedType = u32; using SignedType = i32;
};

template<> struct TypeTraits<u64> {
    static constexpr DataType kTypeId = DataType::Uint64;
    static constexpr sz kSize = 8; static constexpr sz kAlign = 8;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = true;
    static constexpr bool kIsFloating = false; static constexpr bool kIsSigned = false;
    using UnsignedType = u64; using SignedType = i64;
};

template<> struct TypeTraits<f32> {
    static constexpr DataType kTypeId = DataType::Float32;
    static constexpr sz kSize = 4; static constexpr sz kAlign = 4;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = false;
    static constexpr bool kIsFloating = true; static constexpr bool kIsSigned = true;
};

template<> struct TypeTraits<f64> {
    static constexpr DataType kTypeId = DataType::Float64;
    static constexpr sz kSize = 8; static constexpr sz kAlign = 8;
    static constexpr bool kIsNumeric = true; static constexpr bool kIsIntegral = false;
    static constexpr bool kIsFloating = true; static constexpr bool kIsSigned = true;
};

// ============================================================================
// ScalarValue — type-erased 16-byte value container
// ============================================================================

struct alignas(16) ScalarValue {
    u8 Data[16];
    DataType Type;

    ScalarValue() : Data{}, Type(DataType::Null) {}

    template<typename T> requires (sizeof(T) <= 16)
    static ScalarValue From(T v) {
        ScalarValue sv;
        sv.Type = TypeTraits<T>::kTypeId;
        std::memcpy(sv.Data, &v, sizeof(T));
        return sv;
    }

    template<typename T> requires (sizeof(T) <= 16)
    T As() const {
        T v;
        std::memcpy(&v, Data, sizeof(T));
        return v;
    }

    bool IsNull() const { return Type == DataType::Null; }
};

// ============================================================================
// NumericLimits — type-erased numeric range checks
// ============================================================================

struct NumericLimits {
    template<typename T>
    static constexpr T Min() { return std::numeric_limits<T>::lowest(); }

    template<typename T>
    static constexpr T Max() { return std::numeric_limits<T>::max(); }

    static bool FitsInType(f64 value, DataType target) {
        switch (target) {
        case DataType::Int8:     return value >= -128.0 && value <= 127.0;
        case DataType::Int16:    return value >= -32768.0 && value <= 32767.0;
        case DataType::Int32:    return value >= -2147483648.0 && value <= 2147483647.0;
        case DataType::Int64:    return value >= -9.223372036854776e18 && value <= 9.223372036854776e18;
        case DataType::Uint8:    return value >= 0.0 && value <= 255.0;
        case DataType::Uint16:   return value >= 0.0 && value <= 65535.0;
        case DataType::Uint32:   return value >= 0.0 && value <= 4294967295.0;
        case DataType::Uint64:   return value >= 0.0 && value <= 1.8446744073709552e19;
        case DataType::Float32:  return value >= -3.4028235e38 && value <= 3.4028235e38;
        case DataType::Float64:  return true;
        default: return false;
        }
    }

    static bool FitsInType(i64 value, DataType target) {
        switch (target) {
        case DataType::Int8:     return value >= -128 && value <= 127;
        case DataType::Int16:    return value >= -32768 && value <= 32767;
        case DataType::Int32:    return value >= -2147483648LL && value <= 2147483647LL;
        case DataType::Int64:    return true;
        case DataType::Uint8:    return value >= 0 && value <= 255;
        case DataType::Uint16:   return value >= 0 && value <= 65535;
        case DataType::Uint32:   return value >= 0 && value <= 4294967295LL;
        case DataType::Uint64:   return value >= 0;
        case DataType::Float32:  return true;
        case DataType::Float64:  return true;
        default: return false;
        }
    }
};

// ============================================================================
// TypePromotion — result type of binary operations
// ============================================================================

inline constexpr DataType PromoteType(DataType a, DataType b) {
    if (a == b) return a;
    if (a == DataType::Null || b == DataType::Null) return DataType::Null;
    u32 rankA = static_cast<u32>(a);
    u32 rankB = static_cast<u32>(b);
    if (rankA <= 11 && rankB <= 11) {
        return static_cast<DataType>(rankA > rankB ? rankA : rankB);
    }
    return DataType::Null;
}

} // namespace voxel
