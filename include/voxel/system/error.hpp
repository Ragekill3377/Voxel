#pragma once

#include "voxel/core/types.hpp"

#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace voxel {

// ============================================================================
// ErrorCode — well-known error identifiers
// ============================================================================

enum class ErrorCode : u32 {
    None = 0,
    InvalidOpcode,
    DivisionByZero,
    SegmentOutOfBounds,
    StackOverflow,
    StackUnderflow,
    NullDereference,
    TypeMismatch,
    InvalidCast,
    OutOfMemory,
    IoError,
    Timeout,
    InternalError,
    VerifyFailed,
    JitCompilationFailed,
    Count
};

// ============================================================================
// ErrorCodeString — forward-declared for use in VoxelError
// ============================================================================

inline const char* ErrorCodeString(ErrorCode code) {
    switch (code) {
    case ErrorCode::None:                return "None";
    case ErrorCode::InvalidOpcode:       return "InvalidOpcode";
    case ErrorCode::DivisionByZero:      return "DivisionByZero";
    case ErrorCode::SegmentOutOfBounds:  return "SegmentOutOfBounds";
    case ErrorCode::StackOverflow:       return "StackOverflow";
    case ErrorCode::StackUnderflow:      return "StackUnderflow";
    case ErrorCode::NullDereference:     return "NullDereference";
    case ErrorCode::TypeMismatch:        return "TypeMismatch";
    case ErrorCode::InvalidCast:         return "InvalidCast";
    case ErrorCode::OutOfMemory:         return "OutOfMemory";
    case ErrorCode::IoError:             return "IoError";
    case ErrorCode::Timeout:             return "Timeout";
    case ErrorCode::InternalError:       return "InternalError";
    case ErrorCode::VerifyFailed:        return "VerifyFailed";
    case ErrorCode::JitCompilationFailed: return "JitCompilationFailed";
    default:                              return "Unknown";
    }
}

// ============================================================================
// VoxelError — exception with source location
// ============================================================================

class VoxelError : public std::runtime_error {
public:
    ErrorCode Code;
    sz BytecodeOffset;
    const char* File;
    u32 Line;
    std::string Details;

    VoxelError(ErrorCode c, sz offset, const char* file, u32 line, const std::string& details)
        : std::runtime_error(FormatMessage(c, offset, file, line, details))
        , Code(c)
        , BytecodeOffset(offset)
        , File(file)
        , Line(line)
        , Details(details)
    {}

    const char* what() const noexcept override {
        if (CachedWhat_.empty()) {
            CachedWhat_ = FormatMessage(Code, BytecodeOffset, File, Line, Details);
        }
        return CachedWhat_.c_str();
    }

private:
    mutable std::string CachedWhat_;

    static std::string FormatMessage(ErrorCode c, sz offset, const char* file, u32 line, const std::string& details) {
        std::string msg;
        msg.reserve(256);
        msg += ErrorCodeString(c);
        if (file) {
            msg += " at ";
            msg += file;
            msg += ":";
            msg += std::to_string(line);
        }
        if (offset != ~0ULL) {
            msg += " [bytecode offset ";
            msg += std::to_string(offset);
            msg += "]";
        }
        if (!details.empty()) {
            msg += ": ";
            msg += details;
        }
        return msg;
    }
};

// ============================================================================
// ErrorHandler — pluggable error callback interface
// ============================================================================

class ErrorHandler {
public:
    static thread_local ErrorHandler* Instance;

    virtual ~ErrorHandler() = default;

    virtual void OnError(const VoxelError& error) = 0;

    virtual void OnWarning(const std::string& msg) {
        (void)msg;
    }

    static ErrorHandler* Get() {
        return Instance;
    }

    static void Set(ErrorHandler* handler) {
        Instance = handler;
    }
};

// ============================================================================
// DefaultErrorHandler — prints to stderr
// ============================================================================

class DefaultErrorHandler : public ErrorHandler {
public:
    void OnError(const VoxelError& error) override {
        std::fprintf(stderr, "[VOXEL ERROR] %s\n", error.what());
        if (error.File) {
            std::fprintf(stderr, "  Location: %s:%u\n", error.File, error.Line);
        }
        if (error.BytecodeOffset != ~0ULL) {
            std::fprintf(stderr, "  Bytecode Offset: %zu\n", error.BytecodeOffset);
        }
        if (!error.Details.empty()) {
            std::fprintf(stderr, "  Details: %s\n", error.Details.c_str());
        }
        std::fprintf(stderr, "  Error Code: %u (%s)\n",
                     static_cast<u32>(error.Code), ErrorCodeString(error.Code));
    }

    void OnWarning(const std::string& msg) override {
        std::fprintf(stderr, "[VOXEL WARNING] %s\n", msg.c_str());
    }
};

} // namespace voxel
