#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/util/hash.hpp"

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <type_traits>

namespace voxel {

// ============================================================================
// StringView — lightweight non-owning string reference
// ============================================================================

class StringView {
public:
    const char* Data;
    sz Length;

    StringView() : Data(nullptr), Length(0) {}

    StringView(const char* s) : Data(s), Length(s ? std::strlen(s) : 0) {}

    StringView(const char* s, sz len) : Data(s), Length(len) {}

    bool operator==(const StringView& other) const {
        if (Length != other.Length) return false;
        if (Data == other.Data) return true;
        if (!Data || !other.Data) return false;
        return std::memcmp(Data, other.Data, Length) == 0;
    }

    bool operator!=(const StringView& other) const {
        return !(*this == other);
    }

    bool operator<(const StringView& other) const {
        sz minLen = (Length < other.Length) ? Length : other.Length;
        if (minLen > 0 && Data && other.Data) {
            int cmp = std::memcmp(Data, other.Data, minLen);
            if (cmp < 0) return true;
            if (cmp > 0) return false;
        }
        return Length < other.Length;
    }

    char operator[](sz i) const {
        return Data[i];
    }

    sz size() const { return Length; }

    const char* data() const { return Data; }
    const char* begin() const { return Data; }
    const char* end() const { return Data ? (Data + Length) : nullptr; }

    StringView substr(sz pos, sz n) const {
        if (pos >= Length) return StringView();
        sz avail = Length - pos;
        sz count = (n < avail) ? n : avail;
        return StringView(Data + pos, count);
    }

    sz find(char c) const {
        if (!Data) return ~0ULL;
        for (sz i = 0; i < Length; ++i) {
            if (Data[i] == c) return i;
        }
        return ~0ULL;
    }

    bool starts_with(const char* prefix) const {
        if (!prefix) return false;
        sz plen = std::strlen(prefix);
        if (plen > Length) return false;
        return std::memcmp(Data, prefix, plen) == 0;
    }

    bool ends_with(const char* suffix) const {
        if (!suffix) return false;
        sz slen = std::strlen(suffix);
        if (slen > Length) return false;
        return std::memcmp(Data + Length - slen, suffix, slen) == 0;
    }

    i32 compare(const StringView& other) const {
        sz minLen = (Length < other.Length) ? Length : other.Length;
        if (minLen > 0) {
            int cmp = std::memcmp(Data, other.Data, minLen);
            if (cmp != 0) return cmp < 0 ? -1 : 1;
        }
        if (Length < other.Length) return -1;
        if (Length > other.Length) return 1;
        return 0;
    }

    u64 Hash() const {
        if (!Data || Length == 0) return 0;
        return HashBytes(Data, Length);
    }
};

// ============================================================================
// StringBuilder — efficient on-stack string construction
// ============================================================================

class StringBuilder {
public:
    static constexpr sz kBufferSize = 256;

    StringBuilder() : Length_(0) {
        Buffer_[0] = '\0';
    }

    StringBuilder& Append(const char* s) {
        if (!s) return *this;
        sz slen = std::strlen(s);
        for (sz i = 0; i < slen && Length_ + 1 < kBufferSize; ++i) {
            Buffer_[Length_++] = s[i];
        }
        Buffer_[Length_] = '\0';
        return *this;
    }

    StringBuilder& Append(char c) {
        if (Length_ + 1 < kBufferSize) {
            Buffer_[Length_++] = c;
            Buffer_[Length_] = '\0';
        }
        return *this;
    }

    StringBuilder& Append(u64 n) {
        AppendU64(n);
        return *this;
    }

    StringBuilder& Append(i64 n) {
        if (n < 0) {
            Append('-');
            AppendU64(static_cast<u64>(-n));
        } else {
            AppendU64(static_cast<u64>(n));
        }
        return *this;
    }

    StringBuilder& Append(f64 n, u32 precision = 6) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.*f", static_cast<int>(precision), n);
        Append(buf);
        return *this;
    }

    StringBuilder& Append(u32 n) {
        AppendU64(static_cast<u64>(n));
        return *this;
    }

    StringBuilder& Append(i32 n) {
        return Append(static_cast<i64>(n));
    }

    void Clear() {
        Length_ = 0;
        Buffer_[0] = '\0';
    }

    StringView View() const {
        return StringView(Buffer_, Length_);
    }

    const char* CStr() const {
        return Buffer_;
    }

private:
    char Buffer_[kBufferSize];
    sz Length_;

    void AppendU64(u64 n) {
        if (n == 0) {
            Append('0');
            return;
        }
        char tmp[24];
        u32 pos = 0;
        while (n > 0 && pos < sizeof(tmp)) {
            tmp[pos++] = static_cast<char>('0' + (n % 10));
            n /= 10;
        }
        while (pos > 0 && Length_ + 1 < kBufferSize) {
            Buffer_[Length_++] = tmp[--pos];
        }
        Buffer_[Length_] = '\0';
    }
};

} // namespace voxel
