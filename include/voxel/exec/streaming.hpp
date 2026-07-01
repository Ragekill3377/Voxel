#pragma once

#include "voxel/core/types.hpp"
#include "voxel/core/platform.hpp"
#include "voxel/exec/engine.hpp"

namespace voxel {

template<typename T>
class StreamingEngine {
    Engine<T> engine_;
    sz batch_size_;
    std::vector<T> buffer_;
    sz write_pos_ = 0;

public:
    explicit StreamingEngine(sz batchSize = 1024)
        : batch_size_(batchSize), buffer_(batchSize) {}

    Engine<T>& GetEngine() { return engine_; }

    void Feed(const T* data, sz count) {
        for (sz i = 0; i < count; ++i) {
            buffer_[write_pos_++] = data[i];
            if (write_pos_ >= batch_size_) {
                ProcessBatch();
                write_pos_ = 0;
            }
        }
    }

    void Flush() {
        if (write_pos_ > 0) {
            sz segId = engine_.AddSegment(buffer_.data(), write_pos_);
            (void)segId;
            engine_.Run();
            write_pos_ = 0;
        }
    }

    T GetResult(u8 reg = 0) const {
        if constexpr (std::is_floating_point_v<T>)
            return std::bit_cast<T>(engine_.ScalarReg(reg));
        else
            return static_cast<T>(engine_.ScalarReg(reg));
    }

private:
    void ProcessBatch() {
        engine_.AddSegment(buffer_.data(), write_pos_);
        if (engine_.SegmentCount() > 1) {
            // reset engine state for next batch
        }
        engine_.Run();
    }
};

} // namespace voxel
