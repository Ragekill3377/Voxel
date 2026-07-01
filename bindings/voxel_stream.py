"""VoxelVM Streaming — real-time data feed processing."""
import numpy as np
import voxel_py as _vx


class Stream:
    """Process real-time data feeds through VoxelVM bytecode."""

    def __init__(self, batch_size=1024):
        self._engine = _vx.StreamingEngine(batch_size)
        self._results = []

    def feed(self, data):
        """Feed a numpy array into the stream."""
        self._engine.feed(data)
        return self

    def flush(self):
        """Flush remaining data and get results."""
        self._engine.flush()
        return self._engine.get_result(0)

    def setup(self, threshold, op="sum"):
        """Configure the query to run on each batch."""
        e = self._engine.engine
        k_lanes = e.k_lanes
        e.set_scalar_f64(3, float(threshold))
        e.set_scalar(4, k_lanes)
        code = []
        if op == "sum":
            code = [
                _vx.Instruction.vload(0, 1, 0, 0).raw,
                _vx.Instruction.vfilter_gt(1, 0, 3).raw,
                _vx.Instruction.vsum(5, 1).raw,
                _vx.Instruction.addf(0, 0, 5).raw,
                _vx.Instruction.add(1, 1, 4).raw,
                _vx.Instruction.cmp(1, 2).raw,
                _vx.Instruction.jnz(-6).raw,
                _vx.Instruction.halt().raw,
            ]
        e.load_program(code)
        return self


def stream_filter_sum(generator, threshold, batch_size=1024):
    """Stream data from a generator, filter+sum, return final result.
    
    Args:
        generator: yields numpy f64 arrays
        threshold: filter values > threshold
        batch_size: rows per batch
    """
    s = Stream(batch_size)
    s.setup(threshold, "sum")
    for chunk in generator:
        s.feed(chunk)
    return s.flush()
