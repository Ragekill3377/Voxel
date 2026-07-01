"""VoxelVM Large Data API — streaming and memory-mapped processing."""
import numpy as np
import voxel_py as _vx
from voxel_query import from_numpy


class LargeData:
    """Process datasets larger than RAM via mmap or chunked streaming."""

    def __init__(self, source, chunk_size=1_000_000, use_mmap=True):
        self._source = source
        self._chunk_size = chunk_size
        self._is_file = isinstance(source, str)
        self._mmap = None
        if self._is_file and use_mmap:
            self._mmap = _vx.MmapSegment(source)
            if not self._mmap.valid():
                self._mmap = None

    def sum(self, threshold=None):
        """Sum all values, optionally above threshold."""
        if self._mmap is not None:
            seg = self._mmap.view()
            if threshold is not None:
                return from_numpy(np.array(seg[:min(self._chunk_size, len(seg))])) \
                    .filter_gt(threshold).sum().run()
            return from_numpy(np.array(seg[:min(self._chunk_size, len(seg))])).sum().run()
        return from_numpy(self._source).filter_gt(threshold) \
            .sum().run() if threshold else from_numpy(self._source).sum().run()

    def count_above(self, threshold):
        """Count values above threshold."""
        if self._mmap is not None:
            seg = self._mmap.view()
            arr = np.array(seg[:self._chunk_size], dtype=np.float64) if len(seg) > self._chunk_size else np.array(seg, dtype=np.float64)
            return int(from_numpy(arr).filter_gt(threshold).count().run())
        return int(from_numpy(self._source).filter_gt(threshold).count().run())

    def chunked_sum(self, chunk_size=None):
        """Process large file in chunks, accumulating sum."""
        sz = chunk_size or self._chunk_size
        total = 0.0
        if self._mmap is not None:
            seg = self._mmap.view()
            n = len(seg)
            for start in range(0, n, sz):
                end = min(start + sz, n)
                chunk = np.array(seg[start:end], dtype=np.float64)
                total += from_numpy(chunk).sum().run()
        return total

    def __len__(self):
        if self._mmap is not None:
            return self._mmap.__len__()
        return len(self._source)


def from_file(filepath, chunk_size=1_000_000):
    """Create a LargeData reader from a binary file of f64 values."""
    return LargeData(filepath, chunk_size, use_mmap=True)


def from_array(arr):
    """Create a LargeData wrapper from a numpy array."""
    return LargeData(arr, use_mmap=False)
