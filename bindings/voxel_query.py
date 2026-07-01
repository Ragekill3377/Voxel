"""
VoxelVM High-Level Query Builder
Usage:
    import voxel_query as vq
    result = vq.from_numpy(data).filter_gt(500.0).sum().run()
"""
import numpy as np
import voxel_py as _vx

class Query:
    def __init__(self):
        self._engine = None
        self._seg_id = 0
        self._code = []
        self._result_reg = 0
        self._reg_counter = 5  # R0-R4 reserved: acc, offset, total, threshold, lanes
        self._offset_reg = 1
        self._total_reg = 2
        self._thresh_reg = 3
        self._acc_reg = 0
        self._lane_reg = 4
        self._vec_counter = 0

    def _emit(self, instr):
        self._code.append(instr.raw)

    def _alloc_reg(self):
        r = self._reg_counter
        self._reg_counter += 1
        return r

    def _alloc_vec(self):
        v = self._vec_counter
        self._vec_counter = (self._vec_counter + 1) % 2
        return v

    def filter_gt(self, threshold):
        """Keep rows where value > threshold."""
        self._engine.set_scalar_f64(self._thresh_reg, float(threshold))
        self._has_filter = True
        self._filter_cmp = _vx.CMP_GT
        return self

    def filter_ge(self, threshold):
        self._engine.set_scalar_f64(self._thresh_reg, float(threshold))
        self._has_filter = True
        self._filter_cmp = _vx.CMP_GE
        return self

    def filter_lt(self, threshold):
        self._engine.set_scalar_f64(self._thresh_reg, float(threshold))
        self._has_filter = True
        self._filter_cmp = _vx.CMP_LT
        return self

    def filter_le(self, threshold):
        self._engine.set_scalar_f64(self._thresh_reg, float(threshold))
        self._has_filter = True
        self._filter_cmp = _vx.CMP_LE
        return self

    def _emit_loop_body(self):
        """Emit VLOAD [+ VFILTER]. Returns the vector register holding results."""
        self._emit(_vx.Instruction.vload(0, self._offset_reg, self._seg_id, 0))
        if getattr(self, '_has_filter', False):
            cm = getattr(self, '_filter_cmp', _vx.CMP_GT)
            if cm == _vx.CMP_GT:
                self._emit(_vx.Instruction.vfilter_gt(1, 0, self._thresh_reg))
            elif cm == _vx.CMP_GE:
                self._emit(_vx.Instruction.vfilter_ge(1, 0, self._thresh_reg))
            else:
                self._emit(_vx.Instruction.vfilter(1, 0, self._thresh_reg, cm))
            return 1  # V1 holds filtered data
        return 0  # V0 holds raw data

    def sum(self):
        lanes = self._engine.k_lanes
        r5 = self._alloc_reg()
        vr = self._emit_loop_body()
        self._emit(_vx.Instruction.vsum(r5, vr))
        self._emit(_vx.Instruction.addf(self._acc_reg, self._acc_reg, r5))
        self._emit(_vx.Instruction.add(self._offset_reg, self._offset_reg, lanes))
        self._emit(_vx.Instruction.cmp(self._offset_reg, self._total_reg))
        self._emit(_vx.Instruction.jnz(-6))
        return self

    def count(self):
        lanes = self._engine.k_lanes
        r5 = self._alloc_reg()
        vr = self._emit_loop_body()
        self._emit(_vx.Instruction.vcount(r5, vr))
        self._emit(_vx.Instruction.add(self._acc_reg, self._acc_reg, r5))
        self._emit(_vx.Instruction.add(self._offset_reg, self._offset_reg, lanes))
        self._emit(_vx.Instruction.cmp(self._offset_reg, self._total_reg))
        self._emit(_vx.Instruction.jnz(-6))
        return self

    def min(self):
        lanes = self._engine.k_lanes
        r5 = self._alloc_reg()
        vr = self._emit_loop_body()
        self._emit(_vx.Instruction.vmin(r5, vr))
        self._emit(_vx.Instruction.cmp(self._acc_reg, r5))
        self._emit(_vx.Instruction.jle(2))
        self._emit(_vx.Instruction.movr(self._acc_reg, r5))
        self._emit(_vx.Instruction.add(self._offset_reg, self._offset_reg, lanes))
        self._emit(_vx.Instruction.cmp(self._offset_reg, self._total_reg))
        self._emit(_vx.Instruction.jnz(-8))
        return self

    def max(self):
        lanes = self._engine.k_lanes
        r5 = self._alloc_reg()
        vr = self._emit_loop_body()
        self._emit(_vx.Instruction.vmax(r5, vr))
        self._emit(_vx.Instruction.cmp(self._acc_reg, r5))
        self._emit(_vx.Instruction.jge(2))
        self._emit(_vx.Instruction.movr(self._acc_reg, r5))
        self._emit(_vx.Instruction.add(self._offset_reg, self._offset_reg, lanes))
        self._emit(_vx.Instruction.cmp(self._offset_reg, self._total_reg))
        self._emit(_vx.Instruction.jnz(-8))
        return self

    def sort(self, ascending=True):
        """Sort and return all values. Runs in C++ engine."""
        # Not bytecode-level — uses the SortOperator directly
        # Handled at run() time
        self._sort_ascending = ascending
        return self

    def topk(self, k, largest=True):
        """Select top-k values."""
        self._topk = k
        self._topk_largest = largest
        return self

    def group_by(self, group_keys):
        """Group by another column (group_keys: numpy array of u32)."""
        self._group_keys = group_keys
        return self

    def run(self):
        """Compile and execute the query. Returns the result."""
        lanes = self._engine.k_lanes
        self._engine.set_scalar_f64(self._acc_reg, float(0))
        self._engine.set_scalar(self._offset_reg, 0)
        self._engine.set_scalar(self._total_reg, self._total_count)

        # Emit lane constant at the START of the program
        self._code.insert(0, _vx.Instruction.mov(self._lane_reg, lanes).raw)

        # If no loop body was emitted, default to sum all values
        if len(self._code) == 1:
            self.sum()

        self._emit(_vx.Instruction.halt())
        self._engine.load_program(self._code)
        self._engine.run()

        if hasattr(self, '_group_keys'):
            # Group-by: use HashAggregator
            import numpy as np
            data = self._data
            arena = _vx.Arena()
            agg = _vx.HashAggregator(arena)
            agg.init(128)
            agg.accumulate(self._group_keys, data)
            return agg.group_count()

        if hasattr(self, '_sort_ascending'):
            return _vx.sort_ascending(self._data)

        if hasattr(self, '_topk'):
            return _vx.topk_select(self._data, self._topk, self._topk_largest)

        return self._engine.get_scalar_f64(self._acc_reg)


def from_numpy(data: np.ndarray):
    """Create a query from a numpy array."""
    if data.dtype != np.float64:
        data = data.astype(np.float64)

    q = Query()
    q._engine = _vx.EngineF64()
    q._data = data
    q._seg_id = q._engine.add_segment(data)
    q._total_count = len(data)
    return q


def from_list(values):
    """Create a query from a Python list."""
    return from_numpy(np.array(values, dtype=np.float64))


def filter_and_sum(data, threshold):
    """One-shot filter+sum. Fastest path."""
    return from_numpy(data).filter_gt(threshold).sum().run()


def filter_and_count(data, threshold):
    """One-shot filter+count."""
    return from_numpy(data).filter_gt(threshold).count().run()


def group_sum(data, groups):
    """Group by and sum."""
    q = from_numpy(data)
    q._group_keys = np.array(groups, dtype=np.uint32)
    return q.group_by(q._group_keys).run()
