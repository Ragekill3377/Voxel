"""
VoxelVM SQL Frontend — PostgreSQL-compatible SQL via libpg_query

Usage:
    import voxel_sql as vsql
    result = vsql.execute("SELECT SUM(price) FROM trades WHERE price > 500",
                          data={"trades": np.array([...])})
    result = vsql.execute("SELECT * FROM 'data.csv' WHERE col > 100")

Architecture:
    SQL text → pglast (PostgreSQL parser) → AST → VoxelVM query → result
"""
import numpy as np
import pglast
import csv as _csv
import os as _os

import voxel_py as _vx
import voxel_query as _vq


class SqlCompiler:
    """Walks a PostgreSQL AST and compiles to VoxelVM queries."""

    def __init__(self, data_sources=None):
        self._data = data_sources or {}
        self._columns = {}

    def execute(self, sql: str):
        """Parse SQL and execute against registered data sources."""
        trees = pglast.parse_sql(sql)
        if not trees:
            raise ValueError("Empty SQL statement")
        stmt = trees[0].stmt
        return self._compile(stmt)

    def _compile(self, node):
        t = type(node).__name__
        if t == "SelectStmt":
            return self._compile_select(node)
        elif t == "InsertStmt":
            raise NotImplementedError("INSERT not yet supported")
        elif t == "DeleteStmt":
            raise NotImplementedError("DELETE not yet supported")
        else:
            raise NotImplementedError(f"Statement type {t} not supported")

    def _compile_select(self, sel):
        from_clause = getattr(sel, "fromClause", None)
        if not from_clause or len(from_clause) == 0:
            raise ValueError("FROM clause required")

        table = self._resolve_table(from_clause[0])
        where = getattr(sel, "whereClause", None)
        group_by = getattr(sel, "groupClause", None)
        sort = getattr(sel, "sortClause", None)
        limit = getattr(sel, "limitCount", None)
        targets = getattr(sel, "targetList", None)

        # Determine aggregate function from targets
        agg_func = None
        if targets:
            agg_func = self._get_agg_func(targets[0].val)

        # ---- GROUP BY path ----
        if group_by is not None:
            return self._execute_group_by(table, where, group_by, targets, sort, limit)

        # ---- Standard query path (bytecode) ----
        q = _vq.from_numpy(table)
        if where is not None:
            self._compile_where(q, where)
        if targets:
            self._compile_targets(q, targets)
        if sort:
            ascending = self._get_sort_dir(sort)
            q.sort(ascending=ascending)
        if limit is not None:
            k = self._eval_const(limit)
            if k:
                q.topk(int(k), largest=True)
        return q.run()

    # ================================================================
    # GROUP BY execution
    # ================================================================
    def _execute_group_by(self, table, where, group_by, targets, sort, limit):
        """Execute GROUP BY using HashAggregator."""
        # Get group column from the GROUP BY clause
        group_col_names = self._resolve_group_columns(group_by)

        # Build a filtered mask from WHERE clause
        filtered_table = table
        filtered_offsets = None
        if where is not None:
            mask = self._evaluate_where_numpy(table, where)
            filtered_table = table[mask]
            filtered_offsets = np.where(mask)[0]

        # Get aggregate function
        agg_func = "sum"
        if targets:
            agg_func = self._get_agg_func(targets[0].val) or "sum"

        # Get group keys (from data dict or as int index)
        group_keys = None
        if group_col_names and group_col_names[0] in self._data:
            raw_keys = self._data[group_col_names[0]]
            if filtered_offsets is not None:
                group_keys = np.array(raw_keys[filtered_offsets], dtype=np.uint32)
            else:
                group_keys = np.array(raw_keys, dtype=np.uint32)
        else:
            # No group column in data — use sequential indices
            group_keys = np.arange(len(filtered_table), dtype=np.uint32)

        # Run HashAggregator
        arena = _vx.Arena()
        agg = _vx.HashAggregator(arena)
        agg.init(max(128, int(len(group_keys) / 10) + 1))
        agg.accumulate(group_keys, filtered_table)
        ngroups = agg.group_count()

        # Extract results
        return {"groups": int(ngroups), "function": agg_func}

    def _evaluate_where_numpy(self, table, where):
        """Evaluate WHERE clause using numpy to get boolean mask."""
        t = type(where).__name__
        if t == "A_Expr":
            return self._eval_aexpr_numpy(table, where)
        elif t == "BoolExpr":
            result = np.ones(len(table), dtype=bool)
            for arg in where.args:
                result &= self._evaluate_where_numpy(table, arg)
            return result
        return np.ones(len(table), dtype=bool)

    def _eval_aexpr_numpy(self, table, expr):
        op = self._resolve_opname(expr.name)
        val = self._eval_const(expr.rexpr)
        if op == ">":
            return table > val
        elif op == ">=":
            return table >= val
        elif op == "<":
            return table < val
        elif op == "<=":
            return table <= val
        elif op == "=":
            return np.abs(table - val) < 1e-10
        return np.ones(len(table), dtype=bool)

    def _resolve_group_columns(self, group_by):
        """Resolve GROUP BY column names."""
        cols = []
        if isinstance(group_by, (list, tuple)):
            for item in group_by:
                name = self._resolve_column(item)
                if name:
                    cols.append(name)
        return cols

    def _get_agg_func(self, val):
        """Get aggregate function name from target value."""
        vt = type(val).__name__
        if vt == "FuncCall":
            return self._resolve_opname(val.funcname)
        return None

    def _get_sort_dir(self, sort_clause):
        """Get sort direction from ORDER BY clause."""
        if isinstance(sort_clause, (list, tuple)):
            for item in sort_clause:
                sd = getattr(item, 'sortby_dir', None)
                if sd is not None:
                    # SORTBY_DEFAULT/SORTBY_ASC = ascending, SORTBY_DESC = descending
                    dir_name = str(sd).lower()
                    return "desc" not in dir_name
        return True  # default ascending

    # ================================================================
    # Table resolution
    # ================================================================
    def _resolve_table(self, from_item):
        t = type(from_item).__name__
        if t == "RangeVar":
            name = from_item.relname.value if hasattr(from_item.relname, 'value') else str(from_item.relname)
            if name in self._data:
                val = self._data[name]
                return val.astype(np.float64) if isinstance(val, np.ndarray) else val
            csv_path = name.strip("'\"")
            if _os.path.exists(csv_path):
                return self._load_csv(csv_path)
            if name in self._columns:
                return self._columns[name]
            raise ValueError(f"Cannot resolve table: {name}")
        if t == "RangeSubselect":
            return self._compile(from_item.subquery)
        raise ValueError(f"Cannot resolve table: {type(from_item).__name__}")

    def _load_csv(self, path):
        rows = []
        with open(path, 'r') as f:
            reader = _csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    try:
                        rows.append(float(row[0]))
                    except ValueError:
                        pass
        return np.array(rows, dtype=np.float64)

    # ================================================================
    # WHERE clause compilation
    # ================================================================
    def _compile_where(self, q, clause):
        t = type(clause).__name__
        if t == "A_Expr":
            return self._compile_aexpr(q, clause)
        elif t == "BoolExpr":
            return self._compile_boolexpr(q, clause)

    def _compile_aexpr(self, q, expr):
        left = self._resolve_column(expr.lexpr)
        op_name = self._resolve_opname(expr.name)
        right = self._eval_const(expr.rexpr)
        if left is None:
            return
        if op_name == ">":
            q.filter_gt(right)
        elif op_name == ">=":
            q.filter_ge(right)
        elif op_name == "<":
            q.filter_lt(right)
        elif op_name == "<=":
            q.filter_le(right)
        elif op_name == "=":
            q.filter_eq(right)

    def _compile_boolexpr(self, q, expr):
        for arg in expr.args:
            self._compile_where(q, arg)

    # ================================================================
    # Targets and aggregates
    # ================================================================
    def _compile_targets(self, q, targets):
        for t in targets:
            val = t.val
            vt = type(val).__name__
            if vt == "FuncCall":
                self._compile_func(q, val)

    def _compile_func(self, q, func):
        name = self._resolve_opname(func.funcname)
        if name in ("sum",):
            q.sum()
        elif name in ("count",):
            q.count()
        elif name in ("avg",):
            q.sum()
            q.count()
        elif name in ("min",):
            q.min()
        elif name in ("max",):
            q.max()

    # ================================================================
    # Helpers
    # ================================================================
    def _resolve_column(self, col):
        t = type(col).__name__
        if t == "ColumnRef":
            fields = col.fields
            if fields:
                return fields[0].value if hasattr(fields[0], 'value') else str(fields[0])
        elif t == "A_Const":
            return self._eval_const(col)
        return None

    def _resolve_opname(self, name_node):
        if isinstance(name_node, (list, tuple)):
            result = ""
            for n in name_node:
                s = self._resolve_opname(n)
                result += s if s else ""
            return result.lower()
        if hasattr(name_node, 'sval'):
            return str(name_node.sval).lower()
        if hasattr(name_node, 'value'):
            return str(name_node.value).lower()
        return str(name_node).lower()

    def _eval_const(self, node):
        if node is None:
            return None
        t = type(node).__name__
        if t == "A_Const":
            val = node.val
            vt = type(val).__name__
            if vt == "Integer":
                return float(val.ival)
            elif vt == "Float":
                return float(val.fval.value if hasattr(val.fval, 'value') else str(val.fval))
            elif vt == "String":
                return str(val.sval)
            elif vt == "Null":
                return None
        elif t == "Integer":
            return float(node.ival)
        elif t == "Float":
            return float(node.fval.value if hasattr(node.fval, 'value') else str(node.fval))
        elif t == "TypeCast":
            return self._eval_const(node.arg)
        return None


# ================================================================
# Public API
# ================================================================
def execute(sql: str, data=None):
    """Execute SQL against VoxelVM.
    Args:
        sql: PostgreSQL-compatible SQL string
        data: dict mapping table/column names to numpy arrays
    Returns:
        Query result (float for aggregates, dict for GROUP BY)
    """
    compiler = SqlCompiler(data if isinstance(data, dict) else {"t": data})
    return compiler.execute(sql)


def load_csv(path, column=0):
    """Load a CSV file column into a numpy array."""
    rows = []
    with open(path, 'r') as f:
        reader = _csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and column < len(row):
                try:
                    rows.append(float(row[column]))
                except ValueError:
                    pass
    return np.array(rows, dtype=np.float64)
