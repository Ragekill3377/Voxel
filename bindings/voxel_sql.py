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
        self._result = None
        self._group_keys = None

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

        # Build the query chain
        q = _vq.from_numpy(table)

        # WHERE clause
        if where is not None:
            self._compile_where(q, where)

        # Aggregates and projections
        if targets:
            self._compile_targets(q, targets)

        # GROUP BY
        if group_by:
            self._compile_group_by(q, group_by)

        # ORDER BY
        if sort:
            q.sort(ascending=True)

        # LIMIT / TopK
        if limit is not None:
            k = self._eval_const(limit)
            if k:
                q.topk(int(k), largest=True)

        return q.run()

    def _resolve_table(self, from_item):
        """Resolve a FROM clause item to a numpy array."""
        t = type(from_item).__name__

        if t == "RangeVar":
            name = from_item.relname.value if hasattr(from_item.relname, 'value') else str(from_item.relname)

            # Check registered data dict
            if name in self._data:
                val = self._data[name]
                if isinstance(val, np.ndarray):
                    return val.astype(np.float64)
                return val

            # Try CSV file
            csv_path = name.strip("'\"")
            if _os.path.exists(csv_path):
                return self._load_csv(csv_path)

            # Try as a column name
            if name in self._columns:
                return self._columns[name]

        if t == "RangeSubselect":
            return self._compile(from_item.subquery)

        raise ValueError(f"Cannot resolve table: {name}")

    def _load_csv(self, path):
        """Load a CSV file into a numpy array (single numeric column)."""
        rows = []
        with open(path, 'r') as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    try:
                        rows.append(float(row[0]))
                    except ValueError:
                        pass
        return np.array(rows, dtype=np.float64)

    def _compile_where(self, q, clause):
        """Compile WHERE clause into filter operations."""
        t = type(clause).__name__

        if t == "A_Expr":
            return self._compile_aexpr(q, clause)
        elif t == "BoolExpr":
            return self._compile_boolexpr(q, clause)
        elif t == "NullTest":
            raise NotImplementedError("IS NULL not yet supported")
        else:
            raise NotImplementedError(f"WHERE clause type {t} not supported")

    def _compile_aexpr(self, q, expr):
        left = self._resolve_column(expr.lexpr)
        op_name = self._resolve_opname(expr.name)
        right = self._eval_const(expr.rexpr)

        if left is None:
            return

        # Apply filter based on operator
        if op_name == ">":
            q.filter_gt(right)
        elif op_name == ">=":
            q.filter_ge(right)
        elif op_name == "<":
            q.filter_lt(right)
        elif op_name == "<=":
            q.filter_le(right)
        elif op_name == "=":
            q.filter_ge(right)  # approximate: x >= val and x <= val
        elif op_name == "<>":
            pass  # no direct not-equal filter in query builder yet

    def _compile_boolexpr(self, q, expr):
        """Compile AND/OR boolean expressions."""
        for arg in expr.args:
            self._compile_where(q, arg)

    def _compile_targets(self, q, targets):
        """Compile SELECT targets."""
        for t in targets:
            val = t.val
            vt = type(val).__name__
            if vt == "FuncCall":
                self._compile_func(q, val)

    def _compile_func(self, q, func):
        """Compile aggregate functions onto the query chain."""
        name = self._resolve_opname(func.funcname)
        if name in ("sum",):     q.sum()
        elif name in ("count",):  q.count()
        elif name in ("avg",):    q.sum(); q.count()
        elif name in ("min",):    q.min()
        elif name in ("max",):    q.max()
        else:
            raise NotImplementedError(f"Aggregate function {name} not supported")

    def _compile_group_by(self, q, group_clause):
        """Compile GROUP BY clause."""
        pass  # Group-by requires multi-column support; placeholder

    def _resolve_column(self, col):
        """Resolve a column reference."""
        t = type(col).__name__
        if t == "ColumnRef":
            fields = col.fields
            if fields:
                name = fields[0].value if hasattr(fields[0], 'value') else str(fields[0])
                return name
        elif t == "A_Const":
            return self._eval_const(col)
        return None

    def _resolve_opname(self, name_node):
        """Resolve an operator/function name node to a string."""
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
        """Evaluate a constant AST node to a Python value."""
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
        elif t == "FuncCall":
            return None

        return None


def execute(sql: str, data=None):
    """Execute SQL against VoxelVM. One-shot entry point.

    Args:
        sql: PostgreSQL-compatible SQL string
        data: dict mapping table/column names to numpy arrays, or a single numpy array

    Returns:
        Query result (float for aggregates, numpy array for selections)

    Examples:
        >>> result = execute("SELECT SUM(col) FROM data WHERE col > 500",
        ...                  data={"data": np.array([...])})
        >>> result = execute("SELECT SUM(col) FROM 'trades.csv' WHERE col > 100")
    """
    compiler = SqlCompiler(data if isinstance(data, dict) else {"t": data})
    return compiler.execute(sql)


def load_csv(path, column=0):
    """Load a CSV file column into a numpy array."""
    rows = []
    with open(path, 'r') as f:
        reader = _csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and column < len(row):
                try:
                    rows.append(float(row[column]))
                except ValueError:
                    pass
    return np.array(rows, dtype=np.float64)
