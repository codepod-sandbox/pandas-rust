"""GroupBy wrapper for _pandas_native.GroupBy."""


class GroupBy:
    """GroupBy object wrapping native PyGroupBy."""

    def __init__(self, native_groupby):
        self._native = native_groupby

    def __repr__(self):
        return repr(self._native)

    def _wrap(self, result):
        from .frame import DataFrame
        return DataFrame._from_native(result)

    def sum(self): return self._wrap(self._native.sum())
    def mean(self): return self._wrap(self._native.mean())
    def min(self): return self._wrap(self._native.min())
    def max(self): return self._wrap(self._native.max())
    def count(self): return self._wrap(self._native.count())
    def std(self): return self._wrap(self._native.std())
    def var(self): return self._wrap(self._native.var())
    def median(self): return self._wrap(self._native.median())
    def first(self): return self._wrap(self._native.first())
    def last(self): return self._wrap(self._native.last())
    def size(self): return self._wrap(self._native.size())

    def __getitem__(self, key):
        """Select column(s) to aggregate, returning a _ColumnGroupBy."""
        if isinstance(key, str):
            return _ColumnGroupBy(self, [key])
        return _ColumnGroupBy(self, list(key))

    def agg(self, func):
        """Aggregate using one or more functions.

        Parameters
        ----------
        func : str or dict
            If str, call that aggregation method.
            If dict, maps column name -> agg function name.
        """
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, dict):
            results = {}
            for col, agg_func in func.items():
                if isinstance(agg_func, list):
                    # Multiple agg functions per col not yet supported; use first
                    agg_func = agg_func[0]
                full = getattr(self, agg_func)()
                try:
                    results[col] = full[col].tolist()
                except Exception:
                    results[col] = []
            from .frame import DataFrame
            return DataFrame(results)
        raise TypeError("agg expects a string or dict, got {}".format(type(func)))


class _ColumnGroupBy:
    """GroupBy with a subset of columns selected for aggregation."""

    def __init__(self, groupby, columns):
        self._groupby = groupby
        self._columns = columns

    def _apply_and_select(self, method_name):
        full_result = getattr(self._groupby, method_name)()
        if len(self._columns) == 1:
            return full_result[self._columns[0]]
        return full_result[self._columns]

    def sum(self): return self._apply_and_select("sum")
    def mean(self): return self._apply_and_select("mean")
    def min(self): return self._apply_and_select("min")
    def max(self): return self._apply_and_select("max")
    def count(self): return self._apply_and_select("count")
    def std(self): return self._apply_and_select("std")
    def var(self): return self._apply_and_select("var")
    def median(self): return self._apply_and_select("median")
    def first(self): return self._apply_and_select("first")
    def last(self): return self._apply_and_select("last")
    def size(self): return self._apply_and_select("size")

    def agg(self, func):
        if isinstance(func, str):
            return self._apply_and_select(func)
        raise TypeError("agg expects a string for _ColumnGroupBy")

    def __repr__(self):
        return "_ColumnGroupBy(columns={})".format(self._columns)
