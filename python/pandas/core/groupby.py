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
