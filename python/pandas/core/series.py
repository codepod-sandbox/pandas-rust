"""Series wrapper for _pandas_native.Series."""
import _pandas_native as _native


class Series:
    """One-dimensional labeled array."""

    def __init__(self, data=None, name=None, index=None):
        if isinstance(data, _native.Series):
            self._native = data
        elif isinstance(data, list):
            col_name = name or "0"
            df = _native.DataFrame({col_name: data})
            self._native = df.get_column(col_name)
        else:
            raise TypeError("Cannot construct Series from {}".format(type(data)))

    @classmethod
    def _from_native(cls, native_series):
        obj = cls.__new__(cls)
        obj._native = native_series
        return obj

    @property
    def name(self):
        return self._native.name

    @property
    def dtype(self):
        return self._native.dtype

    def __len__(self):
        return self._native.count()

    def __repr__(self):
        return repr(self._native)

    def __getitem__(self, key):
        return self._native[key]

    # Arithmetic operators
    def __add__(self, other):
        other_n = other._native if isinstance(other, Series) else other
        return Series._from_native(self._native + other_n)

    def __sub__(self, other):
        other_n = other._native if isinstance(other, Series) else other
        return Series._from_native(self._native - other_n)

    def __mul__(self, other):
        other_n = other._native if isinstance(other, Series) else other
        return Series._from_native(self._native * other_n)

    def __truediv__(self, other):
        other_n = other._native if isinstance(other, Series) else other
        return Series._from_native(self._native / other_n)

    def __neg__(self):
        return Series._from_native(-self._native)

    def __radd__(self, other):
        return Series._from_native(self._native + other)

    def __rsub__(self, other):
        # other - self: create a scalar series and subtract
        return Series._from_native(self._native * (-1) + other)

    def __rmul__(self, other):
        return Series._from_native(self._native * other)

    # Comparison — native ops require a Series, so we wrap scalars
    def _to_native_comparable(self, other):
        """Convert scalar to same-length Series for comparison."""
        if isinstance(other, Series):
            return other._native
        # Build a constant series of same length
        n = self._native.count()
        if isinstance(other, float):
            col = [other] * n
        else:
            col = [other] * n
        df = _native.DataFrame({"__cmp__": col})
        return df.get_column("__cmp__")

    def __eq__(self, other):
        return Series._from_native(self._native.eq(self._to_native_comparable(other)))

    def __ne__(self, other):
        return Series._from_native(self._native.ne(self._to_native_comparable(other)))

    def __lt__(self, other):
        return Series._from_native(self._native.lt(self._to_native_comparable(other)))

    def __le__(self, other):
        return Series._from_native(self._native.le(self._to_native_comparable(other)))

    def __gt__(self, other):
        return Series._from_native(self._native.gt(self._to_native_comparable(other)))

    def __ge__(self, other):
        return Series._from_native(self._native.ge(self._to_native_comparable(other)))

    # Aggregations
    def sum(self): return self._native.sum()
    def mean(self): return self._native.mean()
    def min(self): return self._native.min()
    def max(self): return self._native.max()
    def count(self): return self._native.count()
    def std(self, ddof=1): return self._native.std()
    def var(self, ddof=1): return self._native.var()
    def median(self): return self._native.median()

    # Null ops
    def isna(self): return Series._from_native(self._native.isna())
    def notna(self): return Series._from_native(self._native.notna())
    def fillna(self, value): return Series._from_native(self._native.fillna(value))
    def dropna(self): return Series._from_native(self._native.dropna())

    # Other
    def sort_values(self, ascending=True):
        return Series._from_native(self._native.sort_values(ascending))

    def astype(self, dtype):
        return Series._from_native(self._native.astype(str(dtype)))

    def copy(self):
        return Series._from_native(self._native.copy())

    def tolist(self):
        return self._native.tolist()

    def to_numpy(self):
        return self._native.to_numpy()

    @property
    def values(self):
        return self._native.values

    def to_dict(self):
        # Native returns {0: val, 1: val, ...}, preserve that
        return self._native.to_dict()

    def value_counts(self, sort=True, ascending=False, dropna=True):
        from .frame import DataFrame
        result = self._native.value_counts(sort, ascending, dropna)
        return DataFrame._from_native(result)

    def unique(self):
        return self._native.unique()

    def nunique(self, dropna=True):
        return self._native.nunique(dropna)

    def duplicated(self, keep="first"):
        return Series._from_native(self._native.duplicated(keep))

    def gt(self, other):
        return Series._from_native(self._native.gt(self._to_native_comparable(other)))

    def ge(self, other):
        return Series._from_native(self._native.ge(self._to_native_comparable(other)))

    def lt(self, other):
        return Series._from_native(self._native.lt(self._to_native_comparable(other)))

    def le(self, other):
        return Series._from_native(self._native.le(self._to_native_comparable(other)))

    def eq(self, other):
        return Series._from_native(self._native.eq(self._to_native_comparable(other)))

    def ne(self, other):
        return Series._from_native(self._native.ne(self._to_native_comparable(other)))
