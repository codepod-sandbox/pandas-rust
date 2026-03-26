"""Series wrapper for _pandas_native.Series."""
import _pandas_native as _native


class _PythonSeries:
    """Lightweight pure-Python series for types the native layer cannot store (e.g. list of lists)."""

    def __init__(self, data, name=None):
        self._data = list(data)
        self._name = name

    @property
    def name(self):
        return self._name

    def tolist(self):
        return list(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "_PythonSeries({})".format(self._data)


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

    @property
    def shape(self):
        return (self._native.count(),)

    def __len__(self):
        return self._native.count()

    def __repr__(self):
        return repr(self._native)

    @property
    def iloc(self):
        from .frame import _SeriesiLocIndexer
        return _SeriesiLocIndexer(self)

    @property
    def loc(self):
        # For default RangeIndex, label == position; delegate to iloc
        from .frame import _SeriesiLocIndexer
        return _SeriesiLocIndexer(self)

    def __getitem__(self, key):
        return self._native.__getitem__(key)

    # Arithmetic operators
    def _coerce_scalar(self, other):
        """Coerce numeric scalars to float for native arithmetic."""
        if isinstance(other, int) and not isinstance(other, bool):
            return float(other)
        return other

    def __add__(self, other):
        if isinstance(other, Series):
            return Series._from_native(self._native + other._native)
        return Series._from_native(self._native + self._coerce_scalar(other))

    def __sub__(self, other):
        if isinstance(other, Series):
            return Series._from_native(self._native - other._native)
        return Series._from_native(self._native - self._coerce_scalar(other))

    def __mul__(self, other):
        if isinstance(other, Series):
            return Series._from_native(self._native * other._native)
        return Series._from_native(self._native * self._coerce_scalar(other))

    def __truediv__(self, other):
        if isinstance(other, Series):
            return Series._from_native(self._native / other._native)
        return Series._from_native(self._native / self._coerce_scalar(other))

    def __neg__(self):
        return Series._from_native(-self._native)

    def __radd__(self, other):
        return Series._from_native(self._native + self._coerce_scalar(other))

    def __rsub__(self, other):
        # other - self: negate and add scalar
        return Series._from_native(self._native * (-1.0) + self._coerce_scalar(other))

    def __rmul__(self, other):
        return Series._from_native(self._native * self._coerce_scalar(other))

    # Comparison — native ops require a Series, so we wrap scalars
    def _to_native_comparable(self, other):
        """Convert scalar to same-length Series for comparison."""
        if isinstance(other, Series):
            return other._native
        # Build a constant series of same length
        n = self._native.count()
        # Coerce int to float for native compatibility
        val = self._coerce_scalar(other)
        col = [val] * n
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
        # Support Python type objects: float -> "float64", int -> "int64", str -> "object"
        _type_map = {float: "float64", int: "int64", str: "object", bool: "bool"}
        if dtype in _type_map:
            dtype = _type_map[dtype]
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

    # --- New methods ---

    def apply(self, func):
        """Apply function element-wise, returning a new Series."""
        results = self._native.apply(func)
        return Series(results)

    def map(self, func_or_dict):
        """Map values using dict or callable."""
        if callable(func_or_dict):
            return self.apply(func_or_dict)
        return Series._from_native(self._native.map(func_or_dict))

    def replace(self, to_replace, value):
        """Replace scalar value with another scalar."""
        return Series._from_native(self._native.replace(to_replace, value))

    def between(self, left, right, inclusive="both"):
        """Return boolean mask: left <= self <= right."""
        return Series._from_native(self._native.between(float(left), float(right)))

    def isin(self, values):
        """Return boolean mask: element is in values list."""
        return Series._from_native(self._native.isin(list(values)))

    def nlargest(self, n=5):
        """Return n largest values."""
        return Series._from_native(self._native.nlargest(n))

    def nsmallest(self, n=5):
        """Return n smallest values."""
        return Series._from_native(self._native.nsmallest(n))

    def abs(self):
        """Return absolute value."""
        return Series._from_native(self._native.abs())

    def clip(self, lower=None, upper=None):
        """Clip values to [lower, upper]."""
        lo = float(lower) if lower is not None else None
        hi = float(upper) if upper is not None else None
        return Series._from_native(self._native.clip(lo, hi))

    def any(self):
        """Return True if any value is truthy."""
        return self._native.any()

    def all(self):
        """Return True if all values are truthy."""
        return self._native.all()

    @property
    def str(self):
        """String accessor for object-dtype Series."""
        return _StringAccessor(self)


class _StringAccessor:
    """Vectorized string operations for Series."""

    def __init__(self, series):
        self._series = series

    def upper(self):
        return self._series.map(lambda x: x.upper() if isinstance(x, str) else x)

    def lower(self):
        return self._series.map(lambda x: x.lower() if isinstance(x, str) else x)

    def strip(self):
        return self._series.map(lambda x: x.strip() if isinstance(x, str) else x)

    def lstrip(self):
        return self._series.map(lambda x: x.lstrip() if isinstance(x, str) else x)

    def rstrip(self):
        return self._series.map(lambda x: x.rstrip() if isinstance(x, str) else x)

    def len(self):
        return self._series.map(lambda x: len(x) if isinstance(x, str) else None)

    def contains(self, pat, case=True, na=False):
        if case:
            return self._series.map(lambda x: pat in x if isinstance(x, str) else na)
        else:
            pat_lower = pat.lower()
            return self._series.map(lambda x: pat_lower in x.lower() if isinstance(x, str) else na)

    def startswith(self, pat):
        return self._series.map(lambda x: x.startswith(pat) if isinstance(x, str) else False)

    def endswith(self, pat):
        return self._series.map(lambda x: x.endswith(pat) if isinstance(x, str) else False)

    def replace(self, pat, repl, n=-1, case=True, regex=False):
        return self._series.map(lambda x: x.replace(pat, repl) if isinstance(x, str) else x)

    def split(self, pat=None, n=-1, expand=False):
        vals = self._series.tolist()
        if pat is None:
            results = [x.split() if isinstance(x, str) else x for x in vals]
        else:
            results = [x.split(pat) if isinstance(x, str) else x for x in vals]
        return _PythonSeries(results, name=self._series.name)

    def cat(self, others=None, sep="", na_rep=None):
        vals = self._series.tolist()
        return sep.join(str(v) for v in vals if v is not None)

    def slice(self, start=None, stop=None, step=None):
        return self._series.map(lambda x: x[start:stop:step] if isinstance(x, str) else x)

    def get(self, i):
        return self._series.map(lambda x: x[i] if isinstance(x, str) and 0 <= i < len(x) else None)

    def zfill(self, width):
        return self._series.map(lambda x: x.zfill(width) if isinstance(x, str) else x)

    def title(self):
        return self._series.map(lambda x: x.title() if isinstance(x, str) else x)

    def capitalize(self):
        return self._series.map(lambda x: x.capitalize() if isinstance(x, str) else x)

    def __repr__(self):
        return "_StringAccessor"
