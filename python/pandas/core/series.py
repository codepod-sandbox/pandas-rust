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
        elif isinstance(data, (int, float, str, bool)) and not isinstance(data, type):
            # Scalar — if index provided, repeat; else single element
            if index is not None:
                data_list = [data] * len(index)
            else:
                data_list = [data]
            col_name = name or "0"
            df = _native.DataFrame({col_name: data_list})
            self._native = df.get_column(col_name)
        elif data is None:
            col_name = name or "0"
            df = _native.DataFrame({col_name: []})
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
        return (self._native.__len__(),)

    def __len__(self):
        return self._native.__len__()

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

    def __and__(self, other):
        lhs = self.tolist()
        if isinstance(other, Series):
            rhs = other.tolist()
        else:
            rhs = list(other)
        result = [bool(a) and bool(b) for a, b in zip(lhs, rhs)]
        return Series(result, name=self.name)

    def __or__(self, other):
        lhs = self.tolist()
        if isinstance(other, Series):
            rhs = other.tolist()
        else:
            rhs = list(other)
        result = [bool(a) or bool(b) for a, b in zip(lhs, rhs)]
        return Series(result, name=self.name)

    def __invert__(self):
        result = [not bool(v) for v in self.tolist()]
        return Series(result, name=self.name)

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
    def sum(self):
        try:
            return self._native.sum()
        except TypeError:
            # Bool dtype: cast to int first
            return self.astype("int64")._native.sum()
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

    # Alias matching upstream pandas API
    to_list = tolist

    def rename(self, index=None, **kwargs):
        """Return a new Series with the name changed."""
        # When called as s.rename("new_name"), index is the new name
        new_name = index
        vals = self.tolist()
        return Series(vals, name=new_name)

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
        """Return boolean mask based on inclusive setting."""
        if inclusive == "both":
            return (self >= left) & (self <= right)
        elif inclusive == "left":
            return (self >= left) & (self < right)
        elif inclusive == "right":
            return (self > left) & (self <= right)
        elif inclusive == "neither":
            return (self > left) & (self < right)
        # fallback
        return (self >= left) & (self <= right)

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

    def __iter__(self):
        return iter(self.tolist())

    def to_frame(self, name=None):
        from .frame import DataFrame
        col_name = name if name is not None else self.name
        return DataFrame({col_name: self.tolist()})

    def idxmax(self):
        vals = self.tolist()
        best_idx = 0
        best_val = vals[0]
        for i, v in enumerate(vals):
            if v is not None and (best_val is None or v > best_val):
                best_idx = i
                best_val = v
        return best_idx

    def idxmin(self):
        vals = self.tolist()
        best_idx = 0
        best_val = vals[0]
        for i, v in enumerate(vals):
            if v is not None and (best_val is None or v < best_val):
                best_idx = i
                best_val = v
        return best_idx

    def shift(self, periods=1, fill_value=None):
        """Shift elements by periods positions."""
        return Series._from_native(self._native.shift(periods))

    def where(self, cond, other=None):
        """Keep values where cond is True, replace False positions with other."""
        if isinstance(cond, Series):
            mask = cond.tolist()
        else:
            mask = list(cond)
        vals = self.tolist()
        result = [v if m else other for v, m in zip(vals, mask)]
        return Series(result, name=self.name)

    def equals(self, other):
        """Test whether two Series are identical element-by-element (NaN-safe)."""
        if not isinstance(other, Series):
            return False
        if len(self) != len(other):
            return False
        l = self.tolist()
        r = other.tolist()
        for lv, rv in zip(l, r):
            if lv is None and rv is None:
                continue
            if isinstance(lv, float) and isinstance(rv, float):
                if lv != lv and rv != rv:  # both NaN
                    continue
            if lv != rv:
                return False
        return True

    def cumsum(self):
        return Series._from_native(self._native.cumsum())

    def cumprod(self):
        return Series._from_native(self._native.cumprod())

    def cummax(self):
        return Series._from_native(self._native.cummax())

    def cummin(self):
        return Series._from_native(self._native.cummin())

    def sample(self, n=None, frac=None, random_state=None):
        import random
        if random_state is not None:
            random.seed(random_state)
        total = len(self)
        if frac is not None:
            n = int(total * frac)
        if n is None:
            n = 1
        indices = random.sample(range(total), min(n, total))
        vals = self.tolist()
        return Series([vals[i] for i in indices], name=self.name)

    def diff(self, periods=1):
        """Difference between consecutive elements."""
        return Series._from_native(self._native.diff(periods))

    def pct_change(self, periods=1):
        """Percentage change between current and prior element."""
        shifted = self.shift(periods)
        curr = self.tolist()
        prev = shifted.tolist()
        result = []
        for c, p in zip(curr, prev):
            if c is None or p is None or p == 0:
                result.append(None)
            else:
                result.append((c - p) / p)
        return Series(result, name=self.name)

    def rank(self, method="average", ascending=True):
        """Rank values. Null values get null rank."""
        return Series._from_native(self._native.rank(method, ascending))

    def round(self, decimals=0):
        """Round float values to given number of decimal places."""
        return Series._from_native(self._native.round(decimals))

    def item(self):
        """Return the first element of the underlying data as a Python scalar."""
        if len(self) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return self.tolist()[0]

    def head(self, n=5):
        """Return first n elements."""
        if n < 0:
            return self.iloc[:len(self) + n]
        return self.iloc[:n]

    def tail(self, n=5):
        """Return last n elements."""
        if n < 0:
            return self.iloc[-n:]
        if n == 0:
            return self.iloc[0:0]
        return self.iloc[-n:]

    def rolling(self, window):
        """Return a _Rolling object for window calculations."""
        return _Rolling(self, window)

    def expanding(self):
        """Return an _Expanding object for expanding window calculations."""
        return _Expanding(self)

    @property
    def str(self):
        """String accessor for object-dtype Series."""
        return _StringAccessor(self)

    def mode(self):
        """Return the most frequent value(s) as a Series."""
        vals = [v for v in self.tolist() if v is not None]
        if not vals:
            return Series([], name=self.name)
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        modes = sorted(v for v, c in counts.items() if c == max_count)
        return Series(modes, name=self.name)

    # --- Properties: size, ndim, empty, is_unique ---

    @property
    def size(self):
        """Number of elements in the Series."""
        return len(self)

    @property
    def ndim(self):
        """Number of dimensions (always 1 for Series)."""
        return 1

    @property
    def empty(self):
        """True if the Series has no elements."""
        return len(self) == 0

    @property
    def is_unique(self):
        """True if all values in the Series are unique."""
        return self.nunique(dropna=False) == len(self)

    # --- to_string ---

    def to_string(self):
        """Render Series to a console-friendly string."""
        return repr(self)

    # --- squeeze ---

    def squeeze(self):
        """Squeeze a single-element Series to a scalar."""
        if len(self) == 1:
            return self.tolist()[0]
        return self

    # --- add_prefix / add_suffix ---

    def add_prefix(self, prefix):
        """Prefix labels (index values) — returns copy of series with same name."""
        # For RangeIndex, this is a no-op on the data; we return a copy
        return self.copy()

    def add_suffix(self, suffix):
        """Suffix labels (index values) — returns copy of series with same name."""
        return self.copy()

    def items(self):
        """Iterate over (index, value) pairs."""
        vals = self.tolist()
        for i, v in enumerate(vals):
            yield i, v

    iteritems = items

    @property
    def is_monotonic_increasing(self):
        """Return True if values are monotonically non-decreasing."""
        vals = self.tolist()
        for i in range(1, len(vals)):
            if vals[i] is None or vals[i - 1] is None:
                return False
            if vals[i] < vals[i - 1]:
                return False
        return True

    @property
    def is_monotonic_decreasing(self):
        """Return True if values are monotonically non-increasing."""
        vals = self.tolist()
        for i in range(1, len(vals)):
            if vals[i] is None or vals[i - 1] is None:
                return False
            if vals[i] > vals[i - 1]:
                return False
        return True

    def argmax(self):
        """Return integer index of the maximum value."""
        return self.idxmax()

    def argmin(self):
        """Return integer index of the minimum value."""
        return self.idxmin()

    def prod(self):
        """Return product of all non-null values."""
        vals = [v for v in self.tolist() if v is not None]
        if not vals:
            return None
        p = 1
        for v in vals:
            p *= v
        return p

    product = prod

    def drop(self, labels):
        """Drop values by integer index position(s)."""
        if isinstance(labels, int):
            labels = [labels]
        label_set = set(labels)
        vals = self.tolist()
        new_vals = [v for i, v in enumerate(vals) if i not in label_set]
        return Series(new_vals, name=self.name)

    def mask(self, cond, other=None):
        """Replace values where cond is True with other (opposite of where)."""
        if isinstance(cond, Series):
            mask_list = cond.tolist()
        else:
            mask_list = list(cond)
        vals = self.tolist()
        result = [other if m else v for v, m in zip(vals, mask_list)]
        return Series(result, name=self.name)

    def reset_index(self, drop=False, name=None):
        """Reset the index of the Series.

        If drop=True, just return a new Series with RangeIndex.
        If drop=False, return a DataFrame with the old index as a column plus values.
        """
        from .frame import DataFrame
        series_name = name if name is not None else (self.name if self.name is not None else 0)
        vals = self.tolist()
        if drop:
            return Series(vals, name=self.name)
        # Return a DataFrame with "index" column and the series values column
        index_col = list(range(len(vals)))
        return DataFrame({"index": index_col, series_name: vals})


class _Rolling:
    """Rolling window calculations."""

    def __init__(self, series, window):
        self._series = series
        self._window = window

    def mean(self):
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            if window_vals:
                result.append(sum(window_vals) / len(window_vals))
            else:
                result.append(None)
        return Series(result, name=self._series.name)

    def sum(self):
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            result.append(sum(window_vals) if window_vals else None)
        return Series(result, name=self._series.name)

    def min(self):
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            result.append(min(window_vals) if window_vals else None)
        return Series(result, name=self._series.name)

    def max(self):
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            result.append(max(window_vals) if window_vals else None)
        return Series(result, name=self._series.name)

    def std(self):
        import math as _math
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            if len(window_vals) < 2:
                result.append(None)
            else:
                m = sum(window_vals) / len(window_vals)
                variance = sum((v - m) ** 2 for v in window_vals) / (len(window_vals) - 1)
                result.append(_math.sqrt(variance))
        return Series(result, name=self._series.name)

    def count(self):
        vals = self._series.tolist()
        n = len(vals)
        w = self._window
        result = [None] * (w - 1)
        for i in range(w - 1, n):
            window_vals = [v for v in vals[i - w + 1:i + 1] if v is not None]
            result.append(float(len(window_vals)))
        return Series(result, name=self._series.name)


class _Expanding:
    """Expanding window calculations."""

    def __init__(self, series):
        self._series = series

    def sum(self):
        vals = self._series.tolist()
        result = []
        running = 0
        for v in vals:
            if v is not None:
                running += v
            result.append(running)
        return Series(result, name=self._series.name)

    def mean(self):
        vals = self._series.tolist()
        result = []
        running = 0
        count = 0
        for v in vals:
            if v is not None:
                running += v
                count += 1
            result.append(running / count if count > 0 else None)
        return Series(result, name=self._series.name)

    def min(self):
        vals = self._series.tolist()
        result = []
        current_min = None
        for v in vals:
            if v is not None:
                current_min = v if current_min is None else min(current_min, v)
            result.append(current_min)
        return Series(result, name=self._series.name)

    def max(self):
        vals = self._series.tolist()
        result = []
        current_max = None
        for v in vals:
            if v is not None:
                current_max = v if current_max is None else max(current_max, v)
            result.append(current_max)
        return Series(result, name=self._series.name)


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

    def find(self, sub, start=0, end=None):
        return self._series.map(lambda x: x.find(sub, start, end) if isinstance(x, str) else -1)

    def zfill(self, width):
        return self._series.map(lambda x: x.zfill(width) if isinstance(x, str) else x)

    def title(self):
        return self._series.map(lambda x: x.title() if isinstance(x, str) else x)

    def capitalize(self):
        return self._series.map(lambda x: x.capitalize() if isinstance(x, str) else x)

    def isnumeric(self):
        return self._series.map(lambda x: x.isnumeric() if isinstance(x, str) else False)

    def isalpha(self):
        return self._series.map(lambda x: x.isalpha() if isinstance(x, str) else False)

    def isdigit(self):
        return self._series.map(lambda x: x.isdigit() if isinstance(x, str) else False)

    def isalnum(self):
        return self._series.map(lambda x: x.isalnum() if isinstance(x, str) else False)

    def isupper(self):
        return self._series.map(lambda x: x.isupper() if isinstance(x, str) else False)

    def islower(self):
        return self._series.map(lambda x: x.islower() if isinstance(x, str) else False)

    def count(self, pat):
        return self._series.map(lambda x: x.count(pat) if isinstance(x, str) else 0)

    def match(self, pat):
        import re
        regex = re.compile(pat)
        return self._series.map(lambda x: bool(regex.match(x)) if isinstance(x, str) else False)

    def pad(self, width, side="left", fillchar=" "):
        if side == "left":
            return self._series.map(lambda x: x.rjust(width, fillchar) if isinstance(x, str) else x)
        elif side == "right":
            return self._series.map(lambda x: x.ljust(width, fillchar) if isinstance(x, str) else x)
        elif side == "both":
            return self._series.map(lambda x: x.center(width, fillchar) if isinstance(x, str) else x)
        return self._series.map(lambda x: x.rjust(width, fillchar) if isinstance(x, str) else x)

    def center(self, width, fillchar=" "):
        return self._series.map(lambda x: x.center(width, fillchar) if isinstance(x, str) else x)

    def ljust(self, width, fillchar=" "):
        return self._series.map(lambda x: x.ljust(width, fillchar) if isinstance(x, str) else x)

    def rjust(self, width, fillchar=" "):
        return self._series.map(lambda x: x.rjust(width, fillchar) if isinstance(x, str) else x)

    def wrap(self, width):
        import textwrap
        return self._series.map(lambda x: "\n".join(textwrap.wrap(x, width)) if isinstance(x, str) else x)

    def repeat(self, repeats):
        return self._series.map(lambda x: x * repeats if isinstance(x, str) else x)

    def normalize(self, form):
        import unicodedata
        return self._series.map(lambda x: unicodedata.normalize(form, x) if isinstance(x, str) else x)

    def encode(self, encoding):
        raise NotImplementedError("str.encode not supported")

    def __repr__(self):
        return "_StringAccessor"
