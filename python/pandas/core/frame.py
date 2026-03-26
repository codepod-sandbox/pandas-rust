"""DataFrame wrapper for _pandas_native.DataFrame."""
import _pandas_native as _native


class DataFrame:
    """Two-dimensional labeled data structure."""

    def __init__(self, data=None, columns=None, index=None):
        if data is None:
            self._native = _native.DataFrame({})
        elif isinstance(data, dict):
            self._native = _native.DataFrame(data)
        elif isinstance(data, _native.DataFrame):
            self._native = data
        elif isinstance(data, list):
            if len(data) == 0:
                self._native = _native.DataFrame({})
            elif isinstance(data[0], dict):
                # Collect all keys in insertion order (preserving first-seen order).
                keys = []
                seen = set()
                for row in data:
                    for k in row:
                        if k not in seen:
                            keys.append(k)
                            seen.add(k)
                # Build column lists; missing keys default to None.
                col_data = {k: [] for k in keys}
                for row in data:
                    for k in keys:
                        col_data[k].append(row.get(k, None))
                self._native = _native.DataFrame(col_data)
            else:
                raise TypeError(
                    "Cannot construct DataFrame from list of {}".format(type(data[0]))
                )
        else:
            raise TypeError("Cannot construct DataFrame from {}".format(type(data)))

    @classmethod
    def _from_native(cls, native_df):
        """Wrap a native DataFrame without copying."""
        obj = cls.__new__(cls)
        obj._native = native_df
        return obj

    @property
    def shape(self):
        return self._native.shape

    @property
    def dtypes(self):
        return self._native.dtypes

    @property
    def columns(self):
        return self._native.columns

    def __len__(self):
        return self._native.shape[0]

    def __repr__(self):
        return repr(self._native)

    def __getitem__(self, key):
        result = self._native[key]
        if isinstance(result, _native.Series):
            from .series import Series
            return Series._from_native(result)
        elif isinstance(result, _native.DataFrame):
            return DataFrame._from_native(result)
        return result

    def __setitem__(self, key, value):
        from .series import Series
        if isinstance(value, Series):
            self._native[key] = value.tolist()
        else:
            self._native[key] = value

    def head(self, n=5):
        return DataFrame._from_native(self._native.head(n))

    def tail(self, n=5):
        return DataFrame._from_native(self._native.tail(n))

    def sort_values(self, by, ascending=True):
        return DataFrame._from_native(self._native.sort_values(by, ascending))

    def drop(self, columns=None, **kwargs):
        if columns is None:
            columns = kwargs.get('labels', [])
        if isinstance(columns, str):
            columns = [columns]
        return DataFrame._from_native(self._native.drop(columns))

    def rename(self, columns=None, **kwargs):
        if columns is None:
            return self.copy()
        return DataFrame._from_native(self._native.rename(columns))

    def copy(self):
        return DataFrame._from_native(self._native.copy())

    def astype(self, dtype):
        result = self.copy()
        if isinstance(dtype, dict):
            for col, dt in dtype.items():
                from .series import Series
                s = result[col].astype(dt)
                result[col] = s.tolist()
        return result

    # Aggregations
    def sum(self, axis=0):
        return self._native.sum()

    def mean(self, axis=0):
        return self._native.mean()

    def min(self, axis=0):
        return self._native.min()

    def max(self, axis=0):
        return self._native.max()

    def count(self):
        return self._native.count()

    def std(self, ddof=1):
        return self._native.std()

    def var(self, ddof=1):
        return self._native.var()

    def median(self):
        return self._native.median()

    def describe(self):
        return DataFrame._from_native(self._native.describe())

    # Null ops
    def fillna(self, value):
        return DataFrame._from_native(self._native.fillna(value))

    def dropna(self, how="any", **kwargs):
        return DataFrame._from_native(self._native.dropna(how))

    def isna(self):
        return DataFrame._from_native(self._native.isna())

    def notna(self):
        return DataFrame._from_native(self._native.notna())

    def duplicated(self, subset=None, keep="first"):
        from .series import Series
        result = self._native.duplicated(subset, keep)
        return Series._from_native(result)

    def drop_duplicates(self, subset=None, keep="first"):
        result = self._native.drop_duplicates(subset, keep)
        return DataFrame._from_native(result)

    # GroupBy & Merge
    def groupby(self, by):
        if isinstance(by, str):
            by = [by]
        from .groupby import GroupBy
        return GroupBy(self._native.groupby(by))

    def merge(self, right, on=None, how="inner", suffixes=("_x", "_y")):
        if isinstance(right, DataFrame):
            right = right._native
        if isinstance(on, str):
            on = [on]
        return DataFrame._from_native(
            self._native.merge(right, on=on, how=how)
        )

    # I/O
    def to_csv(self, path=None):
        if path is None:
            # Write to a temp file and read it back as a string
            import os
            tmp = "/tmp/_pandas_rust_to_csv_tmp.csv"
            self._native.to_csv(tmp)
            with open(tmp) as f:
                result = f.read()
            os.remove(tmp)
            return result
        return self._native.to_csv(path)

    def to_dict(self, orient="dict"):
        return self._native.to_dict()

    def to_numpy(self):
        return self._native.to_numpy()

    @property
    def values(self):
        return self._native.values

    def info(self):
        """Print summary info about the DataFrame."""
        print("<class 'pandas.core.frame.DataFrame'>")
        nrows = self.shape[0]
        if nrows > 0:
            print("RangeIndex: {} entries, 0 to {}".format(nrows, nrows - 1))
        else:
            print("Empty DataFrame")
        print("Data columns (total {} columns):".format(self.shape[1]))
        dtypes_info = self.dtypes
        if isinstance(dtypes_info, dict):
            for col_name, dt in dtypes_info.items():
                print(" {}    {} non-null    {}".format(col_name, nrows, dt))
        print("dtypes: {} columns".format(self.shape[1]))

    def reset_index(self, drop=True):
        """Reset index to RangeIndex. drop=True is always assumed for v1."""
        return self.copy()

    def select_dtypes(self, include=None, exclude=None):
        """Filter columns by dtype string."""
        dtypes_info = self.dtypes
        keep = []
        for col_name, dt in dtypes_info.items():
            # Normalize dtype to category
            if dt in ("int64",):
                cats = {"number", "numeric", "int64", "integer", "int"}
            elif dt in ("float64",):
                cats = {"number", "numeric", "float64", "float", "floating"}
            elif dt in ("object", "str"):
                cats = {"object", "str", "string"}
            elif dt in ("bool",):
                cats = {"bool", "boolean"}
            else:
                cats = {dt}

            if include is not None:
                inc = [include] if isinstance(include, str) else list(include)
                if not any(i in cats for i in inc):
                    continue
            if exclude is not None:
                exc = [exclude] if isinstance(exclude, str) else list(exclude)
                if any(e in cats for e in exc):
                    continue
            keep.append(col_name)

        return self[keep] if keep else DataFrame._from_native(self._native.head(0))

    def nlargest(self, n, columns):
        """Return the n rows with the largest values in the given column."""
        return DataFrame._from_native(self._native.nlargest(n, columns))

    def nsmallest(self, n, columns):
        """Return the n rows with the smallest values in the given column."""
        return DataFrame._from_native(self._native.nsmallest(n, columns))

    def assign(self, **kwargs):
        """Return a new DataFrame with new columns added."""
        result = self.copy()
        for col_name, value in kwargs.items():
            if callable(value):
                value = value(result)
            from .series import Series
            if isinstance(value, Series):
                result[col_name] = value.tolist()
            else:
                result[col_name] = value
        return result

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, *args, **kwargs)."""
        return func(self, *args, **kwargs)

    def abs(self):
        """Return absolute value for numeric columns."""
        return DataFrame._from_native(self._native.abs())

    def clip(self, lower=None, upper=None):
        """Clip values to [lower, upper]."""
        lo = float(lower) if lower is not None else None
        hi = float(upper) if upper is not None else None
        return DataFrame._from_native(self._native.clip(lo, hi))

    def transpose(self):
        """Transpose rows and columns."""
        return DataFrame._from_native(self._native.transpose())

    @property
    def T(self):
        """Transpose property."""
        return self.transpose()

    def iterrows(self):
        """Iterate over rows as (index, dict) pairs."""
        return iter(self._native.iterrows())

    def itertuples(self, index=True, name="Pandas"):
        """Iterate over rows as tuples."""
        return iter(self._native.itertuples(index))

    def apply(self, func, axis=0, **kwargs):
        """Apply function along an axis."""
        if axis == 0:
            result = self._native.apply(func, 0)
            if isinstance(result, dict):
                return result
            return result
        else:
            results = self._native.apply(func, 1)
            from .series import Series
            return Series(results)

    def applymap(self, func):
        """Apply function element-wise."""
        return DataFrame._from_native(self._native.applymap(func))
