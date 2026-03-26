"""DataFrame wrapper for _pandas_native.DataFrame."""
import _pandas_native as _native


class _iLocIndexer:
    """Integer-location based indexer for DataFrame."""

    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = len(self._df) + key
            row = {}
            for col in self._df.columns:
                row[col] = self._df._native.get_column(col).tolist()[key]
            return row
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._df))
            indices = list(range(start, stop, step))
            return self._df._take_rows(indices)
        elif isinstance(key, list):
            return self._df._take_rows(key)
        elif isinstance(key, tuple):
            row_key, col_key = key
            sub = self._df.iloc[row_key]
            if isinstance(sub, dict):
                # single row returned as dict; col_key selects from it
                if isinstance(col_key, int):
                    col_name = self._df.columns[col_key]
                    return sub[col_name]
                elif isinstance(col_key, str):
                    return sub[col_key]
                raise TypeError("Invalid col key for single-row iloc: {}".format(type(col_key)))
            # sub is a DataFrame
            if isinstance(col_key, int):
                col_name = sub.columns[col_key]
                return sub[col_name]
            elif isinstance(col_key, list):
                col_names = [sub.columns[i] for i in col_key]
                return sub[col_names]
            elif isinstance(col_key, slice):
                start, stop, step = col_key.indices(len(sub.columns))
                col_names = [sub.columns[i] for i in range(start, stop, step)]
                return sub[col_names]
        raise TypeError("Invalid iloc key type: {}".format(type(key)))


class _LocIndexer:
    """Label-based indexer for DataFrame."""

    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._df.iloc[key]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop
            if stop is not None:
                stop = stop + 1  # loc is inclusive on stop
            new_slice = slice(start, stop, key.step)
            return self._df.iloc[new_slice]
        elif isinstance(key, list):
            return self._df.iloc[key]
        elif isinstance(key, tuple):
            row_key, col_key = key
            sub = self._df.loc[row_key]
            if isinstance(sub, dict):
                if isinstance(col_key, str):
                    return sub[col_key]
                raise TypeError("Invalid col key for single-row loc: {}".format(type(col_key)))
            if isinstance(col_key, str):
                return sub[col_key]
            elif isinstance(col_key, list):
                return sub[col_key]
        raise TypeError("Invalid loc key type: {}".format(type(key)))


class _RangeIndex:
    """Thin Python wrapper around native PyIndex to support len() and iteration."""

    def __init__(self, native_index):
        self._native = native_index

    def __len__(self):
        return self._native.__len__()

    def __repr__(self):
        return repr(self._native)

    def tolist(self):
        return self._native.tolist()

    def __iter__(self):
        return iter(self.tolist())


class _SeriesiLocIndexer:
    """Integer-location based indexer for Series."""

    def __init__(self, series):
        self._series = series

    def __getitem__(self, key):
        from .series import Series
        data = self._series._native.tolist()
        n = len(data)
        if isinstance(key, int):
            if key < 0:
                key = n + key
            return data[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(n)
            vals = data[start:stop:step]
            return Series(vals, name=self._series.name)
        elif isinstance(key, list):
            vals = [data[i] for i in key]
            return Series(vals, name=self._series.name)
        raise TypeError("Invalid iloc key type for Series: {}".format(type(key)))


class DataFrame:
    """Two-dimensional labeled data structure."""

    def __init__(self, data=None, columns=None, index=None):
        if data is None:
            self._native = _native.DataFrame({})
        elif isinstance(data, dict):
            from .series import Series
            # Check if any values are Series objects and unwrap them
            processed = {}
            for k, v in data.items():
                if isinstance(v, Series):
                    processed[k] = v.tolist()
                else:
                    processed[k] = v
            self._native = _native.DataFrame(processed)
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

    @property
    def index(self):
        return _RangeIndex(self._native.index)

    @property
    def iloc(self):
        return _iLocIndexer(self)

    @property
    def loc(self):
        return _LocIndexer(self)

    def _take_rows(self, indices):
        """Select rows by integer position indices."""
        return DataFrame._from_native(self._native.take_rows(indices))

    def __getitem__(self, key):
        from .series import Series
        # Boolean Series mask → filter rows
        if isinstance(key, Series):
            mask_list = key.tolist()
            indices = [i for i, v in enumerate(mask_list) if v]
            return self._take_rows(indices)
        # Boolean list mask
        if isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
            indices = [i for i, v in enumerate(key) if v]
            return self._take_rows(indices)
        result = self._native[key]
        if isinstance(result, _native.Series):
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
