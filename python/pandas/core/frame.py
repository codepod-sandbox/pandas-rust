"""DataFrame wrapper for _pandas_native.DataFrame."""
import _pandas_native as _native


class _iLocIndexer:
    """Integer-location based indexer for DataFrame."""

    def __init__(self, df):
        self._df = df

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_idx = key
            if isinstance(col_idx, int):
                col_name = self._df.columns[col_idx]
            else:
                col_name = col_idx
            col_vals = self._df[col_name].tolist()
            if isinstance(row_idx, int):
                if row_idx < 0:
                    row_idx = len(self._df) + row_idx
                col_vals[row_idx] = value
            self._df[col_name] = col_vals
        elif isinstance(key, int):
            if isinstance(value, dict):
                actual_key = key
                if actual_key < 0:
                    actual_key = len(self._df) + actual_key
                for col_name, val in value.items():
                    col_vals = self._df[col_name].tolist()
                    col_vals[actual_key] = val
                    self._df[col_name] = col_vals

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
                elif isinstance(col_key, list):
                    return {self._df.columns[i]: sub[self._df.columns[i]] for i in col_key}
                elif isinstance(col_key, slice):
                    start, stop, step = col_key.indices(len(self._df.columns))
                    return {self._df.columns[i]: sub[self._df.columns[i]] for i in range(start, stop, step)}
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

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row_label, col_name = key
            col_vals = self._df[col_name].tolist()
            col_vals[row_label] = value
            self._df[col_name] = col_vals
        elif isinstance(key, int):
            if isinstance(value, dict):
                for col_name, val in value.items():
                    col_vals = self._df[col_name].tolist()
                    col_vals[key] = val
                    self._df[col_name] = col_vals

    def __getitem__(self, key):
        from .series import Series
        if isinstance(key, Series):
            # Boolean mask filtering
            mask = key.tolist()
            indices = [i for i, m in enumerate(mask) if m]
            return self._df._take_rows(indices)
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
            # List of booleans — treat as mask
            if key and isinstance(key[0], bool):
                indices = [i for i, m in enumerate(key) if m]
                return self._df._take_rows(indices)
            return self._df.iloc[key]
        elif isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, Series):
                # loc[bool_mask, col_key]
                sub = self._df.loc[row_key]
                if isinstance(col_key, str):
                    return sub[col_key]
                elif isinstance(col_key, list):
                    return sub[col_key]
                return sub
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


class _ColumnIndex(list):
    """A list subclass with .tolist() for pandas compatibility."""
    def tolist(self):
        return list(self)


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
        # _py_cols stores columns whose values can't be held by native (e.g., list-of-lists)
        self._py_cols = {}
        if data is None:
            self._native = _native.DataFrame({})
        elif isinstance(data, dict):
            from .series import Series
            # Check if any values are Series objects and unwrap them
            processed = {}
            for k, v in data.items():
                if isinstance(v, Series):
                    processed[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)):
                    # Scalar value — broadcast to index length
                    if index is not None:
                        processed[k] = [v] * len(index)
                    else:
                        processed[k] = [v]
                else:
                    processed[k] = v
            # Separate columns that contain list-of-lists from native-compatible columns
            native_processed = {}
            for k, v in processed.items():
                if isinstance(v, list) and v and isinstance(v[0], list):
                    self._py_cols[k] = v
                else:
                    native_processed[k] = v
            self._native = _native.DataFrame(native_processed)
        elif isinstance(data, _native.DataFrame):
            self._native = data
        elif hasattr(data, 'shape') and hasattr(data, 'tolist'):
            # numpy ndarray or similar
            rows = data.tolist()
            if not rows:
                self._native = _native.DataFrame({})
            else:
                if not isinstance(rows[0], list):
                    rows = [[v] for v in rows]  # 1D array
                ncols = len(rows[0]) if rows else 0
                if columns is None:
                    columns = [str(i) for i in range(ncols)]
                col_data = {c: [] for c in columns}
                for row in rows:
                    for j, c in enumerate(columns):
                        col_data[c].append(row[j] if j < len(row) else None)
                self._native = _native.DataFrame(col_data)
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
            elif isinstance(data[0], (list, tuple)):
                # List of lists/tuples — each inner list is a row
                ncols = len(data[0])
                if columns is None:
                    columns = [str(i) for i in range(ncols)]
                col_data = {c: [] for c in columns}
                for row in data:
                    for i, c in enumerate(columns):
                        col_data[c].append(row[i] if i < len(row) else None)
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
        obj._py_cols = {}
        return obj

    @property
    def shape(self):
        native_shape = self._native.shape
        total_cols = native_shape[1] + len(self._py_cols)
        nrows = native_shape[0]
        if nrows == 0 and self._py_cols:
            first = next(iter(self._py_cols.values()))
            nrows = len(first)
        return (nrows, total_cols)

    @property
    def dtypes(self):
        d = dict(self._native.dtypes)
        for k in self._py_cols:
            d[k] = "object"
        return d

    @property
    def columns(self):
        native_cols = list(self._native.columns)
        py_col_keys = [k for k in self._py_cols if k not in native_cols]
        return _ColumnIndex(native_cols + py_col_keys)

    @columns.setter
    def columns(self, new_columns):
        """Rename columns in place."""
        old_cols = list(self._native.columns)
        if len(new_columns) != len(old_cols):
            raise ValueError("Length mismatch: expected {} columns, got {}".format(
                len(old_cols), len(new_columns)))
        mapping = dict(zip(old_cols, new_columns))
        self._native = self._native.rename(mapping)

    def __len__(self):
        native_rows = self._native.shape[0]
        if native_rows == 0 and self._py_cols:
            # All columns are py_cols
            first = next(iter(self._py_cols.values()))
            return len(first)
        return native_rows

    def __iter__(self):
        return iter(self.columns)

    def __repr__(self):
        return repr(self._native)

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails
        try:
            return self[name]
        except (KeyError, TypeError):
            raise AttributeError("'DataFrame' object has no attribute '{}'".format(name))

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
        result = DataFrame._from_native(self._native.take_rows(indices))
        for k, v in self._py_cols.items():
            result._py_cols[k] = [v[i] for i in indices]
        return result

    def __getitem__(self, key):
        from .series import Series, _PythonSeries
        # Boolean Series mask → filter rows
        if isinstance(key, Series):
            mask_list = key.tolist()
            indices = [i for i, v in enumerate(mask_list) if v]
            return self._take_rows(indices)
        # Boolean list mask
        if isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
            indices = [i for i, v in enumerate(key) if v]
            return self._take_rows(indices)
        # Single column name — check py_cols first
        if isinstance(key, str) and key in self._py_cols:
            return _PythonSeries(self._py_cols[key], name=key)
        # List of column names
        if isinstance(key, list):
            result_df = DataFrame.__new__(DataFrame)
            result_df._py_cols = {}
            native_keys = [k for k in key if k not in self._py_cols]
            if native_keys:
                native_sub = self._native[native_keys]
                if isinstance(native_sub, _native.DataFrame):
                    result_df._native = native_sub
                else:
                    result_df._native = _native.DataFrame({native_keys[0]: native_sub.tolist()})
            else:
                result_df._native = _native.DataFrame({})
            for k in key:
                if k in self._py_cols:
                    result_df._py_cols[k] = self._py_cols[k]
            return result_df
        result = self._native[key]
        if isinstance(result, _native.Series):
            return Series._from_native(result)
        elif isinstance(result, _native.DataFrame):
            r = DataFrame._from_native(result)
            r._py_cols = {}
            return r
        return result

    def __setitem__(self, key, value):
        from .series import Series, _PythonSeries
        if isinstance(value, _PythonSeries):
            self._py_cols[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], list):
            self._py_cols[key] = value
        elif isinstance(value, Series):
            self._native[key] = value.tolist()
        else:
            self._native[key] = value

    def head(self, n=5):
        if n < 0:
            return self.iloc[:len(self) + n]
        return DataFrame._from_native(self._native.head(n))

    def tail(self, n=5):
        if n < 0:
            return self.iloc[-n:]
        if n == 0:
            return self.iloc[0:0]
        return DataFrame._from_native(self._native.tail(n))

    def sort_values(self, by, ascending=True, inplace=False, **kwargs):
        # Pass ascending directly (Rust now handles bool, list of bools)
        # Handle empty by list — return copy unchanged
        if isinstance(by, list) and len(by) == 0:
            result = self.copy()
        else:
            result = DataFrame._from_native(self._native.sort_values(by, ascending))
        if inplace:
            self._native = result._native
            return None
        return result

    def __contains__(self, key):
        return key in self.columns

    def drop(self, columns=None, **kwargs):
        if columns is None:
            columns = kwargs.get('labels', [])
        if isinstance(columns, str):
            columns = [columns]
        return DataFrame._from_native(self._native.drop(columns))

    def rename(self, columns=None, **kwargs):
        if columns is None:
            return self.copy()
        if callable(columns):
            mapping = {c: columns(c) for c in self.columns}
            return DataFrame._from_native(self._native.rename(mapping))
        return DataFrame._from_native(self._native.rename(columns))

    def copy(self):
        result = DataFrame._from_native(self._native.copy())
        result._py_cols = {k: list(v) for k, v in self._py_cols.items()}
        return result

    def astype(self, dtype):
        result = self.copy()
        if isinstance(dtype, dict):
            for col, dt in dtype.items():
                from .series import Series
                s = result[col].astype(dt)
                result[col] = s.tolist()
        return result

    def sort_index(self, ascending=True):
        """Sort by index. For RangeIndex (default), data is already sorted; return a copy."""
        return self.copy()

    def stack(self):
        """Flatten columns to a long-format Series."""
        from .series import Series
        result_vals = []
        for i in range(len(self)):
            for col in self.columns:
                result_vals.append(self[col].tolist()[i])
        return Series(result_vals, name=None)

    def rename_axis(self, name, axis=0):
        """Set the name of the index or columns axis (no-op — named axes not yet supported)."""
        return self.copy()

    # Aggregations
    def _row_agg(self, func, name):
        """Helper for axis=1 (row-wise) aggregation."""
        from .series import Series
        numeric_cols = [c for c in self.columns if self.dtypes[c] in ("int64", "float64")]
        col_vals = {c: self[c].tolist() for c in numeric_cols}
        n = len(self)
        vals = []
        for i in range(n):
            row = [col_vals[c][i] for c in numeric_cols if col_vals[c][i] is not None]
            if row:
                vals.append(func(row))
            else:
                vals.append(None)
        return Series(vals, name=name)

    def sum(self, axis=0):
        if axis == 1:
            return self._row_agg(sum, "sum")
        return self._native.sum()

    def mean(self, axis=0):
        if axis == 1:
            return self._row_agg(lambda r: sum(r) / len(r), "mean")
        return self._native.mean()

    def min(self, axis=0):
        if axis == 1:
            return self._row_agg(min, "min")
        return self._native.min()

    def max(self, axis=0):
        if axis == 1:
            return self._row_agg(max, "max")
        return self._native.max()

    def count(self):
        return self._native.count()

    def std(self, ddof=1, axis=0):
        if axis == 1:
            import math
            return self._row_agg(
                lambda r: math.sqrt(sum((x - sum(r) / len(r)) ** 2 for x in r) / (len(r) - 1)) if len(r) > 1 else 0.0,
                "std",
            )
        return self._native.std()

    def var(self, ddof=1, axis=0):
        if axis == 1:
            return self._row_agg(
                lambda r: sum((x - sum(r) / len(r)) ** 2 for x in r) / (len(r) - 1) if len(r) > 1 else 0.0,
                "var",
            )
        return self._native.var()

    def median(self, axis=0):
        if axis == 1:
            def _median(r):
                s = sorted(r)
                n = len(s)
                if n % 2 == 1:
                    return s[n // 2]
                return (s[n // 2 - 1] + s[n // 2]) / 2
            return self._row_agg(_median, "median")
        return self._native.median()

    def cov(self):
        """Compute pairwise covariance of columns."""
        numeric_cols = [c for c in self.columns if self.dtypes[c] in ("int64", "float64")]
        n = len(self)
        col_vals = {c: self[c].tolist() for c in numeric_cols}
        means = {c: sum(v for v in col_vals[c] if v is not None) / n for c in numeric_cols}
        result = {}
        for c1 in numeric_cols:
            row = []
            for c2 in numeric_cols:
                cov_sum = sum(
                    (col_vals[c1][i] - means[c1]) * (col_vals[c2][i] - means[c2])
                    for i in range(n)
                    if col_vals[c1][i] is not None and col_vals[c2][i] is not None
                )
                row.append(cov_sum / (n - 1) if n > 1 else 0.0)
            result[c1] = row
        return DataFrame(result)

    def corrwith(self, other):
        """Compute pairwise correlation with another DataFrame."""
        import math
        result = {}
        for col in self.columns:
            if col in other.columns:
                vals1 = self[col].tolist()
                vals2 = other[col].tolist()
                n = min(len(vals1), len(vals2))
                if n < 2:
                    result[col] = None
                    continue
                mean1 = sum(vals1[:n]) / n
                mean2 = sum(vals2[:n]) / n
                cov = sum((vals1[i] - mean1) * (vals2[i] - mean2) for i in range(n)) / (n - 1)
                std1 = math.sqrt(sum((v - mean1) ** 2 for v in vals1[:n]) / (n - 1))
                std2 = math.sqrt(sum((v - mean2) ** 2 for v in vals2[:n]) / (n - 1))
                result[col] = cov / (std1 * std2) if std1 > 0 and std2 > 0 else None
        return result

    @classmethod
    def from_dict(cls, data, orient="columns"):
        """Construct DataFrame from dict."""
        return cls(data)

    def describe(self):
        return DataFrame._from_native(self._native.describe())

    # Null ops
    def fillna(self, value):
        # Coerce int scalar to float for float columns
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        return DataFrame._from_native(self._native.fillna(value))

    def dropna(self, how="any", subset=None, **kwargs):
        if subset is not None:
            # Filter using Python logic on the specified subset of columns
            if isinstance(subset, str):
                subset = [subset]
            indices = []
            for i in range(len(self)):
                row = self.iloc[i]
                if how == "any":
                    if all(row.get(col) is not None for col in subset):
                        indices.append(i)
                else:  # "all"
                    if any(row.get(col) is not None for col in subset):
                        indices.append(i)
            return self._take_rows(indices)
        return DataFrame._from_native(self._native.dropna(how))

    def isna(self):
        return DataFrame._from_native(self._native.isna())

    def notna(self):
        return DataFrame._from_native(self._native.notna())

    def duplicated(self, subset=None, keep="first"):
        from .series import Series
        result = self._native.duplicated(subset, keep)
        return Series._from_native(result)

    def drop_duplicates(self, subset=None, keep="first", inplace=False, ignore_index=False):
        result = DataFrame._from_native(self._native.drop_duplicates(subset, keep))
        if inplace:
            self._native = result._native
            return None
        return result

    # GroupBy & Merge
    def groupby(self, by, sort=True, as_index=True, **kwargs):
        # sort and as_index accepted but not yet implemented
        if isinstance(by, str):
            by = [by]
        from .groupby import GroupBy
        return GroupBy(self._native.groupby(by), parent_df=self, by_cols=by)

    def merge(self, right, on=None, how="inner", left_on=None, right_on=None, suffixes=("_x", "_y"), indicator=False):
        if left_on is not None and right_on is not None:
            if isinstance(left_on, str):
                left_on = [left_on]
            if isinstance(right_on, str):
                right_on = [right_on]
            # Rename right key columns to match left key names so native merge works
            rename_map = {}
            for lk, rk in zip(left_on, right_on):
                if lk != rk:
                    rename_map[rk] = lk
            if rename_map:
                if isinstance(right, DataFrame):
                    right = right.rename(columns=rename_map)
                else:
                    # native DataFrame — wrap, rename, unwrap
                    right = DataFrame._from_native(right).rename(columns=rename_map)
            on = left_on

        if isinstance(right, DataFrame):
            right = right._native
        if isinstance(on, str):
            on = [on]
        return DataFrame._from_native(
            self._native.merge(right, on=on, how=how)
        )

    # I/O
    def to_csv(self, path=None, index=True, **kwargs):
        if path is None:
            # Write to a temp file and read it back as a string
            import os
            tmp = "/tmp/_pandas_rust_to_csv_tmp.csv"
            self._native.to_csv(tmp)
            with open(tmp) as f:
                result = f.read()
            os.remove(tmp)
            # index param is accepted but our native CSV has no index column anyway
            return result
        return self._native.to_csv(path)

    def isin(self, values):
        """Check whether each element is in values (list or dict)."""
        result = {}
        if isinstance(values, dict):
            for col in self.columns:
                if col in values:
                    result[col] = self[col].isin(values[col]).tolist()
                else:
                    result[col] = [False] * len(self)
        elif isinstance(values, list):
            for col in self.columns:
                result[col] = self[col].isin(values).tolist()
        else:
            for col in self.columns:
                result[col] = self[col].isin(list(values)).tolist()
        return DataFrame(result)

    def replace(self, to_replace, value=None):
        """Replace values in the DataFrame."""
        result = {}
        for col in self.columns:
            vals = self[col].tolist()
            if isinstance(to_replace, dict):
                if col in to_replace:
                    mapping = to_replace[col] if isinstance(to_replace[col], dict) else {to_replace[col]: value}
                    result[col] = [mapping.get(v, v) for v in vals]
                else:
                    result[col] = vals[:]
            else:
                result[col] = [value if v == to_replace else v for v in vals]
        return DataFrame(result)

    def to_dict(self, orient="dict"):
        if orient == "records":
            n = len(self)
            records = []
            col_vals = {col: self[col].tolist() for col in self.columns}
            for i in range(n):
                row = {col: col_vals[col][i] for col in self.columns}
                records.append(row)
            return records
        # "dict" and "list" both return the native {col: [values]} format
        return self._native.to_dict()

    def to_numpy(self):
        return self._native.to_numpy()

    @property
    def values(self):
        return self._native.values

    # --- Properties: size, ndim, empty ---

    @property
    def size(self):
        """Total number of elements (rows * columns)."""
        s = self.shape
        return s[0] * s[1]

    @property
    def ndim(self):
        """Number of dimensions (always 2 for DataFrame)."""
        return 2

    @property
    def empty(self):
        """True if DataFrame has no rows or no columns."""
        s = self.shape
        return s[0] == 0 or s[1] == 0

    # --- filter ---

    def filter(self, items=None, like=None, regex=None, axis=1):
        """Subset rows or columns according to labels."""
        if items is not None:
            cols = [c for c in items if c in self.columns]
            return self[cols]
        elif like is not None:
            cols = [c for c in self.columns if like in c]
            return self[cols]
        elif regex is not None:
            import re
            pat = re.compile(regex)
            cols = [c for c in self.columns if pat.search(c)]
            return self[cols]
        raise TypeError("Must pass either `items`, `like`, or `regex`")

    # --- pop ---

    def pop(self, item):
        """Remove and return column as a Series."""
        result = self[item]
        cols = [c for c in self.columns if c != item]
        new_data = {c: self[c].tolist() for c in cols}
        self._native = _native.DataFrame(new_data)
        return result

    # --- add_prefix / add_suffix ---

    def add_prefix(self, prefix):
        """Prefix column labels with string prefix."""
        return self.rename(columns={c: str(prefix) + c for c in self.columns})

    def add_suffix(self, suffix):
        """Suffix column labels with string suffix."""
        return self.rename(columns={c: c + str(suffix) for c in self.columns})

    # --- to_string ---

    def to_string(self):
        """Render DataFrame to a console-friendly tabular output."""
        return repr(self)

    # --- axes ---

    @property
    def axes(self):
        """Return list of axes [row_index, columns]."""
        return [self.index, self.columns]

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

    def diff(self, periods=1, axis=0):
        """Difference between consecutive rows (axis=0) for each numeric column."""
        from .series import Series
        if not isinstance(periods, int):
            raise ValueError("periods must be an integer")
        result = {}
        for col in self.columns:
            result[col] = self[col].diff(periods).tolist()
        return DataFrame(result, index=self.index)

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
        # Pure Python implementation to avoid native transpose bugs
        nrows = len(self)
        col_names = list(self.columns)
        # New columns are "0", "1", ..., "nrows-1"
        new_col_names = [str(i) for i in range(nrows)]
        result = {}
        for new_col_idx in range(nrows):
            vals = []
            for old_col in col_names:
                vals.append(self[old_col].tolist()[new_col_idx])
            result[new_col_names[new_col_idx]] = vals
        return DataFrame(result)

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

    # pandas 2.1+ alias: DataFrame.map is element-wise (was applymap)
    map = applymap

    def items(self):
        """Iterate over (column name, Series) pairs."""
        for col in self.columns:
            yield col, self[col]

    # deprecated alias
    iteritems = items

    def prod(self, axis=0):
        """Return product of numeric values per column."""
        if axis == 1:
            def _prod(r):
                p = 1
                for v in r:
                    p *= v
                return p
            return self._row_agg(_prod, "prod")
        result = {}
        for col in self.columns:
            try:
                vals = [v for v in self[col].tolist() if v is not None]
                p = 1
                for v in vals:
                    p *= v
                result[col] = p
            except TypeError:
                pass
        return result

    product = prod

    def cumsum(self):
        """Cumulative sum per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].cumsum().tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def cummax(self):
        """Cumulative max per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].cummax().tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def cummin(self):
        """Cumulative min per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].cummin().tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def cumprod(self):
        """Cumulative product per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].cumprod().tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def shift(self, periods=1):
        """Shift each column by periods positions."""
        result = {}
        for col in self.columns:
            result[col] = self[col].shift(periods).tolist()
        return DataFrame(result)

    def pct_change(self, periods=1):
        """Percentage change per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].pct_change(periods).tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def rank(self, method="average", ascending=True):
        """Rank values per column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].rank(method=method, ascending=ascending).tolist()
            except TypeError:
                pass
        return DataFrame(result)

    def reindex(self, columns=None, **kwargs):
        """Reorder/add/drop columns. Missing columns filled with None."""
        if columns is not None:
            data = {}
            for col in columns:
                if col in self.columns:
                    data[col] = self[col].tolist()
                else:
                    data[col] = [None] * len(self)
            return DataFrame(data)
        return self.copy()

    def eq(self, other):
        return self._cmp_op(other, '__eq__')

    def ne(self, other):
        return self._cmp_op(other, '__ne__')

    def lt(self, other):
        return self._cmp_op(other, '__lt__')

    def le(self, other):
        return self._cmp_op(other, '__le__')

    def gt(self, other):
        return self._cmp_op(other, '__gt__')

    def ge(self, other):
        return self._cmp_op(other, '__ge__')

    # --- Comparison operators ---
    def _cmp_op(self, other, op):
        result = {}
        for col in self.columns:
            result[col] = getattr(self[col], op)(other[col] if isinstance(other, DataFrame) else other).tolist()
        return DataFrame(result)

    def __eq__(self, other):
        return self._cmp_op(other, '__eq__')

    def __ne__(self, other):
        return self._cmp_op(other, '__ne__')

    def __lt__(self, other):
        return self._cmp_op(other, '__lt__')

    def __le__(self, other):
        return self._cmp_op(other, '__le__')

    def __gt__(self, other):
        return self._cmp_op(other, '__gt__')

    def __ge__(self, other):
        return self._cmp_op(other, '__ge__')

    # --- sample ---
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
        return self._take_rows(indices)

    # --- corr ---
    def corr(self):
        import math
        numeric_cols = [c for c in self.columns if self.dtypes[c] in ("int64", "float64")]
        n = len(self)
        result = {}
        for c1 in numeric_cols:
            vals1 = self[c1].tolist()
            mean1 = sum(v for v in vals1 if v is not None) / n
            row = []
            for c2 in numeric_cols:
                vals2 = self[c2].tolist()
                mean2 = sum(v for v in vals2 if v is not None) / n
                cov = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(vals1, vals2)) / (n - 1)
                std1 = math.sqrt(sum((v - mean1) ** 2 for v in vals1) / (n - 1))
                std2 = math.sqrt(sum((v - mean2) ** 2 for v in vals2) / (n - 1))
                if std1 == 0 or std2 == 0:
                    row.append(float('nan'))
                else:
                    row.append(cov / (std1 * std2))
            result[c1] = row
        return DataFrame(result)

    def set_index(self, keys, drop=True):
        """Set the DataFrame index using existing columns."""
        if isinstance(keys, str):
            keys = [keys]
        result = self.copy()
        if drop:
            for k in keys:
                result = result.drop(columns=[k])
        return result

    def where(self, cond, other=None):
        """Keep values where cond is True, replace False positions with other."""
        from .series import Series
        if isinstance(cond, Series):
            mask = cond.tolist()
            result = {}
            for col in self.columns:
                vals = self[col].tolist()
                result[col] = [v if m else other for v, m in zip(vals, mask)]
            return DataFrame(result)
        elif isinstance(cond, DataFrame):
            result = {}
            for col in self.columns:
                cond_vals = cond[col].tolist()
                vals = self[col].tolist()
                result[col] = [v if m else other for v, m in zip(vals, cond_vals)]
            return DataFrame(result)
        else:
            # Assume iterable mask
            mask = list(cond)
            result = {}
            for col in self.columns:
                vals = self[col].tolist()
                result[col] = [v if m else other for v, m in zip(vals, mask)]
            return DataFrame(result)

    def equals(self, other):
        """Test whether two DataFrames are identical element-by-element (NaN-safe)."""
        if not isinstance(other, DataFrame):
            return False
        if self.shape != other.shape:
            return False
        if list(self.columns) != list(other.columns):
            return False
        for col in self.columns:
            l = self[col].tolist()
            r = other[col].tolist()
            for lv, rv in zip(l, r):
                if lv is None and rv is None:
                    continue
                if isinstance(lv, float) and isinstance(rv, float):
                    if lv != lv and rv != rv:  # both NaN
                        continue
                if lv != rv:
                    return False
        return True

    def round(self, decimals=0):
        """Round numeric columns to given decimal places."""
        return DataFrame._from_native(self._native.round(decimals))

    def nunique(self, axis=0):
        """Return number of unique values per column as a dict."""
        result = {}
        for col in self.columns:
            result[col] = self[col].nunique()
        return result

    def value_counts(self, subset=None, normalize=False, sort=True, ascending=False, dropna=True):
        """Return a Series containing counts of unique rows."""
        from .series import Series
        cols = subset if subset is not None else list(self.columns)
        if isinstance(cols, str):
            cols = [cols]
        # Build tuples for each row (across selected columns)
        rows = []
        for i in range(len(self)):
            row_key = tuple(self[c].tolist()[i] for c in cols)
            rows.append(row_key)
        counts = {}
        for key in rows:
            if dropna and any(v is None for v in key):
                continue
            counts[key] = counts.get(key, 0) + 1
        if sort:
            sorted_keys = sorted(counts, key=lambda k: counts[k], reverse=not ascending)
        else:
            sorted_keys = list(counts.keys())
        total = sum(counts.values()) if normalize else None
        result_vals = []
        for key in sorted_keys:
            v = counts[key]
            result_vals.append(v / total if normalize else v)
        # Return as a Series with tuple keys as labels (simplified: return counts dict)
        # For compatibility, return as Series of count values
        if len(cols) == 1:
            # Single column: keys are single values
            result = {key[0]: val for key, val in zip(sorted_keys, result_vals)}
        else:
            result = {key: val for key, val in zip(sorted_keys, result_vals)}
        return result

    def pivot_table(self, values=None, index=None, columns=None, aggfunc="mean"):
        if index is None:
            raise ValueError("index is required")
        if isinstance(index, str):
            index = [index]
        if isinstance(values, str):
            values = [values]
        g = self.groupby(index)
        if values:
            g = g[values]
        result = getattr(g, aggfunc)()
        # If result is a Series (single-value column selected), wrap in DataFrame
        from .series import Series
        if isinstance(result, Series):
            return DataFrame({result.name: result.tolist()})
        return result

    def agg(self, func):
        if isinstance(func, str):
            return getattr(self, func)()
        elif isinstance(func, list):
            # List of functions -> DataFrame with function names as rows
            result = {}
            for col in self.columns:
                col_results = []
                for f in func:
                    val = getattr(self[col], f)()
                    # Coerce all numeric results to float for type uniformity
                    if isinstance(val, int) and not isinstance(val, bool):
                        val = float(val)
                    col_results.append(val)
                result[col] = col_results
            return DataFrame(result)
        elif isinstance(func, dict):
            result = {}
            for col, f in func.items():
                if isinstance(f, str):
                    result[col] = [getattr(self[col], f)()]
                elif isinstance(f, list):
                    result[col] = [getattr(self[col], fn)() for fn in f]
            return DataFrame(result)
        raise TypeError("agg expects str, list, or dict")

    def aggregate(self, func):
        return self.agg(func)

    def join(self, other, how="left", **kwargs):
        """Join two DataFrames horizontally (by index)."""
        import pandas as pd
        if isinstance(other, DataFrame):
            return pd.concat([self, other], axis=1)
        raise TypeError("Can only join DataFrame objects")

    # --- Arithmetic operators ---
    def _arith_op(self, other, op):
        """Apply element-wise arithmetic between two DataFrames or DataFrame and scalar."""
        result = {}
        if isinstance(other, DataFrame):
            for col in self.columns:
                result[col] = getattr(self[col], op)(other[col]).tolist()
        else:
            for col in self.columns:
                result[col] = getattr(self[col], op)(other).tolist()
        return DataFrame(result)

    def __add__(self, other):
        return self._arith_op(other, '__add__')

    def __sub__(self, other):
        return self._arith_op(other, '__sub__')

    def __mul__(self, other):
        return self._arith_op(other, '__mul__')

    def __truediv__(self, other):
        return self._arith_op(other, '__truediv__')

    def __radd__(self, other):
        return self._arith_op(other, '__radd__')

    def __rmul__(self, other):
        return self._arith_op(other, '__rmul__')

    def mask(self, cond, other=None):
        """Replace values where cond is True with other (opposite of where)."""
        from .series import Series
        if isinstance(cond, Series):
            mask_vals = cond.tolist()
            result = {}
            for col in self.columns:
                vals = self[col].tolist()
                result[col] = [other if m else v for v, m in zip(vals, mask_vals)]
            return DataFrame(result)
        elif isinstance(cond, DataFrame):
            result = {}
            for col in self.columns:
                cond_vals = cond[col].tolist()
                vals = self[col].tolist()
                result[col] = [other if m else v for v, m in zip(vals, cond_vals)]
            return DataFrame(result)
        else:
            mask_vals = list(cond)
            result = {}
            for col in self.columns:
                vals = self[col].tolist()
                result[col] = [other if m else v for v, m in zip(vals, mask_vals)]
            return DataFrame(result)

    def any(self, axis=0):
        """Return True for each column if any value is truthy."""
        from .series import Series
        result = {}
        for col in self.columns:
            result[col] = [any(v for v in self[col].tolist() if v is not None)]
        if axis == 0:
            return Series([any(v for v in self[col].tolist() if v is not None) for col in self.columns],
                          name=None)
        return Series([any(v for v in self.iloc[i].values() if v) for i in range(len(self))],
                      name=None)

    def all(self, axis=0):
        """Return True for each column if all values are truthy."""
        from .series import Series
        if axis == 0:
            return Series([all(bool(v) for v in self[col].tolist() if v is not None) for col in self.columns],
                          name=None)
        return Series([all(bool(v) for v in self.iloc[i].values()) for i in range(len(self))],
                      name=None)

    def update(self, other):
        """Modify in-place using non-NA values from another DataFrame (aligned by index/column)."""
        if isinstance(other, DataFrame):
            for col in other.columns:
                if col in self.columns:
                    other_vals = other[col].tolist()
                    self_vals = self[col].tolist()
                    n = min(len(self_vals), len(other_vals))
                    for i in range(n):
                        if other_vals[i] is not None:
                            self_vals[i] = other_vals[i]
                    self[col] = self_vals

    def query(self, expr):
        """Filter rows by a string expression like 'col > 2' or 'a > 1 and b < 5'.

        Supports: col op value, col op col, and/or connectors.
        """
        import builtins
        # Keep 'and'/'or' as-is (Python keywords work fine in eval for booleans)
        mask = []
        safe_builtins = {"True": True, "False": False}
        for i in range(len(self)):
            row_vars = {col: self[col].tolist()[i] for col in self.columns}
            try:
                code = builtins.compile(expr, "<query>", "eval")
                result = builtins.eval(code, {"__builtins__": safe_builtins}, row_vars)
                mask.append(bool(result))
            except Exception:
                mask.append(False)
        indices = [i for i, m in enumerate(mask) if m]
        return self._take_rows(indices)

    def set_axis(self, labels, axis=0):
        """Set the axis labels. axis=1 renames columns."""
        if axis == 1 or axis == "columns":
            mapping = dict(zip(self.columns, labels))
            return self.rename(columns=mapping)
        return self.copy()  # axis=0 (index) not fully supported

    def fillna(self, value):
        """Fill NA values. value may be scalar or dict mapping column->fill."""
        if isinstance(value, dict):
            result = self.copy()
            for col, fill_val in value.items():
                if col in result.columns:
                    # Coerce int fill to float if the column dtype is float
                    col_dtype = result.dtypes.get(col, "") if isinstance(result.dtypes, dict) else ""
                    if col_dtype == "float64" and isinstance(fill_val, int) and not isinstance(fill_val, bool):
                        fill_val = float(fill_val)
                    vals = result[col].tolist()
                    filled = [fill_val if v is None else v for v in vals]
                    result[col] = filled
            return result
        # Coerce int scalar to float to avoid dtype mismatch with float columns
        scalar = float(value) if isinstance(value, int) and not isinstance(value, bool) else value
        return DataFrame._from_native(self._native.fillna(scalar))

    def idxmax(self, axis=0):
        """Return index of maximum value for each column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].idxmax()
            except (TypeError, AttributeError):
                pass
        return result

    def idxmin(self, axis=0):
        """Return index of minimum value for each column."""
        result = {}
        for col in self.columns:
            try:
                result[col] = self[col].idxmin()
            except (TypeError, AttributeError):
                pass
        return result

    def rolling(self, window):
        """Return a _DataFrameRolling object for window calculations."""
        return _DataFrameRolling(self, window)

    def expanding(self):
        """Return a _DataFrameExpanding object for expanding window calculations."""
        return _DataFrameExpanding(self)

    def explode(self, column):
        """Expand list-like values in column into rows."""
        from .series import Series
        col_vals = self[column].tolist()
        other_cols = [c for c in self.columns if c != column]

        new_data = {c: [] for c in self.columns}
        for i, val in enumerate(col_vals):
            if isinstance(val, list):
                for v in val:
                    new_data[column].append(v)
                    for c in other_cols:
                        new_data[c].append(self[c].tolist()[i])
            else:
                new_data[column].append(val)
                for c in other_cols:
                    new_data[c].append(self[c].tolist()[i])
        return DataFrame(new_data)


class _DataFrameRolling:
    """Rolling window calculations for DataFrame."""

    def __init__(self, df, window):
        self._df = df
        self._window = window

    def _apply(self, method):
        result = {}
        for col in self._df.columns:
            try:
                r = getattr(self._df[col].rolling(self._window), method)()
                result[col] = r.tolist()
            except (TypeError, AttributeError):
                pass
        return DataFrame(result)

    def mean(self): return self._apply("mean")
    def sum(self): return self._apply("sum")
    def min(self): return self._apply("min")
    def max(self): return self._apply("max")
    def std(self): return self._apply("std")
    def count(self): return self._apply("count")


class _DataFrameExpanding:
    """Expanding window calculations for DataFrame."""

    def __init__(self, df):
        self._df = df

    def _apply(self, method):
        result = {}
        for col in self._df.columns:
            try:
                r = getattr(self._df[col].expanding(), method)()
                result[col] = r.tolist()
            except (TypeError, AttributeError):
                pass
        return DataFrame(result)

    def sum(self): return self._apply("sum")
    def mean(self): return self._apply("mean")
    def min(self): return self._apply("min")
    def max(self): return self._apply("max")
    def count(self): return self._apply("count")
