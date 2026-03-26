"""GroupBy wrapper for _pandas_native.GroupBy."""


class GroupBy:
    """GroupBy object wrapping native PyGroupBy."""

    def __init__(self, native_groupby, parent_df=None, by_cols=None):
        self._native = native_groupby
        self._parent_df = parent_df  # original DataFrame (Python wrapper)
        self._by_cols = by_cols or []

    def __repr__(self):
        return repr(self._native)

    def __len__(self):
        """Return the number of groups."""
        result = self._wrap(self._native.size())
        return len(result)

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

    def transform(self, func):
        """Broadcast aggregated values back to the original DataFrame length.

        Parameters
        ----------
        func : str
            Name of an aggregation function (e.g. "sum", "mean").
        """
        from .frame import DataFrame
        from .series import Series
        if self._parent_df is None:
            raise ValueError("transform requires parent DataFrame reference")

        agg_result = getattr(self, func)()  # returns a DataFrame indexed by group keys

        # Determine value columns (all columns except the by_cols)
        value_cols = [c for c in self._parent_df.columns if c not in self._by_cols]

        # Build lookup: tuple of key values -> dict of {value_col: agg_val}
        agg_key_lists = [agg_result[bc].tolist() for bc in self._by_cols]
        lookup = {}
        n_groups = len(agg_result)
        for i in range(n_groups):
            key = tuple(agg_key_lists[bi][i] for bi in range(len(self._by_cols)))
            lookup[key] = {c: agg_result[c].tolist()[i] for c in value_cols}

        # Map each original row to its group's aggregated value
        parent_key_lists = [self._parent_df[bc].tolist() for bc in self._by_cols]
        n = len(self._parent_df)
        result_cols = {c: [] for c in value_cols}
        for i in range(n):
            key = tuple(parent_key_lists[bi][i] for bi in range(len(self._by_cols)))
            group_vals = lookup.get(key, {c: None for c in value_cols})
            for c in value_cols:
                result_cols[c].append(group_vals.get(c))

        if len(value_cols) == 1:
            return Series(result_cols[value_cols[0]], name=value_cols[0])
        return DataFrame(result_cols)

    def apply(self, func):
        """Apply func to each sub-DataFrame group and combine results."""
        from .frame import DataFrame
        if self._parent_df is None:
            raise ValueError("apply requires parent DataFrame")

        # Get group keys from first() aggregation
        agg_result = self.first()
        results = []
        for i in range(len(agg_result)):
            # Build a boolean mask for this group
            key_mask = None
            for by_col in self._by_cols:
                key_val = agg_result[by_col].tolist()[i]
                col_mask = self._parent_df[by_col] == key_val
                if key_mask is None:
                    key_mask = col_mask
                else:
                    key_mask = key_mask & col_mask
            sub_df = self._parent_df[key_mask]
            result = func(sub_df)
            results.append(result)

        if not results:
            return DataFrame({})
        if isinstance(results[0], DataFrame):
            import pandas as _pd
            return _pd.concat(results)
        # func returned a scalar per group
        return DataFrame({"result": results})

    def filter(self, func):
        """Keep groups where func(sub_df) returns True."""
        from .frame import DataFrame
        if self._parent_df is None:
            raise ValueError("filter requires parent DataFrame")

        keep_indices = []
        agg_result = self.first()

        for i in range(len(agg_result)):
            key_mask = None
            for by_col in self._by_cols:
                key_val = agg_result[by_col].tolist()[i]
                col_mask = self._parent_df[by_col] == key_val
                if key_mask is None:
                    key_mask = col_mask
                else:
                    key_mask = key_mask & col_mask
            sub_df = self._parent_df[key_mask]
            if func(sub_df):
                mask_list = key_mask.tolist()
                for j, m in enumerate(mask_list):
                    if m:
                        keep_indices.append(j)

        return self._parent_df._take_rows(sorted(keep_indices))

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
        self._parent_df = groupby._parent_df
        self._by_cols = groupby._by_cols

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

    def transform(self, func):
        from .series import Series
        from .frame import DataFrame
        if self._parent_df is None:
            raise ValueError("transform requires parent DataFrame reference")

        # Get aggregated result (includes by_cols + value cols)
        agg_result = getattr(self._groupby, func)()

        # Build lookup: group_key_values -> agg_value for each value column
        result_cols = {}
        for col_name in self._columns:
            agg_vals = agg_result[col_name].tolist()
            # Get the by-column values from the aggregated result
            agg_keys = []
            for by_col in self._by_cols:
                agg_keys.append(agg_result[by_col].tolist())

            # Build lookup dict: tuple(key_vals) -> agg_value
            lookup = {}
            for i in range(len(agg_vals)):
                key = tuple(k[i] for k in agg_keys)
                lookup[key] = agg_vals[i]

            # Get by-column values from the parent (original) DataFrame
            parent_keys = []
            for by_col in self._by_cols:
                parent_keys.append(self._parent_df[by_col].tolist())

            # Broadcast aggregated values back to original DataFrame length
            n = len(self._parent_df)
            broadcast = []
            for i in range(n):
                key = tuple(k[i] for k in parent_keys)
                broadcast.append(lookup.get(key))
            result_cols[col_name] = broadcast

        if len(self._columns) == 1:
            return Series(result_cols[self._columns[0]], name=self._columns[0])
        return DataFrame(result_cols)

    def __repr__(self):
        return "_ColumnGroupBy(columns={})".format(self._columns)
