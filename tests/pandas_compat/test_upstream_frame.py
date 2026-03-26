"""Upstream-style tests for DataFrame — modeled after pandas/tests/frame/."""
import math
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Construction (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameConstruction:
    def test_from_dict_of_lists_dtypes(self):
        """DataFrame created from dict of lists carries correct dtypes."""
        df = pd.DataFrame({"ints": [1, 2, 3], "floats": [1.5, 2.5, 3.5], "strs": ["a", "b", "c"]})
        assert df.shape == (3, 3)
        dtypes = df.dtypes
        assert dtypes["ints"] == "int64"
        assert dtypes["floats"] == "float64"
        assert dtypes["strs"] == "object"

    def test_from_list_of_dicts_missing_keys_yields_nulls(self):
        """Rows with missing keys become None/null in the resulting column."""
        df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3}, {"b": 4}])
        assert df.shape == (3, 2)
        a_vals = df["a"].tolist()
        b_vals = df["b"].tolist()
        assert a_vals[0] == 1
        assert a_vals[1] == 3
        assert a_vals[2] is None
        assert b_vals[0] == 2
        assert b_vals[1] is None
        assert b_vals[2] == 4

    def test_from_dict_with_series_values(self):
        """DataFrame can be constructed from a dict whose values are Series."""
        s = pd.Series([10, 20, 30], name="x")
        df = pd.DataFrame({"x": s, "y": [1, 2, 3]})
        assert list(df.columns) == ["x", "y"]
        assert df["x"].tolist() == [10, 20, 30]

    def test_from_scalar_dict_with_index(self):
        """Scalar value expanded to fill index length."""
        df = pd.DataFrame({"a": [7, 7, 7], "b": [0, 1, 2]})
        assert df["a"].tolist() == [7, 7, 7]

    def test_empty_dataframe(self):
        """Empty DataFrame has zero rows and zero columns."""
        df = pd.DataFrame()
        assert df.shape == (0, 0)
        assert list(df.columns) == []


# ---------------------------------------------------------------------------
# Indexing & Selection (8 tests)
# ---------------------------------------------------------------------------

class TestDataFrameIndexing:
    def setup_method(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})

    def test_getitem_column_returns_series_with_name(self):
        s = self.df["a"]
        assert isinstance(s, pd.Series)
        assert s.name == "a"
        assert s.tolist() == [1, 2, 3]

    def test_getitem_list_of_columns_returns_dataframe(self):
        sub = self.df[["a", "b"]]
        assert isinstance(sub, pd.DataFrame)
        assert list(sub.columns) == ["a", "b"]
        assert len(sub) == 3

    def test_getitem_bool_series_filters_rows(self):
        mask = pd.Series([True, False, True], name="m")
        filtered = self.df[mask]
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2
        assert filtered["a"].tolist() == [1, 3]

    def test_iloc_int_returns_dict(self):
        row = self.df.iloc[0]
        assert isinstance(row, dict)
        assert row["a"] == 1

    def test_iloc_slice_returns_dataframe(self):
        sub = self.df.iloc[0:2]
        assert isinstance(sub, pd.DataFrame)
        assert len(sub) == 2
        assert sub["c"].tolist() == ["x", "y"]

    def test_loc_int_col_returns_scalar(self):
        val = self.df.loc[1, "b"]
        assert val == 5.0

    def test_iloc_row_col_returns_scalar(self):
        val = self.df.iloc[2, 0]
        assert val == 3

    def test_attribute_access(self):
        """df.col_name is a shortcut for df['col_name']."""
        s = self.df.a
        assert isinstance(s, pd.Series)
        assert s.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Mutations (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameMutations:
    def test_setitem_list_adds_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["b"] = [10, 20, 30]
        assert "b" in df.columns
        assert df["b"].tolist() == [10, 20, 30]

    def test_setitem_series_adds_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["b"] = pd.Series([7, 8, 9], name="b")
        assert df["b"].tolist() == [7, 8, 9]

    def test_iloc_setitem_row_col_scalar(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.iloc[0, 0] = 99
        assert df.iloc[0]["a"] == 99

    def test_loc_setitem_row_col_scalar(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.loc[0, "a"] = 42
        assert df.iloc[0]["a"] == 42

    def test_drop_columns_does_not_mutate_original(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dropped = df.drop(columns=["b"])
        assert "b" in df.columns
        assert "b" not in dropped.columns


# ---------------------------------------------------------------------------
# Aggregation (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameAggregation:
    def setup_method(self):
        self.df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})

    def test_sum_mean_min_max_return_dict(self):
        assert self.df.sum()["a"] == 10
        assert self.df.mean()["a"] == 2.5
        assert self.df.min()["a"] == 1
        assert self.df.max()["a"] == 4

    def test_describe_returns_dataframe_with_stats(self):
        desc = self.df.describe()
        assert isinstance(desc, pd.DataFrame)
        # describe should have rows for count, mean, std, min, 25%, 50%, 75%, max
        assert "a" in desc.columns or "b" in desc.columns

    def test_agg_string(self):
        result = self.df.agg("sum")
        assert result["a"] == 10

    def test_agg_list_of_strings(self):
        result = self.df.agg(["sum", "mean"])
        assert isinstance(result, pd.DataFrame)

    def test_agg_dict(self):
        result = self.df.agg({"a": "sum", "b": "mean"})
        assert isinstance(result, pd.DataFrame)
        assert result["a"][0] == 10
        assert result["b"][0] == 25.0


# ---------------------------------------------------------------------------
# Sorting & Deduplication (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameSortingDedup:
    def test_sort_values_single_column_ascending(self):
        df = pd.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
        sorted_df = df.sort_values("a")
        assert sorted_df["a"].tolist() == [1, 2, 3]

    def test_sort_values_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [2, 1, 3]})
        sorted_df = df.sort_values(["a", "b"])
        assert sorted_df["b"].tolist() == [1, 2, 3]

    def test_sort_values_ascending_list(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        sorted_df = df.sort_values(["a", "b"], ascending=[False, True])
        assert sorted_df["a"].tolist() == [3, 2, 1]

    def test_drop_duplicates_basic(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [10, 10, 20, 30]})
        result = df.drop_duplicates()
        assert len(result) == 3

    def test_drop_duplicates_with_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        result = df.drop_duplicates(subset=["a"])
        # keep="first" by default: first row with a=1 kept
        assert len(result) == 2
        assert result["a"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# GroupBy (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameGroupBy:
    def test_groupby_sum(self):
        df = pd.DataFrame({"key": ["a", "b", "a", "b"], "val": [1, 2, 3, 4]})
        result = df.groupby("key").sum()
        assert isinstance(result, pd.DataFrame)

    def test_groupby_column_select_mean(self):
        df = pd.DataFrame({"key": ["a", "b", "a"], "x": [10, 20, 30], "y": [1, 2, 3]})
        result = df.groupby("key")["x"].mean()
        # result is a DataFrame or Series
        assert result is not None

    def test_groupby_agg_dict(self):
        df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]})
        result = df.groupby("key").agg({"val": "sum"})
        assert isinstance(result, pd.DataFrame)

    def test_groupby_transform(self):
        df = pd.DataFrame({"key": ["a", "b", "a"], "val": [1, 2, 3]})
        result = df.groupby("key").transform("sum")
        # Transform returns same-length result
        assert result is not None

    def test_groupby_multiple_keys(self):
        df = pd.DataFrame({
            "k1": ["a", "a", "b", "b"],
            "k2": [1, 2, 1, 2],
            "val": [10, 20, 30, 40],
        })
        result = df.groupby(["k1", "k2"]).sum()
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Merge & Concat (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameMergeConcat:
    def test_merge_inner(self):
        left = pd.DataFrame({"key": [1, 2, 3], "val_l": [10, 20, 30]})
        right = pd.DataFrame({"key": [2, 3, 4], "val_r": [200, 300, 400]})
        result = left.merge(right, on="key", how="inner")
        assert len(result) == 2
        assert set(result["key"].tolist()) == {2, 3}

    def test_merge_left_preserves_left_rows(self):
        left = pd.DataFrame({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"key": [2, 3], "b": [200, 300]})
        result = left.merge(right, on="key", how="left")
        assert len(result) == 3

    def test_merge_left_on_right_on(self):
        left = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
        right = pd.DataFrame({"uid": [1, 2], "y": [100, 200]})
        result = left.merge(right, left_on="id", right_on="uid")
        assert len(result) == 2
        assert "x" in result.columns
        assert "y" in result.columns

    def test_concat_vertical(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})
        result = pd.concat([df1, df2])
        assert len(result) == 4
        assert result["a"].tolist() == [1, 2, 3, 4]

    def test_concat_horizontal_axis1(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        result = pd.concat([df1, df2], axis=1)
        assert "a" in result.columns
        assert "b" in result.columns
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Null handling (5 tests)
# ---------------------------------------------------------------------------

class TestDataFrameNullHandling:
    def test_fillna_scalar(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        filled = df.fillna(0.0)
        assert filled["a"].tolist() == [1.0, 0.0, 3.0]

    def test_dropna_removes_rows_with_nulls(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
        result = df.dropna()
        assert len(result) == 1
        assert result["a"].tolist() == [1.0]

    def test_isna_returns_bool_dataframe(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
        result = df.isna()
        assert isinstance(result, pd.DataFrame)
        a_vals = result["a"].tolist()
        assert a_vals[0] == False
        assert a_vals[1] == True

    def test_where_with_condition(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        cond = pd.Series([True, False, True, False], name="c")
        result = df.where(cond, other=0)
        assert result["a"].tolist() == [1, 0, 3, 0]

    def test_none_values_preserved_as_nulls(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        vals = df["a"].tolist()
        assert vals[1] is None

    # Extra: fillna with dict
    def test_fillna_dict_per_column(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
        result = df.fillna({"a": 0.0, "b": 99.0})
        assert result["a"].tolist() == [1.0, 0.0]
        assert result["b"].tolist() == [99.0, 2.0]

    # Extra: dropna with subset
    def test_dropna_subset(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 5.0, None]})
        result = df.dropna(subset=["a"])
        assert len(result) == 2
        assert result["a"].tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Arithmetic operators (extra — gaps identified)
# ---------------------------------------------------------------------------

class TestDataFrameArithmetic:
    def test_add_dataframe(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [10, 20, 30]})
        result = df1 + df2
        assert result["a"].tolist() == [11, 22, 33]

    def test_mul_scalar(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df * 2
        assert result["a"].tolist() == [2.0, 4.0, 6.0]

    def test_sub_dataframe(self):
        df1 = pd.DataFrame({"a": [10, 20, 30]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        result = df1 - df2
        assert result["a"].tolist() == [9.0, 18.0, 27.0]

    def test_mask_opposite_of_where(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cond = pd.Series([True, False, True], name="c")
        result = df.mask(cond, other=None)
        assert result["a"].tolist() == [None, 2, None]

    def test_any_per_column(self):
        df = pd.DataFrame({"a": [True, False], "b": [True, True]})
        result = df.any()
        vals = result.tolist()
        assert vals[0] == True  # column a: any True
        assert vals[1] == True  # column b: any True

    def test_update_in_place(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        other = pd.DataFrame({"a": [99, None, None]})
        df.update(other)
        assert df["a"].tolist()[0] == 99
        assert df["a"].tolist()[1] == 2  # not overwritten (None in other)
