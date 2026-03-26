"""Tests for query, set_axis, pd.merge, reindex, mask, groupby.filter."""
import pandas as pd
import pytest


class TestQuery:
    def setup_method(self):
        self.df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

    def test_query_gt(self):
        result = self.df.query("a > 2")
        assert list(result["a"].tolist()) == [3, 4, 5]

    def test_query_eq(self):
        result = self.df.query("a == 1")
        assert list(result["a"].tolist()) == [1]

    def test_query_and(self):
        result = self.df.query("a > 1 and b < 40")
        assert list(result["a"].tolist()) == [2, 3]

    def test_query_or(self):
        result = self.df.query("a == 1 or b == 50")
        assert list(result["a"].tolist()) == [1, 5]

    def test_query_float(self):
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3], "y": [0.5, 1.5, 2.5]})
        result = df.query("x > 2.0")
        assert len(result) == 2

    def test_query_shape(self):
        result = self.df.query("a > 3")
        assert result.shape[0] == 2

    def test_query_preserves_columns(self):
        result = self.df.query("a > 2")
        assert list(result.columns) == ["a", "b"]

    def test_query_returns_dataframe(self):
        result = self.df.query("b == 30")
        assert isinstance(result, pd.DataFrame)
        assert result["b"].tolist()[0] == 30


class TestSetAxis:
    def setup_method(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_set_axis_renames_columns(self):
        result = self.df.set_axis(["x", "y"], axis=1)
        assert list(result.columns) == ["x", "y"]

    def test_set_axis_preserves_data(self):
        result = self.df.set_axis(["x", "y"], axis=1)
        assert result["x"].tolist() == [1, 2, 3]
        assert result["y"].tolist() == [4, 5, 6]

    def test_set_axis_preserves_shape(self):
        result = self.df.set_axis(["col1", "col2"], axis=1)
        assert result.shape == self.df.shape

    def test_set_axis_columns_keyword(self):
        result = self.df.set_axis(["p", "q"], axis="columns")
        assert list(result.columns) == ["p", "q"]


class TestMergeTopLevel:
    def setup_method(self):
        self.left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        self.right = pd.DataFrame({"id": [2, 3, 4], "info": ["b", "c", "d"]})

    def test_merge_function_exists(self):
        result = pd.merge(self.left, self.right, on="id")
        assert isinstance(result, pd.DataFrame)

    def test_merge_inner(self):
        result = pd.merge(self.left, self.right, on="id", how="inner")
        assert len(result) == 2
        assert sorted(result["id"].tolist()) == [2, 3]

    def test_merge_left(self):
        result = pd.merge(self.left, self.right, on="id", how="left")
        assert len(result) == 3

    def test_merge_on_param(self):
        result = pd.merge(self.left, self.right, on="id")
        assert "val" in result.columns
        assert "info" in result.columns


class TestReindex:
    def setup_method(self):
        self.df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    def test_reindex_adds_missing_columns(self):
        result = self.df.reindex(columns=["a", "b", "d"])
        assert "d" in result.columns
        assert result["d"].tolist() == [None, None]

    def test_reindex_reorders_columns(self):
        result = self.df.reindex(columns=["c", "a"])
        assert list(result.columns) == ["c", "a"]

    def test_reindex_drops_missing(self):
        result = self.df.reindex(columns=["a"])
        assert list(result.columns) == ["a"]
        assert "b" not in result.columns

    def test_reindex_preserves_row_count(self):
        result = self.df.reindex(columns=["a", "b"])
        assert len(result) == 2


class TestSeriesMask:
    def setup_method(self):
        self.s = pd.Series([1, 2, 3, 4, 5])

    def test_mask_replaces_where_true(self):
        cond = pd.Series([True, False, True, False, True])
        result = self.s.mask(cond)
        assert result.tolist() == [None, 2, None, 4, None]

    def test_mask_custom_other(self):
        cond = pd.Series([True, False, True, False, False])
        result = self.s.mask(cond, other=0)
        assert result.tolist() == [0, 2, 0, 4, 5]

    def test_mask_preserves_where_false(self):
        cond = pd.Series([False, False, True, False, False])
        result = self.s.mask(cond, other=-1)
        assert result.tolist()[0] == 1
        assert result.tolist()[1] == 2
        assert result.tolist()[2] == -1


class TestGroupByFilter:
    def setup_method(self):
        self.df = pd.DataFrame({
            "group": ["a", "a", "b", "b", "c"],
            "val": [10, 20, 5, 5, 100],
        })

    def test_filter_keeps_passing_groups(self):
        result = self.df.groupby("group").filter(lambda x: x["val"].sum() > 15)
        groups_kept = sorted(result["group"].tolist())
        assert "a" in groups_kept

    def test_filter_removes_failing_groups(self):
        result = self.df.groupby("group").filter(lambda x: x["val"].sum() > 15)
        groups_kept = result["group"].tolist()
        # group "b" sums to 10, should be removed
        assert "b" not in groups_kept

    def test_filter_preserves_all_columns(self):
        result = self.df.groupby("group").filter(lambda x: x["val"].sum() > 15)
        assert "group" in result.columns
        assert "val" in result.columns
