"""Tests for filter, pop, add_prefix/suffix, rename with callable."""
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# filter — DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameFilter:
    def test_filter_items_basic(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = df.filter(items=["a", "c"])
        assert list(result.columns) == ["a", "c"]

    def test_filter_items_skips_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.filter(items=["a", "missing"])
        assert list(result.columns) == ["a"]

    def test_filter_items_preserves_row_count(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.filter(items=["a"])
        assert len(result) == 3

    def test_filter_like_basic(self):
        df = pd.DataFrame({"ax": [1], "ay": [2], "b": [3]})
        result = df.filter(like="a")
        assert list(result.columns) == ["ax", "ay"]

    def test_filter_like_no_match(self):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        result = df.filter(like="xyz")
        assert list(result.columns) == []

    def test_filter_like_single_col(self):
        df = pd.DataFrame({"name": [1], "age": [2], "grade": [3]})
        result = df.filter(like="age")
        assert list(result.columns) == ["age"]

    def test_filter_preserves_data(self):
        df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        result = df.filter(items=["a"])
        assert result["a"].tolist() == [10, 20]

    def test_filter_regex_basic(self):
        df = pd.DataFrame({"a1": [1], "a2": [2], "b1": [3]})
        result = df.filter(regex="^a")
        assert list(result.columns) == ["a1", "a2"]

    def test_filter_no_args_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            df.filter()


# ---------------------------------------------------------------------------
# pop — DataFrame
# ---------------------------------------------------------------------------

class TestDataFramePop:
    def test_pop_returns_series(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.pop("b")
        assert isinstance(result, pd.Series)

    def test_pop_returns_correct_values(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.pop("b")
        assert result.tolist() == [3, 4]

    def test_pop_removes_column(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df.pop("b")
        assert "b" not in df.columns

    def test_pop_reduces_column_count(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        before = len(df.columns)
        df.pop("b")
        assert len(df.columns) == before - 1

    def test_pop_preserves_other_columns(self):
        df = pd.DataFrame({"a": [10], "b": [20], "c": [30]})
        df.pop("b")
        assert list(df.columns) == ["a", "c"]
        assert df["a"].tolist() == [10]
        assert df["c"].tolist() == [30]

    def test_pop_single_column_leaves_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.pop("a")
        assert df.empty is True


# ---------------------------------------------------------------------------
# add_prefix / add_suffix — DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameAddPrefixSuffix:
    def test_add_prefix_renames_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.add_prefix("x_")
        assert list(result.columns) == ["x_a", "x_b"]

    def test_add_suffix_renames_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.add_suffix("_y")
        assert list(result.columns) == ["a_y", "b_y"]

    def test_add_prefix_preserves_data(self):
        df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        result = df.add_prefix("p_")
        assert result["p_a"].tolist() == [10, 20]
        assert result["p_b"].tolist() == [30, 40]

    def test_add_suffix_preserves_shape(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = df.add_suffix("_z")
        assert result.shape == df.shape

    def test_add_prefix_does_not_mutate(self):
        df = pd.DataFrame({"a": [1]})
        _ = df.add_prefix("p_")
        assert list(df.columns) == ["a"]

    def test_chained_prefix_suffix(self):
        df = pd.DataFrame({"col": [1]})
        result = df.add_prefix("x_").add_suffix("_y")
        assert list(result.columns) == ["x_col_y"]

    def test_add_prefix_empty_string(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.add_prefix("")
        assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# rename with callable
# ---------------------------------------------------------------------------

class TestDataFrameRenameCallable:
    def test_rename_with_upper(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.rename(columns=str.upper)
        assert list(result.columns) == ["A", "B"]

    def test_rename_with_lambda(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.rename(columns=lambda x: x + "_new")
        assert list(result.columns) == ["a_new", "b_new"]

    def test_rename_callable_preserves_data(self):
        df = pd.DataFrame({"a": [10, 20]})
        result = df.rename(columns=str.upper)
        assert result["A"].tolist() == [10, 20]

    def test_rename_dict_basic(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.rename(columns={"a": "alpha", "b": "beta"})
        assert list(result.columns) == ["alpha", "beta"]

    def test_rename_dict_partial(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = df.rename(columns={"a": "A"})
        assert "A" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns


# ---------------------------------------------------------------------------
# str.find
# ---------------------------------------------------------------------------

class TestStrFind:
    def test_str_find_found(self):
        s = pd.Series(["foo", "bar", "baz"], name="x")
        result = s.str.find("a")
        vals = result.tolist()
        assert vals[0] == -1   # "foo" has no "a"
        assert vals[1] == 1    # "bar" has "a" at index 1
        assert vals[2] == 1    # "baz" has "a" at index 1

    def test_str_find_not_found(self):
        s = pd.Series(["hello", "world"], name="x")
        result = s.str.find("z")
        assert result.tolist() == [-1, -1]

    def test_str_find_returns_series(self):
        s = pd.Series(["abc"], name="x")
        result = s.str.find("b")
        assert isinstance(result, pd.Series)
