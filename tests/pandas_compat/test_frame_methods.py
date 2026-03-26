"""Upstream-style pandas tests for DataFrame methods."""
import pytest
from pandas import DataFrame, Series


# ---------------------------------------------------------------------------
# sort_values
# ---------------------------------------------------------------------------

def test_sort_values_single_col_ascending():
    df = DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
    result = df.sort_values("a")
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == ["a", "b", "c"]


def test_sort_values_single_col_descending():
    df = DataFrame({"a": [3, 1, 2]})
    result = df.sort_values("a", ascending=False)
    assert result["a"].tolist() == [3, 2, 1]


def test_sort_values_float_col():
    df = DataFrame({"x": [1.5, 0.5, 2.5]})
    result = df.sort_values("x")
    assert result["x"].tolist() == [0.5, 1.5, 2.5]


def test_sort_values_string_col():
    df = DataFrame({"name": ["banana", "apple", "cherry"]})
    result = df.sort_values("name")
    assert result["name"].tolist() == ["apple", "banana", "cherry"]


def test_sort_values_preserves_shape():
    df = DataFrame({"a": [3, 1, 2], "b": [10, 20, 30]})
    result = df.sort_values("a")
    assert result.shape == (3, 2)


def test_sort_values_stable():
    # equal values should preserve original order (stable sort)
    df = DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    result = df.sort_values("a")
    assert result["b"].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_column():
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df.drop(columns="b")
    assert list(result.columns) == ["a", "c"]


def test_drop_multiple_columns():
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    result = df.drop(columns=["a", "c"])
    assert list(result.columns) == ["b"]


def test_drop_preserves_data():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.drop(columns="b")
    assert result["a"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------

def test_drop_duplicates_basic():
    df = DataFrame({"a": [1, 2, 1, 3], "b": [1, 2, 1, 3]})
    result = df.drop_duplicates()
    assert result.shape[0] == 3


def test_drop_duplicates_keep_first():
    df = DataFrame({"a": [1, 2, 1], "b": [10, 20, 99]})
    result = df.drop_duplicates(subset=["a"], keep="first")
    assert result["b"].tolist() == [10, 20]


def test_drop_duplicates_keep_last():
    df = DataFrame({"a": [1, 2, 1], "b": [10, 20, 99]})
    result = df.drop_duplicates(subset=["a"], keep="last")
    assert result["b"].tolist() == [20, 99]


def test_drop_duplicates_keep_false():
    df = DataFrame({"a": [1, 2, 1]})
    result = df.drop_duplicates(keep=False)
    assert result["a"].tolist() == [2]


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_columns():
    df = DataFrame({"a": [1], "b": [2]})
    result = df.rename(columns={"a": "x", "b": "y"})
    assert list(result.columns) == ["x", "y"]


def test_rename_partial():
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    result = df.rename(columns={"a": "x"})
    assert list(result.columns) == ["x", "b", "c"]


def test_rename_no_op_for_unmapped():
    df = DataFrame({"a": [1], "b": [2]})
    result = df.rename(columns={"z": "q"})  # z doesn't exist
    assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------

def test_fillna_int():
    df = DataFrame({"a": [1, None, 3]})
    result = df.fillna(0)
    assert result["a"].tolist() == [1, 0, 3]


def test_fillna_float():
    df = DataFrame({"a": [1.0, None, 3.0]})
    result = df.fillna(0.0)
    vals = result["a"].tolist()
    assert vals[0] == 1.0
    assert vals[1] == 0.0
    assert vals[2] == 3.0


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_returns_dataframe():
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
    result = df.describe()
    assert isinstance(result, DataFrame)


def test_describe_numeric_only():
    df = DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    result = df.describe()
    assert "a" in result.columns
    assert "name" not in result.columns


def test_describe_shape():
    df = DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df.describe()
    # Should have 5 stats (count, mean, std, min, max)
    assert result.shape[0] == 5


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------

def test_copy_independence():
    df = DataFrame({"a": [1, 2, 3]})
    copy = df.copy()
    copy["a"] = [10, 20, 30]
    # Original should be unchanged
    assert df["a"].tolist() == [1, 2, 3]


def test_copy_same_values():
    df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
    copy = df.copy()
    assert copy["a"].tolist() == [1, 2]
    assert copy["b"].tolist() == ["x", "y"]


# ---------------------------------------------------------------------------
# head / tail
# ---------------------------------------------------------------------------

def test_head_default():
    df = DataFrame({"a": list(range(10))})
    result = df.head()
    assert result.shape[0] == 5


def test_head_custom_n():
    df = DataFrame({"a": list(range(10))})
    result = df.head(3)
    assert result["a"].tolist() == [0, 1, 2]


def test_head_n_zero():
    df = DataFrame({"a": [1, 2, 3]})
    result = df.head(0)
    assert result.shape[0] == 0


def test_head_n_greater_than_len():
    df = DataFrame({"a": [1, 2, 3]})
    result = df.head(100)
    assert result.shape[0] == 3


def test_tail_default():
    df = DataFrame({"a": list(range(10))})
    result = df.tail()
    assert result.shape[0] == 5
    assert result["a"].tolist() == [5, 6, 7, 8, 9]


def test_tail_custom_n():
    df = DataFrame({"a": list(range(10))})
    result = df.tail(3)
    assert result["a"].tolist() == [7, 8, 9]


# ---------------------------------------------------------------------------
# select_dtypes
# ---------------------------------------------------------------------------

def test_select_dtypes_include_numeric():
    df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})
    result = df.select_dtypes(include=["number"])
    assert "a" in result.columns
    assert "c" in result.columns
    assert "b" not in result.columns


def test_select_dtypes_exclude_string():
    df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = df.select_dtypes(exclude=["object"])
    assert "a" in result.columns
    assert "b" not in result.columns


# ---------------------------------------------------------------------------
# nlargest / nsmallest
# ---------------------------------------------------------------------------

def test_nlargest_basic():
    df = DataFrame({"a": [3, 1, 4, 1, 5, 9, 2, 6]})
    result = df.nlargest(3, "a")
    assert result.shape[0] == 3
    vals = result["a"].tolist()
    assert max(vals) == 9


def test_nsmallest_basic():
    df = DataFrame({"a": [3, 1, 4, 1, 5, 9, 2, 6]})
    result = df.nsmallest(3, "a")
    assert result.shape[0] == 3
    vals = result["a"].tolist()
    assert min(vals) == 1


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------

def test_assign_new_column():
    df = DataFrame({"a": [1, 2, 3]})
    result = df.assign(b=[4, 5, 6])
    assert "b" in result.columns
    assert result["b"].tolist() == [4, 5, 6]


def test_assign_does_not_mutate_original():
    df = DataFrame({"a": [1, 2, 3]})
    _ = df.assign(b=[4, 5, 6])
    assert "b" not in df.columns


def test_assign_multiple_columns():
    df = DataFrame({"a": [1, 2]})
    result = df.assign(b=[3, 4], c=[5, 6])
    assert "b" in result.columns
    assert "c" in result.columns


# ---------------------------------------------------------------------------
# abs
# ---------------------------------------------------------------------------

def test_abs_numeric_df():
    df = DataFrame({"a": [-1, -2, 3], "b": [-1.5, 0.0, 2.5]})
    result = df.abs()
    assert result["a"].tolist() == [1, 2, 3]
    b_vals = result["b"].tolist()
    assert b_vals[0] == 1.5
    assert b_vals[2] == 2.5


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------

def test_clip_lower_upper():
    df = DataFrame({"a": [1, 5, 10]})
    result = df.clip(lower=2, upper=8)
    assert result["a"].tolist() == [2, 5, 8]


def test_clip_lower_only():
    df = DataFrame({"a": [-5, 0, 5]})
    result = df.clip(lower=0)
    assert result["a"].tolist() == [0, 0, 5]


# ---------------------------------------------------------------------------
# reset_index
# ---------------------------------------------------------------------------

def test_reset_index_returns_copy():
    df = DataFrame({"a": [1, 2, 3]})
    result = df.reset_index(drop=True)
    assert result.shape == df.shape
    assert result["a"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

def test_transpose_basic():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.T
    assert isinstance(result, DataFrame)
    # Original 2 rows, 2 cols -> transposed: 2 rows, 3 cols (index col + 2 original rows)
    assert result.shape[1] == 3  # "" + "0" + "1"


def test_transpose_numeric_values():
    df = DataFrame({"x": [1, 2, 3]})
    result = df.transpose()
    # Should have columns "", "0", "1", "2"
    assert "" in result.columns
    assert result[""].tolist() == ["x"]
