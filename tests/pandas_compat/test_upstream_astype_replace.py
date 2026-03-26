"""
Upstream-adapted pandas compatibility tests for:
  - DataFrame.astype / Series.astype
  - Series.replace / DataFrame.replace
  - Series.rank
  - Series.rename / Series.to_list

Adapted from:
  pandas/tests/frame/methods/test_astype.py
  pandas/tests/series/methods/test_replace.py
  pandas/tests/series/methods/test_rank.py
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame.astype
# ---------------------------------------------------------------------------

def test_df_astype_int_to_float():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.astype({"a": "float64"})
    assert isinstance(result, pd.DataFrame)
    vals = result["a"].tolist()
    assert vals == [1.0, 2.0, 3.0]


def test_df_astype_float_to_int():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df.astype({"a": "int64"})
    vals = result["a"].tolist()
    assert vals == [1, 2, 3]


def test_df_astype_preserves_other_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.astype({"a": "float64"})
    # Column b should be untouched
    assert result["b"].tolist() == [3, 4]


def test_df_astype_multiple_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.astype({"a": "float64", "b": "float64"})
    assert result["a"].tolist() == [1.0, 2.0]
    assert result["b"].tolist() == [3.0, 4.0]


def test_df_astype_preserves_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.astype({"a": "float64"})
    assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# Series.astype
# ---------------------------------------------------------------------------

def test_series_astype_int_to_float():
    s = pd.Series([1, 2, 3], name="x")
    result = s.astype("float64")
    vals = result.tolist()
    assert vals == [1.0, 2.0, 3.0]


def test_series_astype_float_to_int():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    result = s.astype("int64")
    vals = result.tolist()
    assert vals == [1, 2, 3]


def test_series_astype_preserves_name():
    s = pd.Series([1, 2, 3], name="myname")
    result = s.astype("float64")
    assert result.name == "myname"


def test_series_astype_preserves_length():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = s.astype("float64")
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Series.replace
# ---------------------------------------------------------------------------

def test_series_replace_scalar():
    s = pd.Series([1, 2, 3, 1, 2], name="x")
    result = s.replace(1, 99)
    vals = result.tolist()
    assert vals == [99, 2, 3, 99, 2]


def test_series_replace_only_target():
    s = pd.Series([1, 2, 3], name="x")
    result = s.replace(4, 99)  # 4 is not present — no change
    assert result.tolist() == [1, 2, 3]


def test_series_replace_preserves_name():
    s = pd.Series([1, 2, 3], name="myname")
    result = s.replace(1, 0)
    assert result.name == "myname"


def test_series_replace_preserves_length():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = s.replace(1, 99)
    assert len(result) == 5


def test_series_replace_all_occurrences():
    s = pd.Series([7, 7, 7, 7], name="x")
    result = s.replace(7, 0)
    assert result.tolist() == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# DataFrame.replace
# ---------------------------------------------------------------------------

def test_df_replace_scalar():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 5, 6]})
    result = df.replace(1, 99)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [99, 2, 3]
    assert result["b"].tolist() == [99, 5, 6]


def test_df_replace_preserves_shape():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.replace(1, 0)
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Series.rank
# ---------------------------------------------------------------------------

def test_series_rank_basic_ascending():
    s = pd.Series([3.0, 1.0, 2.0], name="x")
    result = s.rank()
    # Default: ascending average
    # 1.0 -> rank 1, 2.0 -> rank 2, 3.0 -> rank 3
    vals = result.tolist()
    assert vals[0] == 3.0  # 3.0 is 3rd
    assert vals[1] == 1.0  # 1.0 is 1st
    assert vals[2] == 2.0  # 2.0 is 2nd


def test_series_rank_descending():
    s = pd.Series([3.0, 1.0, 2.0], name="x")
    result = s.rank(ascending=False)
    vals = result.tolist()
    assert vals[0] == 1.0  # 3.0 is ranked 1st in descending
    assert vals[1] == 3.0  # 1.0 is ranked 3rd
    assert vals[2] == 2.0  # 2.0 is ranked 2nd


def test_series_rank_ties_average():
    s = pd.Series([1.0, 1.0, 3.0], name="x")
    result = s.rank(method="average")
    vals = result.tolist()
    # Two values tie at 1: average of rank 1 and 2 = 1.5
    assert abs(vals[0] - 1.5) < 1e-9
    assert abs(vals[1] - 1.5) < 1e-9
    assert vals[2] == 3.0


def test_series_rank_preserves_length():
    s = pd.Series([5.0, 3.0, 1.0, 4.0, 2.0], name="x")
    result = s.rank()
    assert len(result) == 5


def test_series_rank_preserves_name():
    s = pd.Series([3.0, 1.0, 2.0], name="myrank")
    result = s.rank()
    assert result.name == "myrank"


def test_series_rank_returns_series():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    result = s.rank()
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Series.rename and Series.to_list
# ---------------------------------------------------------------------------

def test_series_rename_changes_name():
    s = pd.Series([1, 2, 3], name="old")
    result = s.rename("new")
    assert result.name == "new"


def test_series_rename_preserves_values():
    s = pd.Series([10, 20, 30], name="x")
    result = s.rename("y")
    assert result.tolist() == [10, 20, 30]


def test_series_rename_preserves_length():
    s = pd.Series([1, 2, 3, 4], name="x")
    result = s.rename("y")
    assert len(result) == 4


def test_series_to_list_alias():
    s = pd.Series([1, 2, 3], name="x")
    assert s.to_list() == [1, 2, 3]


def test_series_to_list_same_as_tolist():
    s = pd.Series([4, 5, 6], name="x")
    assert s.to_list() == s.tolist()


def test_series_to_list_empty():
    s = pd.Series([], name="x")
    assert s.to_list() == []
