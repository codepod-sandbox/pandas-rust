"""Tests for shift, isna/notna, where, equals, join, to_numeric, int arithmetic."""
import math
import pandas as pd
from pandas import Series, DataFrame


# ---------------------------------------------------------------------------
# shift tests (5)
# ---------------------------------------------------------------------------

def test_shift_down_by_one():
    s = Series([10, 20, 30, 40])
    result = s.shift(1)
    vals = result.tolist()
    assert vals[0] is None
    assert vals[1] == 10
    assert vals[2] == 20
    assert vals[3] == 30


def test_shift_up_by_one():
    s = Series([10, 20, 30, 40])
    result = s.shift(-1)
    vals = result.tolist()
    assert vals[0] == 20
    assert vals[1] == 30
    assert vals[2] == 40
    assert vals[3] is None


def test_shift_zero_unchanged():
    s = Series([1, 2, 3])
    result = s.shift(0)
    assert result.tolist() == [1, 2, 3]


def test_shift_by_two():
    s = Series([1, 2, 3, 4, 5])
    result = s.shift(2)
    vals = result.tolist()
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] == 1
    assert vals[3] == 2
    assert vals[4] == 3


def test_shift_float_series():
    s = Series([1.1, 2.2, 3.3])
    result = s.shift(1)
    vals = result.tolist()
    assert vals[0] is None
    assert abs(vals[1] - 1.1) < 1e-9
    assert abs(vals[2] - 2.2) < 1e-9


# ---------------------------------------------------------------------------
# pd.isna / pd.notna (4)
# ---------------------------------------------------------------------------

def test_isna_none():
    assert pd.isna(None) is True


def test_isna_regular_float():
    assert pd.isna(1.0) is False


def test_isna_nan():
    assert pd.isna(float("nan")) is True


def test_notna_int():
    assert pd.notna(1) is True


# Extra alias checks
def test_isnull_alias():
    assert pd.isnull(None) is True
    assert pd.isnull(0) is False


def test_notnull_alias():
    assert pd.notnull(None) is False
    assert pd.notnull(42) is True


# ---------------------------------------------------------------------------
# where tests (5)
# ---------------------------------------------------------------------------

def test_series_where_keeps_true():
    s = Series([1, 2, 3, 4, 5])
    cond = Series([True, True, False, True, False])
    result = s.where(cond)
    vals = result.tolist()
    assert vals[0] == 1
    assert vals[1] == 2
    assert vals[2] is None
    assert vals[3] == 4
    assert vals[4] is None


def test_series_where_all_true():
    s = Series([10, 20, 30])
    cond = Series([True, True, True])
    result = s.where(cond)
    assert result.tolist() == [10, 20, 30]


def test_series_where_with_custom_other():
    s = Series([1, 2, 3])
    cond = Series([True, False, True])
    result = s.where(cond, other=99)
    assert result.tolist() == [1, 99, 3]


def test_dataframe_where_with_series_mask():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cond = Series([True, False, True])
    result = df.where(cond)
    assert result["a"].tolist() == [1, None, 3]
    assert result["b"].tolist() == [4, None, 6]


def test_dataframe_where_with_dataframe_mask():
    df = DataFrame({"x": [10, 20, 30], "y": [1, 2, 3]})
    cond_df = DataFrame({"x": [True, False, True], "y": [False, True, True]})
    result = df.where(cond_df)
    assert result["x"].tolist() == [10, None, 30]
    assert result["y"].tolist() == [None, 2, 3]


# ---------------------------------------------------------------------------
# equals tests (4)
# ---------------------------------------------------------------------------

def test_dataframe_equals_same():
    df = DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert df.equals(df.copy()) is True


def test_dataframe_equals_different_values():
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"a": [1, 2, 99]})
    assert df1.equals(df2) is False


def test_series_equals_same():
    s = Series([1, 2, 3])
    assert s.equals(s.copy()) is True


def test_series_equals_different():
    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 4])
    assert s1.equals(s2) is False


def test_dataframe_equals_different_columns():
    df1 = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [1, 2]})
    assert df1.equals(df2) is False


# ---------------------------------------------------------------------------
# join tests (3)
# ---------------------------------------------------------------------------

def test_join_basic():
    df_a = DataFrame({"a": [1, 2, 3]})
    df_b = DataFrame({"b": [4, 5, 6]})
    result = df_a.join(df_b)
    assert "a" in result.columns
    assert "b" in result.columns


def test_join_preserves_columns():
    df_a = DataFrame({"x": [10, 20]})
    df_b = DataFrame({"y": [30, 40]})
    result = df_a.join(df_b)
    assert list(result.columns) == ["x", "y"]


def test_join_different_names():
    df_a = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df_b = DataFrame({"col3": [7, 8, 9]})
    result = df_a.join(df_b)
    assert result.shape == (3, 3)
    assert "col1" in result.columns
    assert "col3" in result.columns


# ---------------------------------------------------------------------------
# to_numeric tests (3)
# ---------------------------------------------------------------------------

def test_to_numeric_series():
    s = Series([1, 2, 3])
    result = pd.to_numeric(s)
    assert result.dtype == "float64"


def test_to_numeric_int_series():
    s = Series([10, 20, 30])
    result = pd.to_numeric(s)
    vals = result.tolist()
    assert vals[0] == 10.0
    assert vals[1] == 20.0
    assert vals[2] == 30.0


def test_to_numeric_scalar():
    result = pd.to_numeric(42)
    assert result == 42.0


# ---------------------------------------------------------------------------
# int arithmetic type preservation (3)
# ---------------------------------------------------------------------------

def test_int_plus_int_series_stays_int():
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = s1 + s2
    assert result.dtype == "int64"
    assert result.tolist() == [5, 7, 9]


def test_int_mul_int_stays_int():
    s = Series([2, 3, 4])
    result = s * 3
    assert result.dtype == "int64"
    assert result.tolist() == [6, 9, 12]


def test_int_plus_float_promotes():
    s = Series([1, 2, 3])
    result = s + 1.5
    assert result.dtype == "float64"
    vals = result.tolist()
    assert abs(vals[0] - 2.5) < 1e-9


def test_int_plus_int_scalar_stays_int():
    s = Series([10, 20, 30])
    result = s + 5
    assert result.dtype == "int64"
    assert result.tolist() == [15, 25, 35]


# ---------------------------------------------------------------------------
# groupby with extra kwargs
# ---------------------------------------------------------------------------

def test_groupby_accepts_sort_kwarg():
    df = DataFrame({"cat": ["a", "b", "a"], "val": [1, 2, 3]})
    gb = df.groupby("cat", sort=True)
    result = gb.sum()
    assert result is not None


def test_groupby_accepts_as_index_kwarg():
    df = DataFrame({"cat": ["a", "b", "a"], "val": [1, 2, 3]})
    gb = df.groupby("cat", as_index=False)
    result = gb.sum()
    assert result is not None
