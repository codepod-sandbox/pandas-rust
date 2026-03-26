"""
Upstream-adapted pandas compatibility tests for:
  - DataFrame.equals / Series.equals
  - DataFrame.count / Series.count
  - DataFrame.describe

Adapted from:
  pandas/tests/frame/methods/test_equals.py
  pandas/tests/frame/methods/test_count.py
  pandas/tests/frame/methods/test_describe.py
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame.equals
# ---------------------------------------------------------------------------

def test_df_equals_same_data():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert df1.equals(df2) is True


def test_df_equals_different_values():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 9]})
    assert df1.equals(df2) is False


def test_df_equals_different_columns():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [1, 2]})
    assert df1.equals(df2) is False


def test_df_equals_different_shape():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2]})
    assert df1.equals(df2) is False


def test_df_equals_non_dataframe():
    df = pd.DataFrame({"a": [1, 2]})
    assert df.equals([1, 2]) is False
    assert df.equals(None) is False
    assert df.equals(42) is False


def test_df_equals_multi_column_same():
    df1 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    df2 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    assert df1.equals(df2) is True


def test_df_equals_multi_column_diff():
    df1 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    df2 = pd.DataFrame({"x": [1, 2], "y": ["a", "Z"]})
    assert df1.equals(df2) is False


def test_df_equals_with_nans():
    # NaN should equal NaN in equals() (NaN-safe comparison)
    df1 = pd.DataFrame({"a": [1.0, None, 3.0]})
    df2 = pd.DataFrame({"a": [1.0, None, 3.0]})
    assert df1.equals(df2) is True


def test_df_equals_nan_vs_value():
    df1 = pd.DataFrame({"a": [1.0, None]})
    df2 = pd.DataFrame({"a": [1.0, 2.0]})
    assert df1.equals(df2) is False


# ---------------------------------------------------------------------------
# Series.equals
# ---------------------------------------------------------------------------

def test_series_equals_same():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([1, 2, 3], name="x")
    assert s1.equals(s2) is True


def test_series_equals_different_values():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([1, 2, 99], name="x")
    assert s1.equals(s2) is False


def test_series_equals_different_length():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([1, 2], name="x")
    assert s1.equals(s2) is False


def test_series_equals_non_series():
    s = pd.Series([1, 2], name="x")
    assert s.equals([1, 2]) is False


def test_series_equals_with_nans():
    s1 = pd.Series([1.0, None, 3.0], name="x")
    s2 = pd.Series([1.0, None, 3.0], name="x")
    assert s1.equals(s2) is True


# ---------------------------------------------------------------------------
# DataFrame.count
# ---------------------------------------------------------------------------

def test_df_count_returns_dict():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, 6.0]})
    result = df.count()
    assert isinstance(result, dict)
    assert result["a"] == 2
    assert result["b"] == 3


def test_df_count_all_nulls():
    df = pd.DataFrame({"a": [None, None, None]})
    result = df.count()
    assert result["a"] == 0


def test_df_count_no_nulls():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.count()
    assert result["a"] == 3
    assert result["b"] == 3


def test_df_count_mixed_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    result = df.count()
    assert result["a"] == 2
    assert result["b"] == 2


# ---------------------------------------------------------------------------
# Series.count
# ---------------------------------------------------------------------------

def test_series_count_with_nulls():
    s = pd.Series([1, None, 3, None, 5], name="x")
    assert s.count() == 3


def test_series_count_no_nulls():
    s = pd.Series([1, 2, 3, 4], name="x")
    assert s.count() == 4


def test_series_count_all_nulls():
    s = pd.Series([None, None, None], name="x")
    assert s.count() == 0


def test_series_count_empty():
    s = pd.Series([], name="x")
    assert s.count() == 0


# ---------------------------------------------------------------------------
# DataFrame.describe
# ---------------------------------------------------------------------------

def test_describe_numeric_has_stats_rows():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df.describe()
    # describe returns a DataFrame; check it has expected stats
    assert isinstance(result, pd.DataFrame)


def test_describe_count_row():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df.describe()
    rows = result.to_dict()
    # The 'a' column at index 0 (count) should be 5
    col_a = result["a"].tolist()
    assert col_a[0] == 5  # count


def test_describe_mean_row():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = df.describe()
    col_a = result["a"].tolist()
    # mean is row index 1
    assert abs(col_a[1] - 3.0) < 1e-9


def test_describe_min_max():
    df = pd.DataFrame({"a": [10, 20, 30]})
    result = df.describe()
    col_a = result["a"].tolist()
    # min and max should appear somewhere in describe output
    assert min(col_a) <= 10  # 10 is the actual minimum
    assert max(col_a) >= 30  # 30 is the actual maximum


def test_describe_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = df.describe()
    assert "a" in result.columns
    assert "b" in result.columns


def test_describe_count_correct_with_nulls():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    result = df.describe()
    col_a = result["a"].tolist()
    # count should be 2 (excludes nulls)
    assert col_a[0] == 2
