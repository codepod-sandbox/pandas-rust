"""
Upstream-adapted tests for fillna, dropna, isna, notna.
Source: pandas/tests/frame/methods/test_fillna.py
        pandas/tests/frame/methods/test_dropna.py
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame.fillna — scalar
# ---------------------------------------------------------------------------

def test_fillna_scalar_float():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    result = df.fillna(0.0)
    assert result["a"].tolist() == [1.0, 0.0, 3.0]
    assert result["b"].tolist() == [0.0, 2.0, 3.0]


def test_fillna_scalar_int_on_float_col():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = df.fillna(0)
    assert result["x"].tolist() == [1.0, 0.0, 3.0]


def test_fillna_preserves_non_null_values():
    df = pd.DataFrame({"a": [10.0, None, 30.0]})
    result = df.fillna(99.0)
    assert result["a"].tolist()[0] == 10.0
    assert result["a"].tolist()[2] == 30.0


def test_fillna_does_not_modify_original():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    original_vals = df["a"].tolist()
    _ = df.fillna(0.0)
    assert df["a"].tolist() == original_vals


def test_fillna_all_none():
    df = pd.DataFrame({"a": [None, None, None]})
    result = df.fillna(7.0)
    assert result["a"].tolist() == [7.0, 7.0, 7.0]


def test_fillna_no_nulls_unchanged():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df.fillna(99.0)
    assert result["a"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# DataFrame.fillna — dict (per-column values)
# ---------------------------------------------------------------------------

def test_fillna_dict_per_column():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, None]})
    result = df.fillna({"a": 0.0, "b": -1.0})
    assert result["a"].tolist() == [1.0, 0.0, 3.0]
    assert result["b"].tolist() == [-1.0, 2.0, -1.0]


def test_fillna_dict_partial_columns():
    df = pd.DataFrame({"a": [None, 2.0], "b": [None, 4.0]})
    result = df.fillna({"a": 99.0})
    assert result["a"].tolist() == [99.0, 2.0]
    # b should still have None
    b_vals = result["b"].tolist()
    assert b_vals[0] is None


# ---------------------------------------------------------------------------
# Series.fillna
# ---------------------------------------------------------------------------

def test_series_fillna_scalar():
    s = pd.Series([1.0, None, 3.0])
    result = s.fillna(0.0)
    assert result.tolist() == [1.0, 0.0, 3.0]


def test_series_fillna_no_effect_when_no_null():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.fillna(99.0)
    assert result.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# DataFrame.dropna — default (any)
# ---------------------------------------------------------------------------

def test_dropna_any_removes_rows_with_any_null():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    result = df.dropna()
    assert len(result) == 1
    assert result["a"].tolist() == [1.0]


def test_dropna_any_no_nulls_unchanged():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.dropna()
    assert len(result) == 3


def test_dropna_all_removes_only_fully_null_rows():
    df = pd.DataFrame({"a": [None, None, 3.0], "b": [None, 2.0, 4.0]})
    result = df.dropna(how="all")
    # Only row 0 (both a and b are None) should be dropped
    assert len(result) == 2


def test_dropna_with_subset():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 5.0, None]})
    # Drop rows where 'a' is null (subset=["a"])
    result = df.dropna(subset=["a"])
    assert len(result) == 2
    assert result["a"].tolist() == [1.0, 3.0]


def test_dropna_subset_string():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.dropna(subset="a")
    assert len(result) == 2


def test_dropna_preserves_columns():
    df = pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    result = df.dropna()
    assert list(result.columns) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Series.dropna
# ---------------------------------------------------------------------------

def test_series_dropna():
    s = pd.Series([1.0, None, 3.0, None, 5.0])
    result = s.dropna()
    assert result.tolist() == [1.0, 3.0, 5.0]


def test_series_dropna_no_nulls():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.dropna()
    assert result.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# isna / notna
# ---------------------------------------------------------------------------

def test_isna_returns_bool_dataframe():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, None]})
    result = df.isna()
    assert result.shape == (3, 2)
    assert result["a"].tolist() == [False, True, False]
    assert result["b"].tolist() == [True, False, True]


def test_notna_returns_bool_dataframe():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    result = df.notna()
    assert result["a"].tolist() == [True, False, True]


def test_isna_all_null():
    df = pd.DataFrame({"a": [None, None]})
    result = df.isna()
    assert all(result["a"].tolist())


def test_notna_no_nulls():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df.notna()
    assert all(result["a"].tolist())
