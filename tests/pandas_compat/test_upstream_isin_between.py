"""
Upstream-adapted tests for Series.isin, DataFrame.isin, Series.between.
Source: pandas/tests/series/methods/test_isin.py
        pandas/tests/frame/methods/test_isin.py
        pandas/tests/series/methods/test_between.py
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Series.isin
# ---------------------------------------------------------------------------

def test_series_isin_strings():
    s = pd.Series(["A", "B", "C", "a", "B", "B", "A", "C"])
    result = s.isin(["A", "C"])
    expected = [True, False, True, False, False, False, True, True]
    assert result.tolist() == expected


def test_series_isin_ints():
    s = pd.Series([1, 2, 3, 4, 5])
    result = s.isin([1, 3, 5])
    assert result.tolist() == [True, False, True, False, True]


def test_series_isin_empty_list_all_false():
    s = pd.Series([1, 2, 3])
    result = s.isin([])
    assert result.tolist() == [False, False, False]


def test_series_isin_all_match():
    s = pd.Series([10, 20, 30])
    result = s.isin([10, 20, 30])
    assert all(result.tolist())


def test_series_isin_no_match():
    s = pd.Series([1, 2, 3])
    result = s.isin([99, 100])
    assert not any(result.tolist())


def test_series_isin_preserves_index_length():
    s = pd.Series(["x", "y", "z", "x"])
    result = s.isin(["x"])
    assert len(result) == len(s)


def test_series_isin_with_none():
    s = pd.Series([1.0, None, 3.0])
    result = s.isin([1.0, 3.0])
    # None is not in the list, so it should be False
    vals = result.tolist()
    assert vals[0] is True
    assert vals[2] is True


# ---------------------------------------------------------------------------
# DataFrame.isin
# ---------------------------------------------------------------------------

def test_dataframe_isin_list():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.isin([1, 3, 5])
    assert result.shape == (3, 2)
    assert result["a"].tolist() == [True, False, True]
    assert result["b"].tolist() == [False, True, False]


def test_dataframe_isin_dict_per_column():
    df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})
    result = df.isin({"A": ["a"]})
    assert result.shape == (3, 2)
    # Only A[0] should be True
    assert result["A"].tolist() == [True, False, False]
    # B column not in dict, so all False
    assert result["B"].tolist() == [False, False, False]


def test_dataframe_isin_empty_list():
    df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})
    result = df.isin([])
    assert result.shape == (3, 2)
    assert not any(result["A"].tolist())
    assert not any(result["B"].tolist())


def test_dataframe_isin_returns_bool_df():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = df.isin([2])
    assert result["x"].tolist() == [False, True, False]


def test_dataframe_isin_shape_preserved():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = df.isin([1, 5, 9])
    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Series.between
# ---------------------------------------------------------------------------

def test_between_inclusive_both():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.between(2.0, 4.0)
    assert result.tolist() == [False, True, True, True, False]


def test_between_exclusive():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.between(2.0, 4.0, inclusive="neither")
    assert result.tolist() == [False, False, True, False, False]


def test_between_equal_bounds():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.between(2.0, 2.0)
    assert result.tolist() == [False, True, False]


def test_between_all_below():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.between(10.0, 20.0)
    assert not any(result.tolist())


def test_between_all_above():
    s = pd.Series([10.0, 20.0, 30.0])
    result = s.between(1.0, 5.0)
    assert not any(result.tolist())


def test_between_entire_range():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.between(1.0, 5.0)
    assert all(result.tolist())


def test_between_float_series():
    s = pd.Series([0.1, 0.5, 0.9, 1.5, 2.0])
    result = s.between(0.4, 1.0)
    assert result.tolist() == [False, True, True, False, False]


def test_between_preserves_length():
    s = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = s.between(3.0, 7.0)
    assert len(result) == len(s)
