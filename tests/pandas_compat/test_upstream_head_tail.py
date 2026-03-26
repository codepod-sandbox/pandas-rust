"""
Upstream-adapted tests for DataFrame.head() and DataFrame.tail().
Source: pandas/tests/frame/methods/test_head_tail.py
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame head tests
# ---------------------------------------------------------------------------

def test_head_default_returns_5_rows():
    df = pd.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
    result = df.head()
    assert result.shape == (5, 2)
    assert result["a"].tolist() == [0, 1, 2, 3, 4]


def test_head_explicit_n():
    df = pd.DataFrame({"x": list(range(8))})
    result = df.head(3)
    assert len(result) == 3
    assert result["x"].tolist() == [0, 1, 2]


def test_head_zero_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.head(0)
    assert result.shape == (0, 2)
    assert list(result.columns) == ["a", "b"]


def test_head_n_greater_than_len_returns_full():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.head(100)
    assert len(result) == 3
    assert result["a"].tolist() == [1, 2, 3]


def test_head_negative_n_all_but_last():
    # head(-1) should return all rows except the last one
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df.head(-1)
    assert result["a"].tolist() == [1, 2, 3, 4]


def test_head_negative_2():
    df = pd.DataFrame({"a": list(range(6))})
    result = df.head(-2)
    expected = df.iloc[: len(df) - 2]
    assert result["a"].tolist() == expected["a"].tolist()


def test_head_negative_equals_iloc():
    df = pd.DataFrame({"v": list(range(10))})
    result = df.head(-3)
    expected = df.iloc[: len(df) - 3]
    assert result["v"].tolist() == expected["v"].tolist()


# ---------------------------------------------------------------------------
# DataFrame tail tests
# ---------------------------------------------------------------------------

def test_tail_default_returns_5_rows():
    df = pd.DataFrame({"a": list(range(10))})
    result = df.tail()
    assert result.shape == (5, 1)
    assert result["a"].tolist() == [5, 6, 7, 8, 9]


def test_tail_explicit_n():
    df = pd.DataFrame({"x": list(range(8))})
    result = df.tail(3)
    assert len(result) == 3
    assert result["x"].tolist() == [5, 6, 7]


def test_tail_zero_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.tail(0)
    assert result.shape == (0, 2)


def test_tail_n_greater_than_len_returns_full():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.tail(100)
    assert len(result) == 3


def test_tail_negative_n_skips_first():
    # tail(-1) should return all rows except the first one
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df.tail(-1)
    assert result["a"].tolist() == [2, 3, 4, 5]


def test_tail_negative_2():
    df = pd.DataFrame({"a": list(range(6))})
    result = df.tail(-2)
    expected = df.iloc[2:]
    assert result["a"].tolist() == expected["a"].tolist()


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------

def test_head_empty_dataframe():
    df = pd.DataFrame()
    result = df.head()
    assert result.shape == (0, 0)


def test_tail_empty_dataframe():
    df = pd.DataFrame()
    result = df.tail()
    assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# Series head / tail
# ---------------------------------------------------------------------------

def test_series_head_default():
    s = pd.Series(list(range(10)), name="s")
    result = s.head()
    assert len(result) == 5
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_series_head_n():
    s = pd.Series([10, 20, 30, 40, 50])
    result = s.head(2)
    assert result.tolist() == [10, 20]


def test_series_head_zero():
    s = pd.Series([1, 2, 3])
    result = s.head(0)
    assert len(result) == 0


def test_series_tail_default():
    s = pd.Series(list(range(10)))
    result = s.tail()
    assert result.tolist() == [5, 6, 7, 8, 9]


def test_series_tail_n():
    s = pd.Series([10, 20, 30, 40, 50])
    result = s.tail(2)
    assert result.tolist() == [40, 50]


def test_series_tail_zero():
    s = pd.Series([1, 2, 3])
    result = s.tail(0)
    assert len(result) == 0


def test_series_head_negative():
    s = pd.Series(list(range(5)))
    result = s.head(-1)
    assert result.tolist() == [0, 1, 2, 3]


def test_series_tail_negative():
    s = pd.Series(list(range(5)))
    result = s.tail(-1)
    assert result.tolist() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Preserves column structure
# ---------------------------------------------------------------------------

def test_head_preserves_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = df.head(2)
    assert list(result.columns) == ["a", "b", "c"]


def test_tail_preserves_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = df.tail(1)
    assert list(result.columns) == ["x", "y"]
    assert result["x"].tolist() == [3]
