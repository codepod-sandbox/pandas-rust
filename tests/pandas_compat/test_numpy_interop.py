"""
Pandas compatibility tests for numpy interoperability.
Covers: Series.to_numpy(), Series.values, DataFrame.to_numpy(), DataFrame.values.
Requires numpy import to be available.
"""
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Series.to_numpy()
# ---------------------------------------------------------------------------

def test_series_to_numpy_returns_ndarray():
    s = pd.Series([1, 2, 3], name="x")
    arr = s.to_numpy()
    assert isinstance(arr, np.ndarray)


def test_series_to_numpy_int_values():
    s = pd.Series([1, 2, 3], name="x")
    arr = s.to_numpy()
    assert list(arr) == [1, 2, 3]


def test_series_to_numpy_float_values():
    s = pd.Series([1.5, 2.5, 3.5], name="x")
    arr = s.to_numpy()
    for a, b in zip(arr, [1.5, 2.5, 3.5]):
        assert abs(a - b) < 1e-10


def test_series_to_numpy_string_values():
    s = pd.Series(["hello", "world"], name="x")
    arr = s.to_numpy()
    assert list(arr) == ["hello", "world"]


def test_series_to_numpy_length():
    s = pd.Series([10, 20, 30, 40, 50], name="x")
    arr = s.to_numpy()
    assert len(arr) == 5


# ---------------------------------------------------------------------------
# Series.values
# ---------------------------------------------------------------------------

def test_series_values_returns_ndarray():
    s = pd.Series([1, 2, 3], name="x")
    v = s.values
    assert isinstance(v, np.ndarray)


def test_series_values_int_data():
    s = pd.Series([10, 20, 30], name="x")
    v = s.values
    assert list(v) == [10, 20, 30]


def test_series_values_float_data():
    s = pd.Series([1.1, 2.2, 3.3], name="x")
    v = s.values
    for a, b in zip(v, [1.1, 2.2, 3.3]):
        assert abs(a - b) < 1e-10


def test_series_values_same_as_to_numpy():
    s = pd.Series([1, 2, 3], name="x")
    assert list(s.values) == list(s.to_numpy())


# ---------------------------------------------------------------------------
# DataFrame.to_numpy()
# ---------------------------------------------------------------------------

def test_dataframe_to_numpy_returns_ndarray():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = df.to_numpy()
    assert isinstance(arr, np.ndarray)


def test_dataframe_to_numpy_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = df.to_numpy()
    assert arr.shape == (3, 2)


def test_dataframe_to_numpy_single_col_shape():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    arr = df.to_numpy()
    assert arr.shape == (5, 1)


def test_dataframe_to_numpy_values():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    arr = df.to_numpy()
    # First row should have values from row 0
    assert list(arr[0]) == [1, 3]
    assert list(arr[1]) == [2, 4]


# ---------------------------------------------------------------------------
# DataFrame.values
# ---------------------------------------------------------------------------

def test_dataframe_values_returns_ndarray():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    v = df.values
    assert isinstance(v, np.ndarray)


def test_dataframe_values_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    v = df.values
    assert v.shape == (3, 3)


def test_dataframe_values_same_as_to_numpy():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    v = df.values
    arr = df.to_numpy()
    assert v.shape == arr.shape
    for row_v, row_a in zip(v, arr):
        assert list(row_v) == list(row_a)
