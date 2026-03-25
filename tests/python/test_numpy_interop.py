import pandas as pd
import numpy as np


def test_series_to_numpy():
    s = pd.Series([1, 2, 3], name="x")
    arr = s.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]


def test_series_values_property():
    s = pd.Series([4.0, 5.0, 6.0], name="y")
    arr = s.values
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [4.0, 5.0, 6.0]


def test_dataframe_to_numpy():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    arr = df.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_dataframe_values_property():
    df = pd.DataFrame({"x": [10, 20], "y": [30.0, 40.0]})
    arr = df.values
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
