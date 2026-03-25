import pandas as pd


def test_series_from_list():
    s = pd.Series([1, 2, 3], name="x")
    assert s.name == "x"
    assert len(s) == 3


def test_series_arithmetic():
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = s1 + s2
    assert result.tolist() == [5, 7, 9]


def test_series_subtraction():
    s1 = pd.Series([10, 20, 30], name="a")
    s2 = pd.Series([1, 2, 3], name="b")
    result = s1 - s2
    assert result.tolist() == [9, 18, 27]


def test_series_multiplication():
    s = pd.Series([2, 3, 4], name="x")
    s2 = pd.Series([5, 6, 7], name="y")
    result = s * s2
    assert result.tolist() == [10, 18, 28]


def test_series_comparison():
    s = pd.Series([1, 2, 3], name="x")
    result = s.gt(2)
    assert result.tolist() == [False, False, True]


def test_series_aggregations():
    s = pd.Series([1, 2, 3], name="x")
    assert s.sum() == 6
    assert s.mean() == 2.0
    assert s.min() == 1
    assert s.max() == 3
    assert s.count() == 3


def test_series_sort():
    s = pd.Series([3, 1, 2], name="x")
    result = s.sort_values()
    assert result.tolist() == [1, 2, 3]


def test_series_tolist():
    s = pd.Series([1, 2, 3], name="x")
    assert s.tolist() == [1, 2, 3]


def test_series_negation():
    s = pd.Series([1, -2, 3], name="x")
    result = -s
    assert result.tolist() == [-1, 2, -3]


def test_series_copy():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.copy()
    assert s2.tolist() == [1, 2, 3]
    assert s2.name == "x"
