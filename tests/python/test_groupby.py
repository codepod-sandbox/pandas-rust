import pandas as pd


def test_groupby_sum():
    df = pd.DataFrame({"key": ["a", "b", "a", "b"], "val": [1, 2, 3, 4]})
    result = df.groupby("key").sum()
    assert result.shape[0] == 2


def test_groupby_mean():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1.0, 3.0, 5.0]})
    result = df.groupby("key").mean()
    assert result.shape[0] == 2


def test_groupby_count():
    df = pd.DataFrame({"key": ["x", "y", "x"], "val": [1, 2, 3]})
    result = df.groupby("key").count()
    assert result.shape[0] == 2


def test_groupby_min_max():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [10, 20, 30]})
    assert df.groupby("key").min().shape[0] == 2
    assert df.groupby("key").max().shape[0] == 2


def test_groupby_first_last():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [10, 20, 30]})
    assert df.groupby("key").first().shape[0] == 2
    assert df.groupby("key").last().shape[0] == 2
