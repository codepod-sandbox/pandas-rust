import pandas as pd


def test_merge_inner():
    left = pd.DataFrame({"key": [1, 2, 3], "val_l": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val_r": [200, 300, 400]})
    result = left.merge(right, on="key", how="inner")
    assert result.shape[0] == 2


def test_merge_left():
    left = pd.DataFrame({"key": [1, 2], "val_l": [10, 20]})
    right = pd.DataFrame({"key": [2, 3], "val_r": [200, 300]})
    result = left.merge(right, on="key", how="left")
    assert result.shape[0] == 2
