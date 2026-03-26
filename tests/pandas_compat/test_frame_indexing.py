"""Pandas compatibility tests for DataFrame indexing (iloc, loc, bool mask, concat)."""
import pandas as pd


def test_iloc_int():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1.0, 2.0, 3.0]})
    row = df.iloc[0]
    assert row["a"] == 10
    assert row["b"] == 1.0


def test_iloc_slice():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    sub = df.iloc[1:4]
    assert sub.shape == (3, 1)
    assert sub["a"].tolist() == [2, 3, 4]


def test_iloc_list():
    df = pd.DataFrame({"a": [10, 20, 30, 40]})
    sub = df.iloc[[0, 3]]
    assert sub.shape == (2, 1)
    assert sub["a"].tolist() == [10, 40]


def test_iloc_negative():
    df = pd.DataFrame({"a": [1, 2, 3]})
    row = df.iloc[-1]
    assert row["a"] == 3
    row2 = df.iloc[-2]
    assert row2["a"] == 2


def test_loc_int():
    df = pd.DataFrame({"a": [10, 20, 30]})
    row = df.loc[1]
    assert row["a"] == 20


def test_loc_slice_inclusive():
    df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
    # loc[1:3] is inclusive on both ends → rows 1,2,3 → values 20,30,40
    sub = df.loc[1:3]
    assert sub.shape == (3, 1)
    assert sub["a"].tolist() == [20, 30, 40]


def test_boolean_mask_filter():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    mask = df["a"] > 2
    sub = df[mask]
    assert sub.shape == (2, 2)
    assert sub["a"].tolist() == [3, 4]
    assert sub["b"].tolist() == [30, 40]


def test_concat_vertical():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    result = pd.concat([df1, df2])
    assert result.shape == (4, 2)
    assert result["a"].tolist() == [1, 2, 5, 6]
    assert result["b"].tolist() == [3, 4, 7, 8]


def test_concat_horizontal():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})
    result = pd.concat([df1, df2], axis=1)
    assert result.shape == (3, 2)
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == [4, 5, 6]


def test_from_dict_of_series():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([4, 5, 6], name="y")
    df = pd.DataFrame({"col1": s1, "col2": s2})
    assert df.shape == (3, 2)
    assert df["col1"].tolist() == [1, 2, 3]
    assert df["col2"].tolist() == [4, 5, 6]


def test_index_property():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    idx = df.index
    assert len(idx) == 4


def test_series_iloc_scalar():
    s = pd.Series([10, 20, 30], name="x")
    assert s.iloc[0] == 10
    assert s.iloc[2] == 30
    assert s.iloc[-1] == 30


def test_series_iloc_slice():
    s = pd.Series([10, 20, 30, 40], name="x")
    sub = s.iloc[1:3]
    assert sub.tolist() == [20, 30]


def test_series_iloc_list():
    s = pd.Series([10, 20, 30, 40], name="x")
    sub = s.iloc[[0, 2]]
    assert sub.tolist() == [10, 30]
