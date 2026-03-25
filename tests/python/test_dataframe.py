import pandas as pd


def test_dataframe_from_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert df.shape == (3, 2)
    assert list(df.columns) == ["a", "b"]


def test_dataframe_empty():
    df = pd.DataFrame({})
    assert df.shape == (0, 0)


def test_dataframe_head_tail():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    h = df.head(3)
    assert h.shape == (3, 1)
    t = df.tail(2)
    assert t.shape == (2, 1)


def test_dataframe_column_access():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    s = df["a"]
    assert s.name == "a"
    assert len(s) == 2


def test_dataframe_set_column():
    df = pd.DataFrame({"a": [1, 2]})
    df["b"] = [3, 4]
    assert df.shape == (2, 2)


def test_dataframe_drop():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    df2 = df.drop(columns=["b"])
    assert list(df2.columns) == ["a", "c"]


def test_dataframe_rename():
    df = pd.DataFrame({"a": [1], "b": [2]})
    df2 = df.rename(columns={"a": "alpha"})
    assert "alpha" in df2.columns


def test_dataframe_sort_values():
    df = pd.DataFrame({"a": [3, 1, 2]})
    df2 = df.sort_values("a")
    vals = df2["a"].tolist()
    assert vals == [1, 2, 3]


def test_dataframe_copy():
    df = pd.DataFrame({"a": [1, 2]})
    df2 = df.copy()
    assert df2.shape == df.shape


def test_dataframe_aggregations():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    s = df.sum()
    assert isinstance(s, dict)


def test_dataframe_repr():
    df = pd.DataFrame({"a": [1, 2]})
    r = repr(df)
    assert "a" in r


def test_dataframe_to_dict():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    d = df.to_dict()
    assert "a" in d
    assert "b" in d


def test_dataframe_to_csv():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    csv_str = df.to_csv()
    assert "a" in csv_str
    assert "b" in csv_str


def test_dataframe_describe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    desc = df.describe()
    assert desc.shape[0] > 0


def test_dataframe_fillna():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df.fillna(0)
    assert df2.shape == df.shape


def test_dataframe_isna():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.isna()
    assert result.shape == df.shape


def test_dataframe_len():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3
