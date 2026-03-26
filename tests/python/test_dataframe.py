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


def test_from_list_of_dicts():
    data = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
    df = pd.DataFrame(data)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1, 2, 3]
    assert df["b"].tolist() == [10, 20, 30]


def test_from_empty_list():
    df = pd.DataFrame([])
    assert df.shape == (0, 0)


def test_from_list_of_dicts_missing_keys():
    data = [{"a": 1}, {"a": 2, "b": 20}]
    df = pd.DataFrame(data)
    assert df.shape == (2, 2)
    assert "a" in list(df.columns)
    assert "b" in list(df.columns)


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": [10, 20, 10, 30]})
    df2 = df.drop_duplicates()
    assert df2.shape == (3, 2)


def test_drop_duplicates_subset():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    df2 = df.drop_duplicates(subset=["a"])
    assert df2.shape == (2, 2)


def test_duplicated():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [10, 20, 10]})
    d = df.duplicated()
    assert d.tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# iterrows / itertuples
# ---------------------------------------------------------------------------

def test_iterrows():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    rows = list(df.iterrows())
    assert len(rows) == 2
    assert rows[0][0] == 0  # index
    assert rows[0][1]["a"] == 1
    assert rows[0][1]["b"] == 3.0
    assert rows[1][0] == 1
    assert rows[1][1]["a"] == 2


def test_itertuples():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    tuples = list(df.itertuples())
    assert len(tuples) == 2
    assert tuples[0][0] == 0  # index
    assert tuples[0][1] == 1  # a
    assert tuples[0][2] == 3.0  # b


def test_itertuples_no_index():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    tuples = list(df.itertuples(index=False))
    assert tuples[0][0] == 1  # a (no index)
    assert tuples[0][1] == 3.0  # b


# ---------------------------------------------------------------------------
# apply / applymap
# ---------------------------------------------------------------------------

def test_apply_axis0():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = df.apply(lambda s: s.sum(), axis=0)
    assert isinstance(result, dict)
    assert result["a"] == 6
    assert result["b"] == 15.0


def test_apply_axis1():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = df.apply(lambda row: row["a"] + row["b"], axis=1)
    assert result.tolist() == [11, 22, 33]


def test_applymap():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.applymap(lambda x: x * 2)
    assert result["a"].tolist() == [2, 4]


def test_iloc_slice():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    sub = df.iloc[1:3]
    assert sub.shape == (2, 1)
    assert sub["a"].tolist() == [2, 3]


def test_iloc_single_row():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1.0, 2.0, 3.0]})
    row = df.iloc[0]
    assert row["a"] == 10
    assert row["b"] == 1.0


def test_iloc_negative():
    df = pd.DataFrame({"a": [1, 2, 3]})
    row = df.iloc[-1]
    assert row["a"] == 3


def test_iloc_list():
    df = pd.DataFrame({"a": [10, 20, 30, 40]})
    sub = df.iloc[[0, 2]]
    assert sub.shape == (2, 1)
    assert sub["a"].tolist() == [10, 30]


def test_loc_slice():
    df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
    sub = df.loc[1:3]  # inclusive both ends
    assert sub.shape == (3, 1)
    assert sub["a"].tolist() == [20, 30, 40]


def test_loc_single():
    df = pd.DataFrame({"a": [10, 20, 30]})
    row = df.loc[1]
    assert row["a"] == 20


def test_boolean_mask_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    mask = df["a"].gt(1)
    sub = df[mask]
    assert sub.shape == (2, 2)
    assert sub["a"].tolist() == [2, 3]


def test_concat_basic():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    result = pd.concat([df1, df2])
    assert result.shape == (4, 1)
    assert result["a"].tolist() == [1, 2, 3, 4]


def test_from_dict_of_series():
    s1 = pd.Series([1, 2, 3], name="x")
    s2 = pd.Series([4, 5, 6], name="y")
    df = pd.DataFrame({"a": s1, "b": s2})
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 2, 3]


def test_index_property():
    df = pd.DataFrame({"a": [1, 2, 3]})
    idx = df.index
    assert len(idx) == 3


def test_series_iloc():
    s = pd.Series([10, 20, 30], name="x")
    assert s.iloc[0] == 10
    assert s.iloc[-1] == 30
    sub = s.iloc[0:2]
    assert sub.tolist() == [10, 20]
