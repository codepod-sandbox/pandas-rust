"""Tests for bool ops, df.attr, iloc/loc setitem, transform, pivot_table, agg, Series from scalar."""
import pytest
import pandas as pd


# ---- Bool ops (& | ~) ----

def test_bool_and():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    m1 = df["a"] > 1
    m2 = df["b"] < 30
    result = (m1 & m2).tolist()
    assert result == [False, True, False]


def test_bool_or():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    m1 = df["a"] > 2
    m2 = df["b"] < 20
    result = (m1 | m2).tolist()
    assert result == [True, False, True]


def test_bool_invert():
    df = pd.DataFrame({"a": [1, 2, 3]})
    m = df["a"] > 2
    result = (~m).tolist()
    assert result == [True, True, False]


def test_bool_and_filter_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    filtered = df[(df["a"] > 1) & (df["b"] < 40)]
    assert len(filtered) == 2
    assert filtered["a"].tolist() == [2, 3]


def test_bool_all_true():
    df = pd.DataFrame({"a": [1, 2, 3]})
    m1 = df["a"] > 0
    m2 = df["a"] < 10
    result = (m1 & m2).tolist()
    assert result == [True, True, True]


def test_bool_all_false():
    df = pd.DataFrame({"a": [1, 2, 3]})
    m1 = df["a"] > 5
    m2 = df["a"] < 0
    result = (m1 | m2).tolist()
    assert result == [False, False, False]


# ---- Attribute access (df.col) ----

def test_df_attr_access_returns_series():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = df.a
    assert s.tolist() == [1, 2, 3]


def test_df_attr_access_matches_getitem():
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    assert df.x.tolist() == df["x"].tolist()
    assert df.y.tolist() == df["y"].tolist()


def test_df_missing_attr_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(AttributeError):
        _ = df.nonexistent_column


# ---- iloc/loc setitem ----

def test_iloc_setitem_single_cell():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.iloc[0, 0] = 99
    assert df["a"].tolist()[0] == 99
    assert df["a"].tolist()[1] == 2  # other values unchanged


def test_iloc_setitem_by_col_index():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.iloc[1, 1] = 99
    assert df["b"].tolist()[1] == 99
    assert df["b"].tolist()[0] == 10


def test_iloc_setitem_row_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.iloc[0] = {"a": 99, "b": 88}
    assert df["a"].tolist()[0] == 99
    assert df["b"].tolist()[0] == 88
    assert df["a"].tolist()[1] == 2  # unchanged


def test_loc_setitem_single_cell():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.loc[0, "a"] = 55
    assert df["a"].tolist()[0] == 55
    assert df["a"].tolist()[1] == 2


def test_setitem_preserves_other_values():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.iloc[2, 0] = 100
    assert df["a"].tolist() == [1, 2, 100]
    assert df["b"].tolist() == [10, 20, 30]  # b untouched


def test_iloc_setitem_negative_index():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.iloc[-1, 0] = 99
    assert df["a"].tolist()[-1] == 99
    assert df["a"].tolist()[0] == 1


# ---- transform ----

def test_transform_sum_broadcasts():
    df = pd.DataFrame({"group": ["a", "a", "b"], "val": [1, 2, 3]})
    result = df.groupby("group")["val"].transform("sum")
    # group a: sum=3, group b: sum=3
    assert result.tolist() == [3, 3, 3]


def test_transform_preserves_length():
    df = pd.DataFrame({"group": ["x", "x", "y", "y"], "val": [10, 20, 30, 40]})
    result = df.groupby("group")["val"].transform("sum")
    assert len(result) == 4


def test_transform_mean():
    df = pd.DataFrame({"group": ["a", "a", "b", "b"], "val": [1.0, 3.0, 2.0, 4.0]})
    result = df.groupby("group")["val"].transform("mean")
    vals = result.tolist()
    assert vals[0] == 2.0  # mean of a: (1+3)/2
    assert vals[1] == 2.0
    assert vals[2] == 3.0  # mean of b: (2+4)/2
    assert vals[3] == 3.0


def test_transform_different_groups():
    df = pd.DataFrame({"group": ["a", "b", "a", "b"], "val": [10, 20, 30, 40]})
    result = df.groupby("group")["val"].transform("sum")
    vals = result.tolist()
    # a sum=40, b sum=60
    assert vals[0] == 40
    assert vals[1] == 60
    assert vals[2] == 40
    assert vals[3] == 60


# ---- pivot_table ----

def test_pivot_table_sum():
    df = pd.DataFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
    result = df.pivot_table(values="val", index="cat", aggfunc="sum")
    vals = result["val"].tolist()
    assert 3 in vals  # a: 1+2=3
    assert 3 in vals  # b: 3


def test_pivot_table_mean():
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 3.0, 2.0, 4.0]})
    result = df.pivot_table(values="val", index="cat", aggfunc="mean")
    vals = result["val"].tolist()
    assert 2.0 in vals  # a mean
    assert 3.0 in vals  # b mean


def test_pivot_table_as_groupby():
    df = pd.DataFrame({"cat": ["x", "y", "x"], "val": [10, 20, 30]})
    pt = df.pivot_table(values="val", index="cat", aggfunc="sum")
    gb = df.groupby("cat")["val"].sum()
    # Both should give same results
    assert sorted(pt["val"].tolist()) == sorted(gb.tolist())


# ---- agg ----

def test_df_agg_str():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.agg("sum")
    # Returns dict or similar with sum per column
    # Native sum returns a dict
    assert isinstance(result, dict) or hasattr(result, "__getitem__")


def test_df_agg_list():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.agg(["sum", "mean"])
    from pandas.core.frame import DataFrame
    assert isinstance(result, DataFrame)
    assert result["a"].tolist()[0] == 6.0  # sum of a
    assert result["b"].tolist()[0] == 15.0  # sum of b


def test_df_agg_dict():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.agg({"a": "sum", "b": "mean"})
    from pandas.core.frame import DataFrame
    assert isinstance(result, DataFrame)
    assert result["a"].tolist()[0] == 6.0  # sum of a
    assert result["b"].tolist()[0] == 5.0  # mean of b


# ---- Series from scalar ----

def test_series_from_int_scalar():
    s = pd.Series(5)
    assert s.tolist() == [5]


def test_series_from_float_scalar_with_index():
    s = pd.Series(1.0, index=[0, 1, 2])
    assert s.tolist() == [1.0, 1.0, 1.0]
    assert len(s) == 3


def test_series_from_str_scalar():
    s = pd.Series("hello")
    assert s.tolist() == ["hello"]


def test_series_from_bool_scalar():
    s = pd.Series(True)
    assert s.tolist() == [True]


def test_series_from_scalar_with_index():
    s = pd.Series(42, index=list(range(5)))
    assert len(s) == 5
    assert all(v == 42 for v in s.tolist())
