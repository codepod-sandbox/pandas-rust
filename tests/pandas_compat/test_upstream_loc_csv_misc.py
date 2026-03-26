"""Tests for loc bool mask, to_csv index param, value_counts normalize, sort_index, stack, rename_axis, merge indicator."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "z", "x", "y"]})


# ---------------------------------------------------------------------------
# loc with boolean Series mask (8 tests)
# ---------------------------------------------------------------------------

def test_loc_bool_series_filters_rows():
    df = make_df()
    mask = df["a"] > 2
    result = df.loc[mask]
    assert len(result) == 3


def test_loc_bool_series_col_returns_series():
    df = make_df()
    mask = df["a"] > 2
    result = df.loc[mask, "b"]
    assert list(result.tolist()) == [30, 40, 50]


def test_loc_bool_series_multicol_returns_dataframe():
    df = make_df()
    mask = df["a"] > 2
    result = df.loc[mask, ["a", "b"]]
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_loc_bool_mask_preserves_columns():
    df = make_df()
    mask = df["b"] >= 30
    result = df.loc[mask]
    assert set(result.columns) == {"a", "b", "c"}


def test_loc_bool_mask_preserves_values():
    df = make_df()
    mask = df["a"] == 3
    result = df.loc[mask]
    assert result["a"].tolist() == [3]
    assert result["b"].tolist() == [30]


def test_loc_all_true_mask():
    df = make_df()
    mask = df["a"] > 0
    result = df.loc[mask]
    assert len(result) == len(df)


def test_loc_all_false_mask_empty():
    df = make_df()
    mask = df["a"] > 100
    result = df.loc[mask]
    assert len(result) == 0


def test_loc_comparison_inline():
    df = make_df()
    result = df.loc[df["a"] > 2]
    assert len(result) == 3
    assert result["a"].tolist() == [3, 4, 5]


# ---------------------------------------------------------------------------
# to_csv (5 tests)
# ---------------------------------------------------------------------------

def test_to_csv_returns_string():
    df = make_df()
    result = df.to_csv()
    assert isinstance(result, str)


def test_to_csv_index_false_accepted():
    df = make_df()
    result = df.to_csv(index=False)
    assert isinstance(result, str)


def test_to_csv_to_file(tmp_path):
    df = make_df()
    path = str(tmp_path / "out.csv")
    df.to_csv(path)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "a" in content


def test_to_csv_roundtrip_preserves_data(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    path = str(tmp_path / "rt.csv")
    df.to_csv(path)
    df2 = pd.read_csv(path)
    assert list(df2["x"].tolist()) == [1, 2, 3]


def test_to_csv_string_contains_header():
    df = make_df()
    csv = df.to_csv()
    assert "a" in csv
    assert "b" in csv


# ---------------------------------------------------------------------------
# value_counts normalize (5 tests)
# ---------------------------------------------------------------------------

def test_value_counts_normalize_returns_fractions():
    s = pd.Series([1, 1, 2, 3])
    result = s.value_counts(normalize=True)
    counts = result["count"].tolist()
    # All values should be < 1
    assert all(c < 1.0 for c in counts)


def test_value_counts_normalize_sums_to_one():
    s = pd.Series(["a", "b", "a", "c", "a"])
    result = s.value_counts(normalize=True)
    total = sum(result["count"].tolist())
    assert abs(total - 1.0) < 1e-9


def test_value_counts_sort_default_descending():
    s = pd.Series([1, 1, 1, 2, 2, 3])
    result = s.value_counts()
    counts = result["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_value_counts_ascending():
    s = pd.Series([1, 1, 1, 2, 2, 3])
    result = s.value_counts(ascending=True)
    counts = result["count"].tolist()
    assert counts == sorted(counts)


def test_value_counts_dropna():
    s = pd.Series([1, 2, None, 1])
    result = s.value_counts(dropna=True)
    values = result["value"].tolist()
    assert None not in values


# ---------------------------------------------------------------------------
# sort_index (3 tests)
# ---------------------------------------------------------------------------

def test_sort_index_returns_dataframe():
    df = make_df()
    result = df.sort_index()
    assert isinstance(result, pd.DataFrame)


def test_sort_index_preserves_data():
    df = make_df()
    result = df.sort_index()
    assert result["a"].tolist() == df["a"].tolist()


def test_sort_index_ascending_param():
    df = make_df()
    result = df.sort_index(ascending=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# stack (3 tests)
# ---------------------------------------------------------------------------

def test_stack_flattens_columns():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    result = df.stack()
    assert hasattr(result, "tolist")


def test_stack_result_length():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = df.stack()
    assert len(result.tolist()) == 6  # 3 rows * 2 cols


def test_stack_preserves_values():
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    result = df.stack()
    vals = result.tolist()
    assert set(vals) == {10, 20, 30, 40}


# ---------------------------------------------------------------------------
# Integration patterns (8 tests)
# ---------------------------------------------------------------------------

def test_loc_then_groupby_agg():
    df = make_df()
    sub = df.loc[df["a"] > 2]
    grouped = sub.groupby("c")
    result = grouped["a"].sum()
    assert result is not None


def test_loc_col_then_value_counts():
    df = make_df()
    mask = df["a"] > 1
    s = df.loc[mask, "c"]
    result = s.value_counts()
    assert isinstance(result, pd.DataFrame)


def test_to_csv_read_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    path = str(tmp_path / "rtrip.csv")
    df.to_csv(path)
    df2 = pd.read_csv(path)
    assert list(df2["val"].tolist()) == ["a", "b", "c"]


def test_merge_with_indicator_doesnt_crash():
    df1 = pd.DataFrame({"key": [1, 2, 3], "v": [10, 20, 30]})
    df2 = pd.DataFrame({"key": [2, 3, 4], "w": [200, 300, 400]})
    result = df1.merge(df2, on="key", how="inner", indicator=True)
    assert isinstance(result, pd.DataFrame)


def test_rename_axis_doesnt_crash():
    df = make_df()
    result = df.rename_axis("my_index")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_loc_mask_then_sort_values():
    df = make_df()
    sub = df.loc[df["a"] > 2].sort_values("b")
    assert sub["b"].tolist() == sorted(sub["b"].tolist())


def test_masked_df_describe():
    df = make_df()
    sub = df.loc[df["a"] > 2]
    desc = sub.describe()
    assert desc is not None


def test_groupby_then_loc_on_result():
    df = make_df()
    result = df.loc[df["c"] == "x"]
    assert len(result) == 2
    assert all(v == "x" for v in result["c"].tolist())
