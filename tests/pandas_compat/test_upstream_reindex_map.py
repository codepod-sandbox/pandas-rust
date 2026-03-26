"""Tests for reindex, DataFrame.map, comparison methods, groupby.apply, patterns."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import pandas as our_pd


# ---------------------------------------------------------------------------
# DataFrame.reindex
# ---------------------------------------------------------------------------

def test_reindex_existing_columns_same_data():
    df = our_pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.reindex(columns=["a", "b"])
    assert result["a"].tolist() == [1, 2]
    assert result["b"].tolist() == [3, 4]


def test_reindex_new_column_filled_none():
    df = our_pd.DataFrame({"a": [1, 2]})
    result = df.reindex(columns=["a", "c"])
    assert result["a"].tolist() == [1, 2]
    assert result["c"].tolist() == [None, None]


def test_reindex_reorders_columns():
    df = our_pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = df.reindex(columns=["c", "a", "b"])
    assert list(result.columns) == ["c", "a", "b"]


def test_reindex_drops_columns_not_listed():
    df = our_pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = df.reindex(columns=["a"])
    assert list(result.columns) == ["a"]
    assert "b" not in result.columns


def test_reindex_preserves_row_count():
    df = our_pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.reindex(columns=["a", "b", "new"])
    assert len(result) == 3


# ---------------------------------------------------------------------------
# DataFrame.map — element-wise (pandas 2.1+ alias for applymap)
# ---------------------------------------------------------------------------

def test_df_map_doubles_values():
    df = our_pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.map(lambda x: x * 2)
    assert result["a"].tolist() == [2, 4, 6]
    assert result["b"].tolist() == [8, 10, 12]


def test_df_map_preserves_shape():
    df = our_pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    result = df.map(lambda x: x + 1)
    assert result.shape == df.shape


def test_df_map_preserves_column_names():
    df = our_pd.DataFrame({"foo": [1], "bar": [2]})
    result = df.map(lambda x: x * 0)
    assert list(result.columns) == ["foo", "bar"]


def test_df_map_type_conversion():
    df = our_pd.DataFrame({"a": [1, 2, 3]})
    result = df.map(lambda x: float(x))
    vals = result["a"].tolist()
    assert all(isinstance(v, float) for v in vals)


# ---------------------------------------------------------------------------
# DataFrame comparison methods: eq, ne, lt, le, gt, ge
# ---------------------------------------------------------------------------

def test_df_eq_other_df():
    df1 = our_pd.DataFrame({"a": [1, 2, 3]})
    df2 = our_pd.DataFrame({"a": [1, 0, 3]})
    result = df1.eq(df2)
    assert result["a"].tolist() == [True, False, True]


def test_df_ne_opposite_of_eq():
    df1 = our_pd.DataFrame({"a": [1, 2, 3]})
    df2 = our_pd.DataFrame({"a": [1, 2, 0]})
    eq_res = df1.eq(df2)["a"].tolist()
    ne_res = df1.ne(df2)["a"].tolist()
    for e, n in zip(eq_res, ne_res):
        assert e != n


def test_df_gt_scalar():
    df = our_pd.DataFrame({"a": [1, 5, 3], "b": [2, 4, 6]})
    result = df.gt(3)
    assert result["a"].tolist() == [False, True, False]
    assert result["b"].tolist() == [False, True, True]


def test_df_lt_scalar():
    df = our_pd.DataFrame({"a": [1, 5, 3]})
    result = df.lt(3)
    assert result["a"].tolist() == [True, False, False]


def test_df_comparison_preserves_shape():
    df = our_pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.ge(3)
    assert result.shape == df.shape


# ---------------------------------------------------------------------------
# GroupBy.apply
# ---------------------------------------------------------------------------

def test_groupby_apply_sum_per_group():
    df = our_pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    result = df.groupby("cat").apply(lambda sub: sub["val"].sum())
    # result should have two entries, one per group
    assert len(result) == 2


def test_groupby_apply_returns_combined():
    df = our_pd.DataFrame({"cat": ["x", "x", "y"], "val": [10, 20, 30]})
    result = df.groupby("cat").apply(lambda sub: sub["val"].sum())
    vals = result["result"].tolist() if "result" in result.columns else list(result.values()) if hasattr(result, "values") else result.tolist()
    assert 30 in vals
    assert 30 in vals  # sum of x: 30


def test_groupby_apply_lambda():
    df = our_pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]})
    result = df.groupby("g").apply(lambda x: x["v"].max())
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Additional patterns
# ---------------------------------------------------------------------------

def test_rename_columns_str_upper():
    df = our_pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    result = df.rename(columns=str.upper)
    assert "FOO" in result.columns
    assert "BAR" in result.columns


def test_add_prefix_then_filter():
    df = our_pd.DataFrame({"a": [1], "b": [2]})
    prefixed = df.add_prefix("col_")
    assert "col_a" in prefixed.columns
    assert "col_b" in prefixed.columns


def test_pipe_chaining():
    df = our_pd.DataFrame({"a": [1, 2, 3]})
    result = df.pipe(lambda d: d.assign(b=d["a"] * 2)).pipe(lambda d: d.assign(c=d["b"] + 1))
    assert result["b"].tolist() == [2, 4, 6]
    assert result["c"].tolist() == [3, 5, 7]
