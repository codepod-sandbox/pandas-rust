"""Tests for diff, pct_change, rank, round, rolling, expanding, nunique, melt."""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# diff (4 tests)
# ---------------------------------------------------------------------------

def test_diff_basic():
    s = pd.Series([1, 3, 6, 10])
    result = s.diff(1).tolist()
    assert result[0] is None
    assert result[1] == 2
    assert result[2] == 3
    assert result[3] == 4


def test_diff_periods2():
    s = pd.Series([1, 2, 5, 9])
    result = s.diff(2).tolist()
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 4
    assert result[3] == 7


def test_diff_float():
    s = pd.Series([1.0, 1.5, 3.0])
    result = s.diff(1).tolist()
    assert result[0] is None
    assert abs(result[1] - 0.5) < 1e-9
    assert abs(result[2] - 1.5) < 1e-9


def test_diff_first_elements_null():
    s = pd.Series([10, 20, 30, 40, 50])
    result = s.diff(3).tolist()
    # First 3 are null
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None
    assert result[3] == 30
    assert result[4] == 30


# ---------------------------------------------------------------------------
# pct_change (3 tests)
# ---------------------------------------------------------------------------

def test_pct_change_basic():
    s = pd.Series([100.0, 110.0, 121.0])
    result = s.pct_change(1).tolist()
    assert result[0] is None
    assert abs(result[1] - 0.1) < 1e-9
    assert abs(result[2] - 0.1) < 1e-9


def test_pct_change_first_is_null():
    s = pd.Series([5, 10, 15])
    result = s.pct_change().tolist()
    assert result[0] is None


def test_pct_change_zero_division():
    s = pd.Series([0, 5, 10])
    result = s.pct_change().tolist()
    # prev=0 → null (zero division)
    assert result[1] is None


# ---------------------------------------------------------------------------
# rank (4 tests)
# ---------------------------------------------------------------------------

def test_rank_basic_ascending():
    s = pd.Series([3, 1, 4, 1, 5])
    result = s.rank(method="average", ascending=True).tolist()
    # Values sorted: 1(i1), 1(i3), 3(i0), 4(i2), 5(i4)
    # rank 1,2 -> avg 1.5 for 1s; rank 3 for 3; rank 4 for 4; rank 5 for 5
    assert result[0] == 3.0
    assert result[1] == 1.5
    assert result[2] == 4.0
    assert result[3] == 1.5
    assert result[4] == 5.0


def test_rank_descending():
    s = pd.Series([1, 2, 3])
    result = s.rank(method="average", ascending=False).tolist()
    # Descending: 3->1, 2->2, 1->3
    assert result[0] == 3.0
    assert result[1] == 2.0
    assert result[2] == 1.0


def test_rank_ties_average():
    s = pd.Series([1, 1, 1])
    result = s.rank(method="average").tolist()
    # All tied: ranks 1,2,3 -> avg 2.0
    assert all(r == 2.0 for r in result)


def test_rank_nulls_excluded():
    s = pd.Series([1, None, 3])
    result = s.rank(method="average").tolist()
    # None stays None
    assert result[1] is None
    assert result[0] == 1.0
    assert result[2] == 2.0


# ---------------------------------------------------------------------------
# round (4 tests)
# ---------------------------------------------------------------------------

def test_round_float_zero_decimals():
    s = pd.Series([1.4, 1.5, 2.7, -0.5])
    result = s.round(0).tolist()
    assert result[0] == 1.0
    assert result[2] == 3.0


def test_round_float_one_decimal():
    s = pd.Series([1.23, 4.56, 7.89])
    result = s.round(1).tolist()
    assert abs(result[0] - 1.2) < 1e-9
    assert abs(result[1] - 4.6) < 1e-9
    assert abs(result[2] - 7.9) < 1e-9


def test_round_float_two_decimals():
    s = pd.Series([1.234, 5.678])
    result = s.round(2).tolist()
    assert abs(result[0] - 1.23) < 1e-9


def test_round_dataframe():
    df = pd.DataFrame({"a": [1.111, 2.222], "b": [3.555, 4.666]})
    result = df.round(1)
    a_vals = result["a"].tolist()
    b_vals = result["b"].tolist()
    assert abs(a_vals[0] - 1.1) < 1e-9
    assert abs(b_vals[1] - 4.7) < 1e-9


def test_round_int_noop():
    s = pd.Series([1, 2, 3])
    result = s.round(2).tolist()
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# rolling (5 tests)
# ---------------------------------------------------------------------------

def test_rolling_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.rolling(2).mean().tolist()
    assert result[0] is None
    assert abs(result[1] - 1.5) < 1e-9
    assert abs(result[2] - 2.5) < 1e-9
    assert abs(result[4] - 4.5) < 1e-9


def test_rolling_sum():
    s = pd.Series([1, 2, 3, 4, 5])
    result = s.rolling(3).sum().tolist()
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 6
    assert result[3] == 9
    assert result[4] == 12


def test_rolling_min():
    s = pd.Series([3, 1, 4, 1, 5])
    result = s.rolling(2).min().tolist()
    assert result[0] is None
    assert result[1] == 1
    assert result[2] == 1
    assert result[3] == 1
    assert result[4] == 1


def test_rolling_max():
    s = pd.Series([3, 1, 4, 1, 5])
    result = s.rolling(2).max().tolist()
    assert result[0] is None
    assert result[1] == 3
    assert result[2] == 4
    assert result[3] == 4
    assert result[4] == 5


def test_rolling_first_values_null():
    s = pd.Series([1, 2, 3, 4])
    # window=3: first 2 are null
    result = s.rolling(3).mean().tolist()
    assert result[0] is None
    assert result[1] is None
    assert result[2] is not None


# ---------------------------------------------------------------------------
# expanding (3 tests)
# ---------------------------------------------------------------------------

def test_expanding_sum():
    s = pd.Series([1, 2, 3, 4])
    result = s.expanding().sum().tolist()
    assert result[0] == 1
    assert result[1] == 3
    assert result[2] == 6
    assert result[3] == 10


def test_expanding_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = s.expanding().mean().tolist()
    assert abs(result[0] - 1.0) < 1e-9
    assert abs(result[1] - 1.5) < 1e-9
    assert abs(result[2] - 2.0) < 1e-9
    assert abs(result[3] - 2.5) < 1e-9


def test_expanding_with_none():
    s = pd.Series([1, None, 3])
    result = s.expanding().sum().tolist()
    # None doesn't add to sum but result is still appended
    assert result[0] == 1
    assert result[1] == 1  # None skipped
    assert result[2] == 4


# ---------------------------------------------------------------------------
# nunique (2 tests)
# ---------------------------------------------------------------------------

def test_df_nunique():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.nunique()
    assert result["a"] == 3
    assert result["b"] == 3


def test_df_nunique_with_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2, 3, 3], "y": [5, 5, 5, 5, 5]})
    result = df.nunique()
    assert result["x"] == 3
    assert result["y"] == 1


# ---------------------------------------------------------------------------
# melt (3 tests)
# ---------------------------------------------------------------------------

def test_melt_basic():
    df = pd.DataFrame({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
    result = pd.melt(df, id_vars=["id"])
    assert len(result) == 4
    assert "variable" in result.columns
    assert "value" in result.columns
    assert "id" in result.columns


def test_melt_custom_names():
    df = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    result = pd.melt(df, id_vars="id", var_name="col", value_name="val")
    assert "col" in result.columns
    assert "val" in result.columns
    assert result["col"].tolist() == ["x", "x"]
    assert result["val"].tolist() == [10, 20]


def test_melt_subset_value_vars():
    df = pd.DataFrame({"id": [1], "a": [10], "b": [20], "c": [30]})
    result = pd.melt(df, id_vars="id", value_vars=["a", "b"])
    assert len(result) == 2
    vars_col = result["variable"].tolist()
    assert "a" in vars_col
    assert "b" in vars_col
    assert "c" not in vars_col


# ---------------------------------------------------------------------------
# pct_change + diff combo (2 tests)
# ---------------------------------------------------------------------------

def test_pct_change_and_diff_compose():
    """pct_change and diff should work on the same series sequentially."""
    s = pd.Series([100.0, 110.0, 121.0])
    d = s.diff(1)
    p = s.pct_change(1)
    # diff values at index 2: 121 - 110 = 11
    assert abs(d.tolist()[2] - 11.0) < 1e-9
    # pct_change at index 2: (121 - 110) / 110 = 0.1
    assert abs(p.tolist()[2] - 0.1) < 1e-9


def test_diff_then_round():
    """diff result can be rounded."""
    s = pd.Series([1.1, 2.3, 3.7])
    d = s.diff(1)
    r = d.round(1)
    vals = r.tolist()
    assert vals[0] is None
    assert abs(vals[1] - 1.2) < 1e-9
    assert abs(vals[2] - 1.4) < 1e-9
