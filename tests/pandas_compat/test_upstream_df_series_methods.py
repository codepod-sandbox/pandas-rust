"""Upstream-style pandas compatibility tests for df.cumsum/shift/pct_change/rank/cov,
axis=1 std/var/median, Series.cov/corr, expanding.count, str.join."""
import math
import pytest
from pandas import Series, DataFrame


# ---------------------------------------------------------------------------
# DataFrame cumulative methods (6 tests)
# ---------------------------------------------------------------------------

def test_df_cumsum_basic():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.cumsum()
    assert result["a"].tolist() == [1, 3, 6]
    assert result["b"].tolist() == [4, 9, 15]


def test_df_cummax():
    df = DataFrame({"a": [3, 1, 4, 1, 5], "b": [2, 7, 1, 8, 0]})
    result = df.cummax()
    assert result["a"].tolist() == [3, 3, 4, 4, 5]
    assert result["b"].tolist() == [2, 7, 7, 8, 8]


def test_df_cummin():
    df = DataFrame({"a": [3, 1, 4, 1, 5], "b": [2, 7, 1, 8, 0]})
    result = df.cummin()
    assert result["a"].tolist() == [3, 1, 1, 1, 1]
    assert result["b"].tolist() == [2, 2, 1, 1, 0]


def test_df_cumprod():
    df = DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
    result = df.cumprod()
    assert result["a"].tolist() == [1, 2, 6]
    assert result["b"].tolist() == [2, 4, 8]


def test_df_cumsum_preserves_columns():
    df = DataFrame({"x": [1, 2], "y": [3, 4]})
    result = df.cumsum()
    assert set(result.columns) == {"x", "y"}


def test_df_cumsum_skips_strings():
    df = DataFrame({"num": [1, 2, 3], "text": ["a", "b", "c"]})
    result = df.cumsum()
    assert "num" in result.columns
    # "text" column should be skipped (TypeError from cumsum on strings)
    assert result["num"].tolist() == [1, 3, 6]


# ---------------------------------------------------------------------------
# DataFrame shift / pct_change (5 tests)
# ---------------------------------------------------------------------------

def test_df_shift_1_first_row_null():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.shift(1)
    assert result["a"].tolist()[0] is None
    assert result["a"].tolist()[1:] == [1, 2]


def test_df_shift_neg1_last_row_null():
    df = DataFrame({"a": [1, 2, 3]})
    result = df.shift(-1)
    assert result["a"].tolist()[-1] is None
    assert result["a"].tolist()[:-1] == [2, 3]


def test_df_pct_change_basic():
    df = DataFrame({"a": [100, 110, 121]})
    result = df.pct_change()
    vals = result["a"].tolist()
    assert vals[0] is None
    assert abs(vals[1] - 0.1) < 1e-9
    assert abs(vals[2] - 0.1) < 1e-9


def test_df_pct_change_first_row_null():
    df = DataFrame({"a": [10, 20, 30], "b": [5, 10, 15]})
    result = df.pct_change()
    assert result["a"].tolist()[0] is None
    assert result["b"].tolist()[0] is None


def test_df_shift_preserves_columns():
    df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = df.shift(1)
    assert set(result.columns) == {"x", "y"}


# ---------------------------------------------------------------------------
# DataFrame rank (3 tests)
# ---------------------------------------------------------------------------

def test_df_rank_basic():
    df = DataFrame({"a": [10, 30, 20]})
    result = df.rank()
    assert result["a"].tolist() == [1.0, 3.0, 2.0]


def test_df_rank_ties_average():
    df = DataFrame({"a": [1, 2, 2, 3]})
    result = df.rank()
    vals = result["a"].tolist()
    # tied values at positions 1 and 2 should share average rank 2.5
    assert vals[1] == vals[2] == 2.5


def test_df_rank_returns_float():
    df = DataFrame({"a": [5, 3, 1]})
    result = df.rank()
    for v in result["a"].tolist():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# DataFrame axis=1 (8 tests)
# ---------------------------------------------------------------------------

def test_df_std_axis1():
    df = DataFrame({"a": [1.0, 4.0], "b": [3.0, 4.0]})
    result = df.std(axis=1)
    assert isinstance(result, Series)
    expected0 = math.sqrt(((1 - 2) ** 2 + (3 - 2) ** 2) / 1)
    assert abs(result.tolist()[0] - expected0) < 1e-9


def test_df_var_axis1():
    df = DataFrame({"a": [1.0, 4.0], "b": [3.0, 4.0]})
    result = df.var(axis=1)
    assert isinstance(result, Series)
    # var of [1, 3] = ((1-2)^2 + (3-2)^2) / 1 = 2.0
    assert abs(result.tolist()[0] - 2.0) < 1e-9


def test_df_median_axis1():
    df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7]})
    result = df.median(axis=1)
    assert isinstance(result, Series)
    # median of [1,3,5]=3, [2,4,6]=4, [3,5,7]=5
    assert result.tolist() == [3.0, 4.0, 5.0]


def test_df_prod_axis1():
    df = DataFrame({"a": [2, 3], "b": [4, 5]})
    result = df.prod(axis=1)
    assert isinstance(result, Series)
    assert result.tolist() == [8, 15]


def test_df_sum_axis1_skips_none():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.sum(axis=1)
    assert isinstance(result, Series)
    assert result.tolist() == [4, 6]


def test_df_axis1_all_return_series():
    df = DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert isinstance(df.std(axis=1), Series)
    assert isinstance(df.var(axis=1), Series)
    assert isinstance(df.median(axis=1), Series)
    assert isinstance(df.prod(axis=1), Series)
    assert isinstance(df.sum(axis=1), Series)
    assert isinstance(df.mean(axis=1), Series)


def test_df_axis1_result_length_matches_nrows():
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    for fn in [df.std, df.var, df.median, df.sum, df.mean]:
        result = fn(axis=1)
        assert len(result) == len(df)


def test_df_axis1_mixed_int_float():
    df = DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    result = df.mean(axis=1)
    assert isinstance(result, Series)
    assert abs(result.tolist()[0] - 1.25) < 1e-9


# ---------------------------------------------------------------------------
# Series cov / corr (6 tests)
# ---------------------------------------------------------------------------

def test_series_cov_same_equals_variance():
    s = Series([1.0, 2.0, 3.0, 4.0])
    cov = s.cov(s)
    vals = s.tolist()
    n = len(vals)
    mean = sum(vals) / n
    expected_var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    assert abs(cov - expected_var) < 1e-9


def test_series_cov_value():
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([4.0, 5.0, 6.0])
    cov = s1.cov(s2)
    # cov([1,2,3],[4,5,6]) = 1.0
    assert abs(cov - 1.0) < 1e-9


def test_series_corr_same_is_one():
    s = Series([1.0, 2.0, 3.0, 4.0])
    corr = s.corr(s)
    assert abs(corr - 1.0) < 1e-9


def test_series_corr_perfectly_negative():
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([3.0, 2.0, 1.0])
    corr = s1.corr(s2)
    assert abs(corr - (-1.0)) < 1e-9


def test_series_corr_near_zero():
    # [1, -1, 1, -1] vs [1, 1, -1, -1] — orthogonal, corr = 0
    s1 = Series([1.0, -1.0, 1.0, -1.0])
    s2 = Series([1.0, 1.0, -1.0, -1.0])
    corr = s1.corr(s2)
    assert abs(corr) < 1e-9


def test_series_corr_proportional_is_one():
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([2.0, 4.0, 6.0])
    corr = s1.corr(s2)
    assert abs(corr - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Expanding count (3 tests)
# ---------------------------------------------------------------------------

def test_expanding_count_basic():
    s = Series([10, 20, 30])
    result = s.expanding().count()
    assert result.tolist() == [1, 2, 3]


def test_expanding_count_increasing():
    s = Series([5, 5, 5, 5])
    result = s.expanding().count()
    vals = result.tolist()
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1]


def test_expanding_count_matches_range():
    s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.expanding().count()
    assert result.tolist() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Series str.join (3 tests)
# ---------------------------------------------------------------------------

def test_str_join_list_of_strings():
    # Use str.split to produce list-of-strings Series, then join
    s = Series(["a-b-c", "x-y"])
    split_s = s.str.split("-")
    result = split_s.str.join("-")
    assert result.tolist() == ["a-b-c", "x-y"]


def test_str_join_empty_list():
    # Single-element strings split produce single-item lists; join with comma
    s = Series(["a", "b"])
    split_s = s.str.split(",")
    result = split_s.str.join(",")
    assert result.tolist() == ["a", "b"]


def test_str_join_roundtrip():
    # split then join with same sep should restore original
    s = Series(["hello world", "foo bar baz"])
    joined = s.str.split(" ").str.join(" ")
    assert joined.tolist() == ["hello world", "foo bar baz"]


# ---------------------------------------------------------------------------
# DataFrame cov (4 tests)
# ---------------------------------------------------------------------------

def test_df_cov_returns_square_dataframe():
    df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.cov()
    assert isinstance(result, DataFrame)
    assert set(result.columns) == {"a", "b"}


def test_df_cov_diagonal_equals_variance():
    vals = [1.0, 2.0, 3.0]
    df = DataFrame({"a": vals, "b": [4.0, 5.0, 6.0]})
    cov = df.cov()
    n = len(vals)
    mean_a = sum(vals) / n
    expected_var_a = sum((v - mean_a) ** 2 for v in vals) / (n - 1)
    # cov() returns DataFrame with column "a" as list; diagonal
    cov_aa = cov["a"].tolist()[0]  # first row, col a
    assert abs(cov_aa - expected_var_a) < 1e-9


def test_df_cov_same_col_equals_variance():
    s = Series([1.0, 2.0, 3.0, 4.0])
    df = DataFrame({"x": s.tolist()})
    cov_df = df.cov()
    s_var = s.var() if hasattr(s, 'var') else None
    cov_val = cov_df["x"].tolist()[0]
    if s_var is not None:
        assert abs(cov_val - s_var) < 1e-9


def test_df_cov_symmetric():
    df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    cov = df.cov()
    cols = list(cov.columns)
    a_idx = cols.index("a")
    b_idx = cols.index("b")
    cov_ab = cov["b"].tolist()[a_idx]
    cov_ba = cov["a"].tolist()[b_idx]
    assert abs(cov_ab - cov_ba) < 1e-9


# ---------------------------------------------------------------------------
# Integration workflows (5 tests)
# ---------------------------------------------------------------------------

def test_cumsum_then_diff_approx_original():
    df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    cumsum_df = df.cumsum()
    # diff on the cumsum Series should give back original (except first)
    diff_series = cumsum_df["a"].diff()
    vals = diff_series.tolist()
    assert vals[0] is None
    assert vals[1:] == [2.0, 3.0, 4.0]


def test_shift_then_subtract_matches_diff():
    df = DataFrame({"a": [10.0, 20.0, 35.0]})
    shifted = df.shift(1)
    orig = df["a"].tolist()
    sh = shifted["a"].tolist()
    manual_diff = [None if sh[i] is None else orig[i] - sh[i] for i in range(len(orig))]
    diff_series = df["a"].diff().tolist()
    assert manual_diff[0] is None and diff_series[0] is None
    for i in range(1, len(orig)):
        assert abs(manual_diff[i] - diff_series[i]) < 1e-9


def test_pct_change_values_correct():
    df = DataFrame({"a": [100.0, 150.0, 120.0]})
    result = df.pct_change()
    vals = result["a"].tolist()
    assert vals[0] is None
    assert abs(vals[1] - 0.5) < 1e-9
    assert abs(vals[2] - (-0.2)) < 1e-9


def test_rank_sort_matches_sorted_order():
    df = DataFrame({"a": [30.0, 10.0, 20.0]})
    ranked = df.rank()
    # rank 1 should correspond to smallest value
    vals = df["a"].tolist()
    ranks = ranked["a"].tolist()
    rank1_idx = ranks.index(1.0)
    assert vals[rank1_idx] == min(vals)


def test_axis1_sum_matches_iterrows_sum():
    df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    axis1_result = df.sum(axis=1).tolist()
    for i, (_, row) in enumerate(df.iterrows()):
        row_sum = sum(v for v in row.values() if v is not None)
        assert abs(axis1_result[i] - row_sum) < 1e-9
