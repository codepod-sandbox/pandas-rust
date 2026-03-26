"""Tests for axis=1 aggregation, cov, corrwith, from_dict, and str methods."""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import pandas as pd


# ---------------------------------------------------------------------------
# axis=1 aggregation (10 tests)
# ---------------------------------------------------------------------------

def test_sum_axis1_returns_series():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sum(axis=1)
    assert isinstance(result, pd.Series)


def test_sum_axis1_values_correct():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sum(axis=1)
    assert result.tolist() == [5, 7, 9]


def test_mean_axis1():
    df = pd.DataFrame({"a": [2.0, 4.0], "b": [6.0, 8.0]})
    result = df.mean(axis=1)
    assert result.tolist() == [4.0, 6.0]


def test_min_axis1():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [4, 2, 6]})
    result = df.min(axis=1)
    assert result.tolist() == [1, 2, 3]


def test_max_axis1():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [4, 2, 6]})
    result = df.max(axis=1)
    assert result.tolist() == [4, 5, 6]


def test_sum_axis1_mixed_int_float():
    df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    result = df.sum(axis=1)
    assert result.tolist() == [2.5, 4.5]


def test_sum_axis1_skips_string_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4, 5, 6]})
    result = df.sum(axis=1)
    # Only numeric cols a and c should be summed
    assert result.tolist() == [5, 7, 9]


def test_sum_axis1_single_column():
    df = pd.DataFrame({"a": [10, 20, 30]})
    result = df.sum(axis=1)
    assert result.tolist() == [10, 20, 30]


def test_sum_axis0_still_returns_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sum(axis=0)
    assert isinstance(result, dict)
    assert result["a"] == 6
    assert result["b"] == 15


def test_sum_axis1_length_matches_nrows():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    result = df.sum(axis=1)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# cov (5 tests)
# ---------------------------------------------------------------------------

def test_cov_returns_square_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.cov()
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"a", "b"}
    assert len(result) == 2


def test_cov_diagonal_is_variance():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    cov_df = df.cov()
    # variance of [1,2,3,4] with ddof=1 = 5/3 ≈ 1.6667
    cov_aa = cov_df["a"].tolist()[0]
    expected_var = sum((x - 2.5) ** 2 for x in [1.0, 2.0, 3.0, 4.0]) / 3
    assert abs(cov_aa - expected_var) < 1e-10


def test_cov_symmetric():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    cov_df = df.cov()
    a_vals = cov_df["a"].tolist()
    b_vals = cov_df["b"].tolist()
    # cov(a,b) == cov(b,a)
    assert abs(a_vals[1] - b_vals[0]) < 1e-10


def test_cov_identical_columns_equals_variance():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    cov_df = df.cov()
    a_diag = cov_df["a"].tolist()[0]
    b_diag = cov_df["b"].tolist()[1]
    # Both diagonals should be the same (variance of [1,2,3,4])
    assert abs(a_diag - b_diag) < 1e-10


def test_cov_single_column_is_variance():
    df = pd.DataFrame({"x": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
    cov_df = df.cov()
    expected_var = sum((v - 5.0) ** 2 for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) / 7
    cov_val = cov_df["x"].tolist()[0]
    assert abs(cov_val - expected_var) < 1e-10


# ---------------------------------------------------------------------------
# corrwith (3 tests)
# ---------------------------------------------------------------------------

def test_corrwith_same_df_all_ones():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    result = df.corrwith(df)
    assert abs(result["a"] - 1.0) < 1e-10
    assert abs(result["b"] - 1.0) < 1e-10


def test_corrwith_different_values_between_minus1_and_1():
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    df2 = pd.DataFrame({"a": [4.0, 3.0, 2.0, 1.0]})
    result = df1.corrwith(df2)
    assert "a" in result
    assert result["a"] is not None
    # perfectly negatively correlated, allow floating point tolerance
    assert abs(result["a"]) <= 1.0 + 1e-10


def test_corrwith_missing_column_not_in_result():
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df1.corrwith(df2)
    # only shared columns are in result
    assert "a" in result
    assert "b" not in result


# ---------------------------------------------------------------------------
# from_dict (3 tests)
# ---------------------------------------------------------------------------

def test_from_dict_basic():
    data = {"x": [1, 2, 3], "y": [4, 5, 6]}
    df = pd.DataFrame.from_dict(data)
    assert isinstance(df, pd.DataFrame)
    assert df["x"].tolist() == [1, 2, 3]
    assert df["y"].tolist() == [4, 5, 6]


def test_from_dict_same_as_constructor():
    data = {"a": [10, 20], "b": [30, 40]}
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame.from_dict(data)
    assert df1["a"].tolist() == df2["a"].tolist()
    assert df1["b"].tolist() == df2["b"].tolist()


def test_from_dict_preserves_columns():
    data = {"z": [1], "a": [2], "m": [3]}
    df = pd.DataFrame.from_dict(data)
    # All columns present
    assert set(df.columns) == {"z", "a", "m"}


# ---------------------------------------------------------------------------
# str methods (8 tests)
# ---------------------------------------------------------------------------

def test_str_swapcase():
    s = pd.Series(["Hello", "WORLD", "foo"])
    result = s.str.swapcase().tolist()
    assert result == ["hELLO", "world", "FOO"]


def test_str_fullmatch_true():
    s = pd.Series(["abc", "ab", "abcd"])
    result = s.str.fullmatch(r"abc").tolist()
    assert result == [True, False, False]


def test_str_fullmatch_false():
    s = pd.Series(["hello", "world"])
    result = s.str.fullmatch(r"\d+").tolist()
    assert result == [False, False]


def test_str_extract_captures_pattern():
    s = pd.Series(["foo123", "bar456", "baz"])
    result = s.str.extract(r"\d+").tolist()
    assert result == ["123", "456", None]


def test_str_extract_no_match_returns_none():
    s = pd.Series(["abc", "def"])
    result = s.str.extract(r"\d+").tolist()
    assert result == [None, None]


def test_str_repeat():
    s = pd.Series(["ab", "c"])
    result = s.str.repeat(3).tolist()
    assert result == ["ababab", "ccc"]


def test_str_center_ljust_rjust():
    s = pd.Series(["hi"])
    assert s.str.center(6).tolist() == ["  hi  "]
    assert s.str.ljust(6).tolist() == ["hi    "]
    assert s.str.rjust(6).tolist() == ["    hi"]


def test_str_methods_on_none_values():
    s = pd.Series(["Hello", None, "world"])
    swap = s.str.swapcase().tolist()
    assert swap[0] == "hELLO"
    assert swap[1] is None
    assert swap[2] == "WORLD"


# ---------------------------------------------------------------------------
# Additional workflow tests (6 tests)
# ---------------------------------------------------------------------------

def test_sum_axis1_assign_as_new_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df["row_sum"] = df.sum(axis=1).tolist()
    assert df["row_sum"].tolist() == [5, 7, 9]


def test_notna_sum_counts_non_null_per_column():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    notna_df = df.notna()
    # Count True values per column using Python list
    count_a = sum(1 for v in notna_df["a"].tolist() if v)
    count_b = sum(1 for v in notna_df["b"].tolist() if v)
    assert count_a == 2
    assert count_b == 2


def test_cov_extract_diagonal():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    cov_df = df.cov()
    # Extract diagonal: a[0], b[1]
    a_var = cov_df["a"].tolist()[0]
    b_var = cov_df["b"].tolist()[1]
    assert a_var > 0
    assert b_var > 0


def test_groupby_then_mean_axis0():
    df = pd.DataFrame({"grp": ["a", "a", "b"], "val": [1.0, 3.0, 5.0]})
    result = df.groupby("grp").mean()
    assert isinstance(result, pd.DataFrame)


def test_apply_axis1_vs_sum_axis1():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sum_result = df.sum(axis=1).tolist()
    apply_result = df.apply(lambda row: row["a"] + row["b"], axis=1).tolist()
    assert sum_result == apply_result


def test_transpose_then_sum():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = df.T
    assert isinstance(t, pd.DataFrame)
    result = t.sum()
    assert isinstance(result, dict)
