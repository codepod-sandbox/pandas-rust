"""Tests for new pandas compatibility features: iteration, to_frame, idxmax/min,
cumulative ops, DataFrame comparison, sample, and corr."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import pandas as pd


# ---------------------------------------------------------------------------
# Iteration (4 tests)
# ---------------------------------------------------------------------------

def test_iter_df_yields_column_names():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cols = list(df)
    assert cols == ["a", "b"]


def test_list_df_returns_column_names():
    df = pd.DataFrame({"x": [10], "y": [20], "z": [30]})
    assert list(df) == ["x", "y", "z"]


def test_columns_tolist():
    df = pd.DataFrame({"p": [1], "q": [2]})
    result = df.columns.tolist()
    assert result == ["p", "q"]
    assert isinstance(result, list)


def test_len_columns():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert len(df.columns) == 3


# ---------------------------------------------------------------------------
# to_frame (3 tests)
# ---------------------------------------------------------------------------

def test_to_frame_returns_dataframe():
    s = pd.Series([1, 2, 3], name="vals")
    result = s.to_frame()
    assert isinstance(result, pd.DataFrame)


def test_to_frame_preserves_values():
    s = pd.Series([10, 20, 30], name="nums")
    df = s.to_frame()
    assert df["nums"].tolist() == [10, 20, 30]


def test_to_frame_custom_name():
    s = pd.Series([1.0, 2.0], name="original")
    df = s.to_frame(name="renamed")
    assert "renamed" in df.columns
    assert df["renamed"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# idxmax / idxmin (4 tests)
# ---------------------------------------------------------------------------

def test_idxmax_ints():
    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    assert s.idxmax() == 5


def test_idxmin_ints():
    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    assert s.idxmin() == 1


def test_idxmax_floats():
    s = pd.Series([1.5, 3.7, 2.2, 0.1])
    assert s.idxmax() == 1


def test_idxmin_correct_position():
    s = pd.Series([10, 5, 8, 3, 7])
    assert s.idxmin() == 3


# ---------------------------------------------------------------------------
# Cumulative operations (6 tests)
# ---------------------------------------------------------------------------

def test_cumsum_ints():
    s = pd.Series([1, 2, 3, 4], name="x")
    result = s.cumsum()
    assert result.tolist() == [1, 3, 6, 10]


def test_cumsum_floats():
    s = pd.Series([1.0, 2.0, 3.0], name="f")
    result = s.cumsum()
    assert result.tolist() == [1.0, 3.0, 6.0]


def test_cummax():
    s = pd.Series([1, 5, 3, 7, 2])
    result = s.cummax()
    assert result.tolist() == [1, 5, 5, 7, 7]


def test_cummin():
    s = pd.Series([5, 3, 7, 1, 4])
    result = s.cummin()
    assert result.tolist() == [5, 3, 3, 1, 1]


def test_cumprod():
    s = pd.Series([1, 2, 3, 4])
    result = s.cumprod()
    assert result.tolist() == [1, 2, 6, 24]


def test_cumsum_preserves_name():
    s = pd.Series([1, 2, 3], name="myname")
    result = s.cumsum()
    assert result.name == "myname"


# ---------------------------------------------------------------------------
# DataFrame comparison operators (5 tests)
# ---------------------------------------------------------------------------

def test_df_eq_self_all_true():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df == df
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [True, True, True]
    assert result["b"].tolist() == [True, True, True]


def test_df_ne_self_all_false():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df != df
    assert result["a"].tolist() == [False, False, False]


def test_df_gt_scalar():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [2, 4, 6]})
    result = df > 3
    assert result["a"].tolist() == [False, True, False]
    assert result["b"].tolist() == [False, True, True]


def test_df_eq_scalar():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = df == 2
    assert result["x"].tolist() == [False, True, False]


def test_comparison_returns_dataframe_of_bools():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [5, 25, 15]})
    result = df >= 20
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [False, True, True]
    assert result["b"].tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# sample (3 tests)
# ---------------------------------------------------------------------------

def test_sample_n_shape():
    df = pd.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
    result = df.sample(n=3)
    assert result.shape == (3, 2)


def test_sample_random_state_reproducible():
    df = pd.DataFrame({"a": list(range(20))})
    r1 = df.sample(n=5, random_state=42)
    r2 = df.sample(n=5, random_state=42)
    assert r1["a"].tolist() == r2["a"].tolist()


def test_sample_frac():
    df = pd.DataFrame({"a": list(range(10))})
    result = df.sample(frac=0.5, random_state=7)
    assert result.shape == (5, 1)


# ---------------------------------------------------------------------------
# corr (3 tests)
# ---------------------------------------------------------------------------

def test_corr_identity():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = df.corr()
    assert isinstance(result, pd.DataFrame)
    # correlation of a column with itself should be 1.0
    assert abs(result["a"].tolist()[0] - 1.0) < 1e-9


def test_corr_known_values():
    # perfectly correlated: b = 2*a
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    result = df.corr()
    cols = result.columns.tolist()
    a_idx = cols.index("a")
    b_idx = cols.index("b")
    a_vals = result["a"].tolist()
    # corr(a, b) should be 1.0
    assert abs(a_vals[b_idx] - 1.0) < 1e-9


def test_corr_square_dataframe():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0], "z": [1.0, 3.0, 2.0]})
    result = df.corr()
    assert isinstance(result, pd.DataFrame)
    # result should be 3x3 (number of numeric cols)
    assert len(result.columns) == 3
    assert len(result) == 3
