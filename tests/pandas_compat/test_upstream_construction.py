"""Upstream-style pandas tests: list-of-lists, scalar-dict, transpose, copy, operators."""
import pytest
from pandas import DataFrame, Series


# ---------------------------------------------------------------------------
# From list of lists (8 tests)
# ---------------------------------------------------------------------------

def test_list_of_lists_2x2_basic():
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 3]
    assert df["b"].tolist() == [2, 4]


def test_list_of_lists_single_row():
    df = DataFrame([[10, 20, 30]], columns=["x", "y", "z"])
    assert df.shape == (1, 3)
    assert df["x"].tolist() == [10]
    assert df["z"].tolist() == [30]


def test_list_of_lists_single_column():
    df = DataFrame([[1], [2], [3]], columns=["a"])
    assert df.shape == (3, 1)
    assert df["a"].tolist() == [1, 2, 3]


def test_list_of_lists_with_column_names():
    df = DataFrame([[5, 6], [7, 8]], columns=["col1", "col2"])
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 5
    assert df.iloc[1]["col2"] == 8


def test_list_of_lists_auto_column_names():
    """Without columns= arg, column names should be string ints: '0', '1', ..."""
    df = DataFrame([[1, 2], [3, 4]])
    cols = df.columns.tolist()
    assert cols == ["0", "1"]
    assert df.shape == (2, 2)


def test_list_of_lists_single_row_iloc_access():
    df = DataFrame([[1, 2]], columns=["a", "b"])
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["b"] == 2


def test_list_of_lists_empty():
    """DataFrame([[]] ) should produce an empty frame."""
    df = DataFrame([[]])
    assert df.shape == (0, 0)


def test_list_of_lists_mixed_int_float():
    """Columns retain their type independently."""
    df = DataFrame([[1, 2.5], [3, 4.0]], columns=["a", "b"])
    assert df.shape == (2, 2)
    assert df["a"].dtype == "int64"
    assert df["b"].dtype == "float64"


# ---------------------------------------------------------------------------
# From scalar dict (5 tests)
# ---------------------------------------------------------------------------

def test_scalar_dict_with_index():
    df = DataFrame({"a": 1, "b": 2}, index=[0, 1, 2])
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 1, 1]
    assert df["b"].tolist() == [2, 2, 2]


def test_scalar_dict_without_index():
    """Scalar dict with no index should produce a 1-row DataFrame."""
    df = DataFrame({"a": 1, "b": 2})
    assert df.shape == (1, 2)
    assert df["a"].tolist() == [1]
    assert df["b"].tolist() == [2]


def test_scalar_dict_multiple_same_type():
    df = DataFrame({"a": 5, "b": 10, "c": 15}, index=[0, 1])
    assert df.shape == (2, 3)
    assert df["a"].tolist() == [5, 5]
    assert df["b"].tolist() == [10, 10]
    assert df["c"].tolist() == [15, 15]


def test_scalar_dict_mixed_types():
    df = DataFrame({"a": 1, "b": 2.5}, index=[0, 1])
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 1]
    assert df["b"].tolist() == [2.5, 2.5]


def test_scalar_dict_string_index():
    df = DataFrame({"x": 42, "y": 99}, index=["r0", "r1", "r2"])
    assert df.shape == (3, 2)
    assert df["x"].tolist() == [42, 42, 42]


# ---------------------------------------------------------------------------
# From dict of Series (4 tests)
# ---------------------------------------------------------------------------

def test_dict_of_series_basic():
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    df = DataFrame({"a": s1, "b": s2})
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 2, 3]
    assert df["b"].tolist() == [4, 5, 6]


def test_dict_of_named_series():
    s1 = Series([1, 2, 3], name="x")
    s2 = Series([4, 5, 6], name="y")
    df = DataFrame({"a": s1, "b": s2})
    # Column names come from dict keys, not Series names
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (3, 2)


def test_dict_mixed_series_and_lists():
    s1 = Series([1, 2, 3])
    df = DataFrame({"a": s1, "b": [4, 5, 6]})
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 2, 3]
    assert df["b"].tolist() == [4, 5, 6]


def test_dict_single_series():
    s = Series([10, 20, 30])
    df = DataFrame({"val": s})
    assert df.shape == (3, 1)
    assert df["val"].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# Copy semantics (5 tests)
# ---------------------------------------------------------------------------

def test_copy_independence_column_assign():
    """Assigning a column to a copy must not affect the original."""
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cp = df.copy()
    cp["a"] = [10, 20, 30]
    assert df["a"].tolist() == [1, 2, 3]
    assert cp["a"].tolist() == [10, 20, 30]


def test_copy_preserves_shape_and_columns():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    cp = df.copy()
    assert cp.shape == df.shape
    assert cp.columns.tolist() == df.columns.tolist()


def test_copy_new_column_does_not_affect_original():
    df = DataFrame({"a": [1, 2, 3]})
    cp = df.copy()
    cp["b"] = [10, 20, 30]
    assert "b" not in df.columns
    assert "b" in cp.columns


def test_copy_iloc_setitem_independence():
    """iloc-based row assignment on a copy should not affect original."""
    df = DataFrame({"a": [1, 2, 3]})
    cp = df.copy()
    cp.iloc[0] = {"a": 99}
    assert df["a"].tolist() == [1, 2, 3]
    assert cp["a"].tolist() == [99, 2, 3]


def test_copy_shape_preserved_after_modifications():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cp = df.copy()
    # Mutate copy
    cp["a"] = [0, 0, 0]
    cp["c"] = [7, 8, 9]
    # Original unchanged
    assert df.shape == (3, 2)
    assert list(df.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# Transpose (6 tests)
# ---------------------------------------------------------------------------

def test_transpose_shape_is_ncols_nrows():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = df.T
    assert t.shape == (2, 3)


def test_transpose_double_transpose_restores_shape():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.T.T.shape == df.shape


def test_transpose_column_access():
    """After T, the original row indices become column names (as strings)."""
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    t = df.T
    # Columns of T are the original row indices: "0", "1"
    assert t.columns.tolist() == ["0", "1"]
    assert t["0"].tolist() == [1, 3]
    assert t["1"].tolist() == [2, 4]


def test_transpose_single_column_df():
    df = DataFrame({"a": [1, 2, 3]})
    t = df.T
    assert t.shape == (1, 3)


def test_transpose_single_row_df():
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    t = df.T
    assert t.shape == (3, 1)


def test_transpose_via_method():
    """transpose() method should behave identically to .T."""
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    t1 = df.T
    t2 = df.transpose()
    assert t1.shape == t2.shape
    assert t1.columns.tolist() == t2.columns.tolist()


# ---------------------------------------------------------------------------
# axis=0 aggregations (the only axis currently supported correctly) (6 tests)
# ---------------------------------------------------------------------------

def test_sum_axis0_returns_correct_column_sums():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = df.sum()
    assert s["a"] == 6
    assert s["b"] == 15


def test_sum_default_is_axis0():
    df = DataFrame({"x": [10, 20], "y": [1, 2]})
    s = df.sum()
    assert s["x"] == 30
    assert s["y"] == 3


def test_mean_axis0_correct():
    df = DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]})
    m = df.mean()
    assert m["a"] == 2.0
    assert m["b"] == 3.0


def test_min_axis0_correct():
    df = DataFrame({"a": [5, 1, 3], "b": [2, 8, 4]})
    mn = df.min()
    assert mn["a"] == 1
    assert mn["b"] == 2


def test_max_axis0_correct():
    df = DataFrame({"a": [5, 1, 3], "b": [2, 8, 4]})
    mx = df.max()
    assert mx["a"] == 5
    assert mx["b"] == 8


def test_sum_float_columns():
    df = DataFrame({"a": [1.5, 2.5], "b": [0.5, 1.5]})
    s = df.sum()
    assert s["a"] == 4.0
    assert s["b"] == 2.0


# ---------------------------------------------------------------------------
# DataFrame comparison operators (5 tests)
# ---------------------------------------------------------------------------

def test_df_eq_scalar():
    df = DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
    res = df == 1
    assert res.shape == (3, 2)
    assert res["a"].tolist() == [True, False, False]
    assert res["b"].tolist() == [True, True, True]


def test_df_gt_scalar():
    df = DataFrame({"a": [1, 2, 3]})
    res = df > 2
    assert res["a"].tolist() == [False, False, True]


def test_df_add_df():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    res = df + df
    assert res["a"].tolist() == [2, 4]
    assert res["b"].tolist() == [6, 8]


def test_df_mul_scalar():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    res = df * 3
    assert res["a"].tolist() == [3, 6]
    assert res["b"].tolist() == [9, 12]


def test_df_sub_self_is_zeros():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res = df - df
    assert res["a"].tolist() == [0, 0, 0]
    assert res["b"].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# Properties (5 tests)
# ---------------------------------------------------------------------------

def test_columns_tolist():
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    assert df.columns.tolist() == ["a", "b", "c"]


def test_columns_len():
    df = DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert len(df.columns) == 3


def test_col_in_df():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    assert "a" in df
    assert "b" in df
    assert "c" not in df


def test_list_df_gives_column_names():
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    assert list(df) == ["a", "b", "c"]


def test_dtypes_returns_mapping():
    """dtypes should return a mapping from column name to dtype string."""
    df = DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
    dt = df.dtypes
    assert dt["a"] == "int64"
    assert dt["b"] == "float64"
    assert dt["c"] == "object"
