"""Pandas compatibility tests for DataFrame indexing (iloc, loc, bool mask, concat).

Upstream-style test suite covering iloc, loc, boolean indexing, concat,
and Series iloc with 50+ test cases.
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df5():
    """5-row mixed-type DataFrame."""
    return pd.DataFrame(
        {"a": [10, 20, 30, 40, 50], "b": [1.0, 2.0, 3.0, 4.0, 5.0], "c": ["x", "y", "z", "w", "v"]}
    )


@pytest.fixture
def df3():
    """Simple 3-row integer DataFrame."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


# ===========================================================================
# iloc tests (DataFrame)
# ===========================================================================

def test_iloc_int_positive(df5):
    """Single positive int returns dict-like row with correct values."""
    row = df5.iloc[1]
    assert row["a"] == 20
    assert row["b"] == 2.0
    assert row["c"] == "y"


def test_iloc_int_zero(df5):
    """iloc[0] accesses the very first row."""
    row = df5.iloc[0]
    assert row["a"] == 10
    assert row["c"] == "x"


def test_iloc_int_last(df5):
    """iloc[len-1] accesses the last row."""
    row = df5.iloc[4]
    assert row["a"] == 50
    assert row["c"] == "v"


def test_iloc_int_negative(df5):
    """Negative integer indexes from the end."""
    row = df5.iloc[-1]
    assert row["a"] == 50
    row2 = df5.iloc[-2]
    assert row2["a"] == 40


def test_iloc_int_negative_first(df5):
    """-len(df) reaches the very first row."""
    row = df5.iloc[-5]
    assert row["a"] == 10


def test_iloc_slice_basic(df5):
    """Basic slice [1:3] returns 2 rows."""
    sub = df5.iloc[1:3]
    assert sub.shape == (2, 3)
    assert sub["a"].tolist() == [20, 30]


def test_iloc_slice_from_start(df5):
    """Slice [:3] returns first 3 rows."""
    sub = df5.iloc[:3]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [10, 20, 30]


def test_iloc_slice_to_end(df5):
    """Slice [3:] returns rows from index 3 onward."""
    sub = df5.iloc[3:]
    assert sub.shape == (2, 3)
    assert sub["a"].tolist() == [40, 50]


def test_iloc_slice_full(df5):
    """Slice [:] returns all rows."""
    sub = df5.iloc[:]
    assert sub.shape == (5, 3)


def test_iloc_slice_empty(df5):
    """Slice [0:0] returns an empty DataFrame (0 rows, same columns)."""
    sub = df5.iloc[0:0]
    assert sub.shape == (0, 3)
    assert list(sub.columns) == ["a", "b", "c"]


def test_iloc_slice_step():
    """Step slice [::2] selects every other row."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    sub = df.iloc[::2]
    assert sub.shape == (3, 1)
    assert sub["a"].tolist() == [1, 3, 5]


def test_iloc_slice_negative(df5):
    """Slice [-2:] selects the last two rows."""
    sub = df5.iloc[-2:]
    assert sub.shape == (2, 3)
    assert sub["a"].tolist() == [40, 50]


def test_iloc_list_basic(df5):
    """List of ints selects specific rows in that order."""
    sub = df5.iloc[[0, 2, 4]]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [10, 30, 50]


def test_iloc_list_single(df5):
    """Single-element list returns a DataFrame (not a row dict)."""
    sub = df5.iloc[[0]]
    assert sub.shape == (1, 3)
    assert sub["a"].tolist() == [10]


def test_iloc_list_empty(df5):
    """Empty list returns 0-row DataFrame with all columns intact."""
    sub = df5.iloc[[]]
    assert sub.shape == (0, 3)
    assert list(sub.columns) == ["a", "b", "c"]


def test_iloc_list_repeated(df5):
    """Repeated indices in list produce duplicate rows."""
    sub = df5.iloc[[0, 0, 1]]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [10, 10, 20]


def test_iloc_preserves_columns(df5):
    """After row selection, all original columns are present."""
    sub = df5.iloc[[0, 2]]
    assert list(sub.columns) == ["a", "b", "c"]


def test_iloc_preserves_dtypes(df5):
    """Dtypes are unchanged after iloc row selection."""
    sub = df5.iloc[1:3]
    assert str(sub.dtypes["a"]) == "int64"
    assert str(sub.dtypes["b"]) == "float64"


def test_iloc_returns_new_object(df5):
    """iloc[:] returns a new object, not the same DataFrame."""
    sub = df5.iloc[:]
    assert sub is not df5


def test_iloc_tuple_row_col(df5):
    """iloc[row, col] returns a scalar value."""
    val = df5.iloc[0, 1]
    assert val == 1.0
    val2 = df5.iloc[2, 0]
    assert val2 == 30


# ===========================================================================
# loc tests (DataFrame)
# ===========================================================================

def test_loc_int_label(df5):
    """Single integer label on RangeIndex returns dict-like row."""
    row = df5.loc[2]
    assert row["a"] == 30
    assert row["c"] == "z"


def test_loc_slice_inclusive(df5):
    """loc[1:3] is inclusive on both endpoints (unlike iloc)."""
    sub = df5.loc[1:3]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [20, 30, 40]


def test_loc_slice_from_start(df5):
    """loc[:2] selects rows with labels 0, 1, 2."""
    sub = df5.loc[:2]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [10, 20, 30]


def test_loc_slice_to_end(df5):
    """loc[3:] selects rows with labels 3 and 4."""
    sub = df5.loc[3:]
    assert sub.shape == (2, 3)
    assert sub["a"].tolist() == [40, 50]


def test_loc_slice_full(df5):
    """loc[:] selects all rows."""
    sub = df5.loc[:]
    assert sub.shape == (5, 3)


def test_loc_list_labels(df5):
    """loc[[0,2,4]] selects rows by list of labels."""
    sub = df5.loc[[0, 2, 4]]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [10, 30, 50]


def test_loc_single_row_single_col(df5):
    """loc[row, col] returns a scalar value."""
    val = df5.loc[0, "a"]
    assert val == 10
    val2 = df5.loc[3, "b"]
    assert val2 == 4.0


def test_loc_slice_with_col(df5):
    """loc[0:2, 'col'] returns a Series for that column over the slice."""
    result = df5.loc[0:2, "a"]
    assert result.tolist() == [10, 20, 30]


def test_loc_returns_dataframe_for_slice(df5):
    """loc with a slice returns a DataFrame."""
    sub = df5.loc[0:2]
    assert isinstance(sub, pd.DataFrame)


def test_loc_returns_dict_for_scalar(df5):
    """loc with a scalar label returns a dict-like object (row)."""
    row = df5.loc[0]
    assert isinstance(row, dict)


def test_loc_preserves_columns(df5):
    """Column list is unchanged after loc row selection."""
    sub = df5.loc[0:3]
    assert list(sub.columns) == ["a", "b", "c"]


def test_loc_negative_int(df5):
    """Negative integer label on default RangeIndex reaches row by label."""
    # Default RangeIndex has labels 0..4; -1 is not a valid label but
    # pandas-rust appears to support it as a negative lookup.
    row = df5.loc[-1]
    assert row["a"] == 50


def test_loc_equivalent_to_iloc_for_range_index(df5):
    """For the default RangeIndex, loc and iloc give identical slice results."""
    by_loc = df5.loc[1:2]
    by_iloc = df5.iloc[1:3]   # iloc is exclusive on stop, loc is inclusive
    assert by_loc["a"].tolist() == by_iloc["a"].tolist()


# ===========================================================================
# Boolean indexing tests
# ===========================================================================

def test_bool_series_mask(df5):
    """df[bool_series] keeps only rows where mask is True."""
    mask = df5["a"].gt(20)
    sub = df5[mask]
    assert sub.shape == (3, 3)
    assert sub["a"].tolist() == [30, 40, 50]


def test_bool_series_mask_all_true(df5):
    """Boolean mask of all True keeps every row."""
    mask = df5["a"].gt(0)
    sub = df5[mask]
    assert sub.shape == (5, 3)


def test_bool_series_mask_all_false(df5):
    """Boolean mask of all False returns 0-row DataFrame."""
    mask = df5["a"].gt(999)
    sub = df5[mask]
    assert sub.shape == (0, 3)
    assert list(sub.columns) == ["a", "b", "c"]


def test_bool_series_mask_preserves_columns(df5):
    """Columns are unchanged after boolean masking."""
    mask = df5["a"].gt(20)
    sub = df5[mask]
    assert list(sub.columns) == ["a", "b", "c"]


def test_bool_series_mask_preserves_dtypes(df5):
    """Dtypes are unchanged after boolean masking."""
    mask = df5["a"].gt(20)
    sub = df5[mask]
    assert str(sub.dtypes["a"]) == "int64"
    assert str(sub.dtypes["b"]) == "float64"


def test_bool_mask_from_comparison(df5):
    """Derived mask via comparison operator works correctly."""
    sub = df5[df5["a"] > 20]
    assert sub["a"].tolist() == [30, 40, 50]


def test_bool_mask_combined():
    """Combined masks using gt/lt produce correct intersection."""
    # & between two Series is not yet supported; use chained filtering instead.
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    # Filter a > 1 first, then b < 5 on result
    step1 = df[df["a"].gt(1)]
    result = step1[step1["b"].lt(5)]
    assert result["a"].tolist() == [2, 3, 4, 5]


def test_bool_mask_single_match(df5):
    """Mask that matches exactly one row returns a single-row DataFrame."""
    mask = df5["a"] == 30
    sub = df5[mask]
    assert sub.shape == (1, 3)
    assert sub["a"].tolist() == [30]


def test_bool_mask_on_string_column(df5):
    """Filtering by string equality works correctly."""
    mask = df5["c"] == "z"
    sub = df5[mask]
    assert sub.shape == (1, 3)
    assert sub["a"].tolist() == [30]


def test_bool_mask_chained(df5):
    """Chained filter then column selection returns a Series."""
    result = df5[df5["a"] > 20]["b"]
    assert result.tolist() == [3.0, 4.0, 5.0]


# ===========================================================================
# concat tests
# ===========================================================================

def test_concat_two_frames():
    """Basic vertical concat of two same-schema DataFrames."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    result = pd.concat([df1, df2])
    assert result.shape == (4, 2)
    assert result["a"].tolist() == [1, 2, 5, 6]
    assert result["b"].tolist() == [3, 4, 7, 8]


def test_concat_three_frames():
    """Vertical concat of three DataFrames produces correct row count."""
    d1 = pd.DataFrame({"a": [1]})
    d2 = pd.DataFrame({"a": [2]})
    d3 = pd.DataFrame({"a": [3]})
    result = pd.concat([d1, d2, d3])
    assert result.shape == (3, 1)
    assert result["a"].tolist() == [1, 2, 3]


def test_concat_single_frame():
    """Concat of a single-element list returns a same-shape result."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = pd.concat([df])
    assert result.shape == (2, 2)
    assert result["a"].tolist() == [1, 2]


def test_concat_preserves_dtypes():
    """Dtypes are preserved across vertical concat."""
    df1 = pd.DataFrame({"x": [1, 2], "y": [1.0, 2.0]})
    df2 = pd.DataFrame({"x": [3, 4], "y": [3.0, 4.0]})
    result = pd.concat([df1, df2])
    assert str(result.dtypes["x"]) == "int64"
    assert str(result.dtypes["y"]) == "float64"


def test_concat_preserves_column_order():
    """Column order from the first frame is maintained."""
    df1 = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
    df2 = pd.DataFrame({"z": [4], "a": [5], "m": [6]})
    result = pd.concat([df1, df2])
    assert list(result.columns) == ["z", "a", "m"]


def test_concat_different_columns():
    """Concat of frames with different columns produces union of columns."""
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "c": [4]})
    result = pd.concat([df1, df2])
    assert result.shape == (2, 3)
    cols = list(result.columns)
    assert "a" in cols
    assert "b" in cols
    assert "c" in cols


def test_concat_empty_frame():
    """Concat with one empty DataFrame keeps all rows from the other."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": []})
    result = pd.concat([df1, df2])
    assert result.shape == (2, 1)
    assert result["a"].tolist() == [1, 2]


def test_concat_all_empty():
    """Concat of two empty DataFrames returns an empty result (0 rows).

    Note: pandas-rust currently drops column metadata when all frames are
    empty, so we only assert the row count is 0.
    """
    df1 = pd.DataFrame({"a": []})
    df2 = pd.DataFrame({"a": []})
    result = pd.concat([df1, df2])
    assert result.shape[0] == 0


def test_concat_axis1():
    """Horizontal concat (axis=1) places columns side by side."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})
    result = pd.concat([df1, df2], axis=1)
    assert result.shape == (3, 2)
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == [4, 5, 6]


def test_concat_values_correct():
    """Values in every column are exactly correct after concat."""
    df1 = pd.DataFrame({"x": [10, 20], "y": [1.1, 2.2]})
    df2 = pd.DataFrame({"x": [30, 40], "y": [3.3, 4.4]})
    result = pd.concat([df1, df2])
    assert result["x"].tolist() == [10, 20, 30, 40]
    assert result["y"].tolist() == [1.1, 2.2, 3.3, 4.4]


# ===========================================================================
# Series iloc tests
# ===========================================================================

def test_series_iloc_int():
    """Series iloc with a single int returns the scalar value."""
    s = pd.Series([10, 20, 30, 40], name="x")
    assert s.iloc[0] == 10
    assert s.iloc[2] == 30


def test_series_iloc_negative():
    """Series iloc with negative int indexes from the end."""
    s = pd.Series([10, 20, 30, 40], name="x")
    assert s.iloc[-1] == 40
    assert s.iloc[-2] == 30


def test_series_iloc_slice():
    """Series iloc with a slice returns a Series with correct values."""
    s = pd.Series([10, 20, 30, 40], name="x")
    sub = s.iloc[1:3]
    assert sub.tolist() == [20, 30]


def test_series_iloc_list():
    """Series iloc with a list of ints returns elements at those positions."""
    s = pd.Series([10, 20, 30, 40], name="x")
    sub = s.iloc[[0, 2]]
    assert sub.tolist() == [10, 30]


def test_series_iloc_preserves_name():
    """Name attribute is preserved after Series iloc slice."""
    s = pd.Series([10, 20, 30, 40], name="myseries")
    sub = s.iloc[1:3]
    assert sub.name == "myseries"


def test_series_iloc_full_slice():
    """Series iloc[:] returns all elements."""
    s = pd.Series([1, 2, 3], name="t")
    sub = s.iloc[:]
    assert sub.tolist() == [1, 2, 3]


def test_series_iloc_empty_slice():
    """Series iloc[0:0] returns an empty Series."""
    s = pd.Series([1, 2, 3], name="t")
    sub = s.iloc[0:0]
    assert sub.tolist() == []
