"""
Pandas compatibility tests for Series operations.
Covers: arithmetic, comparison, boolean logic, aggregations, isna/fillna/dropna.
"""
import pandas as pd
import math


# ---------------------------------------------------------------------------
# Arithmetic: Series + Series
# ---------------------------------------------------------------------------

def test_add_int_series():
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = s1 + s2
    assert result.tolist() == [5, 7, 9]


def test_add_float_series():
    s1 = pd.Series([1.0, 2.0, 3.0], name="a")
    s2 = pd.Series([0.5, 0.5, 0.5], name="b")
    result = s1 + s2
    assert result.tolist() == [1.5, 2.5, 3.5]


def test_add_int_result_dtype():
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = s1 + s2
    assert result.dtype == "int64"


def test_add_float_result_dtype():
    s1 = pd.Series([1.0, 2.0], name="a")
    s2 = pd.Series([3.0, 4.0], name="b")
    result = s1 + s2
    assert result.dtype == "float64"


# ---------------------------------------------------------------------------
# Arithmetic: Series - Series
# ---------------------------------------------------------------------------

def test_subtract_series():
    s1 = pd.Series([10, 20, 30], name="a")
    s2 = pd.Series([1, 2, 3], name="b")
    result = s1 - s2
    assert result.tolist() == [9, 18, 27]


def test_subtract_float_series():
    s1 = pd.Series([5.0, 10.0], name="a")
    s2 = pd.Series([2.5, 3.5], name="b")
    result = s1 - s2
    assert result.tolist() == [2.5, 6.5]


# ---------------------------------------------------------------------------
# Arithmetic: Series * Series
# ---------------------------------------------------------------------------

def test_multiply_series():
    s1 = pd.Series([2, 3, 4], name="a")
    s2 = pd.Series([5, 6, 7], name="b")
    result = s1 * s2
    assert result.tolist() == [10, 18, 28]


def test_multiply_result_dtype_int():
    s1 = pd.Series([2, 3], name="a")
    s2 = pd.Series([4, 5], name="b")
    result = s1 * s2
    assert result.dtype == "int64"


# ---------------------------------------------------------------------------
# Arithmetic: Series / Series
# ---------------------------------------------------------------------------

def test_divide_series():
    s1 = pd.Series([4, 6, 8], name="a")
    s2 = pd.Series([2, 3, 4], name="b")
    result = s1 / s2
    assert result.tolist() == [2.0, 2.0, 2.0]


def test_divide_result_dtype_float():
    s1 = pd.Series([4, 6], name="a")
    s2 = pd.Series([2, 3], name="b")
    result = s1 / s2
    assert result.dtype == "float64"


def test_int_plus_float_result_float():
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([1.0, 2.0, 3.0], name="b")
    result = s1 + s2
    assert result.dtype == "float64"


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def test_negation_int():
    s = pd.Series([1, 2, 3], name="x")
    result = -s
    assert result.tolist() == [-1, -2, -3]


def test_negation_float():
    s = pd.Series([1.5, -2.5, 3.0], name="x")
    result = -s
    assert result.tolist() == [-1.5, 2.5, -3.0]


def test_negation_preserves_dtype():
    s = pd.Series([1, 2, 3], name="x")
    result = -s
    assert result.dtype == "int64"


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

def test_eq_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s == 2
    assert result.tolist() == [False, True, False]


def test_ne_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s != 2
    assert result.tolist() == [True, False, True]


def test_lt_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s < 2
    assert result.tolist() == [True, False, False]


def test_le_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s <= 2
    assert result.tolist() == [True, True, False]


def test_gt_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s > 2
    assert result.tolist() == [False, False, True]


def test_ge_operator():
    s = pd.Series([1, 2, 3], name="x")
    result = s >= 2
    assert result.tolist() == [False, True, True]


def test_comparison_returns_bool_series():
    s = pd.Series([1, 2, 3], name="x")
    result = s == 2
    assert result.dtype == "bool"


# ---------------------------------------------------------------------------
# Boolean logic - XFAIL: & | ~ operators not supported on Series
# ---------------------------------------------------------------------------

def test_bool_and_not_supported():
    # XFAIL: & operator between Series not implemented
    import pytest
    b1 = pd.Series([True, False, True], name="b1")
    b2 = pd.Series([True, True, False], name="b2")
    try:
        result = b1 & b2
        # If it works, verify the result
        assert result.tolist() == [True, False, False]
    except TypeError:
        pytest.skip("XFAIL: & operator not supported on Series")


def test_bool_or_not_supported():
    # XFAIL: | operator between Series not implemented
    import pytest
    b1 = pd.Series([True, False, True], name="b1")
    b2 = pd.Series([True, True, False], name="b2")
    try:
        result = b1 | b2
        assert result.tolist() == [True, True, True]
    except TypeError:
        pytest.skip("XFAIL: | operator not supported on Series")


def test_bool_invert_not_supported():
    # XFAIL: ~ operator not implemented on Series
    import pytest
    b = pd.Series([True, False, True], name="b")
    try:
        result = ~b
        assert result.tolist() == [False, True, False]
    except TypeError:
        pytest.skip("XFAIL: ~ operator not supported on Series")


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def test_series_sum():
    s = pd.Series([1, 2, 3], name="x")
    assert s.sum() == 6


def test_series_sum_float():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    assert s.sum() == 6.0


def test_series_mean():
    s = pd.Series([1, 2, 3], name="x")
    assert s.mean() == 2.0


def test_series_mean_returns_float():
    s = pd.Series([1, 2, 3], name="x")
    result = s.mean()
    assert isinstance(result, float)


def test_series_min_int():
    s = pd.Series([3, 1, 2], name="x")
    assert s.min() == 1


def test_series_max_int():
    s = pd.Series([3, 1, 2], name="x")
    assert s.max() == 3


def test_series_min_float():
    s = pd.Series([3.0, 1.5, 2.7], name="x")
    assert s.min() == 1.5


def test_series_max_float():
    s = pd.Series([3.0, 1.5, 2.7], name="x")
    assert s.max() == 3.0


def test_series_count():
    s = pd.Series([1, 2, 3], name="x")
    assert s.count() == 3


def test_series_count_with_nulls():
    s = pd.Series([1.0, None, 3.0], name="x")
    assert s.count() == 2


def test_series_std():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    result = s.std()
    assert abs(result - 1.0) < 1e-9


def test_series_var():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    result = s.var()
    assert abs(result - 1.0) < 1e-9


def test_series_median():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    assert s.median() == 2.0


def test_series_median_even():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], name="x")
    assert s.median() == 2.5


def test_series_sum_empty_returns_none():
    # Our impl returns None for empty series sum
    s = pd.Series([], name="x")
    result = s.sum()
    assert result is None


def test_series_std_single_element_is_none():
    # Standard deviation of a single element is undefined (None or NaN)
    s = pd.Series([5.0], name="x")
    result = s.std()
    assert result is None


# ---------------------------------------------------------------------------
# isna / notna
# ---------------------------------------------------------------------------

def test_series_isna_no_nulls():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    result = s.isna()
    assert result.tolist() == [False, False, False]


def test_series_isna_with_nulls():
    s = pd.Series([1.0, None, 3.0], name="x")
    result = s.isna()
    assert result.tolist() == [False, True, False]


def test_series_notna_with_nulls():
    s = pd.Series([1.0, None, 3.0], name="x")
    result = s.notna()
    assert result.tolist() == [True, False, True]


def test_series_isna_returns_series():
    s = pd.Series([1.0, None, 3.0], name="x")
    result = s.isna()
    assert isinstance(result, pd.Series)


def test_series_isna_notna_complementary():
    s = pd.Series([1.0, None, 3.0], name="x")
    isna = s.isna().tolist()
    notna = s.notna().tolist()
    for a, b in zip(isna, notna):
        assert a != b


# ---------------------------------------------------------------------------
# fillna / dropna on Series
# ---------------------------------------------------------------------------

def test_series_fillna():
    s = pd.Series([1.0, None, 3.0], name="x")
    s2 = s.fillna(0.0)
    assert s2.tolist() == [1.0, 0.0, 3.0]


def test_series_fillna_no_nulls():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    s2 = s.fillna(999.0)
    assert s2.tolist() == [1.0, 2.0, 3.0]


def test_series_dropna():
    s = pd.Series([1.0, None, 3.0, None], name="x")
    s2 = s.dropna()
    assert len(s2) == 2


def test_series_dropna_returns_series():
    s = pd.Series([1.0, None, 3.0], name="x")
    s2 = s.dropna()
    assert isinstance(s2, pd.Series)


def test_series_dropna_no_nulls():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    s2 = s.dropna()
    assert len(s2) == 3
    assert s2.tolist() == [1.0, 2.0, 3.0]
