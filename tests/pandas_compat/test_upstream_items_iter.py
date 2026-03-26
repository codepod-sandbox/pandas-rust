"""Tests for items/iteritems, is_monotonic, argmax/argmin, prod, drop."""
import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import pandas as our_pd


# ---------------------------------------------------------------------------
# items / iteritems — DataFrame
# ---------------------------------------------------------------------------

def test_df_items_yields_col_series_pairs():
    df = our_pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pairs = list(df.items())
    assert len(pairs) == 2
    col_names = [p[0] for p in pairs]
    assert col_names == ["a", "b"]


def test_df_items_length_equals_ncols():
    df = our_pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    assert len(list(df.items())) == 3


def test_df_items_values_are_series():
    df = our_pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    for name, col in df.items():
        assert hasattr(col, "tolist"), "Expected Series-like object"


def test_df_items_values_correct():
    df = our_pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    d = {name: col.tolist() for name, col in df.items()}
    assert d == {"a": [1, 2], "b": [3, 4]}


def test_df_iteritems_alias():
    df = our_pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert list(df.iteritems()) == list(df.items())


def test_df_items_empty():
    df = our_pd.DataFrame({})
    assert list(df.items()) == []


# ---------------------------------------------------------------------------
# items / iteritems — Series
# ---------------------------------------------------------------------------

def test_series_items_yields_index_value_pairs():
    s = our_pd.Series([10, 20, 30])
    pairs = list(s.items())
    assert pairs == [(0, 10), (1, 20), (2, 30)]


def test_series_items_length():
    s = our_pd.Series([1, 2, 3, 4, 5])
    assert len(list(s.items())) == 5


def test_series_items_order_preserved():
    s = our_pd.Series([100, 200, 300])
    indices = [i for i, _ in s.items()]
    assert indices == [0, 1, 2]


def test_series_iteritems_alias():
    s = our_pd.Series([1, 2, 3])
    assert list(s.iteritems()) == list(s.items())


def test_series_items_empty():
    s = our_pd.Series([])
    assert list(s.items()) == []


# ---------------------------------------------------------------------------
# is_monotonic_increasing / is_monotonic_decreasing
# ---------------------------------------------------------------------------

def test_increasing_true():
    s = our_pd.Series([1, 2, 3, 4])
    assert s.is_monotonic_increasing is True


def test_increasing_false_for_decreasing():
    s = our_pd.Series([4, 3, 2, 1])
    assert s.is_monotonic_increasing is False


def test_decreasing_true():
    s = our_pd.Series([4, 3, 2, 1])
    assert s.is_monotonic_decreasing is True


def test_decreasing_false_for_increasing():
    s = our_pd.Series([1, 2, 3, 4])
    assert s.is_monotonic_decreasing is False


def test_constant_both_monotonic():
    s = our_pd.Series([5, 5, 5])
    assert s.is_monotonic_increasing is True
    assert s.is_monotonic_decreasing is True


def test_empty_series_monotonic():
    s = our_pd.Series([])
    assert s.is_monotonic_increasing is True
    assert s.is_monotonic_decreasing is True


# ---------------------------------------------------------------------------
# argmax / argmin
# ---------------------------------------------------------------------------

def test_argmax_basic():
    s = our_pd.Series([1, 5, 3, 2])
    assert s.argmax() == 1


def test_argmin_basic():
    s = our_pd.Series([3, 1, 4, 1, 5])
    assert s.argmin() == 1


def test_argmax_floats():
    s = our_pd.Series([1.1, 2.2, 3.3])
    assert s.argmax() == 2


def test_argmax_equals_idxmax():
    s = our_pd.Series([10, 30, 20])
    assert s.argmax() == s.idxmax()


# ---------------------------------------------------------------------------
# prod / product
# ---------------------------------------------------------------------------

def test_series_prod_basic():
    s = our_pd.Series([2, 3, 4])
    assert s.prod() == 24


def test_df_prod_per_column():
    df = our_pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.prod()
    assert result["a"] == 6
    assert result["b"] == 120


def test_series_prod_ones():
    s = our_pd.Series([1, 1, 1])
    assert s.prod() == 1


def test_series_product_alias():
    s = our_pd.Series([2, 3, 4])
    assert s.product() == s.prod()


# ---------------------------------------------------------------------------
# Series.drop
# ---------------------------------------------------------------------------

def test_series_drop_single():
    s = our_pd.Series([10, 20, 30])
    result = s.drop(0)
    assert result.tolist() == [20, 30]


def test_series_drop_multiple():
    s = our_pd.Series([10, 20, 30, 40])
    result = s.drop([0, 2])
    assert result.tolist() == [20, 40]


def test_series_drop_last():
    s = our_pd.Series([1, 2, 3])
    result = s.drop(2)
    assert result.tolist() == [1, 2]


def test_series_drop_preserves_remaining():
    s = our_pd.Series([100, 200, 300, 400])
    result = s.drop([1, 3])
    assert result.tolist() == [100, 300]
