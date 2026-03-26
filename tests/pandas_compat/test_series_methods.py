"""Upstream-style pandas tests for Series methods."""
import pytest
from pandas import Series, DataFrame


# ---------------------------------------------------------------------------
# sort_values
# ---------------------------------------------------------------------------

def test_sort_values_ascending():
    s = Series([3, 1, 2])
    result = s.sort_values()
    assert result.tolist() == [1, 2, 3]


def test_sort_values_descending():
    s = Series([3, 1, 2])
    result = s.sort_values(ascending=False)
    assert result.tolist() == [3, 2, 1]


def test_sort_values_float():
    s = Series([1.5, 0.5, 2.5, -1.0])
    result = s.sort_values()
    assert result.tolist() == [-1.0, 0.5, 1.5, 2.5]


def test_sort_values_string():
    s = Series(["banana", "apple", "cherry"])
    result = s.sort_values()
    assert result.tolist() == ["apple", "banana", "cherry"]


# ---------------------------------------------------------------------------
# value_counts
# ---------------------------------------------------------------------------

def test_value_counts_sorted():
    s = Series([1, 2, 2, 3, 3, 3])
    result = s.value_counts()
    assert isinstance(result, DataFrame)
    counts = result["count"].tolist()
    # sorted by frequency descending by default
    assert counts[0] == 3


def test_value_counts_ascending():
    s = Series([1, 2, 2, 3, 3, 3])
    result = s.value_counts(sort=True, ascending=True)
    counts = result["count"].tolist()
    assert counts[0] == 1


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_basic():
    s = Series([1, 2, 2, 3, 1])
    result = s.unique()
    # Should contain 1, 2, 3 (preserves first-seen order)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_unique_first_seen_order():
    s = Series([3, 1, 2, 1, 3])
    result = s.unique()
    assert list(result) == [3, 1, 2]


def test_unique_all_unique():
    s = Series([5, 4, 3, 2, 1])
    result = s.unique()
    assert len(result) == 5


# ---------------------------------------------------------------------------
# nunique
# ---------------------------------------------------------------------------

def test_nunique_basic():
    s = Series([1, 2, 2, 3, 3, 3])
    assert s.nunique() == 3


def test_nunique_with_null():
    s = Series([1, None, 2, None, 1])
    assert s.nunique(dropna=True) == 2
    assert s.nunique(dropna=False) == 3


# ---------------------------------------------------------------------------
# duplicated
# ---------------------------------------------------------------------------

def test_duplicated_keep_first():
    s = Series([1, 2, 1, 3, 2])
    result = s.duplicated(keep="first")
    assert result.tolist() == [False, False, True, False, True]


def test_duplicated_keep_last():
    s = Series([1, 2, 1, 3, 2])
    result = s.duplicated(keep="last")
    assert result.tolist() == [True, True, False, False, False]


# ---------------------------------------------------------------------------
# astype
# ---------------------------------------------------------------------------

def test_astype_int_to_float():
    s = Series([1, 2, 3])
    result = s.astype(float)
    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_astype_float_to_int():
    s = Series([1.0, 2.0, 3.0])
    result = s.astype(int)
    assert result.dtype == "int64"
    assert result.tolist() == [1, 2, 3]


def test_astype_int_to_str():
    s = Series([1, 2, 3])
    result = s.astype(str)
    assert result.dtype == "object"
    assert result.tolist() == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# abs
# ---------------------------------------------------------------------------

def test_abs_int():
    s = Series([-3, 0, 5, -1])
    result = s.abs()
    assert result.tolist() == [3, 0, 5, 1]


def test_abs_float():
    s = Series([-1.5, 0.0, 2.5])
    result = s.abs()
    assert result.tolist() == [1.5, 0.0, 2.5]


def test_abs_already_positive():
    s = Series([1, 2, 3])
    result = s.abs()
    assert result.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------

def test_clip_lower():
    s = Series([-5, 0, 5, 10])
    result = s.clip(lower=0)
    assert result.tolist() == [0, 0, 5, 10]


def test_clip_upper():
    s = Series([1, 5, 10, 20])
    result = s.clip(upper=9)
    assert result.tolist() == [1, 5, 9, 9]


def test_clip_both():
    s = Series([1, 5, 10])
    result = s.clip(lower=2, upper=8)
    assert result.tolist() == [2, 5, 8]


def test_clip_float():
    s = Series([0.5, 2.0, 4.5])
    result = s.clip(lower=1.0, upper=3.0)
    assert result.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# isin
# ---------------------------------------------------------------------------

def test_isin_basic():
    s = Series([1, 2, 3, 4, 5])
    result = s.isin([2, 4])
    assert result.tolist() == [False, True, False, True, False]


def test_isin_strings():
    s = Series(["a", "b", "c", "d"])
    result = s.isin(["a", "c"])
    assert result.tolist() == [True, False, True, False]


def test_isin_empty_list():
    s = Series([1, 2, 3])
    result = s.isin([])
    assert result.tolist() == [False, False, False]


def test_isin_all_match():
    s = Series([1, 2, 3])
    result = s.isin([1, 2, 3])
    assert result.tolist() == [True, True, True]


# ---------------------------------------------------------------------------
# between
# ---------------------------------------------------------------------------

def test_between_basic():
    s = Series([1, 2, 3, 4, 5])
    result = s.between(2, 4)
    assert result.tolist() == [False, True, True, True, False]


def test_between_includes_endpoints():
    s = Series([1, 2, 3])
    result = s.between(1, 3)
    assert result.tolist() == [True, True, True]


def test_between_float():
    s = Series([0.5, 1.5, 2.5, 3.5])
    result = s.between(1.0, 3.0)
    assert result.tolist() == [False, True, True, False]


# ---------------------------------------------------------------------------
# nlargest / nsmallest
# ---------------------------------------------------------------------------

def test_nlargest_basic():
    s = Series([3, 1, 4, 1, 5, 9, 2, 6])
    result = s.nlargest(3)
    assert result.shape[0] == 3
    vals = result.tolist()
    assert 9 in vals
    assert 6 in vals


def test_nsmallest_basic():
    s = Series([3, 1, 4, 1, 5, 9, 2, 6])
    result = s.nsmallest(3)
    assert result.shape[0] == 3
    vals = result.tolist()
    assert 1 in vals


def test_nlargest_n_equals_len():
    s = Series([3, 1, 2])
    result = s.nlargest(3)
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# map (dict)
# ---------------------------------------------------------------------------

def test_map_dict_basic():
    s = Series([1, 2, 3])
    result = s.map({1: 10, 2: 20, 3: 30})
    assert result.tolist() == [10, 20, 30]


def test_map_dict_missing_key():
    s = Series([1, 2, 3])
    result = s.map({1: 10, 2: 20})
    # 3 is not in dict -> None/NaN
    vals = result.tolist()
    assert vals[0] == 10
    assert vals[1] == 20
    assert vals[2] is None


def test_map_dict_str():
    s = Series(["a", "b", "c"])
    result = s.map({"a": "x", "b": "y", "c": "z"})
    assert result.tolist() == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

def test_replace_int():
    s = Series([1, 2, 3, 2, 1])
    result = s.replace(2, 99)
    assert result.tolist() == [1, 99, 3, 99, 1]


def test_replace_string():
    s = Series(["a", "b", "a", "c"])
    result = s.replace("a", "x")
    assert result.tolist() == ["x", "b", "x", "c"]


def test_replace_no_match():
    s = Series([1, 2, 3])
    result = s.replace(9, 99)
    assert result.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# any / all
# ---------------------------------------------------------------------------

def test_any_true():
    s = Series([False, True, False])
    assert s.any() == True


def test_any_false():
    s = Series([False, False, False])
    assert s.any() == False


def test_all_true():
    s = Series([True, True, True])
    assert s.all() == True


def test_all_false():
    s = Series([True, False, True])
    assert s.all() == False


def test_any_numeric():
    s = Series([0, 0, 1])
    assert s.any() == True


def test_all_numeric():
    s = Series([1, 2, 3])
    assert s.all() == True


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------

def test_copy_independence():
    s = Series([1, 2, 3])
    copy = s.copy()
    # They should have same values
    assert copy.tolist() == s.tolist()


def test_copy_is_different_object():
    s = Series([1, 2, 3])
    copy = s.copy()
    assert copy is not s
