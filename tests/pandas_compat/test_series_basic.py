"""
Pandas compatibility tests for Series basic properties.
Covers: construction, name, dtype, len, tolist, copy, repr, sort_values, astype.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_series_from_int_list():
    s = pd.Series([1, 2, 3], name="x")
    assert len(s) == 3
    assert s.dtype == "int64"


def test_series_from_float_list():
    s = pd.Series([1.0, 2.5, 3.7], name="f")
    assert len(s) == 3
    assert s.dtype == "float64"


def test_series_from_string_list():
    s = pd.Series(["a", "b", "c"], name="s")
    assert len(s) == 3
    assert s.dtype == "object"


def test_series_from_bool_list():
    s = pd.Series([True, False, True], name="b")
    assert len(s) == 3
    assert s.dtype == "bool"


def test_series_from_single_element():
    s = pd.Series([42], name="single")
    assert len(s) == 1
    assert s.tolist() == [42]


def test_series_from_empty_list():
    s = pd.Series([], name="empty")
    assert len(s) == 0
    assert s.tolist() == []


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------

def test_series_name():
    s = pd.Series([1, 2, 3], name="myname")
    assert s.name == "myname"


def test_series_name_string():
    s = pd.Series([1], name="test_col")
    assert isinstance(s.name, str)


def test_series_name_from_dataframe_column():
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    s = df["col_a"]
    assert s.name == "col_a"


# ---------------------------------------------------------------------------
# dtype property
# ---------------------------------------------------------------------------

def test_series_dtype_int64():
    s = pd.Series([1, 2, 3], name="x")
    assert s.dtype == "int64"


def test_series_dtype_float64():
    s = pd.Series([1.0, 2.0], name="x")
    assert s.dtype == "float64"


def test_series_dtype_object():
    s = pd.Series(["a", "b"], name="x")
    assert s.dtype == "object"


def test_series_dtype_bool():
    s = pd.Series([True, False], name="x")
    assert s.dtype == "bool"


def test_series_dtype_is_string():
    s = pd.Series([1, 2, 3], name="x")
    assert isinstance(s.dtype, str)


# ---------------------------------------------------------------------------
# len()
# ---------------------------------------------------------------------------

def test_series_len():
    s = pd.Series([10, 20, 30, 40], name="x")
    assert len(s) == 4


def test_series_len_empty():
    s = pd.Series([], name="x")
    assert len(s) == 0


def test_series_len_single():
    s = pd.Series([99], name="x")
    assert len(s) == 1


# ---------------------------------------------------------------------------
# tolist() roundtrip
# ---------------------------------------------------------------------------

def test_tolist_int_roundtrip():
    values = [1, 2, 3, 4, 5]
    s = pd.Series(values, name="x")
    assert s.tolist() == values


def test_tolist_float_roundtrip():
    values = [1.1, 2.2, 3.3]
    s = pd.Series(values, name="x")
    result = s.tolist()
    for a, b in zip(result, values):
        assert abs(a - b) < 1e-10


def test_tolist_string_roundtrip():
    values = ["hello", "world", "foo"]
    s = pd.Series(values, name="x")
    assert s.tolist() == values


def test_tolist_bool_roundtrip():
    values = [True, False, True]
    s = pd.Series(values, name="x")
    assert s.tolist() == values


def test_tolist_returns_list():
    s = pd.Series([1, 2, 3], name="x")
    assert isinstance(s.tolist(), list)


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------

def test_series_copy_same_values():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.copy()
    assert s2.tolist() == [1, 2, 3]


def test_series_copy_same_name():
    s = pd.Series([1, 2, 3], name="myname")
    s2 = s.copy()
    assert s2.name == "myname"


def test_series_copy_same_len():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.copy()
    assert len(s2) == len(s)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_returns_string():
    s = pd.Series([1, 2, 3], name="x")
    r = repr(s)
    assert isinstance(r, str)


def test_repr_contains_name():
    s = pd.Series([1, 2, 3], name="mycolname")
    r = repr(s)
    assert "mycolname" in r


def test_repr_nonempty():
    s = pd.Series([1, 2, 3], name="x")
    assert len(repr(s)) > 0


# ---------------------------------------------------------------------------
# sort_values
# ---------------------------------------------------------------------------

def test_sort_values_ascending():
    s = pd.Series([3, 1, 2], name="x")
    s2 = s.sort_values()
    assert s2.tolist() == [1, 2, 3]


def test_sort_values_ascending_explicit():
    s = pd.Series([3, 1, 2], name="x")
    s2 = s.sort_values(ascending=True)
    assert s2.tolist() == [1, 2, 3]


def test_sort_values_descending():
    s = pd.Series([3, 1, 2], name="x")
    s2 = s.sort_values(ascending=False)
    assert s2.tolist() == [3, 2, 1]


def test_sort_values_does_not_mutate():
    s = pd.Series([3, 1, 2], name="x")
    _ = s.sort_values()
    assert s.tolist() == [3, 1, 2]


def test_sort_values_strings():
    s = pd.Series(["banana", "apple", "cherry"], name="x")
    s2 = s.sort_values()
    assert s2.tolist() == ["apple", "banana", "cherry"]


def test_sort_values_already_sorted():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.sort_values()
    assert s2.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# astype
# ---------------------------------------------------------------------------

def test_astype_int_to_float():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.astype("float64")
    assert s2.dtype == "float64"
    assert s2.tolist() == [1.0, 2.0, 3.0]


def test_astype_float_to_int():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    s2 = s.astype("int64")
    assert s2.dtype == "int64"
    assert s2.tolist() == [1, 2, 3]


def test_astype_int_to_object():
    s = pd.Series([1, 2, 3], name="x")
    s2 = s.astype("object")
    assert s2.dtype == "object"
    assert s2.tolist() == ["1", "2", "3"]


def test_astype_preserves_length():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    s2 = s.astype("float64")
    assert len(s2) == 5
