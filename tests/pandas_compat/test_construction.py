"""Upstream-style pandas tests for DataFrame and Series construction."""
import pytest
from pandas import DataFrame, Series


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def test_dataframe_from_dict_of_lists():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 2, 3]
    assert df["b"].tolist() == [4, 5, 6]


def test_dataframe_from_dict_preserves_column_order():
    df = DataFrame({"z": [1], "a": [2], "m": [3]})
    assert list(df.columns) == ["z", "a", "m"]


def test_dataframe_from_dict_float():
    df = DataFrame({"x": [1.0, 2.5, 3.14]})
    assert df["x"].dtype == "float64"


def test_dataframe_from_dict_string():
    df = DataFrame({"name": ["alice", "bob"]})
    assert df["name"].tolist() == ["alice", "bob"]
    assert df["name"].dtype == "object"


def test_dataframe_from_dict_bool():
    df = DataFrame({"flag": [True, False, True]})
    assert df["flag"].tolist() == [True, False, True]


def test_dataframe_from_list_of_dicts():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df = DataFrame(data)
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 3]
    assert df["b"].tolist() == [2, 4]


def test_dataframe_from_list_of_dicts_missing_keys():
    data = [{"a": 1, "b": 2}, {"a": 3}]
    df = DataFrame(data)
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 3]
    # Missing key b in second row -> None
    b_vals = df["b"].tolist()
    assert b_vals[0] == 2
    assert b_vals[1] is None


def test_dataframe_from_list_of_dicts_extra_key_in_second():
    data = [{"a": 1}, {"a": 2, "b": 3}]
    df = DataFrame(data)
    # Both keys should exist
    assert "a" in df.columns
    assert "b" in df.columns
    a_vals = df["a"].tolist()
    assert a_vals == [1, 2]


def test_dataframe_from_empty_dict():
    df = DataFrame({})
    assert df.shape == (0, 0)


def test_dataframe_from_empty_list():
    df = DataFrame([])
    assert df.shape == (0, 0)


def test_dataframe_dtypes_dict():
    df = DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
    dtypes = df.dtypes
    assert isinstance(dtypes, dict)
    assert dtypes["a"] == "int64"
    assert dtypes["b"] == "float64"
    assert dtypes["c"] == "object"


def test_dataframe_shape():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert df.shape == (3, 3)


def test_dataframe_len():
    df = DataFrame({"a": [1, 2, 3, 4]})
    assert len(df) == 4


def test_dataframe_single_column():
    df = DataFrame({"x": [42]})
    assert df.shape == (1, 1)
    assert df["x"].tolist() == [42]


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------

def test_series_from_int_list():
    s = Series([1, 2, 3])
    assert s.dtype == "int64"
    assert s.tolist() == [1, 2, 3]


def test_series_from_float_list():
    s = Series([1.0, 2.5, 3.14])
    assert s.dtype == "float64"
    assert s.tolist() == [1.0, 2.5, 3.14]


def test_series_from_str_list():
    s = Series(["a", "b", "c"])
    assert s.dtype == "object"
    assert s.tolist() == ["a", "b", "c"]


def test_series_from_bool_list():
    s = Series([True, False, True])
    assert s.dtype == "bool"
    assert s.tolist() == [True, False, True]


def test_series_name_property():
    s = Series([1, 2, 3], name="myname")
    assert s.name == "myname"


def test_series_default_name():
    s = Series([1, 2, 3])
    # Default name is "0"
    assert s.name == "0"


def test_series_len():
    s = Series([10, 20, 30, 40])
    assert len(s) == 4


def test_series_from_empty_list():
    # Empty list should not crash
    # Default dtype is float64
    s = Series([])
    assert len(s) == 0


def test_series_mixed_int_float_promotes():
    # Passing floats should give float dtype
    s = Series([1.0, 2.0, 3.0])
    assert s.dtype == "float64"


def test_series_getitem():
    s = Series([10, 20, 30])
    assert s[0] == 10
    assert s[1] == 20
    assert s[2] == 30


def test_series_negative_index():
    s = Series([10, 20, 30])
    assert s[-1] == 30


def test_series_arithmetic():
    s = Series([1, 2, 3])
    result = s + 1
    assert result.tolist() == [2, 3, 4]


def test_series_comparison():
    s = Series([1, 2, 3])
    mask = s > 1
    assert mask.tolist() == [False, True, True]
