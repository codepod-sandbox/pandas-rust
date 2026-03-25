"""
Pandas compatibility tests for DataFrame basic construction and properties.
Modeled after pandas upstream tests, adapted for our supported functionality.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construct_from_dict_ints():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)


def test_construct_from_dict_floats():
    df = pd.DataFrame({"x": [1.0, 2.5, 3.7]})
    assert df.shape == (3, 1)
    assert df.dtypes["x"] == "float64"


def test_construct_from_dict_strings():
    df = pd.DataFrame({"s": ["hello", "world", "foo"]})
    assert df.shape == (3, 1)
    assert df.dtypes["s"] == "object"


def test_construct_from_dict_bools():
    df = pd.DataFrame({"flag": [True, False, True]})
    assert df.shape == (3, 1)
    assert df.dtypes["flag"] == "bool"


def test_construct_from_dict_mixed_types():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    assert df.shape == (3, 3)
    assert df.dtypes["a"] == "int64"
    assert df.dtypes["b"] == "float64"
    assert df.dtypes["c"] == "object"


def test_construct_from_empty_dict():
    df = pd.DataFrame({})
    assert df.shape == (0, 0)
    assert list(df.columns) == []


def test_construct_single_column():
    df = pd.DataFrame({"only": [42]})
    assert df.shape == (1, 1)


def test_construct_many_columns():
    data = {str(i): [i * 10] for i in range(10)}
    df = pd.DataFrame(data)
    assert df.shape == (1, 10)


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------

def test_shape_returns_tuple():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = df.shape
    assert isinstance(s, tuple)
    assert len(s) == 2


def test_shape_nrows_ncols():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert df.shape[0] == 5
    assert df.shape[1] == 2


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------

def test_dtypes_returns_dict():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    d = df.dtypes
    assert isinstance(d, dict)


def test_dtypes_int_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert df.dtypes["a"] == "int64"


def test_dtypes_float_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert df.dtypes["a"] == "float64"


def test_dtypes_object_column():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    assert df.dtypes["a"] == "object"


def test_dtypes_bool_column():
    df = pd.DataFrame({"a": [True, False, True]})
    assert df.dtypes["a"] == "bool"


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------

def test_columns_returns_list():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    cols = df.columns
    assert isinstance(cols, list)


def test_columns_preserves_order():
    df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
    assert list(df.columns) == ["z", "a", "m"]


def test_columns_contains_all_keys():
    df = pd.DataFrame({"x": [1], "y": [2]})
    assert "x" in df.columns
    assert "y" in df.columns


def test_columns_strings():
    df = pd.DataFrame({"col1": [1], "col2": [2]})
    for col in df.columns:
        assert isinstance(col, str)


# ---------------------------------------------------------------------------
# len()
# ---------------------------------------------------------------------------

def test_len_matches_nrows():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert len(df) == 5


def test_len_empty():
    df = pd.DataFrame({})
    assert len(df) == 0


def test_len_single_row():
    df = pd.DataFrame({"a": [99]})
    assert len(df) == 1


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------

def test_copy_same_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df2 = df.copy()
    assert df2.shape == df.shape


def test_copy_independent_modification():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df.copy()
    df2["a"] = [10, 20, 30]
    assert df["a"].tolist() == [1, 2, 3]
    assert df2["a"].tolist() == [10, 20, 30]


def test_copy_same_columns():
    df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    df2 = df.copy()
    assert list(df2.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_returns_string():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    r = repr(df)
    assert isinstance(r, str)


def test_repr_contains_column_names():
    df = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
    r = repr(df)
    assert "alpha" in r
    assert "beta" in r


def test_repr_nonempty():
    df = pd.DataFrame({"a": [1]})
    assert len(repr(df)) > 0


# ---------------------------------------------------------------------------
# head() and tail()
# ---------------------------------------------------------------------------

def test_head_default():
    df = pd.DataFrame({"a": list(range(10))})
    h = df.head()
    assert h.shape == (5, 1)


def test_head_custom_n():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    h = df.head(3)
    assert h.shape == (3, 1)
    assert h["a"].tolist() == [1, 2, 3]


def test_head_zero():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    h = df.head(0)
    assert h.shape == (0, 2)
    assert list(h.columns) == ["a", "b"]


def test_head_exceeds_rows():
    df = pd.DataFrame({"a": [1, 2, 3]})
    h = df.head(100)
    assert h.shape == (3, 1)


def test_tail_default():
    df = pd.DataFrame({"a": list(range(10))})
    t = df.tail()
    assert t.shape == (5, 1)


def test_tail_custom_n():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    t = df.tail(2)
    assert t.shape == (2, 1)
    assert t["a"].tolist() == [4, 5]


def test_tail_zero():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = df.tail(0)
    assert t.shape == (0, 2)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

def test_describe_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    desc = df.describe()
    assert isinstance(desc, pd.DataFrame)


def test_describe_numeric_only():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    desc = df.describe()
    assert "a" in desc.columns
    assert "b" in desc.columns
    assert "c" not in desc.columns


def test_describe_shape():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    desc = df.describe()
    # count, mean, std, min, max = 5 rows
    assert desc.shape[0] == 5
    assert desc.shape[1] == 2


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_returns_dict():
    df = pd.DataFrame({"a": [1, 2, 3]})
    d = df.to_dict()
    assert isinstance(d, dict)


def test_to_dict_has_all_columns():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    d = df.to_dict()
    assert "a" in d
    assert "b" in d


def test_to_dict_values_are_lists():
    df = pd.DataFrame({"a": [1, 2, 3]})
    d = df.to_dict()
    assert isinstance(d["a"], list)


def test_to_dict_values_correct():
    df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
    d = df.to_dict()
    assert d["a"] == [10, 20, 30]
    assert d["b"] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------

def test_info_does_not_raise():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    df.info()  # should not raise


def test_info_empty_df_does_not_raise():
    df = pd.DataFrame({})
    df.info()  # should not raise


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

def test_getitem_single_col_returns_series():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = df["a"]
    assert isinstance(s, pd.Series)
    assert s.name == "a"


def test_getitem_list_of_cols_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df2 = df[["c", "a"]]
    assert isinstance(df2, pd.DataFrame)
    assert list(df2.columns) == ["c", "a"]
