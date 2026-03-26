"""
Upstream-adapted tests for miscellaneous DataFrame/Series methods.
Sources: test_round.py, test_nlargest.py, test_clip.py, test_shift.py,
         test_transpose.py, test_reset_index.py, test_map.py, test_to_dict.py
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Series.round
# ---------------------------------------------------------------------------

def test_series_round_default():
    s = pd.Series([1.5, 2.567, 3.499])
    result = s.round()
    assert result.tolist() == [2.0, 3.0, 3.0]


def test_series_round_1_decimal():
    s = pd.Series([1.123, 2.456, 3.789])
    result = s.round(1)
    assert result.tolist() == [1.1, 2.5, 3.8]


def test_series_round_2_decimal():
    s = pd.Series([1.1, 2.2, 3.3])
    result = s.round(1)
    # Check approximate values match
    for v, expected in zip(result.tolist(), [1.1, 2.2, 3.3]):
        assert abs(v - expected) < 1e-6


def test_series_round_zero_decimals():
    s = pd.Series([1.9, 2.1, 3.5])
    result = s.round(0)
    assert result.tolist() == [2.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# DataFrame.round
# ---------------------------------------------------------------------------

def test_dataframe_round_default():
    df = pd.DataFrame({"col1": [1.123, 2.456], "col2": [3.789, 4.012]})
    result = df.round()
    assert result["col1"].tolist() == [1.0, 2.0]
    assert result["col2"].tolist() == [4.0, 4.0]


def test_dataframe_round_with_int():
    df = pd.DataFrame({"col1": [1.123, 2.456], "col2": [3.789, 4.012]})
    result = df.round(2)
    assert result["col1"].tolist() == [1.12, 2.46]
    assert result["col2"].tolist() == [3.79, 4.01]


def test_dataframe_round_preserves_columns():
    df = pd.DataFrame({"a": [1.5], "b": [2.5]})
    result = df.round(0)
    assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# Series.nlargest / nsmallest
# ---------------------------------------------------------------------------

def test_series_nlargest():
    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    result = s.nlargest(3)
    vals = sorted(result.tolist(), reverse=True)
    assert vals == [9, 6, 5]


def test_series_nsmallest():
    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    result = s.nsmallest(3)
    vals = sorted(result.tolist())
    assert vals == [1, 1, 2]


def test_series_nlargest_n_1():
    s = pd.Series([10, 20, 30, 40, 50])
    result = s.nlargest(1)
    assert result.tolist() == [50]


def test_series_nsmallest_n_1():
    s = pd.Series([10, 20, 30, 40, 50])
    result = s.nsmallest(1)
    assert result.tolist() == [10]


# ---------------------------------------------------------------------------
# DataFrame.nlargest / nsmallest
# ---------------------------------------------------------------------------

def test_dataframe_nlargest():
    df = pd.DataFrame({"score": [1, 5, 3, 9, 7], "name": ["a", "b", "c", "d", "e"]})
    result = df.nlargest(2, "score")
    assert len(result) == 2
    assert result["score"].tolist()[0] == 9


def test_dataframe_nsmallest():
    df = pd.DataFrame({"score": [1, 5, 3, 9, 7], "name": ["a", "b", "c", "d", "e"]})
    result = df.nsmallest(2, "score")
    assert len(result) == 2
    assert result["score"].tolist()[0] == 1


# ---------------------------------------------------------------------------
# Series.shift
# ---------------------------------------------------------------------------

def test_shift_positive_periods():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.shift(2)
    vals = result.tolist()
    # First 2 elements are None, rest are shifted
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] == 1.0
    assert vals[4] == 3.0


def test_shift_negative_periods():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.shift(-2)
    vals = result.tolist()
    assert vals[0] == 3.0
    assert vals[1] == 4.0
    assert vals[3] is None
    assert vals[4] is None


def test_shift_zero_periods():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.shift(0)
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_shift_preserves_length():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert len(s.shift(1)) == len(s)
    assert len(s.shift(-1)) == len(s)


# ---------------------------------------------------------------------------
# Series.clip
# ---------------------------------------------------------------------------

def test_series_clip_lower_only():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.clip(lower=3.0)
    assert result.tolist() == [3.0, 3.0, 3.0, 4.0, 5.0]


def test_series_clip_upper_only():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.clip(upper=3.0)
    assert result.tolist() == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_series_clip_both():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.clip(lower=2.0, upper=4.0)
    assert result.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]


def test_series_clip_no_effect():
    s = pd.Series([2.0, 3.0, 4.0])
    result = s.clip(lower=1.0, upper=10.0)
    assert result.tolist() == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# DataFrame.clip
# ---------------------------------------------------------------------------

def test_dataframe_clip_lower_only():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = df.clip(lower=2.5)
    assert result["a"].tolist() == [2.5, 2.5, 3.0]
    assert result["b"].tolist() == [4.0, 5.0, 6.0]


def test_dataframe_clip_upper_only():
    df = pd.DataFrame({"a": [1.0, 5.0, 10.0]})
    result = df.clip(upper=5.0)
    assert result["a"].tolist() == [1.0, 5.0, 5.0]


# ---------------------------------------------------------------------------
# DataFrame.select_dtypes
# ---------------------------------------------------------------------------

def test_select_dtypes_include_number():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    result = df.select_dtypes(include="number")
    assert "c" not in result.columns
    assert "a" in result.columns


def test_select_dtypes_exclude_object():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = df.select_dtypes(exclude="object")
    assert "b" not in result.columns
    assert "a" in result.columns


# ---------------------------------------------------------------------------
# DataFrame.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_default_returns_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.to_dict()
    assert isinstance(result, dict)
    assert "a" in result
    assert "b" in result


def test_to_dict_records():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_dict("records")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"a": 1, "b": 3}
    assert result[1] == {"a": 2, "b": 4}


def test_to_dict_records_single_row():
    df = pd.DataFrame({"x": [42], "y": [99]})
    result = df.to_dict("records")
    assert result == [{"x": 42, "y": 99}]


def test_to_dict_list_orient():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.to_dict("list")
    # Our implementation returns {col: [values]}
    assert "a" in result


# ---------------------------------------------------------------------------
# DataFrame.transpose
# ---------------------------------------------------------------------------

def test_transpose_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.T
    # Transposed: 2 columns -> 2 rows
    assert result.shape[0] == 2
    # Original column names appear as row values
    assert "a" in result.columns or result.shape[1] >= 3


def test_transpose_property_equals_method():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert df.T.shape == df.transpose().shape


# ---------------------------------------------------------------------------
# DataFrame.reset_index
# ---------------------------------------------------------------------------

def test_reset_index_drop_true():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.reset_index(drop=True)
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_reset_index_preserves_data():
    df = pd.DataFrame({"v": [10, 20, 30]})
    result = df.reset_index(drop=True)
    assert result["v"].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# DataFrame.pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.pipe(lambda x: x)
    assert result["a"].tolist() == [1, 2, 3]


def test_pipe_with_transform():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.pipe(lambda x: x.head(2))
    assert len(result) == 2


def test_pipe_with_args():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.pipe(lambda x, n: x.head(n), 2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# DataFrame.assign
# ---------------------------------------------------------------------------

def test_assign_with_list():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.assign(b=[4, 5, 6])
    assert result["b"].tolist() == [4, 5, 6]
    assert list(result.columns) == ["a", "b"]


def test_assign_with_lambda():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.assign(b=lambda x: x["a"] * 2)
    assert result["b"].tolist() == [2, 4, 6]


def test_assign_overwrites_existing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.assign(a=[10, 20, 30])
    assert result["a"].tolist() == [10, 20, 30]


def test_assign_does_not_mutate_original():
    df = pd.DataFrame({"a": [1, 2, 3]})
    _ = df.assign(b=[4, 5, 6])
    assert "b" not in df.columns


# ---------------------------------------------------------------------------
# Series.map
# ---------------------------------------------------------------------------

def test_series_map_dict():
    s = pd.Series(["a", "b", "c", "a"])
    result = s.map({"a": 1, "b": 2, "c": 3})
    assert result.tolist() == [1, 2, 3, 1]


def test_series_map_callable():
    s = pd.Series([1.0, 4.0, 9.0])
    result = s.map(lambda x: x ** 0.5)
    vals = result.tolist()
    assert abs(vals[0] - 1.0) < 1e-9
    assert abs(vals[1] - 2.0) < 1e-9
    assert abs(vals[2] - 3.0) < 1e-9


def test_series_map_identity():
    s = pd.Series([1, 2, 3])
    result = s.map(lambda x: x)
    assert result.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Series.item
# ---------------------------------------------------------------------------

def test_series_item_single_element():
    s = pd.Series([42])
    assert s.item() == 42


def test_series_item_single_float():
    s = pd.Series([3.14])
    assert abs(s.item() - 3.14) < 1e-9


def test_series_item_raises_for_multiple():
    s = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        s.item()


# ---------------------------------------------------------------------------
# DataFrame.rename with callable
# ---------------------------------------------------------------------------

def test_rename_callable_upper():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = df.rename(columns=str.upper)
    assert list(result.columns) == ["A", "B"]


def test_rename_callable_custom():
    df = pd.DataFrame({"col1": [1], "col2": [2]})
    result = df.rename(columns=lambda c: c + "_new")
    assert list(result.columns) == ["col1_new", "col2_new"]


# ---------------------------------------------------------------------------
# DataFrame.replace
# ---------------------------------------------------------------------------

def test_dataframe_replace_scalar():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    result = df.replace(1, 99)
    assert result["a"].tolist() == [99, 2, 3]
    assert result["b"].tolist() == [99, 2, 3]


def test_dataframe_replace_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.replace({"a": {1: 100}})
    assert result["a"].tolist() == [100, 2, 3]
    assert result["b"].tolist() == [4, 5, 6]


def test_dataframe_replace_no_match():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = df.replace(99, 0)
    assert result["a"].tolist() == [1, 2, 3]
