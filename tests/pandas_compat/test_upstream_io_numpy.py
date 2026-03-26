"""
Upstream-adapted pandas compatibility tests for:
  - DataFrame.to_csv / pd.read_csv
  - DataFrame.to_numpy / Series.to_numpy / .values
  - DataFrame.to_dict
  - Series.to_frame
  - pd.melt
  - DataFrame.pivot_table

Adapted from:
  pandas/tests/frame/methods/test_to_csv.py
  pandas/tests/frame/methods/test_to_numpy.py
  pandas/tests/series/methods/test_to_frame.py
  pandas/tests/reshape/test_melt.py
  pandas/tests/reshape/test_pivot.py
"""
import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame.to_csv
# ---------------------------------------------------------------------------

def test_to_csv_returns_string():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_csv()
    assert isinstance(result, str)


def test_to_csv_contains_header():
    df = pd.DataFrame({"col1": [1], "col2": [2]})
    result = df.to_csv()
    assert "col1" in result
    assert "col2" in result


def test_to_csv_contains_values():
    df = pd.DataFrame({"a": [42, 99]})
    result = df.to_csv()
    assert "42" in result
    assert "99" in result


def test_to_csv_to_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = str(tmp_path / "out.csv")
    df.to_csv(path)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "x" in content


def test_to_csv_round_trip_preserves_shape(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = str(tmp_path / "round.csv")
    df.to_csv(path)
    df2 = pd.read_csv(path)
    # After round-trip we may have an extra index column; check value columns exist
    assert "a" in df2.columns or any("a" in str(c) for c in df2.columns)


def test_to_csv_round_trip_column_names(tmp_path):
    df = pd.DataFrame({"alpha": [1], "beta": [2]})
    path = str(tmp_path / "cols.csv")
    df.to_csv(path)
    df2 = pd.read_csv(path)
    # to_csv includes index column; alpha and beta should still appear
    assert "alpha" in df2.columns
    assert "beta" in df2.columns


# ---------------------------------------------------------------------------
# pd.read_csv
# ---------------------------------------------------------------------------

def test_read_csv_basic(tmp_path):
    path = str(tmp_path / "basic.csv")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(path)
    result = pd.read_csv(path)
    assert isinstance(result, pd.DataFrame)
    assert "a" in result.columns
    assert "b" in result.columns


def test_read_csv_preserves_row_count(tmp_path):
    path = str(tmp_path / "rows.csv")
    df = pd.DataFrame({"x": [10, 20, 30, 40]})
    df.to_csv(path)
    result = pd.read_csv(path)
    assert len(result) == 4


def test_read_csv_preserves_values(tmp_path):
    path = str(tmp_path / "vals.csv")
    df = pd.DataFrame({"v": [100, 200, 300]})
    df.to_csv(path)
    result = pd.read_csv(path)
    vals = result["v"].tolist()
    assert vals == [100, 200, 300]


# ---------------------------------------------------------------------------
# DataFrame.to_numpy
# ---------------------------------------------------------------------------

def test_df_to_numpy_returns_ndarray():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_numpy()
    assert isinstance(result, np.ndarray)


def test_df_to_numpy_shape_matches():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.to_numpy()
    assert result.shape == (3, 2)


def test_df_to_numpy_int_values():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_numpy()
    assert list(result[0]) == [1.0, 3.0] or list(result[0]) == [1, 3]


def test_df_to_numpy_float_values():
    df = pd.DataFrame({"a": [1.5, 2.5], "b": [3.5, 4.5]})
    result = df.to_numpy()
    assert abs(result[0][0] - 1.5) < 1e-9
    assert abs(result[0][1] - 3.5) < 1e-9


# ---------------------------------------------------------------------------
# Series.to_numpy / Series.values
# ---------------------------------------------------------------------------

def test_series_to_numpy_returns_ndarray():
    s = pd.Series([1, 2, 3], name="x")
    result = s.to_numpy()
    assert isinstance(result, np.ndarray)


def test_series_to_numpy_1d():
    s = pd.Series([1, 2, 3], name="x")
    result = s.to_numpy()
    assert result.ndim == 1


def test_series_to_numpy_values():
    s = pd.Series([10, 20, 30], name="x")
    result = s.to_numpy()
    assert list(result) == [10, 20, 30]


def test_series_values_returns_ndarray():
    s = pd.Series([1, 2, 3], name="x")
    result = s.values
    assert isinstance(result, np.ndarray)


def test_series_values_same_data():
    s = pd.Series([5, 6, 7], name="x")
    arr = s.values
    assert list(arr) == [5, 6, 7]


# ---------------------------------------------------------------------------
# DataFrame.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_default_returns_dict():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_dict()
    assert isinstance(result, dict)


def test_to_dict_default_has_column_keys():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_dict()
    assert "a" in result
    assert "b" in result


def test_to_dict_records_returns_list():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.to_dict("records")
    assert isinstance(result, list)
    assert len(result) == 2


def test_to_dict_records_has_correct_keys():
    df = pd.DataFrame({"x": [10], "y": [20]})
    result = df.to_dict("records")
    assert result[0]["x"] == 10
    assert result[0]["y"] == 20


# ---------------------------------------------------------------------------
# Series.to_frame
# ---------------------------------------------------------------------------

def test_to_frame_returns_dataframe():
    s = pd.Series([1, 2, 3], name="vals")
    result = s.to_frame()
    assert isinstance(result, pd.DataFrame)


def test_to_frame_uses_series_name():
    s = pd.Series([1, 2, 3], name="myname")
    result = s.to_frame()
    assert "myname" in result.columns


def test_to_frame_custom_name():
    s = pd.Series([1, 2, 3], name="x")
    result = s.to_frame("custom")
    assert "custom" in result.columns


def test_to_frame_preserves_values():
    s = pd.Series([10, 20, 30], name="x")
    result = s.to_frame()
    assert result["x"].tolist() == [10, 20, 30]


def test_to_frame_shape():
    s = pd.Series([1, 2, 3, 4], name="x")
    result = s.to_frame()
    assert result.shape == (4, 1)


# ---------------------------------------------------------------------------
# pd.melt
# ---------------------------------------------------------------------------

def test_melt_basic():
    df = pd.DataFrame({"id": [1, 2], "a": [3, 4], "b": [5, 6]})
    result = pd.melt(df, id_vars=["id"])
    assert isinstance(result, pd.DataFrame)


def test_melt_with_id_vars_columns():
    df = pd.DataFrame({"id": [1, 2], "a": [3, 4], "b": [5, 6]})
    result = pd.melt(df, id_vars=["id"])
    assert "id" in result.columns
    assert "variable" in result.columns
    assert "value" in result.columns


def test_melt_row_count():
    # 2 rows, 2 value columns -> 4 rows in melted
    df = pd.DataFrame({"id": [1, 2], "a": [3, 4], "b": [5, 6]})
    result = pd.melt(df, id_vars=["id"])
    assert len(result) == 4


def test_melt_custom_names():
    df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    result = pd.melt(df, id_vars=["id"], var_name="my_var", value_name="my_val")
    assert "my_var" in result.columns
    assert "my_val" in result.columns


# ---------------------------------------------------------------------------
# DataFrame.pivot_table
# ---------------------------------------------------------------------------

def test_pivot_table_basic():
    df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    result = df.pivot_table(values="v", index="k", aggfunc="sum")
    assert isinstance(result, pd.DataFrame)


def test_pivot_table_sum_values():
    df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    result = df.pivot_table(values="v", index="k", aggfunc="sum")
    # "a" -> 1+3=4, "b" -> 2
    vals = result["v"].tolist()
    assert 4 in vals
    assert 2 in vals


def test_pivot_table_mean_aggfunc():
    df = pd.DataFrame({"k": ["a", "a", "b"], "v": [10.0, 20.0, 30.0]})
    result = df.pivot_table(values="v", index="k", aggfunc="mean")
    assert isinstance(result, pd.DataFrame)
    vals = result["v"].tolist()
    assert abs(vals[0] - 15.0) < 1e-9 or abs(vals[1] - 15.0) < 1e-9
