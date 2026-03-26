"""Tests for idxmax/min, rolling/expanding for DataFrame, explode, describe/sem for Series,
and ndarray/dict construction."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import pandas as pd


# ---------------------------------------------------------------------------
# idxmax / idxmin DataFrame (4 tests)
# ---------------------------------------------------------------------------

def test_df_idxmax_returns_dict():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [9, 2, 7]})
    result = df.idxmax()
    assert isinstance(result, dict)
    assert "a" in result and "b" in result


def test_df_idxmin_returns_dict():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [9, 2, 7]})
    result = df.idxmin()
    assert isinstance(result, dict)
    assert "a" in result and "b" in result


def test_df_idxmax_correct_values():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [9, 2, 7]})
    result = df.idxmax()
    assert result["a"] == 1   # index 1 has value 5
    assert result["b"] == 0   # index 0 has value 9


def test_df_idxmin_correct_values():
    df = pd.DataFrame({"a": [1, 5, 3], "b": [9, 2, 7]})
    result = df.idxmin()
    assert result["a"] == 0   # index 0 has value 1
    assert result["b"] == 1   # index 1 has value 2


# ---------------------------------------------------------------------------
# rolling DataFrame (6 tests)
# ---------------------------------------------------------------------------

def test_df_rolling_mean_shape():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    result = df.rolling(2).mean()
    assert result.shape == (4, 2)


def test_df_rolling_sum_values():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    result = df.rolling(2).sum()
    vals = result["a"].tolist()
    assert vals[0] is None
    assert vals[1] == pytest.approx(3.0)
    assert vals[2] == pytest.approx(5.0)
    assert vals[3] == pytest.approx(7.0)


def test_df_rolling_min():
    df = pd.DataFrame({"a": [5.0, 3.0, 8.0, 1.0]})
    result = df.rolling(2).min()
    vals = result["a"].tolist()
    assert vals[0] is None
    assert vals[1] == pytest.approx(3.0)
    assert vals[3] == pytest.approx(1.0)


def test_df_rolling_preserves_columns():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    result = df.rolling(2).mean()
    assert list(result.columns) == ["x", "y"]


def test_df_rolling_first_values_are_none():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    result = df.rolling(3).mean()
    vals = result["a"].tolist()
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is not None


def test_df_rolling_mixed_types_skips_strings():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    result = df.rolling(2).mean()
    # numeric column should be present
    assert "a" in result.columns
    a_vals = result["a"].tolist()
    assert a_vals[1] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# expanding DataFrame (5 tests)
# ---------------------------------------------------------------------------

def test_df_expanding_sum_values():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df.expanding().sum()
    vals = result["a"].tolist()
    assert vals == pytest.approx([1.0, 3.0, 6.0])


def test_df_expanding_mean():
    df = pd.DataFrame({"a": [2.0, 4.0, 6.0]})
    result = df.expanding().mean()
    vals = result["a"].tolist()
    assert vals == pytest.approx([2.0, 3.0, 4.0])


def test_df_expanding_count():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = df.expanding().count()
    vals = result["a"].tolist()
    assert vals[0] == 1
    assert vals[1] == 2
    assert vals[2] == 3


def test_df_expanding_preserves_columns():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    result = df.expanding().sum()
    assert "x" in result.columns and "y" in result.columns


def test_df_expanding_mixed_types():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["p", "q", "r"]})
    result = df.expanding().sum()
    assert "a" in result.columns


# ---------------------------------------------------------------------------
# explode (6 tests)
# ---------------------------------------------------------------------------

def test_df_explode_expands_lists():
    df = pd.DataFrame({"a": [[1, 2], [3]], "b": [10, 20]})
    result = df.explode("a")
    assert len(result) == 3


def test_df_explode_preserves_other_columns():
    df = pd.DataFrame({"a": [[1, 2], [3]], "b": [10, 20]})
    result = df.explode("a")
    b_vals = result["b"].tolist()
    assert b_vals == [10, 10, 20]


def test_df_explode_scalar_values_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.explode("a")
    assert len(result) == 3
    assert result["a"].tolist() == [1, 2, 3]


def test_series_explode_list_of_lists():
    s = pd.Series([[1, 2], [3, 4]])
    result = s.explode()
    assert result.tolist() == [1, 2, 3, 4]


def test_series_explode_mixed_lists_and_scalars():
    s = pd.Series([[1, 2], 3, [4]])
    result = s.explode()
    assert result.tolist() == [1, 2, 3, 4]


def test_series_explode_empty_lists():
    s = pd.Series([[], [1, 2]])
    result = s.explode()
    assert result.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Series.describe (4 tests)
# ---------------------------------------------------------------------------

def test_series_describe_numeric_includes_stats():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.describe()
    cols = list(result.columns)
    for stat in ("count", "mean", "std", "min", "max"):
        assert stat in cols


def test_series_describe_includes_percentiles():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = s.describe()
    cols = list(result.columns)
    assert "25%" in cols
    assert "50%" in cols
    assert "75%" in cols


def test_series_describe_float_series():
    s = pd.Series([10.0, 20.0, 30.0])
    result = s.describe()
    mean_val = result["mean"].tolist()[0]
    assert mean_val == pytest.approx(20.0)


def test_series_describe_shape():
    s = pd.Series([1.0, 2.0, 3.0])
    result = s.describe()
    # Should have 1 row and 8 stat columns
    assert result.shape[0] == 1
    assert result.shape[1] == 8


# ---------------------------------------------------------------------------
# sem (3 tests)
# ---------------------------------------------------------------------------

def test_series_sem_basic():
    s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    result = s.sem()
    assert result is not None
    assert result > 0


def test_series_sem_equals_std_over_sqrt_n():
    import math
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = s.std() / math.sqrt(s.count())
    assert s.sem() == pytest.approx(expected)


def test_series_sem_single_value():
    s = pd.Series([42.0])
    # std of single value is None or 0 depending on impl; sem should handle gracefully
    result = s.sem()
    # With n=1, std is None or 0, sem should be None or 0
    assert result is None or result == 0.0


# ---------------------------------------------------------------------------
# Construction (6 tests)
# ---------------------------------------------------------------------------

def test_series_from_dict():
    s = pd.Series({"a": 1, "b": 2, "c": 3})
    vals = s.tolist()
    assert 1 in vals
    assert 2 in vals
    assert 3 in vals


def test_df_from_ndarray():
    try:
        import numpy as np
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        df = pd.DataFrame(arr)
        assert df.shape == (3, 2)
    except ImportError:
        pytest.skip("numpy not available")


def test_df_from_ndarray_with_columns():
    try:
        import numpy as np
        arr = np.array([[1, 2], [3, 4]])
        df = pd.DataFrame(arr, columns=["x", "y"])
        assert list(df.columns) == ["x", "y"]
    except ImportError:
        pytest.skip("numpy not available")


def test_series_dict_preserves_values():
    s = pd.Series({"x": 10, "y": 20})
    assert 10 in s.tolist()
    assert 20 in s.tolist()
    assert len(s) == 2


def test_df_ndarray_roundtrip():
    try:
        import numpy as np
        original = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        arr = original.to_numpy()
        df2 = pd.DataFrame(arr, columns=["a", "b"])
        assert df2.shape == original.shape
    except ImportError:
        pytest.skip("numpy not available")


def test_df_2d_ndarray_shape():
    try:
        import numpy as np
        arr = np.zeros((5, 3))
        df = pd.DataFrame(arr)
        assert df.shape == (5, 3)
    except ImportError:
        pytest.skip("numpy not available")


# ---------------------------------------------------------------------------
# Integration (5 tests)
# ---------------------------------------------------------------------------

def test_rolling_mean_on_real_data():
    df = pd.DataFrame({"sales": [100.0, 200.0, 150.0, 250.0, 300.0]})
    result = df.rolling(3).mean()
    vals = result["sales"].tolist()
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] == pytest.approx((100 + 200 + 150) / 3)
    assert vals[4] == pytest.approx((150 + 250 + 300) / 3)


def test_expanding_sum_matches_cumsum():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    expanding_sum = s.expanding().sum().tolist()
    cumsum = s.cumsum().tolist()
    assert expanding_sum == pytest.approx(cumsum)


def test_explode_then_groupby():
    df = pd.DataFrame({"group": ["A", "B"], "values": [[1, 2], [3, 4]]})
    exploded = df.explode("values")
    assert len(exploded) == 4
    assert set(exploded["group"].tolist()) == {"A", "B"}


def test_describe_transpose_shows_all_stats():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    desc = s.describe()
    transposed = desc.T
    # After transpose, row count equals number of stat columns
    assert transposed.shape[0] == 8


def test_idxmax_then_iloc_retrieves_max():
    df = pd.DataFrame({"a": [10.0, 99.0, 50.0]})
    idx = df.idxmax()["a"]
    row = df.iloc[idx]
    assert row["a"] == pytest.approx(99.0)
