"""Tests for size, ndim, empty, is_unique, squeeze, to_string properties."""
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# size — DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameSize:
    def test_size_normal(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert df.size == 6

    def test_size_empty_df(self):
        df = pd.DataFrame({})
        assert df.size == 0

    def test_size_one_col(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        assert df.size == 5

    def test_size_one_row(self):
        df = pd.DataFrame({"a": [7], "b": [8], "c": [9]})
        assert df.size == 3

    def test_size_matches_shape_product(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        nrows, ncols = df.shape
        assert df.size == nrows * ncols


# ---------------------------------------------------------------------------
# size — Series
# ---------------------------------------------------------------------------

class TestSeriesSize:
    def test_size_normal(self):
        s = pd.Series([1, 2, 3], name="x")
        assert s.size == 3

    def test_size_empty(self):
        s = pd.Series([], name="x")
        assert s.size == 0

    def test_size_single(self):
        s = pd.Series([42], name="x")
        assert s.size == 1

    def test_size_equals_len(self):
        s = pd.Series([10, 20, 30, 40], name="x")
        assert s.size == len(s)


# ---------------------------------------------------------------------------
# ndim — DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameNdim:
    def test_ndim_is_2(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert df.ndim == 2

    def test_ndim_empty_df(self):
        df = pd.DataFrame({})
        assert df.ndim == 2

    def test_ndim_single_col(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert df.ndim == 2


# ---------------------------------------------------------------------------
# ndim — Series
# ---------------------------------------------------------------------------

class TestSeriesNdim:
    def test_ndim_is_1(self):
        s = pd.Series([1, 2, 3], name="x")
        assert s.ndim == 1

    def test_ndim_empty_series(self):
        s = pd.Series([], name="x")
        assert s.ndim == 1

    def test_ndim_single_element(self):
        s = pd.Series([99], name="x")
        assert s.ndim == 1


# ---------------------------------------------------------------------------
# empty — DataFrame
# ---------------------------------------------------------------------------

class TestDataFrameEmpty:
    def test_empty_no_columns(self):
        df = pd.DataFrame({})
        assert df.empty is True

    def test_empty_with_data(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert df.empty is False

    def test_empty_single_row(self):
        df = pd.DataFrame({"a": [1]})
        assert df.empty is False

    def test_empty_one_col_no_rows(self):
        # Build via filtering to get 0-row DataFrame
        df = pd.DataFrame({"a": [1, 2]})
        filtered = df[df["a"] > 100]
        assert filtered.empty is True

    def test_not_empty_multi_col(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert df.empty is False


# ---------------------------------------------------------------------------
# empty — Series
# ---------------------------------------------------------------------------

class TestSeriesEmpty:
    def test_empty_true(self):
        s = pd.Series([], name="x")
        assert s.empty is True

    def test_empty_false(self):
        s = pd.Series([1], name="x")
        assert s.empty is False

    def test_empty_false_multi(self):
        s = pd.Series([1, 2, 3], name="x")
        assert s.empty is False

    def test_empty_after_dropna(self):
        s = pd.Series([None, None], name="x")
        dropped = s.dropna()
        assert dropped.empty is True


# ---------------------------------------------------------------------------
# is_unique — Series
# ---------------------------------------------------------------------------

class TestSeriesIsUnique:
    def test_all_unique(self):
        s = pd.Series([1, 2, 3, 4, 5], name="x")
        assert s.is_unique is True

    def test_has_duplicates(self):
        s = pd.Series([1, 2, 2, 3], name="x")
        assert s.is_unique is False

    def test_single_element(self):
        s = pd.Series([42], name="x")
        assert s.is_unique is True

    def test_all_same(self):
        s = pd.Series([7, 7, 7], name="x")
        assert s.is_unique is False

    def test_strings_unique(self):
        s = pd.Series(["a", "b", "c"], name="x")
        assert s.is_unique is True

    def test_strings_not_unique(self):
        s = pd.Series(["a", "b", "a"], name="x")
        assert s.is_unique is False


# ---------------------------------------------------------------------------
# squeeze — Series
# ---------------------------------------------------------------------------

class TestSeriesSqueeze:
    def test_squeeze_single_element(self):
        s = pd.Series([42], name="x")
        result = s.squeeze()
        assert result == 42

    def test_squeeze_multiple_elements_returns_series(self):
        s = pd.Series([1, 2, 3], name="x")
        result = s.squeeze()
        assert isinstance(result, pd.Series)

    def test_squeeze_preserves_value(self):
        s = pd.Series([3.14], name="x")
        result = s.squeeze()
        assert abs(result - 3.14) < 1e-9

    def test_squeeze_string(self):
        s = pd.Series(["hello"], name="x")
        result = s.squeeze()
        assert result == "hello"


# ---------------------------------------------------------------------------
# to_string — DataFrame and Series
# ---------------------------------------------------------------------------

class TestToString:
    def test_df_to_string_returns_str(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.to_string()
        assert isinstance(result, str)

    def test_series_to_string_returns_str(self):
        s = pd.Series([1, 2, 3], name="x")
        result = s.to_string()
        assert isinstance(result, str)

    def test_df_to_string_contains_column(self):
        df = pd.DataFrame({"mycolumn": [1, 2]})
        result = df.to_string()
        assert "mycolumn" in result

    def test_series_to_string_contains_value(self):
        s = pd.Series([99], name="x")
        result = s.to_string()
        assert "99" in result

    def test_df_to_string_nonempty(self):
        df = pd.DataFrame({"a": [1]})
        assert len(df.to_string()) > 0

    def test_series_to_string_nonempty(self):
        s = pd.Series([1], name="x")
        assert len(s.to_string()) > 0
