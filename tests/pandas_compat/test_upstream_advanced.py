"""Advanced DataFrame and Series pattern tests."""
import pandas as pd
import pytest


class TestAdvancedDataFrame:
    def setup_method(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
        })

    def test_df_prod_basic(self):
        result = self.df.prod()
        assert result["a"] == 24   # 1*2*3*4
        assert result["b"] == 240000  # 10*20*30*40

    def test_df_prod_mixed_types(self):
        df = pd.DataFrame({"a": [2, 3], "label": ["x", "y"]})
        result = df.prod()
        assert result["a"] == 6
        assert "label" not in result  # string columns skipped

    def test_df_bool_list_filtering(self):
        result = self.df[[True, False, True, False]]
        assert len(result) == 2
        assert result["a"].tolist() == [1, 3]

    def test_df_items_iteration_count(self):
        count = sum(1 for _ in self.df.items())
        assert count == 2

    def test_df_map_element_wise(self):
        result = self.df.map(lambda x: x * 2)
        assert result["a"].tolist() == [2, 4, 6, 8]

    def test_df_eq_method(self):
        result = self.df.eq(2)
        assert result["a"].tolist() == [False, True, False, False]

    def test_df_ne_method(self):
        result = self.df.ne(2)
        assert result["a"].tolist() == [True, False, True, True]

    def test_df_gt_method(self):
        result = self.df.gt(2)
        assert result["a"].tolist() == [False, False, True, True]

    def test_df_corr_identity(self):
        result = self.df.corr()
        # A column correlated with itself should be 1.0
        aa = result["a"].tolist()
        # corr returns a DataFrame; find the diagonal
        assert abs(aa[0] - 1.0) < 1e-9

    def test_df_T_returns_dataframe(self):
        t = self.df.T
        # Transposed result is still a DataFrame
        assert isinstance(t, pd.DataFrame)
        # Has fewer rows than the original had rows
        orig_rows, orig_cols = self.df.shape
        t_rows, _ = t.shape
        assert t_rows == orig_cols

    def test_df_empty_on_full_df(self):
        assert self.df.empty is False

    def test_df_empty_on_empty_df(self):
        empty_df = pd.DataFrame({})
        assert empty_df.empty is True

    def test_df_size_equals_rows_times_cols(self):
        rows, cols = self.df.shape
        assert self.df.size == rows * cols

    def test_df_query_basic(self):
        result = self.df.query("a > 2")
        assert len(result) == 2

    def test_df_set_axis_renames(self):
        result = self.df.set_axis(["x", "y"], axis=1)
        assert list(result.columns) == ["x", "y"]

    def test_df_reindex_fills_none(self):
        result = self.df.reindex(columns=["a", "z"])
        assert result["z"].tolist() == [None, None, None, None]

    def test_df_mask_replaces_true(self):
        result = self.df.mask(self.df > 20, other=0)
        assert result["b"].tolist() == [10, 20, 0, 0]

    def test_df_where_keeps_true(self):
        result = self.df.where(self.df > 20, other=0)
        assert result["b"].tolist() == [0, 0, 30, 40]

    def test_df_prod_returns_dict(self):
        result = self.df.prod()
        assert isinstance(result, dict)

    def test_df_items_yields_series(self):
        for col_name, col_series in self.df.items():
            assert isinstance(col_series, pd.Series)
            break


class TestAdvancedSeries:
    def setup_method(self):
        self.s = pd.Series([1, 2, 3, 4, 5])

    def test_series_is_monotonic_increasing(self):
        assert self.s.is_monotonic_increasing is True

    def test_series_is_monotonic_decreasing(self):
        assert self.s.is_monotonic_decreasing is False

    def test_series_is_monotonic_decreasing_true(self):
        s = pd.Series([5, 4, 3, 2, 1])
        assert s.is_monotonic_decreasing is True

    def test_series_argmax(self):
        assert self.s.argmax() == 4

    def test_series_argmin(self):
        assert self.s.argmin() == 0

    def test_series_prod(self):
        assert self.s.prod() == 120  # 1*2*3*4*5

    def test_series_drop_by_index(self):
        result = self.s.drop(2)
        vals = result.tolist()
        assert len(vals) == 4
        assert 3 not in vals

    def test_series_squeeze_single(self):
        s = pd.Series([42])
        assert s.squeeze() == 42

    def test_series_squeeze_multi(self):
        result = self.s.squeeze()
        assert isinstance(result, pd.Series)

    def test_series_item(self):
        s = pd.Series([99])
        assert s.item() == 99

    def test_series_is_unique_true(self):
        assert self.s.is_unique is True

    def test_series_is_unique_false(self):
        s = pd.Series([1, 2, 2, 3])
        assert s.is_unique is False

    def test_series_cumsum_diff(self):
        s = pd.Series([1.0, 2.0, 3.0])
        cumsum = s.cumsum()
        # diff of cumsum should recover original (except first element)
        diff = cumsum.diff()
        vals = diff.tolist()
        assert vals[1] == 2.0
        assert vals[2] == 3.0

    def test_series_rolling_shift_combo(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rolled = s.rolling(2).mean()
        shifted = rolled.shift(1)
        # shifted version should be one step behind
        vals = shifted.tolist()
        assert vals[0] is None  # was None after shift

    def test_series_mask_basic(self):
        cond = pd.Series([True, False, True, False, True])
        result = self.s.mask(cond, other=0)
        assert result.tolist() == [0, 2, 0, 4, 0]

    def test_series_where_basic(self):
        cond = pd.Series([True, False, True, False, True])
        result = self.s.where(cond, other=0)
        assert result.tolist() == [1, 0, 3, 0, 5]
