"""
Round 12 comprehensive tests covering:
- iloc advanced indexing
- loc with boolean masks
- columns setter
- Series construction
- query
- stack
- explode
- describe
- cov/corr
- cumulative/shift/rank/rolling/expanding
"""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0], 'c': ['x', 'y', 'z']})


def make_num_df():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# iloc advanced (6 tests)
# ---------------------------------------------------------------------------

class TestIlocAdvanced:
    def test_iloc_single_row_all_cols_returns_dict(self):
        df = make_df()
        row = df.iloc[0, :]
        assert isinstance(row, dict)

    def test_iloc_single_row_all_cols_values_match(self):
        df = make_df()
        row = df.iloc[0, :]
        assert row['a'] == 1
        assert row['b'] == 4.0
        assert row['c'] == 'x'

    def test_iloc_all_rows_first_col_returns_series(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        col = df.iloc[:, 0]
        assert hasattr(col, 'tolist')
        assert col.tolist() == [1, 2, 3]

    def test_iloc_scalar(self):
        df = make_df()
        val = df.iloc[0, 0]
        assert val == 1

    def test_iloc_negative_indexing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        val = df.iloc[-1, -1]
        assert val == 6

    def test_iloc_step_slicing(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        result = df.iloc[::2]
        assert result.shape == (3, 2)
        assert result['a'].tolist() == [1, 3, 5]


# ---------------------------------------------------------------------------
# loc with boolean mask (6 tests)
# ---------------------------------------------------------------------------

class TestLocBooleanMask:
    def test_loc_bool_series_filters_rows(self):
        df = make_df()
        result = df.loc[df['a'] > 1]
        assert result.shape == (2, 3)
        assert result['a'].tolist() == [2, 3]

    def test_loc_bool_series_single_col_returns_series_values(self):
        df = make_df()
        result = df.loc[df['a'] > 1, 'b']
        assert result.tolist() == [5.0, 6.0]

    def test_loc_bool_series_multi_col_returns_dataframe(self):
        df = make_df()
        result = df.loc[df['a'] > 1, ['a', 'b']]
        assert result.shape == (2, 2)
        assert result.columns.tolist() == ['a', 'b']

    def test_loc_compound_mask(self):
        df = make_num_df()
        mask = (df['a'] > 1) & (df['b'] < 6)
        result = df.loc[mask]
        assert result.shape == (1, 2)
        assert result['a'].tolist() == [2]

    def test_loc_all_true_returns_full_df(self):
        df = make_num_df()
        all_true = df['a'] > 0
        result = df.loc[all_true]
        assert result.shape == df.shape

    def test_loc_all_false_returns_empty(self):
        df = make_num_df()
        all_false = df['a'] > 100
        result = df.loc[all_false]
        assert result.shape[0] == 0
        assert result.shape[1] == 2


# ---------------------------------------------------------------------------
# columns setter (4 tests)
# ---------------------------------------------------------------------------

class TestColumnsSetter:
    def test_set_columns_renames_all(self):
        df = make_df()
        df.columns = ['X', 'Y', 'Z']
        assert df.columns.tolist() == ['X', 'Y', 'Z']

    def test_set_columns_preserves_data(self):
        df = make_df()
        original_a = df['a'].tolist()
        df.columns = ['X', 'Y', 'Z']
        assert df['X'].tolist() == original_a

    def test_set_columns_wrong_length_raises(self):
        df = make_df()
        with pytest.raises((ValueError, Exception)):
            df.columns = ['X', 'Y']  # 2 instead of 3

    def test_set_columns_then_access_by_new_name(self):
        df = make_df()
        df.columns = ['p', 'q', 'r']
        assert df['q'].tolist() == [4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# Series construction (5 tests)
# ---------------------------------------------------------------------------

class TestSeriesConstruction:
    def test_from_range(self):
        s = pd.Series(range(5), name='r')
        assert s.tolist() == [0, 1, 2, 3, 4]

    def test_from_numpy_array(self):
        s = pd.Series(np.array([10, 20, 30]), name='np')
        assert s.tolist() == [10, 20, 30]

    def test_from_dict(self):
        s = pd.Series({'a': 1, 'b': 2, 'c': 3}, name='d')
        assert s.tolist() == [1, 2, 3]

    def test_from_scalar_with_index(self):
        s = pd.Series(7, index=[0, 1, 2], name='x')
        assert s.tolist() == [7, 7, 7]

    def test_from_empty_list(self):
        s = pd.Series([], name='empty')
        assert len(s) == 0


# ---------------------------------------------------------------------------
# query (5 tests)
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_basic_gt(self):
        df = make_num_df()
        result = df.query('a > 2')
        assert result['a'].tolist() == [3]

    def test_query_equality(self):
        df = make_num_df()
        result = df.query('a == 1')
        assert result['a'].tolist() == [1]

    def test_query_compound_and(self):
        df = make_num_df()
        result = df.query('a > 1 and b < 6')
        assert result.shape == (1, 2)
        assert result['a'].tolist() == [2]

    def test_query_preserves_columns(self):
        df = make_num_df()
        result = df.query('a > 1')
        assert result.columns.tolist() == ['a', 'b']

    def test_query_empty_result(self):
        df = make_num_df()
        result = df.query('a > 100')
        assert result.shape[0] == 0
        assert result.columns.tolist() == ['a', 'b']


# ---------------------------------------------------------------------------
# stack (3 tests)
# ---------------------------------------------------------------------------

class TestStack:
    def test_stack_int_df_flattens(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        stacked = df.stack()
        assert hasattr(stacked, 'tolist')

    def test_stack_length_equals_nrows_times_ncols(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        stacked = df.stack()
        assert len(stacked) == 6  # 3 rows * 2 cols

    def test_stack_float_df(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        stacked = df.stack()
        assert len(stacked) == 6


# ---------------------------------------------------------------------------
# explode (4 tests)
# ---------------------------------------------------------------------------

class TestExplode:
    def test_series_explode_lists(self):
        s = pd.Series([[1, 2], [3, 4]], name='x')
        result = s.explode()
        assert result.tolist() == [1, 2, 3, 4]

    def test_series_explode_mixed(self):
        s = pd.Series([[1, 2], [3]], name='x')
        result = s.explode()
        assert result.tolist() == [1, 2, 3]

    def test_dataframe_explode_column(self):
        df = pd.DataFrame({'a': [[1, 2], [3]], 'b': [10, 20]})
        result = df.explode('a')
        assert result.shape == (3, 2)
        assert result['a'].tolist() == [1, 2, 3]

    def test_dataframe_explode_preserves_other_columns(self):
        df = pd.DataFrame({'a': [[1, 2], [3]], 'b': [10, 20]})
        result = df.explode('a')
        assert result['b'].tolist() == [10, 10, 20]


# ---------------------------------------------------------------------------
# describe (4 tests)
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_series_describe_numeric(self):
        s = pd.Series([1, 2, 3, 4, 5], name='x')
        desc = s.describe()
        assert desc is not None

    def test_series_describe_includes_stats(self):
        s = pd.Series([1, 2, 3, 4, 5], name='x')
        desc = s.describe()
        cols = desc.columns.tolist()
        assert 'count' in cols
        assert 'mean' in cols
        assert 'std' in cols
        assert 'min' in cols
        assert 'max' in cols

    def test_dataframe_describe_numeric_cols(self):
        df = make_num_df()
        desc = df.describe()
        assert 'a' in desc.columns.tolist()
        assert 'b' in desc.columns.tolist()

    def test_dataframe_describe_shape(self):
        df = make_num_df()
        desc = df.describe()
        # 5 stats rows (count, mean, std, min, max at minimum), 2 numeric cols
        assert desc.shape[1] == 2
        assert desc.shape[0] >= 5


# ---------------------------------------------------------------------------
# cov/corr (5 tests)
# ---------------------------------------------------------------------------

class TestCovCorr:
    def test_df_cov_shape(self):
        df = make_num_df()
        cov = df.cov()
        assert cov.shape == (2, 2)

    def test_df_cov_diagonal_equals_variance(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 4.0, 6.0]})
        cov = df.cov()
        # var([1,2,3], ddof=1) == 1.0
        a_var = cov['a'].tolist()[0]
        assert abs(a_var - 1.0) < 1e-9

    def test_series_corr_perfect_positive(self):
        s1 = pd.Series([1, 2, 3], name='x')
        s2 = pd.Series([2, 4, 6], name='y')
        assert abs(s1.corr(s2) - 1.0) < 1e-9

    def test_series_corr_perfect_negative(self):
        s1 = pd.Series([1, 2, 3], name='x')
        s2 = pd.Series([3, 2, 1], name='y')
        assert abs(s1.corr(s2) - (-1.0)) < 1e-9

    def test_series_cov_basic_value(self):
        s1 = pd.Series([1, 2, 3], name='x')
        s2 = pd.Series([2, 4, 6], name='y')
        # cov([1,2,3], [2,4,6]) = 2.0
        assert abs(s1.cov(s2) - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# DataFrame cumulative / shift / rank / rolling / expanding (8 tests)
# ---------------------------------------------------------------------------

class TestCumulativeShiftRankRolling:
    def test_df_cumsum_values(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = df.cumsum()
        assert result['a'].tolist() == [1, 3, 6]

    def test_df_shift_first_row_is_null(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        shifted = df.shift(1)
        assert shifted['a'].iloc[0] is None

    def test_df_shift_subsequent_values(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        shifted = df.shift(1)
        assert shifted['a'].tolist()[1:] == [1, 2]

    def test_df_rank_values(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = df.rank()
        assert result['a'].tolist() == [1.0, 2.0, 3.0]

    def test_df_rolling_mean(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = df.rolling(2).mean()
        vals = result['a'].tolist()
        assert vals[0] is None
        assert abs(vals[1] - 1.5) < 1e-9
        assert abs(vals[2] - 2.5) < 1e-9

    def test_df_expanding_sum_matches_cumsum(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        expanding = df.expanding().sum()
        cumsum = df.cumsum()
        assert expanding['a'].tolist() == cumsum['a'].tolist()

    def test_df_cummax(self):
        df = pd.DataFrame({'a': [1, 3, 2]})
        result = df.cummax()
        assert result['a'].tolist() == [1, 3, 3]

    def test_df_idxmax_returns_dict(self):
        df = pd.DataFrame({'a': [1, 3, 2], 'b': [5, 4, 6]})
        result = df.idxmax()
        assert isinstance(result, dict)
        assert result['a'] == 1  # index of max value 3
        assert result['b'] == 2  # index of max value 6
