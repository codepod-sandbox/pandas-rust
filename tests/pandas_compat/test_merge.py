"""
Pandas compatibility tests for DataFrame.merge().
Covers: inner, left, right, outer joins; multi-column keys; suffixes; edge cases.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Inner join
# ---------------------------------------------------------------------------

def test_inner_join_basic():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="inner")
    assert m.shape == (2, 3)


def test_inner_join_correct_keys():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="inner")
    assert sorted(m["key"].tolist()) == [2, 3]


def test_inner_join_correct_values():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="inner")
    assert sorted(m["val"].tolist()) == [20, 30]
    assert sorted(m["val2"].tolist()) == [200, 300]


def test_inner_join_no_matches_empty():
    left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"key": [3, 4], "val2": [30, 40]})
    m = left.merge(right, on="key", how="inner")
    assert m.shape[0] == 0


def test_inner_join_default_how():
    # how='inner' is default
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m_default = left.merge(right, on="key")
    m_inner = left.merge(right, on="key", how="inner")
    assert m_default.shape == m_inner.shape


# ---------------------------------------------------------------------------
# Left join
# ---------------------------------------------------------------------------

def test_left_join_basic():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="left")
    assert m.shape == (3, 3)


def test_left_join_preserves_left_keys():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="left")
    assert sorted(m["key"].tolist()) == [1, 2, 3]


def test_left_join_unmatched_rows_have_none():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="left")
    # key=1 has no match, val2 should be None
    val2_list = m["val2"].tolist()
    assert None in val2_list


# ---------------------------------------------------------------------------
# Right join
# ---------------------------------------------------------------------------

def test_right_join_basic():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="right")
    assert m.shape == (3, 3)


def test_right_join_preserves_right_keys():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="right")
    assert sorted(m["key"].tolist()) == [2, 3, 4]


def test_right_join_unmatched_rows_have_none():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="right")
    # key=4 has no match in left, val should be None
    val_list = m["val"].tolist()
    assert None in val_list


# ---------------------------------------------------------------------------
# Outer join
# ---------------------------------------------------------------------------

def test_outer_join_basic():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="outer")
    assert m.shape == (4, 3)


def test_outer_join_all_keys():
    left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"key": [2, 3, 4], "val2": [200, 300, 400]})
    m = left.merge(right, on="key", how="outer")
    assert sorted(m["key"].tolist()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Multi-column key
# ---------------------------------------------------------------------------

def test_merge_multi_column_key():
    left = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "val": [10, 20, 30]})
    right = pd.DataFrame({"a": [1, 2], "b": ["x", "x"], "val2": [100, 200]})
    m = left.merge(right, on=["a", "b"], how="inner")
    assert m.shape == (2, 4)


def test_merge_multi_column_key_values():
    left = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "val": [10, 20]})
    right = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "val2": [100, 200]})
    m = left.merge(right, on=["a", "b"], how="inner")
    assert sorted(m["val"].tolist()) == [10, 20]
    assert sorted(m["val2"].tolist()) == [100, 200]


# ---------------------------------------------------------------------------
# Overlapping non-key columns get suffixes
# ---------------------------------------------------------------------------

def test_merge_overlapping_columns_get_suffixes():
    left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"key": [1, 2], "val": [100, 200]})
    m = left.merge(right, on="key", how="inner")
    assert "val_x" in m.columns
    assert "val_y" in m.columns
    assert "val" not in m.columns


def test_merge_suffix_values_correct():
    left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"key": [1, 2], "val": [100, 200]})
    m = left.merge(right, on="key", how="inner")
    assert sorted(m["val_x"].tolist()) == [10, 20]
    assert sorted(m["val_y"].tolist()) == [100, 200]


# ---------------------------------------------------------------------------
# One-to-many join
# ---------------------------------------------------------------------------

def test_one_to_many_join():
    left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"key": [1, 1, 2], "val2": [100, 150, 200]})
    m = left.merge(right, on="key", how="inner")
    assert m.shape == (3, 3)


def test_one_to_many_join_values():
    left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"key": [1, 1, 2], "val2": [100, 150, 200]})
    m = left.merge(right, on="key", how="inner")
    # val for key=1 appears twice
    assert m["val"].tolist().count(10) == 2
