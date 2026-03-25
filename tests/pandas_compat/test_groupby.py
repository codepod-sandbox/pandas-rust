"""
Pandas compatibility tests for GroupBy operations.
Covers: sum, mean, min, max, count, std, var, median, first, last, size.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Single key groupby + sum
# ---------------------------------------------------------------------------

def test_groupby_sum_single_key():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 40]})
    result = df.groupby("a").sum()
    assert isinstance(result, pd.DataFrame)


def test_groupby_sum_values():
    df = pd.DataFrame({"cat": ["x", "x", "y", "y"], "val": [1, 2, 3, 4]})
    result = df.groupby("cat").sum()
    # Find x group
    row_x = [r for r in result.to_dict()["val"] if r == 3]
    row_y = [r for r in result.to_dict()["val"] if r == 7]
    assert len(row_x) == 1
    assert len(row_y) == 1


def test_groupby_sum_shape():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 20, 30, 40, 50]})
    result = df.groupby("a").sum()
    assert result.shape[0] == 3  # 3 groups


# ---------------------------------------------------------------------------
# Single key groupby + mean
# ---------------------------------------------------------------------------

def test_groupby_mean():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10.0, 20.0, 30.0, 40.0]})
    result = df.groupby("a").mean()
    d = result.to_dict()
    vals = d["b"]
    assert 15.0 in vals
    assert 35.0 in vals


# ---------------------------------------------------------------------------
# Single key groupby + min / max
# ---------------------------------------------------------------------------

def test_groupby_min():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 40]})
    result = df.groupby("a").min()
    d = result.to_dict()
    vals = d["b"]
    assert 10 in vals
    assert 30 in vals


def test_groupby_max():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 40]})
    result = df.groupby("a").max()
    d = result.to_dict()
    vals = d["b"]
    assert 20 in vals
    assert 40 in vals


# ---------------------------------------------------------------------------
# Single key groupby + count
# ---------------------------------------------------------------------------

def test_groupby_count():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": [10, 20, 30, 40, 50]})
    result = df.groupby("a").count()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# Multi-key groupby
# ---------------------------------------------------------------------------

def test_groupby_multi_key_sum():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"], "c": [10, 20, 30, 40]})
    result = df.groupby(["a", "b"]).sum()
    assert result.shape[0] == 4  # 4 unique (a,b) combinations


def test_groupby_multi_key_count():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"], "c": [10, 20, 30, 40]})
    result = df.groupby(["a", "b"]).count()
    assert result.shape[0] == 2  # 2 unique (a,b) combinations


def test_groupby_multi_key_mean():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"], "c": [10.0, 20.0, 30.0, 40.0]})
    result = df.groupby(["a", "b"]).mean()
    assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# groupby preserves group order (first seen)
# ---------------------------------------------------------------------------

def test_groupby_group_order():
    # Groups should appear in first-seen order
    df = pd.DataFrame({"a": ["b", "a", "b", "a", "c"], "val": [1, 2, 3, 4, 5]})
    result = df.groupby("a").sum()
    key_col = result.to_dict()["a"]
    # b is seen first, then a, then c
    assert key_col == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# groupby.first() and .last()
# ---------------------------------------------------------------------------

def test_groupby_first():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [1, 2, 3, 4], "c": [10, 20, 30, 40]})
    result = df.groupby("a").first()
    assert isinstance(result, pd.DataFrame)
    d = result.to_dict()
    # first b for group x should be 1
    assert 1 in d["b"]
    # first b for group y should be 3
    assert 3 in d["b"]


def test_groupby_last():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [1, 2, 3, 4], "c": [10, 20, 30, 40]})
    result = df.groupby("a").last()
    assert isinstance(result, pd.DataFrame)
    d = result.to_dict()
    # last b for group x should be 2
    assert 2 in d["b"]
    # last b for group y should be 4
    assert 4 in d["b"]


# ---------------------------------------------------------------------------
# groupby.size()
# ---------------------------------------------------------------------------

def test_groupby_size_returns_dataframe():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": [10, 20, 30, 40, 50]})
    result = df.groupby("a").size()
    assert isinstance(result, pd.DataFrame)


def test_groupby_size_correct_shape():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": [10, 20, 30, 40, 50]})
    result = df.groupby("a").size()
    assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# groupby on string keys
# ---------------------------------------------------------------------------

def test_groupby_string_keys_sum():
    df = pd.DataFrame({"cat": ["apple", "banana", "apple", "banana"], "val": [1, 2, 3, 4]})
    result = df.groupby("cat").sum()
    d = result.to_dict()
    vals = sorted(d["val"])
    assert vals == [4, 6]


def test_groupby_string_keys_shape():
    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"], "val": [1, 2, 3, 4, 5]})
    result = df.groupby("cat").sum()
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# groupby on int keys
# ---------------------------------------------------------------------------

def test_groupby_int_keys_mean():
    df = pd.DataFrame({"k": [1, 2, 1, 2], "val": [10, 20, 30, 40]})
    result = df.groupby("k").mean()
    d = result.to_dict()
    vals = d["val"]
    assert 20.0 in vals
    assert 30.0 in vals


def test_groupby_int_keys_shape():
    df = pd.DataFrame({"k": [1, 2, 3, 1, 2], "val": [10, 20, 30, 40, 50]})
    result = df.groupby("k").sum()
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# groupby std / var / median
# ---------------------------------------------------------------------------

def test_groupby_std():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10.0, 20.0, 30.0, 40.0]})
    result = df.groupby("a").std()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2


def test_groupby_var():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10.0, 20.0, 30.0, 40.0]})
    result = df.groupby("a").var()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2


def test_groupby_median():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
    result = df.groupby("a").median()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
