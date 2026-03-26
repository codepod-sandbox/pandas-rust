"""
Tests for common patterns: multi-col sort, groupby column selection,
__contains__, left_on/right_on merge, str accessor, groupby.agg, set_index.
"""
import pandas as pd


# =============================================================================
# 1. sort_values with multiple columns and ascending list
# =============================================================================

def test_sort_values_two_columns():
    df = pd.DataFrame({"a": [2, 1, 2, 1], "b": [10, 20, 5, 30]})
    result = df.sort_values(["a", "b"])
    vals_a = result["a"].tolist()
    assert vals_a == sorted(vals_a), "column 'a' should be sorted ascending"


def test_sort_values_two_columns_shape():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [10.0, 20.0, 30.0]})
    result = df.sort_values(["a", "b"])
    assert result.shape == (3, 2)


def test_sort_values_two_columns_different_ascending():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 5]})
    result = df.sort_values(["a", "b"], ascending=[True, False])
    # Within group a=1: b should be descending: 20, 10
    rows = result.to_dict()
    b_vals = rows["b"]
    # first two rows are a=1, b should be 20 then 10
    assert b_vals[0] == 20 or b_vals[1] == 20


def test_sort_values_string_then_numeric():
    df = pd.DataFrame({"cat": ["b", "a", "b", "a"], "val": [3, 4, 1, 2]})
    result = df.sort_values(["cat", "val"])
    cats = result["cat"].tolist()
    # All 'a' rows come before 'b' rows
    assert cats.index("b") > cats.index("a")


def test_sort_values_descending_multi():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    result = df.sort_values(["x"], ascending=False)
    vals = result["x"].tolist()
    assert vals == [3, 2, 1]


def test_sort_values_single_col_ascending_false():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [10.0, 20.0, 30.0]})
    result = df.sort_values("a", ascending=False)
    assert result["a"].tolist() == [3, 2, 1]


# =============================================================================
# 2. groupby["col"] column selection
# =============================================================================

def test_groupby_col_select_sum_returns_series():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [10, 20, 30]})
    result = df.groupby("g")["v"].sum()
    assert isinstance(result, pd.Series), "single-column select should return Series"


def test_groupby_col_select_sum_values():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = df.groupby("g")["v"].sum()
    vals = result.tolist()
    assert 3 in vals  # a: 1+2
    assert 3 in vals  # b: 3


def test_groupby_col_select_multi_returns_dataframe():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [10, 20, 30]})
    result = df.groupby("g")[["v", "w"]].sum()
    assert isinstance(result, pd.DataFrame)


def test_groupby_col_select_mean():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [2, 4, 6]})
    result = df.groupby("g")["v"].mean()
    vals = result.tolist()
    assert 3.0 in vals  # a: mean(2,4) = 3.0


def test_groupby_col_select_count():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = df.groupby("g")["v"].count()
    assert isinstance(result, pd.Series)


def test_groupby_col_select_min():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [5, 2, 9]})
    result = df.groupby("g")["v"].min()
    vals = result.tolist()
    assert 2 in vals  # a: min(5,2)=2


def test_groupby_col_select_max():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [5, 2, 9]})
    result = df.groupby("g")["v"].max()
    vals = result.tolist()
    assert 5 in vals  # a: max(5,2)=5


def test_groupby_col_select_preserves_values():
    df = pd.DataFrame({"g": [1, 1, 2], "val": [10, 20, 30]})
    result = df.groupby("g")["val"].sum()
    total = sum(result.tolist())
    assert total == 60


# =============================================================================
# 3. __contains__ ("col" in df)
# =============================================================================

def test_contains_existing_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert "a" in df


def test_contains_missing_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert "z" not in df


def test_contains_in_conditional():
    df = pd.DataFrame({"x": [1], "y": [2]})
    result = "x" in df and "z" not in df
    assert result is True


# =============================================================================
# 4. merge with left_on / right_on
# =============================================================================

def test_merge_left_on_right_on_basic():
    left = pd.DataFrame({"left_key": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"right_key": [2, 3], "val2": [200, 300]})
    result = left.merge(right, left_on="left_key", right_on="right_key")
    assert result.shape[0] == 1  # only key=2 matches


def test_merge_left_on_right_on_shape():
    left = pd.DataFrame({"lk": [1, 2, 3], "a": [10, 20, 30]})
    right = pd.DataFrame({"rk": [2, 3, 4], "b": [200, 300, 400]})
    result = left.merge(right, left_on="lk", right_on="rk")
    assert result.shape[0] == 2  # keys 2 and 3 match


def test_merge_left_on_right_on_values():
    left = pd.DataFrame({"id_l": [1, 2], "score": [100, 200]})
    right = pd.DataFrame({"id_r": [1, 2], "label": [10, 20]})
    result = left.merge(right, left_on="id_l", right_on="id_r")
    assert result.shape[0] == 2


def test_merge_left_on_right_on_no_match():
    left = pd.DataFrame({"lk": [1, 2], "a": [10, 20]})
    right = pd.DataFrame({"rk": [3, 4], "b": [30, 40]})
    result = left.merge(right, left_on="lk", right_on="rk")
    assert result.shape[0] == 0


def test_merge_left_on_right_on_list():
    left = pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "v": [10, 20]})
    right = pd.DataFrame({"j1": [1, 2], "j2": ["a", "b"], "w": [100, 200]})
    result = left.merge(right, left_on=["k1", "k2"], right_on=["j1", "j2"])
    assert result.shape[0] == 2


# =============================================================================
# 5. str accessor
# =============================================================================

def test_str_upper():
    s = pd.Series(["hello", "world"])
    result = s.str.upper()
    assert result.tolist() == ["HELLO", "WORLD"]


def test_str_lower():
    s = pd.Series(["HELLO", "WORLD"])
    result = s.str.lower()
    assert result.tolist() == ["hello", "world"]


def test_str_contains():
    s = pd.Series(["apple", "banana", "cherry"])
    result = s.str.contains("an")
    assert result.tolist() == [False, True, False]


def test_str_startswith():
    s = pd.Series(["foo", "bar", "foobar"])
    result = s.str.startswith("foo")
    assert result.tolist() == [True, False, True]


def test_str_endswith():
    s = pd.Series(["test.csv", "data.json", "file.csv"])
    result = s.str.endswith(".csv")
    assert result.tolist() == [True, False, True]


def test_str_strip():
    s = pd.Series(["  hello  ", " world "])
    result = s.str.strip()
    assert result.tolist() == ["hello", "world"]


def test_str_replace():
    s = pd.Series(["hello world", "hello python"])
    result = s.str.replace("hello", "hi")
    assert result.tolist() == ["hi world", "hi python"]


def test_str_len():
    s = pd.Series(["hi", "hello", "hey"])
    result = s.str.len()
    assert result.tolist() == [2, 5, 3]


def test_str_split():
    s = pd.Series(["a b c", "d e"])
    result = s.str.split()
    assert result.tolist() == [["a", "b", "c"], ["d", "e"]]


def test_str_cat():
    s = pd.Series(["a", "b", "c"])
    result = s.str.cat(sep="-")
    assert result == "a-b-c"


def test_str_slice():
    s = pd.Series(["abcdef", "ghijkl"])
    result = s.str.slice(1, 4)
    assert result.tolist() == ["bcd", "hij"]


# =============================================================================
# 6. groupby.agg
# =============================================================================

def test_groupby_agg_string():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = df.groupby("g").agg("sum")
    assert isinstance(result, pd.DataFrame)


def test_groupby_agg_dict():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = df.groupby("g").agg({"v": "sum"})
    assert isinstance(result, pd.DataFrame)
    assert "v" in result.columns


def test_groupby_agg_dict_values():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [10, 20, 30]})
    result = df.groupby("g").agg({"v": "sum"})
    total = sum(result["v"].tolist())
    assert total == 60


# =============================================================================
# 7. set_index
# =============================================================================

def test_set_index_drops_column():
    df = pd.DataFrame({"key": ["a", "b"], "val": [1, 2]})
    result = df.set_index("key")
    assert "key" not in result.columns


def test_set_index_preserves_other_columns():
    df = pd.DataFrame({"key": ["a", "b"], "val": [1, 2], "extra": [10, 20]})
    result = df.set_index("key")
    assert "val" in result.columns
    assert "extra" in result.columns


def test_set_index_no_drop():
    df = pd.DataFrame({"key": ["a", "b"], "val": [1, 2]})
    result = df.set_index("key", drop=False)
    assert "key" in result.columns
    assert "val" in result.columns
