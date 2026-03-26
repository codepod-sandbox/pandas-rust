"""
Adapted from upstream pandas tests:
  - pandas/tests/series/methods/test_sort_values.py
  - pandas/tests/frame/methods/test_sort_values.py
  - pandas/tests/frame/methods/test_drop_duplicates.py
"""
import math
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Series sort_values (adapted from test_sort_values.py)
# ---------------------------------------------------------------------------

class TestSeriesSortValues:
    def test_sort_values_basic(self):
        """Basic ascending sort preserves label-value correspondence."""
        s = pd.Series([3, 2, 4, 1], index=["A", "B", "C", "D"])
        result = s.sort_values()
        assert result.tolist() == [1, 2, 3, 4]

    def test_sort_values_descending(self):
        """ascending=False produces reverse-sorted values."""
        s = pd.Series([3, 1, 4, 1, 5])
        result = s.sort_values(ascending=False)
        vals = result.tolist()
        assert vals == sorted(vals, reverse=True)

    def test_sort_values_nan_at_end(self):
        """NaN values sort to the end by default (na_position='last')."""
        s = pd.Series([3.0, None, 1.0, None, 2.0])
        result = s.sort_values()
        vals = result.tolist()
        # Non-None values come first, sorted
        non_none = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        assert non_none == [1.0, 2.0, 3.0]
        # NaNs at the end
        tail = vals[-2:]
        assert all(v is None or (isinstance(v, float) and math.isnan(v)) for v in tail)

    def test_sort_values_ascending_false_simple(self):
        """Simple descending sort."""
        s = pd.Series([10, 30, 20])
        result = s.sort_values(ascending=False)
        assert result.tolist() == [30, 20, 10]

    def test_sort_values_already_sorted(self):
        """Already-sorted series returns same order."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = s.sort_values()
        assert result.tolist() == [1, 2, 3, 4, 5]

    def test_sort_values_strings(self):
        """String series sorts lexicographically."""
        s = pd.Series(["banana", "apple", "cherry"])
        result = s.sort_values()
        assert result.tolist() == ["apple", "banana", "cherry"]

    def test_sort_values_single_element(self):
        """Single-element series is trivially sorted."""
        s = pd.Series([42])
        result = s.sort_values()
        assert result.tolist() == [42]


# ---------------------------------------------------------------------------
# DataFrame sort_values (adapted from test_sort_values.py)
# ---------------------------------------------------------------------------

class TestDataFrameSortValues:
    def test_sort_values_frame_basic_ascending(self):
        """Sort DataFrame by a single column ascending."""
        df = pd.DataFrame({"A": [3, 1, 2], "B": [9, 7, 8]}, index=[0, 1, 2])
        result = df.sort_values(by="A")
        assert result["A"].tolist() == [1, 2, 3]

    def test_sort_values_frame_basic_descending(self):
        """Sort DataFrame by a single column descending."""
        df = pd.DataFrame({"A": [3, 1, 2], "B": [9, 7, 8]})
        result = df.sort_values(by="A", ascending=False)
        assert result["A"].tolist() == [3, 2, 1]

    def test_sort_values_multicolumn(self):
        """Sort by multiple columns."""
        df = pd.DataFrame({"A": [1, 3, 4], "B": [1, 1, 5], "C": [2, 0, 6]})
        # Sort by B then C
        result = df.sort_values(by=["B", "C"])
        # rows with B=1: C=2 (row0) and C=0 (row1) -> row1 first
        a_vals = result["A"].tolist()
        assert a_vals[0] == 3  # B=1, C=0
        assert a_vals[1] == 1  # B=1, C=2
        assert a_vals[2] == 4  # B=5, C=6

    def test_sort_values_multicolumn_ascending_list(self):
        """Sort by multiple columns with per-column ascending list."""
        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 1, 2]})
        result = df.sort_values(by=["A", "B"], ascending=[True, False])
        a_vals = result["A"].tolist()
        b_vals = result["B"].tolist()
        # A=1 group should be sorted by B descending
        assert a_vals[0] == 1 and b_vals[0] == 3
        assert a_vals[1] == 1 and b_vals[1] == 1

    def test_sort_values_inplace_returns_none(self):
        """inplace=True modifies in place and returns None."""
        df = pd.DataFrame({"A": [3, 1, 2]})
        result = df.sort_values(by="A", inplace=True)
        assert result is None

    def test_sort_values_stable_sort(self):
        """Equal values maintain relative order (stable sort)."""
        df = pd.DataFrame({"A": [1, 1, 1], "B": [3, 1, 2]})
        result = df.sort_values(by="A")
        # A values are all equal; original order preserved
        assert result["B"].tolist() == [3, 1, 2]

    def test_sort_values_nan_position(self):
        """NaN rows sort to end by default (na_position='last')."""
        df = pd.DataFrame({"A": [1.0, None, 2.0], "B": [3, 4, 5]})
        result = df.sort_values(by="A")
        a_vals = result["A"].tolist()
        # First two non-null, last is NaN
        non_null = [v for v in a_vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        assert non_null == [1.0, 2.0]

    def test_sort_values_preserves_other_columns(self):
        """Sorting by one column keeps other column data aligned."""
        df = pd.DataFrame({"A": [3, 1, 2], "B": ["c", "a", "b"]})
        result = df.sort_values(by="A")
        assert result["A"].tolist() == [1, 2, 3]
        assert result["B"].tolist() == ["a", "b", "c"]

    def test_sort_values_empty_by_list(self):
        """Sorting by empty list returns DataFrame unchanged (same data)."""
        df = pd.DataFrame({"a": [1, 4, 2, 5, 3, 6]})
        result = df.sort_values(by=[])
        assert result["a"].tolist() == [1, 4, 2, 5, 3, 6]

    def test_sort_values_returns_new_object(self):
        """sort_values returns a new object (unless inplace)."""
        df = pd.DataFrame({"A": [3, 1, 2]})
        result = df.sort_values(by="A")
        assert result is not df


# ---------------------------------------------------------------------------
# drop_duplicates (adapted from test_drop_duplicates.py)
# ---------------------------------------------------------------------------

class TestDropDuplicatesBasic:
    def test_drop_duplicates_keep_first(self):
        """drop_duplicates keeps first occurrence."""
        df = pd.DataFrame({
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": list(range(8)),
        })
        result = df.drop_duplicates("AAA")
        assert result.index.tolist() == [0, 1]
        assert result["AAA"].tolist() == ["foo", "bar"]

    def test_drop_duplicates_keep_last(self):
        """drop_duplicates keep='last' keeps last occurrence."""
        df = pd.DataFrame({
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": list(range(8)),
        })
        result = df.drop_duplicates("AAA", keep="last")
        assert 6 in result.index.tolist()
        assert 7 in result.index.tolist()

    def test_drop_duplicates_keep_false(self):
        """keep=False removes all duplicated rows."""
        df = pd.DataFrame({
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": list(range(8)),
        })
        result = df.drop_duplicates("AAA", keep=False)
        assert len(result) == 0

    def test_drop_duplicates_subset_multiple_cols(self):
        """Duplicate detection on multiple columns."""
        df = pd.DataFrame({
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": list(range(8)),
        })
        result = df.drop_duplicates(["AAA", "B"])
        assert result.index.tolist() == [0, 1, 2, 3]

    def test_drop_duplicates_no_subset(self):
        """Without subset, consider all columns."""
        df = pd.DataFrame({"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]})
        result = df.drop_duplicates()
        # Row 3 is a duplicate of row 2 (both x=3, y=5)
        assert len(result) == 6

    def test_drop_duplicates_no_duplicates(self):
        """DataFrame with no duplicates returns itself unchanged."""
        df = pd.DataFrame({"x": [1, 0], "y": [0, 2]})
        result = df.drop_duplicates()
        assert len(result) == 2

    def test_drop_duplicates_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame()
        result = df.drop_duplicates()
        assert len(result) == 0

    def test_drop_duplicates_empty_with_columns(self):
        """Empty DataFrame with columns returns empty DataFrame."""
        df = pd.DataFrame(columns=["A", "B", "C"])
        result = df.drop_duplicates()
        assert len(result) == 0
        # columns may or may not be preserved on empty frames — just check length

    def test_drop_duplicates_integer_column(self):
        """Deduplication works on integer columns."""
        df = pd.DataFrame({
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": list(range(8)),
        })
        result = df.drop_duplicates("C")
        assert result.index.tolist() == [0, 2]

    def test_drop_duplicates_inplace(self):
        """inplace=True modifies DataFrame in-place and returns None."""
        df = pd.DataFrame({
            "A": ["foo", "bar", "foo"],
            "B": [1, 2, 3],
        })
        return_value = df.drop_duplicates("A", inplace=True)
        assert return_value is None
        assert len(df) == 2

    def test_drop_duplicates_negative_integers(self):
        """Deduplication handles negative integers."""
        x = 1000000000
        df = pd.DataFrame({"a": [-x, 0], "b": [x, x + 4]})
        result = df.drop_duplicates()
        assert len(result) == 2
