"""
Adapted from upstream pandas tests:
  - pandas/tests/frame/methods/test_assign.py
  - pandas/tests/frame/methods/test_clip.py
  - pandas/tests/frame/methods/test_copy.py
  - pandas/tests/frame/methods/test_diff.py
"""
import math
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# assign tests (adapted from test_assign.py)
# ---------------------------------------------------------------------------

class TestAssign:
    def test_assign_basic_column(self):
        """Basic assign with a Series value."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(C=df["B"] / df["A"])
        assert list(result.columns) == ["A", "B", "C"]
        c_vals = result["C"].tolist()
        assert c_vals[0] == 4.0
        assert abs(c_vals[1] - 2.5) < 1e-9
        assert abs(c_vals[2] - 2.0) < 1e-9

    def test_assign_lambda(self):
        """assign with a callable (lambda) that receives the DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(C=lambda x: x["B"] / x["A"])
        c_vals = result["C"].tolist()
        assert c_vals[0] == 4.0
        assert abs(c_vals[1] - 2.5) < 1e-9

    def test_assign_list(self):
        """assign with a plain integer list."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(C=[10, 20, 30])
        assert result["C"].tolist() == [10, 20, 30]

    def test_assign_does_not_modify_original(self):
        """Original DataFrame is not modified by assign."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        original_cols = list(df.columns)
        _ = df.assign(C=[10, 20, 30])
        assert list(df.columns) == original_cols

    def test_assign_overwrite_existing_column(self):
        """Assigning to an existing column name overwrites it."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(A=df["A"] + df["B"])
        assert result["A"].tolist() == [5, 7, 9]
        # B is unchanged
        assert result["B"].tolist() == [4, 5, 6]

    def test_assign_overwrite_with_lambda(self):
        """Overwrite existing column using lambda."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(A=lambda x: x["A"] + x["B"])
        assert result["A"].tolist() == [5, 7, 9]

    def test_assign_multiple(self):
        """Assign multiple columns at once."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(C=[7, 8, 9], D=df["A"], E=lambda x: x["B"])
        assert list(result.columns) == list("ABCDE")
        assert result["C"].tolist() == [7, 8, 9]
        assert result["D"].tolist() == [1, 2, 3]
        assert result["E"].tolist() == [4, 5, 6]

    def test_assign_order_preserved(self):
        """Column order follows assignment order (Python 3.7+ dicts are ordered)."""
        df = pd.DataFrame({"A": [1, 3], "B": [2, 4]})
        result = df.assign(D=df["A"] + df["B"], C=df["A"] - df["B"])
        cols = list(result.columns)
        # A and B appear first; D and C in assignment order
        assert cols.index("D") < cols.index("C")

    def test_assign_dependent_columns(self):
        """Lambda can reference a previously assigned column."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = df.assign(C=df["A"], D=lambda x: x["A"] + x["C"])
        assert result["C"].tolist() == [1, 2]
        assert result["D"].tolist() == [2, 4]

    def test_assign_dependent_both_lambdas(self):
        """Both new columns defined as lambdas, second uses first."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = df.assign(C=lambda df: df["A"], D=lambda df: df["A"] + df["C"])
        assert result["C"].tolist() == [1, 2]
        assert result["D"].tolist() == [2, 4]

    def test_assign_returns_new_dataframe(self):
        """assign always returns a new object."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = df.assign(B=[4, 5, 6])
        assert result is not df

    def test_assign_rename_via_overwrite(self):
        """Overwriting B with B/A effectively renames the computation."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.assign(B=df["B"] / df["A"])
        b_vals = result["B"].tolist()
        assert b_vals[0] == 4.0
        assert abs(b_vals[1] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# clip tests (adapted from test_clip.py)
# ---------------------------------------------------------------------------

class TestClip:
    def test_clip_upper(self):
        """clip(upper=x) caps values above x."""
        df = pd.DataFrame({"A": [1, 5, 10], "B": [2, 6, 11]})
        result = df.clip(upper=5)
        assert result["A"].tolist() == [1, 5, 5]
        assert result["B"].tolist() == [2, 5, 5]

    def test_clip_lower(self):
        """clip(lower=x) raises values below x."""
        df = pd.DataFrame({"A": [1, 5, 10], "B": [-3, 0, 4]})
        result = df.clip(lower=0)
        assert result["A"].tolist() == [1, 5, 10]
        assert result["B"].tolist() == [0, 0, 4]

    def test_clip_both_bounds(self):
        """clip(lower, upper) clamps values within [lower, upper]."""
        df = pd.DataFrame({"A": [1, 5, 10], "B": [-1, 3, 8]})
        result = df.clip(lower=2, upper=7)
        assert result["A"].tolist() == [2, 5, 7]
        assert result["B"].tolist() == [2, 3, 7]

    def test_clip_equal_bounds(self):
        """When lower == upper all values become that value."""
        df = pd.DataFrame({"A": [1, 5, 10]})
        median = 5.0
        result = df.clip(lower=median, upper=median)
        assert all(v == median for v in result["A"].tolist())

    def test_clip_does_not_modify_original(self):
        """clip returns a new DataFrame; original is unchanged."""
        df = pd.DataFrame({"A": [1, 5, 10]})
        original = df["A"].tolist()
        _ = df.clip(upper=3)
        assert df["A"].tolist() == original

    def test_clip_mixed_numeric(self):
        """clip on integer and float columns."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, None, 3.0]})
        result = df.clip(lower=1, upper=2)
        assert result["A"].tolist() == [1, 2, 2]
        b_clipped = result["B"].tolist()
        assert b_clipped[0] == 1.0
        # NaN propagates
        assert b_clipped[1] is None or (isinstance(b_clipped[1], float) and math.isnan(b_clipped[1]))
        assert b_clipped[2] == 2.0

    def test_clip_no_effect_when_all_in_range(self):
        """Values already within bounds are unchanged."""
        df = pd.DataFrame({"A": [3, 4, 5]})
        result = df.clip(lower=1, upper=10)
        assert result["A"].tolist() == [3, 4, 5]

    def test_clip_single_column(self):
        """clip on a single-column DataFrame."""
        df = pd.DataFrame({"x": [0, 5, 15]})
        result = df.clip(lower=2, upper=10)
        assert result["x"].tolist() == [2, 5, 10]


# ---------------------------------------------------------------------------
# copy tests (adapted from test_copy.py)
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_independence(self):
        """Modifying copy does not affect the original."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        cop = df.copy()
        cop["E"] = [10, 20, 30]
        assert "E" not in df.columns

    def test_copy_data_equality(self):
        """Copy has same data as original."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        cop = df.copy()
        assert cop["A"].tolist() == df["A"].tolist()
        assert cop["B"].tolist() == df["B"].tolist()

    def test_copy_returns_new_object(self):
        """copy() returns a distinct object."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        cop = df.copy()
        assert cop is not df

    def test_copy_columns_preserved(self):
        """Column names are preserved in the copy."""
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4], "Z": [5, 6]})
        cop = df.copy()
        assert list(cop.columns) == ["X", "Y", "Z"]

    def test_copy_shape_preserved(self):
        """Shape is preserved in the copy."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        cop = df.copy()
        assert cop.shape == df.shape


# ---------------------------------------------------------------------------
# diff tests (adapted from test_diff.py)
# ---------------------------------------------------------------------------

class TestDataFrameDiff:
    def test_diff_basic(self):
        """diff(1) computes row-by-row differences per column."""
        df = pd.DataFrame({"A": [1, 3, 6], "B": [10, 20, 35]})
        result = df.diff(1)
        # First row is NaN
        assert result["A"].tolist()[0] is None or math.isnan(result["A"].tolist()[0])
        assert result["A"].tolist()[1] == 2
        assert result["A"].tolist()[2] == 3
        assert result["B"].tolist()[1] == 10
        assert result["B"].tolist()[2] == 15

    def test_diff_negative_periods(self):
        """diff(-1) looks forward: first elements are non-null, last is NaN."""
        df = pd.DataFrame({"A": [1, 3, 6]})
        result = df.diff(-1)
        a_vals = result["A"].tolist()
        # First two values are non-null
        assert a_vals[0] is not None and not (isinstance(a_vals[0], float) and math.isnan(a_vals[0]))
        assert a_vals[1] is not None and not (isinstance(a_vals[1], float) and math.isnan(a_vals[1]))
        # Last row is NaN (no element to look forward to)
        assert a_vals[2] is None or (isinstance(a_vals[2], float) and math.isnan(a_vals[2]))

    def test_diff_int_dtype(self):
        """diff on integer column yields integer-like differences."""
        a = 10_000_000_000_000_000
        b = a + 1
        df = pd.DataFrame({"s": [a, b]})
        result = df.diff()
        assert result["s"].tolist()[1] == 1

    def test_diff_default_period_one(self):
        """diff() with no args defaults to periods=1."""
        df = pd.DataFrame({"A": [5, 10, 20]})
        result = df.diff()
        assert result["A"].tolist()[1] == 5
        assert result["A"].tolist()[2] == 10

    def test_diff_multiple_columns(self):
        """diff applied independently to each column."""
        df = pd.DataFrame({"A": [1, 2, 4], "B": [10, 13, 17]})
        result = df.diff()
        assert result["A"].tolist()[1] == 1
        assert result["A"].tolist()[2] == 2
        assert result["B"].tolist()[1] == 3
        assert result["B"].tolist()[2] == 4

    def test_diff_period_two(self):
        """diff(2) computes difference two rows apart."""
        df = pd.DataFrame({"A": [1, 2, 5, 9]})
        result = df.diff(2)
        a_vals = result["A"].tolist()
        assert a_vals[0] is None or math.isnan(a_vals[0])
        assert a_vals[1] is None or math.isnan(a_vals[1])
        assert a_vals[2] == 4  # 5 - 1
        assert a_vals[3] == 7  # 9 - 2
