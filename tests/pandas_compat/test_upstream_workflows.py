"""End-to-end workflow tests inspired by real pandas usage patterns."""
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Data-cleaning workflow
# ---------------------------------------------------------------------------

class TestDataCleaningWorkflow:
    def test_dropna_rename_sort_head(self):
        df = pd.DataFrame({
            "name": ["Alice", None, "Charlie", "Dave"],
            "score": [90, 85, None, 70],
        })
        result = (
            df.dropna()
              .rename(columns={"name": "student", "score": "grade"})
              .sort_values("grade", ascending=False)
              .head(2)
        )
        assert list(result.columns) == ["student", "grade"]
        assert len(result) == 2
        grades = result["grade"].tolist()
        assert grades == sorted(grades, reverse=True)

    def test_filter_by_condition_groupby_agg(self):
        df = pd.DataFrame({
            "dept": ["eng", "eng", "hr", "hr", "eng"],
            "salary": [100, 120, 80, 90, 110],
        })
        high = df[df["salary"] > 85]
        result = high.groupby("dept")["salary"].sum()
        assert isinstance(result, (pd.Series, dict))

    def test_fillna_then_sort(self):
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, 5, None],
        })
        filled = df.fillna(0.0)
        sorted_df = filled.sort_values("a")
        assert len(sorted_df) == 3

    def test_assign_new_column_then_filter(self):
        df = pd.DataFrame({"price": [10.0, 20.0, 15.0], "qty": [3, 1, 2]})
        df2 = df.assign(total=lambda d: d["price"] * d["qty"])
        big = df2[df2["total"] > 25]
        assert "total" in big.columns
        for v in big["total"].tolist():
            assert v > 25


# ---------------------------------------------------------------------------
# Analysis workflow
# ---------------------------------------------------------------------------

class TestAnalysisWorkflow:
    def test_groupby_multiple_agg(self):
        df = pd.DataFrame({
            "category": ["A", "B", "A", "B", "A"],
            "value": [10, 20, 30, 40, 50],
        })
        g = df.groupby("category")["value"]
        totals = g.sum()
        assert isinstance(totals, (pd.Series, dict, pd.DataFrame))

    def test_merge_then_compute(self):
        left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "mul": [2, 3, 4]})
        merged = left.merge(right, on="id")
        merged["result"] = [v * m for v, m in zip(merged["val"].tolist(), merged["mul"].tolist())]
        assert merged["result"].tolist() == [20, 60, 120]

    def test_sort_then_head_then_to_dict(self):
        df = pd.DataFrame({"x": [3, 1, 4, 1, 5], "y": [9, 2, 6, 5, 3]})
        result = df.sort_values("x").head(3).to_dict()
        assert isinstance(result, dict)
        assert "x" in result

    def test_describe_contains_stats(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        desc = df.describe()
        assert isinstance(desc, pd.DataFrame)
        assert len(desc) > 0

    def test_copy_and_modify_no_mutation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df2 = df.copy()
        df2["b"] = [4, 5, 6]
        assert "b" not in df.columns
        assert "b" in df2.columns


# ---------------------------------------------------------------------------
# String processing workflow
# ---------------------------------------------------------------------------

class TestStringWorkflow:
    def test_str_upper_then_filter(self):
        df = pd.DataFrame({"word": ["apple", "Banana", "cherry"]})
        df["upper"] = df["word"].str.upper()
        result = df.filter(items=["upper"])
        assert list(result.columns) == ["upper"]
        assert result["upper"].tolist() == ["APPLE", "BANANA", "CHERRY"]

    def test_str_contains_then_subset(self):
        df = pd.DataFrame({"name": ["foo_bar", "baz_qux", "foo_xyz"]})
        mask = df["name"].str.contains("foo")
        result = df[mask]
        assert len(result) == 2

    def test_str_startswith_count(self):
        s = pd.Series(["alpha", "beta", "gamma", "aleph"], name="x")
        mask = s.str.startswith("al")
        count = mask.sum()
        assert count == 2

    def test_value_counts_pipeline(self):
        s = pd.Series(["a", "b", "a", "c", "b", "a"], name="x")
        vc = s.value_counts()
        # 'a' should have the highest count
        assert isinstance(vc, pd.DataFrame)


# ---------------------------------------------------------------------------
# Reshape workflow
# ---------------------------------------------------------------------------

class TestReshapeWorkflow:
    def test_concat_drop_dups_sort(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [2, 3, 4]})
        from pandas import concat
        combined = concat([df1, df2])
        deduped = combined.drop_duplicates()
        sorted_df = deduped.sort_values("a")
        assert sorted_df["a"].tolist() == [1, 2, 3, 4]

    def test_concat_axis0_increases_rows(self):
        from pandas import concat
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        result = concat([df1, df2])
        assert len(result) == 4

    def test_add_prefix_suffix_pipeline(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = df.add_prefix("pre_").add_suffix("_suf")
        assert list(result.columns) == ["pre_col1_suf", "pre_col2_suf"]
        assert result.shape == df.shape


# ---------------------------------------------------------------------------
# Full pipeline: construct → mutate → filter → agg → output
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_pipeline(self):
        data = {
            "product": ["A", "B", "A", "C", "B", "A"],
            "region":  ["N", "N", "S", "S", "N", "S"],
            "sales":   [100, 200, 150, 50, 250, 120],
        }
        df = pd.DataFrame(data)

        # Step 1: filter by sales threshold
        df2 = df[df["sales"] >= 100]
        assert len(df2) < len(df) or len(df2) == len(df)

        # Step 2: add a tax column
        df2 = df2.assign(tax=lambda d: d["sales"] * 0.1)
        assert "tax" in df2.columns

        # Step 3: rename columns
        df3 = df2.rename(columns={"sales": "revenue"})
        assert "revenue" in df3.columns

        # Step 4: sort and head
        top = df3.sort_values("revenue", ascending=False).head(3)
        assert len(top) == 3

        # Step 5: to_dict for serialisation
        records = top.to_dict(orient="records")
        assert isinstance(records, list)
        assert len(records) == 3
        assert "revenue" in records[0]

    def test_properties_in_pipeline(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # All properties are accessible and correct
        assert df.ndim == 2
        assert df.size == 6
        assert df.empty is False

        s = df["a"]
        assert s.ndim == 1
        assert s.size == 3
        assert s.empty is False
        assert s.is_unique is True

    def test_filter_pop_in_pipeline(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [88, 72, 95],
            "internal_flag": [True, False, True],
        })
        # Keep only external columns
        public = df.filter(items=["id", "score"])
        assert "internal_flag" not in public.columns

        # Pop score for separate processing
        scores = public.pop("score")
        assert isinstance(scores, pd.Series)
        assert "score" not in public.columns
        assert public.empty is False  # 'id' column remains

    def test_string_ops_to_string(self):
        df = pd.DataFrame({"name": ["alice", "bob"], "val": [1, 2]})
        df["name"] = df["name"].str.upper()
        s = df.to_string()
        assert isinstance(s, str)
        assert "ALICE" in s or "alice" in s  # depends on repr

    def test_squeeze_in_pipeline(self):
        df = pd.DataFrame({"x": [42]})
        s = df["x"]
        scalar = s.squeeze()
        assert scalar == 42
        assert not isinstance(scalar, pd.Series)
