"""Tests for extended string accessor methods."""
import pandas as pd
import pytest


class TestStrPadding:
    def setup_method(self):
        self.s = pd.Series(["hi", "hello", "hey"])

    def test_str_pad_left(self):
        result = self.s.str.pad(10, side="left")
        vals = result.tolist()
        assert vals[0] == "        hi"
        assert len(vals[0]) == 10

    def test_str_pad_right(self):
        result = self.s.str.pad(10, side="right")
        vals = result.tolist()
        assert vals[0] == "hi        "
        assert len(vals[0]) == 10

    def test_str_pad_both(self):
        result = self.s.str.pad(10, side="both")
        vals = result.tolist()
        assert len(vals[0]) == 10

    def test_str_center(self):
        s = pd.Series(["ab"])
        result = s.str.center(6)
        assert result.tolist()[0] == "  ab  "

    def test_str_ljust(self):
        s = pd.Series(["ab"])
        result = s.str.ljust(6)
        assert result.tolist()[0] == "ab    "

    def test_str_rjust(self):
        s = pd.Series(["ab"])
        result = s.str.rjust(6)
        assert result.tolist()[0] == "    ab"


class TestStrChecks:
    def test_str_isnumeric(self):
        s = pd.Series(["123", "abc", "12a"])
        result = s.str.isnumeric()
        assert result.tolist() == [True, False, False]

    def test_str_isalpha(self):
        s = pd.Series(["abc", "123", "abc1"])
        result = s.str.isalpha()
        assert result.tolist() == [True, False, False]

    def test_str_isdigit(self):
        s = pd.Series(["123", "abc", "12.3"])
        result = s.str.isdigit()
        assert result.tolist() == [True, False, False]

    def test_str_isalnum(self):
        s = pd.Series(["abc123", "abc!", "123"])
        result = s.str.isalnum()
        assert result.tolist() == [True, False, True]

    def test_str_isupper(self):
        s = pd.Series(["ABC", "abc", "Abc"])
        result = s.str.isupper()
        assert result.tolist() == [True, False, False]

    def test_str_islower(self):
        s = pd.Series(["abc", "ABC", "Abc"])
        result = s.str.islower()
        assert result.tolist() == [True, False, False]


class TestStrOperations:
    def test_str_count(self):
        s = pd.Series(["hello world", "foo bar baz", "aaa"])
        result = s.str.count("a")
        assert result.tolist() == [0, 2, 3]

    def test_str_match(self):
        s = pd.Series(["foo123", "bar", "123foo"])
        result = s.str.match(r"foo\d+")
        assert result.tolist() == [True, False, False]

    def test_str_repeat(self):
        s = pd.Series(["ab", "cd"])
        result = s.str.repeat(3)
        assert result.tolist() == ["ababab", "cdcdcd"]

    def test_str_find(self):
        s = pd.Series(["hello", "world"])
        result = s.str.find("l")
        assert result.tolist()[0] == 2
        assert result.tolist()[1] == 3

    def test_str_get(self):
        s = pd.Series(["hello", "world"])
        result = s.str.get(0)
        assert result.tolist() == ["h", "w"]

    def test_str_slice(self):
        s = pd.Series(["hello", "world"])
        result = s.str.slice(1, 4)
        assert result.tolist() == ["ell", "orl"]


class TestStrWorkflows:
    def test_filter_df_by_str_contains(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "score": [90, 80, 70]})
        result = df[df["name"].str.contains("li")]
        assert len(result) == 2  # Alice and Charlie

    def test_str_upper_then_startswith(self):
        s = pd.Series(["hello", "world", "help"])
        upper = s.str.upper()
        result = upper.str.startswith("H")
        assert result.tolist() == [True, False, True]

    def test_str_strip_then_len(self):
        s = pd.Series(["  hi  ", "  hello  "])
        stripped = s.str.strip()
        lengths = stripped.str.len()
        assert lengths.tolist() == [2, 5]

    def test_str_replace_chain(self):
        s = pd.Series(["hello world"])
        result = s.str.replace("hello", "bye").str.replace("world", "earth")
        assert result.tolist() == ["bye earth"]

    def test_str_none_handling(self):
        s = pd.Series(["hello", None, "world"])
        # Operations on None should not raise
        result = s.str.upper()
        vals = result.tolist()
        assert vals[0] == "HELLO"
        assert vals[2] == "WORLD"
        # None passes through
        assert vals[1] is None
