"""Tests for string accessor methods: type checks, ops, nulls, workflows."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import pandas as our_pd


# ---------------------------------------------------------------------------
# String type checks
# ---------------------------------------------------------------------------

def test_str_isnumeric_true():
    s = our_pd.Series(["123", "456"])
    result = s.str.isnumeric()
    assert result.tolist() == [True, True]


def test_str_isnumeric_false():
    s = our_pd.Series(["abc", "12x"])
    result = s.str.isnumeric()
    assert result.tolist() == [False, False]


def test_str_isalpha_true():
    s = our_pd.Series(["abc", "XYZ"])
    result = s.str.isalpha()
    assert result.tolist() == [True, True]


def test_str_isalpha_false():
    s = our_pd.Series(["abc1", "123"])
    result = s.str.isalpha()
    assert result.tolist() == [False, False]


def test_str_isdigit_true():
    s = our_pd.Series(["123", "0"])
    result = s.str.isdigit()
    assert result.tolist() == [True, True]


def test_str_isdigit_false():
    s = our_pd.Series(["12a", "abc"])
    result = s.str.isdigit()
    assert result.tolist() == [False, False]


def test_str_isalnum_mixed():
    s = our_pd.Series(["abc1", "!!!"])
    result = s.str.isalnum()
    assert result.tolist() == [True, False]


def test_str_isupper_lower():
    s = our_pd.Series(["HELLO", "world"])
    assert s.str.isupper().tolist() == [True, False]
    assert s.str.islower().tolist() == [False, True]


def test_str_type_check_returns_bool_series():
    s = our_pd.Series(["a", "1"])
    result = s.str.isalpha()
    for v in result.tolist():
        assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# String operations (count, match, find, strip, title, zfill, get, slice)
# ---------------------------------------------------------------------------

def test_str_count_occurrences():
    s = our_pd.Series(["banana", "abcabc"])
    result = s.str.count("a")
    assert result.tolist() == [3, 2]


def test_str_match_regex():
    s = our_pd.Series(["hello world", "foo bar", "123"])
    result = s.str.match(r"\d+")
    assert result.tolist() == [False, False, True]


def test_str_find_position():
    s = our_pd.Series(["hello", "world"])
    result = s.str.find("l")
    assert result.tolist() == [2, 3]


def test_str_strip():
    s = our_pd.Series(["  hello  ", "  world  "])
    result = s.str.strip()
    assert result.tolist() == ["hello", "world"]


def test_str_lstrip():
    s = our_pd.Series(["  hello"])
    assert s.str.lstrip().tolist() == ["hello"]


def test_str_rstrip():
    s = our_pd.Series(["hello  "])
    assert s.str.rstrip().tolist() == ["hello"]


def test_str_title():
    s = our_pd.Series(["hello world"])
    assert s.str.title().tolist() == ["Hello World"]


def test_str_capitalize():
    s = our_pd.Series(["hello world"])
    assert s.str.capitalize().tolist() == ["Hello world"]


def test_str_zfill():
    s = our_pd.Series(["42", "7"])
    result = s.str.zfill(5)
    assert result.tolist() == ["00042", "00007"]


def test_str_get_first_char():
    s = our_pd.Series(["hello", "world"])
    result = s.str.get(0)
    assert result.tolist() == ["h", "w"]


def test_str_slice():
    s = our_pd.Series(["hello", "world"])
    result = s.str.slice(0, 3)
    assert result.tolist() == ["hel", "wor"]


# ---------------------------------------------------------------------------
# String with nulls / mixed types
# ---------------------------------------------------------------------------

def test_str_isnumeric_with_none():
    s = our_pd.Series(["123", None, "abc"])
    result = s.str.isnumeric()
    vals = result.tolist()
    assert vals[0] is True
    assert vals[1] is False  # None -> False
    assert vals[2] is False


def test_str_upper_with_none():
    s = our_pd.Series(["hello", None, "world"])
    result = s.str.upper()
    vals = result.tolist()
    assert vals[0] == "HELLO"
    assert vals[2] == "WORLD"


def test_str_contains_with_none():
    s = our_pd.Series(["hello", None, "world"])
    result = s.str.contains("llo")
    vals = result.tolist()
    assert vals[0] is True
    assert vals[2] is False


def test_str_len_with_none():
    s = our_pd.Series(["hello", None, "hi"])
    result = s.str.len()
    vals = result.tolist()
    assert vals[0] == 5
    assert vals[1] is None
    assert vals[2] == 2


# ---------------------------------------------------------------------------
# End-to-end string workflows
# ---------------------------------------------------------------------------

def test_str_filter_rows_contains():
    df = our_pd.DataFrame({"name": ["Alice", "Bob", "Albert"]})
    mask = df["name"].str.contains("Al")
    filtered = df[mask]
    assert len(filtered) == 2


def test_str_upper_then_startswith():
    s = our_pd.Series(["hello", "world", "hi"])
    upper = s.str.upper()
    starts = upper.str.startswith("H")
    assert starts.tolist() == [True, False, True]


def test_str_replace_then_len():
    s = our_pd.Series(["hello", "world"])
    replaced = s.str.replace("l", "")
    lengths = replaced.str.len()
    assert lengths.tolist() == [3, 4]


def test_str_split_result_type():
    s = our_pd.Series(["a b c", "x y"])
    result = s.str.split(" ")
    vals = result.tolist()
    assert isinstance(vals[0], list)
    assert vals[0] == ["a", "b", "c"]


def test_str_cat_joins():
    s = our_pd.Series(["hello", "world"])
    result = s.str.cat(sep=" ")
    assert result == "hello world"
