"""
Pandas compatibility tests for CSV I/O.
Covers: to_csv, read_csv, roundtrip, dtypes, missing values.
"""
import pandas as pd
import tempfile
import os


def _make_csv_file(content):
    """Write content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

def test_to_csv_returns_string():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = df.to_csv()
    assert isinstance(result, str)


def test_to_csv_contains_column_headers():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    csv = df.to_csv()
    assert "col_a" in csv
    assert "col_b" in csv


def test_to_csv_contains_values():
    df = pd.DataFrame({"a": [10, 20, 30]})
    csv = df.to_csv()
    assert "10" in csv
    assert "20" in csv
    assert "30" in csv


def test_to_csv_string_values():
    df = pd.DataFrame({"s": ["hello", "world"]})
    csv = df.to_csv()
    assert "hello" in csv
    assert "world" in csv


def test_to_csv_nonempty():
    df = pd.DataFrame({"a": [1]})
    csv = df.to_csv()
    assert len(csv) > 0


def test_to_csv_write_to_file():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        assert os.path.exists(tmpf)
        with open(tmpf) as f:
            content = f.read()
        assert "a" in content
        assert "b" in content
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------

def test_read_csv_basic():
    path = _make_csv_file("a,b,c\n1,2.0,hello\n2,3.0,world\n")
    try:
        df = pd.read_csv(path)
        assert df.shape == (2, 3)
    finally:
        os.unlink(path)


def test_read_csv_preserves_column_names():
    path = _make_csv_file("col_x,col_y\n1,2\n3,4\n")
    try:
        df = pd.read_csv(path)
        assert "col_x" in df.columns
        assert "col_y" in df.columns
    finally:
        os.unlink(path)


def test_read_csv_column_order():
    path = _make_csv_file("z_col,a_col,m_col\n1,2,3\n")
    try:
        df = pd.read_csv(path)
        assert list(df.columns) == ["z_col", "a_col", "m_col"]
    finally:
        os.unlink(path)


def test_read_csv_int_dtype():
    path = _make_csv_file("a,b\n1,2\n3,4\n")
    try:
        df = pd.read_csv(path)
        assert df.dtypes["a"] == "int64"
    finally:
        os.unlink(path)


def test_read_csv_float_dtype():
    path = _make_csv_file("x,y\n1.5,2.5\n3.5,4.5\n")
    try:
        df = pd.read_csv(path)
        assert df.dtypes["x"] == "float64"
    finally:
        os.unlink(path)


def test_read_csv_string_dtype():
    path = _make_csv_file("name,age\nhello,25\nworld,30\n")
    try:
        df = pd.read_csv(path)
        assert df.dtypes["name"] == "object"
    finally:
        os.unlink(path)


def test_read_csv_mixed_dtypes():
    path = _make_csv_file("x,y,z\n1,2.0,hello\n2,3.0,world\n")
    try:
        df = pd.read_csv(path)
        assert df.dtypes["x"] == "int64"
        assert df.dtypes["y"] == "float64"
        assert df.dtypes["z"] == "object"
    finally:
        os.unlink(path)


def test_read_csv_correct_values():
    path = _make_csv_file("a,b\n10,20\n30,40\n")
    try:
        df = pd.read_csv(path)
        assert df["a"].tolist() == [10, 30]
        assert df["b"].tolist() == [20, 40]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Roundtrip: to_csv -> read_csv
# ---------------------------------------------------------------------------

def test_csv_roundtrip_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        df2 = pd.read_csv(tmpf)
        assert df2.shape == df.shape
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)


def test_csv_roundtrip_column_names():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        df2 = pd.read_csv(tmpf)
        assert list(df2.columns) == list(df.columns)
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)


def test_csv_roundtrip_int_values():
    df = pd.DataFrame({"a": [10, 20, 30]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        df2 = pd.read_csv(tmpf)
        assert df2["a"].tolist() == [10, 20, 30]
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)


def test_csv_roundtrip_float_values():
    df = pd.DataFrame({"x": [1.5, 2.5, 3.5]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        df2 = pd.read_csv(tmpf)
        result = df2["x"].tolist()
        for a, b in zip(result, [1.5, 2.5, 3.5]):
            assert abs(a - b) < 1e-10
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)


def test_csv_roundtrip_string_values():
    df = pd.DataFrame({"s": ["hello", "world", "foo"]})
    tmpf = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(tmpf)
        df2 = pd.read_csv(tmpf)
        assert df2["s"].tolist() == ["hello", "world", "foo"]
    finally:
        if os.path.exists(tmpf):
            os.unlink(tmpf)
