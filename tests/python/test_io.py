import pandas as pd
import os


def test_to_csv_string():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    result = df.to_csv()
    assert "a" in result
    assert "1" in result


def test_read_csv_roundtrip():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    csv_str = df.to_csv()
    tmp = "/tmp/test_pandas_rust.csv"
    with open(tmp, "w") as f:
        f.write(csv_str)
    df2 = pd.read_csv(tmp)
    assert df2.shape == df.shape
    os.remove(tmp)
