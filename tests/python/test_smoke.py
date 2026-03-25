"""Smoke test to verify the pandas-python binary works."""

def test_import_pandas_native():
    import _pandas_native
    assert _pandas_native._version() == "0.1.0"
