"""pandas-compatible Python package wrapping the Rust native module."""
import _pandas_native as _native

__version__ = "0.1.0"

from .core.frame import DataFrame
from .core.series import Series


def read_csv(path, **kwargs):
    """Read a CSV file into a DataFrame."""
    raw = _native.read_csv(path)
    return DataFrame._from_native(raw)
