import _pandas_native as _native
from ..core.frame import DataFrame


def read_csv(filepath, **kwargs):
    raw = _native.read_csv(filepath)
    return DataFrame._from_native(raw)
