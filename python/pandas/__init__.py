"""pandas-compatible Python package wrapping the Rust native module."""
import _pandas_native as _native

__version__ = "0.1.0"

from .core.frame import DataFrame
from .core.series import Series


def read_csv(path, **kwargs):
    """Read a CSV file into a DataFrame."""
    raw = _native.read_csv(path)
    return DataFrame._from_native(raw)


def concat(objs, axis=0, **kwargs):
    """Concatenate DataFrames along an axis (0=rows, 1=cols)."""
    # Extract native DataFrames from Python wrappers
    native_dfs = []
    for obj in objs:
        if isinstance(obj, DataFrame):
            native_dfs.append(obj._native)
        else:
            raise TypeError("concat() expects a list of DataFrame objects")
    result = _native.concat(native_dfs, axis)
    return DataFrame._from_native(result)


def isna(obj):
    """Return True/False or element-wise boolean for NA/NaN/None values."""
    if obj is None:
        return True
    if isinstance(obj, float):
        return obj != obj  # NaN check
    if isinstance(obj, Series):
        return obj.isna()
    if isinstance(obj, DataFrame):
        return obj.isna()
    return False


def notna(obj):
    """Return True/False or element-wise boolean for non-NA values."""
    if isinstance(obj, (Series, DataFrame)):
        return obj.notna()
    return not isna(obj)


# Aliases
isnull = isna
notnull = notna


def to_numeric(arg, errors="raise"):
    """Convert argument to numeric type."""
    if isinstance(arg, Series):
        return arg.astype("float64")
    if isinstance(arg, list):
        return Series(arg).astype("float64")
    try:
        return float(arg)
    except (ValueError, TypeError):
        if errors == "raise":
            raise
        return float("nan")


def melt(frame, id_vars=None, value_vars=None, var_name="variable", value_name="value"):
    """Unpivot a DataFrame from wide to long format."""
    if id_vars is None:
        id_vars = []
    if isinstance(id_vars, str):
        id_vars = [id_vars]
    if value_vars is None:
        value_vars = [c for c in frame.columns if c not in id_vars]
    if isinstance(value_vars, str):
        value_vars = [value_vars]

    result = {v: [] for v in id_vars}
    result[var_name] = []
    result[value_name] = []

    n = len(frame)
    for val_col in value_vars:
        for id_col in id_vars:
            result[id_col].extend(frame[id_col].tolist())
        result[var_name].extend([val_col] * n)
        result[value_name].extend(frame[val_col].tolist())

    return DataFrame(result)
