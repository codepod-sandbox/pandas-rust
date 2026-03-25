# pandas-rust Design Spec

## Overview

A Rust-backed implementation of Python's pandas library for WASM environments (RustPython inside codepod). Follows the same architecture as numpy-rust, pillow-rust, and matplotlib-rust: heavy Rust core for data-intensive operations, thin Python API layer for the familiar pandas interface.

**Initial scope:** Core data structures (DataFrame, Series), indexing (loc/iloc), basic operations (arithmetic, comparison, aggregation, sort, filter), groupby, merge/join, and CSV I/O.

**License:** BSD 3-Clause (Codepod), same as numpy-rust.

## Repository Structure

```
pandas-rust/
├── Cargo.toml                          # Workspace root
├── LICENSE                             # BSD 3-Clause (Codepod)
├── Makefile                            # setup, build, test, clean
├── pyproject.toml                      # pytest config
├── .gitmodules                         # numpy-rust submodule
├── .githooks/
│   ├── pre-commit                      # fmt + clippy
│   └── pre-push                        # build + pytest
├── .github/workflows/ci.yml
│
├── packages/
│   └── numpy-rust/                     # Git submodule
│
├── crates/
│   ├── pandas-rust-core/               # Pure Rust columnar engine
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── column.rs               # Typed column storage with null bitmask
│   │       ├── dataframe.rs            # DataFrame (ordered map of named columns)
│   │       ├── series.rs               # Single named column
│   │       ├── dtype.rs                # DType enum
│   │       ├── index.rs                # RangeIndex, Int64Index, StringIndex
│   │       ├── error.rs                # PandasError
│   │       ├── ops/
│   │       │   ├── mod.rs
│   │       │   ├── arithmetic.rs       # Column-level +, -, *, /
│   │       │   ├── comparison.rs       # ==, !=, <, >, boolean masks
│   │       │   ├── aggregation.rs      # sum, mean, min, max, count, std, var, median
│   │       │   ├── filter.rs           # Boolean mask filtering
│   │       │   ├── sort.rs             # Sort by values/index
│   │       │   └── nulls.rs            # fillna, dropna, isna, notna
│   │       ├── concat.rs               # Row-wise and column-wise concatenation
│   │       ├── groupby.rs              # Hash-based groupby engine
│   │       ├── merge.rs                # Hash join / sort-merge join
│   │       ├── io/
│   │       │   ├── mod.rs
│   │       │   └── csv.rs              # CSV reader/writer (wrapping `csv` crate)
│   │       └── casting.rs              # Dtype coercion
│   │
│   ├── pandas-rust-python/             # RustPython bindings
│   │   └── src/
│   │       ├── lib.rs                  # pandas_module_def, #[pymodule] _pandas_native
│   │       ├── py_column.rs            # PyColumn (internal)
│   │       ├── py_dataframe.rs         # PyDataFrame class
│   │       ├── py_series.rs            # PySeries class
│   │       ├── py_groupby.rs           # PyGroupBy class
│   │       ├── py_index.rs             # PyIndex classes
│   │       └── py_io.rs                # read_csv, to_csv wrappers
│   │
│   └── pandas-rust-wasm/               # Standalone binary
│       └── src/
│           └── main.rs                 # Registers numpy + pandas native modules
│
├── python/
│   └── pandas/
│       ├── __init__.py                 # Imports from _pandas_native, re-exports API
│       ├── core/
│       │   ├── __init__.py
│       │   ├── frame.py               # DataFrame Python API
│       │   ├── series.py              # Series Python API
│       │   ├── indexes.py             # Index classes
│       │   ├── groupby.py             # GroupBy Python API
│       │   └── dtypes.py              # Dtype helpers
│       ├── io/
│       │   ├── __init__.py
│       │   └── parsers.py             # read_csv() top-level function
│       └── _testing.py                # Test utilities (assert_frame_equal, etc.)
│
└── tests/
    ├── python/
    │   ├── run_tests.sh
    │   ├── test_dataframe.py
    │   ├── test_series.py
    │   ├── test_indexing.py
    │   ├── test_groupby.py
    │   ├── test_merge.py
    │   └── test_io.py
    └── pandas_compat/
        └── run_compat.py
```

## Rust Core: Column Storage & DTypes

### Data Model

```rust
// dtype.rs
pub enum DType {
    Bool,
    Int64,
    Float64,
    Str,
}

// column.rs
pub enum ColumnData {
    Bool(Vec<bool>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Str(Vec<String>),
}

pub struct Column {
    data: ColumnData,
    null_mask: Option<Vec<bool>>,  // true = null at that position
    name: String,
}
```

### Design Decisions

- **Explicit nulls via bitmask** — consistent null handling across all dtypes. The Python layer maps to pandas' `None`/`NaN` semantics.
- **`Vec<String>` for string columns** — simple, owned. No interning for v1.
- **No generic `T` parameter** — `ColumnData` enum dispatches at runtime, matching pandas' mixed-dtype DataFrames. Keeps the binding layer simple.

### DataFrame & Index

```rust
// dataframe.rs
use indexmap::IndexMap;

pub struct DataFrame {
    columns: IndexMap<String, Column>,  // preserves insertion order, O(1) lookup by name
    index: Index,
}

// index.rs
pub enum Index {
    Range(RangeIndex),
    Int64(Vec<i64>),
    Str(Vec<String>),
}

pub struct RangeIndex {
    start: i64,
    stop: i64,
    step: i64,
}

// series.rs
pub struct Series {
    column: Column,
    index: Index,
}
```

## Rust Core: Operations

All operations that iterate over data live in Rust.

### Arithmetic & Comparison (`ops/arithmetic.rs`, `ops/comparison.rs`)
- Column + Column, Column + scalar for all numeric types
- Null propagation: any op with null -> null
- Type promotion: Int64 + Float64 -> Float64
- Boolean mask generation: `col > 5` -> `Vec<bool>`

### Aggregation (`ops/aggregation.rs`)
- `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `quantile`
- All null-aware (skip nulls by default)
- Return scalar results
- `quantile(q)` uses linear interpolation between adjacent values

### Filtering (`ops/filter.rs`)
- Apply boolean mask to Column -> new Column
- Apply boolean mask to DataFrame -> new DataFrame

### Sorting (`ops/sort.rs`)
- Stable sort by values
- Sort DataFrame by one or more columns, ascending/descending per column
- `argsort` -> index permutation applied to all columns

### GroupBy (`groupby.rs`)
- Hash-based grouping: `HashMap<GroupKey, Vec<usize>>`
- GroupKey supports single column or tuple of columns
- Per-group aggregation: `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `first`, `last`, `size`

### Merge (`merge.rs`)
- Hash join (default): hash table on smaller side, probe with larger
- Join types: `inner`, `left`, `right`, `outer`
- Join on one or more key columns
- Suffix handling for overlapping names (`_x`, `_y`)

### CSV I/O (`io/csv.rs`)
- Wraps the `csv` crate for parsing
- Two-pass: first pass infers dtypes (sample rows), second reads with known types
- Configurable: delimiter, header row, column selection, dtype overrides, na_values
- Writer: DataFrame -> CSV string/file

### Null Operations (`ops/nulls.rs`)
- `isna(column)` -> Bool column (true where null)
- `notna(column)` -> Bool column (true where not null)
- `fillna(column, value)` -> new Column with nulls replaced by scalar value. No forward-fill/back-fill for v1.
- `dropna(dataframe, axis, how)` -> new DataFrame with null-containing rows/columns removed. `how="any"` (default) drops if any null; `how="all"` drops only if all null.

### Concatenation (`concat.rs`)
- `concat_rows(frames)` -> new DataFrame. Aligns columns by name; missing columns filled with nulls. Dtypes promoted if needed (e.g., Int64 + Float64 -> Float64).
- `concat_cols(frames)` -> new DataFrame. Appends columns side-by-side. Requires matching row counts (error otherwise). Duplicate column names get suffix.

### Casting (`casting.rs`)
- Int64 <-> Float64, numeric -> Str, Str -> numeric
- Bool <-> Int64/Float64

## Python Binding Layer

`pandas-rust-python` exposes RustPython `#[pyclass]` types.

### Native Module (`_pandas_native`)
- `PyDataFrame` — holds `pandas_rust_core::DataFrame`
- `PySeries` — holds `pandas_rust_core::Series`
- `PyGroupBy` — holds grouped row indices + source DataFrame reference
- `PyIndex` — wraps `Index` enum
- `read_csv(path, **kwargs)` -> `PyDataFrame`
- `concat(frames, axis)` -> `PyDataFrame`

### Python API Surface

```python
class DataFrame:
    # Construction
    def __init__(self, data=None, columns=None, index=None): ...

    # Indexing
    @property
    def loc(self): ...        # Label-based
    @property
    def iloc(self): ...       # Integer-based
    def __getitem__(self, key): ...
    def __setitem__(self, key, val): ...

    # Info
    shape, dtypes, columns, index, values (-> numpy ndarray)
    head(), tail(), describe(), info(), __repr__(), __len__()

    # Operations (delegate to Rust)
    sort_values(), drop(), rename(), astype(), copy()
    fillna(), dropna(), isna(), notna()

    # Aggregation
    sum(), mean(), min(), max(), count(), std(), var(), median()

    # GroupBy & Merge
    groupby(by) -> GroupBy
    merge(right, on, how, suffixes)

    # I/O
    to_csv(), to_dict(), to_numpy()
```

`Series` has a similar surface with arithmetic/comparison operators and aggregations.

`loc`/`iloc` are Python classes implementing `__getitem__`/`__setitem__`, translating label/integer slicing into Rust-side operations.

## numpy-rust Interop

### Dependency Structure

```
pandas-rust-core  (no numpy dependency, pure columnar engine)
       |
pandas-rust-python  -> numpy-rust-python (for PyNdArray interop)
       |                    |
pandas-rust-wasm   -> numpy-rust-python (registers both modules)
```

`pandas-rust-core` has zero numpy dependency. Interop happens in the Python binding layer:

- **`to_numpy()`** — copies column data into numpy-rust-core `NdArray`, wraps as `PyNdArray`. Int64 columns with nulls promote to Float64 (NaN for nulls).
- **Construction from ndarrays** — Python layer detects ndarray inputs and extracts data into columns.

### WASM Binary

Registers both numpy and pandas native modules. Sets PYTHONPATH to include both `python/` and `packages/numpy-rust/python/`.

## Testing Strategy

Upstream pandas tests are the primary guide.

### Three Tiers

1. **Rust unit tests** — `#[cfg(test)]` modules in pandas-rust-core. Column ops, null handling, dtype coercion, groupby, merge, CSV parsing. Run via `cargo test`.

2. **Python integration tests** (`tests/python/`) — pytest-style tests for end-to-end Python API. DataFrame/Series construction, indexing, operators, aggregations, groupby, merge, CSV I/O, repr, numpy interop.

3. **Upstream pandas compat** (`tests/pandas_compat/`) — vendored test files from real pandas (BSD-3 licensed). Adapted with xfail/skip for unsupported features. Initial set:
   - `test_frame_basic.py`
   - `test_frame_indexing.py`
   - `test_series_basic.py`
   - `test_groupby_basic.py`
   - `test_merge.py`
   - `test_io_csv.py`

### CI Pipeline

```yaml
jobs:
  test:          # cargo test -p pandas-rust-core, cargo test --workspace
  lint:          # cargo fmt --check, cargo clippy
  python-tests:  # build binary, run vendored + compat tests
  wasm-build:    # cargo build -p pandas-rust-wasm --target wasm32-wasip1
```

## Dependencies

### pandas-rust-core
- `thiserror` — error types
- `csv` — CSV parsing
- `indexmap` — ordered map for column storage (preserves insertion order with O(1) lookup)

### pandas-rust-python
- `pandas-rust-core`
- `numpy-rust-core` (for NdArray interop)
- `numpy-rust-python` (for PyNdArray interop)
- `rustpython-vm`, `rustpython-derive`

### pandas-rust-wasm
- `pandas-rust-python`
- `numpy-rust-python`
- `rustpython` (with freeze-stdlib, threading, host_env) — same feature set as numpy-rust-wasm

### Workspace
- RustPython pinned to same rev as numpy-rust (`f9ca63893`)

## Mutation Semantics

DataFrames and Series are **always-copy** for v1. Every operation that transforms data returns a new DataFrame/Series; the original is never mutated. This is simple, safe, and matches pre-2.0 pandas behavior.

The one exception is `__setitem__` (`df["col"] = values`), which mutates in place — it replaces or adds a column on the existing DataFrame. This matches pandas behavior and avoids the awkwardness of requiring `df = df.assign(col=values)` for basic column assignment.

`copy()` produces a deep copy (all column data cloned).

## Error Mapping

Rust `PandasError` variants map to Python exceptions following the same pattern as numpy-rust:

| Rust Error | Python Exception | When |
|---|---|---|
| `KeyError(name)` | `KeyError` | Column name not found |
| `IndexError(msg)` | `IndexError` | Row index out of bounds |
| `TypeError(msg)` | `TypeError` | Incompatible dtypes for operation |
| `ValueError(msg)` | `ValueError` | Invalid argument (e.g., bad axis, shape mismatch) |
| `IoError(msg)` | `OSError` | File not found, permission denied |
| `ParseError(msg)` | `ValueError` | CSV parse failure, dtype conversion failure |

The Python binding layer converts `Result<T, PandasError>` to `PyResult<T>` using `vm.new_key_error(...)`, `vm.new_type_error(...)`, etc.

## `values` Property (Mixed-Dtype DataFrames)

`DataFrame.values` returns a 2D numpy ndarray. For homogeneous numeric DataFrames, the dtype matches the column dtype. For mixed-dtype DataFrames (e.g., Int64 + Str), all columns are converted to `Str` (object-like semantics). This is a simplification of real pandas' object array behavior, but adequate for v1.

## `head()` / `tail()` / `describe()`

`head(n)` and `tail(n)` are implemented in pure Python as thin wrappers around `iloc[:n]` and `iloc[-n:]`. No Rust-side implementation needed.

`describe()` delegates to the Rust aggregation functions (count, mean, std, min, max) plus `quantile` for 25th/50th/75th percentiles. Quantile computation (linear interpolation) is added to `ops/aggregation.rs`. For non-numeric columns (Str, Bool), `describe()` excludes them from the output — only numeric columns get statistics. String-column describe (count/unique/top/freq) is out of scope for v1.

## WASM Binary: Module Registration

The `pandas-rust-wasm` binary registers both numpy and pandas native modules, following the same pattern as matplotlib-rust (which registers both numpy and pillow). This is for **standalone testing only** — the codepod orchestrator has its own module registration that composes all available packages. Each `-wasm` binary is self-contained so it can run tests independently without the full codepod stack.

The binary is built and run natively for tests (`cargo build -p pandas-rust-wasm`) and also verified to compile for `wasm32-wasip1` in CI. The `freeze-stdlib, threading, host_env` feature set works for both native and WASM targets, matching numpy-rust-wasm. The `main.rs` prepends both `python/` and `packages/numpy-rust/python/` to PYTHONPATH, following the matplotlib-rust pattern.

## Out of Scope (v1)

- Datetime dtype and `.dt` accessor
- Categorical dtype
- String accessor (`.str`)
- MultiIndex
- Window operations (`rolling`, `expanding`)
- `pivot_table`, `melt`, `stack`/`unstack`
- JSON, Excel, Parquet I/O
- `apply` with arbitrary Python functions (requires RustPython callback overhead)
- String interning for groupby optimization
