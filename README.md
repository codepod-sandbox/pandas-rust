# pandas-rust

A pandas implementation in Rust for Python code running in sandboxed environments (RustPython/WASM).

**1,745 tests passing (`2026-03-26`)**

## How it works

```
Python code (import pandas as pd)
        │
   pandas/ package (Python wrappers: DataFrame, Series, GroupBy)
        │
   _pandas_native (Rust ← RustPython bindings)
        │
   pandas-rust-core (columnar storage, groupby, merge, CSV, ops)
        │
   numpy-rust (interop via to_numpy/values)
```

Heavy-lifting operations (anything that iterates over data) run in native Rust. The Python layer handles API surface, method chaining, and pandas-compatible semantics.

## Test coverage

| Suite | Result |
|---|---|
| `cargo test -p pandas-rust-core` | 265 passed |
| Python integration tests | 72 passed |
| Pandas compatibility tests (upstream-style) | 1,408 passed, 1 skipped |

## Architecture

Three-crate Rust workspace:

| Crate | Purpose |
|---|---|
| `pandas-rust-core` | Pure Rust columnar engine — Column, DataFrame, Series, Index, all operations |
| `pandas-rust-python` | RustPython bindings — PyDataFrame, PySeries, PyGroupBy, PyIndex |
| `pandas-rust-wasm` | Standalone binary — registers numpy + pandas modules, runs RustPython |

### Data model

Columnar storage with explicit null bitmask:

```rust
enum ColumnData { Bool(Vec<bool>), Int64(Vec<i64>), Float64(Vec<f64>), Str(Vec<String>) }
struct Column { data: ColumnData, null_mask: Option<Vec<bool>>, name: String }
struct DataFrame { columns: IndexMap<String, Column>, index: Index }
struct Series { column: Column, index: Index }
```

### What runs in Rust

- Arithmetic, comparison, boolean ops with null propagation and type promotion
- Aggregation: sum, mean, min, max, count, std, var, median, quantile, prod
- GroupBy: hash-based grouping, 11 aggregation functions
- Merge/Join: hash join (inner, left, right, outer)
- Sort: stable multi-column sort with null handling
- CSV: reader (with dtype inference) and writer, wrapping the `csv` crate
- Filter: boolean mask application
- Null ops: isna, notna, fillna, dropna
- Unique: unique, nunique, value_counts, duplicated
- Math: abs, clip, cumsum, cummax, cummin, cumprod, diff, shift, rank, round
- Casting: all type conversions between Bool/Int64/Float64/Str
- Concat: row-wise and column-wise with dtype promotion
- Transpose: with type promotion
- numpy interop: to_numpy / values via numpy-rust-core

### What runs in Python

Thin wrappers providing the familiar pandas API:

- `DataFrame`: construction (dict, list-of-dicts, dict-of-Series), iloc/loc indexing, `__getitem__`/`__setitem__`, `__getattr__` column access, method chaining, `__repr__`, `query()`, `apply()`/`applymap()`
- `Series`: arithmetic/comparison operators, `.str` accessor (30+ methods), `rolling()`/`expanding()` windows, `map()`/`apply()`, `pct_change()`
- `GroupBy`: column selection (`df.groupby("k")["v"].sum()`), `transform()`, `filter()`, `apply()`, `agg()`
- Top-level: `pd.concat()`, `pd.merge()`, `pd.melt()`, `pd.read_csv()`, `pd.isna()`/`pd.notna()`, `pd.to_numeric()`

## Supported API

### DataFrame

| Category | Methods |
|---|---|
| Construction | `DataFrame(dict)`, `DataFrame([list of dicts])`, `DataFrame(dict of Series)` |
| Indexing | `df["col"]`, `df[["a","b"]]`, `df[bool_mask]`, `df.iloc[...]`, `df.loc[...]`, `df.col_name` |
| Mutation | `df["col"] = ...`, `df.iloc[r,c] = ...`, `df.loc[r,"col"] = ...`, `pop`, `update` |
| Info | `shape`, `dtypes`, `columns`, `index`, `size`, `ndim`, `empty`, `len()` |
| Display | `head`, `tail`, `describe`, `info`, `to_string` |
| Sorting | `sort_values` (single/multi-column, ascending list), `nlargest`, `nsmallest` |
| Selection | `drop`, `rename`, `filter`, `select_dtypes`, `reindex`, `set_axis`, `add_prefix`, `add_suffix` |
| Aggregation | `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `prod`, `corr`, `agg` |
| Null handling | `fillna`, `dropna`, `isna`, `notna`, `where`, `mask` |
| Dedup | `drop_duplicates`, `duplicated`, `value_counts` |
| GroupBy | `groupby` → `sum/mean/min/max/count/std/var/median/first/last/size`, `agg`, `transform`, `filter`, `apply` |
| Merge/Join | `merge` (inner/left/right/outer, on, left_on/right_on), `join` |
| Concat | `pd.concat` (axis=0/1) |
| Transform | `apply`, `applymap`/`map`, `assign`, `pipe`, `query`, `replace`, `astype` |
| Math | `abs`, `clip`, `diff`, `round`, `T`/`transpose` |
| I/O | `to_csv`, `to_dict` (dict/records), `to_numpy`, `values` |
| Iteration | `iterrows`, `itertuples`, `items`, `__iter__` |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=`, `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `equals` |
| Arithmetic | `+`, `-`, `*`, `/` (DataFrame ↔ DataFrame, DataFrame ↔ scalar) |
| Other | `copy`, `sample`, `reset_index`, `set_index`, `isin`, `melt`, `pivot_table` |

### Series

| Category | Methods |
|---|---|
| Construction | `Series(list)`, `Series(scalar, index=[...])` |
| Arithmetic | `+`, `-`, `*`, `/`, `-s` (negation), with type promotion |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=`, `&`, `\|`, `~` |
| Aggregation | `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `prod`, `any`, `all` |
| Cumulative | `cumsum`, `cummax`, `cummin`, `cumprod` |
| Transform | `apply`, `map` (dict or callable), `replace`, `astype`, `abs`, `clip`, `round`, `rank` |
| Null handling | `isna`, `notna`, `fillna`, `dropna`, `where`, `mask` |
| String | `.str.upper/lower/strip/contains/startswith/endswith/replace/split/len/find/count/match/pad/center/ljust/rjust/zfill/title/capitalize/get/slice/cat/repeat` + type checks |
| Window | `rolling(n).mean/sum/min/max/std/count`, `expanding().sum/mean/min/max` |
| Sorting | `sort_values`, `nlargest`, `nsmallest` |
| Dedup | `unique`, `nunique`, `value_counts`, `duplicated`, `is_unique` |
| Index | `idxmax`, `idxmin`, `argmax`, `argmin` |
| Time-series | `shift`, `diff`, `pct_change` |
| Other | `copy`, `to_frame`, `to_numpy`, `values`, `tolist`, `to_dict`, `isin`, `between`, `sample`, `squeeze`, `item`, `mode`, `rename`, `drop`, `head`, `tail` |
| Properties | `name`, `dtype`, `shape`, `size`, `ndim`, `empty`, `is_monotonic_increasing/decreasing` |

## Building

```bash
# Setup
git clone --recursive https://github.com/codepod-sandbox/pandas-rust.git
cd pandas-rust
make setup  # configure git hooks

# Build
cargo build -p pandas-rust-wasm

# Test
cargo test -p pandas-rust-core
target/debug/pandas-python -m pytest tests/python/ -v
target/debug/pandas-python -m pytest tests/pandas_compat/ -v
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
