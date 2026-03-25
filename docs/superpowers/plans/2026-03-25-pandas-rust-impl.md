# pandas-rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust-backed pandas implementation for WASM/RustPython environments with columnar storage, basic operations, groupby, merge, and CSV I/O.

**Architecture:** Three-crate workspace (pandas-rust-core, pandas-rust-python, pandas-rust-wasm) with numpy-rust as a git submodule. Heavy Rust core for data-intensive operations, thin Python API layer. IndexMap-based columnar storage with explicit null bitmask.

**Tech Stack:** Rust, RustPython, indexmap, csv crate, thiserror. Python test layer via pytest.

**Spec:** `docs/superpowers/specs/2026-03-25-pandas-rust-design.md`

**Reference projects:** numpy-rust (`/Users/sunny/work/codepod/numpy-rust`), matplotlib-rust (`/Users/sunny/work/codepod/matplotlib-rust`)

---

## Phase 1: Repository Scaffold + Core Data Model

### Task 1: Initialize Git Repository and Workspace

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `LICENSE`
- Create: `Makefile`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.githooks/pre-commit`
- Create: `.githooks/pre-push`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/sunny/work/codepod/pandas-rust
git init
```

- [ ] **Step 2: Create workspace Cargo.toml**

```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.dependencies]
rustpython = { git = "https://github.com/RustPython/RustPython", rev = "f9ca63893", default-features = false }
rustpython-vm = { git = "https://github.com/RustPython/RustPython", rev = "f9ca63893" }
rustpython-derive = { git = "https://github.com/RustPython/RustPython", rev = "f9ca63893" }
```

Note: The RustPython rev MUST match numpy-rust's Cargo.toml (`f9ca63893`). Verify by reading `/Users/sunny/work/codepod/numpy-rust/Cargo.toml`.

- [ ] **Step 3: Create LICENSE (BSD 3-Clause)**

Copy from `/Users/sunny/work/codepod/numpy-rust/LICENSE` — same text, same copyright holder (Codepod).

- [ ] **Step 4: Create .gitignore**

```
target/
.DS_Store
```

Note: Do NOT gitignore `Cargo.lock` — sibling projects (numpy-rust, matplotlib-rust) track it in git. Use `target/` (no leading slash) to match sibling convention.

- [ ] **Step 5: Create Makefile**

```makefile
.PHONY: setup build test clean

setup:
	git config core.hooksPath .githooks

build:
	cargo build -p pandas-rust-wasm

test: build
	cargo test -p pandas-rust-core
	target/debug/pandas-python -m pytest tests/python/

clean:
	cargo clean
```

- [ ] **Step 6: Create pyproject.toml**

```toml
[tool.pytest.ini_options]
testpaths = ["tests/python"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

- [ ] **Step 7: Create .githooks/pre-commit**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "pre-commit: cargo fmt --check"
cargo fmt --all -- --check

echo "pre-commit: cargo clippy"
cargo clippy --workspace -- -D warnings
```

Make executable: `chmod +x .githooks/pre-commit`

- [ ] **Step 8: Create .githooks/pre-push**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "pre-push: building"
cargo build -p pandas-rust-wasm

echo "pre-push: running python tests"
target/debug/pandas-python -m pytest tests/python/
```

Make executable: `chmod +x .githooks/pre-push`

- [ ] **Step 9: Add numpy-rust as git submodule**

```bash
cd /Users/sunny/work/codepod/pandas-rust
git submodule add https://github.com/codepod-sandbox/numpy-rust.git packages/numpy-rust
```

Verify the URL by reading `/Users/sunny/work/codepod/matplotlib-rust/.gitmodules` (line 3).

- [ ] **Step 10: Commit scaffold**

```bash
git add -A
git commit -m "chore: initialize pandas-rust workspace scaffold"
```

---

### Task 2: Create pandas-rust-core Crate — DType and Error

**Files:**
- Create: `crates/pandas-rust-core/Cargo.toml`
- Create: `crates/pandas-rust-core/src/lib.rs`
- Create: `crates/pandas-rust-core/src/dtype.rs`
- Create: `crates/pandas-rust-core/src/error.rs`

- [ ] **Step 1: Create Cargo.toml for pandas-rust-core**

```toml
[package]
name = "pandas-rust-core"
version = "0.1.0"
edition = "2021"

[dependencies]
indexmap = "2"
thiserror = "2"
csv = "1"
```

- [ ] **Step 2: Create src/dtype.rs with tests**

```rust
use std::fmt;

/// Data types supported by pandas-rust columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int64,
    Float64,
    Str,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int64 => write!(f, "int64"),
            DType::Float64 => write!(f, "float64"),
            DType::Str => write!(f, "object"),
        }
    }
}

impl DType {
    /// Parse a dtype string (e.g., "int64", "float64", "bool", "object"/"str").
    pub fn from_str(s: &str) -> Option<DType> {
        match s {
            "bool" => Some(DType::Bool),
            "int64" | "int" | "i64" => Some(DType::Int64),
            "float64" | "float" | "f64" => Some(DType::Float64),
            "object" | "str" | "string" => Some(DType::Str),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::Bool.to_string(), "bool");
        assert_eq!(DType::Int64.to_string(), "int64");
        assert_eq!(DType::Float64.to_string(), "float64");
        assert_eq!(DType::Str.to_string(), "object");
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("int64"), Some(DType::Int64));
        assert_eq!(DType::from_str("float"), Some(DType::Float64));
        assert_eq!(DType::from_str("object"), Some(DType::Str));
        assert_eq!(DType::from_str("str"), Some(DType::Str));
        assert_eq!(DType::from_str("unknown"), None);
    }
}
```

- [ ] **Step 3: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PandasError {
    #[error("KeyError: {0}")]
    KeyError(String),

    #[error("IndexError: {0}")]
    IndexError(String),

    #[error("TypeError: {0}")]
    TypeError(String),

    #[error("ValueError: {0}")]
    ValueError(String),

    #[error("IoError: {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParseError: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, PandasError>;
```

- [ ] **Step 4: Create src/lib.rs**

```rust
pub mod dtype;
pub mod error;

pub use dtype::DType;
pub use error::{PandasError, Result};
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pandas-rust-core
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/pandas-rust-core/
git commit -m "feat: add pandas-rust-core with DType and PandasError"
```

---

### Task 3: Column Type with Null Bitmask

**Files:**
- Create: `crates/pandas-rust-core/src/column.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write column.rs with core struct and constructors**

```rust
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Typed column data storage.
#[derive(Debug, Clone)]
pub enum ColumnData {
    Bool(Vec<bool>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Str(Vec<String>),
}

/// A named, typed column with optional null bitmask.
#[derive(Debug, Clone)]
pub struct Column {
    pub(crate) data: ColumnData,
    pub(crate) null_mask: Option<Vec<bool>>, // true = null at that position
    pub(crate) name: String,
}

impl ColumnData {
    pub fn dtype(&self) -> DType {
        match self {
            ColumnData::Bool(_) => DType::Bool,
            ColumnData::Int64(_) => DType::Int64,
            ColumnData::Float64(_) => DType::Float64,
            ColumnData::Str(_) => DType::Str,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnData::Bool(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Column {
    /// Create a new column without nulls.
    pub fn new(name: impl Into<String>, data: ColumnData) -> Self {
        Column {
            data,
            null_mask: None,
            name: name.into(),
        }
    }

    /// Create a new column with a null bitmask.
    pub fn new_with_nulls(
        name: impl Into<String>,
        data: ColumnData,
        null_mask: Vec<bool>,
    ) -> Result<Self> {
        if null_mask.len() != data.len() {
            return Err(PandasError::ValueError(format!(
                "null_mask length ({}) != data length ({})",
                null_mask.len(),
                data.len()
            )));
        }
        let mask = if null_mask.iter().any(|&b| b) {
            Some(null_mask)
        } else {
            None // no nulls, don't store the mask
        };
        Ok(Column {
            data,
            null_mask: mask,
            name: name.into(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn has_nulls(&self) -> bool {
        self.null_mask.is_some()
    }

    pub fn is_null(&self, idx: usize) -> bool {
        self.null_mask
            .as_ref()
            .map_or(false, |m| m.get(idx).copied().unwrap_or(false))
    }

    pub fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map_or(0, |m| m.iter().filter(|&&b| b).count())
    }

    /// Set the column name, returning a new Column.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Create a column of all nulls with the given dtype and length.
    pub fn all_null(name: impl Into<String>, dtype: DType, len: usize) -> Self {
        let data = match dtype {
            DType::Bool => ColumnData::Bool(vec![false; len]),
            DType::Int64 => ColumnData::Int64(vec![0; len]),
            DType::Float64 => ColumnData::Float64(vec![0.0; len]),
            DType::Str => ColumnData::Str(vec![String::new(); len]),
        };
        Column {
            data,
            null_mask: Some(vec![true; len]),
            name: name.into(),
        }
    }

    /// Get a reference to the underlying data.
    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// Get the null mask (if any).
    pub fn null_mask(&self) -> Option<&[bool]> {
        self.null_mask.as_deref()
    }

    /// Select rows by index positions, returning a new Column.
    pub fn take(&self, indices: &[usize]) -> Result<Column> {
        let data = match &self.data {
            ColumnData::Bool(v) => ColumnData::Bool(
                indices
                    .iter()
                    .map(|&i| v.get(i).copied().ok_or_else(|| idx_err(i, v.len())))
                    .collect::<Result<Vec<_>>>()?,
            ),
            ColumnData::Int64(v) => ColumnData::Int64(
                indices
                    .iter()
                    .map(|&i| v.get(i).copied().ok_or_else(|| idx_err(i, v.len())))
                    .collect::<Result<Vec<_>>>()?,
            ),
            ColumnData::Float64(v) => ColumnData::Float64(
                indices
                    .iter()
                    .map(|&i| v.get(i).copied().ok_or_else(|| idx_err(i, v.len())))
                    .collect::<Result<Vec<_>>>()?,
            ),
            ColumnData::Str(v) => ColumnData::Str(
                indices
                    .iter()
                    .map(|&i| {
                        v.get(i)
                            .cloned()
                            .ok_or_else(|| idx_err(i, v.len()))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        let null_mask = self.null_mask.as_ref().map(|m| {
            indices
                .iter()
                .map(|&i| m.get(i).copied().unwrap_or(false))
                .collect()
        });
        Ok(Column {
            data,
            null_mask,
            name: self.name.clone(),
        })
    }
}

fn idx_err(idx: usize, len: usize) -> PandasError {
    PandasError::IndexError(format!("index {} out of bounds for length {}", idx, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_basics() {
        let col = Column::new("x", ColumnData::Int64(vec![1, 2, 3]));
        assert_eq!(col.name(), "x");
        assert_eq!(col.dtype(), DType::Int64);
        assert_eq!(col.len(), 3);
        assert!(!col.has_nulls());
        assert_eq!(col.null_count(), 0);
    }

    #[test]
    fn test_column_with_nulls() {
        let col = Column::new_with_nulls(
            "y",
            ColumnData::Float64(vec![1.0, 0.0, 3.0]),
            vec![false, true, false],
        )
        .unwrap();
        assert!(col.has_nulls());
        assert_eq!(col.null_count(), 1);
        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));
    }

    #[test]
    fn test_column_no_actual_nulls_optimized() {
        let col = Column::new_with_nulls(
            "z",
            ColumnData::Int64(vec![1, 2]),
            vec![false, false],
        )
        .unwrap();
        // All false mask gets optimized away
        assert!(!col.has_nulls());
    }

    #[test]
    fn test_column_null_mask_length_mismatch() {
        let result = Column::new_with_nulls(
            "bad",
            ColumnData::Int64(vec![1, 2, 3]),
            vec![false, true],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_column_all_null() {
        let col = Column::all_null("n", DType::Str, 3);
        assert_eq!(col.len(), 3);
        assert_eq!(col.null_count(), 3);
        assert!(col.is_null(0));
        assert!(col.is_null(1));
        assert!(col.is_null(2));
    }

    #[test]
    fn test_column_take() {
        let col = Column::new("a", ColumnData::Int64(vec![10, 20, 30, 40]));
        let taken = col.take(&[3, 1, 0]).unwrap();
        match &taken.data {
            ColumnData::Int64(v) => assert_eq!(v, &[40, 20, 10]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_column_take_out_of_bounds() {
        let col = Column::new("a", ColumnData::Int64(vec![10, 20]));
        assert!(col.take(&[5]).is_err());
    }

    #[test]
    fn test_column_take_preserves_nulls() {
        let col = Column::new_with_nulls(
            "a",
            ColumnData::Int64(vec![10, 20, 30]),
            vec![false, true, false],
        )
        .unwrap();
        let taken = col.take(&[2, 1]).unwrap();
        assert!(!taken.is_null(0));
        assert!(taken.is_null(1));
    }
}
```

- [ ] **Step 2: Add column module to lib.rs**

Add `pub mod column;` and re-exports to `src/lib.rs`:

```rust
pub mod column;
pub mod dtype;
pub mod error;

pub use column::{Column, ColumnData};
pub use dtype::DType;
pub use error::{PandasError, Result};
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p pandas-rust-core
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/pandas-rust-core/src/column.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add Column type with null bitmask and take()"
```

---

### Task 4: Index Type

**Files:**
- Create: `crates/pandas-rust-core/src/index.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write index.rs**

```rust
use crate::error::{PandasError, Result};

/// Row index for DataFrame and Series.
#[derive(Debug, Clone)]
pub enum Index {
    Range(RangeIndex),
    Int64(Vec<i64>),
    Str(Vec<String>),
}

/// A range-based index (0..n), no allocation needed.
#[derive(Debug, Clone)]
pub struct RangeIndex {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

impl RangeIndex {
    pub fn new(len: usize) -> Self {
        RangeIndex {
            start: 0,
            stop: len as i64,
            step: 1,
        }
    }

    pub fn len(&self) -> usize {
        if self.step > 0 {
            ((self.stop - self.start + self.step - 1) / self.step).max(0) as usize
        } else if self.step < 0 {
            ((self.start - self.stop - self.step - 1) / (-self.step)).max(0) as usize
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, idx: usize) -> Option<i64> {
        if idx < self.len() {
            Some(self.start + (idx as i64) * self.step)
        } else {
            None
        }
    }
}

impl Index {
    pub fn len(&self) -> usize {
        match self {
            Index::Range(r) => r.len(),
            Index::Int64(v) => v.len(),
            Index::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a default RangeIndex(0..len).
    pub fn default_range(len: usize) -> Self {
        Index::Range(RangeIndex::new(len))
    }

    /// Select rows by positional indices.
    pub fn take(&self, indices: &[usize]) -> Result<Index> {
        match self {
            Index::Range(r) => {
                let values: Result<Vec<i64>> = indices
                    .iter()
                    .map(|&i| {
                        r.get(i).ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                r.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Int64(values?))
            }
            Index::Int64(v) => {
                let values: Result<Vec<i64>> = indices
                    .iter()
                    .map(|&i| {
                        v.get(i).copied().ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                v.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Int64(values?))
            }
            Index::Str(v) => {
                let values: Result<Vec<String>> = indices
                    .iter()
                    .map(|&i| {
                        v.get(i).cloned().ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                v.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Str(values?))
            }
        }
    }

    /// Look up the positional index for a label value.
    /// For RangeIndex and Int64, the label is an i64.
    /// For Str, the label is a string.
    /// Returns None if not found.
    pub fn get_loc_int(&self, label: i64) -> Option<usize> {
        match self {
            Index::Range(r) => {
                if r.step == 0 {
                    return None;
                }
                let offset = label - r.start;
                if offset % r.step != 0 {
                    return None;
                }
                let pos = (offset / r.step) as usize;
                if pos < r.len() {
                    Some(pos)
                } else {
                    None
                }
            }
            Index::Int64(v) => v.iter().position(|&x| x == label),
            Index::Str(_) => None,
        }
    }

    pub fn get_loc_str(&self, label: &str) -> Option<usize> {
        match self {
            Index::Str(v) => v.iter().position(|x| x == label),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_index() {
        let r = RangeIndex::new(5);
        assert_eq!(r.len(), 5);
        assert_eq!(r.get(0), Some(0));
        assert_eq!(r.get(4), Some(4));
        assert_eq!(r.get(5), None);
    }

    #[test]
    fn test_index_default_range() {
        let idx = Index::default_range(3);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_index_take() {
        let idx = Index::Int64(vec![10, 20, 30, 40]);
        let taken = idx.take(&[3, 0]).unwrap();
        match taken {
            Index::Int64(v) => assert_eq!(v, vec![40, 10]),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_index_get_loc_range() {
        let idx = Index::default_range(5);
        assert_eq!(idx.get_loc_int(0), Some(0));
        assert_eq!(idx.get_loc_int(4), Some(4));
        assert_eq!(idx.get_loc_int(5), None);
    }

    #[test]
    fn test_index_get_loc_str() {
        let idx = Index::Str(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(idx.get_loc_str("b"), Some(1));
        assert_eq!(idx.get_loc_str("z"), None);
    }
}
```

- [ ] **Step 2: Add to lib.rs**

Add `pub mod index;` and `pub use index::{Index, RangeIndex};`

- [ ] **Step 3: Run tests**

```bash
cargo test -p pandas-rust-core
```

- [ ] **Step 4: Commit**

```bash
git add crates/pandas-rust-core/src/index.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add Index type (Range, Int64, Str) with take() and get_loc()"
```

---

### Task 5: Series Type

**Files:**
- Create: `crates/pandas-rust-core/src/series.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write series.rs**

```rust
use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};
use crate::index::Index;

/// A single named column with an index — the 1D building block.
#[derive(Debug, Clone)]
pub struct Series {
    pub(crate) column: Column,
    pub(crate) index: Index,
}

impl Series {
    /// Create a Series from a Column with a default RangeIndex.
    pub fn new(column: Column) -> Self {
        let len = column.len();
        Series {
            column,
            index: Index::default_range(len),
        }
    }

    /// Create a Series with a specific index.
    pub fn with_index(column: Column, index: Index) -> Result<Self> {
        if column.len() != index.len() {
            return Err(PandasError::ValueError(format!(
                "column length ({}) != index length ({})",
                column.len(),
                index.len()
            )));
        }
        Ok(Series { column, index })
    }

    pub fn name(&self) -> &str {
        self.column.name()
    }

    pub fn dtype(&self) -> DType {
        self.column.dtype()
    }

    pub fn len(&self) -> usize {
        self.column.len()
    }

    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }

    pub fn column(&self) -> &Column {
        &self.column
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Select rows by positional indices.
    pub fn take(&self, indices: &[usize]) -> Result<Series> {
        let col = self.column.take(indices)?;
        let idx = self.index.take(indices)?;
        Ok(Series {
            column: col,
            index: idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_new() {
        let s = Series::new(Column::new("x", ColumnData::Int64(vec![1, 2, 3])));
        assert_eq!(s.name(), "x");
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::Int64);
    }

    #[test]
    fn test_series_with_index() {
        let col = Column::new("val", ColumnData::Float64(vec![1.0, 2.0]));
        let idx = Index::Str(vec!["a".into(), "b".into()]);
        let s = Series::with_index(col, idx).unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_series_index_length_mismatch() {
        let col = Column::new("val", ColumnData::Float64(vec![1.0, 2.0, 3.0]));
        let idx = Index::Str(vec!["a".into()]);
        assert!(Series::with_index(col, idx).is_err());
    }

    #[test]
    fn test_series_take() {
        let s = Series::new(Column::new("x", ColumnData::Int64(vec![10, 20, 30])));
        let taken = s.take(&[2, 0]).unwrap();
        assert_eq!(taken.len(), 2);
    }
}
```

- [ ] **Step 2: Add to lib.rs**

Add `pub mod series;` and `pub use series::Series;`

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p pandas-rust-core
git add crates/pandas-rust-core/src/series.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add Series type wrapping Column + Index"
```

---

### Task 6: DataFrame Type

**Files:**
- Create: `crates/pandas-rust-core/src/dataframe.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write dataframe.rs**

```rust
use indexmap::IndexMap;

use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};
use crate::index::Index;
use crate::series::Series;

/// A 2D labeled data structure — ordered collection of named columns.
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub(crate) columns: IndexMap<String, Column>,
    pub(crate) index: Index,
}

impl DataFrame {
    /// Create an empty DataFrame.
    pub fn empty() -> Self {
        DataFrame {
            columns: IndexMap::new(),
            index: Index::default_range(0),
        }
    }

    /// Create a DataFrame from a list of columns.
    /// All columns must have the same length.
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Self::empty());
        }
        let len = columns[0].len();
        for col in &columns {
            if col.len() != len {
                return Err(PandasError::ValueError(format!(
                    "column '{}' has length {} but expected {}",
                    col.name(),
                    col.len(),
                    len
                )));
            }
        }
        let mut map = IndexMap::with_capacity(columns.len());
        for col in columns {
            map.insert(col.name().to_string(), col);
        }
        Ok(DataFrame {
            columns: map,
            index: Index::default_range(len),
        })
    }

    /// Create a DataFrame from columns with a specific index.
    pub fn from_columns_with_index(columns: Vec<Column>, index: Index) -> Result<Self> {
        let mut df = Self::from_columns(columns)?;
        if !df.columns.is_empty() && df.nrows() != index.len() {
            return Err(PandasError::ValueError(format!(
                "index length ({}) != row count ({})",
                index.len(),
                df.nrows()
            )));
        }
        df.index = index;
        Ok(df)
    }

    pub fn nrows(&self) -> usize {
        self.index.len()
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    pub fn dtypes(&self) -> Vec<(&str, DType)> {
        self.columns
            .iter()
            .map(|(name, col)| (name.as_str(), col.dtype()))
            .collect()
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get a column by name.
    pub fn get_column(&self, name: &str) -> Result<&Column> {
        self.columns
            .get(name)
            .ok_or_else(|| PandasError::KeyError(name.to_string()))
    }

    /// Get a column by positional index.
    pub fn get_column_by_idx(&self, idx: usize) -> Result<&Column> {
        self.columns
            .get_index(idx)
            .map(|(_, col)| col)
            .ok_or_else(|| {
                PandasError::IndexError(format!(
                    "column index {} out of bounds for {} columns",
                    idx,
                    self.ncols()
                ))
            })
    }

    /// Extract a column as a Series.
    pub fn get_series(&self, name: &str) -> Result<Series> {
        let col = self.get_column(name)?.clone();
        Ok(Series::with_index(col, self.index.clone())
            .expect("column and index lengths match by construction"))
    }

    /// Insert or replace a column. Mutates in place.
    pub fn set_column(&mut self, column: Column) -> Result<()> {
        if !self.columns.is_empty() && column.len() != self.nrows() {
            return Err(PandasError::ValueError(format!(
                "column '{}' has length {} but DataFrame has {} rows",
                column.name(),
                column.len(),
                self.nrows()
            )));
        }
        if self.columns.is_empty() {
            self.index = Index::default_range(column.len());
        }
        self.columns.insert(column.name().to_string(), column);
        Ok(())
    }

    /// Remove a column by name, returning it.
    pub fn drop_column(&mut self, name: &str) -> Result<Column> {
        self.columns
            .shift_remove(name)
            .ok_or_else(|| PandasError::KeyError(name.to_string()))
    }

    /// Select specific columns by name, returning a new DataFrame.
    pub fn select_columns(&self, names: &[&str]) -> Result<DataFrame> {
        let cols: Result<Vec<Column>> = names
            .iter()
            .map(|n| self.get_column(n).map(|c| c.clone()))
            .collect();
        DataFrame::from_columns_with_index(cols?, self.index.clone())
    }

    /// Select rows by positional indices, returning a new DataFrame.
    pub fn take_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        let cols: Result<Vec<Column>> = self
            .columns
            .values()
            .map(|c| c.take(indices))
            .collect();
        let new_index = self.index.take(indices)?;
        DataFrame::from_columns_with_index(cols?, new_index)
    }

    /// Rename columns using a mapping of old_name -> new_name.
    pub fn rename_columns(&self, mapping: &[(&str, &str)]) -> Result<DataFrame> {
        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let new_name = mapping
                .iter()
                .find(|(old, _)| old == &name.as_str())
                .map(|(_, new)| *new)
                .unwrap_or(name.as_str());
            new_columns.insert(
                new_name.to_string(),
                col.clone().with_name(new_name),
            );
        }
        Ok(DataFrame {
            columns: new_columns,
            index: self.index.clone(),
        })
    }

    /// Deep copy.
    pub fn copy(&self) -> DataFrame {
        self.clone()
    }

    /// Iterate over columns.
    pub fn iter_columns(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.columns.iter().map(|(n, c)| (n.as_str(), c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 2, 3])),
            Column::new("b", ColumnData::Float64(vec![1.0, 2.0, 3.0])),
            Column::new("c", ColumnData::Str(vec!["x".into(), "y".into(), "z".into()])),
        ])
        .unwrap()
    }

    #[test]
    fn test_empty_dataframe() {
        let df = DataFrame::empty();
        assert_eq!(df.shape(), (0, 0));
    }

    #[test]
    fn test_from_columns() {
        let df = sample_df();
        assert_eq!(df.shape(), (3, 3));
        assert_eq!(df.column_names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_column_length_mismatch() {
        let result = DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 2])),
            Column::new("b", ColumnData::Int64(vec![1])),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_column() {
        let df = sample_df();
        let col = df.get_column("a").unwrap();
        assert_eq!(col.dtype(), DType::Int64);
        assert!(df.get_column("missing").is_err());
    }

    #[test]
    fn test_get_series() {
        let df = sample_df();
        let s = df.get_series("b").unwrap();
        assert_eq!(s.name(), "b");
        assert_eq!(s.dtype(), DType::Float64);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_set_column() {
        let mut df = sample_df();
        df.set_column(Column::new("d", ColumnData::Bool(vec![true, false, true])))
            .unwrap();
        assert_eq!(df.ncols(), 4);
        assert_eq!(df.get_column("d").unwrap().dtype(), DType::Bool);
    }

    #[test]
    fn test_set_column_wrong_length() {
        let mut df = sample_df();
        let result =
            df.set_column(Column::new("d", ColumnData::Bool(vec![true])));
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_column() {
        let mut df = sample_df();
        let dropped = df.drop_column("b").unwrap();
        assert_eq!(dropped.name(), "b");
        assert_eq!(df.ncols(), 2);
        assert!(df.drop_column("b").is_err());
    }

    #[test]
    fn test_select_columns() {
        let df = sample_df();
        let sub = df.select_columns(&["c", "a"]).unwrap();
        assert_eq!(sub.column_names(), vec!["c", "a"]);
        assert_eq!(sub.nrows(), 3);
    }

    #[test]
    fn test_take_rows() {
        let df = sample_df();
        let sub = df.take_rows(&[2, 0]).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 3);
    }

    #[test]
    fn test_rename_columns() {
        let df = sample_df();
        let renamed = df.rename_columns(&[("a", "alpha"), ("c", "gamma")]).unwrap();
        assert_eq!(renamed.column_names(), vec!["alpha", "b", "gamma"]);
    }

    #[test]
    fn test_dtypes() {
        let df = sample_df();
        let dt = df.dtypes();
        assert_eq!(dt, vec![
            ("a", DType::Int64),
            ("b", DType::Float64),
            ("c", DType::Str),
        ]);
    }
}
```

- [ ] **Step 2: Add to lib.rs**

Add `pub mod dataframe;` and `pub use dataframe::DataFrame;`

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p pandas-rust-core
git add crates/pandas-rust-core/src/dataframe.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add DataFrame with IndexMap columns, take_rows, select, rename"
```

---

### Task 7: WASM Binary Crate (Skeleton)

**Files:**
- Create: `crates/pandas-rust-python/Cargo.toml`
- Create: `crates/pandas-rust-python/src/lib.rs`
- Create: `crates/pandas-rust-wasm/Cargo.toml`
- Create: `crates/pandas-rust-wasm/src/main.rs`

This is a skeleton to verify the workspace compiles end-to-end. The Python bindings are minimal stubs for now.

- [ ] **Step 1: Create pandas-rust-python Cargo.toml**

```toml
[package]
name = "pandas-rust-python"
version = "0.1.0"
edition = "2021"

[dependencies]
pandas-rust-core = { path = "../pandas-rust-core" }
numpy-rust-python = { path = "../../packages/numpy-rust/crates/numpy-rust-python", default-features = false }
numpy-rust-core = { path = "../../packages/numpy-rust/crates/numpy-rust-core", default-features = false }
rustpython-vm = { workspace = true }
rustpython-derive = { workspace = true }
```

Note: numpy-rust features (linalg, fft, random) disabled here — pandas doesn't need them.

- [ ] **Step 2: Create pandas-rust-python src/lib.rs (stub)**

```rust
use rustpython_vm as vm;

/// Return the native pandas module definition for registration with the interpreter builder.
pub fn pandas_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    _pandas_native::module_def(ctx)
}

#[vm::pymodule]
pub mod _pandas_native {
    use vm::PyResult;
    use vm::VirtualMachine;
    use rustpython_vm as vm;

    #[pyfunction]
    fn _version(_vm: &VirtualMachine) -> PyResult<String> {
        Ok("0.1.0".to_string())
    }
}
```

- [ ] **Step 3: Create pandas-rust-wasm Cargo.toml**

```toml
[package]
name = "pandas-rust-wasm"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "pandas-python"
path = "src/main.rs"

[dependencies]
pandas-rust-python = { path = "../pandas-rust-python" }
numpy-rust-python = { path = "../../packages/numpy-rust/crates/numpy-rust-python" }
rustpython = { workspace = true, features = ["freeze-stdlib", "threading", "host_env"] }
rustpython-vm = { workspace = true }
```

- [ ] **Step 4: Create pandas-rust-wasm src/main.rs**

```rust
use std::path::PathBuf;
use std::process::ExitCode;

use rustpython::{InterpreterBuilder, InterpreterBuilderExt};

pub fn main() -> ExitCode {
    let root = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent()?.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let paths = [
        root.join("python"),
        root.join("packages/numpy-rust/python"),
    ];
    let prepend = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    let pythonpath = match std::env::var("PYTHONPATH") {
        Ok(existing) => format!("{}:{}", prepend, existing),
        Err(_) => prepend,
    };
    std::env::set_var("PYTHONPATH", pythonpath);

    let config = InterpreterBuilder::new().init_stdlib();

    let numpy_def = numpy_rust_python::numpy_module_def(&config.ctx);
    let config = config.add_native_module(numpy_def);

    let pandas_def = pandas_rust_python::pandas_module_def(&config.ctx);
    let config = config.add_native_module(pandas_def);

    rustpython::run(config)
}
```

- [ ] **Step 5: Build the workspace**

```bash
cargo build -p pandas-rust-wasm
```

This will take a while (first RustPython compile). Expected: compiles successfully.

- [ ] **Step 6: Verify the binary runs**

```bash
echo "import _pandas_native; print(_pandas_native._version())" | target/debug/pandas-python
```

Expected output: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add crates/pandas-rust-python/ crates/pandas-rust-wasm/
git commit -m "feat: add pandas-rust-python stub and pandas-rust-wasm binary"
```

---

### Task 8: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Run core tests
        run: cargo test -p pandas-rust-core
      - name: Run workspace tests
        run: cargo test --workspace

  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      - name: Check formatting
        run: cargo fmt --all -- --check
      - name: Clippy
        run: cargo clippy --workspace -- -D warnings

  python-tests:
    name: Python Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build pandas-python binary
        run: cargo build -p pandas-rust-wasm --release
      - name: Run vendored Python tests
        run: ./tests/python/run_tests.sh target/release/pandas-python

  wasm-build:
    name: WASM Build Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip1
      - uses: Swatinem/rust-cache@v2
      - name: Build for wasm32-wasip1
        run: cargo build -p pandas-rust-wasm --target wasm32-wasip1
```

- [ ] **Step 2: Create test runner script**

Create `tests/python/run_tests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

BINARY="${1:?Usage: run_tests.sh <pandas-python-binary>}"
exec "$BINARY" -m pytest tests/python/ -v
```

Make executable: `chmod +x tests/python/run_tests.sh`

- [ ] **Step 3: Create a minimal test file placeholder**

Create `tests/python/test_smoke.py`:

```python
"""Smoke test to verify the pandas-python binary works."""

def test_import_pandas_native():
    import _pandas_native
    assert _pandas_native._version() == "0.1.0"
```

- [ ] **Step 4: Commit**

```bash
git add .github/ tests/
git commit -m "ci: add GitHub Actions workflow and smoke test"
```

---

## Phase 2: Column Operations

### Task 9: Arithmetic Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/mod.rs`
- Create: `crates/pandas-rust-core/src/ops/arithmetic.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Create ops/mod.rs**

```rust
pub mod arithmetic;
```

- [ ] **Step 2: Write arithmetic.rs with tests**

Implement column + column, column + scalar for Int64 and Float64 with null propagation and type promotion. Operations: `add`, `sub`, `mul`, `div`, `neg`.

Each operation function takes two `&Column` or a `&Column` + scalar and returns `Result<Column>`. The result column inherits the name from the left operand. Null propagation: if either operand is null at position i, result is null at i. Type promotion: Int64 op Float64 → Float64 (promote the Int64 side).

Include tests:
- Int64 + Int64
- Float64 + Float64
- Int64 + Float64 (promotion)
- Column + scalar
- Null propagation
- Division by zero (Float64 → Inf/NaN, Int64 → error)
- Length mismatch → error
- Negation

- [ ] **Step 3: Add `pub mod ops;` to lib.rs**

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::arithmetic
git add crates/pandas-rust-core/src/ops/
git commit -m "feat: add column arithmetic with null propagation and type promotion"
```

---

### Task 10: Comparison Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/comparison.rs`
- Modify: `crates/pandas-rust-core/src/ops/mod.rs`

- [ ] **Step 1: Write comparison.rs**

Implement `eq`, `ne`, `lt`, `le`, `gt`, `ge` for column vs column and column vs scalar. Returns a Bool `Column`. Null compared to anything → null in the result. Same type-promotion rules as arithmetic for mixed Int64/Float64.

Also implement `and`, `or`, `not` for Bool columns (for chaining conditions like `(df["a"] > 1) & (df["b"] < 5)`).

Include tests:
- Int64 comparisons
- Float64 comparisons
- Cross-type comparison (Int64 vs Float64)
- String equality
- Null propagation
- Bool logical ops
- Length mismatch → error

- [ ] **Step 2: Add `pub mod comparison;` to ops/mod.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::comparison
git add crates/pandas-rust-core/src/ops/comparison.rs crates/pandas-rust-core/src/ops/mod.rs
git commit -m "feat: add column comparison and boolean logic ops"
```

---

### Task 11: Aggregation Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/aggregation.rs`
- Modify: `crates/pandas-rust-core/src/ops/mod.rs`

- [ ] **Step 1: Write aggregation.rs**

Implement for numeric columns (Int64, Float64): `sum`, `mean`, `min`, `max`, `count` (non-null count), `std` (sample std, ddof=1), `var` (sample variance, ddof=1), `median`, `quantile(q)`.

All skip nulls. Return an `AggResult` enum:

```rust
pub enum AggResult {
    Int64(i64),
    Float64(f64),
    Usize(usize),  // for count
    None,           // all nulls
}
```

`quantile(q)` uses linear interpolation: sort non-null values, compute position `q * (n-1)`, interpolate between adjacent values.

Also implement `count` for non-numeric columns (just counts non-nulls).

Include tests:
- sum/mean/min/max for Int64 and Float64
- count with and without nulls
- std/var with ddof=1
- median odd/even length
- quantile at 0.0, 0.25, 0.5, 0.75, 1.0
- All-null column → None result
- Empty column → None result

- [ ] **Step 2: Add to ops/mod.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::aggregation
git add crates/pandas-rust-core/src/ops/aggregation.rs crates/pandas-rust-core/src/ops/mod.rs
git commit -m "feat: add aggregation ops (sum, mean, std, median, quantile, etc.)"
```

---

### Task 12: Filter Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/filter.rs`
- Modify: `crates/pandas-rust-core/src/ops/mod.rs`

- [ ] **Step 1: Write filter.rs**

Implement:
- `filter_column(column, mask) -> Result<Column>` — keep rows where mask is true (not null and true)
- `filter_indices(mask) -> Vec<usize>` — convert bool mask to list of matching indices (useful for DataFrame filtering)

The mask must be a Bool column. Null values in the mask are treated as false (row excluded).

Include tests:
- Basic filtering
- Filtering with nulls in mask
- Filtering with nulls in data
- Empty result
- Mask length mismatch → error

- [ ] **Step 2: Add to ops/mod.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::filter
git add crates/pandas-rust-core/src/ops/filter.rs crates/pandas-rust-core/src/ops/mod.rs
git commit -m "feat: add boolean mask filtering for columns"
```

---

### Task 13: Sort Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/sort.rs`
- Modify: `crates/pandas-rust-core/src/ops/mod.rs`

- [ ] **Step 1: Write sort.rs**

Implement:
- `argsort_column(column, ascending) -> Vec<usize>` — stable sort, returns permutation indices. Nulls sort to the end regardless of direction.
- `sort_column(column, ascending) -> Column` — sort the column by its own values.
- `argsort_multi(columns, ascending_flags) -> Vec<usize>` — multi-column sort for DataFrame `sort_values(by=[...])`. Sorts by first column, breaks ties with second, etc.

For Float64, use total ordering (NaN after all other values, matching pandas).

Include tests:
- Sort Int64 ascending/descending
- Sort Float64 with NaN
- Sort Str
- Nulls at end
- Multi-column sort
- Stable sort verification

- [ ] **Step 2: Add to ops/mod.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::sort
git add crates/pandas-rust-core/src/ops/sort.rs crates/pandas-rust-core/src/ops/mod.rs
git commit -m "feat: add sort operations with null handling and multi-column support"
```

---

### Task 14: Null Operations

**Files:**
- Create: `crates/pandas-rust-core/src/ops/nulls.rs`
- Modify: `crates/pandas-rust-core/src/ops/mod.rs`

- [ ] **Step 1: Write nulls.rs**

Implement:
- `isna(column) -> Column` — Bool column, true where null
- `notna(column) -> Column` — Bool column, true where not null
- `fillna_scalar(column, value) -> Result<Column>` — replace nulls with a scalar value. Value must match column dtype (or be promotable).
- `dropna_rows(columns, how) -> Vec<usize>` — returns row indices to keep. `how="any"`: drop if any column has null. `how="all"`: drop only if all columns have null.

Include tests:
- isna/notna for column with no nulls → all false/true
- isna/notna for column with some nulls
- fillna with matching type
- fillna on column with no nulls → unchanged
- dropna "any" vs "all"

- [ ] **Step 2: Add to ops/mod.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- ops::nulls
git add crates/pandas-rust-core/src/ops/nulls.rs crates/pandas-rust-core/src/ops/mod.rs
git commit -m "feat: add null operations (isna, notna, fillna, dropna)"
```

---

### Task 15: Casting

**Files:**
- Create: `crates/pandas-rust-core/src/casting.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write casting.rs**

Implement `cast_column(column, target_dtype) -> Result<Column>`:
- Int64 → Float64: lossless
- Float64 → Int64: truncate (error if NaN/Inf)
- Bool → Int64: true=1, false=0
- Bool → Float64: true=1.0, false=0.0
- Int64/Float64 → Str: format as string
- Str → Int64: parse (error on failure)
- Str → Float64: parse (error on failure)
- Same type → clone (no-op)

Null values are preserved through casting.

Include tests for each conversion, including error cases and null preservation.

- [ ] **Step 2: Add to lib.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- casting
git add crates/pandas-rust-core/src/casting.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add dtype casting between all supported types"
```

---

### Task 16: Concatenation

**Files:**
- Create: `crates/pandas-rust-core/src/concat.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write concat.rs**

Implement:
- `concat_rows(dataframes) -> Result<DataFrame>` — stack vertically. Union of column names. Missing columns filled with nulls. Dtype promotion for matching columns with different types (Int64 + Float64 → Float64). Index is reset to RangeIndex.
- `concat_cols(dataframes) -> Result<DataFrame>` — stack horizontally. All DataFrames must have same row count. Duplicate column names get `_0`, `_1` suffixes.

Include tests:
- Concat two DataFrames with same columns
- Concat with different columns (null fill)
- Concat with type promotion
- Concat cols same row count
- Concat cols different row count → error
- Concat empty list → empty DataFrame
- Single DataFrame → clone

- [ ] **Step 2: Add to lib.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- concat
git add crates/pandas-rust-core/src/concat.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add row-wise and column-wise DataFrame concatenation"
```

---

## Phase 3: GroupBy + Merge

### Task 17: GroupBy Engine

**Files:**
- Create: `crates/pandas-rust-core/src/groupby.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write groupby.rs**

Implement:
- `GroupKey` — enum of single values (Bool/Int64/Float64/Str) or tuple of values, with Hash + Eq.
- `group_by(dataframe, by_columns) -> Result<GroupedData>` — builds `IndexMap<GroupKey, Vec<usize>>` mapping each unique key to its row indices. Uses IndexMap to preserve first-seen order.
- `GroupedData` struct holds the groups + reference to column names.
- `aggregate(grouped, source_columns, agg_fn) -> Result<DataFrame>` — apply an aggregation function per group per column. The result DataFrame has one row per group with the group keys as columns plus aggregated values.

Supported agg functions: `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `first`, `last`, `size`.

Null handling in group keys: null keys form their own group (like pandas `dropna=False`). Actually, for v1, drop rows with null keys (default pandas behavior with `dropna=True`).

Include tests:
- Single column groupby with sum
- Multi-column groupby
- All aggregation functions
- Groupby with null keys (dropped)
- Groupby on empty DataFrame
- Groupby preserves group order (first seen)

- [ ] **Step 2: Add to lib.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- groupby
git add crates/pandas-rust-core/src/groupby.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add hash-based groupby engine with all aggregation functions"
```

---

### Task 18: Merge/Join Engine

**Files:**
- Create: `crates/pandas-rust-core/src/merge.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write merge.rs**

Implement hash join:
- `merge(left, right, on, how, suffixes) -> Result<DataFrame>`
- `on`: list of column names to join on (must exist in both DataFrames)
- `how`: `Inner`, `Left`, `Right`, `Outer` (enum)
- `suffixes`: tuple of strings for overlapping non-key column names (default `("_x", "_y")`)

Algorithm:
1. Build HashMap from the smaller DataFrame's key columns → row indices
2. Probe with larger DataFrame's key columns
3. For each match, combine rows. For unmatched (Left/Right/Outer), fill with nulls.
4. Result has: key columns (from left), left non-key columns, right non-key columns (with suffixes if overlap).

Null keys never match (standard SQL semantics).

Include tests:
- Inner join
- Left join with unmatched right rows
- Right join
- Outer join
- Multi-column join
- Overlapping column names with suffixes
- No matches → empty DataFrame (inner) or all nulls (outer)
- Null keys don't match

- [ ] **Step 2: Add to lib.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- merge
git add crates/pandas-rust-core/src/merge.rs crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add hash-based merge/join engine (inner, left, right, outer)"
```

---

## Phase 4: CSV I/O

### Task 19: CSV Reader

**Files:**
- Create: `crates/pandas-rust-core/src/io/mod.rs`
- Create: `crates/pandas-rust-core/src/io/csv.rs`
- Modify: `crates/pandas-rust-core/src/lib.rs`

- [ ] **Step 1: Write io/csv.rs — reader**

Implement `read_csv(reader, options) -> Result<DataFrame>`:

```rust
pub struct CsvReadOptions {
    pub delimiter: u8,          // default b','
    pub has_header: bool,       // default true
    pub columns: Option<Vec<String>>,  // select specific columns
    pub dtype_overrides: Vec<(String, DType)>,
    pub na_values: Vec<String>, // additional NA representations
    pub max_rows: Option<usize>,
}
```

Two-pass approach:
1. Read all rows as strings using the `csv` crate
2. Infer dtypes per column: try Bool → Int64 → Float64 → Str. Empty strings and na_values are null.
3. Apply dtype overrides
4. Build columns with null bitmasks

Read from `impl Read` so it works with files and strings.

Include tests:
- Basic CSV with header
- CSV without header (auto-generate column names: `0`, `1`, ...)
- Dtype inference (int, float, string, bool, mixed)
- NA values (empty string, "NA", "null", "NaN")
- Custom delimiter (tab, semicolon)
- Column selection
- Dtype override
- max_rows
- Quoted fields with commas and newlines
- Empty file → empty DataFrame

- [ ] **Step 2: Write io/csv.rs — writer**

Implement `to_csv(dataframe, writer, options) -> Result<()>`:

```rust
pub struct CsvWriteOptions {
    pub delimiter: u8,
    pub header: bool,
    pub na_rep: String,   // default ""
}
```

Write to `impl Write`. Null values written as `na_rep`.

Include tests:
- Round-trip: write then read back, compare
- Custom na_rep
- No header option

- [ ] **Step 3: Create io/mod.rs**

```rust
pub mod csv;
```

- [ ] **Step 4: Add `pub mod io;` to lib.rs, run tests, commit**

```bash
cargo test -p pandas-rust-core -- io
git add crates/pandas-rust-core/src/io/ crates/pandas-rust-core/src/lib.rs
git commit -m "feat: add CSV reader/writer with dtype inference and null handling"
```

---

## Phase 5: Python Bindings

### Task 20: PyDataFrame and PySeries Bindings

**Files:**
- Create: `crates/pandas-rust-python/src/py_dataframe.rs`
- Create: `crates/pandas-rust-python/src/py_series.rs`
- Create: `crates/pandas-rust-python/src/py_index.rs`
- Create: `crates/pandas-rust-python/src/py_column.rs`
- Modify: `crates/pandas-rust-python/src/lib.rs`

- [ ] **Step 1: Write py_column.rs**

Internal helper module — not exposed as a Python class. Provides conversion functions:
- `pyobj_to_column(name, data, vm) -> PyResult<Column>` — convert a Python list to a Column, inferring dtype
- `column_to_pyobj(column, idx, vm) -> PyResult<PyObjectRef>` — extract a single value as a Python object
- `column_to_pylist(column, vm) -> PyResult<PyObjectRef>` — convert full column to Python list

- [ ] **Step 2: Write py_index.rs**

`PyIndex` wraps `pandas_rust_core::Index`. Expose:
- `__repr__` — e.g., `RangeIndex(start=0, stop=5, step=1)` or `Index([1, 2, 3], dtype='int64')`
- `__len__`
- `__getitem__` — positional access
- `tolist()` — convert to Python list

- [ ] **Step 3: Write py_series.rs**

`PySeries` wraps `pandas_rust_core::Series`. Expose:
- Construction: `__init__(data, name, index)` — from Python list
- Properties: `name`, `dtype`, `index`, `values` (→ numpy ndarray via numpy-rust-core interop; Int64 with nulls → Float64+NaN)
- `__repr__` — formatted like pandas Series output
- `__len__`
- `__getitem__` — positional access
- Arithmetic: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`
- Comparison: `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`
- Boolean: `__and__`, `__or__`, `__invert__`
- Aggregation methods: `sum()`, `mean()`, `min()`, `max()`, `count()`, `std()`, `var()`, `median()`
- Null ops: `isna()`, `notna()`, `fillna(value)`, `dropna()`
- `astype(dtype_str)`, `copy()`, `sort_values(ascending=True)`
- `tolist()`, `to_dict()`

- [ ] **Step 4: Write py_dataframe.rs**

`PyDataFrame` wraps `pandas_rust_core::DataFrame`. Expose:
- Construction: `__init__(data)` — from dict of lists
- Properties: `shape`, `dtypes`, `columns`, `index`
- `__repr__` — tabular display
- `__len__`
- `__getitem__(key)` — column selection (string → Series, list → DataFrame)
- `__setitem__(key, value)` — column assignment
- `get_column(name)` → PySeries
- `head(n=5)`, `tail(n=5)`
- Aggregation: `sum()`, `mean()`, `min()`, `max()`, `count()`, `std()`, `var()`, `median()`
- `sort_values(by, ascending=True)`
- `drop(columns=[...])`
- `rename(columns={...})`
- `astype(dtype_map)`
- `copy()`
- `fillna(value)`, `dropna(how="any")`, `isna()`, `notna()`
- `groupby(by)` → PyGroupBy
- `merge(right, on, how, suffixes)`
- `to_csv(path=None)`, `to_dict(orient="dict")`
- `to_numpy()` — convert to 2D numpy ndarray via numpy-rust-core interop. Mixed-dtype DataFrames promote all columns to Str. Int64 columns with nulls promote to Float64 (NaN for nulls).
- `values` property — alias for `to_numpy()`
- `info()` — print column names, non-null counts, and dtypes (pure Python, delegates to per-column `count()` and `dtype`)
- `describe()`

- [ ] **Step 5: Update lib.rs to register all classes and functions**

Register `PyDataFrame`, `PySeries`, `PyIndex` as `#[pyattr]` types. Register `read_csv` and `concat` as `#[pyfunction]`.

Reference: `/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-python/src/lib.rs` for the exact pattern of class registration.

- [ ] **Step 6: Build and test**

```bash
cargo build -p pandas-rust-wasm
echo "import _pandas_native; print(dir(_pandas_native))" | target/debug/pandas-python
```

Expected: shows DataFrame, Series, read_csv, concat, etc.

- [ ] **Step 7: Commit**

```bash
git add crates/pandas-rust-python/
git commit -m "feat: add RustPython bindings for DataFrame, Series, Index, and I/O"
```

---

### Task 21: PyGroupBy Binding

**Files:**
- Create: `crates/pandas-rust-python/src/py_groupby.rs`
- Create: `crates/pandas-rust-python/src/py_io.rs`
- Modify: `crates/pandas-rust-python/src/lib.rs`

- [ ] **Step 1: Write py_groupby.rs**

`PyGroupBy` wraps the grouped data. Expose:
- Aggregation methods: `sum()`, `mean()`, `min()`, `max()`, `count()`, `std()`, `var()`, `median()`, `first()`, `last()`, `size()` — each returns a PyDataFrame
- `__repr__`
- `__iter__` — iterate over (key, sub-DataFrame) pairs

- [ ] **Step 2: Write py_io.rs**

Wrap `read_csv` and `to_csv`:
- `read_csv(path, delimiter=",", header=True, columns=None, dtype=None, na_values=None, nrows=None)` → PyDataFrame
- `to_csv` is a method on PyDataFrame (already defined), but also expose a standalone helper

- [ ] **Step 3: Register in lib.rs, build, commit**

```bash
cargo build -p pandas-rust-wasm
git add crates/pandas-rust-python/
git commit -m "feat: add PyGroupBy and CSV I/O Python bindings"
```

---

### Task 21.5: numpy-rust Interop in Python Bindings

**Files:**
- Modify: `crates/pandas-rust-python/src/py_series.rs`
- Modify: `crates/pandas-rust-python/src/py_dataframe.rs`

This task implements the `to_numpy()` and `values` interop described in the spec's "numpy-rust Interop" section.

- [ ] **Step 1: Implement Series.to_numpy() / values in py_series.rs**

Add a method that converts the underlying `Column` data to a numpy-rust-core `NdArray`:
- Float64 column → 1D Float64 NdArray directly
- Int64 column with no nulls → 1D Float64 NdArray (cast i64 → f64)
- Int64 column with nulls → 1D Float64 NdArray, null positions → f64::NAN
- Bool column → 1D Float64 NdArray (true=1.0, false=0.0)
- Str column → error for `to_numpy()` (strings can't be in a numeric ndarray)

Use `numpy_rust_core::NdArray` and wrap with `numpy_rust_python::py_array::PyNdArray`.

Reference: see how numpy-rust-python creates `PyNdArray` instances in `/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-python/src/py_creation.rs`.

- [ ] **Step 2: Implement DataFrame.to_numpy() / values in py_dataframe.rs**

Build a 2D NdArray from all columns:
- If all numeric (Int64/Float64): stack columns into 2D Float64 NdArray
- If mixed types: convert all columns to Str, return error (or fallback to Python list of lists)
- Null handling: Int64 nulls → NaN in Float64, Float64 nulls → NaN

- [ ] **Step 3: Build and verify**

```bash
cargo build -p pandas-rust-wasm
echo "import pandas as pd; import numpy as np; s = pd.Series([1,2,3], name='x'); arr = s.to_numpy(); print(type(arr), arr)" | target/debug/pandas-python
```

Expected: `<class 'numpy.ndarray'> [1. 2. 3.]`

- [ ] **Step 4: Commit**

```bash
git add crates/pandas-rust-python/
git commit -m "feat: add to_numpy() and values interop with numpy-rust"
```

---

## Phase 6: Python API Layer

### Task 22: pandas Package Init and DataFrame Wrapper

**Files:**
- Create: `python/pandas/__init__.py`
- Create: `python/pandas/core/__init__.py`
- Create: `python/pandas/core/frame.py`
- Create: `python/pandas/core/dtypes.py`

- [ ] **Step 1: Write python/pandas/core/dtypes.py**

Dtype helper constants and the dtype string → DType mapping for the Python layer.

- [ ] **Step 2: Write python/pandas/core/frame.py**

The `DataFrame` class wrapping `_pandas_native.PyDataFrame`. This is the main user-facing class. Key implementation:

- `__init__` accepts: dict of lists, dict of Series, list of dicts, or raw PyDataFrame
- `loc` and `iloc` properties return `_LocIndexer` / `_iLocIndexer` classes
- `_LocIndexer.__getitem__` translates label-based indexing to native calls
- `_iLocIndexer.__getitem__` translates integer-based indexing to native calls
- `__repr__` delegates to native `__repr__`
- Method chaining: all methods that return DataFrames wrap the native result in a new Python DataFrame
- `describe()` computes stats using aggregation methods, builds a new DataFrame
- `info()` prints column names, non-null counts, dtypes, and memory usage to stdout (pure Python)
- `head(n)` → `self.iloc[:n]`
- `tail(n)` → `self.iloc[-n:]`

- [ ] **Step 3: Write python/pandas/__init__.py**

```python
"""pandas-compatible Python package wrapping the Rust native module."""
from .core.frame import DataFrame
from .core.series import Series  # added in next task
from _pandas_native import read_csv, concat

__version__ = "0.1.0"
```

- [ ] **Step 4: Write python/pandas/core/__init__.py**

```python
from .frame import DataFrame
```

- [ ] **Step 5: Test manually**

```bash
echo "import pandas as pd; df = pd.DataFrame({'a': [1,2,3], 'b': [4.0,5.0,6.0]}); print(df)" | target/debug/pandas-python
```

- [ ] **Step 6: Commit**

```bash
git add python/
git commit -m "feat: add Python pandas package with DataFrame wrapper"
```

---

### Task 23: Series Wrapper and GroupBy Wrapper

**Files:**
- Create: `python/pandas/core/series.py`
- Create: `python/pandas/core/groupby.py`
- Create: `python/pandas/core/indexes.py`
- Modify: `python/pandas/__init__.py`

- [ ] **Step 1: Write series.py**

`Series` class wrapping `_pandas_native.PySeries`. Support arithmetic operators, comparison operators, aggregation methods, null ops, and repr. Also support construction from a Python list.

- [ ] **Step 2: Write groupby.py**

`GroupBy` class wrapping `_pandas_native.PyGroupBy`. Support all aggregation methods, `__iter__` for (key, sub_df) iteration.

- [ ] **Step 3: Write indexes.py**

Thin wrapper around `_pandas_native.PyIndex` for `RangeIndex` display.

- [ ] **Step 4: Update __init__.py imports**

Add Series to the public API.

- [ ] **Step 5: Test manually, commit**

```bash
echo "import pandas as pd; s = pd.Series([1,2,3], name='x'); print(s); print(s.mean())" | target/debug/pandas-python
git add python/
git commit -m "feat: add Series, GroupBy, and Index Python wrappers"
```

---

### Task 24: I/O and Testing Utilities

**Files:**
- Create: `python/pandas/io/__init__.py`
- Create: `python/pandas/io/parsers.py`
- Create: `python/pandas/_testing.py`

- [ ] **Step 1: Write io/parsers.py**

Thin wrapper around `_pandas_native.read_csv` with pandas-compatible signature.

- [ ] **Step 2: Write _testing.py**

Test utility functions:
- `assert_frame_equal(left, right, check_dtype=True)` — compare two DataFrames element by element
- `assert_series_equal(left, right, check_dtype=True, check_names=True)` — compare two Series

These compare shapes, column names, dtypes, and values. For float comparisons, use approximate equality.

- [ ] **Step 3: Commit**

```bash
git add python/
git commit -m "feat: add I/O wrappers and testing utilities"
```

---

## Phase 7: Integration Tests + Upstream Compat

### Task 25: Python Integration Tests

**Files:**
- Create: `tests/python/test_dataframe.py`
- Create: `tests/python/test_series.py`
- Create: `tests/python/test_indexing.py`
- Create: `tests/python/test_groupby.py`
- Create: `tests/python/test_merge.py`
- Create: `tests/python/test_io.py`
- Create: `tests/python/test_numpy_interop.py`
- Modify: `tests/python/test_smoke.py` (remove or keep)

- [ ] **Step 1: Write test_dataframe.py**

Test DataFrame construction (dict of lists, empty), properties (shape, dtypes, columns), column access, `__setitem__`, head/tail, sort_values, drop, rename, copy, describe, repr.

- [ ] **Step 2: Write test_series.py**

Test Series construction, arithmetic, comparisons, aggregations (sum, mean, std, median), null ops (isna, fillna, dropna), astype, sort_values.

- [ ] **Step 3: Write test_indexing.py**

Test `iloc` (int, slice, list), `loc` (label, slice, list), `__getitem__` (column name, list of names, boolean mask), `__setitem__` (new column, replace column).

- [ ] **Step 4: Write test_groupby.py**

Test groupby with single/multiple keys, all aggregation functions, iteration over groups.

- [ ] **Step 5: Write test_merge.py**

Test inner/left/right/outer joins, multi-column keys, suffix handling, no-match cases.

- [ ] **Step 6: Write test_io.py**

Test read_csv/to_csv round-trip, dtype inference, NA handling, custom delimiter, column selection.

- [ ] **Step 7: Write test_numpy_interop.py**

Test:
- `Series.to_numpy()` for Int64, Float64, Bool columns
- `Series.values` property returns ndarray
- `DataFrame.to_numpy()` for homogeneous numeric DataFrames
- Int64 with nulls → Float64 with NaN
- `type(result)` is numpy ndarray
- Round-trip: create from numpy array, convert back to numpy

- [ ] **Step 8: Run all tests**

```bash
cargo build -p pandas-rust-wasm && target/debug/pandas-python -m pytest tests/python/ -v
```

- [ ] **Step 9: Commit**

```bash
git add tests/python/
git commit -m "test: add comprehensive Python integration tests"
```

---

### Task 26: Upstream Pandas Compat Tests

**Files:**
- Create: `tests/pandas_compat/run_compat.py`
- Create: `tests/pandas_compat/conftest.py`

- [ ] **Step 1: Create run_compat.py harness**

A test runner that:
1. Discovers vendored test files in `tests/pandas_compat/`
2. Runs them via pytest with xfail markers for unsupported features
3. Reports pass/fail/skip/xfail counts
4. Exits 0 if no unexpected failures

- [ ] **Step 2: Create conftest.py**

Fixtures and markers for the compat tests. Import our pandas as the `pd` fixture.

- [ ] **Step 3: Vendor initial upstream test files**

From the pandas source (BSD-3 licensed), copy and adapt:
- `tests/frame/test_api.py` → `tests/pandas_compat/test_frame_basic.py`
- `tests/frame/indexing/test_getitem.py` → `tests/pandas_compat/test_frame_indexing.py`
- `tests/series/test_api.py` → `tests/pandas_compat/test_series_basic.py`

Strip tests that use features outside v1 scope. Mark remaining unsupported tests with `@pytest.mark.xfail(reason="not yet implemented")`.

- [ ] **Step 4: Run compat tests, commit**

```bash
target/debug/pandas-python tests/pandas_compat/run_compat.py --ci
git add tests/pandas_compat/
git commit -m "test: add upstream pandas compatibility test harness"
```

---

### Task 27: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace
cargo build -p pandas-rust-wasm
target/debug/pandas-python -m pytest tests/python/ -v
```

- [ ] **Step 2: Run clippy and fmt**

```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Verify WASM build**

```bash
rustup target add wasm32-wasip1
cargo build -p pandas-rust-wasm --target wasm32-wasip1
```

- [ ] **Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "fix: address clippy warnings and formatting"
```
