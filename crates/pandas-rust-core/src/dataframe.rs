use indexmap::IndexMap;

use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};
use crate::index::Index;
use crate::ops::unique::{duplicated_multi, Keep};
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
        let cols: Result<Vec<Column>> = names.iter().map(|n| self.get_column(n).cloned()).collect();
        DataFrame::from_columns_with_index(cols?, self.index.clone())
    }

    /// Select rows by positional indices, returning a new DataFrame.
    pub fn take_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        let cols: Result<Vec<Column>> = self.columns.values().map(|c| c.take(indices)).collect();
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
            new_columns.insert(new_name.to_string(), col.clone().with_name(new_name));
        }
        Ok(DataFrame {
            columns: new_columns,
            index: self.index.clone(),
        })
    }

    /// Return a Bool column marking duplicate rows.
    ///
    /// If `subset` is `Some`, only those columns are used for comparison.
    /// Otherwise all columns are used.
    pub fn duplicated_rows(&self, subset: Option<&[&str]>, keep: Keep) -> Result<Column> {
        let col_refs: Vec<&Column> = match subset {
            Some(names) => {
                let mut cols = Vec::with_capacity(names.len());
                for name in names {
                    cols.push(self.get_column(name)?);
                }
                cols
            }
            None => self.columns.values().collect(),
        };
        Ok(duplicated_multi(&col_refs, keep))
    }

    /// Return a new DataFrame with duplicate rows removed.
    ///
    /// If `subset` is `Some`, only those columns are considered when determining
    /// duplicates. `keep` controls which occurrence is retained.
    pub fn drop_duplicates(&self, subset: Option<&[&str]>, keep: Keep) -> Result<DataFrame> {
        let dup_col = self.duplicated_rows(subset, keep)?;
        let keep_indices: Vec<usize> = (0..self.nrows())
            .filter(|&i| match dup_col.data() {
                ColumnData::Bool(v) => !v[i],
                _ => true,
            })
            .collect();
        self.take_rows(&keep_indices)
    }

    /// Deep copy.
    pub fn copy(&self) -> DataFrame {
        self.clone()
    }

    /// Iterate over columns.
    pub fn iter_columns(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.columns.iter().map(|(n, c)| (n.as_str(), c))
    }

    /// Transpose: swap rows and columns.
    ///
    /// Column names become row labels (the first column of the result, named "").
    /// Row indices (0, 1, 2, ...) become new column names.
    /// All numeric values are promoted to Float64; if any Str column exists, all
    /// values become Str.
    pub fn transpose(&self) -> Result<DataFrame> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        if ncols == 0 || nrows == 0 {
            return Ok(DataFrame::empty());
        }

        // Determine target dtype
        let has_str = self.columns.values().any(|c| c.dtype() == DType::Str);

        // Old column names become first column
        let col_names: Vec<String> = self.columns.keys().cloned().collect();

        // New columns: one per original row, named "0", "1", ...
        // Plus the first column of old column names (named "")
        let mut new_cols: Vec<Column> = Vec::with_capacity(nrows + 1);

        // Index column: old column names as strings
        let idx_col = Column::new("", ColumnData::Str(col_names.clone()));
        new_cols.push(idx_col);

        // For each old row, build a new column
        for row_idx in 0..nrows {
            let new_col_name = row_idx.to_string();
            if has_str {
                // Stringify everything
                let vals: Vec<String> = self
                    .columns
                    .values()
                    .map(|c| {
                        if c.is_null(row_idx) {
                            return "".to_string();
                        }
                        match c.data() {
                            ColumnData::Int64(v) => v[row_idx].to_string(),
                            ColumnData::Float64(v) => v[row_idx].to_string(),
                            ColumnData::Str(v) => v[row_idx].clone(),
                            ColumnData::Bool(v) => v[row_idx].to_string(),
                        }
                    })
                    .collect();
                new_cols.push(Column::new(new_col_name, ColumnData::Str(vals)));
            } else {
                // All numeric: promote to Float64
                let mut null_mask = vec![false; ncols];
                let mut has_nulls = false;
                let vals: Vec<f64> = self
                    .columns
                    .values()
                    .enumerate()
                    .map(|(col_i, c)| {
                        if c.is_null(row_idx) {
                            null_mask[col_i] = true;
                            has_nulls = true;
                            return f64::NAN;
                        }
                        match c.data() {
                            ColumnData::Int64(v) => v[row_idx] as f64,
                            ColumnData::Float64(v) => v[row_idx],
                            ColumnData::Bool(v) => {
                                if v[row_idx] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ColumnData::Str(_) => unreachable!(),
                        }
                    })
                    .collect();
                let col = if has_nulls {
                    Column::new_with_nulls(new_col_name, ColumnData::Float64(vals), null_mask)?
                } else {
                    Column::new(new_col_name, ColumnData::Float64(vals))
                };
                new_cols.push(col);
            }
        }

        DataFrame::from_columns(new_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 2, 3])),
            Column::new("b", ColumnData::Float64(vec![1.0, 2.0, 3.0])),
            Column::new(
                "c",
                ColumnData::Str(vec!["x".into(), "y".into(), "z".into()]),
            ),
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
        let result = df.set_column(Column::new("d", ColumnData::Bool(vec![true])));
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
        let renamed = df
            .rename_columns(&[("a", "alpha"), ("c", "gamma")])
            .unwrap();
        assert_eq!(renamed.column_names(), vec!["alpha", "b", "gamma"]);
    }

    #[test]
    fn test_dtypes() {
        let df = sample_df();
        let dt = df.dtypes();
        assert_eq!(
            dt,
            vec![
                ("a", DType::Int64),
                ("b", DType::Float64),
                ("c", DType::Str),
            ]
        );
    }

    #[test]
    fn test_drop_duplicates_all_columns() {
        let df = DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 2, 1, 3])),
            Column::new(
                "b",
                ColumnData::Str(vec!["x".into(), "y".into(), "x".into(), "z".into()]),
            ),
        ])
        .unwrap();
        let deduped = df.drop_duplicates(None, Keep::First).unwrap();
        assert_eq!(deduped.nrows(), 3);
    }

    #[test]
    fn test_drop_duplicates_subset() {
        let df = DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 1, 2])),
            Column::new("b", ColumnData::Int64(vec![10, 20, 30])),
        ])
        .unwrap();
        // Only column "a" considered; rows 0 and 1 share a=1, keep first.
        let deduped = df.drop_duplicates(Some(&["a"]), Keep::First).unwrap();
        assert_eq!(deduped.nrows(), 2);
    }

    #[test]
    fn test_drop_duplicates_keep_last() {
        let df = DataFrame::from_columns(vec![Column::new("a", ColumnData::Int64(vec![1, 2, 1]))])
            .unwrap();
        let deduped = df.drop_duplicates(None, Keep::Last).unwrap();
        assert_eq!(deduped.nrows(), 2);
        // Kept rows should be indices 1 and 2 (last occurrence of 1 is at index 2).
        match deduped.get_column("a").unwrap().data() {
            ColumnData::Int64(v) => assert_eq!(v, &[2, 1]),
            _ => panic!(),
        }
    }
}
