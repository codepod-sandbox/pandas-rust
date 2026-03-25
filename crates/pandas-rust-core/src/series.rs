use crate::column::Column;
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
    use crate::column::ColumnData;

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
