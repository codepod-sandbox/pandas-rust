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
            .is_some_and(|m| m.get(idx).copied().unwrap_or(false))
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
                    .map(|&i| v.get(i).cloned().ok_or_else(|| idx_err(i, v.len())))
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
        let col =
            Column::new_with_nulls("z", ColumnData::Int64(vec![1, 2]), vec![false, false]).unwrap();
        assert!(!col.has_nulls());
    }

    #[test]
    fn test_column_null_mask_length_mismatch() {
        let result =
            Column::new_with_nulls("bad", ColumnData::Int64(vec![1, 2, 3]), vec![false, true]);
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
