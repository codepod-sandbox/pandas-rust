use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Convert a Bool column to a list of indices where the value is true (not null).
pub fn filter_indices(mask: &Column) -> Result<Vec<usize>> {
    if mask.dtype() != DType::Bool {
        return Err(PandasError::TypeError(format!(
            "mask must be Bool dtype, got {:?}",
            mask.dtype()
        )));
    }
    let ColumnData::Bool(vals) = &mask.data else {
        unreachable!()
    };
    let indices = vals
        .iter()
        .enumerate()
        .filter(|(i, &v)| !mask.is_null(*i) && v)
        .map(|(i, _)| i)
        .collect();
    Ok(indices)
}

/// Keep rows where mask is true (not null and true).
/// Mask must be Bool column with the same length as column.
pub fn filter_column(column: &Column, mask: &Column) -> Result<Column> {
    if mask.dtype() != DType::Bool {
        return Err(PandasError::TypeError(format!(
            "mask must be Bool dtype, got {:?}",
            mask.dtype()
        )));
    }
    if mask.len() != column.len() {
        return Err(PandasError::ValueError(format!(
            "mask length ({}) must match column length ({})",
            mask.len(),
            column.len()
        )));
    }
    let indices = filter_indices(mask)?;
    column.take(&indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn bool_col(vals: Vec<bool>) -> Column {
        Column::new("mask", ColumnData::Bool(vals))
    }

    fn bool_col_with_nulls(vals: Vec<bool>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("mask", ColumnData::Bool(vals), mask).unwrap()
    }

    fn int_col(vals: Vec<i64>) -> Column {
        Column::new("x", ColumnData::Int64(vals))
    }

    fn int_col_with_nulls(vals: Vec<i64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Int64(vals), mask).unwrap()
    }

    #[test]
    fn test_filter_indices_basic() {
        let mask = bool_col(vec![true, false, true, false, true]);
        let indices = filter_indices(&mask).unwrap();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_filter_indices_all_false() {
        let mask = bool_col(vec![false, false, false]);
        let indices = filter_indices(&mask).unwrap();
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn test_filter_indices_all_true() {
        let mask = bool_col(vec![true, true, true]);
        let indices = filter_indices(&mask).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_filter_indices_nulls_excluded() {
        // null in mask = exclude row (treated as false)
        let mask = bool_col_with_nulls(vec![true, true, true], vec![false, true, false]);
        let indices = filter_indices(&mask).unwrap();
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_filter_indices_non_bool_error() {
        let col = Column::new("x", ColumnData::Int64(vec![1, 2, 3]));
        assert!(filter_indices(&col).is_err());
    }

    #[test]
    fn test_filter_column_basic() {
        let col = int_col(vec![10, 20, 30, 40, 50]);
        let mask = bool_col(vec![true, false, true, false, true]);
        let result = filter_column(&col, &mask).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[10, 30, 50]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_filter_column_nulls_in_data() {
        let col = int_col_with_nulls(vec![10, 99, 30], vec![false, true, false]);
        let mask = bool_col(vec![true, true, true]);
        let result = filter_column(&col, &mask).unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.is_null(0));
        assert!(result.is_null(1)); // the null is preserved
        assert!(!result.is_null(2));
    }

    #[test]
    fn test_filter_column_nulls_in_mask() {
        let col = int_col(vec![10, 20, 30]);
        // null in mask position 1 -> row 1 excluded
        let mask = bool_col_with_nulls(vec![true, true, true], vec![false, true, false]);
        let result = filter_column(&col, &mask).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[10, 30]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_filter_column_empty_result() {
        let col = int_col(vec![1, 2, 3]);
        let mask = bool_col(vec![false, false, false]);
        let result = filter_column(&col, &mask).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_filter_column_length_mismatch() {
        let col = int_col(vec![1, 2, 3]);
        let mask = bool_col(vec![true, false]);
        assert!(filter_column(&col, &mask).is_err());
    }

    #[test]
    fn test_filter_column_non_bool_mask() {
        let col = int_col(vec![1, 2, 3]);
        let mask = Column::new("m", ColumnData::Int64(vec![1, 0, 1]));
        assert!(filter_column(&col, &mask).is_err());
    }

    #[test]
    fn test_filter_column_str() {
        let col = Column::new(
            "s",
            ColumnData::Str(vec!["a".into(), "b".into(), "c".into()]),
        );
        let mask = bool_col(vec![false, true, false]);
        let result = filter_column(&col, &mask).unwrap();
        match &result.data {
            ColumnData::Str(v) => assert_eq!(v, &["b"]),
            _ => panic!("wrong dtype"),
        }
    }
}
