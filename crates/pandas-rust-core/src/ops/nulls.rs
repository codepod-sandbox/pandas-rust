use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Scalar value for fillna operations.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Str(String),
}

/// Strategy for dropna.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DropHow {
    /// Drop row if ANY column has null.
    Any,
    /// Drop row only if ALL columns have null.
    All,
}

/// Return a Bool column: true where value is null.
pub fn isna(col: &Column) -> Column {
    let vals: Vec<bool> = (0..col.len()).map(|i| col.is_null(i)).collect();
    Column::new(col.name(), ColumnData::Bool(vals))
}

/// Return a Bool column: true where value is not null.
pub fn notna(col: &Column) -> Column {
    let vals: Vec<bool> = (0..col.len()).map(|i| !col.is_null(i)).collect();
    Column::new(col.name(), ColumnData::Bool(vals))
}

/// Replace null values with a scalar. Returns a clone if no nulls.
pub fn fillna_scalar(col: &Column, value: &ScalarValue) -> Result<Column> {
    if !col.has_nulls() {
        return Ok(col.clone());
    }
    match (&col.data, value) {
        (ColumnData::Bool(v), ScalarValue::Bool(fill)) => {
            let data: Vec<bool> = v
                .iter()
                .enumerate()
                .map(|(i, &val)| if col.is_null(i) { *fill } else { val })
                .collect();
            Ok(Column::new(col.name(), ColumnData::Bool(data)))
        }
        (ColumnData::Int64(v), ScalarValue::Int64(fill)) => {
            let data: Vec<i64> = v
                .iter()
                .enumerate()
                .map(|(i, &val)| if col.is_null(i) { *fill } else { val })
                .collect();
            Ok(Column::new(col.name(), ColumnData::Int64(data)))
        }
        (ColumnData::Float64(v), ScalarValue::Float64(fill)) => {
            let data: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, &val)| if col.is_null(i) { *fill } else { val })
                .collect();
            Ok(Column::new(col.name(), ColumnData::Float64(data)))
        }
        // Int64 + Float64 scalar: promote column to Float64
        (ColumnData::Int64(v), ScalarValue::Float64(fill)) => {
            let data: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, &val)| if col.is_null(i) { *fill } else { val as f64 })
                .collect();
            Ok(Column::new(col.name(), ColumnData::Float64(data)))
        }
        (ColumnData::Str(v), ScalarValue::Str(fill)) => {
            let data: Vec<String> = v
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    if col.is_null(i) {
                        fill.clone()
                    } else {
                        val.clone()
                    }
                })
                .collect();
            Ok(Column::new(col.name(), ColumnData::Str(data)))
        }
        // Mismatched type errors
        _ => Err(PandasError::TypeError(format!(
            "cannot fill column of dtype {:?} with {:?}",
            col.dtype(),
            scalar_dtype(value)
        ))),
    }
}

fn scalar_dtype(v: &ScalarValue) -> DType {
    match v {
        ScalarValue::Bool(_) => DType::Bool,
        ScalarValue::Int64(_) => DType::Int64,
        ScalarValue::Float64(_) => DType::Float64,
        ScalarValue::Str(_) => DType::Str,
    }
}

/// Return indices of rows to KEEP based on null strategy across multiple columns.
pub fn dropna_rows(columns: &[&Column], how: DropHow) -> Vec<usize> {
    if columns.is_empty() {
        return vec![];
    }
    let len = columns[0].len();
    (0..len)
        .filter(|&row| {
            match how {
                DropHow::Any => {
                    // Keep if NO column has null at this row
                    columns.iter().all(|col| !col.is_null(row))
                }
                DropHow::All => {
                    // Keep if NOT ALL columns are null at this row
                    !columns.iter().all(|col| col.is_null(row))
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn int_col(vals: Vec<i64>) -> Column {
        Column::new("x", ColumnData::Int64(vals))
    }

    fn int_col_with_nulls(vals: Vec<i64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Int64(vals), mask).unwrap()
    }

    fn float_col_with_nulls(vals: Vec<f64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Float64(vals), mask).unwrap()
    }

    #[test]
    fn test_isna_no_nulls() {
        let col = int_col(vec![1, 2, 3]);
        let result = isna(&col);
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[false, false, false]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_isna_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let result = isna(&col);
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[false, true, false]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_notna_no_nulls() {
        let col = int_col(vec![1, 2, 3]);
        let result = notna(&col);
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[true, true, true]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_notna_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let result = notna(&col);
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[true, false, true]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_isna_name_preserved() {
        let col = Column::new("myname", ColumnData::Int64(vec![1]));
        assert_eq!(isna(&col).name(), "myname");
    }

    #[test]
    fn test_fillna_int64_no_nulls() {
        let col = int_col(vec![1, 2, 3]);
        let result = fillna_scalar(&col, &ScalarValue::Int64(0)).unwrap();
        // Should return clone
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_fillna_int64_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let result = fillna_scalar(&col, &ScalarValue::Int64(0)).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 0, 3]),
            _ => panic!("wrong dtype"),
        }
        assert!(!result.has_nulls());
    }

    #[test]
    fn test_fillna_float64() {
        let col = float_col_with_nulls(vec![1.0, 0.0, 3.0], vec![false, true, false]);
        let result = fillna_scalar(&col, &ScalarValue::Float64(99.0)).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, 99.0, 3.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_fillna_bool() {
        let col = Column::new_with_nulls(
            "b",
            ColumnData::Bool(vec![true, false, true]),
            vec![false, true, false],
        )
        .unwrap();
        let result = fillna_scalar(&col, &ScalarValue::Bool(false)).unwrap();
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[true, false, true]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_fillna_str() {
        let col = Column::new_with_nulls(
            "s",
            ColumnData::Str(vec!["a".into(), "".into(), "c".into()]),
            vec![false, true, false],
        )
        .unwrap();
        let result = fillna_scalar(&col, &ScalarValue::Str("X".into())).unwrap();
        match &result.data {
            ColumnData::Str(v) => assert_eq!(v, &["a", "X", "c"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_fillna_int64_with_float_promotes() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let result = fillna_scalar(&col, &ScalarValue::Float64(0.5)).unwrap();
        assert_eq!(result.dtype(), DType::Float64);
        match &result.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 0.5);
                assert_eq!(v[2], 3.0);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_fillna_type_mismatch() {
        let col = int_col_with_nulls(vec![1, 99], vec![false, true]);
        assert!(fillna_scalar(&col, &ScalarValue::Str("x".into())).is_err());
    }

    #[test]
    fn test_dropna_any() {
        let col1 = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let col2 = int_col(vec![4, 5, 6]);
        let keep = dropna_rows(&[&col1, &col2], DropHow::Any);
        assert_eq!(keep, vec![0, 2]); // row 1 has null in col1
    }

    #[test]
    fn test_dropna_all() {
        let col1 = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let col2 = int_col_with_nulls(vec![4, 99, 6], vec![false, true, false]);
        let keep = dropna_rows(&[&col1, &col2], DropHow::All);
        assert_eq!(keep, vec![0, 2]); // row 1: both null → dropped
    }

    #[test]
    fn test_dropna_all_partial() {
        // Row 1: col1=null, col2=not null → keep with DropHow::All
        let col1 = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        let col2 = int_col(vec![4, 5, 6]);
        let keep = dropna_rows(&[&col1, &col2], DropHow::All);
        assert_eq!(keep, vec![0, 1, 2]); // row 1 not ALL null → kept
    }

    #[test]
    fn test_dropna_no_nulls() {
        let col = int_col(vec![1, 2, 3]);
        let keep = dropna_rows(&[&col], DropHow::Any);
        assert_eq!(keep, vec![0, 1, 2]);
    }

    #[test]
    fn test_dropna_empty_columns() {
        let keep = dropna_rows(&[], DropHow::Any);
        assert_eq!(keep, vec![]);
    }
}
