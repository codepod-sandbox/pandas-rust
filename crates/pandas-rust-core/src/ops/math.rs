use crate::column::{Column, ColumnData};
use crate::error::{PandasError, Result};
use crate::ops::nulls::ScalarValue;

/// Absolute value for Int64 and Float64 columns. Null-aware (nulls pass through).
pub fn abs_column(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let data: Vec<i64> = v.iter().map(|&x| x.abs()).collect();
            if let Some(mask) = &col.null_mask {
                Column::new_with_nulls(col.name(), ColumnData::Int64(data), mask.clone())
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(data)))
            }
        }
        ColumnData::Float64(v) => {
            let data: Vec<f64> = v.iter().map(|&x| x.abs()).collect();
            if let Some(mask) = &col.null_mask {
                Column::new_with_nulls(col.name(), ColumnData::Float64(data), mask.clone())
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(data)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "abs not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Clamp values to [lower, upper]. Null-aware (nulls pass through).
pub fn clip_column(col: &Column, lower: Option<f64>, upper: Option<f64>) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let lo = lower.map(|x| x as i64);
            let hi = upper.map(|x| x as i64);
            let data: Vec<i64> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if col.is_null(i) {
                        return x;
                    }
                    let mut val = x;
                    if let Some(l) = lo {
                        val = val.max(l);
                    }
                    if let Some(h) = hi {
                        val = val.min(h);
                    }
                    val
                })
                .collect();
            if let Some(mask) = &col.null_mask {
                Column::new_with_nulls(col.name(), ColumnData::Int64(data), mask.clone())
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(data)))
            }
        }
        ColumnData::Float64(v) => {
            let data: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if col.is_null(i) {
                        return x;
                    }
                    let mut val = x;
                    if let Some(l) = lower {
                        if val < l {
                            val = l;
                        }
                    }
                    if let Some(h) = upper {
                        if val > h {
                            val = h;
                        }
                    }
                    val
                })
                .collect();
            if let Some(mask) = &col.null_mask {
                Column::new_with_nulls(col.name(), ColumnData::Float64(data), mask.clone())
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(data)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "clip not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Returns Bool column: true where value is in the given set.
pub fn isin_column(col: &Column, values: &[ScalarValue]) -> Column {
    let result: Vec<bool> = (0..col.len())
        .map(|i| {
            if col.is_null(i) {
                return false;
            }
            values.iter().any(|sv| match (&col.data, sv) {
                (ColumnData::Int64(v), ScalarValue::Int64(target)) => v[i] == *target,
                (ColumnData::Int64(v), ScalarValue::Float64(target)) => v[i] as f64 == *target,
                (ColumnData::Float64(v), ScalarValue::Float64(target)) => v[i] == *target,
                (ColumnData::Float64(v), ScalarValue::Int64(target)) => v[i] == *target as f64,
                (ColumnData::Str(v), ScalarValue::Str(target)) => v[i] == *target,
                (ColumnData::Bool(v), ScalarValue::Bool(target)) => v[i] == *target,
                _ => false,
            })
        })
        .collect();
    Column::new(col.name(), ColumnData::Bool(result))
}

/// Returns true if any non-null value is truthy.
/// For Bool: any true. For Int64/Float64: any nonzero.
pub fn any_col(col: &Column) -> bool {
    match &col.data {
        ColumnData::Bool(v) => v.iter().enumerate().any(|(i, &b)| !col.is_null(i) && b),
        ColumnData::Int64(v) => v
            .iter()
            .enumerate()
            .any(|(i, &x)| !col.is_null(i) && x != 0),
        ColumnData::Float64(v) => v
            .iter()
            .enumerate()
            .any(|(i, &x)| !col.is_null(i) && x != 0.0),
        _ => false,
    }
}

/// Returns true if all non-null values are truthy.
/// For Bool: all true. For Int64/Float64: all nonzero.
pub fn all_col(col: &Column) -> bool {
    let non_null_count = (0..col.len()).filter(|&i| !col.is_null(i)).count();
    if non_null_count == 0 {
        return true; // vacuous truth
    }
    match &col.data {
        ColumnData::Bool(v) => v
            .iter()
            .enumerate()
            .filter(|(i, _)| !col.is_null(*i))
            .all(|(_, &b)| b),
        ColumnData::Int64(v) => v
            .iter()
            .enumerate()
            .filter(|(i, _)| !col.is_null(*i))
            .all(|(_, &x)| x != 0),
        ColumnData::Float64(v) => v
            .iter()
            .enumerate()
            .filter(|(i, _)| !col.is_null(*i))
            .all(|(_, &x)| x != 0.0),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn int_col(vals: Vec<i64>) -> Column {
        Column::new("x", ColumnData::Int64(vals))
    }

    fn float_col(vals: Vec<f64>) -> Column {
        Column::new("x", ColumnData::Float64(vals))
    }

    #[test]
    fn test_abs_int64() {
        let col = int_col(vec![-3, 0, 5, -1]);
        let result = abs_column(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[3, 0, 5, 1]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_abs_float64() {
        let col = float_col(vec![-1.5, 0.0, 2.5]);
        let result = abs_column(&col).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.5, 0.0, 2.5]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_abs_str_error() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into()]));
        assert!(abs_column(&col).is_err());
    }

    #[test]
    fn test_clip_int64() {
        let col = int_col(vec![1, 5, 10, -3, 8]);
        let result = clip_column(&col, Some(2.0), Some(7.0)).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[2, 5, 7, 2, 7]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_clip_float64() {
        let col = float_col(vec![0.5, 3.0, 5.5]);
        let result = clip_column(&col, Some(1.0), Some(5.0)).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, 3.0, 5.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_isin_int64() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        let values = vec![ScalarValue::Int64(2), ScalarValue::Int64(4)];
        let result = isin_column(&col, &values);
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[false, true, false, true, false]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_any_bool_true() {
        let col = Column::new("x", ColumnData::Bool(vec![false, true, false]));
        assert!(any_col(&col));
    }

    #[test]
    fn test_any_bool_false() {
        let col = Column::new("x", ColumnData::Bool(vec![false, false]));
        assert!(!any_col(&col));
    }

    #[test]
    fn test_all_bool_true() {
        let col = Column::new("x", ColumnData::Bool(vec![true, true, true]));
        assert!(all_col(&col));
    }

    #[test]
    fn test_all_bool_false() {
        let col = Column::new("x", ColumnData::Bool(vec![true, false, true]));
        assert!(!all_col(&col));
    }
}
