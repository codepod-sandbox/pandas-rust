use crate::column::{Column, ColumnData};
use crate::error::{PandasError, Result};

fn merge_null_masks(left: &Column, right: &Column) -> Option<Vec<bool>> {
    match (&left.null_mask, &right.null_mask) {
        (None, None) => None,
        (Some(l), None) => Some(l.clone()),
        (None, Some(r)) => Some(r.clone()),
        (Some(l), Some(r)) => Some(l.iter().zip(r.iter()).map(|(&a, &b)| a || b).collect()),
    }
}

fn check_lengths(left: &Column, right: &Column) -> Result<()> {
    if left.len() != right.len() {
        return Err(PandasError::ValueError(format!(
            "length mismatch: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}

fn binary_op_numeric(
    left: &Column,
    right: &Column,
    op_i64: impl Fn(i64, i64) -> Result<i64>,
    op_f64: impl Fn(f64, f64) -> Result<f64>,
    always_float: bool,
) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let data = match (&left.data, &right.data) {
        (ColumnData::Int64(l), ColumnData::Int64(r)) => {
            if always_float {
                let vals: Result<Vec<f64>> = l
                    .iter()
                    .zip(r.iter())
                    .map(|(&a, &b)| op_f64(a as f64, b as f64))
                    .collect();
                ColumnData::Float64(vals?)
            } else {
                let vals: Result<Vec<i64>> = l
                    .iter()
                    .zip(r.iter())
                    .map(|(&a, &b)| op_i64(a, b))
                    .collect();
                ColumnData::Int64(vals?)
            }
        }
        (ColumnData::Float64(l), ColumnData::Float64(r)) => {
            let vals: Result<Vec<f64>> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| op_f64(a, b))
                .collect();
            ColumnData::Float64(vals?)
        }
        (ColumnData::Int64(l), ColumnData::Float64(r)) => {
            let vals: Result<Vec<f64>> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| op_f64(a as f64, b))
                .collect();
            ColumnData::Float64(vals?)
        }
        (ColumnData::Float64(l), ColumnData::Int64(r)) => {
            let vals: Result<Vec<f64>> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| op_f64(a, b as f64))
                .collect();
            ColumnData::Float64(vals?)
        }
        (ColumnData::Bool(_), _) | (_, ColumnData::Bool(_)) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Bool columns".to_string(),
            ));
        }
        (ColumnData::Str(_), _) | (_, ColumnData::Str(_)) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Str columns".to_string(),
            ));
        }
    };

    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn add(left: &Column, right: &Column) -> Result<Column> {
    binary_op_numeric(left, right, |a, b| Ok(a + b), |a, b| Ok(a + b), false)
}

pub fn sub(left: &Column, right: &Column) -> Result<Column> {
    binary_op_numeric(left, right, |a, b| Ok(a - b), |a, b| Ok(a - b), false)
}

pub fn mul(left: &Column, right: &Column) -> Result<Column> {
    binary_op_numeric(left, right, |a, b| Ok(a * b), |a, b| Ok(a * b), false)
}

pub fn div(left: &Column, right: &Column) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let data = match (&left.data, &right.data) {
        (ColumnData::Int64(l), ColumnData::Int64(r)) => {
            let vals: Result<Vec<f64>> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| {
                    if b == 0 {
                        Err(PandasError::ValueError(
                            "integer division by zero".to_string(),
                        ))
                    } else {
                        Ok(a as f64 / b as f64)
                    }
                })
                .collect();
            ColumnData::Float64(vals?)
        }
        (ColumnData::Float64(l), ColumnData::Float64(r)) => {
            ColumnData::Float64(l.iter().zip(r.iter()).map(|(&a, &b)| a / b).collect())
        }
        (ColumnData::Int64(l), ColumnData::Float64(r)) => ColumnData::Float64(
            l.iter()
                .zip(r.iter())
                .map(|(&a, &b)| a as f64 / b)
                .collect(),
        ),
        (ColumnData::Float64(l), ColumnData::Int64(r)) => ColumnData::Float64(
            l.iter()
                .zip(r.iter())
                .map(|(&a, &b)| a / b as f64)
                .collect(),
        ),
        (ColumnData::Bool(_), _) | (_, ColumnData::Bool(_)) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Bool columns".to_string(),
            ));
        }
        (ColumnData::Str(_), _) | (_, ColumnData::Str(_)) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Str columns".to_string(),
            ));
        }
    };

    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn neg(col: &Column) -> Result<Column> {
    let data = match &col.data {
        ColumnData::Int64(v) => ColumnData::Int64(v.iter().map(|&x| -x).collect()),
        ColumnData::Float64(v) => ColumnData::Float64(v.iter().map(|&x| -x).collect()),
        ColumnData::Bool(_) => {
            return Err(PandasError::TypeError(
                "negation not supported for Bool columns".to_string(),
            ));
        }
        ColumnData::Str(_) => {
            return Err(PandasError::TypeError(
                "negation not supported for Str columns".to_string(),
            ));
        }
    };
    let result = match col.null_mask.clone() {
        Some(mask) => Column::new_with_nulls(col.name.clone(), data, mask)?,
        None => Column::new(col.name.clone(), data),
    };
    Ok(result)
}

fn scalar_op_numeric(col: &Column, scalar: f64, op: impl Fn(f64, f64) -> f64) -> Result<Column> {
    // If the column is Int64 and the scalar is a whole number, preserve Int64 dtype.
    let data = match &col.data {
        ColumnData::Int64(v) => {
            if scalar.fract() == 0.0 && scalar >= i64::MIN as f64 && scalar <= i64::MAX as f64 {
                let scalar_i = scalar as i64;
                ColumnData::Int64(v.iter().map(|&x| op(x as f64, scalar_i as f64) as i64).collect())
            } else {
                ColumnData::Float64(v.iter().map(|&x| op(x as f64, scalar)).collect())
            }
        }
        ColumnData::Float64(v) => ColumnData::Float64(v.iter().map(|&x| op(x, scalar)).collect()),
        ColumnData::Bool(_) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Bool columns".to_string(),
            ));
        }
        ColumnData::Str(_) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Str columns".to_string(),
            ));
        }
    };
    let result = match col.null_mask.clone() {
        Some(mask) => Column::new_with_nulls(col.name.clone(), data, mask)?,
        None => Column::new(col.name.clone(), data),
    };
    Ok(result)
}

pub fn add_scalar(col: &Column, scalar: f64) -> Result<Column> {
    scalar_op_numeric(col, scalar, |a, b| a + b)
}

pub fn sub_scalar(col: &Column, scalar: f64) -> Result<Column> {
    scalar_op_numeric(col, scalar, |a, b| a - b)
}

pub fn mul_scalar(col: &Column, scalar: f64) -> Result<Column> {
    scalar_op_numeric(col, scalar, |a, b| a * b)
}

pub fn div_scalar(col: &Column, scalar: f64) -> Result<Column> {
    // Division always returns float (matching pandas behavior)
    let data = match &col.data {
        ColumnData::Int64(v) => {
            ColumnData::Float64(v.iter().map(|&x| x as f64 / scalar).collect())
        }
        ColumnData::Float64(v) => ColumnData::Float64(v.iter().map(|&x| x / scalar).collect()),
        ColumnData::Bool(_) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Bool columns".to_string(),
            ));
        }
        ColumnData::Str(_) => {
            return Err(PandasError::TypeError(
                "arithmetic not supported for Str columns".to_string(),
            ));
        }
    };
    let result = match col.null_mask.clone() {
        Some(mask) => Column::new_with_nulls(col.name.clone(), data, mask)?,
        None => Column::new(col.name.clone(), data),
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_col(name: &str, vals: Vec<i64>) -> Column {
        Column::new(name, ColumnData::Int64(vals))
    }

    fn float_col(name: &str, vals: Vec<f64>) -> Column {
        Column::new(name, ColumnData::Float64(vals))
    }

    fn str_col(name: &str, vals: Vec<&str>) -> Column {
        Column::new(
            name,
            ColumnData::Str(vals.into_iter().map(String::from).collect()),
        )
    }

    #[test]
    fn test_add_int64() {
        let left = int_col("a", vec![1, 2, 3]);
        let right = int_col("b", vec![4, 5, 6]);
        let result = add(&left, &right).unwrap();
        assert_eq!(result.name(), "a");
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[5, 7, 9]),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_add_float64() {
        let left = float_col("a", vec![1.0, 2.0]);
        let right = float_col("b", vec![3.0, 4.0]);
        let result = add(&left, &right).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[4.0, 6.0]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_add_promotion() {
        let left = int_col("a", vec![1, 2, 3]);
        let right = float_col("b", vec![1.5, 2.5, 3.5]);
        let result = add(&left, &right).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[2.5, 4.5, 6.5]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_add_with_nulls() {
        let left = Column::new_with_nulls(
            "a",
            ColumnData::Int64(vec![1, 2, 3]),
            vec![false, true, false],
        )
        .unwrap();
        let right = int_col("b", vec![10, 20, 30]);
        let result = add(&left, &right).unwrap();
        assert!(!result.is_null(0));
        assert!(result.is_null(1));
        assert!(!result.is_null(2));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 11);
                assert_eq!(v[2], 33);
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_sub() {
        let left = int_col("a", vec![10, 20, 30]);
        let right = int_col("b", vec![1, 2, 3]);
        let result = sub(&left, &right).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[9, 18, 27]),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_mul() {
        let left = int_col("a", vec![2, 3, 4]);
        let right = int_col("b", vec![5, 6, 7]);
        let result = mul(&left, &right).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[10, 18, 28]),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_div_float() {
        let left = float_col("a", vec![10.0, 9.0]);
        let right = float_col("b", vec![2.0, 3.0]);
        let result = div(&left, &right).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                assert!((v[0] - 5.0).abs() < 1e-10);
                assert!((v[1] - 3.0).abs() < 1e-10);
            }
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_div_int_returns_float() {
        let left = int_col("a", vec![10, 9]);
        let right = int_col("b", vec![2, 3]);
        let result = div(&left, &right).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                assert!((v[0] - 5.0).abs() < 1e-10);
                assert!((v[1] - 3.0).abs() < 1e-10);
            }
            _ => panic!("expected Float64 for int division"),
        }
    }

    #[test]
    fn test_div_int_by_zero_error() {
        let left = int_col("a", vec![10]);
        let right = int_col("b", vec![0]);
        assert!(div(&left, &right).is_err());
    }

    #[test]
    fn test_neg() {
        let col = int_col("a", vec![1, -2, 3]);
        let result = neg(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[-1, 2, -3]),
            _ => panic!("expected Int64"),
        }

        let col = float_col("b", vec![1.5, -2.5]);
        let result = neg(&col).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[-1.5, 2.5]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_add_scalar() {
        let col = int_col("a", vec![1, 2, 3]);
        let result = add_scalar(&col, 10.0).unwrap();
        // Int64 + whole-number scalar stays Int64
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[11, 12, 13]),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_add_scalar_float_promotes() {
        let col = int_col("a", vec![1, 2, 3]);
        let result = add_scalar(&col, 0.5).unwrap();
        // Int64 + fractional scalar promotes to Float64
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.5, 2.5, 3.5]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_sub_scalar() {
        let col = float_col("a", vec![10.0, 20.0]);
        let result = sub_scalar(&col, 5.0).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[5.0, 15.0]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_mul_scalar() {
        let col = int_col("a", vec![2, 3]);
        let result = mul_scalar(&col, 3.0).unwrap();
        // Int64 * whole-number scalar stays Int64
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[6, 9]),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_div_scalar() {
        let col = float_col("a", vec![10.0, 20.0]);
        let result = div_scalar(&col, 4.0).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[2.5, 5.0]),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_length_mismatch() {
        let left = int_col("a", vec![1, 2, 3]);
        let right = int_col("b", vec![1, 2]);
        assert!(add(&left, &right).is_err());
    }

    #[test]
    fn test_str_arithmetic_error() {
        let left = str_col("a", vec!["x", "y"]);
        let right = str_col("b", vec!["a", "b"]);
        assert!(add(&left, &right).is_err());
    }
}
