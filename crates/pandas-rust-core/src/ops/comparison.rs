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

fn binary_cmp_any(
    left: &Column,
    right: &Column,
    op_i64: impl Fn(i64, i64) -> bool,
    op_f64: impl Fn(f64, f64) -> bool,
    op_str: Option<impl Fn(&str, &str) -> bool>,
    op_bool: Option<impl Fn(bool, bool) -> bool>,
) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let bools: Vec<bool> = match (&left.data, &right.data) {
        (ColumnData::Int64(l), ColumnData::Int64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| op_i64(a, b))
            .collect(),
        (ColumnData::Float64(l), ColumnData::Float64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| op_f64(a, b))
            .collect(),
        (ColumnData::Int64(l), ColumnData::Float64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| op_f64(a as f64, b))
            .collect(),
        (ColumnData::Float64(l), ColumnData::Int64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| op_f64(a, b as f64))
            .collect(),
        (ColumnData::Str(l), ColumnData::Str(r)) => match &op_str {
            Some(f) => l
                .iter()
                .zip(r.iter())
                .map(|(a, b)| f(a.as_str(), b.as_str()))
                .collect(),
            None => {
                return Err(PandasError::TypeError(
                    "ordering comparison not supported for Str columns".to_string(),
                ));
            }
        },
        (ColumnData::Bool(l), ColumnData::Bool(r)) => match &op_bool {
            Some(f) => l.iter().zip(r.iter()).map(|(&a, &b)| f(a, b)).collect(),
            None => {
                return Err(PandasError::TypeError(
                    "ordering comparison not supported for Bool columns".to_string(),
                ));
            }
        },
        _ => {
            return Err(PandasError::TypeError(
                "mixed-type comparison not supported".to_string(),
            ));
        }
    };

    let data = ColumnData::Bool(bools);
    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn eq(left: &Column, right: &Column) -> Result<Column> {
    binary_cmp_any(
        left,
        right,
        |a, b| a == b,
        |a, b| a == b,
        Some(|a: &str, b: &str| a == b),
        Some(|a: bool, b: bool| a == b),
    )
}

pub fn ne(left: &Column, right: &Column) -> Result<Column> {
    binary_cmp_any(
        left,
        right,
        |a, b| a != b,
        |a, b| a != b,
        Some(|a: &str, b: &str| a != b),
        Some(|a: bool, b: bool| a != b),
    )
}

pub fn lt(left: &Column, right: &Column) -> Result<Column> {
    // For strings, support lexicographic ordering; for bools, reject
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let bools: Vec<bool> = match (&left.data, &right.data) {
        (ColumnData::Int64(l), ColumnData::Int64(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a < b).collect()
        }
        (ColumnData::Float64(l), ColumnData::Float64(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a < b).collect()
        }
        (ColumnData::Int64(l), ColumnData::Float64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| (a as f64) < b)
            .collect(),
        (ColumnData::Float64(l), ColumnData::Int64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| a < b as f64)
            .collect(),
        (ColumnData::Str(l), ColumnData::Str(r)) => {
            l.iter().zip(r.iter()).map(|(a, b)| a < b).collect()
        }
        (ColumnData::Bool(_), _) | (_, ColumnData::Bool(_)) => {
            return Err(PandasError::TypeError(
                "ordering comparison not supported for Bool columns".to_string(),
            ));
        }
        _ => {
            return Err(PandasError::TypeError(
                "mixed-type comparison not supported".to_string(),
            ));
        }
    };

    let data = ColumnData::Bool(bools);
    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn le(left: &Column, right: &Column) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let bools: Vec<bool> = match (&left.data, &right.data) {
        (ColumnData::Int64(l), ColumnData::Int64(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a <= b).collect()
        }
        (ColumnData::Float64(l), ColumnData::Float64(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a <= b).collect()
        }
        (ColumnData::Int64(l), ColumnData::Float64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| (a as f64) <= b)
            .collect(),
        (ColumnData::Float64(l), ColumnData::Int64(r)) => l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| a <= b as f64)
            .collect(),
        (ColumnData::Str(l), ColumnData::Str(r)) => {
            l.iter().zip(r.iter()).map(|(a, b)| a <= b).collect()
        }
        (ColumnData::Bool(_), _) | (_, ColumnData::Bool(_)) => {
            return Err(PandasError::TypeError(
                "ordering comparison not supported for Bool columns".to_string(),
            ));
        }
        _ => {
            return Err(PandasError::TypeError(
                "mixed-type comparison not supported".to_string(),
            ));
        }
    };

    let data = ColumnData::Bool(bools);
    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn gt(left: &Column, right: &Column) -> Result<Column> {
    lt(right, left).map(|c| c.with_name(left.name.clone()))
}

pub fn ge(left: &Column, right: &Column) -> Result<Column> {
    le(right, left).map(|c| c.with_name(left.name.clone()))
}

pub fn eq_scalar(col: &Column, scalar: f64) -> Result<Column> {
    let bools: Vec<bool> = match &col.data {
        ColumnData::Int64(v) => v.iter().map(|&x| (x as f64) == scalar).collect(),
        ColumnData::Float64(v) => v.iter().map(|&x| x == scalar).collect(),
        ColumnData::Bool(_) => {
            return Err(PandasError::TypeError(
                "scalar comparison not supported for Bool columns".to_string(),
            ));
        }
        ColumnData::Str(_) => {
            return Err(PandasError::TypeError(
                "scalar comparison not supported for Str columns".to_string(),
            ));
        }
    };
    let data = ColumnData::Bool(bools);
    let result = match col.null_mask.clone() {
        Some(mask) => Column::new_with_nulls(col.name.clone(), data, mask)?,
        None => Column::new(col.name.clone(), data),
    };
    Ok(result)
}

pub fn and(left: &Column, right: &Column) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let bools: Vec<bool> = match (&left.data, &right.data) {
        (ColumnData::Bool(l), ColumnData::Bool(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a && b).collect()
        }
        _ => {
            return Err(PandasError::TypeError(
                "and operation only supported for Bool columns".to_string(),
            ));
        }
    };

    let data = ColumnData::Bool(bools);
    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn or(left: &Column, right: &Column) -> Result<Column> {
    check_lengths(left, right)?;
    let null_mask = merge_null_masks(left, right);

    let bools: Vec<bool> = match (&left.data, &right.data) {
        (ColumnData::Bool(l), ColumnData::Bool(r)) => {
            l.iter().zip(r.iter()).map(|(&a, &b)| a || b).collect()
        }
        _ => {
            return Err(PandasError::TypeError(
                "or operation only supported for Bool columns".to_string(),
            ));
        }
    };

    let data = ColumnData::Bool(bools);
    let col = match null_mask {
        Some(mask) => Column::new_with_nulls(left.name.clone(), data, mask)?,
        None => Column::new(left.name.clone(), data),
    };
    Ok(col)
}

pub fn not(col: &Column) -> Result<Column> {
    let bools: Vec<bool> = match &col.data {
        ColumnData::Bool(v) => v.iter().map(|&b| !b).collect(),
        _ => {
            return Err(PandasError::TypeError(
                "not operation only supported for Bool columns".to_string(),
            ));
        }
    };
    let data = ColumnData::Bool(bools);
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

    fn bool_col(name: &str, vals: Vec<bool>) -> Column {
        Column::new(name, ColumnData::Bool(vals))
    }

    fn str_col(name: &str, vals: Vec<&str>) -> Column {
        Column::new(
            name,
            ColumnData::Str(vals.into_iter().map(String::from).collect()),
        )
    }

    fn extract_bools(col: &Column) -> Vec<bool> {
        match &col.data {
            ColumnData::Bool(v) => v.clone(),
            _ => panic!("expected Bool column"),
        }
    }

    #[test]
    fn test_int64_comparisons() {
        let a = int_col("a", vec![1, 2, 3, 4, 5]);
        let b = int_col("b", vec![3, 2, 1, 4, 6]);

        assert_eq!(
            extract_bools(&eq(&a, &b).unwrap()),
            vec![false, true, false, true, false]
        );
        assert_eq!(
            extract_bools(&ne(&a, &b).unwrap()),
            vec![true, false, true, false, true]
        );
        assert_eq!(
            extract_bools(&lt(&a, &b).unwrap()),
            vec![true, false, false, false, true]
        );
        assert_eq!(
            extract_bools(&le(&a, &b).unwrap()),
            vec![true, true, false, true, true]
        );
        assert_eq!(
            extract_bools(&gt(&a, &b).unwrap()),
            vec![false, false, true, false, false]
        );
        assert_eq!(
            extract_bools(&ge(&a, &b).unwrap()),
            vec![false, true, true, true, false]
        );
    }

    #[test]
    fn test_float64_comparisons() {
        let a = float_col("a", vec![1.0, 2.0, 3.0]);
        let b = float_col("b", vec![2.0, 2.0, 1.0]);

        assert_eq!(
            extract_bools(&lt(&a, &b).unwrap()),
            vec![true, false, false]
        );
        assert_eq!(extract_bools(&ge(&a, &b).unwrap()), vec![false, true, true]);
    }

    #[test]
    fn test_cross_type_int_float() {
        let a = int_col("a", vec![1, 2, 3]);
        let b = float_col("b", vec![1.5, 2.0, 2.5]);

        assert_eq!(
            extract_bools(&lt(&a, &b).unwrap()),
            vec![true, false, false]
        );
        assert_eq!(
            extract_bools(&eq(&a, &b).unwrap()),
            vec![false, true, false]
        );
    }

    #[test]
    fn test_string_eq_ne() {
        let a = str_col("a", vec!["apple", "banana", "cherry"]);
        let b = str_col("b", vec!["apple", "mango", "cherry"]);

        assert_eq!(extract_bools(&eq(&a, &b).unwrap()), vec![true, false, true]);
        assert_eq!(
            extract_bools(&ne(&a, &b).unwrap()),
            vec![false, true, false]
        );
    }

    #[test]
    fn test_string_ordering() {
        let a = str_col("a", vec!["apple", "banana"]);
        let b = str_col("b", vec!["banana", "apple"]);

        assert_eq!(extract_bools(&lt(&a, &b).unwrap()), vec![true, false]);
        assert_eq!(extract_bools(&gt(&a, &b).unwrap()), vec![false, true]);
    }

    #[test]
    fn test_null_propagation() {
        let a = Column::new_with_nulls(
            "a",
            ColumnData::Int64(vec![1, 2, 3]),
            vec![false, true, false],
        )
        .unwrap();
        let b = int_col("b", vec![1, 2, 3]);

        let result = eq(&a, &b).unwrap();
        assert!(!result.is_null(0));
        assert!(result.is_null(1));
        assert!(!result.is_null(2));
    }

    #[test]
    fn test_bool_and() {
        let a = bool_col("a", vec![true, true, false, false]);
        let b = bool_col("b", vec![true, false, true, false]);
        assert_eq!(
            extract_bools(&and(&a, &b).unwrap()),
            vec![true, false, false, false]
        );
    }

    #[test]
    fn test_bool_or() {
        let a = bool_col("a", vec![true, true, false, false]);
        let b = bool_col("b", vec![true, false, true, false]);
        assert_eq!(
            extract_bools(&or(&a, &b).unwrap()),
            vec![true, true, true, false]
        );
    }

    #[test]
    fn test_bool_not() {
        let a = bool_col("a", vec![true, false, true]);
        assert_eq!(extract_bools(&not(&a).unwrap()), vec![false, true, false]);
    }

    #[test]
    fn test_length_mismatch() {
        let a = int_col("a", vec![1, 2, 3]);
        let b = int_col("b", vec![1, 2]);
        assert!(eq(&a, &b).is_err());
    }

    #[test]
    fn test_wrong_type_for_and() {
        let a = int_col("a", vec![1, 2]);
        let b = int_col("b", vec![3, 4]);
        assert!(and(&a, &b).is_err());
    }

    #[test]
    fn test_wrong_type_for_or() {
        let a = float_col("a", vec![1.0]);
        let b = float_col("b", vec![2.0]);
        assert!(or(&a, &b).is_err());
    }

    #[test]
    fn test_bool_ordering_error() {
        let a = bool_col("a", vec![true, false]);
        let b = bool_col("b", vec![false, true]);
        assert!(lt(&a, &b).is_err());
    }

    #[test]
    fn test_result_name_is_left() {
        let a = int_col("left_name", vec![1, 2]);
        let b = int_col("right_name", vec![1, 2]);
        let result = eq(&a, &b).unwrap();
        assert_eq!(result.name(), "left_name");
    }

    #[test]
    fn test_eq_scalar() {
        let col = int_col("a", vec![1, 2, 3]);
        let result = eq_scalar(&col, 2.0).unwrap();
        assert_eq!(extract_bools(&result), vec![false, true, false]);
    }

    #[test]
    fn test_bool_eq_ne() {
        let a = bool_col("a", vec![true, false, true]);
        let b = bool_col("b", vec![true, true, false]);
        assert_eq!(
            extract_bools(&eq(&a, &b).unwrap()),
            vec![true, false, false]
        );
        assert_eq!(extract_bools(&ne(&a, &b).unwrap()), vec![false, true, true]);
    }
}
