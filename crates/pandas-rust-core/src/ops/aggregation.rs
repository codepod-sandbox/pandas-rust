use crate::column::{Column, ColumnData};
use crate::error::{PandasError, Result};

/// The result of an aggregation operation.
#[derive(Debug, Clone, PartialEq)]
pub enum AggResult {
    Int64(i64),
    Float64(f64),
    Str(String),
    Usize(usize),
    None, // all nulls or empty
}

/// Sum of non-null values. Int64→Int64, Float64→Float64.
pub fn sum(col: &Column) -> Result<AggResult> {
    match &col.data {
        ColumnData::Int64(v) => {
            let mut total: i64 = 0;
            let mut any = false;
            for (i, &val) in v.iter().enumerate() {
                if !col.is_null(i) {
                    total += val;
                    any = true;
                }
            }
            Ok(if any {
                AggResult::Int64(total)
            } else {
                AggResult::None
            })
        }
        ColumnData::Float64(v) => {
            let mut total: f64 = 0.0;
            let mut any = false;
            for (i, &val) in v.iter().enumerate() {
                if !col.is_null(i) {
                    total += val;
                    any = true;
                }
            }
            Ok(if any {
                AggResult::Float64(total)
            } else {
                AggResult::None
            })
        }
        _ => Err(PandasError::TypeError(format!(
            "sum not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Mean of non-null values as Float64.
pub fn mean(col: &Column) -> Result<AggResult> {
    let (sum_val, count) = match &col.data {
        ColumnData::Int64(v) => {
            let mut s = 0.0f64;
            let mut c = 0usize;
            for (i, &val) in v.iter().enumerate() {
                if !col.is_null(i) {
                    s += val as f64;
                    c += 1;
                }
            }
            (s, c)
        }
        ColumnData::Float64(v) => {
            let mut s = 0.0f64;
            let mut c = 0usize;
            for (i, &val) in v.iter().enumerate() {
                if !col.is_null(i) {
                    s += val;
                    c += 1;
                }
            }
            (s, c)
        }
        _ => {
            return Err(PandasError::TypeError(format!(
                "mean not supported for dtype {:?}",
                col.dtype()
            )))
        }
    };
    if count == 0 {
        Ok(AggResult::None)
    } else {
        Ok(AggResult::Float64(sum_val / count as f64))
    }
}

/// Minimum value. Works for Int64, Float64, and Str (lexicographic).
pub fn min(col: &Column) -> Result<AggResult> {
    match &col.data {
        ColumnData::Int64(v) => {
            let min_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, &val)| val)
                .min();
            Ok(min_val.map(AggResult::Int64).unwrap_or(AggResult::None))
        }
        ColumnData::Float64(v) => {
            let min_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, &val)| val)
                .reduce(f64::min);
            Ok(min_val.map(AggResult::Float64).unwrap_or(AggResult::None))
        }
        ColumnData::Str(v) => {
            let min_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, val)| val.clone())
                .min();
            Ok(min_val.map(AggResult::Str).unwrap_or(AggResult::None))
        }
        _ => Err(PandasError::TypeError(format!(
            "min not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Maximum value. Works for Int64, Float64, and Str (lexicographic).
pub fn max(col: &Column) -> Result<AggResult> {
    match &col.data {
        ColumnData::Int64(v) => {
            let max_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, &val)| val)
                .max();
            Ok(max_val.map(AggResult::Int64).unwrap_or(AggResult::None))
        }
        ColumnData::Float64(v) => {
            let max_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, &val)| val)
                .reduce(f64::max);
            Ok(max_val.map(AggResult::Float64).unwrap_or(AggResult::None))
        }
        ColumnData::Str(v) => {
            let max_val = v
                .iter()
                .enumerate()
                .filter(|(i, _)| !col.is_null(*i))
                .map(|(_, val)| val.clone())
                .max();
            Ok(max_val.map(AggResult::Str).unwrap_or(AggResult::None))
        }
        _ => Err(PandasError::TypeError(format!(
            "max not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Count of non-null values. Works for ALL dtypes.
pub fn count(col: &Column) -> AggResult {
    let cnt = (0..col.len()).filter(|&i| !col.is_null(i)).count();
    AggResult::Usize(cnt)
}

/// Variance. ddof=1 for sample variance.
pub fn var(col: &Column, ddof: usize) -> Result<AggResult> {
    let vals = collect_numeric_vals(col)?;
    if vals.len() <= ddof {
        return Ok(AggResult::None);
    }
    let mean_val = vals.iter().sum::<f64>() / vals.len() as f64;
    let variance =
        vals.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / (vals.len() - ddof) as f64;
    Ok(AggResult::Float64(variance))
}

/// Sample standard deviation. ddof=1 by default.
pub fn std(col: &Column, ddof: usize) -> Result<AggResult> {
    match var(col, ddof)? {
        AggResult::Float64(v) => Ok(AggResult::Float64(v.sqrt())),
        AggResult::None => Ok(AggResult::None),
        other => Ok(other),
    }
}

/// Median as Float64. Sort non-null values, take middle.
pub fn median(col: &Column) -> Result<AggResult> {
    let mut vals = collect_numeric_vals(col)?;
    if vals.is_empty() {
        return Ok(AggResult::None);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let med = if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    };
    Ok(AggResult::Float64(med))
}

/// Quantile with linear interpolation. q must be in [0,1].
pub fn quantile(col: &Column, q: f64) -> Result<AggResult> {
    if !(0.0..=1.0).contains(&q) {
        return Err(PandasError::ValueError(format!(
            "quantile q={q} must be in [0, 1]"
        )));
    }
    let mut vals = collect_numeric_vals(col)?;
    if vals.is_empty() {
        return Ok(AggResult::None);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n == 1 {
        return Ok(AggResult::Float64(vals[0]));
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    let result = vals[lo] + frac * (vals[hi] - vals[lo]);
    Ok(AggResult::Float64(result))
}

/// Helper: collect non-null numeric values as f64.
fn collect_numeric_vals(col: &Column) -> Result<Vec<f64>> {
    match &col.data {
        ColumnData::Int64(v) => Ok(v
            .iter()
            .enumerate()
            .filter(|(i, _)| !col.is_null(*i))
            .map(|(_, &val)| val as f64)
            .collect()),
        ColumnData::Float64(v) => Ok(v
            .iter()
            .enumerate()
            .filter(|(i, _)| !col.is_null(*i))
            .filter(|(_, &val)| !val.is_nan())
            .map(|(_, &val)| val)
            .collect()),
        _ => Err(PandasError::TypeError(format!(
            "operation not supported for dtype {:?}",
            col.dtype()
        ))),
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

    fn int_col_with_nulls(vals: Vec<i64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Int64(vals), mask).unwrap()
    }

    fn float_col_with_nulls(vals: Vec<f64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Float64(vals), mask).unwrap()
    }

    #[test]
    fn test_sum_int64() {
        let col = int_col(vec![1, 2, 3, 4]);
        assert_eq!(sum(&col).unwrap(), AggResult::Int64(10));
    }

    #[test]
    fn test_sum_float64() {
        let col = float_col(vec![1.0, 2.0, 3.0]);
        assert_eq!(sum(&col).unwrap(), AggResult::Float64(6.0));
    }

    #[test]
    fn test_sum_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        assert_eq!(sum(&col).unwrap(), AggResult::Int64(4));
    }

    #[test]
    fn test_sum_all_nulls() {
        let col = int_col_with_nulls(vec![1, 2], vec![true, true]);
        assert_eq!(sum(&col).unwrap(), AggResult::None);
    }

    #[test]
    fn test_sum_empty() {
        let col = int_col(vec![]);
        assert_eq!(sum(&col).unwrap(), AggResult::None);
    }

    #[test]
    fn test_sum_type_error() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into()]));
        assert!(sum(&col).is_err());
    }

    #[test]
    fn test_mean_int64() {
        let col = int_col(vec![1, 2, 3]);
        assert_eq!(mean(&col).unwrap(), AggResult::Float64(2.0));
    }

    #[test]
    fn test_mean_float64() {
        let col = float_col(vec![1.0, 3.0]);
        assert_eq!(mean(&col).unwrap(), AggResult::Float64(2.0));
    }

    #[test]
    fn test_mean_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        assert_eq!(mean(&col).unwrap(), AggResult::Float64(2.0));
    }

    #[test]
    fn test_mean_all_nulls() {
        let col = int_col_with_nulls(vec![1], vec![true]);
        assert_eq!(mean(&col).unwrap(), AggResult::None);
    }

    #[test]
    fn test_min_int64() {
        let col = int_col(vec![3, 1, 2]);
        assert_eq!(min(&col).unwrap(), AggResult::Int64(1));
    }

    #[test]
    fn test_max_int64() {
        let col = int_col(vec![3, 1, 2]);
        assert_eq!(max(&col).unwrap(), AggResult::Int64(3));
    }

    #[test]
    fn test_min_float64() {
        let col = float_col(vec![3.0, 1.0, 2.0]);
        assert_eq!(min(&col).unwrap(), AggResult::Float64(1.0));
    }

    #[test]
    fn test_max_float64() {
        let col = float_col(vec![3.0, 1.0, 2.0]);
        assert_eq!(max(&col).unwrap(), AggResult::Float64(3.0));
    }

    #[test]
    fn test_min_with_nulls() {
        let col = int_col_with_nulls(vec![5, 1, 3], vec![false, true, false]);
        assert_eq!(min(&col).unwrap(), AggResult::Int64(3));
    }

    #[test]
    fn test_max_with_nulls() {
        let col = int_col_with_nulls(vec![5, 1, 3], vec![false, true, false]);
        assert_eq!(max(&col).unwrap(), AggResult::Int64(5));
    }

    #[test]
    fn test_min_str() {
        let col = Column::new(
            "x",
            ColumnData::Str(vec!["banana".into(), "apple".into(), "cherry".into()]),
        );
        assert_eq!(min(&col).unwrap(), AggResult::Str("apple".to_string()));
    }

    #[test]
    fn test_max_str() {
        let col = Column::new(
            "x",
            ColumnData::Str(vec!["banana".into(), "apple".into(), "cherry".into()]),
        );
        assert_eq!(max(&col).unwrap(), AggResult::Str("cherry".to_string()));
    }

    #[test]
    fn test_min_empty() {
        let col = int_col(vec![]);
        assert_eq!(min(&col).unwrap(), AggResult::None);
    }

    #[test]
    fn test_count_int64() {
        let col = int_col(vec![1, 2, 3]);
        assert_eq!(count(&col), AggResult::Usize(3));
    }

    #[test]
    fn test_count_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3], vec![false, true, false]);
        assert_eq!(count(&col), AggResult::Usize(2));
    }

    #[test]
    fn test_count_str() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into(), "b".into()]));
        assert_eq!(count(&col), AggResult::Usize(2));
    }

    #[test]
    fn test_count_bool() {
        let col = Column::new("x", ColumnData::Bool(vec![true, false, true]));
        assert_eq!(count(&col), AggResult::Usize(3));
    }

    #[test]
    fn test_count_all_nulls() {
        let col = int_col_with_nulls(vec![1, 2], vec![true, true]);
        assert_eq!(count(&col), AggResult::Usize(0));
    }

    #[test]
    fn test_std_sample() {
        // vals = [2,4,4,4,5,5,7,9], sample std (ddof=1)
        // mean=5, sum_sq_dev=32, var=32/7≈4.571, std≈2.138
        let col = float_col(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        match std(&col, 1).unwrap() {
            AggResult::Float64(v) => assert!((v - (32.0f64 / 7.0).sqrt()).abs() < 1e-10),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_std_not_enough_vals() {
        let col = float_col(vec![1.0]);
        assert_eq!(std(&col, 1).unwrap(), AggResult::None);
    }

    #[test]
    fn test_var_sample() {
        // vals = [2,4,4,4,5,5,7,9], mean=5, sum_sq_dev=32, sample var = 32/7
        let col = float_col(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        match var(&col, 1).unwrap() {
            AggResult::Float64(v) => assert!((v - 32.0 / 7.0).abs() < 1e-10),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_median_odd() {
        let col = int_col(vec![3, 1, 2]);
        assert_eq!(median(&col).unwrap(), AggResult::Float64(2.0));
    }

    #[test]
    fn test_median_even() {
        let col = int_col(vec![1, 2, 3, 4]);
        assert_eq!(median(&col).unwrap(), AggResult::Float64(2.5));
    }

    #[test]
    fn test_median_with_nulls() {
        let col = int_col_with_nulls(vec![1, 99, 3, 5], vec![false, true, false, false]);
        // non-null: [1, 3, 5], median = 3.0
        assert_eq!(median(&col).unwrap(), AggResult::Float64(3.0));
    }

    #[test]
    fn test_median_float64() {
        let col = float_col_with_nulls(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        assert_eq!(median(&col).unwrap(), AggResult::Float64(2.0));
    }

    #[test]
    fn test_quantile_0() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        assert_eq!(quantile(&col, 0.0).unwrap(), AggResult::Float64(1.0));
    }

    #[test]
    fn test_quantile_1() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        assert_eq!(quantile(&col, 1.0).unwrap(), AggResult::Float64(5.0));
    }

    #[test]
    fn test_quantile_025() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        match quantile(&col, 0.25).unwrap() {
            AggResult::Float64(v) => assert!((v - 2.0).abs() < 1e-10),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_quantile_05() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        assert_eq!(quantile(&col, 0.5).unwrap(), AggResult::Float64(3.0));
    }

    #[test]
    fn test_quantile_075() {
        let col = int_col(vec![1, 2, 3, 4, 5]);
        match quantile(&col, 0.75).unwrap() {
            AggResult::Float64(v) => assert!((v - 4.0).abs() < 1e-10),
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_quantile_invalid() {
        let col = int_col(vec![1, 2, 3]);
        assert!(quantile(&col, 1.5).is_err());
        assert!(quantile(&col, -0.1).is_err());
    }

    #[test]
    fn test_quantile_empty() {
        let col = int_col(vec![]);
        assert_eq!(quantile(&col, 0.5).unwrap(), AggResult::None);
    }

    #[test]
    fn test_std_type_error() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into()]));
        assert!(std(&col, 1).is_err());
    }
}
