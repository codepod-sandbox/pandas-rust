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

/// Running cumulative sum. Nulls produce null at that position but don't reset accumulator.
pub fn cumsum(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let mut acc: i64 = 0;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0i64);
                    any_null = true;
                } else {
                    acc += x;
                    result.push(acc);
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Int64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(result)))
            }
        }
        ColumnData::Float64(v) => {
            let mut acc: f64 = 0.0;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0.0f64);
                    any_null = true;
                } else {
                    acc += x;
                    result.push(acc);
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(result)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "cumsum not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Running cumulative product. Nulls produce null at that position but don't reset accumulator.
pub fn cumprod(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let mut acc: i64 = 1;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0i64);
                    any_null = true;
                } else {
                    acc *= x;
                    result.push(acc);
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Int64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(result)))
            }
        }
        ColumnData::Float64(v) => {
            let mut acc: f64 = 1.0;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0.0f64);
                    any_null = true;
                } else {
                    acc *= x;
                    result.push(acc);
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(result)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "cumprod not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Running cumulative maximum. Nulls produce null at that position but don't reset accumulator.
pub fn cummax(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let mut acc: Option<i64> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0i64);
                    any_null = true;
                } else {
                    acc = Some(acc.map_or(x, |a| a.max(x)));
                    result.push(acc.unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Int64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(result)))
            }
        }
        ColumnData::Float64(v) => {
            let mut acc: Option<f64> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0.0f64);
                    any_null = true;
                } else {
                    acc = Some(acc.map_or(x, |a| if x > a { x } else { a }));
                    result.push(acc.unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(result)))
            }
        }
        ColumnData::Str(v) => {
            let mut acc: Option<String> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(String::new());
                    any_null = true;
                } else {
                    acc = Some(acc.as_ref().map_or_else(|| x.clone(), |a| if x > a { x.clone() } else { a.clone() }));
                    result.push(acc.clone().unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Str(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Str(result)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "cummax not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Running cumulative minimum. Nulls produce null at that position but don't reset accumulator.
pub fn cummin(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let mut acc: Option<i64> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0i64);
                    any_null = true;
                } else {
                    acc = Some(acc.map_or(x, |a| a.min(x)));
                    result.push(acc.unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Int64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(result)))
            }
        }
        ColumnData::Float64(v) => {
            let mut acc: Option<f64> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, &x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(0.0f64);
                    any_null = true;
                } else {
                    acc = Some(acc.map_or(x, |a| if x < a { x } else { a }));
                    result.push(acc.unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(result)))
            }
        }
        ColumnData::Str(v) => {
            let mut acc: Option<String> = None;
            let mut result = Vec::with_capacity(v.len());
            let mut null_mask = Vec::with_capacity(v.len());
            let mut any_null = false;
            for (i, x) in v.iter().enumerate() {
                if col.is_null(i) {
                    null_mask.push(true);
                    result.push(String::new());
                    any_null = true;
                } else {
                    acc = Some(acc.as_ref().map_or_else(|| x.clone(), |a| if x < a { x.clone() } else { a.clone() }));
                    result.push(acc.clone().unwrap());
                    null_mask.push(false);
                }
            }
            if any_null {
                Column::new_with_nulls(col.name(), ColumnData::Str(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Str(result)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "cummin not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Shift column elements by `periods` positions.
/// periods > 0: shift down (first `periods` positions become null)
/// periods < 0: shift up (last `|periods|` positions become null)
/// periods == 0: return a clone
pub fn shift_column(col: &Column, periods: i64) -> Column {
    let n = col.len();
    if n == 0 || periods == 0 {
        return col.clone();
    }

    let abs_periods = periods.unsigned_abs() as usize;
    if abs_periods >= n {
        // All values become null
        let null_mask = vec![true; n];
        return match &col.data {
            ColumnData::Int64(v) => {
                Column::new_with_nulls(col.name(), ColumnData::Int64(vec![0i64; v.len()]), null_mask)
                    .expect("valid shift null mask")
            }
            ColumnData::Float64(v) => {
                Column::new_with_nulls(col.name(), ColumnData::Float64(vec![0.0f64; v.len()]), null_mask)
                    .expect("valid shift null mask")
            }
            ColumnData::Str(v) => {
                Column::new_with_nulls(col.name(), ColumnData::Str(vec![String::new(); v.len()]), null_mask)
                    .expect("valid shift null mask")
            }
            ColumnData::Bool(v) => {
                Column::new_with_nulls(col.name(), ColumnData::Bool(vec![false; v.len()]), null_mask)
                    .expect("valid shift null mask")
            }
        };
    }

    let mut null_mask = vec![false; n];

    if periods > 0 {
        // Shift down: result[i] = col[i - periods] for i >= periods, null for i < periods
        for i in 0..abs_periods {
            null_mask[i] = true;
        }
        // Also carry over existing nulls shifted down
        for i in abs_periods..n {
            if col.is_null(i - abs_periods) {
                null_mask[i] = true;
            }
        }
    } else {
        // Shift up: result[i] = col[i + |periods|] for i < n - |periods|, null for i >= n - |periods|
        for i in (n - abs_periods)..n {
            null_mask[i] = true;
        }
        for i in 0..(n - abs_periods) {
            if col.is_null(i + abs_periods) {
                null_mask[i] = true;
            }
        }
    }

    let any_null = null_mask.iter().any(|&b| b);

    let data = match &col.data {
        ColumnData::Int64(v) => {
            let mut new_v = vec![0i64; n];
            if periods > 0 {
                for i in abs_periods..n {
                    new_v[i] = v[i - abs_periods];
                }
            } else {
                for i in 0..(n - abs_periods) {
                    new_v[i] = v[i + abs_periods];
                }
            }
            ColumnData::Int64(new_v)
        }
        ColumnData::Float64(v) => {
            let mut new_v = vec![0.0f64; n];
            if periods > 0 {
                for i in abs_periods..n {
                    new_v[i] = v[i - abs_periods];
                }
            } else {
                for i in 0..(n - abs_periods) {
                    new_v[i] = v[i + abs_periods];
                }
            }
            ColumnData::Float64(new_v)
        }
        ColumnData::Str(v) => {
            let mut new_v = vec![String::new(); n];
            if periods > 0 {
                for i in abs_periods..n {
                    new_v[i] = v[i - abs_periods].clone();
                }
            } else {
                for i in 0..(n - abs_periods) {
                    new_v[i] = v[i + abs_periods].clone();
                }
            }
            ColumnData::Str(new_v)
        }
        ColumnData::Bool(v) => {
            let mut new_v = vec![false; n];
            if periods > 0 {
                for i in abs_periods..n {
                    new_v[i] = v[i - abs_periods];
                }
            } else {
                for i in 0..(n - abs_periods) {
                    new_v[i] = v[i + abs_periods];
                }
            }
            ColumnData::Bool(new_v)
        }
    };

    if any_null {
        Column::new_with_nulls(col.name(), data, null_mask).expect("valid shift null mask")
    } else {
        Column::new(col.name(), data)
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

    #[test]
    fn test_cumsum_int64() {
        let col = int_col(vec![1, 2, 3, 4]);
        let result = cumsum(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 3, 6, 10]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_cumsum_float64() {
        let col = float_col(vec![1.0, 2.0, 3.0]);
        let result = cumsum(&col).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, 3.0, 6.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_cumprod_int64() {
        let col = int_col(vec![1, 2, 3, 4]);
        let result = cumprod(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 6, 24]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_cummax_int64() {
        let col = int_col(vec![1, 5, 3, 7, 2]);
        let result = cummax(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 5, 5, 7, 7]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_cummin_int64() {
        let col = int_col(vec![5, 3, 7, 1, 4]);
        let result = cummin(&col).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[5, 3, 3, 1, 1]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_cumsum_error_str() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into()]));
        assert!(cumsum(&col).is_err());
    }

    #[test]
    fn test_shift_down_int64() {
        let col = int_col(vec![1, 2, 3, 4]);
        let result = shift_column(&col, 1);
        assert!(result.is_null(0));
        assert!(!result.is_null(1));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[1], 1);
                assert_eq!(v[2], 2);
                assert_eq!(v[3], 3);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_shift_up_int64() {
        let col = int_col(vec![1, 2, 3, 4]);
        let result = shift_column(&col, -1);
        assert!(!result.is_null(0));
        assert!(result.is_null(3));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 2);
                assert_eq!(v[1], 3);
                assert_eq!(v[2], 4);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_shift_zero() {
        let col = int_col(vec![1, 2, 3]);
        let result = shift_column(&col, 0);
        assert!(!result.is_null(0));
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_shift_by_two() {
        let col = int_col(vec![1, 2, 3, 4]);
        let result = shift_column(&col, 2);
        assert!(result.is_null(0));
        assert!(result.is_null(1));
        assert!(!result.is_null(2));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[2], 1);
                assert_eq!(v[3], 2);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_shift_float64() {
        let col = float_col(vec![1.0, 2.0, 3.0]);
        let result = shift_column(&col, 1);
        assert!(result.is_null(0));
        match &result.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[1], 1.0);
                assert_eq!(v[2], 2.0);
            }
            _ => panic!("wrong dtype"),
        }
    }
}
