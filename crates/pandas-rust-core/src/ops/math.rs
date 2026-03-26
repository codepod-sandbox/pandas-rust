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

/// Difference between consecutive elements: result[i] = col[i] - col[i - periods].
/// First `periods` elements (or last `|periods|` for negative) become null.
/// Works for Int64 (result stays Int64) and Float64. TypeError for Str/Bool.
pub fn diff_column(col: &Column, periods: i64) -> Result<Column> {
    let n = col.len();
    let abs_p = periods.unsigned_abs() as usize;

    match &col.data {
        ColumnData::Int64(v) => {
            let mut result = vec![0i64; n];
            let mut null_mask = vec![false; n];
            if periods >= 0 {
                let p = abs_p;
                for i in 0..p.min(n) { null_mask[i] = true; }
                for i in p..n {
                    if col.is_null(i) || col.is_null(i - p) {
                        null_mask[i] = true;
                    } else {
                        result[i] = v[i] - v[i - p];
                    }
                }
            } else {
                let p = abs_p;
                for i in n.saturating_sub(p)..n { null_mask[i] = true; }
                for i in 0..(n.saturating_sub(p)) {
                    if col.is_null(i) || col.is_null(i + p) {
                        null_mask[i] = true;
                    } else {
                        result[i] = v[i + p] - v[i];
                    }
                }
            }
            if null_mask.iter().any(|&b| b) {
                Column::new_with_nulls(col.name(), ColumnData::Int64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Int64(result)))
            }
        }
        ColumnData::Float64(v) => {
            let mut result = vec![0.0f64; n];
            let mut null_mask = vec![false; n];
            if periods >= 0 {
                let p = abs_p;
                for i in 0..p.min(n) { null_mask[i] = true; }
                for i in p..n {
                    if col.is_null(i) || col.is_null(i - p) {
                        null_mask[i] = true;
                    } else {
                        result[i] = v[i] - v[i - p];
                    }
                }
            } else {
                let p = abs_p;
                for i in n.saturating_sub(p)..n { null_mask[i] = true; }
                for i in 0..(n.saturating_sub(p)) {
                    if col.is_null(i) || col.is_null(i + p) {
                        null_mask[i] = true;
                    } else {
                        result[i] = v[i + p] - v[i];
                    }
                }
            }
            if null_mask.iter().any(|&b| b) {
                Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(result)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "diff not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Round Float64 values to `decimals` decimal places. Int64 is a no-op. Null-aware.
pub fn round_column(col: &Column, decimals: i32) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(_) => {
            // Int64 is already whole numbers — return a clone
            Ok(col.clone())
        }
        ColumnData::Float64(v) => {
            let factor = 10f64.powi(decimals);
            let data: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if col.is_null(i) { x } else { (x * factor).round() / factor }
                })
                .collect();
            if let Some(mask) = &col.null_mask {
                Column::new_with_nulls(col.name(), ColumnData::Float64(data), mask.clone())
            } else {
                Ok(Column::new(col.name(), ColumnData::Float64(data)))
            }
        }
        _ => Err(PandasError::TypeError(format!(
            "round not supported for dtype {:?}",
            col.dtype()
        ))),
    }
}

/// Rank method enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RankMethod {
    Average,
    Min,
    Max,
    First,
    Dense,
}

/// Rank values in a column. Null values receive null rank.
/// Returns a Float64 column (ranks can be fractional with Average method).
pub fn rank_column(col: &Column, method: RankMethod, ascending: bool) -> Result<Column> {
    let n = col.len();
    let mut result = vec![0.0f64; n];
    let mut null_mask = vec![false; n];

    // Collect non-null indices and their values as f64 for sorting
    let mut items: Vec<(usize, f64)> = Vec::new();
    for i in 0..n {
        if col.is_null(i) {
            null_mask[i] = true;
        } else {
            let val = match &col.data {
                ColumnData::Int64(v) => v[i] as f64,
                ColumnData::Float64(v) => v[i],
                ColumnData::Bool(v) => if v[i] { 1.0 } else { 0.0 },
                ColumnData::Str(_) => {
                    return Err(PandasError::TypeError(
                        "rank not supported for Str dtype".to_string(),
                    ))
                }
            };
            items.push((i, val));
        }
    }

    if items.is_empty() {
        return Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask);
    }

    // Sort by value (then by original index for First method stability)
    items.sort_by(|a, b| {
        let ord = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        if !ascending { ord.reverse() } else { ord }.then(a.0.cmp(&b.0))
    });

    let m = items.len();
    // Assign ranks (1-based)
    let mut i = 0;
    while i < m {
        // Find end of group with same value
        let val = items[i].1;
        let mut j = i + 1;
        while j < m && (items[j].1 - val).abs() < f64::EPSILON * val.abs().max(1.0) {
            j += 1;
        }
        // items[i..j] is a group of equal values
        let group_size = j - i;
        let rank_start = (i + 1) as f64; // 1-based
        let rank_end = (j) as f64;

        match method {
            RankMethod::Average => {
                let avg = (rank_start + rank_end) / 2.0;
                for k in i..j {
                    result[items[k].0] = avg;
                }
            }
            RankMethod::Min => {
                for k in i..j {
                    result[items[k].0] = rank_start;
                }
            }
            RankMethod::Max => {
                for k in i..j {
                    result[items[k].0] = rank_end;
                }
            }
            RankMethod::First => {
                // Already stable by original index in sort
                for k in i..j {
                    result[items[k].0] = (i + 1 + (k - i)) as f64;
                }
            }
            RankMethod::Dense => {
                // Dense ranks: same value gets same rank, next unique value gets +1
                // We'll compute dense ranks in a second pass; for now use placeholder
                // (handled below via second pass)
                for k in i..j {
                    result[items[k].0] = rank_start; // will be overwritten
                }
            }
        }
        i = j;
        let _ = group_size;
    }

    // For Dense method, do a proper pass
    if method == RankMethod::Dense {
        let mut dense_rank = 1usize;
        let mut prev_val: Option<f64> = None;
        // Re-sort by value ascending (or descending) — items is already sorted
        for k in 0..m {
            let val = items[k].1;
            if let Some(pv) = prev_val {
                if (val - pv).abs() > f64::EPSILON * pv.abs().max(1.0) {
                    dense_rank += 1;
                }
            }
            result[items[k].0] = dense_rank as f64;
            prev_val = Some(val);
        }
    }

    let any_null = null_mask.iter().any(|&b| b);
    if any_null {
        Column::new_with_nulls(col.name(), ColumnData::Float64(result), null_mask)
    } else {
        Ok(Column::new(col.name(), ColumnData::Float64(result)))
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

    // --- diff tests ---

    #[test]
    fn test_diff_int64_periods1() {
        let col = int_col(vec![1, 3, 6, 10]);
        let result = diff_column(&col, 1).unwrap();
        assert!(result.is_null(0));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[1], 2);
                assert_eq!(v[2], 3);
                assert_eq!(v[3], 4);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_diff_int64_periods2() {
        let col = int_col(vec![1, 2, 5, 9]);
        let result = diff_column(&col, 2).unwrap();
        assert!(result.is_null(0));
        assert!(result.is_null(1));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[2], 4);
                assert_eq!(v[3], 7);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_diff_float64() {
        let col = float_col(vec![1.0, 1.5, 3.0]);
        let result = diff_column(&col, 1).unwrap();
        assert!(result.is_null(0));
        match &result.data {
            ColumnData::Float64(v) => {
                assert!((v[1] - 0.5).abs() < 1e-10);
                assert!((v[2] - 1.5).abs() < 1e-10);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_diff_str_error() {
        let col = Column::new("x", ColumnData::Str(vec!["a".into()]));
        assert!(diff_column(&col, 1).is_err());
    }

    // --- round tests ---

    #[test]
    fn test_round_float64_zero_decimals() {
        let col = float_col(vec![1.4, 1.5, 2.7]);
        let result = round_column(&col, 0).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[2], 3.0);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_round_float64_two_decimals() {
        let col = float_col(vec![1.234, 2.675]);
        let result = round_column(&col, 2).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                assert!((v[0] - 1.23).abs() < 1e-10);
                // 2.675 rounding may vary due to floating point
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_round_int64_noop() {
        let col = int_col(vec![1, 2, 3]);
        let result = round_column(&col, 2).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    // --- rank tests ---

    #[test]
    fn test_rank_average_ascending() {
        let col = int_col(vec![3, 1, 4, 1, 5]);
        let result = rank_column(&col, RankMethod::Average, true).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                // sorted: 1(idx1), 1(idx3), 3(idx0), 4(idx2), 5(idx4)
                // ranks: 1,3(idx0), 1.5(idx1), 1.5(idx3), 2(idx2), ???
                // 1->rank1.5, 1->rank1.5, 3->rank3, 4->rank4, 5->rank5
                assert_eq!(v[0], 3.0); // 3 is rank 3
                assert_eq!(v[1], 1.5); // first 1 gets avg rank 1.5
                assert_eq!(v[2], 4.0); // 4 is rank 4
                assert_eq!(v[3], 1.5); // second 1 gets avg rank 1.5
                assert_eq!(v[4], 5.0); // 5 is rank 5
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_rank_with_nulls() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![0, 2, 0, 4]),
            vec![false, false, true, false],
        ).unwrap();
        let result = rank_column(&col, RankMethod::Average, true).unwrap();
        assert!(result.is_null(2));
        match &result.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[3], 3.0);
            }
            _ => panic!("wrong dtype"),
        }
    }
}
