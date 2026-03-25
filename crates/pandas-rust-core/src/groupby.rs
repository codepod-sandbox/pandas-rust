use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;

use crate::column::{Column, ColumnData};
use crate::dataframe::DataFrame;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::aggregation::{self, AggResult};

/// A single group key value — one entry per groupby column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupKeyValue {
    Bool(bool),
    Int64(i64),
    Str(String),
    /// f64 stored as bits for hashing/equality.
    Float64Bits(u64),
}

/// Holds the grouped row indices, keyed by group key.
pub struct GroupedData {
    /// Maps group key → list of row indices (in first-seen order).
    pub groups: IndexMap<Vec<GroupKeyValue>, Vec<usize>>,
    /// Names of the columns used for grouping.
    pub by_columns: Vec<String>,
}

/// Aggregation functions supported by `aggregate`.
#[derive(Debug, Clone, Copy)]
pub enum AggFn {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Std,
    Var,
    Median,
    First,
    Last,
    Size,
}

/// Extract the `GroupKeyValue` for a single cell, or `None` if the cell is null.
fn extract_key_value(col: &Column, row: usize) -> Option<GroupKeyValue> {
    if col.is_null(row) {
        return None;
    }
    Some(match &col.data {
        ColumnData::Bool(v) => GroupKeyValue::Bool(v[row]),
        ColumnData::Int64(v) => GroupKeyValue::Int64(v[row]),
        ColumnData::Float64(v) => GroupKeyValue::Float64Bits(v[row].to_bits()),
        ColumnData::Str(v) => GroupKeyValue::Str(v[row].clone()),
    })
}

/// Build a `GroupedData` from `df` grouped by the columns named in `by`.
///
/// Rows where ANY key column is null are dropped (pandas default `dropna=True`).
pub fn group_by(df: &DataFrame, by: &[&str]) -> Result<GroupedData> {
    // Validate columns exist
    for &name in by {
        df.get_column(name)?;
    }

    let nrows = df.nrows();
    let mut groups: IndexMap<Vec<GroupKeyValue>, Vec<usize>> = IndexMap::new();

    // Fetch column references up front
    let key_cols: Vec<&Column> = by
        .iter()
        .map(|&name| df.get_column(name))
        .collect::<Result<Vec<_>>>()?;

    for row in 0..nrows {
        // Build key, dropping rows with any null key
        let mut key = Vec::with_capacity(by.len());
        let mut has_null = false;
        for col in &key_cols {
            match extract_key_value(col, row) {
                Some(v) => key.push(v),
                None => {
                    has_null = true;
                    break;
                }
            }
        }
        if has_null {
            continue;
        }
        groups.entry(key).or_default().push(row);
    }

    Ok(GroupedData {
        groups,
        by_columns: by.iter().map(|s| s.to_string()).collect(),
    })
}

/// Hash a key to a u64 (not used for grouping directly — IndexMap handles equality).
#[allow(dead_code)]
fn hash_key(key: &[GroupKeyValue]) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Apply `agg_fn` to `grouped` data from `df`.
///
/// Returns a DataFrame with one row per group:
///   - key columns first
///   - then one aggregated column per non-key numeric (or compatible) column
pub fn aggregate(grouped: &GroupedData, df: &DataFrame, agg_fn: AggFn) -> Result<DataFrame> {
    let by_set: std::collections::HashSet<&str> =
        grouped.by_columns.iter().map(|s| s.as_str()).collect();

    // Non-key columns to aggregate
    let agg_cols: Vec<&str> = df
        .column_names()
        .into_iter()
        .filter(|name| !by_set.contains(name))
        .collect();

    let ngroups = grouped.groups.len();

    // Build key columns
    let mut result_columns: Vec<Column> = Vec::new();

    for (key_idx, key_name) in grouped.by_columns.iter().enumerate() {
        let src_col = df.get_column(key_name)?;
        let dtype = src_col.dtype();

        // Build a column from the first element of each group key
        let mut null_mask = vec![false; ngroups];
        let data = match dtype {
            DType::Bool => {
                let mut vals = vec![false; ngroups];
                for (group_idx, key) in grouped.groups.keys().enumerate() {
                    match &key[key_idx] {
                        GroupKeyValue::Bool(b) => vals[group_idx] = *b,
                        _ => null_mask[group_idx] = true,
                    }
                }
                ColumnData::Bool(vals)
            }
            DType::Int64 => {
                let mut vals = vec![0i64; ngroups];
                for (group_idx, key) in grouped.groups.keys().enumerate() {
                    match &key[key_idx] {
                        GroupKeyValue::Int64(v) => vals[group_idx] = *v,
                        _ => null_mask[group_idx] = true,
                    }
                }
                ColumnData::Int64(vals)
            }
            DType::Float64 => {
                let mut vals = vec![0.0f64; ngroups];
                for (group_idx, key) in grouped.groups.keys().enumerate() {
                    match &key[key_idx] {
                        GroupKeyValue::Float64Bits(bits) => vals[group_idx] = f64::from_bits(*bits),
                        _ => null_mask[group_idx] = true,
                    }
                }
                ColumnData::Float64(vals)
            }
            DType::Str => {
                let mut vals = vec![String::new(); ngroups];
                for (group_idx, key) in grouped.groups.keys().enumerate() {
                    match &key[key_idx] {
                        GroupKeyValue::Str(s) => vals[group_idx] = s.clone(),
                        _ => null_mask[group_idx] = true,
                    }
                }
                ColumnData::Str(vals)
            }
        };
        let col = Column::new_with_nulls(key_name.as_str(), data, null_mask)?;
        result_columns.push(col);
    }

    // Build aggregated columns
    for &col_name in &agg_cols {
        let src_col = df.get_column(col_name)?;

        // For size, always return Int64 regardless of dtype
        if matches!(agg_fn, AggFn::Size) {
            let vals: Vec<i64> = grouped
                .groups
                .values()
                .map(|rows| rows.len() as i64)
                .collect();
            result_columns.push(Column::new(col_name, ColumnData::Int64(vals)));
            continue;
        }

        // Determine if this column is numeric
        let is_numeric = matches!(src_col.dtype(), DType::Int64 | DType::Float64);
        let is_str = matches!(src_col.dtype(), DType::Str);

        // Skip non-compatible columns for numeric-only aggregations
        let skip = match agg_fn {
            AggFn::Sum | AggFn::Mean | AggFn::Std | AggFn::Var | AggFn::Median => !is_numeric,
            AggFn::Min | AggFn::Max => !is_numeric && !is_str,
            AggFn::Count | AggFn::First | AggFn::Last => false,
            AggFn::Size => false, // handled above
        };
        if skip {
            continue;
        }

        // Aggregate each group
        let mut agg_results: Vec<AggResult> = Vec::with_capacity(ngroups);

        for rows in grouped.groups.values() {
            let sub_col = src_col.take(rows)?;
            let result = match agg_fn {
                AggFn::Sum => aggregation::sum(&sub_col)?,
                AggFn::Mean => aggregation::mean(&sub_col)?,
                AggFn::Min => aggregation::min(&sub_col)?,
                AggFn::Max => aggregation::max(&sub_col)?,
                AggFn::Count => aggregation::count(&sub_col),
                AggFn::Std => aggregation::std(&sub_col, 1)?,
                AggFn::Var => aggregation::var(&sub_col, 1)?,
                AggFn::Median => aggregation::median(&sub_col)?,
                AggFn::First => first_value(&sub_col),
                AggFn::Last => last_value(&sub_col),
                AggFn::Size => AggResult::Int64(rows.len() as i64), // already handled
            };
            agg_results.push(result);
        }

        // Determine output dtype from results
        let col = agg_results_to_column(col_name, &agg_results, src_col.dtype())?;
        result_columns.push(col);
    }

    DataFrame::from_columns(result_columns)
}

/// Return the first non-null value in `col`, or `AggResult::None` if all null.
fn first_value(col: &Column) -> AggResult {
    for i in 0..col.len() {
        if !col.is_null(i) {
            return match &col.data {
                ColumnData::Bool(v) => AggResult::Int64(v[i] as i64),
                ColumnData::Int64(v) => AggResult::Int64(v[i]),
                ColumnData::Float64(v) => AggResult::Float64(v[i]),
                ColumnData::Str(v) => AggResult::Str(v[i].clone()),
            };
        }
    }
    AggResult::None
}

/// Return the last non-null value in `col`, or `AggResult::None` if all null.
fn last_value(col: &Column) -> AggResult {
    for i in (0..col.len()).rev() {
        if !col.is_null(i) {
            return match &col.data {
                ColumnData::Bool(v) => AggResult::Int64(v[i] as i64),
                ColumnData::Int64(v) => AggResult::Int64(v[i]),
                ColumnData::Float64(v) => AggResult::Float64(v[i]),
                ColumnData::Str(v) => AggResult::Str(v[i].clone()),
            };
        }
    }
    AggResult::None
}

/// Convert a Vec<AggResult> into a Column, inferring the output dtype.
fn agg_results_to_column(name: &str, results: &[AggResult], src_dtype: DType) -> Result<Column> {
    // Determine output dtype by scanning results
    let has_float = results.iter().any(|r| matches!(r, AggResult::Float64(_)));
    let has_int = results.iter().any(|r| matches!(r, AggResult::Int64(_)));
    let has_usize = results.iter().any(|r| matches!(r, AggResult::Usize(_)));
    let has_str = results.iter().any(|r| matches!(r, AggResult::Str(_)));

    let out_dtype = if has_float {
        DType::Float64
    } else if has_int || has_usize {
        DType::Int64
    } else if has_str {
        DType::Str
    } else {
        // All None — use source dtype
        src_dtype
    };

    let mut null_mask = vec![false; results.len()];

    let data = match out_dtype {
        DType::Float64 => {
            let vals: Vec<f64> = results
                .iter()
                .enumerate()
                .map(|(i, r)| match r {
                    AggResult::Float64(v) => *v,
                    AggResult::Int64(v) => *v as f64,
                    AggResult::Usize(v) => *v as f64,
                    AggResult::None => {
                        null_mask[i] = true;
                        0.0
                    }
                    AggResult::Str(_) => {
                        null_mask[i] = true;
                        0.0
                    }
                })
                .collect();
            ColumnData::Float64(vals)
        }
        DType::Int64 => {
            let vals: Vec<i64> = results
                .iter()
                .enumerate()
                .map(|(i, r)| match r {
                    AggResult::Int64(v) => *v,
                    AggResult::Usize(v) => *v as i64,
                    AggResult::Float64(v) => *v as i64,
                    AggResult::None => {
                        null_mask[i] = true;
                        0
                    }
                    AggResult::Str(_) => {
                        null_mask[i] = true;
                        0
                    }
                })
                .collect();
            ColumnData::Int64(vals)
        }
        DType::Str => {
            let vals: Vec<String> = results
                .iter()
                .enumerate()
                .map(|(i, r)| match r {
                    AggResult::Str(s) => s.clone(),
                    AggResult::None => {
                        null_mask[i] = true;
                        String::new()
                    }
                    AggResult::Int64(v) => v.to_string(),
                    AggResult::Usize(v) => v.to_string(),
                    AggResult::Float64(v) => v.to_string(),
                })
                .collect();
            ColumnData::Str(vals)
        }
        DType::Bool => {
            // Shouldn't normally happen for aggregations
            let vals = vec![false; results.len()];
            null_mask = vec![true; results.len()];
            ColumnData::Bool(vals)
        }
    };

    Column::new_with_nulls(name, data, null_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn make_df() -> DataFrame {
        DataFrame::from_columns(vec![
            Column::new(
                "cat",
                ColumnData::Str(vec![
                    "a".into(),
                    "b".into(),
                    "a".into(),
                    "b".into(),
                    "a".into(),
                ]),
            ),
            Column::new("val", ColumnData::Int64(vec![1, 2, 3, 4, 5])),
            Column::new("score", ColumnData::Float64(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ])
        .unwrap()
    }

    #[test]
    fn test_groupby_single_column_sum() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        assert_eq!(grouped.groups.len(), 2);

        let result = aggregate(&grouped, &df, AggFn::Sum).unwrap();
        assert_eq!(result.nrows(), 2);
        // cat "a" has rows [0,2,4] → val sum = 1+3+5 = 9
        // cat "b" has rows [1,3]   → val sum = 2+4   = 6
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 9); // a
                assert_eq!(v[1], 6); // b
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_mean() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        let result = aggregate(&grouped, &df, AggFn::Mean).unwrap();
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Float64(v) => {
                assert!((v[0] - 3.0).abs() < 1e-10); // (1+3+5)/3 = 3
                assert!((v[1] - 3.0).abs() < 1e-10); // (2+4)/2 = 3
            }
            _ => panic!("expected Float64"),
        }
    }

    #[test]
    fn test_groupby_count() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        let result = aggregate(&grouped, &df, AggFn::Count).unwrap();
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 3); // a: 3 non-null
                assert_eq!(v[1], 2); // b: 2 non-null
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_size() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        let result = aggregate(&grouped, &df, AggFn::Size).unwrap();
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 3);
                assert_eq!(v[1], 2);
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_first() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        let result = aggregate(&grouped, &df, AggFn::First).unwrap();
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 1); // first "a" row has val=1
                assert_eq!(v[1], 2); // first "b" row has val=2
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_last() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();
        let result = aggregate(&grouped, &df, AggFn::Last).unwrap();
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 5); // last "a" row has val=5
                assert_eq!(v[1], 4); // last "b" row has val=4
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_multi_column() {
        let df = DataFrame::from_columns(vec![
            Column::new(
                "cat",
                ColumnData::Str(vec!["a".into(), "a".into(), "b".into(), "b".into()]),
            ),
            Column::new("sub", ColumnData::Int64(vec![1, 2, 1, 1])),
            Column::new("val", ColumnData::Int64(vec![10, 20, 30, 40])),
        ])
        .unwrap();

        let grouped = group_by(&df, &["cat", "sub"]).unwrap();
        // Groups: (a,1)→[0], (a,2)→[1], (b,1)→[2,3]
        assert_eq!(grouped.groups.len(), 3);

        let result = aggregate(&grouped, &df, AggFn::Sum).unwrap();
        assert_eq!(result.nrows(), 3);
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 10); // (a,1)
                assert_eq!(v[1], 20); // (a,2)
                assert_eq!(v[2], 70); // (b,1): 30+40
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_drops_null_keys() {
        let df = DataFrame::from_columns(vec![
            Column::new_with_nulls(
                "cat",
                ColumnData::Str(vec!["a".into(), "b".into(), "a".into()]),
                vec![false, true, false],
            )
            .unwrap(),
            Column::new("val", ColumnData::Int64(vec![1, 2, 3])),
        ])
        .unwrap();

        let grouped = group_by(&df, &["cat"]).unwrap();
        // Row 1 has null key → dropped
        assert_eq!(grouped.groups.len(), 1); // only group "a"
        let result = aggregate(&grouped, &df, AggFn::Sum).unwrap();
        assert_eq!(result.nrows(), 1);
        let val_col = result.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => assert_eq!(v[0], 4), // 1+3
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_empty_dataframe() {
        let df = DataFrame::from_columns(vec![
            Column::new("cat", ColumnData::Str(vec![])),
            Column::new("val", ColumnData::Int64(vec![])),
        ])
        .unwrap();

        let grouped = group_by(&df, &["cat"]).unwrap();
        assert_eq!(grouped.groups.len(), 0);

        let result = aggregate(&grouped, &df, AggFn::Sum).unwrap();
        assert_eq!(result.nrows(), 0);
    }

    #[test]
    fn test_groupby_preserves_order() {
        // Groups should appear in first-seen order: b, a, c
        let df = DataFrame::from_columns(vec![
            Column::new(
                "cat",
                ColumnData::Str(vec![
                    "b".into(),
                    "a".into(),
                    "c".into(),
                    "b".into(),
                    "a".into(),
                ]),
            ),
            Column::new("val", ColumnData::Int64(vec![1, 2, 3, 4, 5])),
        ])
        .unwrap();

        let grouped = group_by(&df, &["cat"]).unwrap();
        let keys: Vec<&Vec<GroupKeyValue>> = grouped.groups.keys().collect();
        assert_eq!(keys[0], &vec![GroupKeyValue::Str("b".into())]);
        assert_eq!(keys[1], &vec![GroupKeyValue::Str("a".into())]);
        assert_eq!(keys[2], &vec![GroupKeyValue::Str("c".into())]);
    }

    #[test]
    fn test_groupby_min_max() {
        let df = make_df();
        let grouped = group_by(&df, &["cat"]).unwrap();

        let result_min = aggregate(&grouped, &df, AggFn::Min).unwrap();
        let val_col = result_min.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 1); // min of a: 1,3,5
                assert_eq!(v[1], 2); // min of b: 2,4
            }
            _ => panic!("expected Int64"),
        }

        let result_max = aggregate(&grouped, &df, AggFn::Max).unwrap();
        let val_col = result_max.get_column("val").unwrap();
        match val_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 5);
                assert_eq!(v[1], 4);
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_groupby_missing_column() {
        let df = make_df();
        let result = group_by(&df, &["nonexistent"]);
        assert!(result.is_err());
    }
}
