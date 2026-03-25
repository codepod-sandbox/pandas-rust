use std::collections::HashMap;

use crate::column::{Column, ColumnData};
use crate::dataframe::DataFrame;
use crate::dtype::DType;
use crate::error::{PandasError, Result};
use crate::groupby::GroupKeyValue;

/// How to perform the join.
#[derive(Debug, Clone, Copy)]
pub enum JoinHow {
    Inner,
    Left,
    Right,
    Outer,
}

/// Extract a `GroupKeyValue` for a single cell.
/// Returns `None` if the cell is null (null keys never match).
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

/// Build a key for a given row from multiple columns.
/// Returns `None` if any key column is null.
fn build_row_key(cols: &[&Column], row: usize) -> Option<Vec<GroupKeyValue>> {
    let mut key = Vec::with_capacity(cols.len());
    for col in cols {
        match extract_key_value(col, row) {
            Some(v) => key.push(v),
            None => return None,
        }
    }
    Some(key)
}

/// Merge (join) two DataFrames on one or more key columns.
///
/// - `on`: column names present in both DataFrames to join on
/// - `how`: join type (Inner / Left / Right / Outer)
/// - `suffixes`: appended to overlapping non-key column names ("_x", "_y")
pub fn merge(
    left: &DataFrame,
    right: &DataFrame,
    on: &[&str],
    how: JoinHow,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    // Validate key columns exist in both DataFrames
    for &key in on {
        left.get_column(key).map_err(|_| {
            PandasError::KeyError(format!("key column '{}' not found in left DataFrame", key))
        })?;
        right.get_column(key).map_err(|_| {
            PandasError::KeyError(format!("key column '{}' not found in right DataFrame", key))
        })?;
    }

    let on_set: std::collections::HashSet<&str> = on.iter().copied().collect();

    // Non-key columns from each side
    let left_non_key: Vec<&str> = left
        .column_names()
        .into_iter()
        .filter(|n| !on_set.contains(n))
        .collect();
    let right_non_key: Vec<&str> = right
        .column_names()
        .into_iter()
        .filter(|n| !on_set.contains(n))
        .collect();

    // Detect overlapping non-key column names and build output names
    let right_non_key_set: std::collections::HashSet<&str> =
        right_non_key.iter().copied().collect();
    let left_non_key_names: Vec<String> = left_non_key
        .iter()
        .map(|&n| {
            if right_non_key_set.contains(n) {
                format!("{}{}", n, suffixes.0)
            } else {
                n.to_string()
            }
        })
        .collect();

    let left_non_key_set: std::collections::HashSet<&str> = left_non_key.iter().copied().collect();
    let right_non_key_names: Vec<String> = right_non_key
        .iter()
        .map(|&n| {
            if left_non_key_set.contains(n) {
                format!("{}{}", n, suffixes.1)
            } else {
                n.to_string()
            }
        })
        .collect();

    // Fetch key columns
    let left_key_cols: Vec<&Column> = on
        .iter()
        .map(|&n| left.get_column(n))
        .collect::<Result<Vec<_>>>()?;
    let right_key_cols: Vec<&Column> = on
        .iter()
        .map(|&n| right.get_column(n))
        .collect::<Result<Vec<_>>>()?;

    // Build hash map: key → list of row indices in RIGHT
    let mut right_index: HashMap<Vec<GroupKeyValue>, Vec<usize>> = HashMap::new();
    for row in 0..right.nrows() {
        if let Some(key) = build_row_key(&right_key_cols, row) {
            right_index.entry(key).or_default().push(row);
        }
        // Null key rows in right are simply not indexed — they can't match
    }

    // Determine row pairs to emit
    // (left_row_or_none, right_row_or_none)
    let mut left_rows: Vec<Option<usize>> = Vec::new();
    let mut right_rows: Vec<Option<usize>> = Vec::new();

    // Track which right rows have been matched (for Right and Outer joins)
    let mut right_matched: Vec<bool> = vec![false; right.nrows()];

    for lrow in 0..left.nrows() {
        let key = build_row_key(&left_key_cols, lrow);
        let matches = key.as_ref().and_then(|k| right_index.get(k));

        match matches {
            Some(rrows) if !rrows.is_empty() => {
                for &rrow in rrows {
                    left_rows.push(Some(lrow));
                    right_rows.push(Some(rrow));
                    right_matched[rrow] = true;
                }
            }
            _ => {
                // No match
                match how {
                    JoinHow::Left | JoinHow::Outer => {
                        left_rows.push(Some(lrow));
                        right_rows.push(None);
                    }
                    JoinHow::Inner | JoinHow::Right => {
                        // Drop unmatched left rows
                    }
                }
            }
        }
    }

    // For Right and Outer: include unmatched right rows
    if matches!(how, JoinHow::Right | JoinHow::Outer) {
        for (rrow, matched) in right_matched.iter().enumerate() {
            if !matched {
                left_rows.push(None);
                right_rows.push(Some(rrow));
            }
        }
    }

    let nrows = left_rows.len();

    // Build result columns
    let mut result_cols: Vec<Column> = Vec::new();

    // Key columns — taken from left when available, else from right
    for &key_name in on {
        let left_col = left.get_column(key_name)?;
        let right_col = right.get_column(key_name)?;
        let dtype = left_col.dtype();

        let col = build_merged_column(
            key_name,
            dtype,
            &left_rows,
            &right_rows,
            Some(left_col),
            Some(right_col),
            nrows,
        )?;
        result_cols.push(col);
    }

    // Left non-key columns
    for (src_name, out_name) in left_non_key.iter().zip(left_non_key_names.iter()) {
        let left_col = left.get_column(src_name)?;
        let dtype = left_col.dtype();
        let col = build_merged_column(
            out_name,
            dtype,
            &left_rows,
            &right_rows,
            Some(left_col),
            None,
            nrows,
        )?;
        result_cols.push(col);
    }

    // Right non-key columns
    for (src_name, out_name) in right_non_key.iter().zip(right_non_key_names.iter()) {
        let right_col = right.get_column(src_name)?;
        let dtype = right_col.dtype();
        let col = build_merged_column(
            out_name,
            dtype,
            &left_rows,
            &right_rows,
            None,
            Some(right_col),
            nrows,
        )?;
        result_cols.push(col);
    }

    DataFrame::from_columns(result_cols)
}

/// Build a merged column from optional left/right source columns.
///
/// For each output row:
///   - If left_col is Some and left_row is Some → take from left
///   - Else if right_col is Some and right_row is Some → take from right
///   - Else → null
fn build_merged_column(
    name: &str,
    dtype: DType,
    left_rows: &[Option<usize>],
    right_rows: &[Option<usize>],
    left_col: Option<&Column>,
    right_col: Option<&Column>,
    nrows: usize,
) -> Result<Column> {
    let mut null_mask = vec![false; nrows];

    macro_rules! build_typed {
        ($default:expr, $data_variant:ident, $extract:expr) => {{
            let mut vals = vec![$default; nrows];
            for i in 0..nrows {
                let lrow = left_rows[i];
                let rrow = right_rows[i];
                let (src_col, src_row) = if let (Some(col), Some(row)) = (left_col, lrow) {
                    (Some(col), Some(row))
                } else if let (Some(col), Some(row)) = (right_col, rrow) {
                    (Some(col), Some(row))
                } else {
                    (None, None)
                };
                match (src_col, src_row) {
                    (Some(col), Some(row)) => {
                        if col.is_null(row) {
                            null_mask[i] = true;
                        } else {
                            vals[i] = $extract(col, row);
                        }
                    }
                    _ => {
                        null_mask[i] = true;
                    }
                }
            }
            ColumnData::$data_variant(vals)
        }};
    }

    let data = match dtype {
        DType::Bool => build_typed!(false, Bool, |col: &Column, row: usize| {
            if let ColumnData::Bool(v) = &col.data {
                v[row]
            } else {
                false
            }
        }),
        DType::Int64 => build_typed!(0i64, Int64, |col: &Column, row: usize| {
            if let ColumnData::Int64(v) = &col.data {
                v[row]
            } else {
                0
            }
        }),
        DType::Float64 => build_typed!(0.0f64, Float64, |col: &Column, row: usize| {
            if let ColumnData::Float64(v) = &col.data {
                v[row]
            } else {
                0.0
            }
        }),
        DType::Str => build_typed!(String::new(), Str, |col: &Column, row: usize| {
            if let ColumnData::Str(v) = &col.data {
                v[row].clone()
            } else {
                String::new()
            }
        }),
    };

    Column::new_with_nulls(name, data, null_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn left_df() -> DataFrame {
        DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![1, 2, 3])),
            Column::new(
                "val_l",
                ColumnData::Str(vec!["a".into(), "b".into(), "c".into()]),
            ),
        ])
        .unwrap()
    }

    fn right_df() -> DataFrame {
        DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![2, 3, 4])),
            Column::new(
                "val_r",
                ColumnData::Str(vec!["x".into(), "y".into(), "z".into()]),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn test_inner_join() {
        let left = left_df();
        let right = right_df();
        let result = merge(&left, &right, &["id"], JoinHow::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 2); // ids 2 and 3 match
        let id_col = result.get_column("id").unwrap();
        match id_col.data() {
            ColumnData::Int64(v) => {
                assert_eq!(v, &[2, 3]);
            }
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_left_join() {
        let left = left_df();
        let right = right_df();
        let result = merge(&left, &right, &["id"], JoinHow::Left, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 3); // all left rows
                                       // id=1 has no match → val_r is null
        let val_r = result.get_column("val_r").unwrap();
        assert!(val_r.is_null(0)); // id=1 unmatched
        assert!(!val_r.is_null(1)); // id=2 matched
        assert!(!val_r.is_null(2)); // id=3 matched
    }

    #[test]
    fn test_right_join() {
        let left = left_df();
        let right = right_df();
        let result = merge(&left, &right, &["id"], JoinHow::Right, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 3); // all right rows
                                       // id=4 has no left match → val_l is null
        let val_l = result.get_column("val_l").unwrap();
        assert!(!val_l.is_null(0)); // id=2
        assert!(!val_l.is_null(1)); // id=3
        assert!(val_l.is_null(2)); // id=4 unmatched
    }

    #[test]
    fn test_outer_join() {
        let left = left_df();
        let right = right_df();
        let result = merge(&left, &right, &["id"], JoinHow::Outer, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 4); // 3 left + 1 unmatched right
    }

    #[test]
    fn test_multi_column_key() {
        let left = DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 1, 2])),
            Column::new(
                "b",
                ColumnData::Str(vec!["x".into(), "y".into(), "x".into()]),
            ),
            Column::new("v", ColumnData::Int64(vec![10, 20, 30])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            Column::new("a", ColumnData::Int64(vec![1, 2])),
            Column::new("b", ColumnData::Str(vec!["x".into(), "x".into()])),
            Column::new("w", ColumnData::Int64(vec![100, 200])),
        ])
        .unwrap();

        let result = merge(&left, &right, &["a", "b"], JoinHow::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 2); // (1,"x") and (2,"x")
    }

    #[test]
    fn test_overlapping_column_names_suffixes() {
        let left = DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![1, 2])),
            Column::new("val", ColumnData::Int64(vec![10, 20])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![1, 2])),
            Column::new("val", ColumnData::Int64(vec![100, 200])),
        ])
        .unwrap();

        let result = merge(&left, &right, &["id"], JoinHow::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 2);
        let names = result.column_names();
        assert!(names.contains(&"val_x"));
        assert!(names.contains(&"val_y"));
    }

    #[test]
    fn test_no_matches_inner() {
        let left = DataFrame::from_columns(vec![Column::new("id", ColumnData::Int64(vec![1, 2]))])
            .unwrap();
        let right = DataFrame::from_columns(vec![Column::new("id", ColumnData::Int64(vec![3, 4]))])
            .unwrap();

        let result = merge(&left, &right, &["id"], JoinHow::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 0);
    }

    #[test]
    fn test_null_keys_dont_match() {
        let left = DataFrame::from_columns(vec![
            Column::new_with_nulls(
                "id",
                ColumnData::Int64(vec![1, 0, 2]),
                vec![false, true, false],
            )
            .unwrap(),
            Column::new("val_l", ColumnData::Int64(vec![10, 20, 30])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            Column::new_with_nulls("id", ColumnData::Int64(vec![0, 2]), vec![true, false]).unwrap(),
            Column::new("val_r", ColumnData::Int64(vec![100, 200])),
        ])
        .unwrap();

        let result = merge(&left, &right, &["id"], JoinHow::Inner, ("_x", "_y")).unwrap();
        // Only id=2 matches (nulls never match)
        assert_eq!(result.nrows(), 1);
        let id_col = result.get_column("id").unwrap();
        match id_col.data() {
            ColumnData::Int64(v) => assert_eq!(v[0], 2),
            _ => panic!("expected Int64"),
        }
    }

    #[test]
    fn test_duplicate_key_one_to_many() {
        let left = DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![1])),
            Column::new("val_l", ColumnData::Str(vec!["left".into()])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            Column::new("id", ColumnData::Int64(vec![1, 1, 1])),
            Column::new(
                "val_r",
                ColumnData::Str(vec!["r1".into(), "r2".into(), "r3".into()]),
            ),
        ])
        .unwrap();

        let result = merge(&left, &right, &["id"], JoinHow::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.nrows(), 3);
    }

    #[test]
    fn test_missing_key_column() {
        let left = left_df();
        let right = right_df();
        let result = merge(
            &left,
            &right,
            &["nonexistent"],
            JoinHow::Inner,
            ("_x", "_y"),
        );
        assert!(result.is_err());
    }
}
