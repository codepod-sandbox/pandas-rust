use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;

use crate::column::{Column, ColumnData};

/// Controls which duplicate occurrences are marked in `duplicated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Keep the first occurrence; mark subsequent ones as duplicate.
    First,
    /// Keep the last occurrence; mark earlier ones as duplicate.
    Last,
    /// Mark all occurrences of any value that appears more than once.
    MarkAll,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A hashable representation of a single cell value (including null).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CellKey {
    Null,
    Bool(bool),
    Int64(i64),
    Float64Bits(u64), // f64 stored as bits for hashing
    Str(String),
}

fn cell_key(col: &Column, row: usize) -> CellKey {
    if col.is_null(row) {
        return CellKey::Null;
    }
    match col.data() {
        ColumnData::Bool(v) => CellKey::Bool(v[row]),
        ColumnData::Int64(v) => CellKey::Int64(v[row]),
        ColumnData::Float64(v) => CellKey::Float64Bits(v[row].to_bits()),
        ColumnData::Str(v) => CellKey::Str(v[row].clone()),
    }
}

/// Hash a row across multiple columns (used for DataFrame-level operations).
pub fn hash_row(cols: &[&Column], row: usize) -> u64 {
    let mut h = DefaultHasher::new();
    for col in cols {
        cell_key(col, row).hash(&mut h);
    }
    h.finish()
}

/// Build an ordered map from CellKey → list-of-row-indices for a single column.
fn build_index(col: &Column) -> IndexMap<CellKey, Vec<usize>> {
    let mut map: IndexMap<CellKey, Vec<usize>> = IndexMap::new();
    for i in 0..col.len() {
        map.entry(cell_key(col, i)).or_default().push(i);
    }
    map
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return a new column containing only the first occurrence of each distinct value.
/// Null is treated as a value — at most one null is kept.
pub fn unique(col: &Column) -> Column {
    let index = build_index(col);
    // First occurrence index for each key (in key-insertion order).
    let keep_indices: Vec<usize> = index.values().map(|rows| rows[0]).collect();
    // Sort to preserve original order.
    let mut sorted = keep_indices;
    sorted.sort_unstable();
    col.take(&sorted).expect("unique indices always valid")
}

/// Count of distinct values. If `dropna` is true, the null bucket is excluded.
pub fn nunique(col: &Column, dropna: bool) -> usize {
    let index = build_index(col);
    index
        .keys()
        .filter(|k| !(dropna && **k == CellKey::Null))
        .count()
}

/// Compute value counts.
///
/// Returns `(values_column, counts_column)` where `values_column` contains each
/// unique value and `counts_column` contains how many times it appears.
///
/// * `sort`      — if true, sort by count.
/// * `ascending` — used when sort is true; false = descending (default).
/// * `dropna`    — if true, exclude null entries.
pub fn value_counts(col: &Column, sort: bool, ascending: bool, dropna: bool) -> (Column, Column) {
    let index = build_index(col);

    // Build (key, count) pairs, filtering nulls if requested.
    let mut pairs: Vec<(CellKey, usize)> = index
        .into_iter()
        .filter(|(k, _)| !(dropna && *k == CellKey::Null))
        .map(|(k, rows)| (k, rows.len()))
        .collect();

    if sort {
        if ascending {
            pairs.sort_by_key(|(_, c)| *c);
        } else {
            pairs.sort_by(|(_, a), (_, b)| b.cmp(a));
        }
    }

    // Build value column and count column, sharing dtype with the original.
    let counts_data: Vec<i64> = pairs.iter().map(|(_, c)| *c as i64).collect();
    let counts_col = Column::new("count", ColumnData::Int64(counts_data));

    let n = pairs.len();
    let values_col = match col.data() {
        ColumnData::Bool(_) => {
            let mut vals = vec![false; n];
            let mut null_mask = vec![false; n];
            for (i, (k, _)) in pairs.iter().enumerate() {
                match k {
                    CellKey::Bool(b) => vals[i] = *b,
                    CellKey::Null => null_mask[i] = true,
                    _ => {}
                }
            }
            let has_null = null_mask.iter().any(|&b| b);
            if has_null {
                Column::new_with_nulls("value", ColumnData::Bool(vals), null_mask)
                    .expect("lengths match")
            } else {
                Column::new("value", ColumnData::Bool(vals))
            }
        }
        ColumnData::Int64(_) => {
            let mut vals = vec![0i64; n];
            let mut null_mask = vec![false; n];
            for (i, (k, _)) in pairs.iter().enumerate() {
                match k {
                    CellKey::Int64(v) => vals[i] = *v,
                    CellKey::Null => null_mask[i] = true,
                    _ => {}
                }
            }
            let has_null = null_mask.iter().any(|&b| b);
            if has_null {
                Column::new_with_nulls("value", ColumnData::Int64(vals), null_mask)
                    .expect("lengths match")
            } else {
                Column::new("value", ColumnData::Int64(vals))
            }
        }
        ColumnData::Float64(_) => {
            let mut vals = vec![0.0f64; n];
            let mut null_mask = vec![false; n];
            for (i, (k, _)) in pairs.iter().enumerate() {
                match k {
                    CellKey::Float64Bits(bits) => vals[i] = f64::from_bits(*bits),
                    CellKey::Null => null_mask[i] = true,
                    _ => {}
                }
            }
            let has_null = null_mask.iter().any(|&b| b);
            if has_null {
                Column::new_with_nulls("value", ColumnData::Float64(vals), null_mask)
                    .expect("lengths match")
            } else {
                Column::new("value", ColumnData::Float64(vals))
            }
        }
        ColumnData::Str(_) => {
            let mut vals = vec![String::new(); n];
            let mut null_mask = vec![false; n];
            for (i, (k, _)) in pairs.iter().enumerate() {
                match k {
                    CellKey::Str(s) => vals[i] = s.clone(),
                    CellKey::Null => null_mask[i] = true,
                    _ => {}
                }
            }
            let has_null = null_mask.iter().any(|&b| b);
            if has_null {
                Column::new_with_nulls("value", ColumnData::Str(vals), null_mask)
                    .expect("lengths match")
            } else {
                Column::new("value", ColumnData::Str(vals))
            }
        }
    };

    (values_col, counts_col)
}

/// Return a Bool column marking which rows are duplicates.
///
/// * `Keep::First`   — the first occurrence is NOT marked; subsequent are.
/// * `Keep::Last`    — the last occurrence is NOT marked; earlier are.
/// * `Keep::MarkAll` — every occurrence of a duplicated value is marked.
pub fn duplicated(col: &Column, keep: Keep) -> Column {
    let index = build_index(col);
    let n = col.len();
    let mut mask = vec![false; n];

    for rows in index.values() {
        if rows.len() <= 1 {
            continue; // unique value — no duplicates
        }
        match keep {
            Keep::First => {
                // Mark all except the first.
                for &r in &rows[1..] {
                    mask[r] = true;
                }
            }
            Keep::Last => {
                // Mark all except the last.
                let last = rows.len() - 1;
                for &r in &rows[..last] {
                    mask[r] = true;
                }
            }
            Keep::MarkAll => {
                for &r in rows {
                    mask[r] = true;
                }
            }
        }
    }

    Column::new("duplicated", ColumnData::Bool(mask))
}

// ---------------------------------------------------------------------------
// Multi-column duplicated (for DataFrame.drop_duplicates)
// ---------------------------------------------------------------------------

/// Return a Bool column marking duplicate rows in a set of columns.
/// Rows are compared across all provided columns simultaneously.
pub fn duplicated_multi(cols: &[&Column], keep: Keep) -> Column {
    if cols.is_empty() {
        return Column::new("duplicated", ColumnData::Bool(vec![]));
    }
    let n = cols[0].len();

    // Build a map from row-key-hash+values → list of row indices.
    // We use a Vec<CellKey> as the full key.
    let mut index: IndexMap<Vec<CellKey>, Vec<usize>> = IndexMap::new();
    for row in 0..n {
        let key: Vec<CellKey> = cols.iter().map(|c| cell_key(c, row)).collect();
        index.entry(key).or_default().push(row);
    }

    let mut mask = vec![false; n];
    for rows in index.values() {
        if rows.len() <= 1 {
            continue;
        }
        match keep {
            Keep::First => {
                for &r in &rows[1..] {
                    mask[r] = true;
                }
            }
            Keep::Last => {
                let last = rows.len() - 1;
                for &r in &rows[..last] {
                    mask[r] = true;
                }
            }
            Keep::MarkAll => {
                for &r in rows {
                    mask[r] = true;
                }
            }
        }
    }

    Column::new("duplicated", ColumnData::Bool(mask))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn int_col(v: Vec<i64>) -> Column {
        Column::new("x", ColumnData::Int64(v))
    }

    fn str_col(v: Vec<&str>) -> Column {
        Column::new(
            "x",
            ColumnData::Str(v.into_iter().map(str::to_owned).collect()),
        )
    }

    fn float_col(v: Vec<f64>) -> Column {
        Column::new("x", ColumnData::Float64(v))
    }

    // ------------------------------------------------------------------
    // unique
    // ------------------------------------------------------------------

    #[test]
    fn test_unique_int() {
        let col = int_col(vec![1, 2, 1, 3, 2]);
        let u = unique(&col);
        match u.data() {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_unique_str() {
        let col = str_col(vec!["a", "b", "a", "c"]);
        let u = unique(&col);
        match u.data() {
            ColumnData::Str(v) => assert_eq!(v, &["a", "b", "c"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_unique_float() {
        let col = float_col(vec![1.0, 2.0, 1.0, 3.0]);
        let u = unique(&col);
        match u.data() {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, 2.0, 3.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_unique_with_nulls() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 1, 2]),
            vec![false, true, false, false],
        )
        .unwrap();
        let u = unique(&col);
        // Should have: 1, null, 2 (in original order)
        assert_eq!(u.len(), 3);
        assert!(!u.is_null(0));
        assert!(u.is_null(1));
        assert!(!u.is_null(2));
    }

    // ------------------------------------------------------------------
    // nunique
    // ------------------------------------------------------------------

    #[test]
    fn test_nunique_basic() {
        let col = int_col(vec![1, 2, 1, 3]);
        assert_eq!(nunique(&col, false), 3);
    }

    #[test]
    fn test_nunique_with_null_dropna_false() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 2]),
            vec![false, true, false],
        )
        .unwrap();
        assert_eq!(nunique(&col, false), 3); // 1, null, 2
    }

    #[test]
    fn test_nunique_with_null_dropna_true() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 2]),
            vec![false, true, false],
        )
        .unwrap();
        assert_eq!(nunique(&col, true), 2); // 1, 2 (null excluded)
    }

    // ------------------------------------------------------------------
    // value_counts
    // ------------------------------------------------------------------

    #[test]
    fn test_value_counts_sorted_desc() {
        let col = int_col(vec![1, 2, 1, 3, 2, 2]);
        let (vals, counts) = value_counts(&col, true, false, true);
        match counts.data() {
            ColumnData::Int64(c) => assert_eq!(c[0], 3), // 2 appears 3x
            _ => panic!(),
        }
        match vals.data() {
            ColumnData::Int64(v) => assert_eq!(v[0], 2),
            _ => panic!(),
        }
    }

    #[test]
    fn test_value_counts_dropna() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 1]),
            vec![false, true, false],
        )
        .unwrap();
        let (vals, _counts) = value_counts(&col, false, false, true);
        assert_eq!(vals.len(), 1); // only non-null value
    }

    #[test]
    fn test_value_counts_include_null() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 1]),
            vec![false, true, false],
        )
        .unwrap();
        let (vals, _counts) = value_counts(&col, false, false, false);
        assert_eq!(vals.len(), 2); // 1 and null
    }

    // ------------------------------------------------------------------
    // duplicated
    // ------------------------------------------------------------------

    #[test]
    fn test_duplicated_first() {
        let col = int_col(vec![1, 2, 1, 3, 2]);
        let d = duplicated(&col, Keep::First);
        match d.data() {
            ColumnData::Bool(v) => assert_eq!(v, &[false, false, true, false, true]),
            _ => panic!(),
        }
    }

    #[test]
    fn test_duplicated_last() {
        let col = int_col(vec![1, 2, 1, 3, 2]);
        let d = duplicated(&col, Keep::Last);
        match d.data() {
            ColumnData::Bool(v) => assert_eq!(v, &[true, true, false, false, false]),
            _ => panic!(),
        }
    }

    #[test]
    fn test_duplicated_mark_all() {
        let col = int_col(vec![1, 2, 1, 3, 2]);
        let d = duplicated(&col, Keep::MarkAll);
        match d.data() {
            ColumnData::Bool(v) => assert_eq!(v, &[true, true, true, false, true]),
            _ => panic!(),
        }
    }

    #[test]
    fn test_duplicated_no_dupes() {
        let col = int_col(vec![1, 2, 3]);
        let d = duplicated(&col, Keep::First);
        match d.data() {
            ColumnData::Bool(v) => assert_eq!(v, &[false, false, false]),
            _ => panic!(),
        }
    }

    // ------------------------------------------------------------------
    // duplicated_multi
    // ------------------------------------------------------------------

    #[test]
    fn test_duplicated_multi_first() {
        let a = int_col(vec![1, 2, 1]);
        let b = int_col(vec![10, 20, 10]);
        let d = duplicated_multi(&[&a, &b], Keep::First);
        match d.data() {
            ColumnData::Bool(v) => assert_eq!(v, &[false, false, true]),
            _ => panic!(),
        }
    }
}
