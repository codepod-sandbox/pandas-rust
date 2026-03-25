use crate::column::{Column, ColumnData};
use crate::error::{PandasError, Result};

/// Compare two non-null values at positions a and b (ascending order only).
fn compare_values_asc(col: &Column, a: usize, b: usize) -> std::cmp::Ordering {
    match &col.data {
        ColumnData::Bool(v) => v[a].cmp(&v[b]),
        ColumnData::Int64(v) => v[a].cmp(&v[b]),
        ColumnData::Float64(v) => {
            let va = v[a];
            let vb = v[b];
            match (va.is_nan(), vb.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // NaN after values
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        ColumnData::Str(v) => v[a].cmp(&v[b]),
    }
}

/// Count NaN values (non-null positions with NaN) for Float64 columns.
fn nan_count(col: &Column) -> usize {
    match &col.data {
        ColumnData::Float64(v) => v
            .iter()
            .enumerate()
            .filter(|(i, &val)| !col.is_null(*i) && val.is_nan())
            .count(),
        _ => 0,
    }
}

/// Return a stable sort permutation. Nulls sort to the END regardless of direction.
/// For Float64, NaN sorts after all values (before nulls).
pub fn argsort_column(col: &Column, ascending: bool) -> Vec<usize> {
    let null_count = col.null_count();
    let nan_cnt = nan_count(col);
    let mut indices: Vec<usize> = (0..col.len()).collect();

    // Sort ascending first (with nulls at end, NaN before nulls but after values)
    indices.sort_by(|&a, &b| {
        let a_null = col.is_null(a);
        let b_null = col.is_null(b);
        match (a_null, b_null) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // nulls at end
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => compare_values_asc(col, a, b),
        }
    });

    // If descending, reverse only the non-null, non-NaN portion
    // NaN stays just before nulls, nulls stay at end
    if !ascending {
        let tail_count = null_count + nan_cnt;
        let non_special_end = indices.len().saturating_sub(tail_count);
        indices[..non_special_end].reverse();
    }

    indices
}

/// Sort a column by values using argsort.
pub fn sort_column(col: &Column, ascending: bool) -> Column {
    let indices = argsort_column(col, ascending);
    col.take(&indices)
        .expect("argsort indices are always valid")
}

/// Multi-column sort. Sort by first column, break ties with second, etc.
/// All columns must have same length.
pub fn argsort_multi(columns: &[&Column], ascending: &[bool]) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok(vec![]);
    }
    let len = columns[0].len();
    for col in columns.iter() {
        if col.len() != len {
            return Err(PandasError::ValueError(format!(
                "all columns must have the same length, got {} and {}",
                len,
                col.len()
            )));
        }
    }
    // Pad ascending slice if needed
    let asc: Vec<bool> = (0..columns.len())
        .map(|i| ascending.get(i).copied().unwrap_or(true))
        .collect();

    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_by(|&a, &b| {
        for (col, &asc_flag) in columns.iter().zip(asc.iter()) {
            let a_null = col.is_null(a);
            let b_null = col.is_null(b);
            let ord = match (a_null, b_null) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => {
                    let o = compare_values_asc(col, a, b);
                    if asc_flag {
                        o
                    } else {
                        o.reverse()
                    }
                }
            };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(indices)
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

    fn str_col(vals: Vec<&str>) -> Column {
        Column::new(
            "x",
            ColumnData::Str(vals.into_iter().map(|s| s.into()).collect()),
        )
    }

    fn int_col_with_nulls(vals: Vec<i64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Int64(vals), mask).unwrap()
    }

    fn float_col_with_nulls(vals: Vec<f64>, mask: Vec<bool>) -> Column {
        Column::new_with_nulls("x", ColumnData::Float64(vals), mask).unwrap()
    }

    #[test]
    fn test_argsort_int64_ascending() {
        let col = int_col(vec![3, 1, 4, 1, 5]);
        let idx = argsort_column(&col, true);
        let sorted: Vec<i64> = idx
            .iter()
            .map(|&i| match &col.data {
                ColumnData::Int64(v) => v[i],
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn test_argsort_int64_descending() {
        let col = int_col(vec![3, 1, 4, 1, 5]);
        let idx = argsort_column(&col, false);
        let sorted: Vec<i64> = idx
            .iter()
            .map(|&i| match &col.data {
                ColumnData::Int64(v) => v[i],
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(sorted, vec![5, 4, 3, 1, 1]);
    }

    #[test]
    fn test_sort_column_ascending() {
        let col = int_col(vec![5, 2, 8, 1]);
        let sorted = sort_column(&col, true);
        match &sorted.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 5, 8]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_sort_column_descending() {
        let col = int_col(vec![5, 2, 8, 1]);
        let sorted = sort_column(&col, false);
        match &sorted.data {
            ColumnData::Int64(v) => assert_eq!(v, &[8, 5, 2, 1]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_sort_float64_with_nan() {
        let col = float_col(vec![3.0, f64::NAN, 1.0, 2.0]);
        let sorted = sort_column(&col, true);
        match &sorted.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[2], 3.0);
                assert!(v[3].is_nan());
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_sort_float64_desc_with_nan() {
        let col = float_col(vec![3.0, f64::NAN, 1.0]);
        let sorted = sort_column(&col, false);
        match &sorted.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 3.0);
                assert_eq!(v[1], 1.0);
                assert!(v[2].is_nan());
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_sort_str() {
        let col = str_col(vec!["banana", "apple", "cherry"]);
        let sorted = sort_column(&col, true);
        match &sorted.data {
            ColumnData::Str(v) => assert_eq!(v, &["apple", "banana", "cherry"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_nulls_at_end_ascending() {
        let col = int_col_with_nulls(vec![3, 99, 1], vec![false, true, false]);
        let sorted = sort_column(&col, true);
        assert!(!sorted.is_null(0));
        assert!(!sorted.is_null(1));
        assert!(sorted.is_null(2)); // null at end
        match &sorted.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 1);
                assert_eq!(v[1], 3);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_nulls_at_end_descending() {
        let col = int_col_with_nulls(vec![3, 99, 1], vec![false, true, false]);
        let sorted = sort_column(&col, false);
        assert!(!sorted.is_null(0));
        assert!(!sorted.is_null(1));
        assert!(sorted.is_null(2)); // null at end
        match &sorted.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 3);
                assert_eq!(v[1], 1);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_argsort_multi() {
        let col1 = int_col(vec![2, 1, 2, 1]);
        let col2 = int_col(vec![10, 20, 5, 30]);
        // Sort by col1 asc, then col2 asc
        // Expected order by (col1,col2): (1,20),(1,30),(2,5),(2,10)
        // Indices: 1,3,2,0
        let idx = argsort_multi(&[&col1, &col2], &[true, true]).unwrap();
        assert_eq!(idx, vec![1, 3, 2, 0]);
    }

    #[test]
    fn test_argsort_multi_length_mismatch() {
        let col1 = int_col(vec![1, 2, 3]);
        let col2 = int_col(vec![1, 2]);
        assert!(argsort_multi(&[&col1, &col2], &[true, true]).is_err());
    }

    #[test]
    fn test_argsort_multi_descending() {
        let col1 = int_col(vec![2, 1, 2, 1]);
        let col2 = int_col(vec![10, 20, 5, 30]);
        // Sort by col1 desc, then col2 desc
        // col1 desc: 2,2,1,1 -> ties broken by col2 desc: (2,10),(2,5),(1,30),(1,20)
        // Indices: 0,2,3,1
        let idx = argsort_multi(&[&col1, &col2], &[false, false]).unwrap();
        assert_eq!(idx, vec![0, 2, 3, 1]);
    }

    #[test]
    fn test_stable_sort() {
        // Equal values should maintain original order
        let col = int_col(vec![2, 1, 2, 1, 2]);
        let idx = argsort_column(&col, true);
        // For equal values, stable sort preserves original order
        // indices of '1': [1, 3], indices of '2': [0, 2, 4]
        assert_eq!(idx, vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_sort_float64_nulls_at_end() {
        let col = float_col_with_nulls(vec![3.0, 0.0, 1.0], vec![false, true, false]);
        let sorted = sort_column(&col, true);
        assert!(!sorted.is_null(0));
        assert!(!sorted.is_null(1));
        assert!(sorted.is_null(2));
    }
}
