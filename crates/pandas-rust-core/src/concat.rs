use std::collections::HashSet;

use crate::casting::cast_column;
use crate::column::{Column, ColumnData};
use crate::dataframe::DataFrame;
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Vertically stack DataFrames (row-wise concatenation).
/// Union of column names across all frames. Missing columns filled with nulls.
/// Dtype promotion for matching columns with different types (Int64 + Float64 → Float64).
/// Index is reset to RangeIndex.
pub fn concat_rows(dataframes: &[&DataFrame]) -> Result<DataFrame> {
    // Filter out empty DataFrames
    let frames: Vec<&DataFrame> = dataframes
        .iter()
        .copied()
        .filter(|df| df.ncols() > 0 && df.nrows() > 0)
        .collect();

    if frames.is_empty() {
        return Ok(DataFrame::empty());
    }
    if frames.len() == 1 {
        // Reset index to RangeIndex
        let df = frames[0];
        let cols: Vec<Column> = df.iter_columns().map(|(_, c)| c.clone()).collect();
        return DataFrame::from_columns(cols);
    }

    // Collect all column names in order (union, preserving first-seen order)
    let mut all_names: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for df in &frames {
        for name in df.column_names() {
            if seen.insert(name.to_string()) {
                all_names.push(name.to_string());
            }
        }
    }

    // Determine target dtype for each column (promote Int64+Float64 → Float64)
    let mut col_dtypes: std::collections::HashMap<String, DType> = std::collections::HashMap::new();
    for df in &frames {
        for (name, col) in df.iter_columns() {
            let entry = col_dtypes.entry(name.to_string()).or_insert(col.dtype());
            *entry = promote_dtype(*entry, col.dtype());
        }
    }

    // Build each output column
    let mut result_cols: Vec<Column> = Vec::new();
    for col_name in &all_names {
        let target_dtype = col_dtypes[col_name];
        let mut parts: Vec<Column> = Vec::new();
        for df in &frames {
            let part = if let Ok(col) = df.get_column(col_name) {
                // Cast to target dtype if needed
                if col.dtype() != target_dtype {
                    cast_column(col, target_dtype)?
                } else {
                    col.clone()
                }
            } else {
                // Missing column: fill with nulls
                Column::all_null(col_name, target_dtype, df.nrows())
            };
            parts.push(part);
        }
        // Concatenate parts
        let concatenated = concat_column_parts(&parts, col_name)?;
        result_cols.push(concatenated);
    }

    DataFrame::from_columns(result_cols)
}

/// Horizontally stack DataFrames (column-wise concatenation).
/// All must have same row count. Duplicate column names get `_0`, `_1` suffixes.
pub fn concat_cols(dataframes: &[&DataFrame]) -> Result<DataFrame> {
    // Filter out empty DataFrames
    let frames: Vec<&DataFrame> = dataframes
        .iter()
        .copied()
        .filter(|df| df.ncols() > 0)
        .collect();

    if frames.is_empty() {
        return Ok(DataFrame::empty());
    }
    if frames.len() == 1 {
        return Ok(frames[0].clone());
    }

    // Check all have same row count
    let nrows = frames[0].nrows();
    for df in frames.iter().skip(1) {
        if df.nrows() != nrows {
            return Err(PandasError::ValueError(format!(
                "all DataFrames must have the same row count for concat_cols, got {} and {}",
                nrows,
                df.nrows()
            )));
        }
    }

    // Collect all columns, renaming duplicates
    let mut all_cols: Vec<Column> = Vec::new();
    let mut name_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    // First pass: count occurrences to know which need suffixes
    let mut occurrence_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for df in &frames {
        for (name, _) in df.iter_columns() {
            *occurrence_count.entry(name.to_string()).or_insert(0) += 1;
        }
    }

    for df in &frames {
        for (name, col) in df.iter_columns() {
            let final_name = if occurrence_count[name] > 1 {
                // Need suffix
                let count = name_counts.entry(name.to_string()).or_insert(0);
                let suffixed = format!("{}_{}", name, count);
                *count += 1;
                suffixed
            } else if seen_names.contains(name) {
                // Same name seen again (shouldn't happen with occurrence_count > 1 check)
                let count = name_counts.entry(name.to_string()).or_insert(0);
                let suffixed = format!("{}_{}", name, count);
                *count += 1;
                suffixed
            } else {
                name.to_string()
            };
            seen_names.insert(name.to_string());
            all_cols.push(col.clone().with_name(final_name));
        }
    }

    DataFrame::from_columns(all_cols)
}

/// Promote two dtypes: Int64 + Float64 → Float64, same type → same.
fn promote_dtype(a: DType, b: DType) -> DType {
    match (a, b) {
        (DType::Int64, DType::Float64) | (DType::Float64, DType::Int64) => DType::Float64,
        _ => a, // keep first type (or could error for incompatible types)
    }
}

/// Concatenate column parts vertically.
fn concat_column_parts(parts: &[Column], name: &str) -> Result<Column> {
    if parts.is_empty() {
        return Ok(Column::new(name, ColumnData::Int64(vec![])));
    }

    // Total length
    let total_len: usize = parts.iter().map(|p| p.len()).sum();

    // Build combined null mask: Some if any part has nulls
    let has_any_nulls = parts.iter().any(|p| p.has_nulls());

    let mut null_mask: Vec<bool> = if has_any_nulls {
        Vec::with_capacity(total_len)
    } else {
        vec![]
    };

    if has_any_nulls {
        for part in parts {
            for i in 0..part.len() {
                null_mask.push(part.is_null(i));
            }
        }
    }

    // Concatenate data
    let data = match parts[0].dtype() {
        DType::Bool => {
            let mut vals: Vec<bool> = Vec::with_capacity(total_len);
            for part in parts {
                match &part.data {
                    ColumnData::Bool(v) => vals.extend_from_slice(v),
                    _ => {
                        return Err(PandasError::TypeError(
                            "dtype mismatch in concat".to_string(),
                        ))
                    }
                }
            }
            ColumnData::Bool(vals)
        }
        DType::Int64 => {
            let mut vals: Vec<i64> = Vec::with_capacity(total_len);
            for part in parts {
                match &part.data {
                    ColumnData::Int64(v) => vals.extend_from_slice(v),
                    _ => {
                        return Err(PandasError::TypeError(
                            "dtype mismatch in concat".to_string(),
                        ))
                    }
                }
            }
            ColumnData::Int64(vals)
        }
        DType::Float64 => {
            let mut vals: Vec<f64> = Vec::with_capacity(total_len);
            for part in parts {
                match &part.data {
                    ColumnData::Float64(v) => vals.extend_from_slice(v),
                    _ => {
                        return Err(PandasError::TypeError(
                            "dtype mismatch in concat".to_string(),
                        ))
                    }
                }
            }
            ColumnData::Float64(vals)
        }
        DType::Str => {
            let mut vals: Vec<String> = Vec::with_capacity(total_len);
            for part in parts {
                match &part.data {
                    ColumnData::Str(v) => vals.extend_from_slice(v),
                    _ => {
                        return Err(PandasError::TypeError(
                            "dtype mismatch in concat".to_string(),
                        ))
                    }
                }
            }
            ColumnData::Str(vals)
        }
    };

    if has_any_nulls {
        Column::new_with_nulls(name, data, null_mask)
    } else {
        Ok(Column::new(name, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn make_df(cols: Vec<Column>) -> DataFrame {
        DataFrame::from_columns(cols).unwrap()
    }

    fn int_col(name: &str, vals: Vec<i64>) -> Column {
        Column::new(name, ColumnData::Int64(vals))
    }

    fn float_col(name: &str, vals: Vec<f64>) -> Column {
        Column::new(name, ColumnData::Float64(vals))
    }

    fn str_col(name: &str, vals: Vec<&str>) -> Column {
        Column::new(
            name,
            ColumnData::Str(vals.into_iter().map(|s| s.into()).collect()),
        )
    }

    #[test]
    fn test_concat_rows_same_columns() {
        let df1 = make_df(vec![int_col("a", vec![1, 2]), int_col("b", vec![3, 4])]);
        let df2 = make_df(vec![int_col("a", vec![5, 6]), int_col("b", vec![7, 8])]);
        let result = concat_rows(&[&df1, &df2]).unwrap();
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        let col_a = result.get_column("a").unwrap();
        match &col_a.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 5, 6]),
            _ => panic!("wrong dtype"),
        }
        // Index should be reset to range
        assert_eq!(result.index().len(), 4);
    }

    #[test]
    fn test_concat_rows_different_columns() {
        let df1 = make_df(vec![int_col("a", vec![1, 2])]);
        let df2 = make_df(vec![int_col("b", vec![3, 4])]);
        let result = concat_rows(&[&df1, &df2]).unwrap();
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        // "a" should have nulls in rows 2,3
        let col_a = result.get_column("a").unwrap();
        assert!(!col_a.is_null(0));
        assert!(!col_a.is_null(1));
        assert!(col_a.is_null(2));
        assert!(col_a.is_null(3));
        // "b" should have nulls in rows 0,1
        let col_b = result.get_column("b").unwrap();
        assert!(col_b.is_null(0));
        assert!(col_b.is_null(1));
        assert!(!col_b.is_null(2));
        assert!(!col_b.is_null(3));
    }

    #[test]
    fn test_concat_rows_type_promotion() {
        let df1 = make_df(vec![int_col("a", vec![1, 2])]);
        let df2 = make_df(vec![float_col("a", vec![3.5, 4.5])]);
        let result = concat_rows(&[&df1, &df2]).unwrap();
        assert_eq!(result.nrows(), 4);
        let col_a = result.get_column("a").unwrap();
        assert_eq!(col_a.dtype(), DType::Float64);
        match &col_a.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[2], 3.5);
                assert_eq!(v[3], 4.5);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_concat_rows_empty_list() {
        let result = concat_rows(&[]).unwrap();
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 0);
    }

    #[test]
    fn test_concat_rows_single_df() {
        let df = make_df(vec![int_col("a", vec![1, 2, 3])]);
        let result = concat_rows(&[&df]).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 1);
    }

    #[test]
    fn test_concat_rows_with_str() {
        let df1 = make_df(vec![str_col("s", vec!["a", "b"])]);
        let df2 = make_df(vec![str_col("s", vec!["c"])]);
        let result = concat_rows(&[&df1, &df2]).unwrap();
        let col = result.get_column("s").unwrap();
        match &col.data {
            ColumnData::Str(v) => assert_eq!(v, &["a", "b", "c"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_concat_cols_same_row_count() {
        let df1 = make_df(vec![int_col("a", vec![1, 2, 3])]);
        let df2 = make_df(vec![int_col("b", vec![4, 5, 6])]);
        let result = concat_cols(&[&df1, &df2]).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert!(result.get_column("a").is_ok());
        assert!(result.get_column("b").is_ok());
    }

    #[test]
    fn test_concat_cols_different_row_count() {
        let df1 = make_df(vec![int_col("a", vec![1, 2, 3])]);
        let df2 = make_df(vec![int_col("b", vec![4, 5])]);
        assert!(concat_cols(&[&df1, &df2]).is_err());
    }

    #[test]
    fn test_concat_cols_duplicate_names() {
        let df1 = make_df(vec![int_col("a", vec![1, 2])]);
        let df2 = make_df(vec![int_col("a", vec![3, 4])]);
        let result = concat_cols(&[&df1, &df2]).unwrap();
        assert_eq!(result.ncols(), 2);
        // Names should be suffixed
        let names = result.column_names();
        assert!(names.contains(&"a_0") || names.contains(&"a_1"));
    }

    #[test]
    fn test_concat_cols_empty_list() {
        let result = concat_cols(&[]).unwrap();
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 0);
    }

    #[test]
    fn test_concat_cols_single_df() {
        let df = make_df(vec![int_col("a", vec![1, 2])]);
        let result = concat_cols(&[&df]).unwrap();
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 1);
    }
}
