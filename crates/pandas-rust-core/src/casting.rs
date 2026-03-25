use crate::column::{Column, ColumnData};
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Cast a column to a different dtype.
/// Null values are preserved through all casts.
pub fn cast_column(col: &Column, target: DType) -> Result<Column> {
    // Same type → clone
    if col.dtype() == target {
        return Ok(col.clone());
    }

    match (&col.data, target) {
        // Int64 → Float64: lossless
        (ColumnData::Int64(v), DType::Float64) => {
            let data: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            make_col(col, ColumnData::Float64(data))
        }

        // Float64 → Int64: truncate. Error if NaN or Inf or non-null
        (ColumnData::Float64(v), DType::Int64) => {
            let mut data = Vec::with_capacity(v.len());
            for (i, &val) in v.iter().enumerate() {
                if col.is_null(i) {
                    data.push(0i64); // placeholder, will be masked
                } else if val.is_nan() || val.is_infinite() {
                    return Err(PandasError::ValueError(format!(
                        "cannot cast {} to Int64",
                        val
                    )));
                } else {
                    data.push(val as i64);
                }
            }
            make_col(col, ColumnData::Int64(data))
        }

        // Bool → Int64
        (ColumnData::Bool(v), DType::Int64) => {
            let data: Vec<i64> = v.iter().map(|&x| if x { 1 } else { 0 }).collect();
            make_col(col, ColumnData::Int64(data))
        }

        // Bool → Float64
        (ColumnData::Bool(v), DType::Float64) => {
            let data: Vec<f64> = v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
            make_col(col, ColumnData::Float64(data))
        }

        // Int64 → Bool: 0=false, nonzero=true
        (ColumnData::Int64(v), DType::Bool) => {
            let data: Vec<bool> = v.iter().map(|&x| x != 0).collect();
            make_col(col, ColumnData::Bool(data))
        }

        // Bool → Str: "True"/"False"
        (ColumnData::Bool(v), DType::Str) => {
            let data: Vec<String> = v
                .iter()
                .map(|&x| if x { "True".into() } else { "False".into() })
                .collect();
            make_col(col, ColumnData::Str(data))
        }

        // Int64 → Str
        (ColumnData::Int64(v), DType::Str) => {
            let data: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            make_col(col, ColumnData::Str(data))
        }

        // Float64 → Str
        (ColumnData::Float64(v), DType::Str) => {
            let data: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            make_col(col, ColumnData::Str(data))
        }

        // Str → Int64
        (ColumnData::Str(v), DType::Int64) => {
            let mut data = Vec::with_capacity(v.len());
            for (i, s) in v.iter().enumerate() {
                if col.is_null(i) {
                    data.push(0i64); // placeholder
                } else {
                    let parsed: i64 = s.parse().map_err(|_| {
                        PandasError::ParseError(format!("cannot parse {:?} as Int64", s))
                    })?;
                    data.push(parsed);
                }
            }
            make_col(col, ColumnData::Int64(data))
        }

        // Str → Float64
        (ColumnData::Str(v), DType::Float64) => {
            let mut data = Vec::with_capacity(v.len());
            for (i, s) in v.iter().enumerate() {
                if col.is_null(i) {
                    data.push(0.0f64); // placeholder
                } else {
                    let parsed: f64 = s.parse().map_err(|_| {
                        PandasError::ParseError(format!("cannot parse {:?} as Float64", s))
                    })?;
                    data.push(parsed);
                }
            }
            make_col(col, ColumnData::Float64(data))
        }

        // Str → Bool
        (ColumnData::Str(v), DType::Bool) => {
            let mut data = Vec::with_capacity(v.len());
            for (i, s) in v.iter().enumerate() {
                if col.is_null(i) {
                    data.push(false); // placeholder
                } else {
                    let b = match s.as_str() {
                        "true" | "True" | "1" => true,
                        "false" | "False" | "0" => false,
                        other => {
                            return Err(PandasError::ParseError(format!(
                                "cannot parse {:?} as Bool",
                                other
                            )))
                        }
                    };
                    data.push(b);
                }
            }
            make_col(col, ColumnData::Bool(data))
        }

        // Float64 → Bool: not supported
        (ColumnData::Float64(_), DType::Bool) => Err(PandasError::TypeError(
            "cannot cast Float64 to Bool".to_string(),
        )),

        _ => Err(PandasError::TypeError(format!(
            "cannot cast {:?} to {:?}",
            col.dtype(),
            target
        ))),
    }
}

/// Helper: build a new column with the same name and null mask as the original.
fn make_col(original: &Column, data: ColumnData) -> Result<Column> {
    match &original.null_mask {
        Some(mask) => Column::new_with_nulls(original.name(), data, mask.clone()),
        None => Ok(Column::new(original.name(), data)),
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

    fn bool_col(vals: Vec<bool>) -> Column {
        Column::new("x", ColumnData::Bool(vals))
    }

    fn str_col(vals: Vec<&str>) -> Column {
        Column::new(
            "x",
            ColumnData::Str(vals.into_iter().map(|s| s.into()).collect()),
        )
    }

    #[test]
    fn test_same_type_clone() {
        let col = int_col(vec![1, 2, 3]);
        let result = cast_column(&col, DType::Int64).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_int64_to_float64() {
        let col = int_col(vec![1, -2, 300]);
        let result = cast_column(&col, DType::Float64).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, -2.0, 300.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_float64_to_int64_truncate() {
        let col = float_col(vec![1.9, -2.7, 3.0]);
        let result = cast_column(&col, DType::Int64).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, -2, 3]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_float64_to_int64_nan_error() {
        let col = float_col(vec![f64::NAN]);
        assert!(cast_column(&col, DType::Int64).is_err());
    }

    #[test]
    fn test_float64_to_int64_inf_error() {
        let col = float_col(vec![f64::INFINITY]);
        assert!(cast_column(&col, DType::Int64).is_err());
    }

    #[test]
    fn test_bool_to_int64() {
        let col = bool_col(vec![true, false, true]);
        let result = cast_column(&col, DType::Int64).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[1, 0, 1]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_bool_to_float64() {
        let col = bool_col(vec![true, false]);
        let result = cast_column(&col, DType::Float64).unwrap();
        match &result.data {
            ColumnData::Float64(v) => assert_eq!(v, &[1.0, 0.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_int64_to_bool() {
        let col = int_col(vec![0, 1, -5, 0]);
        let result = cast_column(&col, DType::Bool).unwrap();
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[false, true, true, false]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_bool_to_str() {
        let col = bool_col(vec![true, false]);
        let result = cast_column(&col, DType::Str).unwrap();
        match &result.data {
            ColumnData::Str(v) => assert_eq!(v, &["True", "False"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_int64_to_str() {
        let col = int_col(vec![42, -7]);
        let result = cast_column(&col, DType::Str).unwrap();
        match &result.data {
            ColumnData::Str(v) => assert_eq!(v, &["42", "-7"]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_float64_to_str() {
        let col = float_col(vec![1.5]);
        let result = cast_column(&col, DType::Str).unwrap();
        assert_eq!(result.dtype(), DType::Str);
    }

    #[test]
    fn test_str_to_int64() {
        let col = str_col(vec!["42", "-7", "0"]);
        let result = cast_column(&col, DType::Int64).unwrap();
        match &result.data {
            ColumnData::Int64(v) => assert_eq!(v, &[42, -7, 0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_str_to_int64_parse_error() {
        let col = str_col(vec!["abc"]);
        assert!(cast_column(&col, DType::Int64).is_err());
    }

    #[test]
    fn test_str_to_float64() {
        let col = str_col(vec!["3.14", "-1.0"]);
        let result = cast_column(&col, DType::Float64).unwrap();
        match &result.data {
            ColumnData::Float64(v) => {
                assert!((v[0] - 3.14).abs() < 1e-10);
                assert_eq!(v[1], -1.0);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_str_to_float64_parse_error() {
        let col = str_col(vec!["not_a_float"]);
        assert!(cast_column(&col, DType::Float64).is_err());
    }

    #[test]
    fn test_str_to_bool() {
        let col = str_col(vec!["true", "False", "1", "0", "True", "false"]);
        let result = cast_column(&col, DType::Bool).unwrap();
        match &result.data {
            ColumnData::Bool(v) => assert_eq!(v, &[true, false, true, false, true, false]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_str_to_bool_parse_error() {
        let col = str_col(vec!["yes"]);
        assert!(cast_column(&col, DType::Bool).is_err());
    }

    #[test]
    fn test_float64_to_bool_error() {
        let col = float_col(vec![1.0, 0.0]);
        assert!(cast_column(&col, DType::Bool).is_err());
    }

    #[test]
    fn test_null_preserved_int64_to_float64() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 99, 3]),
            vec![false, true, false],
        )
        .unwrap();
        let result = cast_column(&col, DType::Float64).unwrap();
        assert!(result.is_null(1));
        assert!(!result.is_null(0));
        assert!(!result.is_null(2));
        match &result.data {
            ColumnData::Float64(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[2], 3.0);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_null_preserved_str_to_int64() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Str(vec!["1".into(), "".into(), "3".into()]),
            vec![false, true, false],
        )
        .unwrap();
        let result = cast_column(&col, DType::Int64).unwrap();
        assert!(result.is_null(1));
        assert!(!result.is_null(0));
        match &result.data {
            ColumnData::Int64(v) => {
                assert_eq!(v[0], 1);
                assert_eq!(v[2], 3);
            }
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_float64_null_skips_nan_check() {
        // A null NaN value should not cause an error when casting to Int64
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Float64(vec![f64::NAN]),
            vec![true], // it's null, not a real NaN
        )
        .unwrap();
        let result = cast_column(&col, DType::Int64).unwrap();
        assert!(result.is_null(0));
    }
}
