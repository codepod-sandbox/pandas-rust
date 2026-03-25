use crate::column::{Column, ColumnData};
use crate::dataframe::DataFrame;
use crate::dtype::DType;
use crate::error::{PandasError, Result};

/// Options for reading a CSV file.
pub struct CsvReadOptions {
    /// Field delimiter (default: `b','`).
    pub delimiter: u8,
    /// Whether the first row is a header (default: `true`).
    pub has_header: bool,
    /// If `Some`, select only these columns by name.
    pub columns: Option<Vec<String>>,
    /// Override the inferred dtype for named columns.
    pub dtype_overrides: Vec<(String, DType)>,
    /// Strings that should be treated as null/NA.
    pub na_values: Vec<String>,
    /// Maximum number of data rows to read (not counting header).
    pub max_rows: Option<usize>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        CsvReadOptions {
            delimiter: b',',
            has_header: true,
            columns: None,
            dtype_overrides: vec![],
            na_values: vec![
                "".into(),
                "NA".into(),
                "NaN".into(),
                "null".into(),
                "None".into(),
            ],
            max_rows: None,
        }
    }
}

/// Options for writing a CSV file.
pub struct CsvWriteOptions {
    /// Field delimiter (default: `b','`).
    pub delimiter: u8,
    /// Whether to write a header row (default: `true`).
    pub header: bool,
    /// String to use for null values (default: `""`).
    pub na_rep: String,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        CsvWriteOptions {
            delimiter: b',',
            header: true,
            na_rep: String::new(),
        }
    }
}

/// Read a CSV from any `std::io::Read` source.
pub fn read_csv<R: std::io::Read>(reader: R, options: CsvReadOptions) -> Result<DataFrame> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(options.delimiter)
        .has_headers(options.has_header)
        .flexible(true)
        .from_reader(reader);

    // Gather column names
    let header_names: Vec<String> = if options.has_header {
        rdr.headers()
            .map_err(|e| PandasError::ValueError(e.to_string()))?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        // Peek at the first record to know how many columns there are
        // We'll handle this below when reading records
        vec![]
    };

    // Read all records (respecting max_rows)
    let mut raw: Vec<Vec<String>> = Vec::new();
    for record in rdr.records() {
        let record = record.map_err(|e| PandasError::ValueError(e.to_string()))?;
        raw.push(record.iter().map(|s| s.to_string()).collect());
        if let Some(max) = options.max_rows {
            if raw.len() >= max {
                break;
            }
        }
    }

    // Determine column names
    let ncols = if !raw.is_empty() {
        raw[0].len()
    } else if !header_names.is_empty() {
        header_names.len()
    } else {
        return Ok(DataFrame::empty());
    };

    let all_col_names: Vec<String> = if options.has_header {
        header_names
    } else {
        (0..ncols).map(|i| i.to_string()).collect()
    };

    // Determine which columns to include
    let selected_indices: Vec<usize> = if let Some(ref wanted) = options.columns {
        let mut idxs = Vec::new();
        for name in wanted {
            match all_col_names.iter().position(|n| n == name) {
                Some(i) => idxs.push(i),
                None => {
                    return Err(PandasError::KeyError(format!(
                        "column '{}' not found in CSV",
                        name
                    )))
                }
            }
        }
        idxs
    } else {
        (0..ncols).collect()
    };

    let selected_names: Vec<&str> = selected_indices
        .iter()
        .map(|&i| all_col_names[i].as_str())
        .collect();

    // Build raw column data (Vec<Vec<String>>), one inner vec per column
    let nrows = raw.len();
    let mut col_raw: Vec<Vec<String>> = selected_indices
        .iter()
        .map(|&ci| {
            raw.iter()
                .map(|row| row.get(ci).cloned().unwrap_or_default())
                .collect()
        })
        .collect();

    // Build dtype override map
    let dtype_override_map: std::collections::HashMap<&str, DType> = options
        .dtype_overrides
        .iter()
        .map(|(n, d)| (n.as_str(), *d))
        .collect();

    // Build columns
    let mut columns: Vec<Column> = Vec::with_capacity(selected_names.len());

    for (col_idx, &col_name) in selected_names.iter().enumerate() {
        let raw_vals = &mut col_raw[col_idx];

        // Mark NA values as a sentinel
        let mut null_mask = vec![false; nrows];
        for (i, val) in raw_vals.iter().enumerate() {
            if options.na_values.iter().any(|na| na == val) {
                null_mask[i] = true;
            }
        }

        // Determine dtype
        let dtype = if let Some(&dt) = dtype_override_map.get(col_name) {
            dt
        } else {
            infer_dtype(raw_vals, &null_mask)
        };

        let col = build_column(col_name, dtype, raw_vals, &null_mask)?;
        columns.push(col);
    }

    if columns.is_empty() {
        return Ok(DataFrame::empty());
    }

    DataFrame::from_columns(columns)
}

/// Infer the best dtype for a column by trying to parse all non-null values.
fn infer_dtype(vals: &[String], null_mask: &[bool]) -> DType {
    let non_null: Vec<&str> = vals
        .iter()
        .zip(null_mask.iter())
        .filter(|(_, &is_null)| !is_null)
        .map(|(v, _)| v.as_str())
        .collect();

    if non_null.is_empty() {
        return DType::Str; // fallback
    }

    // Try bool
    if non_null
        .iter()
        .all(|v| matches!(v.to_lowercase().as_str(), "true" | "false"))
    {
        return DType::Bool;
    }

    // Try int64
    if non_null.iter().all(|v| v.parse::<i64>().is_ok()) {
        return DType::Int64;
    }

    // Try float64
    if non_null.iter().all(|v| v.parse::<f64>().is_ok()) {
        return DType::Float64;
    }

    DType::Str
}

/// Build a Column from raw string values, given a dtype and null mask.
fn build_column(name: &str, dtype: DType, vals: &[String], null_mask: &[bool]) -> Result<Column> {
    match dtype {
        DType::Bool => {
            let data: Vec<bool> = vals
                .iter()
                .zip(null_mask.iter())
                .map(|(v, &is_null)| {
                    if is_null {
                        false
                    } else {
                        v.to_lowercase() == "true"
                    }
                })
                .collect();
            Column::new_with_nulls(name, ColumnData::Bool(data), null_mask.to_vec())
        }
        DType::Int64 => {
            let mut data: Vec<i64> = Vec::with_capacity(vals.len());
            for (v, &is_null) in vals.iter().zip(null_mask.iter()) {
                if is_null {
                    data.push(0);
                } else {
                    let parsed = v.parse::<i64>().map_err(|_| {
                        PandasError::ValueError(format!("cannot parse '{}' as int64", v))
                    })?;
                    data.push(parsed);
                }
            }
            Column::new_with_nulls(name, ColumnData::Int64(data), null_mask.to_vec())
        }
        DType::Float64 => {
            let mut data: Vec<f64> = Vec::with_capacity(vals.len());
            for (v, &is_null) in vals.iter().zip(null_mask.iter()) {
                if is_null {
                    data.push(0.0);
                } else {
                    let parsed = v.parse::<f64>().map_err(|_| {
                        PandasError::ValueError(format!("cannot parse '{}' as float64", v))
                    })?;
                    data.push(parsed);
                }
            }
            Column::new_with_nulls(name, ColumnData::Float64(data), null_mask.to_vec())
        }
        DType::Str => {
            let data: Vec<String> = vals
                .iter()
                .zip(null_mask.iter())
                .map(|(v, &is_null)| if is_null { String::new() } else { v.clone() })
                .collect();
            Column::new_with_nulls(name, ColumnData::Str(data), null_mask.to_vec())
        }
    }
}

/// Write a DataFrame to CSV.
pub fn to_csv<W: std::io::Write>(
    df: &DataFrame,
    writer: W,
    options: CsvWriteOptions,
) -> Result<()> {
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(options.delimiter)
        .from_writer(writer);

    let col_names = df.column_names();

    // Write header
    if options.header {
        wtr.write_record(&col_names)
            .map_err(|e| PandasError::ValueError(e.to_string()))?;
    }

    let ncols = col_names.len();
    let nrows = df.nrows();

    // Fetch columns
    let cols: Vec<&Column> = col_names
        .iter()
        .map(|&n| df.get_column(n))
        .collect::<Result<Vec<_>>>()?;

    for row in 0..nrows {
        let mut record: Vec<String> = Vec::with_capacity(ncols);
        for (ci, col) in cols.iter().enumerate() {
            let _ = ci;
            if col.is_null(row) {
                record.push(options.na_rep.clone());
            } else {
                let val = match &col.data {
                    ColumnData::Bool(v) => v[row].to_string(),
                    ColumnData::Int64(v) => v[row].to_string(),
                    ColumnData::Float64(v) => v[row].to_string(),
                    ColumnData::Str(v) => v[row].clone(),
                };
                record.push(val);
            }
        }
        wtr.write_record(&record)
            .map_err(|e| PandasError::ValueError(e.to_string()))?;
    }

    wtr.flush()
        .map_err(|e| PandasError::ValueError(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;

    fn read_str(s: &str) -> DataFrame {
        read_csv(s.as_bytes(), CsvReadOptions::default()).unwrap()
    }

    #[test]
    fn test_basic_csv_with_header() {
        let csv = "name,age,score\nalice,30,1.5\nbob,25,2.0\n";
        let df = read_str(csv);
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.column_names(), vec!["name", "age", "score"]);
        assert_eq!(df.get_column("age").unwrap().dtype(), DType::Int64);
        assert_eq!(df.get_column("score").unwrap().dtype(), DType::Float64);
        assert_eq!(df.get_column("name").unwrap().dtype(), DType::Str);
    }

    #[test]
    fn test_csv_without_header() {
        let csv = "1,2,3\n4,5,6\n";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                has_header: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.column_names(), vec!["0", "1", "2"]);
    }

    #[test]
    fn test_dtype_inference_int() {
        let csv = "x\n1\n2\n3\n";
        let df = read_str(csv);
        assert_eq!(df.get_column("x").unwrap().dtype(), DType::Int64);
    }

    #[test]
    fn test_dtype_inference_float() {
        let csv = "x\n1.0\n2.5\n3.14\n";
        let df = read_str(csv);
        assert_eq!(df.get_column("x").unwrap().dtype(), DType::Float64);
    }

    #[test]
    fn test_dtype_inference_bool() {
        let csv = "x\ntrue\nfalse\ntrue\n";
        let df = read_str(csv);
        assert_eq!(df.get_column("x").unwrap().dtype(), DType::Bool);
    }

    #[test]
    fn test_dtype_inference_mixed_becomes_str() {
        let csv = "x\n1\nhello\n3\n";
        let df = read_str(csv);
        assert_eq!(df.get_column("x").unwrap().dtype(), DType::Str);
    }

    #[test]
    fn test_na_values_detection() {
        let csv = "x\n1\nNA\n3\n";
        let df = read_str(csv);
        let col = df.get_column("x").unwrap();
        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));
    }

    #[test]
    fn test_empty_string_is_null() {
        // Use a multi-column CSV where the second row has an empty field for "x"
        let csv = "x,y\n1,a\n,b\n3,c\n";
        let df = read_str(csv);
        let col = df.get_column("x").unwrap();
        assert!(!col.is_null(0));
        assert!(col.is_null(1)); // empty field "" → null
        assert!(!col.is_null(2));
    }

    #[test]
    fn test_custom_delimiter_tab() {
        let csv = "a\tb\n1\t2\n3\t4\n";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                delimiter: b'\t',
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.column_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_column_selection() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                columns: Some(vec!["a".into(), "c".into()]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.column_names(), vec!["a", "c"]);
    }

    #[test]
    fn test_dtype_override() {
        let csv = "x\n1\n2\n3\n";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                dtype_overrides: vec![("x".into(), DType::Float64)],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.get_column("x").unwrap().dtype(), DType::Float64);
    }

    #[test]
    fn test_max_rows() {
        let csv = "x\n1\n2\n3\n4\n5\n";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                max_rows: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.nrows(), 3);
    }

    #[test]
    fn test_round_trip() {
        let original = DataFrame::from_columns(vec![
            Column::new("name", ColumnData::Str(vec!["alice".into(), "bob".into()])),
            Column::new("age", ColumnData::Int64(vec![30, 25])),
            Column::new("score", ColumnData::Float64(vec![1.5, 2.0])),
        ])
        .unwrap();

        let mut buf = Vec::new();
        to_csv(&original, &mut buf, CsvWriteOptions::default()).unwrap();
        let csv_str = String::from_utf8(buf).unwrap();

        let restored = read_str(&csv_str);
        assert_eq!(restored.nrows(), 2);
        assert_eq!(restored.column_names(), vec!["name", "age", "score"]);
        assert_eq!(restored.get_column("age").unwrap().dtype(), DType::Int64);
    }

    #[test]
    fn test_empty_file() {
        let csv = "";
        let df = read_csv(
            csv.as_bytes(),
            CsvReadOptions {
                has_header: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
    }

    #[test]
    fn test_header_only_empty_data() {
        let csv = "a,b,c\n";
        let df = read_str(csv);
        assert_eq!(df.nrows(), 0);
    }

    #[test]
    fn test_to_csv_with_custom_na_rep() {
        let col = Column::new_with_nulls(
            "x",
            ColumnData::Int64(vec![1, 0, 3]),
            vec![false, true, false],
        )
        .unwrap();
        let df = DataFrame::from_columns(vec![col]).unwrap();

        let mut buf = Vec::new();
        to_csv(
            &df,
            &mut buf,
            CsvWriteOptions {
                na_rep: "N/A".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("N/A"));
    }

    #[test]
    fn test_to_csv_no_header() {
        let df =
            DataFrame::from_columns(vec![Column::new("x", ColumnData::Int64(vec![1, 2]))]).unwrap();

        let mut buf = Vec::new();
        to_csv(
            &df,
            &mut buf,
            CsvWriteOptions {
                header: false,
                ..Default::default()
            },
        )
        .unwrap();

        let s = String::from_utf8(buf).unwrap();
        assert!(!s.contains("x"));
        assert!(s.contains("1"));
        assert!(s.contains("2"));
    }
}
