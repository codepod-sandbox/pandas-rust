use std::fmt;

/// Data types supported by pandas-rust columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int64,
    Float64,
    Str,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int64 => write!(f, "int64"),
            DType::Float64 => write!(f, "float64"),
            DType::Str => write!(f, "object"),
        }
    }
}

impl DType {
    /// Parse a dtype string (e.g., "int64", "float64", "bool", "object"/"str").
    pub fn from_str(s: &str) -> Option<DType> {
        match s {
            "bool" => Some(DType::Bool),
            "int64" | "int" | "i64" => Some(DType::Int64),
            "float64" | "float" | "f64" => Some(DType::Float64),
            "object" | "str" | "string" => Some(DType::Str),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::Bool.to_string(), "bool");
        assert_eq!(DType::Int64.to_string(), "int64");
        assert_eq!(DType::Float64.to_string(), "float64");
        assert_eq!(DType::Str.to_string(), "object");
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("int64"), Some(DType::Int64));
        assert_eq!(DType::from_str("float"), Some(DType::Float64));
        assert_eq!(DType::from_str("object"), Some(DType::Str));
        assert_eq!(DType::from_str("str"), Some(DType::Str));
        assert_eq!(DType::from_str("unknown"), None);
    }
}
