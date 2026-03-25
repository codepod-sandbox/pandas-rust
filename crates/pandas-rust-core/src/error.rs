use thiserror::Error;

#[derive(Error, Debug)]
pub enum PandasError {
    #[error("KeyError: {0}")]
    KeyError(String),

    #[error("IndexError: {0}")]
    IndexError(String),

    #[error("TypeError: {0}")]
    TypeError(String),

    #[error("ValueError: {0}")]
    ValueError(String),

    #[error("IoError: {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParseError: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, PandasError>;
