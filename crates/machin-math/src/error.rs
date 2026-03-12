use thiserror::Error;

#[derive(Debug, Error)]
pub enum MathError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("matrix is not square: {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("matrix is singular (non-invertible)")]
    Singular,

    #[error("empty input")]
    EmptyInput,

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
