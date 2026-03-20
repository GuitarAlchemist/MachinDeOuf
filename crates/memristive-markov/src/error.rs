use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemristiveError {
    #[error("deserialization failed: {0}")]
    Deserialize(#[from] serde_json::Error),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("invalid state index {index}: exceeds state_count {state_count}")]
    InvalidState { index: usize, state_count: usize },

    #[error("reservoir dimension mismatch: expected {expected}, got {got}")]
    ReservoirDimension { expected: usize, got: usize },

    #[error("engine not warmed up: need at least {needed} observations, have {have}")]
    ColdEngine { needed: usize, have: usize },

    #[cfg(feature = "gpu")]
    #[error("GPU initialization failed: {0}")]
    GpuInit(String),
}

pub type Result<T> = std::result::Result<T, MemristiveError>;
