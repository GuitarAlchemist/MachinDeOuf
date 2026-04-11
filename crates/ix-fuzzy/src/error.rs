//! Error types for [`crate::FuzzyDistribution`] construction and
//! operation failures.

use thiserror::Error;

/// Error produced by fuzzy-distribution construction or operation.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum FuzzyError {
    /// A membership value was outside `[0.0, 1.0]`.
    #[error("membership {value} out of range [0.0, 1.0]")]
    OutOfRange {
        /// The offending value.
        value: f64,
    },

    /// A membership value was `NaN` or infinite.
    #[error("membership must be finite, got {value}")]
    NotFinite {
        /// The offending value.
        value: f64,
    },

    /// A distribution's memberships did not sum to `1.0` within the
    /// documented tolerance (`±0.01`).
    #[error("memberships must sum to 1.0 ± 0.01, got {sum}")]
    BadSum {
        /// The actual sum.
        sum: f64,
    },

    /// A distribution was constructed without any variants.
    #[error("distribution has no variants")]
    Empty,
}
