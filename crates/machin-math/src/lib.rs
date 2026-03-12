//! # machin-math
//!
//! Core math primitives for the machin ML toolkit.
//! Linear algebra, statistics, distances, activation functions, and numerical calculus.

pub mod linalg;
pub mod stats;
pub mod distance;
pub mod activation;
pub mod calculus;
pub mod random;
pub mod hyperbolic;
pub mod error;

pub use ndarray;
