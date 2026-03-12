//! # machin-optimize
//!
//! Optimization algorithms: gradient descent variants, simulated annealing,
//! particle swarm optimization, and convergence utilities.

pub mod traits;
pub mod gradient;
pub mod annealing;
pub mod pso;
pub mod convergence;

pub use traits::{ObjectiveFunction, Optimizer};
