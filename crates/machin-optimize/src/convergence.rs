//! Convergence criteria for optimization loops.

/// Convergence criteria configuration.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}
