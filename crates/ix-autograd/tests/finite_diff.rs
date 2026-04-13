//! Finite-difference verifier for `DifferentiableTool` implementations.
//!
//! This file is Day 1's stub. Day 2 of Week 1 fills in the real verifier.
//! It exists on Day 1 so that the file is committed and wired into CI
//! from the start, even before the real verifier is implemented.
//!
//! # Day 2 plan
//!
//! ```text
//! pub fn verify_gradient<T: DifferentiableTool>(
//!     tool: &T,
//!     inputs: &ValueMap,
//!     epsilon: f64,
//!     tolerance: f64,
//! ) -> anyhow::Result<()>
//! ```
//!
//! For each scalar element of each input tensor:
//!   1. Perturb by +epsilon, run forward, record output.
//!   2. Perturb by -epsilon, run forward, record output.
//!   3. Central finite difference: (fplus - fminus) / (2 * epsilon).
//!   4. Analytical gradient via backward(ones_like(output)).
//!   5. Assert max_abs_diff(numerical, analytical) <= tolerance.
//!
//! Verifier must pass for add, mul, sum, matmul by end of Day 2.

use ix_autograd::prelude::*;

#[test]
fn stub_compiles_and_imports_crate() {
    // Confirms the crate exports the prelude and the public API shape
    // compiles. Day 2 replaces this with real verification tests.
    let mode = ExecutionMode::VerifyFiniteDiff;
    assert!(mode.requires_tape());
    assert!(!mode.allows_non_diff());
}

#[test]
fn execution_modes_have_expected_semantics() {
    assert!(!ExecutionMode::Eager.requires_tape());
    assert!(ExecutionMode::Train.requires_tape());
    assert!(ExecutionMode::Mixed.requires_tape());
    assert!(ExecutionMode::Mixed.allows_non_diff());
    assert!(!ExecutionMode::Train.allows_non_diff());
}

#[test]
fn tensor_from_array_roundtrip() {
    use ndarray::array;
    let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let t = Tensor::from_array(a.clone());
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.as_f64(), &a);
    assert!(!t.requires_grad);

    let t2 = Tensor::from_array_with_grad(a);
    assert!(t2.requires_grad);
}
