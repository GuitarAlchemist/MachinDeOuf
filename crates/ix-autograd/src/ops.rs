//! Primitive differentiable operations.
//!
//! Day 1 scaffold: signatures only. Day 2 implements forward + backward
//! for add, mul, sum, matmul, and verifies them against finite differences.

use crate::tape::{DiffContext, TensorHandle};
use crate::Result;

/// Element-wise addition. `z = a + b`, broadcast-safe.
/// Backward: `dL/da = dL/dz`, `dL/db = dL/dz` (summed over broadcast dims).
pub fn add(_ctx: &mut DiffContext, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 2: implement element-wise add with reverse-mode backward")
}

/// Element-wise multiplication. `z = a * b`.
/// Backward: `dL/da = dL/dz * b`, `dL/db = dL/dz * a`.
pub fn mul(_ctx: &mut DiffContext, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 2: implement element-wise mul with reverse-mode backward")
}

/// Sum-reduction over all elements. `z = sum(a)`.
/// Backward: `dL/da = dL/dz * ones_like(a)`.
pub fn sum(_ctx: &mut DiffContext, _a: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 2: implement sum reduction with reverse-mode backward")
}

/// Matrix multiplication. `z = a @ b` for 2-D inputs.
/// Backward: `dL/da = dL/dz @ b^T`, `dL/db = a^T @ dL/dz`.
pub fn matmul(_ctx: &mut DiffContext, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 2: implement matmul with reverse-mode backward")
}

/// Mean reduction. `z = mean(a) = sum(a) / n`.
pub fn mean(_ctx: &mut DiffContext, _a: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 3: implement mean; depends on sum")
}

/// Variance. `z = mean((a - mean(a))^2)`.
pub fn variance(_ctx: &mut DiffContext, _a: TensorHandle) -> Result<TensorHandle> {
    todo!("Day 3: implement variance; depends on mean, mul, sum")
}
