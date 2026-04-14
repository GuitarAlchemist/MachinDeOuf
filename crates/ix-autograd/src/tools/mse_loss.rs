//! `MseLossTool` — scalar mean-squared-error loss wrapped as a
//! `DifferentiableTool`.
//!
//! R7 Week 2: `LinearRegressionTool` folds the MSE computation into its
//! own forward graph, but most downstream training loops want to
//! compute loss separately from the model's output (so they can swap
//! losses, or compute multiple losses over the same prediction).
//! `MseLossTool` exposes MSE as a standalone differentiable primitive.
//!
//! Forward graph (all composed of existing primitives):
//!
//! ```text
//!     residual = pred - target
//!     sq       = residual * residual
//!     loss     = mean(sq)
//! ```
//!
//! Only `pred` receives a gradient. `target` is treated as an observed
//! constant — the reverse walker technically computes a gradient for it,
//! but we do not surface it on the output since it is never trainable.
//!
//! Inputs:  `pred`, `target` — same-shape tensors.
//! Outputs: `loss` — scalar tensor (rank 0).

use crate::ops;
use crate::tape::{DiffContext, TensorHandle};
use crate::tensor::Tensor;
use crate::tool::{DifferentiableTool, ValueMap};
use crate::{AutogradError, Result};

/// Scalar mean-squared-error loss over matching `pred` and `target`.
pub struct MseLossTool;

/// Typed tool state stored in `DiffContext` between forward and backward.
#[derive(Debug, Clone, Copy)]
pub struct MseLossState {
    /// Leaf handle for the prediction tensor.
    pub pred: TensorHandle,
    /// Leaf handle for the target (observed) tensor.
    pub target: TensorHandle,
    /// Node handle for the scalar MSE loss output.
    pub loss: TensorHandle,
}

const STATE_KEY: &str = "ix_mse_loss.last";

impl MseLossTool {
    /// Build the forward graph on the tape and return handles for the
    /// leaves and the scalar loss. Exposed so callers can use MSE as a
    /// sub-graph under a larger differentiable pipeline.
    pub fn build_graph(
        ctx: &mut DiffContext,
        pred: Tensor,
        target: Tensor,
    ) -> Result<MseLossState> {
        let pred_h = ops::input(ctx, pred);
        let target_h = ops::input(ctx, target);

        let residual = ops::sub(ctx, pred_h, target_h)?;
        let sq = ops::mul(ctx, residual, residual)?;
        let loss = ops::mean(ctx, sq)?;

        Ok(MseLossState {
            pred: pred_h,
            target: target_h,
            loss,
        })
    }
}

impl DifferentiableTool for MseLossTool {
    fn name(&self) -> &'static str {
        "ix_mse_loss"
    }

    fn forward(&self, ctx: &mut DiffContext, inputs: &ValueMap) -> Result<ValueMap> {
        let pred = inputs
            .get("pred")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("pred".into()))?;
        let target = inputs
            .get("target")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("target".into()))?;

        if pred.shape() != target.shape() {
            return Err(AutogradError::Numerical(format!(
                "mse_loss: pred shape {:?} != target shape {:?}",
                pred.shape(),
                target.shape()
            )));
        }

        let state = Self::build_graph(ctx, pred, target)?;
        ctx.set_tool_state(STATE_KEY, state);

        let loss_value = ctx
            .tape
            .get(state.loss)
            .ok_or(AutogradError::InvalidHandle(state.loss))?
            .value
            .clone();

        let mut out = ValueMap::new();
        out.insert("loss".into(), loss_value);
        Ok(out)
    }

    fn backward(&self, ctx: &mut DiffContext, _out_grads: &ValueMap) -> Result<ValueMap> {
        let state = *ctx
            .get_tool_state::<MseLossState>(STATE_KEY)
            .ok_or_else(|| AutogradError::MissingSaved(STATE_KEY.into()))?;

        let seed = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 1.0_f64);
        let grads = ctx.backward(state.loss, seed)?;

        let mut out = ValueMap::new();
        if let Some(g) = grads.get(&state.pred) {
            out.insert("pred".into(), Tensor::from_array(g.clone()));
        }
        // `target` is an observed constant; intentionally not surfaced.
        Ok(out)
    }
}
