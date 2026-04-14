//! `StatsMeanTool` — scalar mean reduction wrapped as a
//! `DifferentiableTool`.
//!
//! R7 Week 2: the simplest possible differentiable tool. It exists
//! primarily to exercise the wrapping pattern for pure compositions of
//! primitive ops — the tape is one `mean` call deep, and backward flows
//! through `sum → div_scalar → input` in the reverse walker without any
//! tool-specific backward logic.
//!
//! Inputs:  `x` — any-rank tensor.
//! Outputs: `mean` — scalar tensor (rank 0).

use crate::ops;
use crate::tape::{DiffContext, TensorHandle};
use crate::tensor::Tensor;
use crate::tool::{DifferentiableTool, ValueMap};
use crate::{AutogradError, Result};

/// Scalar mean reduction over a single tensor input.
pub struct StatsMeanTool;

/// Typed tool state stored in `DiffContext` between forward and backward.
#[derive(Debug, Clone, Copy)]
pub struct StatsMeanState {
    /// Leaf handle for the input tensor `x`.
    pub x: TensorHandle,
    /// Node handle for the scalar mean output.
    pub mean: TensorHandle,
}

const STATE_KEY: &str = "ix_stats_mean.last";

impl DifferentiableTool for StatsMeanTool {
    fn name(&self) -> &'static str {
        "ix_stats_mean"
    }

    fn forward(&self, ctx: &mut DiffContext, inputs: &ValueMap) -> Result<ValueMap> {
        let x = inputs
            .get("x")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("x".into()))?;
        let x_h = ops::input(ctx, x);
        let mean = ops::mean(ctx, x_h)?;

        ctx.set_tool_state(STATE_KEY, StatsMeanState { x: x_h, mean });

        let mean_value = ctx
            .tape
            .get(mean)
            .ok_or(AutogradError::InvalidHandle(mean))?
            .value
            .clone();

        let mut out = ValueMap::new();
        out.insert("mean".into(), mean_value);
        Ok(out)
    }

    fn backward(&self, ctx: &mut DiffContext, _out_grads: &ValueMap) -> Result<ValueMap> {
        let state = *ctx
            .get_tool_state::<StatsMeanState>(STATE_KEY)
            .ok_or_else(|| AutogradError::MissingSaved(STATE_KEY.into()))?;

        let seed = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 1.0_f64);
        let grads = ctx.backward(state.mean, seed)?;

        let mut out = ValueMap::new();
        if let Some(g) = grads.get(&state.x) {
            out.insert("x".into(), Tensor::from_array(g.clone()));
        }
        Ok(out)
    }
}
