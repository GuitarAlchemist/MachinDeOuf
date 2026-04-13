//! Tape-aware tensor type backed by `ndarray`.
//!
//! Codex review decision: stay ndarray-native at the boundary and
//! internally. The `TensorData` enum leaves an explicit future door for
//! an alternative backend (candle, burn, custom GPU) without forcing
//! every downstream crate to migrate.

use ndarray::ArrayD;

#[derive(Debug, Clone)]
pub enum TensorData {
    F64(ArrayD<f64>),
    // Future: Candle(candle_core::Tensor), Gpu(ix_gpu::Buffer), ...
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn from_array(array: ArrayD<f64>) -> Self {
        Self {
            data: TensorData::F64(array),
            requires_grad: false,
        }
    }

    pub fn from_array_with_grad(array: ArrayD<f64>) -> Self {
        Self {
            data: TensorData::F64(array),
            requires_grad: true,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self.data {
            TensorData::F64(a) => a.shape().to_vec(),
        }
    }

    pub fn as_f64(&self) -> &ArrayD<f64> {
        match &self.data {
            TensorData::F64(a) => a,
        }
    }
}
