pub mod error;
pub mod tensor;
pub mod conductance;
pub mod vlmm;
pub mod sampler;
pub mod consolidator;
pub mod engine;
pub mod serde_state;

#[cfg(feature = "reservoir")]
pub mod reservoir;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use error::{MemristiveError, Result};
pub use tensor::MarkovTensor;
pub use conductance::ConductanceMatrix;
pub use vlmm::VariableOrderSelector;
pub use sampler::SamplingStrategy;
pub use consolidator::MemoryConsolidator;
pub mod governance;

pub use engine::MemristiveEngine;
pub use governance::{GovernanceState, GovernanceMarkov};
