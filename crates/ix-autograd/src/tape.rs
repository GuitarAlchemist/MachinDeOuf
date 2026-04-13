//! Wengert-style reverse-mode tape.
//!
//! Day 1 scaffold: types and skeleton API only. Day 2 fills in the
//! backward walk and the finite-difference verifier that consumes it.

use crate::mode::ExecutionMode;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Opaque index into the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorHandle(pub usize);

/// A single node on the Wengert tape.
#[derive(Debug)]
pub struct TapeNode {
    pub op: &'static str,
    pub inputs: Vec<TensorHandle>,
    pub value: Tensor,
    pub grad: Option<Tensor>,
    /// Tool-specific saved state used by `backward`. JSON for now so
    /// tools can record whatever they need without coupling to this crate.
    pub saved: Option<serde_json::Value>,
}

/// The Wengert tape. Append-only during forward, walked in reverse
/// during backward.
#[derive(Debug, Default)]
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, node: TapeNode) -> TensorHandle {
        let id = self.nodes.len();
        self.nodes.push(node);
        TensorHandle(id)
    }

    pub fn get(&self, handle: TensorHandle) -> Option<&TapeNode> {
        self.nodes.get(handle.0)
    }

    pub fn get_mut(&mut self, handle: TensorHandle) -> Option<&mut TapeNode> {
        self.nodes.get_mut(handle.0)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Runtime context threaded through every `DifferentiableTool::forward`
/// and `DifferentiableTool::backward` call. Carries the tape, the
/// execution mode, and a bag of tool-scoped state.
#[derive(Debug)]
pub struct DiffContext {
    pub tape: Tape,
    pub mode: ExecutionMode,
    pub tool_state: HashMap<String, serde_json::Value>,
}

impl DiffContext {
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            tape: Tape::new(),
            mode,
            tool_state: HashMap::new(),
        }
    }
}
