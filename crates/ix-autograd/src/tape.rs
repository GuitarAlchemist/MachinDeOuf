//! Wengert-style reverse-mode tape.
//!
//! Day 1 scaffold: types and skeleton API only. Day 2 fills in the
//! backward walk and the finite-difference verifier that consumes it.

use crate::mode::ExecutionMode;
use crate::tensor::Tensor;
use std::any::Any;
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
///
/// Day 3 refactor (per r7-day2-review.md §3.2): `tool_state` is now a
/// typed `Box<dyn Any>` map instead of `serde_json::Value`. Tools
/// serialize concrete state types without JSON round-tripping. The
/// type parameter on `set_tool_state` / `get_tool_state` ensures the
/// read side recovers the same type the write side stored.
pub struct DiffContext {
    pub tape: Tape,
    pub mode: ExecutionMode,
    tool_state: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for DiffContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffContext")
            .field("tape", &self.tape)
            .field("mode", &self.mode)
            .field("tool_state_keys", &self.tool_state.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl DiffContext {
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            tape: Tape::new(),
            mode,
            tool_state: HashMap::new(),
        }
    }

    /// Store a typed value in the tool-state bag. Overwrites any
    /// previous value stored under the same key.
    pub fn set_tool_state<T>(&mut self, key: impl Into<String>, value: T)
    where
        T: Any + Send + Sync,
    {
        self.tool_state.insert(key.into(), Box::new(value));
    }

    /// Retrieve a typed reference from the tool-state bag. Returns
    /// `None` if the key is missing or if the stored type does not
    /// match `T`.
    pub fn get_tool_state<T>(&self, key: &str) -> Option<&T>
    where
        T: Any + Send + Sync,
    {
        self.tool_state.get(key).and_then(|b| b.downcast_ref::<T>())
    }

    /// Remove and return a typed value from the tool-state bag. Returns
    /// `None` if the key is missing or if the stored type does not
    /// match `T`.
    pub fn take_tool_state<T>(&mut self, key: &str) -> Option<T>
    where
        T: Any + Send + Sync,
    {
        let boxed = self.tool_state.remove(key)?;
        match boxed.downcast::<T>() {
            Ok(b) => Some(*b),
            Err(reinsert) => {
                // Type mismatch — put it back and return None.
                self.tool_state.insert(key.to_string(), reinsert);
                None
            }
        }
    }
}
