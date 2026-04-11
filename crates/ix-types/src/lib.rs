//! Shared type lattice for the ix skill registry.
//!
//! Every `#[ix_skill]`-annotated function accepts and returns values that
//! serialize through this crate's [`Value`] enum. Socket compatibility between
//! pipeline nodes is checked via [`SocketType`], and governance verdicts flow
//! through [`Tetravalent`]. The [`FromValue`] / [`IntoValue`] trait pair is the
//! glue between native Rust types and the universal `Value`.

use ndarray::{Array1, Array2};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod traits;
pub use traits::{FromValue, IntoValue};

/// Six-valued truth for governance beliefs — the ecosystem-wide standard
/// defined in `governance/demerzel/logic/hexavalent-logic.md`.
///
/// Extends classical tetravalent logic (T/F/U/C) with two evidential gradient
/// values that separate **direction of evidence** from **sufficiency**:
///
/// | Symbol | Name          | Meaning                                 |
/// |--------|---------------|-----------------------------------------|
/// | T      | True          | Verified with sufficient evidence       |
/// | P      | Probable      | Evidence leans true, not yet verified   |
/// | U      | Unknown       | Insufficient evidence to determine      |
/// | D      | Doubtful      | Evidence leans false, not yet refuted   |
/// | F      | False         | Refuted with sufficient evidence        |
/// | C      | Contradictory | Evidence supports both true and false   |
///
/// Serialized as the single-letter symbol (`"T"` / `"P"` / …) for cross-repo
/// wire compatibility with Demerzel's `hexavalent-state.schema.json`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
pub enum Hexavalent {
    #[serde(rename = "T")]
    True,
    #[serde(rename = "P")]
    Probable,
    #[serde(rename = "U")]
    Unknown,
    #[serde(rename = "D")]
    Doubtful,
    #[serde(rename = "F")]
    False,
    #[serde(rename = "C")]
    Contradictory,
}

impl Hexavalent {
    /// Single-letter symbol (`T`/`P`/`U`/`D`/`F`/`C`) — wire format.
    pub const fn symbol(self) -> char {
        match self {
            Hexavalent::True => 'T',
            Hexavalent::Probable => 'P',
            Hexavalent::Unknown => 'U',
            Hexavalent::Doubtful => 'D',
            Hexavalent::False => 'F',
            Hexavalent::Contradictory => 'C',
        }
    }

    /// Hexavalent NOT: T↔F, P↔D, U→U, C→C.
    pub const fn not(self) -> Self {
        use Hexavalent::*;
        match self {
            True => False,
            Probable => Doubtful,
            Unknown => Unknown,
            Doubtful => Probable,
            False => True,
            Contradictory => Contradictory,
        }
    }

    /// Hexavalent AND — per `hexavalent-logic.md` truth table.
    /// F absorbs everything; P demotes T to P; D demotes T/P to D.
    pub const fn and(self, other: Self) -> Self {
        use Hexavalent::*;
        match (self, other) {
            // F is absorbing
            (False, _) | (_, False) => False,
            // Anything AND T = itself
            (x, True) => x,
            (True, x) => x,
            // P row/col
            (Probable, Probable) => Probable,
            (Probable, Unknown) | (Unknown, Probable) => Unknown,
            (Probable, Doubtful) | (Doubtful, Probable) => Doubtful,
            (Probable, Contradictory) | (Contradictory, Probable) => Contradictory,
            // U row/col (excluding pairs handled above)
            (Unknown, Unknown) => Unknown,
            (Unknown, Doubtful) | (Doubtful, Unknown) => Unknown,
            (Unknown, Contradictory) | (Contradictory, Unknown) => Contradictory,
            // D row/col
            (Doubtful, Doubtful) => Doubtful,
            (Doubtful, Contradictory) | (Contradictory, Doubtful) => Contradictory,
            // C row/col
            (Contradictory, Contradictory) => Contradictory,
        }
    }

    /// Hexavalent OR — per `hexavalent-logic.md` truth table (derived from
    /// AND via De Morgan: `a OR b = NOT(NOT a AND NOT b)`).
    pub const fn or(self, other: Self) -> Self {
        self.not().and(other.not()).not()
    }

    /// All six values in canonical lattice order (T, P, U, D, F, C).
    pub const fn all() -> [Hexavalent; 6] {
        [
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ]
    }

    /// Is this a "definite" value (T or F)?
    pub const fn is_definite(self) -> bool {
        matches!(self, Hexavalent::True | Hexavalent::False)
    }

    /// Does this value carry evidential direction (P, D, T, F)?
    pub const fn is_directed(self) -> bool {
        matches!(
            self,
            Hexavalent::True | Hexavalent::Probable | Hexavalent::Doubtful | Hexavalent::False
        )
    }
}

impl std::fmt::Display for Hexavalent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Hexavalent::True => "T",
            Hexavalent::Probable => "P",
            Hexavalent::Unknown => "U",
            Hexavalent::Doubtful => "D",
            Hexavalent::False => "F",
            Hexavalent::Contradictory => "C",
        })
    }
}

/// Universal value lattice. All skill inputs and outputs are carried through
/// this enum when they cross the registry boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "data")]
pub enum Value {
    Null,
    Scalar(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Bytes(Vec<u8>),
    Vector(Vec<f64>),
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    Belief(Hexavalent),
    Json(serde_json::Value),
}

/// Static socket-type tag compared at both registration and runtime for edge
/// compatibility in the visual pipeline editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum SocketType {
    Any,
    Scalar,
    Integer,
    Bool,
    Text,
    Bytes,
    Vector,
    Matrix,
    Belief,
    Json,
}

impl SocketType {
    /// Can a socket of type `self` feed into a socket of type `other`?
    ///
    /// `Any` matches everything. Exact matches pass. Safe widening is
    /// allowed: `Scalar → Vector` (broadcast), `Vector → Matrix` (row).
    pub const fn compatible_with(self, other: SocketType) -> bool {
        use SocketType::*;
        if matches!(self, Any) || matches!(other, Any) {
            return true;
        }
        if self as u8 == other as u8 {
            return true;
        }
        matches!(
            (self, other),
            (Scalar, Vector) | (Vector, Matrix) | (Integer, Scalar)
        )
    }
}

/// Type-mismatch error raised when a `Value` cannot be decoded as the expected
/// native type by a skill adapter.
#[derive(Debug, Error)]
#[error("type mismatch: expected {expected:?}, got {actual}")]
pub struct TypeError {
    pub expected: SocketType,
    pub actual: &'static str,
}

impl Value {
    /// Human-readable tag of the active variant — used in error messages.
    pub fn tag(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Scalar(_) => "scalar",
            Value::Integer(_) => "integer",
            Value::Bool(_) => "bool",
            Value::Text(_) => "text",
            Value::Bytes(_) => "bytes",
            Value::Vector(_) => "vector",
            Value::Matrix { .. } => "matrix",
            Value::Belief(_) => "belief",
            Value::Json(_) => "json",
        }
    }
}

// ---------------------------------------------------------------------------
// Newtypes wrapping ndarray for pipeline transport. Skills should use these
// in their signatures rather than raw ndarray types so the macro can emit
// concrete `FromValue`/`IntoValue` calls without orphan-rule issues.
// ---------------------------------------------------------------------------

/// Owned 1-D numeric vector wrapping `ndarray::Array1<f64>`.
#[derive(Debug, Clone)]
pub struct IxVector(pub Array1<f64>);

impl IxVector {
    pub fn new(data: Vec<f64>) -> Self {
        Self(Array1::from_vec(data))
    }
    pub fn into_inner(self) -> Array1<f64> {
        self.0
    }
}

impl From<Array1<f64>> for IxVector {
    fn from(a: Array1<f64>) -> Self {
        Self(a)
    }
}

impl From<Vec<f64>> for IxVector {
    fn from(v: Vec<f64>) -> Self {
        Self::new(v)
    }
}

/// Owned 2-D numeric matrix wrapping `ndarray::Array2<f64>`.
#[derive(Debug, Clone)]
pub struct IxMatrix(pub Array2<f64>);

impl IxMatrix {
    /// Construct from row-major flat data.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, TypeError> {
        Array2::from_shape_vec((rows, cols), data)
            .map(Self)
            .map_err(|_| TypeError {
                expected: SocketType::Matrix,
                actual: "matrix: row×col != data.len()",
            })
    }
    pub fn into_inner(self) -> Array2<f64> {
        self.0
    }
}

impl From<Array2<f64>> for IxMatrix {
    fn from(a: Array2<f64>) -> Self {
        Self(a)
    }
}
