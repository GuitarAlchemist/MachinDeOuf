//! Batch 1 — canary migration of 6 MCP tools to the capability registry.
//!
//! These wrappers delegate to the pre-existing `handlers::*` functions to
//! preserve exact MCP behavior; only the *registration surface* changes.
//! Each wrapper carries its hand-written JSON schema via `schema_fn = ...`.

use crate::handlers;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

// --- ix_stats --------------------------------------------------------------

fn stats_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "number" },
                "description": "List of numbers to compute statistics on"
            }
        },
        "required": ["data"]
    })
}

/// Compute statistics (mean, std, min, max, median) on a list of numbers.
#[ix_skill(
    domain = "math",
    name = "stats",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::stats_schema"
)]
pub fn stats(params: Value) -> Result<Value, String> {
    handlers::stats(params)
}

// --- ix_distance -----------------------------------------------------------

fn distance_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "a": { "type": "array", "items": { "type": "number" }, "description": "First vector" },
            "b": { "type": "array", "items": { "type": "number" }, "description": "Second vector" },
            "metric": {
                "type": "string",
                "enum": ["euclidean", "cosine", "manhattan"],
                "description": "Distance metric"
            }
        },
        "required": ["a", "b", "metric"]
    })
}

/// Compute distance between two vectors (euclidean, cosine, or manhattan).
#[ix_skill(
    domain = "math",
    name = "distance",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::distance_schema"
)]
pub fn distance(params: Value) -> Result<Value, String> {
    handlers::distance(params)
}

// --- ix_fft ----------------------------------------------------------------

fn fft_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "signal": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Real-valued input signal (length must be power of 2)"
            },
            "inverse": {
                "type": "boolean",
                "description": "If true, compute the inverse FFT",
                "default": false
            }
        },
        "required": ["signal"]
    })
}

/// Compute the Fast Fourier Transform of a real-valued signal.
#[ix_skill(
    domain = "signal",
    name = "fft",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::fft_schema"
)]
pub fn fft(params: Value) -> Result<Value, String> {
    handlers::fft(params)
}

// --- ix_kmeans -------------------------------------------------------------

fn kmeans_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Matrix where each row is a data point"
            },
            "k": { "type": "integer", "minimum": 1, "description": "Number of clusters" },
            "max_iter": { "type": "integer", "default": 100, "description": "Max iterations" },
            "seed": { "type": "integer", "default": 42, "description": "RNG seed" }
        },
        "required": ["data", "k"]
    })
}

/// Cluster points using K-Means.
#[ix_skill(
    domain = "unsupervised",
    name = "kmeans",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::kmeans_schema"
)]
pub fn kmeans(params: Value) -> Result<Value, String> {
    handlers::kmeans(params)
}

// --- ix_linear_regression --------------------------------------------------

fn linear_regression_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "X": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Feature matrix (each row an observation)"
            },
            "y": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Target vector"
            }
        },
        "required": ["X", "y"]
    })
}

/// Fit an ordinary least-squares linear regression model.
#[ix_skill(
    domain = "supervised",
    name = "linear_regression",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::linear_regression_schema"
)]
pub fn linear_regression(params: Value) -> Result<Value, String> {
    handlers::linear_regression(params)
}

// --- ix_governance_belief --------------------------------------------------

fn governance_belief_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "proposition": { "type": "string", "description": "Belief proposition text" },
            "truth_value": {
                "type": "string",
                "enum": ["T", "F", "U", "C"],
                "description": "Tetravalent truth (legacy). Hexavalent values P/D supported when present."
            },
            "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "supporting": { "type": "array", "items": { "type": "object" } },
            "contradicting": { "type": "array", "items": { "type": "object" } }
        },
        "required": ["proposition", "truth_value", "confidence"]
    })
}

/// Query the Demerzel belief engine with a belief state and receive a
/// resolved action recommendation.
#[ix_skill(
    domain = "governance",
    name = "governance.belief",
    governance = "safety,reversible",
    schema_fn = "crate::skills::batch1::governance_belief_schema"
)]
pub fn governance_belief(params: Value) -> Result<Value, String> {
    handlers::governance_belief(params)
}
