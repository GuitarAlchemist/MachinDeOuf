//! Probabilistic grammar infrastructure for TARS integration.
//!
//! - [`weighted`] — Bayesian (Beta-Binomial) rule weights, softmax selection
//! - [`replicator`] — Grammar species replicator dynamics and ESS detection
//! - [`constrained`] — EBNF grammar loading + grammar-guided MCTS adapter

pub mod weighted;
pub mod replicator;
pub mod constrained;
