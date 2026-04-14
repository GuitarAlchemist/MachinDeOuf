//! Probabilistic grammar infrastructure for TARS integration.
//!
//! - [`weighted`] — Bayesian (Beta-Binomial) rule weights, softmax selection
//! - [`replicator`] — Grammar species replicator dynamics and ESS detection
//! - [`constrained`] — EBNF grammar loading + grammar-guided MCTS adapter
//! - [`catalog`] — curated index of real-world grammar sources (EBNF, ABNF, PEG, ...)
//! - [`ebnf`] — ISO 14977 EBNF parser (feature `cfg_parsers`)
//! - [`abnf`] — RFC 5234 ABNF parser (feature `cfg_parsers`)

pub mod catalog;
pub mod constrained;
pub mod replicator;
pub mod weighted;

#[cfg(feature = "cfg_parsers")]
pub mod abnf;
#[cfg(feature = "cfg_parsers")]
pub mod ebnf;
