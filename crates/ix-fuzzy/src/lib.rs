//! # ix-fuzzy — generic fuzzy distributions over discrete variants
//!
//! Primitive #5 of the harness primitives roadmap
//! (`docs/brainstorms/2026-04-10-ix-harness-primitives.md`). Closes a
//! documented gap in the Demerzel governance spec
//! (`governance/demerzel/logic/fuzzy-membership.md`,
//! `governance/demerzel/docs/superpowers/specs/2026-03-22-fuzzy-enum-du-design.md`):
//! the schemas define `FuzzyEnum<'T>` and its operations over any enum,
//! but no Rust consumer had implemented them. This crate fills that
//! gap.
//!
//! ## Governance-instrument role
//!
//! `FuzzyDistribution<T>` is the continuous companion to
//! `ix_types::Hexavalent`. Where `Hexavalent` names a single discrete
//! truth state, `FuzzyDistribution` carries the membership mass across
//! every variant, so consumers can:
//!
//! - Project a belief forward under AND / OR / NOT without losing
//!   uncertainty
//! - Bridge evidence accumulation (Bayesian / Zadeh / Multiplicative
//!   combiners) into a single replayable structure
//! - Sharpen to discrete `Hexavalent` when the argmax crosses a
//!   confidence threshold
//!
//! Every public type is `Serialize` + `Deserialize` and iterates its
//! internal `BTreeMap` in deterministic key order, so bit-exact
//! replay across processes is preserved — the same contract the rest
//! of the harness substrate honors.
//!
//! ## Scope (MVP)
//!
//! - Generic `FuzzyDistribution<T>` over any `Ord + Clone` variant
//! - Operations: `and` / `or` / `not_generic` / `renormalize` /
//!   `sharpen` / `argmax` / `is_sharp`
//! - Combiners: `Multiplicative`, `Zadeh` (min), `Bayesian`
//! - Hexavalent specialization: tiebreak order
//!   `C > U > D > P > T > F`, escalation check (`C > 0.3`),
//!   hexavalent-specific NOT (`T↔F`, `P↔D`)
//!
//! ## Non-goals (v2)
//!
//! - Classical FIS / F-AHP / F-TOPSIS inference engines (separate
//!   crate if they ever become needed)
//! - Payload-carrying FuzzyDU
//! - Belief revision / Dempster-Shafer
//! - A `fuzzy { }` computation-expression analog — Rust's trait
//!   system already covers method chaining cleanly

pub mod builder;
pub mod distribution;
pub mod error;
pub mod hexavalent;
pub mod observations;
pub mod ops;

pub use builder::{Combiner, FuzzyBuilder};
pub use distribution::FuzzyDistribution;
pub use error::FuzzyError;
pub use hexavalent::{
    escalation_triggered, hexavalent_not, HexavalentDistribution, ESCALATION_THRESHOLD,
    SHARPEN_THRESHOLD,
};
