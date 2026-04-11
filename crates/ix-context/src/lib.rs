//! # ix-context — deterministic structural context DAG over Rust code
//!
//! ## Governance reframe
//!
//! This crate is **not primarily a better retrieval system for Claude.**
//! Claude is the beneficiary; Demerzel is the customer.
//!
//! Vector RAG is opaque: a wrong answer is unattributable — bad similarity?
//! bad chunking? bad luck? A walked DAG makes the agent's uncertainty
//! *legible* — wrongness localizes to either the walk policy or the
//! reasoning. This is the same move as static typing: you don't eliminate
//! bugs, you **relocate** them to a place where they can be named.
//!
//! Every walk produces a [`model::ContextBundle`] with a replayable
//! `walk_trace`. Given the same [`index::ProjectIndex`] at the same git SHA,
//! feeding the trace back through the walker must reconstruct the exact set
//! of visited nodes and edges. That's the governance instrument: a skeptical
//! auditor can replay any agent action and verify the informational state it
//! acted on.
//!
//! ## Architecture
//!
//! - [`model`] — node, edge, and bundle types with stable IDs and hexavalent
//!   belief labels
//! - [`index`] — project-wide symbol table built by a two-pass tree-sitter
//!   walk over the workspace
//! - [`resolve`] — call-site resolver that consumes
//!   [`ix_code::semantic::CalleeHint`] and produces
//!   [`model::ResolvedOrAmbiguous`] edges, preserving ambiguity as signal
//!   rather than hiding it
//! - [`walk`] — [`walk::Walker`] with four MVP strategies:
//!   callers-transitive, callees-transitive, module-siblings, git-co-change
//! - [`cache`] — SHA + content-hash keyed cache with `notify` +
//!   `ix-cache` pub/sub invalidation
//! - [`mcp`] — thin wrapper registering `ix_context_walk` as an MCP tool
//!
//! ## Scope boundaries
//!
//! - **Per-file parsing lives in [`ix_code::semantic`]**. This crate adds
//!   cross-file resolution on top; it does not re-implement AST walking.
//! - **No cycle-checked pipeline substrate**. [`ix_pipeline::dag::Dag`] is
//!   already the generic DAG; `ix-context` is a *walker* over call/import
//!   graphs, not a DAG executor.
//! - **No belief-weighted walks in MVP**. Frontier ordering by unresolved
//!   count is deferred to v2.
//! - **No persistent-homology stopping rule in MVP**. `ix_topo` integration
//!   is deferred to v2.
//! - **No embedding fallback, ever**. This is a determinism purity project.

pub mod cache;
pub mod index;
pub mod mcp;
pub mod model;
pub mod resolve;
pub mod walk;
