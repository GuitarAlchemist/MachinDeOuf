//! Experimental compact memory formats for session history over
//! mathematical spaces.
//!
//! This crate is a **research sandbox**, not a production component.
//! It exists to validate that several mathematical encodings from
//! other ix crates can serve as compact memory formats for the
//! session history substrate proposed in the 2026-04-11 Path C
//! brainstorm.
//!
//! # Three modules, three math spaces
//!
//! - [`hrr`] — **Holographic Reduced Representations.** Circular
//!   convolution binding + correlation retrieval in a fixed-size
//!   vector. A whole session fits in a single D-dimensional vector
//!   regardless of event count, with graceful degradation as capacity
//!   saturates. Based on Tony Plate's 1991 work.
//!
//! - [`dna`] — **DNA codon encoding.** 2-bit bases, 6-bit codons, a
//!   variant-to-codon table for `SessionEvent` types with built-in
//!   redundancy borrowed from biology. Compact for discrete tags.
//!
//! - [`sedenion_sig`] — **Sedenion session product signature.** Each
//!   event maps to a sedenion basis element; the session signature
//!   is their non-associative left-fold product. Reorder-detecting
//!   by algebra. Uses `ix-sedenion`.
//!
//! # One wrapper format
//!
//! [`mem_file`] defines a minimal binary container (`.mem`) that can
//! carry any of the three payload types:
//!
//! ```text
//! [ Magic: 4 bytes "IXMM" ]
//! [ Version: 1 byte = 0x01 ]
//! [ Kind:    1 byte ('H' | 'D' | 'S') ]
//! [ Payload length: 4 bytes LE u32 ]
//! [ Payload: length bytes ]
//! [ SHA-256: 32 bytes — over magic..payload ]
//! ```
//!
//! Each `Kind` has its own encoding within the payload — HRR writes
//! a float vector, DNA writes packed codon bytes, sedenion writes
//! 16 f64 components.
//!
//! # What this is NOT
//!
//! - Not a drop-in replacement for `ix-session`'s JSONL log. That
//!   substrate stays authoritative; this crate produces sidecar
//!   artifacts for experiments.
//! - Not a complete SessionEvent encoder. The modules here encode
//!   *features derived from* events (variant tags, cumulative
//!   products, associative bindings) — not the full event payload.
//! - Not benchmarked yet. Compression ratios and retrieval fidelity
//!   are goals, not measurements.

pub mod dna;
pub mod hrr;
pub mod mem_file;
pub mod sedenion_sig;

/// Version of the `.mem` container format understood by this crate.
pub const MEM_FORMAT_VERSION: u8 = 0x01;
