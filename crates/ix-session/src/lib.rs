//! # ix-session — JSONL-backed append-only session log
//!
//! Primitive #4 of the harness primitives roadmap (see
//! `docs/brainstorms/2026-04-10-ix-harness-primitives.md`). Provides a
//! persistent [`ix_agent_core::EventSink`] backed by a newline-
//! delimited JSON (JSONL) file, so [`ix_agent_core::SessionEvent`]s
//! emitted by middleware and handlers survive process restarts and
//! can be replayed later.
//!
//! ## Governance-instrument role
//!
//! The design doc
//! `docs/brainstorms/2026-04-10-agent-context-action.md` treats the
//! session event log as the **source of truth** for agent state:
//!
//! > Every `ReadContext` is a pure projection of the session event
//! > log at a specific ordinal. Replay is `f(EventLog, ordinal) ->
//! > ReadContext`. Same inputs → bit-identical outputs, across
//! > processes, because every map in the context uses `BTreeMap`
//! > rather than `HashMap`.
//!
//! This crate is the missing half of that contract. Without a
//! persistent log, "replay across processes" is impossible — the
//! `VecEventSink` that ships in `ix-agent-core` is in-memory only,
//! intended for unit tests. `SessionLog` replaces it for production
//! use.
//!
//! ## Scope
//!
//! - Pure library — no async, no global state, no file watchers
//! - Append-only writes via `std::fs::File` + `BufWriter`
//! - In-memory ordinal counter initialized from the file on open
//! - Reload on open: existing files are scanned to count events and
//!   populate the ordinal
//! - Read-back API: [`SessionLog::events`] returns an iterator over
//!   the on-disk entries
//! - Thread-safe via `Mutex` — concurrent writers share one writer
//!   handle
//!
//! ## Non-goals (MVP)
//!
//! - **No compaction.** Logs grow unbounded. v2 will add size-based
//!   rotation with projections to a separate snapshot file.
//! - **No integrity checking.** Corrupted lines during reload emit a
//!   warning (via [`ReloadError`]) but don't abort the load —
//!   consumers decide how to react.
//! - **No async I/O.** The whole API is synchronous. For
//!   `tokio`-based callers, wrap the sink in `spawn_blocking`.
//! - **No wiring into `ix-agent`'s `dispatch_action` yet.** The new
//!   primitive ships standalone so existing tests don't start
//!   writing to disk. A follow-up commit will make the wiring
//!   opt-in via configuration.
//!
//! ## Example
//!
//! ```no_run
//! use ix_session::SessionLog;
//! use ix_agent_core::{SessionEvent, EventSink};
//!
//! let log = SessionLog::open("session.jsonl").expect("open log");
//! let mut sink = log.sink();
//! sink.emit(SessionEvent::ActionCompleted {
//!     ordinal: sink.next_ordinal(),
//!     value: serde_json::json!({"result": 42}),
//! });
//! // Later, reopen and replay:
//! let replay = SessionLog::open("session.jsonl").expect("reopen");
//! for result in replay.events().expect("iter") {
//!     let event = result.expect("valid line");
//!     println!("{event:?}");
//! }
//! ```

pub mod errors;
pub mod log;
pub mod sink;

pub use errors::{ReloadError, SessionError};
pub use log::SessionLog;
pub use sink::SessionSink;
