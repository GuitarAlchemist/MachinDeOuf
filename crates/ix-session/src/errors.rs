//! Error types for session log I/O and reload.

use std::path::PathBuf;
use thiserror::Error;

/// Top-level errors that can occur opening, writing to, or reading from
/// a session log.
#[derive(Debug, Error)]
pub enum SessionError {
    /// The log directory could not be created or accessed.
    #[error("failed to create or access directory {path}: {source}")]
    Directory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The log file could not be opened.
    #[error("failed to open log file {path}: {source}")]
    OpenFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A write to the log file failed.
    #[error("failed to write to log file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A read from the log file failed.
    #[error("failed to read from log file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// An event could not be serialized to JSON.
    #[error("failed to serialize event: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Per-line errors that can occur while reloading an existing session
/// log. [`crate::SessionLog::open`] returns a vector of these (rather
/// than failing outright) so consumers can decide whether to abort,
/// repair, or skip bad lines.
#[derive(Debug, Error)]
pub enum ReloadError {
    /// A line in the log file was not valid UTF-8.
    #[error("line {line} is not valid UTF-8")]
    InvalidUtf8 { line: usize },

    /// A line failed to deserialize as a [`ix_agent_core::SessionEvent`].
    #[error("line {line} failed to deserialize: {source}")]
    BadJson {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}
