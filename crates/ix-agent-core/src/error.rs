//! Error and result types for the harness primitives.
//!
//! [`ActionError`] is the return-shape tools and middleware produce
//! when something goes wrong; [`ActionResult`] is the standard
//! `Result<ActionOutcome, ActionError>` typedef that every handler
//! returns.

use crate::event::ActionOutcome;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::event::BlockCode;

/// The error shape handlers return from [`crate::handler::AgentHandler::run`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionError {
    /// Middleware halted the action before the handler ran.
    #[error("blocked by {blocker}: {reason} (code: {code:?})")]
    Blocked {
        /// Machine-readable block category.
        code: BlockCode,
        /// Human-readable reason the block fired.
        reason: String,
        /// Name of the middleware that emitted the block.
        blocker: String,
    },
    /// The handler executed but failed with a runtime error.
    #[error("execution failed: {0}")]
    Exec(String),
    /// The handler returned a value that did not match its declared
    /// output shape.
    #[error("handler returned invalid value: {0}")]
    InvalidResult(String),
}

/// Convenience alias for handler return types.
pub type ActionResult = Result<ActionOutcome, ActionError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_display_formats_code_and_reason() {
        let e = ActionError::Blocked {
            code: BlockCode::LoopDetected,
            reason: "10 calls in 5 min".into(),
            blocker: "ix_loop_detect".into(),
        };
        let formatted = format!("{e}");
        assert!(formatted.contains("ix_loop_detect"));
        assert!(formatted.contains("10 calls"));
        assert!(formatted.contains("LoopDetected"));
    }

    #[test]
    fn exec_display_includes_payload() {
        let e = ActionError::Exec("division by zero".into());
        assert_eq!(format!("{e}"), "execution failed: division by zero");
    }

    #[test]
    fn action_error_serde_round_trip() {
        let e = ActionError::Blocked {
            code: BlockCode::BudgetExceeded,
            reason: "quota".into(),
            blocker: "ix_budget".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: ActionError = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }
}
