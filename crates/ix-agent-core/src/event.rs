//! Session event log shape + middleware verdict + outcome types.
//!
//! **"Transforms ARE Events."** When middleware rewrites an action, it
//! emits `ActionReplaced`. When it blocks, it emits `ActionBlocked`.
//! When it annotates, it emits `MetadataMounted`. The dispatcher
//! projects [`crate::context::ReadContext`] from the event log, so
//! downstream handlers see the effect of the rewrite without caring
//! how it happened.

use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::action::AgentAction;
use crate::error::ActionError;

// ---------------------------------------------------------------------------
// Block codes
// ---------------------------------------------------------------------------

/// Machine-readable category for [`crate::error::ActionError::Blocked`]
/// and [`SessionEvent::ActionBlocked`]. Consumers dispatch on the code,
/// not the reason string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockCode {
    /// `ix-loop-detect` fired.
    LoopDetected,
    /// A configured budget (time, tokens, tool calls) was exhausted.
    BudgetExceeded,
    /// An approval classifier requires explicit consent.
    ApprovalRequired,
    /// A policy rule hand-coded against [`BlockCode`] denied the
    /// action.
    PolicyDenied,
    /// The proposed action's blast radius (via ix-context) exceeds the
    /// allowed threshold.
    BlastRadiusTooLarge,
    /// A Demerzel constitutional article was about to be violated.
    GovernanceViolation,
}

// ---------------------------------------------------------------------------
// ActionOutcome
// ---------------------------------------------------------------------------

/// The success return shape from a handler invocation.
///
/// Handlers emit events via the `events` field even though they only
/// see a `&ReadContext` — the dispatcher drains these into the session
/// log after the handler returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// The raw value the handler produced. Untyped JSON because tool
    /// outputs are heterogeneous.
    pub value: JsonValue,
    /// Events the handler wants appended to the session log.
    pub events: Vec<SessionEvent>,
}

impl ActionOutcome {
    /// Build an outcome with no events, carrying just a value.
    pub fn value_only(value: JsonValue) -> Self {
        Self {
            value,
            events: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// MiddlewareVerdict
// ---------------------------------------------------------------------------

/// The return shape of a middleware invocation.
///
/// Middleware produces a verdict AND may emit events through its
/// [`crate::context::WriteContext::sink`]. The two are complementary:
/// the verdict tells the dispatcher what to do next (continue / block /
/// replace), and the events tell the session log what happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum MiddlewareVerdict {
    /// Let the dispatcher proceed to the next middleware or, if this
    /// is the last one, to the tool handler.
    Continue,

    /// Stop the chain and return an [`ActionError::Blocked`]. The
    /// dispatcher emits an [`SessionEvent::ActionBlocked`] event into
    /// the session log automatically — the middleware itself does NOT
    /// need to emit the event manually.
    Block {
        /// Machine-readable block category.
        code: BlockCode,
        /// Human-readable reason.
        reason: String,
    },

    /// Replace the pending action with a new one. The dispatcher emits
    /// an [`SessionEvent::ActionReplaced`] event and re-projects
    /// `ReadContext` for downstream middleware.
    Replace(AgentAction),
}

// ---------------------------------------------------------------------------
// Session event log entries
// ---------------------------------------------------------------------------

/// An entry in the session event log.
///
/// `ix-session` (primitive #4) owns the actual persistence and ordinal
/// assignment; this crate defines the shape that all producers and
/// consumers agree on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEvent {
    /// The agent proposed an action. First event of every turn.
    ActionProposed {
        /// Log ordinal.
        ordinal: u64,
        /// The proposed action.
        action: AgentAction,
    },

    /// Middleware blocked the action. Handler is NOT called.
    ActionBlocked {
        /// Log ordinal of the blocked action.
        ordinal: u64,
        /// Machine-readable block category.
        code: BlockCode,
        /// Human-readable reason.
        reason: String,
        /// Name of the middleware that emitted the block.
        emitted_by: String,
    },

    /// Middleware rewrote the action.
    ActionReplaced {
        /// Log ordinal.
        ordinal: u64,
        /// The action the agent originally proposed.
        original: AgentAction,
        /// The action the dispatcher will actually invoke.
        replacement: AgentAction,
        /// Middleware name.
        emitted_by: String,
    },

    /// Middleware or a handler mounted data at a metadata path.
    MetadataMounted {
        /// Log ordinal.
        ordinal: u64,
        /// VFS-style path, e.g. `"approval/blast_radius"`.
        path: String,
        /// Arbitrary JSON payload.
        value: JsonValue,
        /// Who mounted the data.
        emitted_by: String,
    },

    /// A handler completed successfully with a value.
    ActionCompleted {
        /// Log ordinal.
        ordinal: u64,
        /// Handler return value.
        value: JsonValue,
    },

    /// A handler failed with a typed error.
    ActionFailed {
        /// Log ordinal.
        ordinal: u64,
        /// The error shape.
        error: ActionError,
    },

    /// A hexavalent belief changed. Projected into
    /// [`crate::context::ReadContext::beliefs`] on the next turn.
    BeliefChanged {
        /// Log ordinal.
        ordinal: u64,
        /// The proposition whose truth value changed.
        proposition: String,
        /// Previous value, if any.
        old: Option<Hexavalent>,
        /// New value.
        new: Hexavalent,
        /// Evidence payload.
        evidence: JsonValue,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── BlockCode serialization ────────────────────────────────────────

    #[test]
    fn block_code_serializes_as_snake_case() {
        let cases = [
            (BlockCode::LoopDetected, r#""loop_detected""#),
            (BlockCode::BudgetExceeded, r#""budget_exceeded""#),
            (BlockCode::ApprovalRequired, r#""approval_required""#),
            (BlockCode::PolicyDenied, r#""policy_denied""#),
            (BlockCode::BlastRadiusTooLarge, r#""blast_radius_too_large""#),
            (BlockCode::GovernanceViolation, r#""governance_violation""#),
        ];
        for (code, expected) in cases {
            assert_eq!(serde_json::to_string(&code).unwrap(), expected);
        }
    }

    #[test]
    fn block_code_is_copy() {
        // Compile-time assertion: takes by value, works after .clone()
        fn takes_copy<T: Copy>(_: T) {}
        takes_copy(BlockCode::LoopDetected);
    }

    // ── ActionOutcome ──────────────────────────────────────────────────

    #[test]
    fn action_outcome_value_only_has_no_events() {
        let o = ActionOutcome::value_only(json!(42));
        assert_eq!(o.value, json!(42));
        assert!(o.events.is_empty());
    }

    #[test]
    fn action_outcome_serde_round_trip() {
        let o = ActionOutcome {
            value: json!({"mean": 2.5}),
            events: vec![SessionEvent::ActionCompleted {
                ordinal: 1,
                value: json!({"mean": 2.5}),
            }],
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: ActionOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }

    // ── MiddlewareVerdict ──────────────────────────────────────────────

    #[test]
    fn verdict_continue_serializes_with_tag() {
        let v = MiddlewareVerdict::Continue;
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains(r#""outcome":"continue""#));
    }

    #[test]
    fn verdict_block_serde_round_trip() {
        let v = MiddlewareVerdict::Block {
            code: BlockCode::LoopDetected,
            reason: "too many calls".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: MiddlewareVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn verdict_replace_carries_new_action() {
        let new = AgentAction::Return {
            payload: json!("done"),
            ordinal: 0,
        };
        let v = MiddlewareVerdict::Replace(new.clone());
        let json = serde_json::to_string(&v).unwrap();
        let back: MiddlewareVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }

    // ── SessionEvent round-trips ───────────────────────────────────────

    #[test]
    fn session_event_action_proposed_round_trip() {
        let e = SessionEvent::ActionProposed {
            ordinal: 0,
            action: AgentAction::Return {
                payload: json!("x"),
                ordinal: 0,
            },
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""kind":"action_proposed""#));
        let back: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn session_event_action_blocked_round_trip() {
        let e = SessionEvent::ActionBlocked {
            ordinal: 3,
            code: BlockCode::LoopDetected,
            reason: "11 calls".into(),
            emitted_by: "ix_loop_detect".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn session_event_metadata_mounted_round_trip() {
        let e = SessionEvent::MetadataMounted {
            ordinal: 5,
            path: "approval/blast_radius".into(),
            value: json!({"nodes": 42, "edges": 115}),
            emitted_by: "ix_approval".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn session_event_belief_changed_round_trip() {
        let e = SessionEvent::BeliefChanged {
            ordinal: 7,
            proposition: "api_stable".into(),
            old: Some(Hexavalent::Unknown),
            new: Hexavalent::Probable,
            evidence: json!({"tests_passing": 47}),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }
}
