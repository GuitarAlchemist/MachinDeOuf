//! Agent actions — the structured representation of what an agent wants
//! to do. Tagged enum with four variants covering every action shape the
//! harness primitives currently care about.
//!
//! Actions are **immutable after creation**. Middleware cannot mutate an
//! `AgentAction` — instead, it emits a `Replace` event that the
//! dispatcher consults before invoking the handler, producing a new
//! action without breaking the original's identity. This is the
//! *"Transforms ARE Events"* thesis from the design doc.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// A structured representation of an agent's intended action.
///
/// Every variant carries an `ordinal: u64` — the position of the action
/// in the session event log. The dispatcher assigns this on action
/// construction; middleware must preserve it when emitting `Replace`
/// events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentAction {
    /// Invoke a registered MCP tool.
    InvokeTool {
        /// The MCP tool name (e.g., `"ix_stats"`, `"ix_context_walk"`).
        tool_name: String,
        /// Raw JSON parameters. Untyped here because tools have
        /// heterogeneous schemas; per-tool validation happens inside
        /// the handler.
        params: JsonValue,
        /// Correlation ID — the ordinal of this action in the session
        /// log.
        ordinal: u64,
        /// Optional target argument hint for fine-grained loop
        /// detection. e.g., for `ix_context_walk`, this is the target
        /// function path. `None` means loop detection keys on
        /// `tool_name` alone.
        target_hint: Option<String>,
    },

    /// Emit a non-tool observation — e.g., a summary, a progress
    /// update, a belief assertion. These do not invoke the dispatcher
    /// but are logged for replay.
    EmitObservation {
        /// Logical channel the observation belongs to.
        stream: String,
        /// Observation payload.
        payload: JsonValue,
        /// Correlation ID.
        ordinal: u64,
    },

    /// Request approval for a high-risk action. The approval middleware
    /// projects the subject, computes a verdict, and either blocks
    /// (via a `Block` event) or lets the dispatcher proceed.
    RequestApproval {
        /// Which classifier should evaluate the subject. Reserved for
        /// future approval-crate routing.
        classifier: String,
        /// Opaque subject the classifier will inspect.
        subject: JsonValue,
        /// Correlation ID.
        ordinal: u64,
    },

    /// Return a final result to the caller. Terminates the current turn
    /// for this agent.
    Return {
        /// Final payload.
        payload: JsonValue,
        /// Correlation ID.
        ordinal: u64,
    },
}

impl AgentAction {
    /// A stable key for loop detection.
    ///
    /// For [`AgentAction::InvokeTool`], combines `tool_name` with
    /// `target_hint` if present:
    /// `"ix_context_walk:mini::eigen::jacobi"` vs bare
    /// `"ix_context_walk"`.
    ///
    /// For non-tool actions, returns `None` — loop detection only
    /// applies to repeated tool invocations.
    ///
    /// # Example
    ///
    /// ```
    /// use ix_agent_core::AgentAction;
    /// use serde_json::json;
    ///
    /// let bare = AgentAction::InvokeTool {
    ///     tool_name: "ix_stats".into(),
    ///     params: json!({}),
    ///     ordinal: 0,
    ///     target_hint: None,
    /// };
    /// assert_eq!(bare.loop_key().as_deref(), Some("ix_stats"));
    ///
    /// let targeted = AgentAction::InvokeTool {
    ///     tool_name: "ix_context_walk".into(),
    ///     params: json!({}),
    ///     ordinal: 1,
    ///     target_hint: Some("ix_math::eigen::jacobi".into()),
    /// };
    /// assert_eq!(
    ///     targeted.loop_key().as_deref(),
    ///     Some("ix_context_walk:ix_math::eigen::jacobi"),
    /// );
    /// ```
    pub fn loop_key(&self) -> Option<String> {
        match self {
            AgentAction::InvokeTool {
                tool_name,
                target_hint,
                ..
            } => Some(match target_hint {
                Some(t) => format!("{tool_name}:{t}"),
                None => tool_name.clone(),
            }),
            _ => None,
        }
    }

    /// Monotonic ordinal within the session. Never `None` — every
    /// action has a position in the log.
    pub fn ordinal(&self) -> u64 {
        match self {
            AgentAction::InvokeTool { ordinal, .. }
            | AgentAction::EmitObservation { ordinal, .. }
            | AgentAction::RequestApproval { ordinal, .. }
            | AgentAction::Return { ordinal, .. } => *ordinal,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn invoke(tool: &str, target: Option<&str>) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: json!({"data": [1, 2, 3]}),
            ordinal: 0,
            target_hint: target.map(String::from),
        }
    }

    // ── loop_key semantics ─────────────────────────────────────────────

    #[test]
    fn loop_key_bare_invoke() {
        let a = invoke("ix_stats", None);
        assert_eq!(a.loop_key().as_deref(), Some("ix_stats"));
    }

    #[test]
    fn loop_key_targeted_invoke() {
        let a = invoke("ix_context_walk", Some("ix_math::eigen::jacobi"));
        assert_eq!(
            a.loop_key().as_deref(),
            Some("ix_context_walk:ix_math::eigen::jacobi")
        );
    }

    #[test]
    fn loop_key_for_observation_is_none() {
        let a = AgentAction::EmitObservation {
            stream: "log".into(),
            payload: json!({}),
            ordinal: 0,
        };
        assert_eq!(a.loop_key(), None);
    }

    #[test]
    fn loop_key_for_return_is_none() {
        let a = AgentAction::Return {
            payload: json!({}),
            ordinal: 0,
        };
        assert_eq!(a.loop_key(), None);
    }

    #[test]
    fn loop_key_for_approval_is_none() {
        let a = AgentAction::RequestApproval {
            classifier: "c".into(),
            subject: json!({}),
            ordinal: 0,
        };
        assert_eq!(a.loop_key(), None);
    }

    // ── ordinal() dispatch ─────────────────────────────────────────────

    #[test]
    fn ordinal_invoke_tool() {
        let a = AgentAction::InvokeTool {
            tool_name: "t".into(),
            params: json!({}),
            ordinal: 42,
            target_hint: None,
        };
        assert_eq!(a.ordinal(), 42);
    }

    #[test]
    fn ordinal_all_variants() {
        let mut actions = vec![
            AgentAction::InvokeTool {
                tool_name: "t".into(),
                params: json!({}),
                ordinal: 1,
                target_hint: None,
            },
            AgentAction::EmitObservation {
                stream: "s".into(),
                payload: json!({}),
                ordinal: 2,
            },
            AgentAction::RequestApproval {
                classifier: "c".into(),
                subject: json!({}),
                ordinal: 3,
            },
            AgentAction::Return {
                payload: json!({}),
                ordinal: 4,
            },
        ];
        for (i, a) in actions.drain(..).enumerate() {
            assert_eq!(a.ordinal(), (i as u64) + 1);
        }
    }

    // ── Serde round-trip ───────────────────────────────────────────────

    #[test]
    fn invoke_tool_serde_round_trip() {
        let original = invoke("ix_context_walk", Some("ix_math::eigen::jacobi"));
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""kind":"invoke_tool""#));
        let back: AgentAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn emit_observation_serde_round_trip() {
        let a = AgentAction::EmitObservation {
            stream: "progress".into(),
            payload: json!({"step": 7, "total": 10}),
            ordinal: 42,
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains(r#""kind":"emit_observation""#));
        let back: AgentAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn request_approval_serde_round_trip() {
        let a = AgentAction::RequestApproval {
            classifier: "ix_approval".into(),
            subject: json!({"action": "delete", "target": "src/main.rs"}),
            ordinal: 3,
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: AgentAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn return_serde_round_trip() {
        let a = AgentAction::Return {
            payload: json!("done"),
            ordinal: 99,
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: AgentAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }
}
