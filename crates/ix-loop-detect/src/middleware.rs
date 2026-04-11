//! [`LoopDetectMiddleware`] — the [`ix_agent_core::Middleware`] adapter
//! over [`LoopDetector`].
//!
//! Drops into a [`ix_agent_core::MiddlewareChain`] as the first hop,
//! so every action routes through the circuit breaker before approval
//! classification or any downstream handler runs.
//!
//! ## Behavior
//!
//! - Non-`InvokeTool` actions (observations, returns, approvals) are
//!   always passed through — they cannot loop.
//! - For `InvokeTool`, the middleware calls
//!   [`LoopDetector::record`]. If the verdict is
//!   [`LoopVerdict::TooManyEdits`], it returns
//!   [`MiddlewareVerdict::Block`] with
//!   [`ix_agent_core::event::BlockCode::LoopDetected`]. The
//!   dispatcher — not this middleware — emits the resulting
//!   [`ix_agent_core::SessionEvent::ActionBlocked`] into the sink,
//!   so there is no double-logging.
//! - A success verdict ([`LoopVerdict::Ok`]) returns
//!   [`MiddlewareVerdict::Continue`] without emitting anything —
//!   the session log would otherwise grow unbounded on normal traffic.

use std::sync::Arc;

use ix_agent_core::event::{BlockCode, MiddlewareVerdict};
use ix_agent_core::{AgentAction, Middleware, WriteContext};

use crate::{LoopDetector, LoopDetectorConfig, LoopVerdict};

/// Middleware wrapper around a shared [`LoopDetector`].
///
/// The detector is held as an `Arc<LoopDetector>` so callers can
/// share one process-wide detector between the middleware chain and
/// any legacy dispatch path that still records inline. This keeps
/// both paths counting against the same sliding window.
pub struct LoopDetectMiddleware {
    detector: Arc<LoopDetector>,
}

impl LoopDetectMiddleware {
    /// Construct a middleware with a freshly-built [`LoopDetector`]
    /// using the provided configuration. The detector is wrapped in
    /// a new `Arc`; use [`LoopDetectMiddleware::from_shared`] if you
    /// need to share an existing detector.
    pub fn new(config: LoopDetectorConfig) -> Self {
        Self {
            detector: Arc::new(LoopDetector::new(config)),
        }
    }

    /// Construct a middleware with the detector's default
    /// configuration (10 calls / 5 minutes per key).
    pub fn with_defaults() -> Self {
        Self {
            detector: Arc::new(LoopDetector::with_defaults()),
        }
    }

    /// Construct a middleware from an already-existing
    /// `Arc<LoopDetector>`. Use this when the same detector must be
    /// shared between the chain and a legacy dispatch path so both
    /// contribute to one sliding window.
    pub fn from_shared(detector: Arc<LoopDetector>) -> Self {
        Self { detector }
    }

    /// Clone the detector handle — exposed so callers can observe
    /// or clear the detector without going through the middleware.
    pub fn detector(&self) -> Arc<LoopDetector> {
        Arc::clone(&self.detector)
    }
}

impl Middleware for LoopDetectMiddleware {
    fn name(&self) -> &str {
        "ix_loop_detect"
    }

    fn pre(&self, _cx: &mut WriteContext<'_>, action: &AgentAction) -> MiddlewareVerdict {
        // Non-invoke actions have no loop key and never contribute.
        if action.loop_key().is_none() {
            return MiddlewareVerdict::Continue;
        }

        match self.detector.record(action) {
            LoopVerdict::Ok => MiddlewareVerdict::Continue,
            LoopVerdict::TooManyEdits {
                count,
                window,
                threshold,
            } => MiddlewareVerdict::Block {
                code: BlockCode::LoopDetected,
                reason: format!(
                    "circuit breaker tripped: {count} calls in the last {window:?} \
                     exceeds threshold {threshold}"
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_agent_core::{ReadContext, VecEventSink};

    fn invoke(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        }
    }

    fn ctx() -> ReadContext {
        ReadContext::synthetic_for_legacy()
    }

    #[test]
    fn under_threshold_returns_continue() {
        let mw = LoopDetectMiddleware::new(LoopDetectorConfig {
            threshold: 3,
            window: std::time::Duration::from_secs(60),
        });
        let cx = ctx();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let action = invoke("ix_stats");
        for _ in 0..3 {
            assert!(matches!(
                mw.pre(&mut wc, &action),
                MiddlewareVerdict::Continue
            ));
        }
        // Middleware never emits on its own — the chain handles
        // ActionBlocked emission. Sink stays empty here.
        assert!(sink.events.is_empty());
    }

    #[test]
    fn fourth_call_blocks_with_loop_detected() {
        let mw = LoopDetectMiddleware::new(LoopDetectorConfig {
            threshold: 3,
            window: std::time::Duration::from_secs(60),
        });
        let cx = ctx();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let action = invoke("ix_stats");
        for _ in 0..3 {
            mw.pre(&mut wc, &action);
        }
        match mw.pre(&mut wc, &action) {
            MiddlewareVerdict::Block { code, .. } => {
                assert_eq!(code, BlockCode::LoopDetected);
            }
            other => panic!("expected Block(LoopDetected), got {other:?}"),
        }
        // Middleware itself emits no events; chain-level emission is
        // covered by the integration test in ix-agent.
        assert!(sink.events.is_empty());
    }

    #[test]
    fn non_invoke_actions_pass_through() {
        let mw = LoopDetectMiddleware::with_defaults();
        let cx = ctx();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let action = AgentAction::Return {
            ordinal: 0,
            payload: serde_json::json!(null),
        };
        for _ in 0..1000 {
            assert!(matches!(
                mw.pre(&mut wc, &action),
                MiddlewareVerdict::Continue
            ));
        }
        assert!(sink.events.is_empty());
    }

    #[test]
    fn different_tool_names_isolate_counts() {
        let mw = LoopDetectMiddleware::new(LoopDetectorConfig {
            threshold: 2,
            window: std::time::Duration::from_secs(60),
        });
        let cx = ctx();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        // Two calls each of two tools: neither should trip.
        for _ in 0..2 {
            assert!(matches!(
                mw.pre(&mut wc, &invoke("ix_stats")),
                MiddlewareVerdict::Continue
            ));
            assert!(matches!(
                mw.pre(&mut wc, &invoke("ix_fft")),
                MiddlewareVerdict::Continue
            ));
        }

        // A third call on just one of them now trips.
        assert!(matches!(
            mw.pre(&mut wc, &invoke("ix_stats")),
            MiddlewareVerdict::Block { .. }
        ));
    }
}
