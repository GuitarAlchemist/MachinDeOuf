//! Integration tests for the circuit breaker — both the legacy
//! inline `dispatch()` path and the new `dispatch_action()` path
//! via `LoopDetectMiddleware`.
//!
//! Lives in its own test binary so the shared process-wide
//! `LoopDetector` inside `registry_bridge` doesn't race with the
//! parallel parity tests in `parity.rs` (every `tests/*.rs` file
//! compiles to a separate binary, giving each group a private
//! detector instance).

use ix_agent::registry_bridge;
use ix_agent_core::event::BlockCode;
use ix_agent_core::{ActionError, AgentAction, ReadContext};

/// Chain-level loop detection via the new `LoopDetectMiddleware`
/// registered as the first middleware in
/// `registry_bridge::middleware_chain`. Shares the process-wide
/// detector with the legacy inline path, so both entry points
/// count against the same sliding window.
#[test]
fn dispatch_action_trips_loop_detect_middleware() {
    registry_bridge::shared_loop_detector().clear_key("ix_stats");

    let cx = ReadContext::synthetic_for_legacy();
    let make_action = || AgentAction::InvokeTool {
        tool_name: "ix_stats".to_string(),
        params: serde_json::json!({ "data": [1.0, 2.0, 3.0] }),
        ordinal: 0,
        target_hint: None,
    };

    for i in 0..10 {
        registry_bridge::dispatch_action(&cx, make_action())
            .unwrap_or_else(|e| panic!("call {i} should succeed, got {e:?}"));
    }

    match registry_bridge::dispatch_action(&cx, make_action()) {
        Err(ActionError::Blocked { code, blocker, .. }) => {
            assert_eq!(code, BlockCode::LoopDetected);
            assert_eq!(blocker, "ix_loop_detect");
        }
        other => panic!("expected Blocked(LoopDetected), got {other:?}"),
    }

    registry_bridge::shared_loop_detector().clear_key("ix_stats");
}

/// Legacy inline path: reach into the shared detector, clear it,
/// then spam `ix_stats` 11 times through the raw `dispatch()`
/// function. The 11th call must return a circuit-breaker error.
#[test]
fn loop_detector_trips_on_repeated_dispatch() {
    // Use a distinct tool name so this test does not race with the
    // chain-level test above through the shared detector.
    let detector = registry_bridge::shared_loop_detector();
    detector.clear_key("ix_fft");

    let params = serde_json::json!({ "signal": [1.0, 0.0, -1.0, 0.0] });
    for _ in 0..10 {
        let result = registry_bridge::dispatch("ix_fft", params.clone());
        assert!(result.is_ok(), "first 10 calls should succeed: {result:?}");
    }
    let tripped = registry_bridge::dispatch("ix_fft", params.clone());
    match tripped {
        Err(msg) => {
            assert!(
                msg.contains("circuit breaker tripped"),
                "expected circuit-breaker error, got: {msg}"
            );
            assert!(msg.contains("ix_fft"));
            assert!(msg.contains("threshold"));
        }
        Ok(value) => panic!("11th call should have been blocked, got: {value}"),
    }

    detector.clear_key("ix_fft");
}

/// Prove the two dispatch paths share one detector: alternating
/// calls through `dispatch_action` and legacy `dispatch` on an
/// unregistered tool name still contribute to one sliding window,
/// so the 11th call trips `LoopDetected` even though no single
/// path reached 11 alone.
///
/// Unknown-tool dispatches fail downstream (unknown registry lookup
/// for `dispatch`, approval block for `dispatch_action`), but both
/// paths record into the shared detector *before* those downstream
/// checks run — which is exactly the property we need to verify.
#[test]
fn legacy_and_action_paths_share_detector() {
    const PROBE: &str = "ix_sharing_probe_tool_xyz";
    let detector = registry_bridge::shared_loop_detector();
    detector.clear_key(PROBE);

    let cx = ReadContext::synthetic_for_legacy();
    let make_action = || AgentAction::InvokeTool {
        tool_name: PROBE.to_string(),
        params: serde_json::json!({}),
        ordinal: 0,
        target_hint: None,
    };

    // 5 dispatch_action calls: each fails downstream at approval
    // (Tier 3 unknown tool) but records into the detector first.
    for _ in 0..5 {
        let _ = registry_bridge::dispatch_action(&cx, make_action());
    }
    // 5 legacy dispatch calls: each fails at registry lookup but
    // records into the detector first.
    for _ in 0..5 {
        let _ = registry_bridge::dispatch(PROBE, serde_json::json!({}));
    }

    // Detector should have 10 recorded calls for PROBE.
    assert_eq!(
        detector.count(PROBE),
        10,
        "shared detector should have counted 5 action + 5 legacy calls"
    );

    // 11th call via dispatch_action — LoopDetectMiddleware runs
    // before ApprovalMiddleware, so we get LoopDetected not
    // ApprovalRequired.
    match registry_bridge::dispatch_action(&cx, make_action()) {
        Err(ActionError::Blocked { code, .. }) => {
            assert_eq!(code, BlockCode::LoopDetected);
        }
        other => panic!("expected Blocked(LoopDetected), got {other:?}"),
    }

    detector.clear_key(PROBE);
}
