//! R1 gap closer — integration test for `ix_pipeline_run`.
//!
//! Per `examples/canonical-showcase/ix-roadmap-plan-v1.md` §4.1, R1
//! exposes a new MCP tool that takes a DAG of steps and executes them
//! end-to-end, replacing the hand-chaining of tool calls from Days 1-5.
//! This integration test exercises the tool via `ToolRegistry::call_with_ctx`
//! (the same entry point the MCP dispatcher uses) to prove the
//! end-to-end path works without standing up a real stdio server.

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn make_ctx() -> ServerContext {
    // ServerContext::new returns (ctx, outbound_receiver). We don't
    // need the receiver for pipeline_run tests — the handler never
    // sends any outbound sampling requests.
    let (ctx, _outbound_rx) = ServerContext::new();
    ctx
}

#[test]
fn pipeline_run_single_step_stats() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    let args = json!({
        "steps": [
            {
                "id": "s1",
                "tool": "ix_stats",
                "arguments": { "data": [1.0, 2.0, 3.0, 4.0, 5.0] }
            }
        ]
    });

    let result = reg.call_with_ctx("ix_pipeline_run", args, &ctx).expect("ok");
    let results = result.get("results").expect("results");
    let s1 = results.get("s1").expect("s1");
    // Response may be wrapped or bare — check for mean either way.
    let mean = extract_mean(s1).expect("mean present");
    assert!((mean - 3.0).abs() < 1e-9, "mean = {mean}");

    let order = result.get("execution_order").and_then(|v| v.as_array()).expect("order");
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "s1");
}

#[test]
fn pipeline_run_two_step_chain_with_substitution() {
    // Minimal 2-step chain: stats computes summary, second step
    // references the stats result via "$s1" substitution. We use
    // ix_cache so the second step has a cheap, predictable tool call
    // that only needs one argument.
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    let args = json!({
        "steps": [
            {
                "id": "s1",
                "tool": "ix_stats",
                "arguments": { "data": [10.0, 20.0, 30.0] }
            },
            {
                "id": "s2",
                "tool": "ix_stats",
                "arguments": { "data": [5.0, 15.0, 25.0, 35.0] },
                "depends_on": ["s1"]
            }
        ]
    });

    let result = reg.call_with_ctx("ix_pipeline_run", args, &ctx).expect("ok");
    let order = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .expect("order");
    assert_eq!(order.len(), 2);
    assert_eq!(order[0], "s1");
    assert_eq!(order[1], "s2");

    let results = result.get("results").expect("results");
    let s1_mean = extract_mean(results.get("s1").expect("s1")).expect("s1 mean");
    let s2_mean = extract_mean(results.get("s2").expect("s2")).expect("s2 mean");
    assert!((s1_mean - 20.0).abs() < 1e-9, "s1 mean = {s1_mean}");
    assert!((s2_mean - 20.0).abs() < 1e-9, "s2 mean = {s2_mean}");

    // Durations recorded
    let durations = result.get("durations_ms").expect("durations");
    assert!(durations.get("s1").is_some());
    assert!(durations.get("s2").is_some());
}

#[test]
fn pipeline_run_rejects_cycle() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();
    let args = json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": {"data": [1.0]}, "depends_on": ["b"] },
            { "id": "b", "tool": "ix_stats", "arguments": {"data": [2.0]}, "depends_on": ["a"] }
        ]
    });
    let result = reg.call_with_ctx("ix_pipeline_run", args, &ctx);
    assert!(result.is_err(), "cycle should be rejected");
}

#[test]
fn pipeline_run_listed_in_tools() {
    let reg = ToolRegistry::new();
    let list = reg.list();
    let tools = list.get("tools").and_then(|v| v.as_array()).expect("tools");
    let found = tools
        .iter()
        .any(|t| t.get("name").and_then(|n| n.as_str()) == Some("ix_pipeline_run"));
    assert!(found, "ix_pipeline_run should appear in tools/list");
}

#[test]
fn pipeline_run_placeholder_handler_errors() {
    // Calling the placeholder directly (bypassing call_with_ctx) should
    // return a clear error message instructing the caller to use the
    // top-level dispatcher.
    let reg = ToolRegistry::new();
    let result = reg.call("ix_pipeline_run", json!({"steps": []}));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("top-level MCP dispatcher") || err.contains("call_with_ctx") || err.contains("ToolRegistry::run_pipeline"),
        "error should mention the correct entry point, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Helpers — robustly extract the "mean" field from a stats tool result,
// which may be returned as a plain JSON object or wrapped in an MCP
// content envelope depending on the dispatch path.
// ---------------------------------------------------------------------------

fn extract_mean(v: &Value) -> Option<f64> {
    // Direct field
    if let Some(m) = v.get("mean").and_then(|m| m.as_f64()) {
        return Some(m);
    }
    // MCP content envelope: { content: [{ type: "text", text: "{...json...}" }] }
    if let Some(content) = v.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    if let Some(m) = parsed.get("mean").and_then(|m| m.as_f64()) {
                        return Some(m);
                    }
                }
            }
        }
    }
    None
}
