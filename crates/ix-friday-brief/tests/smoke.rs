//! Smoke test for the Friday Brief pipeline.
//!
//! Redirects the `state/` root to a tempdir via `IX_FRIDAY_BRIEF_STATE_DIR`
//! so the test never touches the real repo state. Asserts both artifacts
//! exist, the brief contains the title, the snapshot carries the
//! `trust: "inferred"` tag, and the pipeline trace contains all 12 nodes
//! in declared order.

use std::fs;

use ix_friday_brief::{run, PIPELINE_NODE_ORDER};

#[test]
fn smoke_run_produces_brief_and_snapshot() {
    // Use a unique subdir inside the system temp dir so parallel runs
    // don't collide, and so we never pollute the repo's state/ tree.
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "ix-friday-brief-smoke-{}",
        std::process::id()
    ));
    if tmp.exists() {
        let _ = fs::remove_dir_all(&tmp);
    }
    fs::create_dir_all(&tmp).unwrap();

    // SAFETY: tests in this file are serial by default at the Rust test
    // runner level for a single test binary, but `set_var` is still
    // `unsafe` on 2024+. We're on 2021 edition so the legacy signature
    // applies.
    std::env::set_var("IX_FRIDAY_BRIEF_STATE_DIR", &tmp);

    let artifacts = run().expect("friday brief run should succeed");

    assert!(
        artifacts.brief_path.exists(),
        "brief not written at {}",
        artifacts.brief_path.display()
    );
    assert!(
        artifacts.snapshot_path.exists(),
        "snapshot not written at {}",
        artifacts.snapshot_path.display()
    );

    let brief = fs::read_to_string(&artifacts.brief_path).unwrap();
    assert!(
        brief.contains("Friday Brief"),
        "brief missing title, got:\n{brief}"
    );

    let snap_raw = fs::read_to_string(&artifacts.snapshot_path).unwrap();
    let snap: serde_json::Value = serde_json::from_str(&snap_raw).unwrap();
    assert_eq!(
        snap.get("trust").and_then(|v| v.as_str()),
        Some("inferred"),
        "snapshot missing trust=inferred tag: {snap_raw}"
    );

    // Pipeline trace must contain all 12 declared nodes, in order.
    assert_eq!(
        artifacts.pipeline_trace.len(),
        PIPELINE_NODE_ORDER.len(),
        "expected {} nodes in trace, got {:?}",
        PIPELINE_NODE_ORDER.len(),
        artifacts.pipeline_trace
    );
    for (i, expected) in PIPELINE_NODE_ORDER.iter().enumerate() {
        assert_eq!(
            &artifacts.pipeline_trace[i], expected,
            "node {i} mismatch: got {:?}, expected {expected}",
            artifacts.pipeline_trace[i]
        );
    }

    // Cleanup — best effort, don't fail the test if it can't remove.
    let _ = fs::remove_dir_all(&tmp);
}
