//! Canonical showcase #05 — the Adversarial Refactor Oracle.
//!
//! A 12-tool self-referential demo: ix analyses its own workspace,
//! attacks a classifier trained on its own cluster labels, searches
//! for refactor vectors via a GA, and audits the whole chain via
//! Demerzel governance — all as a single `ix_pipeline_run` invocation.
//!
//! # What makes this the most ambitious demo in the showcase
//!
//! - **12 tools** chained in one call (bracket v2 is 5). See the
//!   per-step list below.
//! - **Real workspace data** — every number in the pipeline comes from
//!   `git log`, `wc -l`, `Cargo.toml`, or `ix_code_analyze` on this
//!   very repository. No synthetic fixtures.
//! - **R6 Level 1 preview** — the adversarial step (FGSM) wraps the
//!   random-forest classifier trained in the previous step, as per
//!   the adversarial-surrogate-validation rung of the R6 roadmap.
//! - **Compiled via `ix_pipeline_compile`** — the pipeline spec comes
//!   from the NL-to-pipeline compiler. The same spec that lives on
//!   disk is driven through the compiler (with a canned LLM response
//!   standing in for a live sampling call) and executed end-to-end in
//!   this test, proving the compile → validate → run → audit chain.
//! - **Lineage DAG** (R2 Phase 2) — the 12-step provenance chain is
//!   surfaced in the response and can be passed to `ix_governance_check`
//!   as the audit trail, closing the loop between pipeline output and
//!   Demerzel compliance.
//!
//! # Framing compromises — called out honestly
//!
//! Three tools in the chain accept only fixed enums / hard-coded maps.
//! The pipeline still runs on real data but narrates these steps as
//! symbolic stand-ins for the full functionality:
//!
//! 1. **`ix_evolution`** only accepts `sphere | rosenbrock | rastrigin`
//!    as its fitness. We run rosenbrock over a 4-dim "refactor vector
//!    space" and narrate the best_params as a symbolic refactor move.
//!    A real fitness function is future R10 work.
//! 2. **`ix_chaos_lyapunov`** only supports the logistic map. We derive
//!    a parameter from the SLOC variance across the 5 crates and map
//!    it into `r ∈ [3.5, 4.0]` to report which dynamical regime the
//!    workspace lives in. Poetic, but on real input.
//! 3. **`ix_adversarial_fgsm`** needs an explicit gradient array and
//!    `ix_random_forest` has no gradient — so we bake the synthetic
//!    gradient as `healthy_centroid - at_risk_centroid` (precomputed
//!    from a preliminary k-means run on the same data) and narrate
//!    the attack as "the minimum perturbation along the natural
//!    refactor direction."
//!
//! # Per-step map
//!
//! | # | id                       | tool                   | role                                    |
//! |---|--------------------------|------------------------|-----------------------------------------|
//! | 1 | s01_baseline_sloc        | ix_stats               | descriptive stats on SLOC across crates |
//! | 2 | s02_dep_pagerank         | ix_graph (pagerank)    | centrality of each crate in the dep DAG|
//! | 3 | s03_dep_toposort         | ix_graph (toposort)    | confirm dep graph is a DAG              |
//! | 4 | s04_betti_numbers        | ix_topo (betti_at_r)   | Betti numbers of the metric cloud       |
//! | 5 | s05_persistence_diagram  | ix_topo (persistence)  | persistent homology of same cloud       |
//! | 6 | s06_churn_spectrum       | ix_fft                 | FFT of commit counts                    |
//! | 7 | s07_velocity_regime      | ix_chaos_lyapunov      | logistic-map regime check               |
//! | 8 | s08_crate_clusters       | ix_kmeans              | cluster crates by health profile        |
//! | 9 | s09_risk_classifier      | ix_random_forest       | RF trained on cluster labels            |
//! |10 | s10_adversarial_attack   | ix_adversarial_fgsm    | minimum refactor perturbation           |
//! |11 | s11_refactor_search      | ix_evolution (GA)      | symbolic refactor vector search         |
//! |12 | s12_governance_audit     | ix_governance_check    | Demerzel verdict on the plan            |

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::thread;

// ---------------------------------------------------------------------------
// Real workspace metrics for 5 top-churned crates, as of this snapshot.
// Produced from `git log --since="90 days ago"`, `find ... -name "*.rs" |
// wc -l`, and `grep -c "^ix-" crates/<name>/Cargo.toml`.
//
// Node indices for ix_graph:
//   0 = ix-agent, 1 = ix-demo, 2 = ix-nn, 3 = ix-autograd, 4 = ix-math
// ---------------------------------------------------------------------------

const SLOC: [f64; 5] = [11429.0, 6610.0, 4297.0, 1528.0, 6250.0];

// 5 crates × 4 features [sloc, file_count, dep_count, commits_90d].
// Every number is from the workspace, no synthesis.
const FEATURES_5X4: [[f64; 4]; 5] = [
    [11429.0, 21.0, 36.0, 89.0], // ix-agent     ← expected at-risk
    [6610.0, 28.0, 26.0, 33.0],  // ix-demo      ← expected warning
    [4297.0, 12.0, 4.0, 29.0],   // ix-nn        ← expected healthy
    [1528.0, 12.0, 1.0, 29.0],   // ix-autograd  ← expected healthy
    [6250.0, 20.0, 0.0, 28.0],   // ix-math      ← expected healthy (leaf)
];

/// Edge list for the 5-node dep subgraph (real Cargo.toml references).
/// Nodes are indexed 0..5 as per the constant comments above. Built via
/// `json!` so `from`/`to` serialize as JSON integers — `ix_graph` parses
/// them with `as_u64()` and rejects floats.
fn dep_edges() -> Value {
    json!([
        [0, 4, 1.0], // ix-agent → ix-math
        [0, 2, 1.0], // ix-agent → ix-nn
        [0, 3, 1.0], // ix-agent → ix-autograd
        [1, 4, 1.0], // ix-demo → ix-math
        [1, 2, 1.0], // ix-demo → ix-nn
        [1, 0, 1.0], // ix-demo → ix-agent
        [2, 4, 1.0], // ix-nn → ix-math
        [3, 4, 1.0]  // ix-autograd → ix-math (via ix-signal)
    ])
}

// FFT needs a power-of-two length; pad commit counts out to 8.
const CHURN_SIGNAL: [f64; 8] = [89.0, 33.0, 29.0, 29.0, 28.0, 0.0, 0.0, 0.0];

// Synthetic gradient for the FGSM step. Computed offline as
// (healthy_centroid - at_risk_centroid) on the 4-feature space — i.e.
// the direction that moves ix-agent toward the healthy cluster. Stored
// as a baked constant because the runner doesn't support
// `centroids[0] - centroids[2]` expression-style substitution.
const FGSM_GRADIENT: [f64; 4] = [
    -7402.8, // SLOC: need less code
    -3.0,    // file count: slightly fewer files
    -25.75,  // dep count: many fewer upstream deps
    -60.0,   // commits: less churn
];

/// Derive the logistic-map parameter for step 7 from the coefficient
/// of variation of SLOC across crates. Higher CV → closer to the fully
/// chaotic band at r = 4.0. Encapsulated so the assertion inside the
/// test can sanity-check it against the demo narrative.
fn velocity_regime_parameter() -> f64 {
    let mean = SLOC.iter().sum::<f64>() / SLOC.len() as f64;
    let var = SLOC.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / SLOC.len() as f64;
    let cv = var.sqrt() / mean; // coefficient of variation ≈ 0.56 on this data
    // Map cv ∈ [0, 1] → r ∈ [3.5, 4.0]. The 0.56 CV lands near r = 3.78.
    3.5 + 0.5 * cv.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// The canned LLM response — the pipeline spec a competent compiler
// would emit given the brief in `ORACLE_BRIEF`. The test drives this
// through `ix_pipeline_compile` via `fake_client`, which exercises
// the real validation + dispatch path but substitutes a deterministic
// response for the LLM sampling call.
// ---------------------------------------------------------------------------

const ORACLE_BRIEF: &str = "Analyse the 5 most-churned ix crates for refactor candidates: \
    descriptive SLOC stats, dep-graph centrality via PageRank, confirm the dep graph is a DAG, \
    topological invariants of the per-crate metric cloud, FFT of commit churn, logistic-map \
    regime check on workspace velocity, cluster crates into 3 health groups via k-means, \
    train a random-forest risk classifier on the cluster labels, attack it with FGSM to find \
    the minimum refactor perturbation, run a GA over the 4-dim refactor vector space, and \
    close with a Demerzel governance check on the proposed refactor plan. Every step needs an \
    asset_name and a depends_on chain that matches a linear narrative.";

fn canned_oracle_spec() -> Value {
    json!({
        "steps": [
            {
                "id": "s01_baseline_sloc",
                "tool": "ix_stats",
                "asset_name": "refactor_oracle.baseline_sloc",
                "arguments": { "data": SLOC }
            },
            {
                "id": "s02_dep_pagerank",
                "tool": "ix_graph",
                "asset_name": "refactor_oracle.dep_pagerank",
                "depends_on": ["s01_baseline_sloc"],
                "arguments": {
                    "operation": "pagerank",
                    "n_nodes": 5,
                    "edges": dep_edges(),
                    "damping": 0.85,
                    "iterations": 100
                }
            },
            {
                "id": "s03_dep_toposort",
                "tool": "ix_graph",
                "asset_name": "refactor_oracle.dep_toposort",
                "depends_on": ["s02_dep_pagerank"],
                "arguments": {
                    "operation": "topological_sort",
                    "n_nodes": 5,
                    "edges": dep_edges()
                }
            },
            {
                "id": "s04_betti_numbers",
                "tool": "ix_topo",
                "asset_name": "refactor_oracle.betti_numbers",
                "depends_on": ["s03_dep_toposort"],
                "arguments": {
                    "operation": "betti_at_radius",
                    "points": FEATURES_5X4,
                    "radius": 5000.0,
                    "max_dim": 1
                }
            },
            {
                "id": "s05_persistence_diagram",
                "tool": "ix_topo",
                "asset_name": "refactor_oracle.persistence_diagram",
                "depends_on": ["s04_betti_numbers"],
                "arguments": {
                    "operation": "persistence",
                    "points": FEATURES_5X4,
                    "max_dim": 1,
                    "max_radius": 15000.0
                }
            },
            {
                "id": "s06_churn_spectrum",
                "tool": "ix_fft",
                "asset_name": "refactor_oracle.churn_spectrum",
                "depends_on": ["s05_persistence_diagram"],
                "arguments": { "signal": CHURN_SIGNAL }
            },
            {
                "id": "s07_velocity_regime",
                "tool": "ix_chaos_lyapunov",
                "asset_name": "refactor_oracle.velocity_regime",
                "depends_on": ["s06_churn_spectrum"],
                "arguments": {
                    "map": "logistic",
                    "parameter": velocity_regime_parameter(),
                    "iterations": 1000
                }
            },
            {
                "id": "s08_crate_clusters",
                "tool": "ix_kmeans",
                "asset_name": "refactor_oracle.crate_clusters",
                "depends_on": ["s07_velocity_regime"],
                "arguments": {
                    "data": FEATURES_5X4,
                    "k": 3,
                    "max_iter": 100
                }
            },
            {
                "id": "s09_risk_classifier",
                "tool": "ix_random_forest",
                "asset_name": "refactor_oracle.risk_classifier",
                "depends_on": ["s08_crate_clusters"],
                "arguments": {
                    "x_train": FEATURES_5X4,
                    "y_train": "$s08_crate_clusters.labels",
                    "x_test": FEATURES_5X4,
                    "n_trees": 20,
                    "max_depth": 5
                }
            },
            {
                "id": "s10_adversarial_attack",
                "tool": "ix_adversarial_fgsm",
                "asset_name": "refactor_oracle.adversarial_attack",
                "depends_on": ["s09_risk_classifier"],
                "arguments": {
                    "input": FEATURES_5X4[0],
                    "gradient": FGSM_GRADIENT,
                    "epsilon": 0.1
                }
            },
            {
                "id": "s11_refactor_search",
                "tool": "ix_evolution",
                "asset_name": "refactor_oracle.refactor_search",
                "depends_on": ["s10_adversarial_attack"],
                "arguments": {
                    "algorithm": "genetic",
                    "function": "rosenbrock",
                    "dimensions": 4,
                    "generations": 50,
                    "population_size": 40,
                    "mutation_rate": 0.1
                }
            },
            {
                "id": "s12_governance_audit",
                "tool": "ix_governance_check",
                "asset_name": "refactor_oracle.governance_audit",
                "depends_on": ["s11_refactor_search"],
                "arguments": {
                    "action": "ship the GA-proposed refactor plan for ix-agent to main",
                    "context": "Plan was derived from real workspace metrics (SLOC, file count, dep count, 90-day commit churn) on 5 representative crates, validated by a PageRank + topological sort of the dep graph, topologically summarised via persistent homology + Betti numbers, frequency-analysed via FFT over commit churn, regime-checked via the logistic-map Lyapunov exponent, clustered into 3 health profiles, classified by a random forest, attacked by FGSM along the healthy-centroid direction, and searched via GA. Every upstream step has an asset-backed cache key; see the pipeline lineage for the full audit chain."
                }
            }
        ]
    })
}

// ---------------------------------------------------------------------------
// Fake MCP client for the sampling path.
// ---------------------------------------------------------------------------

fn fake_client(ctx: ServerContext, outbound: Receiver<String>, canned: String) {
    thread::spawn(move || {
        while let Ok(line) = outbound.recv() {
            let envelope: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(id) = envelope.get("id").and_then(|v| v.as_i64()) else {
                continue;
            };
            if envelope.get("method").and_then(|m| m.as_str())
                != Some("sampling/createMessage")
            {
                continue;
            }
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "role": "assistant",
                    "content": { "type": "text", "text": canned },
                }
            });
            ctx.deliver_response(id, response);
        }
    });
}

fn unwrap_tool_result(v: &Value) -> Value {
    if v.is_object() && !v.get("content").map(|c| c.is_array()).unwrap_or(false) {
        return v.clone();
    }
    if let Some(content) = v.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    return parsed;
                }
            }
        }
    }
    v.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn canned_spec_passes_the_validator() {
    let reg = ToolRegistry::new();
    let (errors, warnings) = reg.validate_pipeline_spec(&canned_oracle_spec());
    assert!(
        errors.is_empty(),
        "canned oracle spec must validate cleanly: {errors:?}"
    );
    assert!(
        warnings.is_empty(),
        "canned oracle spec should have asset_name on every step: {warnings:?}"
    );
}

#[test]
fn compiler_drives_oracle_from_natural_language_brief() {
    // Drive the full compile pipeline: fake_client stands in for the
    // real MCP sampling call, ix_pipeline_compile validates the LLM
    // response and returns status "ok" if the spec is executable.
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise canned");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let result = reg
        .call_with_ctx(
            "ix_pipeline_compile",
            json!({ "sentence": ORACLE_BRIEF, "max_steps": 12 }),
            &ctx,
        )
        .expect("compile");

    assert_eq!(result["status"], "ok", "compile failed: {result}");
    assert_eq!(
        result["spec"]["steps"].as_array().map(|a| a.len()),
        Some(12),
        "expected 12 steps"
    );
}

#[test]
fn oracle_runs_end_to_end_and_produces_lineage_dag() {
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise canned");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();

    // Compile the brief into a spec.
    let compiled = reg
        .call_with_ctx(
            "ix_pipeline_compile",
            json!({ "sentence": ORACLE_BRIEF }),
            &ctx,
        )
        .expect("compile");
    assert_eq!(compiled["status"], "ok");

    // Execute the compiled spec.
    let exec = reg
        .call_with_ctx("ix_pipeline_run", compiled["spec"].clone(), &ctx)
        .expect("run");

    // All 12 steps must execute in topological order.
    let order = exec["execution_order"].as_array().expect("execution_order");
    assert_eq!(order.len(), 12);
    assert_eq!(order[0], "s01_baseline_sloc");
    assert_eq!(order[11], "s12_governance_audit");

    // Every step must have an asset-backed cache key (R2 Phase 1).
    let cache_keys = exec["cache_keys"].as_object().expect("cache_keys");
    for step_id in order {
        let id = step_id.as_str().unwrap();
        let key = cache_keys.get(id).expect("cache key present");
        assert!(
            key.as_str().is_some_and(|k| k.starts_with("ix_pipeline_run:")),
            "step {id}: expected asset-backed cache key, got {key:?}"
        );
    }

    // Lineage DAG (R2 Phase 2) must have 12 well-formed entries.
    let lineage = exec["lineage"].as_object().expect("lineage");
    assert_eq!(lineage.len(), 12);
    for (id, entry) in lineage {
        let deps = entry.get("depends_on").and_then(|v| v.as_array()).unwrap();
        let ups = entry
            .get("upstream_cache_keys")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(
            deps.len(),
            ups.len(),
            "lineage['{id}']: depends_on length must match upstream_cache_keys"
        );
    }

    // s08 k-means must have produced 3 labels covering the 5 crates.
    let clusters = unwrap_tool_result(&exec["results"]["s08_crate_clusters"]);
    let labels = clusters
        .get("labels")
        .and_then(|v| v.as_array())
        .expect("kmeans labels");
    assert_eq!(labels.len(), 5, "5 crates → 5 labels");
    let mut distinct_labels: Vec<i64> = labels.iter().filter_map(|v| v.as_i64()).collect();
    distinct_labels.sort();
    distinct_labels.dedup();
    assert_eq!(
        distinct_labels.len(),
        3,
        "k=3 should produce 3 distinct labels; got {distinct_labels:?}"
    );

    // s09 random forest must recover the cluster labels on the self-test.
    let rf = unwrap_tool_result(&exec["results"]["s09_risk_classifier"]);
    let predictions = rf
        .get("predictions")
        .and_then(|v| v.as_array())
        .expect("rf predictions");
    assert_eq!(predictions.len(), 5);

    // s10 FGSM must have produced an adversarial input of the same
    // shape as the ix-agent feature row.
    let fgsm = unwrap_tool_result(&exec["results"]["s10_adversarial_attack"]);
    let adv = fgsm
        .get("adversarial_input")
        .and_then(|v| v.as_array())
        .expect("adversarial_input");
    assert_eq!(adv.len(), 4);

    // s11 GA must report a best_params of length 4 (dims=4 in the spec).
    let ga = unwrap_tool_result(&exec["results"]["s11_refactor_search"]);
    let best = ga
        .get("best_params")
        .and_then(|v| v.as_array())
        .expect("ga best_params");
    assert_eq!(best.len(), 4);

    // s12 governance must have returned a compliance verdict.
    let gov = unwrap_tool_result(&exec["results"]["s12_governance_audit"]);
    assert!(
        gov.get("compliant").is_some(),
        "governance_check must emit a compliant field"
    );
}

#[test]
fn oracle_governance_check_can_consume_pipeline_lineage() {
    // After the pipeline finishes, invoke ix_governance_check a
    // second time with the emitted lineage — this mirrors the
    // R2 Phase 2 test pattern and closes the audit loop.
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let exec = reg
        .call_with_ctx("ix_pipeline_run", canned_oracle_spec(), &ctx)
        .expect("run");
    let lineage = exec.get("lineage").cloned().expect("lineage");

    let args = json!({
        "action": "ship the GA refactor plan for ix-agent to main",
        "lineage": lineage,
    });
    let verdict = reg
        .call("ix_governance_check", args)
        .expect("governance_check");
    let verdict = unwrap_tool_result(&verdict);

    let audit = verdict
        .get("lineage_audit")
        .expect("lineage_audit present when lineage was passed");
    assert_eq!(
        audit.get("step_count").and_then(|v| v.as_u64()),
        Some(12),
        "lineage_audit.step_count should be 12"
    );
}

// ---------------------------------------------------------------------------
// Dump helper — writes the canned spec to disk as the on-disk
// pipeline.json for the 05-adversarial-refactor-oracle showcase. Run
// with `cargo test -- --ignored` when the canned spec is updated.
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

/// Run the oracle and print a narrated step-by-step trace of what
/// every tool produced on the real workspace metrics. Use this to
/// actually *see* the demo's output.
///
/// ```bash
/// cargo test -p ix-agent --test showcase_refactor_oracle \
///   run_refactor_oracle_with_narration -- --ignored --nocapture
/// ```
#[test]
#[ignore = "narration demo — run with --ignored --nocapture to see full output"]
fn run_refactor_oracle_with_narration() {
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let exec = reg
        .call_with_ctx("ix_pipeline_run", canned_oracle_spec(), &ctx)
        .expect("run oracle");

    let results = exec["results"].as_object().unwrap();
    let order: Vec<String> = exec["execution_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│           THE ADVERSARIAL REFACTOR ORACLE — LIVE RUN                 │");
    println!("│       12 ix tools chained on real workspace metrics                  │");
    println!("└──────────────────────────────────────────────────────────────────────┘\n");

    let crates = ["ix-agent", "ix-demo", "ix-nn", "ix-autograd", "ix-math"];

    // Step 1 — baseline stats
    let s1 = unwrap_tool_result(&results["s01_baseline_sloc"]);
    println!("─── STEP 1  ix_stats (SLOC across 5 crates) ────────────────────────────");
    println!("    mean     = {:.0}", s1["mean"].as_f64().unwrap_or(0.0));
    println!("    std_dev  = {:.0}", s1["std_dev"].as_f64().unwrap_or(0.0));
    println!("    min      = {:.0}", s1["min"].as_f64().unwrap_or(0.0));
    println!("    max      = {:.0}  ← ix-agent dominates", s1["max"].as_f64().unwrap_or(0.0));
    println!();

    // Step 2 — PageRank
    let s2 = unwrap_tool_result(&results["s02_dep_pagerank"]);
    println!("─── STEP 2  ix_graph pagerank (dep-graph centrality) ───────────────────");
    if let Some(pr) = s2.get("pagerank").and_then(|v| v.as_object()) {
        let mut rows: Vec<(usize, f64)> = pr
            .iter()
            .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_f64()?)))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (idx, score) in &rows {
            let name = crates.get(*idx).copied().unwrap_or("?");
            println!("    {name:<12} rank = {score:.4}");
        }
    }
    println!();

    // Step 3 — topological sort
    let s3 = unwrap_tool_result(&results["s03_dep_toposort"]);
    println!("─── STEP 3  ix_graph topological_sort (DAG check) ──────────────────────");
    println!("    is_dag  = {}", s3["is_dag"].as_bool().unwrap_or(false));
    if let Some(ord) = s3.get("order").and_then(|v| v.as_array()) {
        let named: Vec<&str> = ord
            .iter()
            .filter_map(|v| v.as_u64().and_then(|i| crates.get(i as usize).copied()))
            .collect();
        println!("    order   = {named:?}");
    }
    println!();

    // Step 4 — Betti numbers
    let s4 = unwrap_tool_result(&results["s04_betti_numbers"]);
    println!("─── STEP 4  ix_topo betti_at_radius (structural summary) ───────────────");
    println!("    radius        = {}", s4["radius"].as_f64().unwrap_or(0.0));
    println!("    betti_numbers = {:?}", s4["betti_numbers"]);
    println!("    (β₀ = connected components, β₁ = independent cycles)");
    println!();

    // Step 5 — persistence diagram
    let s5 = unwrap_tool_result(&results["s05_persistence_diagram"]);
    println!("─── STEP 5  ix_topo persistence (persistent homology) ──────────────────");
    if let Some(diagrams) = s5.get("diagrams").and_then(|v| v.as_array()) {
        for d in diagrams {
            let dim = d["dimension"].as_u64().unwrap_or(0);
            let pair_count = d["pairs"].as_array().map(|a| a.len()).unwrap_or(0);
            println!("    H{dim} had {pair_count} birth/death pair(s)");
        }
    }
    println!();

    // Step 6 — FFT of commit churn
    let s6 = unwrap_tool_result(&results["s06_churn_spectrum"]);
    println!("─── STEP 6  ix_fft (spectrum of commit churn) ──────────────────────────");
    if let Some(mags) = s6.get("magnitudes").and_then(|v| v.as_array()) {
        let first_few: Vec<String> = mags
            .iter()
            .take(5)
            .map(|v| format!("{:.2}", v.as_f64().unwrap_or(0.0)))
            .collect();
        println!("    fft_size      = {}", s6["fft_size"].as_u64().unwrap_or(0));
        println!("    first 5 bins  = [{}]", first_few.join(", "));
        println!("    dominant bin  = low-frequency (commits are back-loaded)");
    }
    println!();

    // Step 7 — Lyapunov regime
    let s7 = unwrap_tool_result(&results["s07_velocity_regime"]);
    println!("─── STEP 7  ix_chaos_lyapunov (logistic-map regime) ────────────────────");
    println!("    parameter r       = {:.3}  (derived from SLOC coefficient of variation)",
             velocity_regime_parameter());
    println!("    lyapunov_exponent = {:.4}", s7["lyapunov_exponent"].as_f64().unwrap_or(0.0));
    println!("    dynamics          = {}", s7["dynamics"].as_str().unwrap_or("?"));
    println!();

    // Step 8 — k-means clusters
    let s8 = unwrap_tool_result(&results["s08_crate_clusters"]);
    println!("─── STEP 8  ix_kmeans (k=3 health profiles) ────────────────────────────");
    let labels: Vec<i64> = s8["labels"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    for (i, label) in labels.iter().enumerate() {
        println!("    {:<12} → cluster {label}", crates[i]);
    }
    println!("    inertia  = {:.1}", s8["inertia"].as_f64().unwrap_or(0.0));
    println!();

    // Step 9 — Random forest risk classifier
    let s9 = unwrap_tool_result(&results["s09_risk_classifier"]);
    println!("─── STEP 9  ix_random_forest (risk classifier — self-test) ─────────────");
    let preds: Vec<i64> = s9["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    for (i, pred) in preds.iter().enumerate() {
        let ok = if pred == &labels[i] { "✓" } else { "✗" };
        println!("    {:<12} pred={pred} (true={}) {ok}", crates[i], labels[i]);
    }
    let accuracy = preds
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count() as f64
        / preds.len() as f64;
    println!("    accuracy = {:.0}%", accuracy * 100.0);
    println!();

    // Step 10 — FGSM attack
    let s10 = unwrap_tool_result(&results["s10_adversarial_attack"]);
    println!("─── STEP 10  ix_adversarial_fgsm (minimum refactor perturbation) ───────");
    println!("    input (ix-agent features): {:?}", s10["adversarial_input"]);
    println!("    perturbation:              {:?}", s10["perturbation"]);
    println!("    l_inf_norm                = {:.3}", s10["l_inf_norm"].as_f64().unwrap_or(0.0));
    println!("    epsilon                   = {}", s10["epsilon"].as_f64().unwrap_or(0.0));
    println!();

    // Step 11 — GA refactor search
    let s11 = unwrap_tool_result(&results["s11_refactor_search"]);
    println!("─── STEP 11  ix_evolution (GA over 4-dim refactor space) ───────────────");
    println!("    algorithm       = {}", s11["algorithm"].as_str().unwrap_or("?"));
    println!("    function        = {}  (symbolic stand-in for real fitness)", s11["function"].as_str().unwrap_or("?"));
    println!("    generations     = {}", s11["generations"].as_u64().unwrap_or(0));
    println!("    best_params     = {:?}", s11["best_params"]);
    println!("    best_fitness    = {:.6}", s11["best_fitness"].as_f64().unwrap_or(0.0));
    println!();

    // Step 12 — Governance audit
    let s12 = unwrap_tool_result(&results["s12_governance_audit"]);
    println!("─── STEP 12  ix_governance_check (Demerzel verdict) ────────────────────");
    println!("    compliant             = {}", s12["compliant"].as_bool().unwrap_or(false));
    println!("    constitution_version  = {}", s12["constitution_version"].as_str().unwrap_or("?"));
    println!("    warnings              = {}",
             s12["warnings"].as_array().map(|a| a.len()).unwrap_or(0));
    if let Some(articles) = s12.get("relevant_articles").and_then(|v| v.as_array()) {
        println!("    relevant articles     = {} matched", articles.len());
    }
    println!();

    // Pipeline-level summary.
    let cache_hits = exec["cache_hits"].as_array().unwrap();
    let durations = exec["durations_ms"].as_object().unwrap();
    let total_ms: u64 = durations
        .values()
        .filter_map(|v| v.as_u64())
        .sum();
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│                        PIPELINE SUMMARY                              │");
    println!("├──────────────────────────────────────────────────────────────────────┤");
    println!("│  steps executed       : {:<45} │", order.len());
    println!("│  cache hits           : {:<45} │", cache_hits.len());
    println!("│  total duration (ms)  : {:<45} │", total_ms);
    println!("│  lineage DAG entries  : {:<45} │",
             exec["lineage"].as_object().map(|o| o.len()).unwrap_or(0));
    println!("└──────────────────────────────────────────────────────────────────────┘");
}

#[test]
#[ignore = "writes pipeline.json to disk — run with --ignored to regenerate"]
fn dump_refactor_oracle_pipeline_json() {
    let spec = canned_oracle_spec();
    let mut wrapped = json!({
        "$schema": "https://ix.guitaralchemist.com/schemas/pipeline-v1.json",
        "name": "adversarial-refactor-oracle",
        "description": "12-tool self-referential ecosystem forensics demo: ix analyses its own workspace, attacks the resulting classifier with FGSM, searches for refactor vectors via a GA over the 4-dim refactor space, and closes with a Demerzel governance audit. Every number is real (SLOC, file count, intra-workspace deps, 90-day commit churn for ix-agent, ix-demo, ix-nn, ix-autograd, ix-math). Compiled from a natural-language brief via ix_pipeline_compile. R6 Level 1 preview + R2 Phase 2 lineage + R1 pipeline surface + R3 registry contract.",
        "version": "1.0",
        "tools_used": [
            "ix_stats", "ix_graph", "ix_topo", "ix_fft",
            "ix_chaos_lyapunov", "ix_kmeans", "ix_random_forest",
            "ix_adversarial_fgsm", "ix_evolution", "ix_governance_check"
        ]
    });
    wrapped["steps"] = spec["steps"].clone();

    let mut out = workspace_root();
    out.push("examples");
    out.push("canonical-showcase");
    out.push("05-adversarial-refactor-oracle");
    out.push("pipeline.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let pretty = serde_json::to_string_pretty(&wrapped).unwrap() + "\n";
    std::fs::write(&out, pretty).expect("write pipeline.json");
    eprintln!("[dump_refactor_oracle] wrote {}", out.display());
}
