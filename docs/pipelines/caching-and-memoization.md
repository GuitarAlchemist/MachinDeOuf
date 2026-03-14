# Caching and Memoization

## The Problem

You have a five-stage data pipeline that takes 30 seconds to run end-to-end. You change the normalization logic in stage 4 and re-run. Stages 1 through 3 produce exactly the same output as before --- but the pipeline recomputes them anyway, wasting 25 seconds. You need a way to skip nodes whose inputs have not changed, re-executing only the parts of the DAG that are actually affected.

This is the **incremental recomputation** problem. Build tools (Make, Bazel), notebook environments (Jupyter cell caching), and ETL frameworks (dbt) all solve versions of it. ix's pipeline executor solves it at the node level with a pluggable cache interface.

---

## The Intuition

Imagine a kitchen where each cook writes down their recipe and ingredients on a card. Before they start cooking, they check a pantry of pre-made dishes. If someone already made this exact dish with these exact ingredients, they grab it from the shelf instead of cooking from scratch. If not, they cook it and put a copy on the shelf for next time.

The "shelf" is the cache. The "recipe + ingredients" is the cache key. The `cacheable` flag on each node controls whether that cook bothers checking the shelf at all --- some dishes (like "today's special") should always be made fresh.

---

## How It Works

### The PipelineCache trait

```rust
pub trait PipelineCache: Send + Sync {
    /// Try to get a cached result for a node.
    fn get(&self, cache_key: &str) -> Option<Value>;

    /// Store a result in the cache.
    fn set(&self, cache_key: &str, value: &Value);
}
```

**In plain English:** any type that can look up and store JSON values by string key can serve as a pipeline cache. This could be an in-memory HashMap, a Redis connection, the `ix-cache` embedded store, or a file-based cache.

### Cache key generation

The executor generates a deterministic cache key for each node from:

1. The **node ID** (e.g., `"normalize"`).
2. The **input values** (sorted by input name, then hashed with FNV-1a).

```
cache_key = "pipeline:{node_id}:{hash(sorted_inputs)}"
```

**In plain English:** if the same node receives the same inputs, it produces the same cache key, and a previous result can be reused. If any input changes (because an upstream node produced different output), the hash changes and the node re-executes.

### Per-node cacheability

Each node has a `cacheable` flag (default: `true`). Use the builder's `.no_cache()` method to disable caching for nodes that:

- Have **side effects** (writing to a file, sending an HTTP request).
- Are **non-deterministic** (reading the current time, sampling random numbers).
- Should always reflect **fresh data** (reading from a live database).

```rust
.node("fetch_live_data", |b| b
    .compute(|_| { /* query database */ })
    .no_cache()  // always re-execute
)
```

### The NoCache implementation

For pipelines that do not need caching, pass `&NoCache`:

```rust
pub struct NoCache;

impl PipelineCache for NoCache {
    fn get(&self, _key: &str) -> Option<Value> { None }
    fn set(&self, _key: &str, _value: &Value) {}
}
```

This is a zero-cost way to disable caching entirely.

---

## In Rust

### Seeing cache hits in action

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, PipelineCache, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

// A simple in-memory cache
struct MemoryCache {
    store: Mutex<HashMap<String, Value>>,
}

impl MemoryCache {
    fn new() -> Self {
        Self { store: Mutex::new(HashMap::new()) }
    }
}

impl PipelineCache for MemoryCache {
    fn get(&self, key: &str) -> Option<Value> {
        self.store.lock().unwrap().get(key).cloned()
    }
    fn set(&self, key: &str, value: &Value) {
        self.store.lock().unwrap().insert(key.to_string(), value.clone());
    }
}

let pipeline = PipelineBuilder::new()
    .source("data", || Ok(json!({"values": [1, 2, 3]})))
    .node("expensive_stats", |b| b
        .input("x", "data")
        .cost(10.0)  // high cost -- we want to cache this
        .compute(|inputs| {
            println!("  Computing expensive_stats...");
            let vals = inputs["x"]["values"].as_array().unwrap();
            let sum: f64 = vals.iter().map(|v| v.as_f64().unwrap()).sum();
            Ok(json!({"sum": sum}))
        })
    )
    .build()
    .unwrap();

let cache = MemoryCache::new();
let inputs = HashMap::new();

// First run: computes everything
println!("Run 1:");
let r1 = execute(&pipeline, &inputs, &cache).unwrap();
println!("  Cache hits: {}", r1.cache_hits);  // 0

// Second run: same inputs, cache kicks in
println!("Run 2:");
let r2 = execute(&pipeline, &inputs, &cache).unwrap();
println!("  Cache hits: {}", r2.cache_hits);  // 1 (expensive_stats was cached)

// Both runs produce the same output
assert_eq!(r1.output("expensive_stats"), r2.output("expensive_stats"));
```

### Mixing cacheable and non-cacheable nodes

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, PipelineCache};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

struct MemoryCache {
    store: Mutex<HashMap<String, Value>>,
}
impl MemoryCache {
    fn new() -> Self { Self { store: Mutex::new(HashMap::new()) } }
}
impl PipelineCache for MemoryCache {
    fn get(&self, key: &str) -> Option<Value> {
        self.store.lock().unwrap().get(key).cloned()
    }
    fn set(&self, key: &str, value: &Value) {
        self.store.lock().unwrap().insert(key.to_string(), value.clone());
    }
}

let pipeline = PipelineBuilder::new()
    .source("config", || Ok(json!({"threshold": 0.5})))
    .node("load_data", |b| b
        .compute(|_| {
            // Simulates reading from a live source
            Ok(json!({"records": 42}))
        })
        .no_cache()  // always fetch fresh data
    )
    .node("process", |b| b
        .input("config", "config")
        .input("data", "load_data")
        .compute(|inputs| {
            let threshold = inputs["config"]["threshold"].as_f64().unwrap();
            let records = inputs["data"]["records"].as_f64().unwrap();
            Ok(json!(records * threshold))
        })
        // cacheable by default -- but will re-execute if load_data changes
    )
    .build()
    .unwrap();

let cache = MemoryCache::new();
let r = execute(&pipeline, &HashMap::new(), &cache).unwrap();
println!("Process output: {}", r.output("process").unwrap());
```

### Inspecting execution details from PipelineResult

```rust
use ix_pipeline::executor::PipelineResult;

fn print_execution_report(result: &PipelineResult) {
    println!("Total time: {:?}", result.total_duration);
    println!("Cache hits: {}", result.cache_hits);

    println!("\nExecution order:");
    for (level, nodes) in result.execution_order.iter().enumerate() {
        println!("  Level {}: {:?}", level, nodes);
    }

    println!("\nPer-node details:");
    for (id, node_result) in &result.node_results {
        println!("  {} -- {:?} (cache hit: {})",
            id,
            node_result.duration,
            node_result.cache_hit,
        );
    }
}
```

The `PipelineResult` fields relevant to caching:

| Field | Type | Meaning |
|-------|------|---------|
| `cache_hits` | `usize` | Total number of nodes that returned a cached result |
| `execution_order` | `Vec<Vec<NodeId>>` | Which nodes ran at each level (cache hits still appear here) |
| `node_results[id].cache_hit` | `bool` | Whether this specific node was served from cache |
| `node_results[id].duration` | `Duration` | `Duration::ZERO` for cache hits; actual time for computed nodes |

---

## When To Use This

| Situation | Caching strategy |
|-----------|-----------------|
| Pipeline runs repeatedly with same inputs | Use a persistent cache (file or `ix-cache`) |
| Pipeline runs once | Use `NoCache` to avoid overhead |
| Some nodes are non-deterministic | Mark them `.no_cache()` |
| Debugging: want to see all computations | Use `NoCache` temporarily |
| Distributed pipeline across machines | Use a shared cache (Redis-compatible, or `ix-cache` with TCP) |

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `cacheable` (per node) | Whether the executor checks/stores cache for this node | Default: `true`. Disable with `.no_cache()` for side effects or non-determinism |
| Cache key | `"pipeline:{node_id}:{input_hash}"` | Deterministic from node ID + sorted input values. Changes when any input changes |
| `PipelineCache` implementation | Where cached values are stored | In-memory HashMap for tests; `ix-cache` for production |
| `initial_inputs` | External values fed to the pipeline | Changing these changes the cache keys of all nodes that read them |

---

## Pitfalls

1. **Non-deterministic nodes poison downstream caches.** If a non-cacheable node produces different output on each run, all downstream nodes will miss the cache too (because their inputs changed). This is correct behavior, but it means caching only helps the branches that are purely deterministic.

2. **Cache does not expire automatically.** The pipeline executor does not implement TTL or eviction. If you use a simple HashMap cache, it grows without bound. Connect to `ix-cache` (which supports TTL and LRU eviction) for production workloads.

3. **Large JSON values are expensive to cache.** The cache stores `serde_json::Value` objects. If a node outputs a multi-megabyte JSON array, cloning it into the cache and back out adds overhead. For large intermediate data, consider storing a reference (file path, buffer ID) rather than the data itself.

4. **Cache key collisions are theoretically possible.** The FNV-1a hash used for cache keys is fast but not cryptographic. Two different input sets could (extremely rarely) produce the same hash, causing incorrect cache hits. For safety-critical pipelines, implement a `PipelineCache` that also stores and verifies the full input alongside the result.

5. **No automatic cache invalidation on code changes.** If you change a node's compute function but keep the same inputs, the cache will return stale results from the old function. Clear the cache when you change pipeline logic, or include a version number in your node IDs (e.g., `"normalize_v2"`).

---

## Going Further

- **Connect to `ix-cache`** for a production-grade cache with TTL, LRU eviction, pub/sub notifications, and a Redis-compatible RESP server. Implement `PipelineCache` to call into the cache crate's API.
- **Content-addressed caching** hashes the node's *code* (or a version tag) alongside its inputs, automatically invalidating when logic changes. This is how Bazel and Nix achieve reproducible builds.
- **Partial re-execution.** Use `dag.has_path(changed_node, target_node)` to determine which downstream nodes need to re-run after a change, without re-executing the entire pipeline.
- **[DAG Execution](./dag-execution.md)** covers the pipeline builder, executor, error types, and parallel-levels scheduling in full detail.
