# Context DAG over ix-code — deterministic structural retrieval for Claude agents

**Status:** brainstorm → ready for `/octo:develop` handoff
**Date:** 2026-04-10
**Session format:** Multi-AI brainstorm (Codex + Gemini + Claude) moderated by Claude Opus 4.6
**MVP target:** 1–2 days
**Related:** `docs/brainstorms/2026-04-10-ix-harness-primitives.md` (meta-architecture this feeds into)

---

## Opening reframe — the governance instrument disguised as a retrieval system

A Context DAG over ix-code is **not primarily a better retrieval system for Claude**. Claude is the beneficiary; Demerzel is the customer.

Vector RAG is opaque: a wrong answer is unattributable — bad similarity? bad chunking? bad luck? A walked DAG makes the agent's uncertainty *legible* — wrongness localizes to either the walk policy or the reasoning. This is the same move as static typing: you don't eliminate bugs, you **relocate** them to a place where they can be named.

That relocation is the entire point. Every agent action becomes replayable: which nodes did the agent see? Which edges did it cross? Which governance verdicts flagged which crossings? With a deterministic walker, Demerzel can reconstruct the exact informational state the agent acted on. With embedding RAG, it cannot — not even in principle.

So the design decisions in this doc flow from one axiom: **every walk must be replayable, every edge must carry provenance, every label must map to hexavalent belief**. If a design choice breaks replayability for a latency or ergonomics win, it's the wrong choice.

This is an IX-native answer to the RAG trap. It's also the first concrete payload for the "structural oracle for harnesses" line of thinking developed in the meta doc.

---

## Motivation in one paragraph

IX shipped the Code Observatory this session — 7 layers of structural analysis over Rust code (tree-sitter AST, git2 trajectory, persistent homology, hexavalent gates, hyperbolic/K-theory/BSP, physics). IX also has `ix-pipeline` with a generic cycle-checked `Dag<N>` and `ix-cache` with TTL/LRU/pubsub. The missing piece between them is a **walker** that treats code structure as the substrate for context retrieval. The answer to the Reddit/DAG-vs-RAG debate was already 70% built; this brainstorm pins down the remaining 30%.

---

## Five load-bearing insights from the multi-AI session

1. **[Codex]** Store unresolved edges, not just resolved ones. Ambiguity is signal. An edge labelled "this call might resolve to one of `{A, B, C}`" is more useful than silently dropping it.
2. **[Codex]** Split IDs from labels early. Stable node IDs (`fn:crate::mod::name@path#span`) survive renames; labels don't. Cache invalidation and trajectory joins break without this.
3. **[Gemini]** Git trajectories are *desire paths* — invisible high-weight edges between files that don't import each other but change together. They're already computed by `ix-code::trajectory`; the walker just needs to read them.
4. **[Claude]** Belief-weighted walks: frontier ordered by *unresolved* count, not similarity. *Relevance = unresolved dependency, not similarity.* (Stretch, not MVP.)
5. **[Claude]** Persistent homology as stopping rule: walk until H₁ stabilizes. Uses the `ix-topo` code already shipped. (Stretch, not MVP.)

---

## Scope decisions

### What's in the MVP (1–2 days)

- **Crate:** `crates/ix-context/` (new wrapper crate, not a pipeline-core modification)
- **Symbol resolution:** two-pass tree-sitter walker producing `HashMap<SymbolKey, DefSite>`; resolution by ordered heuristics (local → use-alias → module → impl → trait → ambiguous fallback)
- **Node schema:** `enum ContextNode { Function, Module, File, Test, Commit, Symbol }` with shared `NodeMeta { id, label, confidence, provenance }`
- **Walk strategies:** all four picked in the challenge round
  1. Callers-transitive (upstream)
  2. Callees-transitive (downstream)
  3. Module-siblings
  4. Git co-change (reuses shipped `trajectory::compute_trajectory`)
- **Cache:** keyed by `(git HEAD SHA, file content hash, workspace manifest hash)`; invalidated via `notify` file watcher publishing through `ix-cache` pub/sub. TTL only as leak-prevention fallback, never for correctness.
- **Public API:** library-level `Walker::walk(from, strategy, budget) -> ContextBundle`
- **MCP tool:** thin wrapper `ix_context_walk(target, strategy, budget)` exposed through `ix-agent`
- **Edge provenance:** every edge tagged with source — `ast_call | import_hint | git_cochange | test_reference | sibling`
- **Unresolved edges:** preserved and exposed with the set of candidate targets
- **Stable IDs:** `fn:crate::mod::name@path#span` for functions, analogous for other node kinds
- **Hexavalent labels on nodes:** consumes `ix_types::Hexavalent` already used by `ix-code::gates`; MVP assigns `U` (Unknown) by default except where gates provide a verdict
- **Tests:** unit tests per walk strategy + integration test that walks a real crate in the workspace (`ix-math` is a good victim)

### What's *not* in the MVP (documented stretch)

| # | Stretch feature | Why deferred | Lands in |
|---|---|---|---|
| S1 | **Belief-weighted walks** — frontier ordered by unresolved count, walks *toward* Unknown/Dependent edges | Needs a belief-aware walk policy on top of the basic graph. Basic walks must ship first. | v2 |
| S2 | **Persistent homology stopping rule** — stop walking when H₁ stabilizes using `ix-topo` | Needs subgraph-level homology computation per walk, not per file. Non-trivial to get right. | v2 |
| S3 | **Hybrid landing-zone mode** — Gemini's idea: embeddings to find the first 3 files, DAG for the expansion | Violates the no-embeddings purism; reconsider only if pure-DAG has unacceptable cold-start latency | v3 or never |
| S4 | **Federated cross-repo DAG** — edges crossing Rust/F#/C# boundaries (ix ↔ tars ↔ ga) | Requires instrumentation in TARS and GA that does not exist yet | v3 |
| S5 | **Blast-radius walk** — follow `Result<T, E>` propagation through match arms | Needs richer call-site metadata than MVP symbol resolver produces | v2 |
| S6 | **Feature-flag shadow walk** — traverse mutually-exclusive `#[cfg]` paths | Requires cfg evaluation; rust-analyzer territory | v3 |
| S7 | **Hyperbolic / Laplacian spectral walks** | Expensive and unjustified until basic walks are proven insufficient | v3 |
| S8 | **Walk-as-reasoning-step** (Claude's insight) — Claude drives each walk step interactively through `walk_step(cursor)` instead of batch `walk(from, budget)` | Requires v1 API to stabilize first so the incremental API has something to wrap | v2 |

---

## Architecture

### Crate layout

```
crates/ix-context/
├── Cargo.toml               # depends on ix-code, ix-pipeline, ix-cache, ix-types, ix-governance (optional)
├── src/
│   ├── lib.rs               # re-exports; crate-level docs with the governance reframe
│   ├── model.rs             # ContextNode enum, NodeMeta, EdgeProvenance, ContextBundle, stable IDs
│   ├── index.rs             # project walk + SymbolKey → DefSite table (pass 1)
│   ├── resolve.rs           # call-site resolution heuristics (pass 2), unresolved edge preservation
│   ├── walk.rs              # Walker struct + four strategies + budget enforcement
│   ├── cache.rs             # SHA/hash-keyed cache on top of ix_cache::Cache
│   └── mcp.rs               # thin MCP tool wrapper for ix-agent registration
└── tests/
    ├── symbol_index.rs      # two-pass resolver correctness
    ├── walk_callers.rs      # caller-transitive walks
    ├── walk_callees.rs
    ├── walk_siblings.rs
    ├── walk_cochange.rs     # git co-change over a real repo fixture
    ├── unresolved_edges.rs  # verifies ambiguity is preserved, not dropped
    ├── cache_invalidation.rs
    └── integration.rs       # walks ix-math and asserts sensible bundles
```

Small extension also needed in `crates/ix-code/src/semantic.rs`: expose richer call-site records rather than just `String` names — today `CallEdge.callee` is a bare string, but the resolver needs at least the scoped path seen at the call site (e.g., `foo::bar::baz` vs `baz`) to resolve without ambiguity.

### Node schema (concrete)

```rust
use ix_types::Hexavalent;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ContextNode {
    Function(FunctionNode),
    Module(ModuleNode),
    File(FileNode),
    Test(TestNode),
    Commit(CommitNode),
    Symbol(SymbolNode),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    /// Stable ID, e.g. `fn:ix_math::eigen::jacobi@crates/ix-math/src/eigen.rs#L42-L103`
    pub id: String,
    /// Human-readable label. May change across renames; IDs may not.
    pub label: String,
    /// Hexavalent belief about this node's presence/validity.
    /// Default `Unknown`. `ix-code::gates` verdicts overwrite for analyzed nodes.
    pub belief: Hexavalent,
    /// Where this node was discovered.
    pub provenance: NodeProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeProvenance {
    TreeSitter { file: String, span: (usize, usize) },
    GitHistory { commit: String },
    Topology { cluster_id: u32 },
    Gate { verdict_source: String },
}
```

### Edge schema

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEdge {
    pub from: String,        // NodeMeta.id
    pub to: ResolvedOrAmbiguous,
    pub provenance: EdgeProvenance,
    pub weight: f64,         // 1.0 for AST edges; co-change weight from trajectory
    pub belief: Hexavalent,  // MVP: Unknown unless from gates
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolvedOrAmbiguous {
    Resolved(String),              // single target node ID
    Ambiguous(Vec<String>),        // candidate target IDs (e.g., trait method with multiple impls)
    Unresolved { hint: String },   // best-effort name hint, no candidate set
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeProvenance {
    AstCall { call_site_line: usize },
    ImportHint { via_use_alias: String },
    GitCochange { commits_shared: u32, confidence: f64 },
    TestReference { test_fn_id: String },
    Sibling { parent_module: String },
}
```

### Walker API

```rust
pub struct Walker<'a> {
    index: &'a ProjectIndex,
    cache: &'a ContextCache,
}

#[derive(Debug, Clone, Copy)]
pub enum WalkStrategy {
    CallersTransitive { max_depth: u8 },
    CalleesTransitive { max_depth: u8 },
    ModuleSiblings,
    GitCochange { min_confidence: f64 },
}

#[derive(Debug, Clone)]
pub struct WalkBudget {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub timeout: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBundle {
    pub root: String,               // starting node ID
    pub strategy: WalkStrategy,
    pub nodes: Vec<ContextNode>,
    pub edges: Vec<ContextEdge>,
    pub unresolved_count: usize,    // exposed for agent reasoning
    pub walk_trace: Vec<WalkStep>,  // replayable — the governance-legibility requirement
}

impl<'a> Walker<'a> {
    pub fn walk(
        &self,
        from: &str,
        strategy: WalkStrategy,
        budget: WalkBudget,
    ) -> Result<ContextBundle, WalkError>;
}
```

The `walk_trace: Vec<WalkStep>` is the replayability primitive. Every node visit, every edge traversal, every budget check gets a `WalkStep`. A later governance audit can feed the same trace back through the same walker and verify bit-exact reproduction.

### Cache keying

```rust
pub struct CacheKey {
    pub git_head_sha: String,     // from git2::Repository::head
    pub file_hash: u64,           // xxhash of file content at HEAD
    pub manifest_hash: u64,       // xxhash of Cargo.toml content
    pub strategy: WalkStrategy,
    pub root: String,             // node ID
}
```

Invalidation: `notify` file watcher publishes `{path, new_hash}` events to an `ix-cache` pubsub channel. The walker's cache subscribes and evicts any key whose `file_hash` matches or whose `root` points to the changed file.

### MCP tool wrapper

```rust
// In crates/ix-context/src/mcp.rs, registered from ix-agent
pub fn register(agent: &mut McpAgent) {
    agent.register_tool("ix_context_walk", |params| {
        // params: { target: String, strategy: String, budget: { max_nodes, max_edges, timeout_ms } }
        // returns: ContextBundle as JSON
    });
}
```

The MVP tool is *intentionally dumb*: fixed parameter shape, no policy learning, no incremental API. Smarter versions (walk-as-reasoning-step, belief-weighted walks) ride on top of this once it stabilizes.

---

## Walk strategy details

### 1. Callers-transitive

Input: function node ID, `max_depth`.
Algorithm: BFS over reverse call-graph edges (`edge.to == root`), collecting caller nodes until `max_depth` or budget exhausted.
Cost: cheap. Just a reverse adjacency lookup.
Value: highest. "Who calls this?" is the most common agent question.

### 2. Callees-transitive

Same as callers but forward direction. Slightly noisier because standard-library and third-party callees bloat the frontier.
Mitigation: filter out edges where the target is outside the workspace (detectable via the symbol table knowing which files belong to the project).

### 3. Module siblings

Input: any node.
Algorithm: look up the node's parent module, enumerate all other nodes in that module, return as siblings.
Cost: trivial once the symbol table exists.
Value: moderate. Useful for "show me everything in this file" without re-reading.

### 4. Git co-change

Input: file node.
Algorithm: use `ix-code::trajectory` to get the commits that touched this file. For each commit, enumerate other files touched in the same commit. Aggregate by file: `(file, cochange_count)`. Filter by `min_confidence` (default: present in at least 3 shared commits).
Cost: medium — reuses the already-shipped git2 revwalk path, no checkouts.
Value: high. This is the invisible edge no static tool can see.

---

## Contract with Demerzel (what the auditor gets)

Every `ContextBundle` must satisfy:

1. **Replayable**: `walk_trace` is sufficient to reconstruct the exact set of visited nodes and edges when fed back through the same `ProjectIndex` at the same git SHA.
2. **Hexavalent-labelled**: every node and edge carries a `Hexavalent` value (MVP default: `Unknown` unless ix-code gates say otherwise).
3. **Provenance-tagged**: every node cites its `NodeProvenance`, every edge its `EdgeProvenance`. No anonymous edges.
4. **Unresolved-preserving**: ambiguous call sites are surfaced as `ResolvedOrAmbiguous::Ambiguous(candidates)`, not silently dropped or arbitrarily chosen.
5. **Budget-honouring**: `budget` is a hard limit, not a hint. If the walker hits the budget, it returns a partial bundle with a `truncated: true` flag rather than timing out.

These five properties together mean: a skeptical-auditor persona can ingest the bundle, replay it, check provenance, and flag any crossing of a hexavalent threshold. **That's the governance instrument.**

---

## Open questions (explicit — do not defer silently)

1. **Naming.** Candidates from the brainstorm:
   - *Verve* (Gemini — short, verb-able)
   - *Stanza* (Gemini — music tie-in, GA-adjacent)
   - *Lattice* (Gemini — crystalline determinism)
   - *Incidence* (Claude — graph-theory vs *resonance* for similarity retrieval)
   - *Querywalk* (Claude — contrast with indexwalk and vectorhit)
   - *ix-context* (Codex — boring, descriptive, the crate name regardless)

   Recommendation: ship as `ix-context` (crate name) and call the concept **querywalk** in docs — it's the most memorable non-cute name and contrasts cleanly with RAG.

2. **Unresolved budget.** When a walk hits `Ambiguous(N candidates)` with N large, should it expand all candidates (blows budget) or mark the frontier as "paused here, N alternatives"? MVP: the latter. Document the trade-off.

3. **Cross-file call resolution correctness.** The two-pass resolver will be wrong in predictable ways (trait dispatch with multiple impls, macro-generated calls, re-exports, generics). The question isn't "how do we fix it" — it's "how do we surface the wrongness so agents and auditors can see it." Answer: every unresolved/ambiguous edge counts toward `unresolved_count` in the bundle, and Demerzel can threshold on that ratio.

4. **Integration with `ix-approval`.** The meta doc proposes a deterministic blast-radius policy using the Context DAG. The boundary between `ix-context` (walks) and `ix-approval` (verdicts) needs to be drawn so the walker doesn't grow approval semantics. Decision: walker returns bundles, `ix-approval` consumes them. Bundle contract stays stable.

5. **Does `Walker` depend on `ix-governance`?** MVP says yes (optional), because nodes carry `Hexavalent` which lives in `ix-types` but gate-verdict derivation lives in `ix-code::gates`. The dependency is: `ix-context → ix-code (gates feature) → ix-types`. No direct `ix-governance` dep in MVP; that enters only when `ix-approval` is built.

---

## Multi-perspective analysis

### Provider contributions

| Provider | Key contribution | Unique insight |
|---|---|---|
| 🔴 Codex | Concrete implementation plan: two-pass resolver, enum schema, cache keying, MVP file layout, 1-2 day scope | *"Store unresolved edges, not just resolved ones. Ambiguity is signal."* Also the stable-ID/label split — a seemingly boring detail that prevents cache chaos downstream. |
| 🟡 Gemini | Analogies that reframe the problem: music theory (V7 resolution → callers as harmonic resolution), gene regulatory networks (promoters/suppressors), urban desire paths (git trajectories), hybrid RAG+DAG modes, federated cross-language DAG | *"Stop searching for code. Start navigating the Code Topology."* And the desire-path insight: git trajectories are high-weight edges no static tool can see. |
| 🔵 Claude | Paradoxes and naming: governance-instrument reframe, saturation-horizon concept, belief-weighted walks toward *unresolved* not *similar*, DAG walk as Claude tool not preprocessing, persistent homology as termination theory | *"A Context DAG is a governance instrument disguised as a retrieval system. The agent is the beneficiary; the constitution is the customer."* This reframes the entire design. |

### Cross-provider patterns

- **Convergence on node schema**: Codex (enum variants) and Claude (kind-tagged variants with shared meta) and Gemini (analogies map onto discrete kinds) all reach for structured nodes, not flat dicts. No provider advocated stringly-typed schemas.
- **Convergence on hybrid over pure**: Gemini explicit (embeddings for landing zone), Claude implicit (stochastic exploration operator needed for discovery). Both acknowledge that pure deterministic walks miss the unreachable. Dropped from MVP but noted as real.
- **Convergence on "beyond the obvious walks"**: Gemini's blast-radius walk over `Result<T,E>`, Codex's unresolved-edge preservation, and Claude's belief-weighted walks all point at the same insight — *the interesting walks are determined by question shape, not node similarity*.
- **Divergence on target audience**: Codex treats this as an agent tool. Gemini treats it as a navigation UX. Claude treats it as a governance instrument. The MVP adopts Claude's framing because it has the highest constraint density — a design that satisfies governance replayability is automatically a valid agent tool but not vice versa.

### Historical rhymes (Claude)

The "refuse the popular solution, commit to structure" pattern recurs across CS history: Codd vs. network/hierarchical DBs, Prolog vs. expert-system shells, TDD vs. REPL-debug, static vs. dynamic typing. Structural answers demo slower but compound harder. This brainstorm is in that lineage.

---

## Done criteria for MVP

1. `crates/ix-context/` builds and tests pass under `cargo test -p ix-context`
2. Integration test walks `ix-math` and returns a bundle with at least 10 nodes and 15 edges for a non-trivial root
3. The four walk strategies each have a dedicated test with hand-constructed expected bundles
4. Cache invalidation integration test: modify a file, assert the stale bundle is evicted within 100ms
5. Unresolved-edge preservation test: construct a trait dispatch scenario and assert `Ambiguous(candidates)` is emitted, not a silent guess
6. MCP tool registered in `ix-agent` and callable from Claude Code with a real repo fixture
7. README section in `crates/ix-context/README.md` (or lib.rs docs) explaining the governance reframe and linking to this brainstorm
8. French translation stub created in `docs/fr/brainstorms/2026-04-10-context-dag.fr.md` (full translation can follow — user is French, doc will eventually need full translation per ecosystem convention)

---

## Handoff to `/octo:develop`

- **Crate to create:** `crates/ix-context/`
- **Workspace root:** `C:\Users\spare\source\repos\ix`
- **Depends on (already shipped):** `ix-code`, `ix-pipeline`, `ix-cache`, `ix-types`
- **Does NOT depend on:** `ix-governance` (until `ix-approval` ships)
- **Blocking bug to fix first:** `ix-code::semantic::CallEdge.callee` is `String` — needs to preserve the scoped path seen at the call site, not just the final segment. Small edit in `crates/ix-code/src/semantic.rs` before the resolver is implementable.
- **Entry point:** start with `model.rs` (types), then `index.rs` (pass 1), then `resolve.rs` (pass 2), then `walk.rs` (strategies), then `cache.rs`, then `mcp.rs`. Tests alongside each module.
- **Success signal:** a `/octo:develop` run that ends with a green `cargo test -p ix-context` and a hand-verified walk over `ix-math::eigen::jacobi` returning the Jacobi callers (MDS, Kernel PCA, LDA) deterministically.

---

## Next steps beyond MVP

See `docs/brainstorms/2026-04-10-ix-harness-primitives.md` for the full meta-architecture. Direct dependents of this MVP:

- **v2: belief-weighted walks** — frontier ordered by unresolved-count, walks toward Unknown/Doubtful
- **v2: persistent homology stopping rule** — `ix-topo` Betti stabilization as termination
- **v2: walk-as-reasoning-step** — `walk_step(cursor)` incremental API
- **`ix-approval`** — consumes bundles to compute deterministic blast-radius verdicts for Claude Code auto-mode integration
- **`ix-session`** — append-only event log peer to the Context DAG (structure of history ↔ structure of code)
