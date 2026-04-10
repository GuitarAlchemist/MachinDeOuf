# IX as a structural oracle for agent harnesses — meta-architecture

**Status:** brainstorm / strategic — **NOT** ready for direct `/octo:develop` handoff
**Date:** 2026-04-10
**Companion doc:** `docs/brainstorms/2026-04-10-context-dag.md` (the first concrete deliverable in this line)
**Scope:** 6 primitives + 1 migration + 1 flywheel. Months of work, not days. This doc exists so the insights from this session aren't lost.

---

## Thesis in one sentence

> **Every modern agent harness is converging on the same short list of primitives — append-only session logs, stateless control loops, structural (non-vector) context, deterministic guardrails, evaluator-with-real-tools, middleware hooks, loop detection, trace-driven iteration. IX already ships the math and governance to be the best backend for those primitives, and it doesn't know it yet.**

This doc maps each primitive to an IX home, names the gaps, and proposes a coherent order of construction.

---

## The five-source convergence

Over this session we read (or tried to read) five independent pieces on harness engineering:

1. Anthropic — *Harness design for long-running apps* (Opus 4.6 context anxiety, planner/generator/evaluator, file handoffs)
2. Anthropic — *Managed Agents* (session/harness/sandbox as virtualized primitives, append-only event log as source of truth)
3. Anthropic — *Claude Code auto-mode* (tiered allowlists, transcript classifier, 93% approve-rate problem, blast-radius judgment)
4. LangChain — *Improving deep agents with harness engineering* (+13.7 pp on Terminal Bench from harness-only changes; middleware pattern; LocalContext, PreCompletionChecklist, LoopDetection, Trace Analyzer)
5. OpenAI — *Harness engineering* (fetch blocked 403; reasoning from Swarm/Codex conventions: handoff as primitive, routines + handoffs, single-tool bias, eval-first shipping)
6. SethGammon — *Citadel* (4-tier routing, 22 hooks across 14 lifecycle events, circuit breaker, fleet mode with discovery relay)

| Theme | Anthropic harness | Managed Agents | Auto-mode | LangChain | OpenAI (inferred) | Citadel |
|---|---|---|---|---|---|---|
| Event log as primitive state | ✓ file handoffs | ✓ append-only | — | — | ✓ transcript | ✓ campaigns |
| Stateless control loop | ✓ | ✓ wake(sessionId) | — | — | ✓ | partial |
| Deterministic guardrails | — | — | ✓ tier 1/2 allowlist | partial | — | ✓ 22 hooks |
| Structural/pre-loaded context | implied | ✓ getEvents() | — | ✓ LocalContext | — | — |
| Evaluator with real tools | ✓ | — | ✓ classifier | ✓ PreCompletionChecklist | ✓ | partial |
| Middleware hooks | — | — | — | ✓ | partial | ✓ |
| Loop/spiral detection | — | — | implied | ✓ LoopDetection | — | ✓ circuit breaker |
| Trace-driven harness iteration | — | — | — | ✓ Trace Analyzer | ✓ | — |
| Reasoning budget allocation | — | — | — | ✓ sandwich | — | — |
| Parallel worktree agents | — | — | — | — | — | ✓ Fleet |

The pattern is unmistakable. Every article converges on the same short list.

---

## The six proposed primitives + two support items

| # | Primitive | What it is | Sources validating | IX home | Cost | Ship order |
|---|---|---|---|---|---|---|
| **1** | **Context DAG** (`ix-context`) | Deterministic structural retrieval over code. Walker + bundle + MCP tool. | Anthropic harness, Managed Agents, LangChain LocalContext, Reddit DAG post | new crate `ix-context` | 1–2 days MVP | **Ship first** — companion brainstorm has the detailed plan |
| **2** | **Middleware stack** (`ix-middleware`) | Pre/post hook chain around every MCP tool call. Ships with LoopDetection, PreCompletionChecklist, TimeBudget injectors. | LangChain (directly), Citadel (hooks), Auto-mode (classifier-as-hook) | new crate or module under `ix-agent` | 1–2 days | Ship second |
| **3** | **Approval policy** (`ix-approval`) | Deterministic blast-radius verdict. Tier 1/2/3 classifier backed by `ix-context` walks + `ix-code::trajectory` volatility + Demerzel constitution. Exposes `classify_action(action, context, code_dag) → ApprovalVerdict`. | Auto-mode (directly), Demerzel | module in `ix-governance` or new `ix-approval` crate | 1 day | Ship third (depends on 1) |
| **4** | **Session event log** (`ix-session`) | Append-only JSONL log as source of truth for agent history. Typed belief/PDCA state become *projections* of the log, not the source. | Managed Agents (directly), OpenAI transcript-as-state, Citadel campaigns | new crate `ix-session`; migrates `state/beliefs/*` to projections over `state/events/*.log` | 2–3 days (crate is small; migration is the work) | Ship fourth |
| **5** | **Fuzzy logic** (`ix-fuzzy`) | FIS (Fuzzy Inference Systems), membership functions, F-AHP, F-TOPSIS. Complements hexavalent: fuzzy computes continuous confidence, hexavalent buckets it. Closes a documented gap — Demerzel has fuzzy *schemas* but no Rust *implementation*. | Demerzel specs (`governance/demerzel/logic/fuzzy-membership.md`, `schemas/fuzzy-*.schema.json`, behavioral test cases), Article user pasted | new crate `ix-fuzzy` | 2–3 days | Ship fifth |
| **6** | **Trace flywheel wiring** | Connect `ix-agent` tool invocations to the already-existing `self-improve-loop` skill: emit traces on every session → `ix_trace_ingest` → analysis → `ix-grammar` evolution → harness-change proposals. Uses `ga_bridge` + `tars_bridge` infrastructure already in place. | LangChain Trace Analyzer, OpenAI eval-first | modify `ix-agent` + wire existing federation | 0.5 day | Ship sixth (cheap, high leverage) |
| **A** | **Hexavalent migration** | Retire `ix-governance::TruthValue` (4-valued). Migrate karnaugh / remediation_optimizer / violation_pattern / weight_evolution to `ix_types::Hexavalent` (6-valued). Rename `tetravalent.rs` module → `logic.rs`. Remove `Tetravalent` skill redirect. | Two-truth-types inconsistency discovered this session; user's hexavalent feedback memory | in-place edits in `ix-governance` | 1 day | Ship between #2 and #3 to unblock #3 cleanly |
| **B** | **Ambiguous call metadata fix** | `ix-code::semantic::CallEdge.callee` is a bare `String`. Needs to preserve the full scoped path seen at the call site for the Context DAG resolver. | Codex brainstorm finding | small edit in `crates/ix-code/src/semantic.rs` | <0.5 day | Ship **before** #1 as a prerequisite |

**Total:** roughly 10–13 days of focused work to build the whole system. Individual ships are all 0.5–3 days — so this is compoundable across many sessions, not a single heroic push.

---

## Primitive-by-primitive design sketches

### 1. Context DAG (`ix-context`) — **see companion doc**

Full design in `docs/brainstorms/2026-04-10-context-dag.md`. Summary: deterministic walker over AST + call graph + git trajectory, `enum ContextNode` schema with stable IDs, four MVP walk strategies (callers, callees, siblings, git co-change), belief/homology as v2. Framed as a governance instrument: every bundle is replayable and hexavalent-labelled.

### 2. Middleware stack (`ix-middleware`)

The gap: `ix-agent` is strict request-response. No pre/post hooks around tool calls. LangChain's middleware pattern and Citadel's 22-hook system both solve this with a chain of handlers. Auto-mode's classifier is conceptually a pre-hook too.

```rust
pub trait ToolMiddleware: Send + Sync {
    fn pre(&self, action: &mut ToolAction, ctx: &mut AgentContext) -> MiddlewareVerdict;
    fn post(&self, action: &ToolAction, result: &ToolResult, ctx: &mut AgentContext) -> MiddlewareVerdict;
}

pub enum MiddlewareVerdict {
    Continue,
    Block { reason: String, hexavalent: Hexavalent },
    Transform { new_action: ToolAction },
    Retry,
    Escalate,
}
```

MVP ships three middlewares:

- **`LoopDetection`** — sliding-window per-target edit counter. Fires `Block { reason: "10 edits to same file in 5 min — reconsider" }` when the threshold trips. ~50 LOC.
- **`PreCompletionChecklist`** — before an agent declares a task done, injects a checklist of evaluator-style questions ("did you run the tests? did you check behavior vs. description?"). Invoked by the `skeptical-auditor` persona. Directly mirrors LangChain's `PreCompletionChecklistMiddleware`.
- **`TimeBudgetInjector`** — publishes a "you have N seconds left" system message at T/2, 3T/4, 7T/8 of the configured budget. Trivial. LangChain showed this moves agents from exploration to verification.

Integration: `ix-agent`'s tool dispatcher grows a `Vec<Box<dyn ToolMiddleware>>` that it folds over on every tool invocation. Middleware is config-driven — loaded from `.claude/ix-middleware.toml` or similar.

### 3. Approval policy (`ix-approval`)

The auto-mode article's specific weakness: **17% FNR on blast-radius judgment** because the classifier is uncertain whether user consent covers the specific scope. IX can do better *deterministically* by leveraging Context DAG walks.

```rust
pub struct ApprovalVerdict {
    pub tier: Tier,                    // 1 | 2 | 3
    pub blast_radius: BlastRadius,     // N nodes transitively affected
    pub risk_prior: f64,               // from ix-code::trajectory volatility
    pub verdict: Hexavalent,           // T/P/U/D/F/C
    pub rationale: Vec<Evidence>,      // provenance chain
}

pub enum Tier {
    One,    // Auto-approve (reads, searches, in-project reads)
    Two,    // Auto-approve (in-project edits with git safety net + small blast radius)
    Three,  // Require classifier / escalate
}

pub fn classify_action(
    action: &AgentAction,
    context: &AgentContext,
    code_dag: Option<&ContextBundle>,
) -> ApprovalVerdict;
```

Logic sketch:
- Reads / searches → Tier 1, `T`.
- In-project edit where Context DAG callers-transitive walk stays under `max_blast_radius` nodes → Tier 2, `P`.
- Edit where the target file has high `trajectory::volatility` or recent `discontinuities` → bump one tier up (fragile hot spot).
- Edit that touches a node flagged `Doubtful` or `False` by `ix-code::gates` → Tier 3 mandatory.
- Cross-module / cross-crate edit → Tier 3.
- Out-of-project / shell / web fetch → Tier 3 always.

**Why this is better than auto-mode's classifier alone:** it's deterministic, explainable, unit-testable, and it lives *underneath* the LLM classifier rather than competing with it. Auto-mode can still classify Tier 3 actions; `ix-approval` deflates the 40–60% of actions where blast radius is provably small into Tier 2.

### 4. Session event log (`ix-session`)

The Managed Agents insight: the session log is the primitive; structured state is a projection.

Today IX has:
- `state/beliefs/*.belief.json` — typed belief records
- `state/pdca/*.pdca.json` — PDCA cycles
- `state/knowledge/*.knowledge.json` — knowledge transfers
- `state/snapshots/*.snapshot.json` — reconnaissance snapshots

These are *sources of truth*. The proposal inverts this: the source becomes `state/events/{session-id}.log` (JSONL), and the above typed files become *projections* computed on demand.

```rust
pub struct Session {
    id: SessionId,
    log_path: PathBuf,
}

#[derive(Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,                // monotonic within session
    pub timestamp: DateTime<Utc>,
    pub kind: EventKind,
    pub payload: serde_json::Value,
    pub hexavalent: Hexavalent,
    pub provenance: EventProvenance,
}

pub enum EventKind {
    BeliefAsserted { topic: String },
    BeliefRetracted { topic: String },
    PdcaStep { phase: PdcaPhase },
    ToolInvoked { tool: String },
    ToolResult { tool: String },
    Discovery { scope: String },      // for parallel-agent blackboard
    KnowledgeReceived { source: String },
    Escalation { reason: String },
}

impl Session {
    pub fn emit(&mut self, event: Event) -> Result<()>;
    pub fn events(&self, from: Option<EventId>, to: Option<EventId>) -> impl Iterator<Item = Event>;
    pub fn project_beliefs(&self) -> HashMap<String, BeliefState>;
    pub fn project_pdca(&self) -> Vec<PdcaCycle>;
}
```

Migration strategy: ship `ix-session` with projections first, run both (log + typed files) in parallel for a session, verify projections match typed files bit-exact, then delete the typed files and let projections own the contract.

Bonus: **this is also the Citadel "discovery blackboard" implementation.** Parallel agents emit `Discovery` events into a shared session; subscribers read via `ix-cache` pub/sub on the log's append channel. One primitive, two payoffs.

### 5. Fuzzy logic (`ix-fuzzy`)

The user's 6-valued logic question surfaced a real gap: Demerzel has extensive fuzzy *specs* (`fuzzy-membership.md`, `fuzzy-belief.schema.json`, `fuzzy-distribution.schema.json`, behavioral test cases) but **no Rust implementation**. Demerzel is runtime-free by rule, so the implementation has to live in an IX crate.

Scope:

```rust
// Membership functions
pub trait MembershipFunction {
    fn mu(&self, x: f64) -> f64;  // returns [0, 1]
}
pub struct Triangular { pub a: f64, pub b: f64, pub c: f64 }
pub struct Trapezoidal { pub a: f64, pub b: f64, pub c: f64, pub d: f64 }
pub struct Gaussian { pub mean: f64, pub sigma: f64 }

// Fuzzy inference system (Mamdani)
pub struct FuzzyRule {
    pub antecedents: Vec<(String, Box<dyn MembershipFunction>)>,
    pub consequent: (String, Box<dyn MembershipFunction>),
}

pub struct MamdaniFis {
    pub rules: Vec<FuzzyRule>,
}

impl MamdaniFis {
    pub fn evaluate(&self, inputs: &HashMap<String, f64>) -> HashMap<String, f64>;
}

// Multi-criteria decision making
pub fn f_ahp(pairwise: &FuzzyMatrix) -> Vec<f64>;
pub fn f_topsis(alternatives: &FuzzyMatrix, weights: &[f64]) -> Vec<f64>;
```

Integration with hexavalent: fuzzy outputs a continuous `[0, 1]` confidence; a `bucket()` function maps it to `Hexavalent`:

```rust
pub fn bucket_to_hexavalent(confidence: f64, contradiction: f64) -> Hexavalent {
    if contradiction > 0.5 { return Hexavalent::Contradictory; }
    match confidence {
        c if c >= 0.90 => Hexavalent::True,
        c if c >= 0.70 => Hexavalent::Probable,
        c if c >= 0.50 => Hexavalent::Unknown,
        c if c >= 0.30 => Hexavalent::Doubtful,
        _              => Hexavalent::False,
    }
}
```

The existing `ix-code::gates::verdict_from_delta` pattern is exactly this; we're generalizing it into a reusable crate. Fuzzy and hexavalent are not competitors — they're the continuous and discrete faces of the same uncertainty engine.

### 6. Trace flywheel wiring

IX already has the infrastructure — it's just not wired:
- `ix_trace_ingest` MCP tool exists
- `self-improve-loop` skill exists (export → analysis → pattern promotion → grammar evolution)
- TARS pattern promotion exists via `ix_tars_bridge`
- GA trace export exists via `ix_ga_bridge`

What's missing: **`ix-agent` doesn't emit traces during normal operation.** Every tool invocation should write an event to the session log (→ primitive #4) which a background reader streams to `ix_trace_ingest`. That feeds the analysis pipeline. Pattern-promotion suggestions come back through the federation as directives that `ix-approval` can gate.

This closes the loop: **every session trains the next session's harness**. LangChain's Trace Analyzer insight, made concrete with IX's existing federation.

Cost is the smallest of all the primitives (~0.5 day) because everything it needs is already built.

### A. Hexavalent migration

Discovery this session: two truth types coexist in the workspace.

- `ix_types::Hexavalent` (6 values, shipped, used by `ix-code::gates`, `ix-skill`, new code) ✅
- `ix_governance::TruthValue` (4 values, legacy, used by karnaugh/remediation_optimizer/violation_pattern/weight_evolution) ⚠️

Plan:
1. Add hexavalent `and`/`or`/`not`/`implies`/`xor`/`equiv` to `ix_types::Hexavalent` matching the API of the legacy `TruthValue` (look up truth tables in `governance/demerzel/logic/hexavalent-logic.md`)
2. Update `ix-governance::karnaugh`, `remediation_optimizer`, `violation_pattern`, `weight_evolution` to consume `Hexavalent` — each is a focused edit
3. Rename `crates/ix-governance/src/tetravalent.rs` → `logic.rs`
4. Update `crates/ix-governance/src/lib.rs` docstring (*"four-valued logic"* → *"hexavalent logic"*)
5. Delete the `tetravalent` skill redirect if there is one, or mark it deprecated
6. Update `docs/solutions/feature-implementations/karnaugh-tetravalent-logic.md` with a hexavalent addendum (append-only, don't rewrite history)

This is cleanup work, no new features. It unblocks #3 (approval) from inconsistent truth types and makes the Context DAG's hexavalent labels consistent across the workspace.

### B. Ambiguous call-metadata fix

Tiny prerequisite for #1. In `crates/ix-code/src/semantic.rs`:

```rust
// Current
pub struct CallEdge {
    pub caller: String,
    pub callee: String,                // bare name, e.g. "baz"
    pub call_site_line: usize,
    pub weight: u32,
}

// Proposed
pub struct CallEdge {
    pub caller: String,
    pub callee_hint: CalleeHint,       // scoped path or bare name
    pub call_site_line: usize,
    pub weight: u32,
}

pub enum CalleeHint {
    Bare(String),                      // `baz`
    Scoped(Vec<String>),               // ["foo", "bar", "baz"]
    MethodCall { receiver_hint: Option<String>, method: String },
}
```

Half a day of work, including tests. Ship this before `ix-context` and Codex's two-pass resolver becomes implementable.

---

## Proposed construction order

```
           Now                                       Month later
            │                                             │
   ┌────────┴────────┐                                    │
   │  B. CallEdge    │  (prerequisite, <0.5d)             │
   │  metadata fix   │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │  1. Context DAG │  (MVP, 1-2d) ──── ship as ix-context
   │   ix-context    │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ 2. Middleware   │  (1-2d) ──── three middlewares     │
   │  ix-middleware  │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ A. Hexavalent   │  (1d) ──── cleanup, unblocks #3    │
   │    migration    │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ 3. Approval     │  (1d) ──── auto-mode integration   │
   │   ix-approval   │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ 4. Session log  │  (2-3d) ──── Managed Agents pattern
   │   ix-session    │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ 5. Fuzzy logic  │  (2-3d) ──── closes Demerzel gap   │
   │    ix-fuzzy     │                                    │
   └────────┬────────┘                                    │
            │                                             │
   ┌────────┴────────┐                                    │
   │ 6. Trace        │  (0.5d) ──── wires the flywheel    │
   │   flywheel      │                                    │
   └─────────────────┘                                    │
```

**Total: ~10-13 days of focused work. Individual ships: 0.5-3 days each.** Safe to stop after any step — nothing later is a hard dependency of anything earlier except #3 depending on A and 1.

---

## What this buys IX

Taken together, the six primitives + migrations make IX something new in the ecosystem: **a structural, deterministic, hexavalent oracle that agent harnesses can consume as a backend**. Not a competing harness, not a framework — a *backend* for any harness that wants to:

- Pre-load context from code structure rather than guessing via embeddings (→ Context DAG)
- Hook every tool call with deterministic pre/post policies (→ Middleware)
- Classify action risk from blast radius and history volatility (→ Approval)
- Treat session history as an append-only event log (→ Session)
- Reason about uncertainty continuously and discretely (→ Fuzzy + Hexavalent)
- Learn from its own traces (→ Flywheel)

This reframes IX from "Rust ML library + governance artifacts" to "structural oracle for the agent era." The math primitives shipped over the last months stop looking random and start looking like *the bottom of a stack that didn't exist yet*.

Claude's pattern-spotter observation from the Context DAG brainstorm applies recursively: **"compose capabilities you don't yet need, and the future arrives as a dependency injection."** Unix pipes, LLVM IR, Plan 9's 9P. The math came first; the use case arrived as five articles in one session.

---

## What could kill this direction

Honest risks:

1. **Scope creep.** Six primitives + migrations is a lot. Discipline: ship #1 in isolation, evaluate, decide whether to continue. Don't build all six speculatively.
2. **Demerzel consumer contract.** If `ix-governance::TruthValue` is depended on externally (TARS, GA), the migration breaks consumers. Mitigation: check federation dependencies before the migration, deprecate with a compatibility shim for one release.
3. **Walker correctness edge cases.** Two-pass resolver will be wrong on trait dispatch, macros, re-exports. Mitigation already baked in: preserve ambiguity as signal, don't paper over it.
4. **Middleware overhead.** If every MCP tool call goes through a middleware chain, latency compounds. Mitigation: middleware is opt-in per tool, default chain is empty, hot path stays fast.
5. **Session log size.** Append-only logs grow. Mitigation: rotation at configurable size, compaction to projections, old logs live in `state/events/archive/`.
6. **Fuzzy vs hexavalent confusion.** Two uncertainty engines in the same workspace could be misused. Mitigation: `ix-fuzzy` is the *computation* layer; `Hexavalent` is the *contract* layer. Crates consuming decisions import `Hexavalent`; crates computing them import `ix-fuzzy`. Clean boundary.
7. **"Structural oracle" is a narrative, not a product.** The danger of a meta-doc like this is treating the story as the deliverable. Mitigation: this doc exists only to contextualize the Context DAG ship and to prevent insight loss. If any of #2-#6 don't get built, the project is still fine — they're options, not commitments.

---

## Not doing (yet)

Ideas surfaced in the brainstorm that are deliberately *not* on the roadmap:

- **Federated cross-language Context DAG** (Gemini) — needs TARS/GA instrumentation first
- **Hybrid RAG+DAG with embedding landing zones** (Gemini) — violates purism; revisit only if cold-start is unacceptable
- **Hyperbolic / Laplacian spectral walks** (Codex rank 7-8) — unjustified before basic walks prove insufficient
- **Walk-as-MCP-reasoning-step** (Claude) — wait for v1 API stability
- **Borrow-checker lifetime walks** (Gemini) — rust-analyzer territory
- **Feature-flag shadow walks** (Gemini) — requires cfg evaluation
- **Cloud-hosted IX** — this is local tooling by design, not a service
- **ix-agent as fully stateless with wake(sessionId)** — Managed Agents pattern, but overkill for a local MCP server

---

## Appendix: article citations read in this session

1. https://www.anthropic.com/engineering/harness-design-long-running-apps
2. https://www.anthropic.com/engineering/managed-agents
3. https://www.anthropic.com/engineering/claude-code-auto-mode
4. https://blog.langchain.com/improving-deep-agents-with-harness-engineering/
5. https://openai.com/index/harness-engineering/ *(fetch failed 403 — content inferred from Swarm/Codex public material)*
6. https://github.com/SethGammon/Citadel
7. https://www.reddit.com/r/ClaudeAI/comments/1s3wt3n/rag_is_a_trap_for_claude_code_i_built_a_dagbased/ *(fetch blocked — argument inferred from title + context)*
8. https://platform.claude.com/cookbook/

Plus the user-pasted 6-valued and fuzzy logic reference material that surfaced the `ix-fuzzy` gap.

---

## Handoff

This doc does **not** get handed to `/octo:develop`. Only the companion Context DAG doc is implementation-ready. This meta-doc is:
- A reference for why the Context DAG MVP is shaped the way it is
- A strategic map for follow-up sessions
- An insurance policy against insight loss: if we stop after Context DAG, the other five ideas are on paper, not in memory

Review this doc before starting each subsequent primitive to avoid divergent designs. Update the construction-order diagram when ships change reality.
