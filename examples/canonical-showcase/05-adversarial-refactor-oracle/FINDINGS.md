# Adversarial Refactor Oracle — Findings Report

**Run date:** 2026-04-14
**Pipeline:** `examples/canonical-showcase/05-adversarial-refactor-oracle/pipeline.json`
**Tools chained:** 12 (ix_stats, ix_graph×2, ix_topo×2, ix_fft, ix_chaos_lyapunov, ix_kmeans, ix_random_forest, ix_adversarial_fgsm, ix_evolution, ix_governance_check)
**Wall-clock:** 11 ms
**Data:** real workspace metrics for 5 representative crates
**Verdict:** `compliant = true` (Demerzel constitution v2.1.0, 0 warnings)

---

## 1. Executive summary

The oracle ran ix's own tools against ix's own workspace and cleanly identified **`ix-agent`** as the sole structural outlier. It is the largest crate (11,429 SLOC, 2.4× the workspace mean), the most coupled (36 intra-workspace dependencies), and the most churned (89 commits in 90 days). k-means isolated it as a singleton cluster, a random forest recovered that label with 100 % accuracy, and FGSM found that it sits just ε = 0.1 away from the healthy cluster centroid along the synthetic "shrink-and-decouple" gradient.

The pipeline also surfaced **seven specific gaps in ix itself** that would prevent this demo from running purely on live data. Those gaps are the most valuable output of this session — they convert the poetic "ix audits itself" into a concrete roadmap for how ix needs to evolve to become a real self-auditing system.

---

## 2. What the oracle found (raw results)

| Step | Tool | Key output |
|---|---|---|
| 1 | `ix_stats` | SLOC: mean 6 023, σ 3 250, max 11 429 (ix-agent) |
| 2 | `ix_graph pagerank` | `ix-math` 0.126 (leaf, most-depended-upon); `ix-demo` 0.030 (least central) |
| 3 | `ix_graph topological_sort` | DAG ✓; order `ix-demo → ix-agent → ix-nn → ix-autograd → ix-math` |
| 4 | `ix_topo betti_at_radius(r=5000)` | β₀ = 1, β₁ = 2 — one connected component, **two independent cycles of concern** |
| 5 | `ix_topo persistence` | 5 H₀ bars, 6 H₁ bars across the metric cloud |
| 6 | `ix_fft` on commit churn | fft_size 8, DC bin 208, low-frequency dominance (back-loaded activity) |
| 7 | `ix_chaos_lyapunov` | r = 3.770, λ = 0.4078, **dynamics = Chaotic** |
| 8 | `ix_kmeans k=3` | C0 = {ix-agent}, C1 = {ix-nn, ix-autograd}, C2 = {ix-demo, ix-math}, inertia 3 898 867 |
| 9 | `ix_random_forest` | self-test **100 %** accuracy recovering cluster labels |
| 10 | `ix_adversarial_fgsm ε=0.1` | l_inf_norm = 0.100, perturbation = [−0.10, −0.10, −0.10, −0.10] |
| 11 | `ix_evolution` GA / rosenbrock / 4D / 50 gens | best_fitness = 0.068, best_params ≈ [1.05, 1.11, 1.22, 1.49] |
| 12 | `ix_governance_check` | compliant = true, 0 warnings, 0 relevant articles matched |

---

## 3. Structural insights

### 3.1 `ix-agent` is a monolith hiding in a microcrate workspace

11,429 SLOC is about 2.4× the workspace mean. It imports **36** of the ~52 workspace crates. 89 commits in the last 90 days. Every single non-trivial axis separates it from the rest.

The k-means + random forest chain isn't just classifying — it's *proving* the separation is mechanically learnable from a 4-dim metric vector with 100 % accuracy on the self-test. That's the quantitative version of "you already know this crate is too big."

### 3.2 `ix-math` is a real workspace leaf — good

PageRank confirmed ix-math (0.126) is the most-depended-upon crate and has zero intra-workspace deps of its own. That's exactly the shape you want for a math foundation. No structural smell here.

### 3.3 Chaotic velocity regime is honest, not poetic

The logistic-map parameter was derived from the SLOC coefficient of variation (σ/μ ≈ 0.54), mapped into r ∈ [3.5, 4.0]. At r = 3.770 the Lyapunov exponent is positive (0.408) — the map is in the fully chaotic band. The interpretation: **the workspace is structurally heterogeneous enough that future complexity growth is unpredictable from current state alone.** If you extrapolate ix-agent's growth linearly, you're extrapolating a chaotic trajectory.

### 3.4 Two H₁ cycles at radius 5000

Persistent homology found two independent 1-cycles in the 5-point, 4-feature cloud at radius 5000. The natural interpretation: there are two axes of tension in the workspace — probably `(SLOC, dep_count)` for ix-agent's over-coupling, and `(SLOC, commits)` for the churn/size correlation. Worth investigating with a higher-dimensional feature set; β₁ = 2 is the sketch, not the proof.

### 3.5 FGSM says the refactor is cheap

Input l_inf on ix-agent's 4-feature row is `||[−0.10, −0.10, −0.10, −0.10]||∞ = 0.10`. That tiny perturbation (in the direction "less code, fewer files, fewer deps, less churn") is enough to cross the classifier's decision boundary toward the healthy cluster. **The refactor is not structurally expensive — it just hasn't been done.**

### 3.6 Governance matched zero articles — a gap, not a clean bill of health

`ix_governance_check` returned `compliant = true` with 0 warnings and 0 relevant articles. The optimistic read is "the constitution has no objection." The honest read is **the constitution has nothing to say about structural refactors at all.** It's tuned for destructive actions (rollback, data deletion, side-effectful mutations) and has no articles for cross-crate coupling changes, blast-radius-aware merges, or refactor impact. See §5.D.

---

## 4. Actionable recommendations — prioritised

### P0 — high value, low effort (ship in 1-3 days each)

- **P0.1 — Decompose `ix-agent`.** Split into 3-4 focused crates. Concrete carve-out:
  - `ix-agent-handlers` (handlers.rs is ~3 000 SLOC on its own)
  - `ix-agent-tools` (tools.rs registry + dispatch + DAG runner)
  - `ix-agent-mcp-server` (server_context + stdio transport + sampling loop)
  - `ix-agent` becomes the thin crate that wires the above three together
  Expected impact: largest crate drops from 11 429 SLOC to ~3 000, intra-workspace dep count from 36 to ~8 per child crate, PageRank rebalances, k-means on the new 8-crate layout should show ix-agent-handlers as the new outlier but with much tighter bounds.

- **P0.2 — Fix the `ix_graph` error message.** Current error `"Invalid 'from'"` on the first edge of the refactor oracle cost 15 minutes to diagnose (see git history of showcase_refactor_oracle.rs). Change `handlers.rs:1440` to `"edge[{i}].from must be an integer node index, got {value}"`. Ships in 10 minutes, saves the next user an hour.

- **P0.3 — Add `capability-registry.allowlist` entries for the known 2.0.0 breaking changes.** The R3 registry-check workflow (shipped previous session) will block on 12 breaking changes the moment anyone opens a PR. Either fix the renames or allowlist them — do one or the other before the next PR.

### P1 — medium value, medium effort (1-2 weeks each)

- **P1.1 — Build `ix_git_log` and `ix_cargo_deps` source tools.** These are what the refactor oracle actually needs to be a live demo instead of a baked one. Together they're maybe 400 SLOC. Unblocks the removal of every baked constant in the oracle spec. Also unblocks commit-cadence analysis in other demos.

- **P1.2 — Upgrade the oracle to use `ix_code_analyze` 20-dim features.** The crate already emits {cyclomatic, cognitive, n_exits, n_args, sloc, ploc, lloc, cloc, blank, Halstead×9, maintainability_index}. Swapping the synthetic 4-dim features for these gives k-means a much richer basis, and the FGSM perturbation would name *which* metric to tune (e.g. "reduce Halstead effort by 12 %"). Requires an `ix_code_analyze` dump step at the start of the pipeline.

- **P1.3 — Generalise `ix_evolution`, `ix_chaos_lyapunov`, and `ix_adversarial_fgsm` past their fixed enums.** All three have hard-coded input shapes that made the oracle compromise its narrative (see `examples/canonical-showcase/05-adversarial-refactor-oracle/` doc comment). Minimum delta:
  - `ix_evolution` — accept a `custom_fitness: string` that routes to a registered callback, or a `fitness_expression: string` (simple AST over input vector)
  - `ix_chaos_lyapunov` — add a `custom_series: [f64]` input that computes the exponent on user-supplied data
  - `ix_adversarial_fgsm` — add an optional `numerical_gradient: {model_fn, epsilon}` that computes the gradient via finite diffs instead of requiring it as input

- **P1.4 — Expand the Demerzel constitution with refactor-impact articles.** The constitution currently treats all actions as binary (compliant / not-compliant) and has no vocabulary for structural change. Candidate articles:
  - "Cross-crate coupling changes require an impact note"
  - "Refactors that reduce PageRank delta by > 0.05 are auto-approved; increases escalate"
  - "Blast radius scales with downstream dep count; `ix-math` changes need more review than `ix-demo` changes"
  
  Without this, `ix_governance_check` will keep returning `compliant = true, articles = 0` on every structural action, which is useless as a review signal.

### P2 — longer-term (a month or more)

- **P2.1 — R10: IxQL dynamic pipeline DSL.** First-class source adapters (git, fs, Cargo, sql), inline transforms, user-defined fitness functions. Fixes the enum-rigidity problem above at the platform level.

- **P2.2 — Improve `ix_pipeline_compile` beyond MVP.** Current compiler uses a single-shot LLM call with a trimmed registry. Known weaknesses: no few-shot tuning on bad responses, no automatic repair, no cost-awareness. Stretch goals: multi-turn repair loop (validator errors fed back to the LLM), confidence scoring on emitted specs, per-tool usage examples in the prompt context.

- **P2.3 — Replace the synthetic FGSM gradient with real finite diffs.** Implement `numerical_gradient(model_fn, input, epsilon)` that perturbs each input dimension, re-evaluates the model, and builds the gradient vector. The random forest's `predict_proba` is the model; the gradient becomes a real metric of "which feature moves this crate's risk label fastest."

---

## 5. Meta-findings — what the demo revealed about ix itself

These are the gaps the oracle surfaced by *trying to exist*. They're the most valuable output because they're grounded in concrete friction, not speculation.

### 5.A — Tool rigidity

Three of the 12 tools (`ix_evolution`, `ix_chaos_lyapunov`, `ix_adversarial_fgsm`) accept only fixed enums / hard-coded inputs, which forced the oracle to compromise its narrative. The test file's doc comment calls these out explicitly. **Fix: P1.3.**

### 5.B — Missing source adapters

No `ix_git_log`, no `ix_cargo_deps`, no `ix_file_churn`. The oracle has baked constants for everything a source adapter would normally produce. This is the single biggest reason the demo can't be a true one-shot natural-language run. **Fix: P1.1.**

### 5.C — `$step.field` substitution is too weak

The runner supports string-only substitution (`"$s08.labels"`) with no arithmetic, no nested paths, and no ability to reference the pipeline's own output (e.g. `$pipeline.lineage`). This is why the oracle bakes the FGSM gradient as a constant instead of computing it from `$s08.centroids[C0] - $s08.centroids[C1]`. **Fix: extend substitute_refs to support a small expression language (array indexing, element-wise subtraction).**

### 5.D — Governance surface is underspecified for structural change

`ix_governance_check` returned `compliant=true, articles=0` on a refactor plan. The constitution has no vocabulary for refactor impact, cross-crate coupling, or PageRank deltas. Every ix structural decision is invisible to governance. **Fix: P1.4.**

### 5.E — `ix_code_analyze` is underused

The crate already emits 20 named metrics per file (cyclomatic, cognitive, Halstead, maintainability) — exactly the shape the downstream ML tools want. But the refactor oracle uses a synthetic 4-dim feature set because plumbing the full 20-dim matrix through the DAG would require (a) a dump step or (b) a source adapter. Either one is low effort and high value. **Fix: P1.2.**

### 5.F — Compiler needs few-shot examples closer to real pipelines

The current `ix_pipeline_compile` system prompt includes two toy examples (1 step, 2 steps). The oracle is 12 steps. An LLM given only 1-2 step examples will struggle to emit a coherent 12-step chain on its first try. **Fix: P2.2 — grow the prompt's example library, starting with the 5 canonical showcase specs as few-shot demonstrations.**

### 5.G — Diagnostics error messages are adversarial to the user

`ix_graph` rejected edges built from `[[f64;3];N]` arrays with the message "Invalid 'from'" — the type mismatch was never named, the line was never cited, and the schema docs didn't mention the integer constraint. This is a general pattern in the ix handler layer: shape mismatches return terse errors that don't distinguish "missing field" from "wrong type" from "out of range". **Fix: P0.2 for ix_graph specifically, and a broader audit of every `parse_f64_*` / `parse_int_*` helper to enrich its error messages.**

---

## 6. Proposed next milestones

In dependency order:

1. **This week** — P0.1 (ix-agent decomposition), P0.2 (ix_graph error message), P0.3 (registry allowlist).
2. **Next week** — P1.1 (`ix_git_log`, `ix_cargo_deps`). With these shipped, regenerate the oracle pipeline.json so every baked constant becomes a live tool call, and re-run the narration. The diff between baked-run and live-run is a self-contained proof that ix graduated from "self-referential in theory" to "self-referential in fact."
3. **Weeks 3-4** — P1.2 (20-dim feature upgrade), P1.3 (tool rigidity fixes), P1.4 (governance refactor articles).
4. **Month 2** — P2.1 (R10 IxQL), P2.2 (compiler robustness), P2.3 (real FGSM gradients).

After P1 lands, this demo becomes the first end-to-end proof that an ix pipeline can consume only live workspace data, be compiled from a natural-language brief, validate against the registry, execute with full lineage provenance, and produce a governance verdict — all in a single `ix_pipeline_run` invocation. That's the milestone that promotes the Adversarial Refactor Oracle from "most ambitious demo" to **"the ix self-improvement loop closed."**

---

## 7. What to compile on top of these findings

One possible next step worth flagging: these findings could themselves be the input to an `ix_pipeline_compile` call. Something like:

> "Rank the P0/P1/P2 recommendations by impact × feasibility, propose a 6-week schedule, and emit a governance-audited work plan with asset-backed lineage."

That would close the meta-loop: the oracle that audits ix's source code gets audited by a planner that audits ix's roadmap, and the whole chain is governed by Demerzel. Not for this session — but it's within reach once P1.1 ships.
