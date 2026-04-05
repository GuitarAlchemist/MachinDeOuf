# Advanced demos — real-world problems

Nine pipelines solving problems you'd actually ship + a scheduled
monitoring loop. Each fuses 3-5 signals into a governance-gated decision.

```bash
bash examples/showcase/advanced/run-all.sh
```

Or run any one individually:

```bash
cp examples/showcase/advanced/fraud-detection.yaml /tmp/ix.yaml
cd /tmp && IX_ROOT="$OLDPWD" ix pipeline run
```

---

## 1. `fraud-detection.yaml` — e-commerce transaction screening

**The problem**: a payment API has milliseconds to decide whether to
let a charge through or freeze the account. False negatives cost money;
false positives lose customers.

**The approach**: four independent signals run in parallel, a
constitutional gate fuses them into a block/allow decision with an
audit trail.

| Stage | Skill | Signal |
|---|---|---|
| `amount_profile` | `stats` | Mean, std_dev, max/mean ratio of recent transactions |
| `behavior_clusters` | `kmeans` | K=3 clustering of (recency, amount_tier) for this merchant |
| `watchlist_check` | `bloom_filter` | Is the account_id on our known-bad list? |
| `ip_velocity` | `hyperloglog` | Estimated unique IPs touching this account |
| `fraud_decision` | `governance.check` | Article 1/4/6 audit of block-and-freeze proposal |

The Bloom filter uses 0.01 false-positive-rate; HyperLogLog at
precision=10 gives ~0.8% cardinality error — both ideal for hot-path
screening.

---

## 2. `bandit-ab-test.yaml` — multi-armed A/B test rollout

**The problem**: you have 4 UI variants, don't know which is best,
and need to roll out the winner without blowing traffic on losers.

**The approach**: Thompson-sampling bandit for 1000 rounds discovers
the posterior; stats + game-theoretic equilibrium sanity-check the
choice; governance gate approves the final deployment.

| Stage | Skill | Role |
|---|---|---|
| `explore_arms` | `bandit` (Thompson) | 4 arms × 1000 rounds, CTRs 0.08-0.18 |
| `arm_summary` | `stats` | Descriptive stats on observed arm rewards |
| `game_equilibrium` | `game.nash` | Stable equilibrium if arms are strategic |
| `deployment_review` | `governance.check` | Constitutional audit of 100% rollout |

Thompson sampling converges faster than ε-greedy or UCB1 on
well-separated arms — at 1000 rounds the posterior on arm #4
(the 0.18 CTR winner) should dominate.

---

## 3. `sensor-anomaly.yaml` — industrial sensor alerting

**The problem**: monitoring a 16-sample sliding window from an
industrial sensor. Need to decide whether to page on-call at 3am
based on multi-signal convergence — not just one red light.

**The approach**: spectrum + chaos regime + baseline stats + known
anomaly fingerprints, all running on the same window, fused by a
constitutional gate that weighs "reversibility × proportionality".

| Stage | Skill | Signal |
|---|---|---|
| `spectrum` | `fft` | Frequency decomposition reveals drift sidebands |
| `chaos_probe` | `chaos.lyapunov` | Logistic map at r=3.57 (edge of chaos) → λ sign |
| `baseline_stats` | `stats` | Mean/std_dev excursion from prior window |
| `known_anomaly_patterns` | `bloom_filter` | O(1) match against incident catalog |
| `alert_decision` | `governance.check` | Article 3 + 6 audit of "page SRE now?" |

Setting `parameter: 3.57` is deliberate — that's the period-doubling
accumulation point for the logistic map, where the dynamics transition
from periodic to chaotic. A positive Lyapunov exponent here confirms
we've left the stable regime.

---

## 4. `chained-spectrum.yaml` — cross-stage data flow via `{from:}`

**The problem**: spectral analyses where you want to compute stats *on*
the FFT output (not the raw signal). Demands genuine cross-stage dataflow.

**The approach**: `stats(data: {from: spectrum.magnitudes})` — the
lowering pass extracts the `magnitudes` field from the upstream FFT
result and feeds it as the downstream stats input at runtime.

| Stage | Skill | What it sees |
|---|---|---|
| `time_domain` | `stats` | Raw signal (16 samples, mean=0.625, std=1.28) |
| `spectrum` | `fft` | Raw signal → {frequencies, magnitudes} |
| `spectrum_stats` | `stats` | **Receives `spectrum.magnitudes` via `{from:}`** (mean=3.55, max=13.04) |
| `flatness_review` | `governance.check` | Verdict on spectral flatness |

The ratio `spectrum_stats.std_dev / spectrum_stats.mean` is the
spectral flatness coefficient — flat ≈ broadband noise, spiky ≈
narrowband resonance.

---

## 5. `music-theory.yaml` — GA cross-repo federation

**The problem**: build a harmonic substitution recommender. Need chord
feature vectors that ML clustering can consume.

**The approach**: federate to GuitarAlchemist (`ga_bridge` skill)
which emits pitch-class feature specs for chords / scales /
progressions. ix's clustering consumes those downstream.

| Stage | Skill | Music theory input |
|---|---|---|
| `ii_V_I_features` | `ga_bridge` | Dm7 – G7 – Cmaj7 (canonical jazz turnaround) |
| `scale_features` | `ga_bridge` | C major scale → 12-bit pitch-class vector |
| `progression_feats` | `ga_bridge` | "Cm7 F7 BbMaj7 EbMaj7" (Autumn Leaves, first 4 bars) |
| `workflow_audit` | `governance.check` | Federation-call audit |

Demonstrates the cross-repo story: GA owns music theory, ix owns ML,
both speak JSON through the MCP registry. No Rust coupling required.

---

## 6. `code-quality.yaml` — static-analysis PR triage

**The problem**: CI must decide whether a PR's complexity metrics
(cyclomatic, cognitive, Halstead, SLOC, maintainability index) trigger
a blocker.

**The approach**: analyze three candidate functions in parallel, gate
on the spread between simplest and most complex.

| Stage | Skill | Function analyzed |
|---|---|---|
| `simple_function` | `code_analyze` | `add(a,b) -> a+b` (baseline) |
| `complex_branching` | `code_analyze` | 7-way if/return chain |
| `deep_nesting` | `code_analyze` | 3-level nested loops with conditionals |
| `pr_merge_review` | `governance.check` | Article 4 (Proportionality) + 6 (Escalation) |

If the deep-nesting function's cyclomatic complexity >> baseline's,
the gate should block. No ML required — pure static analysis.

---

---

## 7. `graph-centrality.yaml` — service mesh routing

**The problem**: 6-node service mesh. Which service is most central?
What's the minimum-latency path from API gateway to database? What's
the valid deployment bring-up order?

| Stage | Skill | Result |
|---|---|---|
| `centrality` | `graph` (pagerank) | Ranks nodes by incoming influence |
| `request_path` | `graph` (dijkstra) | Min-latency path from gateway → db |
| `deploy_order` | `graph` (topological_sort) | Valid service startup sequence |
| `deployment_plan_review` | `governance.check` | Reviews the rollout plan |

---

## 8. `topology-circle.yaml` — persistent homology detects the hole

**The problem**: shape classification from point clouds. Given 8 points
on a unit circle, persistent homology should detect one long-lived
1-dimensional hole (β₁=1) plus the single connected component (β₀=1)
once the filtration radius grows.

| Stage | Skill | Output |
|---|---|---|
| `persistence` | `topo` | Birth-death pairs for β₀ and β₁ |
| `betti_curve` | `topo` | β numbers as a function of radius |
| `betti_at_radius` | `topo` | Exact Betti numbers at radius=0.8 |
| `shape_audit` | `governance.check` | Classifier verdict review |

---

## 9. `grammar-evolution.yaml` — replicator dynamics + Bayesian + MCTS

**The problem**: three candidate code-gen grammar rules compete for
"recommended" status. Need to triangulate with three different methods
before promotion.

| Stage | Skill | Method |
|---|---|---|
| `rule_competition` | `grammar.evolve` | Replicator dynamics × 200 steps |
| `weight_update` | `grammar.weights` | Bayesian Beta-Binomial posterior bump |
| `grammar_search` | `grammar.search` | MCTS derivation search |
| `rule_promotion_review` | `governance.check` | Triangulated promotion gate |

---

## Scheduled monitoring loop — `monitoring-loop.sh`

```bash
INTERVAL=2 MAX_RUNS=5 bash examples/showcase/advanced/monitoring-loop.sh
```

Runs the `sensor-anomaly.yaml` pipeline every N seconds, updates the
`sensor_health` hexavalent belief (T/P/U/D/F/C + confidence), and
captures a snapshot whenever the verdict flips. A watchdog-style
pattern for long-running monitoring jobs.

---

## Design principles these share

1. **Parallel-roots-to-governance-leaf** — independent analyses run
   concurrently, fused into a single constitutional decision point.
2. **Mixed skill domains** — each pipeline touches ≥4 different crates
   (stats / probabilistic / RL / game / chaos / governance).
3. **Every decision is auditable** — the `governance.check` leaf's
   output logs the action text + matched articles + verdict, ready
   for compliance replay.
4. **Runnable in <20ms** — none of these need a GPU; the registry's
   pure-Rust skill impls run instantly on CPU.

## Extending them

- **Chain pipelines**: write a new `ix.yaml` where one stage's output
  feeds the next stage's input via `{"from": "prior_stage.field"}`.
- **Swap skills**: `gradient_boosting` in place of `random_forest`,
  `ucb1` in place of `thompson`, `number_theory` gcd in place of a
  primality test — the governance leaf adapts automatically.
- **Add belief persistence**: follow each run with
  `ix beliefs set model_robustness P --confidence 0.85` to record
  audit state for the next cycle.
