# ix

[![CI](https://github.com/GuitarAlchemist/ix/actions/workflows/ci.yml/badge.svg)](https://github.com/GuitarAlchemist/ix/actions/workflows/ci.yml)

A Rust workspace of composable ML/math algorithms and AI governance, designed to be exposed as **Claude Code skills** via an MCP server and CLI. Part of the [GuitarAlchemist](https://github.com/GuitarAlchemist) ecosystem (ix + [tars](https://github.com/GuitarAlchemist/tars) + [ga](https://github.com/GuitarAlchemist/ga) + [Demerzel](https://github.com/GuitarAlchemist/Demerzel)).

32 crates. 37 MCP tools. 80+ Claude Code skills. Pure Rust. No external ML frameworks.

## Quick Start

```bash
# Build everything
cargo build --workspace

# Run tests
cargo test --workspace

# Run the CLI
cargo run -p ix-skill -- optimize --algo pso --function sphere --dim 10

# Start the MCP server (for Claude Code integration)
cargo run -p ix-agent
```

## Crates

### Core Math & Optimization
| Crate | Description |
|-------|-------------|
| **ix-math** | Linear algebra, statistics, distances, activations, calculus, random, hyperbolic geometry |
| **ix-optimize** | SGD, Adam, simulated annealing, particle swarm optimization |

### Supervised Learning
| Crate | Description |
|-------|-------------|
| **ix-supervised** | Linear/logistic regression, decision trees, KNN, Naive Bayes, SVM, metrics (confusion matrix, ROC/AUC), cross-validation, SMOTE, TF-IDF |
| **ix-ensemble** | Random forest, gradient boosted trees |

### Unsupervised Learning
| Crate | Description |
|-------|-------------|
| **ix-unsupervised** | K-Means, DBSCAN, PCA, t-SNE (Barnes-Hut), GMM (EM algorithm) |

### Deep Learning & RL
| Crate | Description |
|-------|-------------|
| **ix-nn** | Neural network layers (Dense, LayerNorm, BatchNorm, Dropout), loss functions, backprop, transformers |
| **ix-rl** | Multi-armed bandits (epsilon-greedy, UCB1, Thompson), Q-learning, GridWorld |
| **ix-evolution** | Genetic algorithms, differential evolution |

### Search & Graphs
| Crate | Description |
|-------|-------------|
| **ix-search** | A\*, Q\* (learned heuristic), MCTS, minimax/alpha-beta, BFS/DFS, hill climbing, tabu search |
| **ix-graph** | Graph algorithms, Markov chains, HMM/Viterbi/Baum-Welch, state spaces, agent routing |
| **ix-game** | Nash equilibria, Shapley value, auctions, evolutionary dynamics, mean field games |

### Signal & Chaos
| Crate | Description |
|-------|-------------|
| **ix-signal** | FFT, wavelets, FIR/IIR filters, Kalman filter, spectral analysis, DCT |
| **ix-chaos** | Lyapunov exponents, bifurcation diagrams, strange attractors, fractal dimensions, delay embedding, chaos control |

### Security & Privacy
| Crate | Description |
|-------|-------------|
| **ix-adversarial** | FGSM, PGD, C&W attacks, adversarial training, poisoning detection, differential privacy, robustness evaluation |

### Advanced Math
| Crate | Description |
|-------|-------------|
| **ix-dynamics** | Inverse kinematics chains, Lie groups/algebras (SO(3), SE(3)), neural ODEs |
| **ix-topo** | Persistent homology, simplicial complexes, Betti numbers, topological data analysis |
| **ix-ktheory** | Graph K-theory, Grothendieck K0/K1, Mayer-Vietoris sequences |
| **ix-category** | Functors, natural transformations, monads, category theory primitives |
| **ix-grammar** | Formal grammars: context-free grammars, Earley parser, CYK, Chomsky normal form |
| **ix-rotation** | Quaternions, SLERP, Euler angles, axis-angle, rotation matrices, Plücker coordinates |
| **ix-sedenion** | Hypercomplex algebra: sedenions, octonions, Cayley-Dickson construction, BSP trees |
| **ix-fractal** | Takagi curves, IFS (Sierpinski, fern), L-systems, Hilbert/Peano/Morton space-filling curves |
| **ix-number-theory** | Prime sieving, Miller-Rabin, modular arithmetic, CRT, elliptic curves |

### Infrastructure
| Crate | Description |
|-------|-------------|
| **ix-gpu** | WGPU compute shaders for cosine similarity, matrix multiply, batch vector search (Vulkan/DX12/Metal) |
| **ix-cache** | Embedded Redis-like cache with sharded concurrency, TTL, LRU eviction, pub/sub, RESP protocol server |
| **ix-pipeline** | DAG executor with topological sort, parallel branch execution, memoization, critical path analysis |
| **ix-probabilistic** | Bloom filter, Count-Min sketch, HyperLogLog, Cuckoo filter |
| **ix-io** | CSV, JSON, file watcher, named pipes, TCP, HTTP, WebSocket, trace bridge |

### Governance
| Crate | Description |
|-------|-------------|
| **ix-governance** | Demerzel governance: tetravalent logic (T/F/U/C), constitution parser, 12 persona loader, policy engine |

### Integration
| Crate | Description |
|-------|-------------|
| **ix-agent** | MCP server: 37 tools via JSON-RPC over stdio (algorithms + governance + federation) |
| **ix-skill** | CLI binary for direct command-line access to all algorithms |
| **ix-demo** | egui desktop app with 22+ interactive demo tabs including governance explorer |

## Claude Code Integration

### MCP Server

Register ix as a Claude Code MCP server in `.mcp.json`:

```json
{
  "mcpServers": {
    "ix": {
      "command": "cargo",
      "args": ["run", "-p", "ix-agent"]
    }
  }
}
```

Claude can then call tools like `ix_kmeans`, `ix_viterbi`, `ix_optimize`, etc. directly during conversations.

### Skills

80+ Claude Code skills organized by domain:

**Algorithm skills** (26): ix-optimize, ix-cluster, ix-search, ix-chaos, ix-hmm, ix-adversarial, ix-game, ix-pipeline, ix-signal, ix-benchmark, ix-nn, ix-bandit, ix-evolution, ix-random-forest, ix-supervised, ix-topo, ix-category, ix-dynamics, ix-ktheory, ix-gpu, ix-cache, ix-grammar, ix-rotation, ix-sedenion, ix-fractal, ix-number-theory

**Governance skills** (3): ix-governance-check, ix-governance-persona, ix-governance-belief

**Federation skills** (4): federation-discover, federation-grammar, federation-music, federation-traces

**Ecosystem skills** (4): governed-execute, ecosystem-audit, roadblock-resolver, delegate-cli

### MCP Federation

ix connects to tars (F# reasoning) and ga (music theory) via MCP:

```json
{
  "mcpServers": {
    "ix": { "command": "cargo", "args": ["run", "--release", "-p", "ix-agent"] },
    "tars": { "command": "dotnet", "args": ["run", "--project", "path/to/Tars.Interface.Cli", "--", "mcp", "server"] },
    "ga": { "command": "dotnet", "args": ["run", "--project", "path/to/GaMcpServer"] }
  }
}
```

### Demerzel Governance

Agents operate under the [Demerzel](https://github.com/GuitarAlchemist/Demerzel) constitution (11 articles, 12 personas, tetravalent logic). Named after [R. Daneel Olivaw](https://asimov.fandom.com/wiki/R._Daneel_Olivaw) — consistent with Asimov's Zeroth Law.

## Examples

16 runnable examples organized by domain:

| # | Example | Domain | Source |
|---|---------|--------|--------|
| 1 | **PSO Rosenbrock** — Minimize a 10D cost function | Optimization | [`examples/optimization/pso_rosenbrock.rs`](examples/optimization/pso_rosenbrock.rs) |
| 2 | **Decision Tree** — CART classification with probabilities | Supervised | [`examples/supervised/decision_tree.rs`](examples/supervised/decision_tree.rs) |
| 3 | **K-Means Clustering** — Segment data into k groups | Unsupervised | [`examples/unsupervised/kmeans_clustering.rs`](examples/unsupervised/kmeans_clustering.rs) |
| 4 | **DBSCAN Anomaly** — Density-based clustering + noise detection | Unsupervised | [`examples/unsupervised/dbscan_anomaly.rs`](examples/unsupervised/dbscan_anomaly.rs) |
| 5 | **Viterbi HMM** — Decode hidden states, Baum-Welch training | Sequence | [`examples/sequence/viterbi_hmm.rs`](examples/sequence/viterbi_hmm.rs) |
| 6 | **Nash Equilibrium** — Prisoner's Dilemma analysis | Game Theory | [`examples/game-theory/nash_equilibrium.rs`](examples/game-theory/nash_equilibrium.rs) |
| 7 | **A\* & Q\* Search** — Hand-crafted vs learned heuristics | Search | [`examples/search/astar_qstar.rs`](examples/search/astar_qstar.rs) |
| 8 | **Logistic Map** — Lyapunov exponents and chaos detection | Chaos | [`examples/chaos/logistic_map.rs`](examples/chaos/logistic_map.rs) |
| 9 | **DAG Pipeline** — Parallel data flow with memoization | Pipeline | [`examples/pipeline/dag_pipeline.rs`](examples/pipeline/dag_pipeline.rs) |
| 10 | **Robustness Test** — FGSM/PGD attacks and defenses | Adversarial | [`examples/adversarial/robustness_test.rs`](examples/adversarial/robustness_test.rs) |
| 11 | **FFT Analysis** — Frequency decomposition of signals | Signal | [`examples/signal/fft_analysis.rs`](examples/signal/fft_analysis.rs) |
| 12 | **Auctions** — First-price, second-price, English, Dutch | Game Theory | [`examples/game-theory/auctions.rs`](examples/game-theory/auctions.rs) |
| 13 | **Bandits** — Thompson sampling for A/B testing | RL | [`examples/reinforcement-learning/bandits.rs`](examples/reinforcement-learning/bandits.rs) |
| 14 | **Bloom Filter** — Probabilistic membership + HyperLogLog | Probabilistic | [`examples/probabilistic/bloom_filter.rs`](examples/probabilistic/bloom_filter.rs) |
| 15 | **GPU Similarity** — WGPU cosine similarity search | GPU | [`examples/gpu/similarity_search.rs`](examples/gpu/similarity_search.rs) |
| 16 | **Embedded Cache** — TTL, LRU, pub/sub, Redis-style ops | Cache | [`examples/cache/embedded_cache.rs`](examples/cache/embedded_cache.rs) |

Run any example:
```bash
cargo run --example pso_rosenbrock
cargo run --example viterbi_hmm
cargo run --example bloom_filter
# etc.
```

## Documentation

**[Learning Path — 60+ tutorials](docs/INDEX.md)** from foundations to advanced topics, with runnable Rust examples.

**[🇫🇷 Parcours d'apprentissage en français](docs/fr/INDEX.md)** — 11 tutoriels d'algorithmes + 4 cas pratiques traduits en français.

Topics: linear algebra, optimization, supervised/unsupervised learning, neural networks, reinforcement learning, game theory, signal processing, chaos theory, adversarial ML, GPU computing, and more.

## Architecture

```
ix/
├── Cargo.toml                 # Workspace root
├── CLAUDE.md                  # Project conventions for Claude Code
├── .mcp.json                  # MCP server configuration
├── .claude/
│   ├── skills/                # 10 Claude Code skills
│   ├── hooks/                 # Pipeline validation, cache lifecycle
│   └── agents/                # Compound engineering agents
└── crates/
    ├── ix-math/           # Core math primitives
    ├── ix-optimize/       # Optimization algorithms
    ├── ix-supervised/     # Regression, classification
    ├── ix-unsupervised/   # Clustering, dimensionality reduction
    ├── ix-ensemble/       # Random forest
    ├── ix-nn/             # Neural networks
    ├── ix-rl/             # Reinforcement learning
    ├── ix-evolution/      # Evolutionary algorithms
    ├── ix-graph/          # Graphs, Markov, HMM
    ├── ix-search/         # Search algorithms
    ├── ix-game/           # Game theory
    ├── ix-chaos/          # Chaos theory
    ├── ix-signal/         # Signal processing
    ├── ix-adversarial/    # Adversarial ML
    ├── ix-probabilistic/  # Probabilistic data structures
    ├── ix-gpu/            # GPU compute (WGPU)
    ├── ix-cache/          # Embedded cache
    ├── ix-pipeline/       # DAG executor
    ├── ix-io/             # Data I/O
    ├── ix-dynamics/       # IK, Lie groups, neural ODEs
    ├── ix-topo/           # Persistent homology
    ├── ix-ktheory/        # Graph K-theory
    ├── ix-category/       # Category theory
    ├── ix-grammar/        # Formal grammars
    ├── ix-agent/          # MCP server
    ├── ix-skill/          # CLI binary
    └── ix-demo/           # egui demo app
```

## Key Dependencies

- **ndarray** 0.17 — Matrix operations (`f64`)
- **rand** 0.9 + **rand_distr** 0.5 — Random number generation
- **wgpu** 28 — Cross-platform GPU compute
- **parking_lot** 0.12 — Fast concurrent locks
- **tokio** 1 — Async runtime (I/O, cache server)
- **thiserror** 2 — Error types
- **clap** 4 — CLI parsing

## Testing

```bash
# Unit + integration tests
cargo test --workspace

# Clippy (zero warnings policy)
cargo clippy --workspace -- -D warnings

# Property-based tests (proptest)
cargo test --workspace -- --include-ignored

# Benchmarks (criterion)
cargo bench -p ix-math
```

- **proptest** for math invariants (commutativity, associativity, norm preservation)
- **criterion** for performance-critical paths (FFT, matrix ops, GPU kernels)

## Conventions

- Pure Rust (except WGPU for GPU compute)
- CPU algorithms use `f64`; GPU uses `f32` via WGSL shaders
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
- Each crate defines domain traits (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, etc.)
- MSRV: Rust 1.80+ (due to wgpu 28)

## License

MIT
