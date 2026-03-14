# ix - ML Algorithms for Claude Code Skills

## Project Overview
Rust workspace (27 crates) implementing foundational ML/math algorithms as composable crates, exposed as Claude Code skills via MCP server (`ix-agent`) and CLI (`ix-skill`).

## Architecture

### Core Math & Optimization
- `crates/ix-math` - Core math: linalg, stats, distance, activation, calculus, random, hyperbolic (Poincaré)
- `crates/ix-optimize` - Optimization: SGD, Adam, simulated annealing, PSO

### Supervised & Unsupervised Learning
- `crates/ix-supervised` - Regression, classification, metrics
- `crates/ix-unsupervised` - Clustering (K-Means, DBSCAN), PCA
- `crates/ix-ensemble` - Random forest, boosting (stub)

### Deep Learning, RL & Evolution
- `crates/ix-nn` - Neural network layers, loss functions, backprop
- `crates/ix-rl` - Bandits (epsilon-greedy, UCB1, Thompson), Q-learning
- `crates/ix-evolution` - Genetic algorithms, differential evolution

### Search, Graphs & Game Theory
- `crates/ix-graph` - Graph algorithms, Markov chains, HMM/Viterbi, state spaces, agent routing
- `crates/ix-search` - Search: A*, Q*, MCTS, minimax, alpha-beta, BFS/DFS, data structure search
- `crates/ix-game` - Game theory: Nash equilibria, Shapley value, auctions, evolutionary, mean field

### Signal, Chaos & Adversarial
- `crates/ix-signal` - Signal processing: FFT, wavelets, filters, Kalman, spectral analysis
- `crates/ix-chaos` - Chaos theory: Lyapunov exponents, bifurcation, attractors, fractals, embedding
- `crates/ix-adversarial` - Adversarial ML: evasion (FGSM, PGD, C&W), defense, poisoning detection, privacy

### Advanced Math
- `crates/ix-dynamics` - Inverse kinematics, Lie groups/algebras, neural ODEs
- `crates/ix-topo` - Persistent homology, simplicial complexes, Betti numbers
- `crates/ix-ktheory` - Graph K-theory, Mayer-Vietoris sequences
- `crates/ix-category` - Functors, natural transformations, monads
- `crates/ix-grammar` - Formal grammars: CFG, Earley parser, CYK, Chomsky normal form

### Probabilistic & Infrastructure
- `crates/ix-probabilistic` - Bloom filters, Count-Min sketch, HyperLogLog, Cuckoo filter
- `crates/ix-io` - Data I/O: CSV, JSON, file watcher, named pipes, TCP, HTTP, WebSocket
- `crates/ix-gpu` - GPU compute via WGPU: cosine similarity, matrix multiply, batch vector search, quaternions, sedenions
- `crates/ix-cache` - Embedded Redis-like cache: concurrent sharded store, TTL, LRU, pub/sub, RESP server
- `crates/ix-pipeline` - DAG pipeline executor: skill orchestration, parallel branches, memoized data flow

### Integration
- `crates/ix-agent` - MCP server exposing algorithms as Claude Code tools via JSON-RPC over stdio
- `crates/ix-skill` - CLI binary (`machin`) for direct command-line access
- `crates/ix-demo` - egui desktop app with 16 interactive demo tabs

## Build
```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps
```

## Testing
- **Unit tests**: `#[test]` for every public function
- **Property tests**: `proptest` for math invariants (commutativity, norm preservation, rotation orthogonality)
- **Benchmarks**: `criterion` for performance-critical paths (FFT, matrix ops, GPU kernels)
- **Doc tests**: All `///` examples must compile and run
- **CI**: GitHub Actions on stable + nightly Rust, Linux + Windows

## Key Dependencies
- `ndarray` 0.17 - Matrix operations
- `rand` 0.9 + `rand_distr` 0.5 - Random number generation
- `thiserror` 2 - Error types
- `clap` 4 - CLI parsing
- `wgpu` 28 - Cross-platform GPU compute (Vulkan/DX12/Metal)
- `tokio` 1 - Async runtime (I/O crate)
- `proptest` 1 - Property-based testing
- `criterion` 0.5 - Benchmarking

## MSRV
Rust 1.80+ (due to wgpu 28)

## Conventions
- Pure Rust, no external ML frameworks (except wgpu for GPU compute)
- CPU algorithms use `f64` and `ndarray::Array{1,2}<f64>`; GPU uses `f32` via WGPU shaders
- Each crate defines traits (Regressor, Classifier, Clusterer, Optimizer, etc.)
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
