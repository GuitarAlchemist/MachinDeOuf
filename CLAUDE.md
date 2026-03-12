# MachinDeOuf - ML Algorithms for Claude Code Skills

## Project Overview
Rust workspace implementing foundational ML/math algorithms as composable crates, designed to be exposed as Claude Code skills via a CLI (`machin`).

## Architecture
- `crates/machin-math` - Core math: linalg, stats, distance, activation, calculus, random, hyperbolic (Poincaré)
- `crates/machin-optimize` - Optimization: SGD, Adam, simulated annealing, PSO
- `crates/machin-supervised` - Regression, classification, metrics
- `crates/machin-unsupervised` - Clustering (K-Means), dimensionality reduction
- `crates/machin-ensemble` - Random forest, boosting (stub)
- `crates/machin-nn` - Neural network layers, loss functions, backprop
- `crates/machin-rl` - Bandits (epsilon-greedy, UCB1, Thompson), Q-learning
- `crates/machin-evolution` - Genetic algorithms, differential evolution
- `crates/machin-graph` - Graph algorithms, Markov chains, HMM/Viterbi, state spaces, agent routing
- `crates/machin-probabilistic` - Bloom filters, Count-Min sketch, HyperLogLog, Cuckoo filter
- `crates/machin-io` - Data I/O: CSV, JSON, file watcher, named pipes, TCP, HTTP, WebSocket
- `crates/machin-signal` - Signal processing: FFT, wavelets, filters, Kalman, spectral analysis
- `crates/machin-chaos` - Chaos theory: Lyapunov exponents, bifurcation, attractors, fractals, embedding
- `crates/machin-game` - Game theory: Nash equilibria, Shapley value, auctions, evolutionary, mean field
- `crates/machin-search` - Search: A*, Q*, MCTS, minimax, alpha-beta, BFS/DFS, data structure search
- `crates/machin-gpu` - GPU compute via WGPU: cosine similarity, matrix multiply, batch vector search
- `crates/machin-cache` - Embedded Redis-like cache: concurrent sharded store, TTL, LRU, pub/sub, RESP server
- `crates/machin-pipeline` - DAG pipeline executor: skill orchestration, parallel branches, memoized data flow
- `crates/machin-adversarial` - Adversarial ML: evasion (FGSM, PGD, C&W), defense, poisoning detection, privacy
- `crates/machin-skill` - CLI exposing all algorithms as Claude Code skills

## Build
```bash
cargo build --workspace
cargo test --workspace
```

## Key Dependencies
- `ndarray` 0.17 - Matrix operations
- `rand` 0.9 + `rand_distr` 0.5 - Random number generation
- `thiserror` 2 - Error types
- `clap` 4 - CLI parsing
- `wgpu` 28 - Cross-platform GPU compute (Vulkan/DX12/Metal)
- `tokio` 1 - Async runtime (I/O crate)

## Conventions
- Pure Rust, no external ML frameworks (except wgpu for GPU compute)
- CPU algorithms use `f64` and `ndarray::Array{1,2}<f64>`; GPU uses `f32` via WGPU shaders
- Each crate defines traits (Regressor, Classifier, Clusterer, Optimizer, etc.)
- Builder pattern for algorithm configuration
- Seeded RNG for reproducibility
