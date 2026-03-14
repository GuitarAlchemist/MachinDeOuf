---
name: machin-evolution
description: Evolutionary optimization — genetic algorithm, differential evolution
---

# Evolutionary Optimization

Population-based global optimization on benchmark functions.

## When to Use
When the user needs gradient-free global optimization, wants to compare GA vs DE, or is working with non-convex objective functions.

## Capabilities
- **Genetic algorithm** — Tournament selection, crossover, mutation
- **Differential evolution** — DE/rand/1/bin strategy
- **Benchmark functions** — Sphere, Rosenbrock, Rastrigin

## Key Concepts
- GA: good for discrete/mixed problems, uses crossover + mutation
- DE: excellent for continuous optimization, uses vector differences
- Both are population-based and gradient-free

## Programmatic Usage
```rust
use machin_evolution::genetic::GeneticAlgorithm;
use machin_evolution::differential::DifferentialEvolution;
```

## MCP Tool
Tool name: `machin_evolution`
Parameters: `algorithm` (genetic/differential), `function`, `dimensions`, `generations`
