# Q* Search: Learned Heuristics

## The Problem

A* search is powerful, but it requires a hand-crafted heuristic -- a function that estimates the distance to the goal. For a grid, Manhattan distance works. For a road network, straight-line distance works. But what about:

- A robot navigating a warehouse where some aisles are periodically blocked?
- A logistics optimizer routing packages through a network with dynamic pricing?
- A game AI where the "distance to winning" depends on a complex combination of factors?

In these domains, writing a good admissible heuristic by hand is somewhere between hard and impossible. You have data (past searches, simulations, game logs) but no formula.

**Q* search** replaces the hand-crafted heuristic with a **learned Q-function** -- a model trained to estimate cost-to-go from any state. The name comes from combining Q-learning (reinforcement learning) with A* search.

## The Intuition

Standard A* asks a human expert: "How far is this state from the goal?" and uses the answer to prioritize search.

Q* asks a trained model the same question. The model has seen thousands of solved instances and learned patterns that a human could not easily articulate. It might be wrong sometimes, but on average it guides the search far better than a naive heuristic.

The key trick: standard A* evaluates the heuristic for **every successor** node. If a state has 100 successors, that is 100 heuristic calls per expansion. Q* evaluates the heuristic **once per expanded node** and adjusts successor estimates by subtracting the step cost. In domains with large branching factors (many actions per state), this reduces heuristic evaluations by orders of magnitude.

Think of it this way: if you have a rough estimate of how far *you* are from the goal, and you take a step that costs 3, your successor is roughly "your estimate minus 3" from the goal. You do not need to re-evaluate from scratch.

## How It Works

Q* uses the same `f(n) = g(n) + h(n)` framework as A*, with two changes:

1. **h(n) comes from a learned Q-function** instead of a hand-crafted formula.
2. **h(successor) is approximated** as `max(0, h(parent) - step_cost)` instead of calling the Q-function again.

```
qstar(start, Q):
    h_start = Q.estimate_cost_to_go(start)
    open = priority_queue with (start, g=0, f=h_start)

    while open is not empty:
        current = pop lowest-f node
        if current is goal: return path

        h_current = Q.estimate_cost_to_go(current)   # One Q-call per expansion

        for (action, successor, step_cost) in current.successors():
            new_g = g(current) + step_cost
            h_succ = max(0, h_current - step_cost)    # No Q-call for successors!
            f_succ = new_g + h_succ
            add successor to open with f_succ
```

**In plain English:** Ask the model "how far am I?" once each time you expand a node. For each successor, estimate "a bit closer by the cost I just paid" without asking the model again.

### Weighted Q*

Like weighted A*, you can inflate the Q-function: `f(n) = g(n) + w * Q(n)`. This trades optimality for speed:

- `w = 1.0`: Optimal (if Q is admissible).
- `w > 1.0`: Faster search, path cost at most `w * optimal`.

### Two-Head Q*

For non-uniform action costs, a single Q-value per state is not enough. The **two-head** variant uses:

- **Head 1:** Estimates the transition cost `c(state, action)`.
- **Head 2:** Estimates the cost-to-go `h(successor)`.

This is more accurate but requires calling the Q-function for each successor (losing the one-call-per-expansion advantage).

## In Rust

The `machin-search` crate provides Q* through the `QFunction` trait:

```rust
use machin_search::astar::SearchState;
use machin_search::qstar::{
    QFunction, TabularQ, qstar_search, qstar_weighted,
    qstar_two_head, compare_qstar_vs_astar, QStarResult,
};

// --- Using TabularQ for small/discrete state spaces ---

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GridPos {
    x: i32, y: i32, goal_x: i32, goal_y: i32, width: i32, height: i32,
}

impl SearchState for GridPos {
    type Action = (i32, i32);
    fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
        let dirs = [(0,1),(0,-1),(1,0),(-1,0)];
        dirs.iter().filter_map(|&(dx,dy)| {
            let (nx, ny) = (self.x + dx, self.y + dy);
            if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                Some(((dx,dy), GridPos { x: nx, y: ny, ..*self }, 1.0))
            } else { None }
        }).collect()
    }
    fn is_goal(&self) -> bool { self.x == self.goal_x && self.y == self.goal_y }
}

fn main() {
    let start = GridPos { x: 0, y: 0, goal_x: 9, goal_y: 9, width: 20, height: 20 };

    // TabularQ: a simple hash-map based Q-function.
    // Default value 10.0 means "unknown states are estimated as 10 steps away."
    let q = TabularQ::new(10.0);
    // In practice you would train this: q.set(state, learned_value);

    // --- Basic Q* search ---
    let result: QStarResult<GridPos> = qstar_search(start.clone(), &q).unwrap();
    println!("Q* cost: {}, nodes expanded: {}, heuristic calls: {}",
             result.cost, result.nodes_expanded, result.heuristic_calls);

    // --- Weighted Q* (faster, bounded suboptimal) ---
    let fast = qstar_weighted(start.clone(), &q, 2.0).unwrap();
    println!("Weighted Q* cost: {} (at most 2x optimal)", fast.cost);

    // --- Compare Q* vs A* on the same problem ---
    let manhattan = |s: &GridPos| ((s.x - s.goal_x).abs() + (s.y - s.goal_y).abs()) as f64;
    let (qr, ar) = compare_qstar_vs_astar(start, &q, manhattan);
    println!("Q* expanded: {}, A* expanded: {}",
             qr.unwrap().nodes_expanded, ar.unwrap().nodes_expanded);
}
```

### Implementing a custom Q-function

For real applications, replace `TabularQ` with a neural network or any learned model:

```rust
use machin_search::qstar::QFunction;

struct MyNeuralQ {
    // Your model weights, ONNX runtime, etc.
}

impl QFunction<MyState> for MyNeuralQ {
    fn estimate_cost_to_go(&self, state: &MyState) -> f64 {
        // Run inference on your model
        self.model.predict(state.to_features())
    }

    // Optional: for two-head Q*
    fn estimate_transition_cost(&self, state: &MyState, action_idx: usize) -> Option<f64> {
        Some(self.model.predict_transition(state.to_features(), action_idx))
    }
}
```

### QStarResult fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Vec<S>` | States from start to goal |
| `actions` | `Vec<S::Action>` | Actions taken |
| `cost` | `f64` | Total path cost |
| `nodes_expanded` | `usize` | Nodes popped from the open list |
| `nodes_generated` | `usize` | Successor nodes created |
| `heuristic_calls` | `usize` | Number of Q-function evaluations (the key efficiency metric) |

### API summary

| Function | Optimal? | Q-calls per expansion | Use case |
|----------|----------|----------------------|----------|
| `qstar_search(start, &q)` | Yes (if Q admissible) | 1 | Standard learned-heuristic search |
| `qstar_weighted(start, &q, w)` | Bounded (cost <= w * optimal) | 1 | Faster search with bounded suboptimality |
| `qstar_two_head(start, &q)` | Depends on model | 1 + per successor | Non-uniform action costs |
| `qstar_bounded(start, &q, eps)` | cost <= (1+eps) * optimal | 1 | Explicit epsilon-bounded search |
| `compare_qstar_vs_astar(start, &q, h)` | Both | -- | Performance comparison |

See the full working example: [examples/search/astar_qstar.rs](../../examples/search/astar_qstar.rs)

## When To Use This

| Situation | Q* or A*? | Why |
|-----------|-----------|-----|
| Good hand-crafted heuristic available | A* | Simpler, no training needed |
| No good heuristic, but have training data | Q* | Learned Q-function fills the gap |
| Large branching factor (many actions) | Q* | One Q-call per expansion vs one per successor |
| Neural network inference is expensive | Q* (with caching) | Minimizes inference calls |
| Need guaranteed optimality | Either (if heuristic/Q is admissible) | Same guarantee |
| Dynamic environment (costs change) | Q* | Retrain the Q-function; hand-crafted heuristics break |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `start` | `S: SearchState` | Initial state |
| `q_function` | `&Q: QFunction<S>` | Learned cost-to-go estimator |
| `weight` | `f64` | Heuristic inflation (1.0 = optimal, >1 = faster) |
| `epsilon` (bounded) | `f64` | Suboptimality bound (cost <= (1+eps) * optimal) |

### TabularQ parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `default` | `f64` | Cost estimate for unseen states. Higher = more conservative (explores more). Lower = more aggressive (may miss paths). |

## Pitfalls

1. **Inadmissible Q-function.** If the Q-function overestimates cost-to-go, Q* may return suboptimal paths. Unlike hand-crafted heuristics where admissibility is provable, learned models can overestimate unpredictably. Use `qstar_weighted` with a known bound if optimality matters.

2. **The subtraction trick can lose information.** Approximating `h(successor) = max(0, h(parent) - step_cost)` works well when costs are uniform, but can be inaccurate for non-uniform costs. Use `qstar_two_head` in those cases.

3. **TabularQ does not generalize.** `TabularQ` is a hash map -- it only knows about states you explicitly inserted. For large or continuous state spaces, you need a function approximator (neural network, random forest, etc.) that implements `QFunction`.

4. **Training the Q-function is a separate problem.** Q* assumes you already have a trained model. Training it well (from search logs, RL, or supervised learning on solved instances) is the hard part. A badly trained Q-function can be worse than Manhattan distance.

5. **Default value matters.** In `TabularQ::new(default)`, the default is the cost estimate for all unknown states. Too low (optimistic) = the search skips promising states. Too high (pessimistic) = the search explores too broadly. Start with a value near the average cost-to-go in your domain.

## Going Further

- **Training a Q-function:** Use Q-learning or Monte Carlo returns from solved instances. Store `(state, actual_cost_to_go)` pairs and train a regressor.
- **Neural network integration:** Implement `QFunction` with an ONNX runtime or a lightweight Rust ML library. Batch inference across multiple states for GPU efficiency.
- **Comparison with A*:** Use `compare_qstar_vs_astar` to benchmark your learned heuristic against a hand-crafted one. Track `heuristic_calls` as the key efficiency metric.
- **A* Search:** For domains where hand-crafted heuristics work well. See [A* Search](./astar-search.md).
- Read: Agostinelli et al., "Solving the Rubik's Cube with Deep Reinforcement Learning and Search" (2019) -- a real-world application of learned heuristics with A*.
