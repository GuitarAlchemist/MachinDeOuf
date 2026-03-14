# A* Search and Variants

## The Problem

You are building a video game where enemies must navigate a tile-based dungeon to chase the player. The dungeon has walls, corridors, and open rooms. You need the shortest path from the enemy to the player -- and you need it computed in milliseconds, not seconds, because there are dozens of enemies on screen at once.

Or: you are routing delivery trucks across a city. Each road segment has a travel time. You want the fastest route from the warehouse to each customer, and you want the algorithm to skip obviously-wrong directions early (heading away from the destination) rather than exploring every road in the city.

These are instances of the **shortest-path problem** on a graph where you have a good guess (heuristic) about how far each node is from the goal.

## The Intuition

Imagine you are lost in a maze. You have two pieces of information at every intersection:

1. **How far you have walked** from the entrance (the *g-cost*).
2. **A rough guess** of how far the exit is from here -- say, the straight-line distance through the walls (the *heuristic h*).

A naive approach (Dijkstra's algorithm) only uses (1): it fans out evenly in all directions like a ripple in a pond. A* adds (2): it says "expand the node that looks cheapest overall -- the one where *walked-so-far + guess-to-go* is smallest." This focuses the search toward the goal, like a flashlight instead of a floodlight.

The guarantee: as long as your guess **never overestimates** the real distance (called *admissibility*), A* will find the true shortest path. If the guess is also *consistent* (the triangle inequality holds), A* will never need to re-expand a node.

**Weighted A*** relaxes optimality for speed: it inflates the heuristic by a factor `w`. The found path is at most `w` times the optimal cost, but the search can be dramatically faster.

**Greedy Best-First Search** throws away the walked-so-far cost entirely and only follows the heuristic. It is fast but the path it finds can be arbitrarily bad.

## How It Works

A* maintains two data structures:

- An **open list** (priority queue) of nodes to explore, sorted by `f(n)`.
- A **closed set** of already-expanded nodes.

For each node `n`:

```
f(n) = g(n) + h(n)
```

| Symbol | Meaning |
|--------|---------|
| `g(n)` | Actual cost of the cheapest known path from **start** to `n` |
| `h(n)` | Heuristic estimate of the cost from `n` to the **goal** |
| `f(n)` | Estimated total cost of the cheapest path through `n` |

**In plain English:** g is what you have spent, h is what you expect to spend, and f is the total budget. A* always picks the node with the lowest total budget next.

The algorithm:

1. Put the start node in the open list with `g = 0`, `f = h(start)`.
2. Pop the node with the lowest `f` from the open list.
3. If it is the goal, reconstruct the path and return.
4. Otherwise, expand it: for each successor, compute `new_g = g(current) + step_cost`. If `new_g` is better than any previously known path to that successor, update it and add it to the open list.
5. Repeat until the goal is found or the open list is empty (no path exists).

**Weighted A*** uses `f(n) = g(n) + w * h(n)`. A larger `w` makes the heuristic louder, causing the search to beeline toward the goal.

**Greedy Best-First** uses `f(n) = h(n)` -- it ignores path cost entirely.

## In Rust

The `ix-search` crate models search problems through the `SearchState` trait:

```rust
use ix_search::astar::{SearchState, astar, weighted_astar, greedy_best_first, SearchResult};

// Define your state by implementing SearchState.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GridPos {
    x: i32,
    y: i32,
    goal_x: i32,
    goal_y: i32,
    width: i32,
    height: i32,
}

impl SearchState for GridPos {
    // Actions are (dx, dy) direction tuples.
    type Action = (i32, i32);

    // Return (action, successor_state, step_cost) triples.
    fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        dirs.iter()
            .filter_map(|&(dx, dy)| {
                let nx = self.x + dx;
                let ny = self.y + dy;
                if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                    Some(((dx, dy), GridPos { x: nx, y: ny, ..*self }, 1.0))
                } else {
                    None
                }
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.x == self.goal_x && self.y == self.goal_y
    }
}

fn main() {
    let start = GridPos { x: 0, y: 0, goal_x: 9, goal_y: 9, width: 20, height: 20 };

    // Manhattan distance heuristic (admissible for 4-connected grids).
    let h = |s: &GridPos| ((s.x - s.goal_x).abs() + (s.y - s.goal_y).abs()) as f64;

    // --- Standard A* (optimal) ---
    let result: SearchResult<GridPos> = astar(start.clone(), h).unwrap();
    println!("A* cost: {}, nodes expanded: {}", result.cost, result.nodes_expanded);
    // result.path    -- Vec<GridPos> from start to goal
    // result.actions -- Vec<(i32,i32)> directions taken
    // result.cost    -- total path cost (18.0 for a 9+9 grid)

    // --- Weighted A* (faster, bounded suboptimal) ---
    let fast = weighted_astar(start.clone(), h, 1.5).unwrap();
    println!("WA* cost: {}, nodes expanded: {}", fast.cost, fast.nodes_expanded);
    // fast.cost <= 1.5 * result.cost  (guaranteed)

    // --- Greedy Best-First (fast, no optimality guarantee) ---
    let greedy = greedy_best_first(start, h).unwrap();
    println!("Greedy cost: {}, nodes expanded: {}", greedy.cost, greedy.nodes_expanded);
}
```

### SearchResult fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Vec<S>` | Sequence of states from start to goal |
| `actions` | `Vec<S::Action>` | Actions taken at each step |
| `cost` | `f64` | Total path cost |
| `nodes_expanded` | `usize` | Nodes popped from the open list |
| `nodes_generated` | `usize` | Total successor nodes created |

See the full working example: [examples/search/astar_qstar.rs](../../examples/search/astar_qstar.rs)

## When To Use This

| Algorithm | Optimal? | Speed | Memory | Best For |
|-----------|----------|-------|--------|----------|
| `astar` | Yes (with admissible h) | Fast with good h | O(nodes) | Most pathfinding |
| `weighted_astar` (w > 1) | No, but bounded (cost <= w * optimal) | Faster | O(nodes) | Real-time games, "good enough" paths |
| `greedy_best_first` | No | Fastest (often) | O(nodes) | Quick-and-dirty, when optimality does not matter |
| `uniform_cost_search` | Yes | Slowest (A* with h=0) | O(nodes) | When you have no heuristic |
| `bidirectional_astar` | Yes | Faster on large symmetric graphs | 2x memory | Large maps with known start and goal |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `start` | `S: SearchState` | The initial state. Must implement `Clone + Eq + Hash`. |
| `heuristic` | `Fn(&S) -> f64` | Estimated cost to goal. Must be >= 0. **Admissible** means it never overestimates. |
| `weight` (weighted A*) | `f64` | Heuristic inflation. 1.0 = standard A*. 2.0 = paths at most 2x optimal but much faster search. |

### Choosing a heuristic

- **Grid (4-connected):** Manhattan distance `|dx| + |dy|`
- **Grid (8-connected, diagonals):** Chebyshev distance `max(|dx|, |dy|)`
- **Euclidean space:** Straight-line distance `sqrt(dx^2 + dy^2)`
- **Road networks:** Haversine (great-circle) distance
- **No domain knowledge:** Use `|_| 0.0` (falls back to Dijkstra)

## Pitfalls

1. **Inadmissible heuristic breaks optimality.** If your heuristic overestimates even once, A* may return a suboptimal path. Weighted A* is the principled way to trade optimality for speed -- use it instead of a sloppy heuristic.

2. **Memory.** A* stores every generated node. On a 10,000 x 10,000 grid with no obstacles, that can be millions of nodes. Consider IDA* (iterative deepening A*) for memory-constrained environments, or weighted A* to reduce the frontier size.

3. **Hash collisions.** The `SearchState` trait requires `Hash + Eq`. If your state hashing is slow or collision-prone, the closed-set lookups become a bottleneck. Keep states small and hashable.

4. **Expensive successors.** `successors()` is called once per expansion. If generating successors is costly (e.g., physics simulation), minimize the branching factor or cache results.

5. **Forgetting the goal in the state.** The `is_goal()` method is called on each state. If your goal information is external to the state struct, you need to embed it (or use a closure) so the trait method can check it.

## Going Further

- **Bidirectional A*:** `bidirectional_astar(start, goal, fwd_h, rev_h)` searches from both ends simultaneously, meeting in the middle. Can cut search time roughly in half on symmetric graphs.
- **Q* Search:** Replace the hand-crafted heuristic with a learned Q-function for domains where good heuristics are hard to design. See [Q* Learned Heuristics](./qstar-learned-heuristics.md).
- **MCTS:** For game trees where the branching factor is too large for A*, use Monte Carlo Tree Search. See [MCTS](./mcts.md).
- **IDA*:** Not yet in ix-search, but easy to build on top of `SearchState` using iterative deepening with an f-cost threshold.
- Read the original paper: Hart, Nilsson, Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (1968).
