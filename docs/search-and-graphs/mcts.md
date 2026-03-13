# Monte Carlo Tree Search (MCTS)

## The Problem

You are building an AI for a board game -- Go, Chess, or a custom strategy game. The game tree is enormous: Go has roughly 10^170 legal positions, far too many to search exhaustively. You cannot write a reliable evaluation function either, because the game is too complex for simple heuristics.

You need an algorithm that can play at a strong level even when it cannot see the entire tree, learns which moves are promising by sampling, and improves with more thinking time.

## The Intuition

Imagine you move to a new city and want to find the best restaurant. You could:

1. **Try a random restaurant** each night (pure random sampling).
2. **Always go back to the best one you have found** (pure exploitation).
3. **Balance both:** mostly go to restaurants you already like, but occasionally try somewhere new in case you are missing something.

MCTS does option 3 for game trees. It repeatedly:

- **Selects** a promising branch of the game tree (exploitation), with a bonus for branches it has not explored much (exploration).
- **Simulates** a random game from that point to the end (rollout).
- **Updates** the win/loss statistics back up the tree.

Over thousands of iterations, the statistics converge: moves that lead to wins get visited more, moves that lead to losses get avoided. The algorithm does not need a hand-crafted evaluation function -- the random rollouts *are* the evaluation.

The exploration-exploitation tradeoff is controlled by the **UCB1 formula**, borrowed from the multi-armed bandit problem. It is the same math that helps online advertisers decide which ad to show.

## How It Works

Each iteration has four phases:

### 1. Selection

Starting from the root, walk down the tree by picking the child with the highest UCB1 score until you reach a node with untried actions or a terminal state.

```
UCB1(child) = (wins / visits) + c * sqrt(ln(parent_visits) / visits)
```

| Term | Meaning |
|------|---------|
| `wins / visits` | **Exploitation:** average reward of this child (between 0 and 1) |
| `c * sqrt(ln(parent_visits) / visits)` | **Exploration:** bonus for rarely-visited children |
| `c` | Exploration constant. Higher = explore more. sqrt(2) ~ 1.41 is the theoretical optimum for [0,1] rewards. |

**In plain English:** pick the child that either (a) has been winning a lot, or (b) has not been tried much yet. The `c` parameter controls how adventurous the algorithm is.

### 2. Expansion

At the selected node, pick one untried action at random. Create a new child node for it.

### 3. Simulation (Rollout)

From the new child, play a completely random game until a terminal state is reached. Record the reward (1.0 = win, 0.0 = loss, 0.5 = draw).

### 4. Backpropagation

Walk back up from the new child to the root, adding 1 to `visits` and adding the reward to `total_reward` at each ancestor.

After all iterations, return the root's **most-visited child** as the best move (most-visited, not highest-average, because visit count is more robust).

## In Rust

The `machin-search` crate provides MCTS through the `MctsState` trait:

```rust
use machin_search::mcts::{MctsState, mcts_search};

// A simple Nim-like game: players take turns adding 1-3, goal is to reach 21.
#[derive(Clone, Debug)]
struct NimState {
    count: i32,
    my_turn: bool,
}

impl MctsState for NimState {
    type Action = i32; // How many to add (1, 2, or 3)

    fn legal_actions(&self) -> Vec<i32> {
        if self.is_terminal() { return vec![]; }
        let max = 3.min(21 - self.count);
        (1..=max).collect()
    }

    fn apply(&self, action: &i32) -> Self {
        NimState {
            count: self.count + action,
            my_turn: !self.my_turn,
        }
    }

    fn is_terminal(&self) -> bool {
        self.count >= 21
    }

    fn reward(&self) -> f64 {
        // The player who just moved to 21 wins.
        if self.count >= 21 {
            if self.my_turn { 0.0 } else { 1.0 }
        } else {
            0.5 // Non-terminal draw
        }
    }
}

fn main() {
    let state = NimState { count: 0, my_turn: true };

    // Run 5000 iterations with exploration constant 1.41, seed 42.
    let best_action: Option<i32> = mcts_search(&state, 5000, 1.41, 42);

    match best_action {
        Some(a) => println!("MCTS recommends adding: {}", a),
        None => println!("No legal moves"),
    }
}
```

### API reference

```rust
pub fn mcts_search<S: MctsState>(
    root: &S,           // Current game state (not consumed)
    iterations: usize,  // Number of selection-expansion-rollout cycles
    exploration: f64,    // UCB1 exploration constant c
    seed: u64,           // RNG seed for reproducibility
) -> Option<S::Action>  // The best action, or None if no legal moves
```

### MctsState trait

| Method | Signature | Purpose |
|--------|-----------|---------|
| `legal_actions` | `&self -> Vec<Action>` | All valid moves from this state |
| `apply` | `&self, &Action -> Self` | Return the state after taking an action (immutable) |
| `is_terminal` | `&self -> bool` | True if the game is over |
| `reward` | `&self -> f64` | Terminal evaluation: 1.0 = win, 0.0 = loss, 0.5 = draw |

## When To Use This

| Situation | Use MCTS? | Alternative |
|-----------|-----------|-------------|
| Large branching factor (Go, complex strategy) | Yes | -- |
| Small game tree (tic-tac-toe) | Overkill | Minimax with alpha-beta |
| Need deterministic perfect play | No | Alpha-beta to full depth |
| No good evaluation function available | Yes | -- |
| Real-time constraint (< 1ms per move) | Maybe not | Alpha-beta with shallow depth |
| Stochastic game (dice, card draws) | Yes, naturally handles randomness | Expectiminimax |
| Single-agent search (pathfinding) | No | A* or Q* |

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `iterations` | 1,000 -- 100,000 | More = stronger play, longer thinking time. Strength scales roughly as sqrt(iterations). |
| `exploration` (c) | 1.41 (sqrt(2)) | Lower = more exploitation (aggressive, can miss good moves). Higher = more exploration (cautious, slower convergence). |
| `seed` | Any u64 | Determines rollout randomness. Same seed = same result for reproducibility. |

### Tuning exploration

- **c = 0.5:** Very exploitative. Good when the game has few traps (most moves are similar).
- **c = 1.41:** Balanced default. Start here.
- **c = 2.0+:** Very exploratory. Use when the game has rare but critical moves that are easy to miss.

## Pitfalls

1. **Reward must be from the right perspective.** `reward()` is called on a terminal state and backpropagated to all ancestors. If your game alternates turns, make sure the reward reflects the perspective of the *root* player, not the player who just moved. The implementation in machin-search backpropagates the same reward to all ancestors, so design `reward()` accordingly.

2. **Random rollouts can be weak.** In games where random play is wildly different from good play (e.g., Chess -- random moves lose material instantly), raw MCTS with random rollouts will need many more iterations. The fix is a better rollout policy (not yet built into machin-search, but you can modify the algorithm).

3. **Rollout depth limit.** The implementation caps rollouts at 500 moves to prevent infinite games. If your game can last longer, the rollout will return the `reward()` of a non-terminal state.

4. **Memory grows with iterations.** Each iteration adds at most one node to the tree. 100,000 iterations = up to 100,000 nodes in memory. For very long-running searches, this can become significant.

5. **Not suitable for single-agent optimization.** MCTS is designed for adversarial or stochastic games. For pathfinding, use A*. For optimization, use simulated annealing or evolutionary algorithms.

## Going Further

- **RAVE (Rapid Action Value Estimation):** Shares statistics across the tree for moves that appear at different positions. Dramatically speeds up convergence in Go.
- **Progressive widening:** Limits the branching factor in continuous action spaces by only expanding new children when visit count exceeds a threshold.
- **Neural network rollouts:** Replace random rollouts with a trained policy network (the AlphaGo approach). The `MctsState` trait makes this straightforward -- implement a custom rollout that queries your model.
- **Minimax + Alpha-Beta:** For smaller, fully-observable, deterministic games, alpha-beta is stronger per node. See [Minimax and Alpha-Beta](./minimax-alpha-beta.md).
- Read the survey: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012).
