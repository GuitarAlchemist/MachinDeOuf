# Minimax and Alpha-Beta Pruning

## The Problem

You are building an AI for tic-tac-toe, checkers, or chess. Two players take turns. One player (the *maximizer*) wants to reach states with the highest score; the other (the *minimizer*) wants the lowest score. Each player plays perfectly -- they assume the opponent will also play perfectly.

You need an algorithm that answers: "What is the best move I can make, assuming my opponent will always respond with *their* best move?"

## The Intuition

Think of it as a negotiation. You propose a deal (your move). Your opponent will always pick the counter-offer that is worst for you. So you should choose the proposal where, even after the opponent picks their best counter, you are still in the best possible position.

**Minimax** formalizes this by building the entire game tree:

- At **your** turns (maximizer), you pick the child with the **highest** value.
- At **opponent** turns (minimizer), they pick the child with the **lowest** value.
- At terminal states (game over or depth limit), you evaluate the position with a static score.

The problem: the game tree is huge. Chess has roughly 35 legal moves per position and games last 40+ moves, giving 35^40 nodes. You cannot search them all.

**Alpha-Beta Pruning** is the insight that you do not need to. If you already know that the maximizer has a move guaranteeing a score of 5, and you discover a branch where the minimizer can force a score of 3, you can skip the rest of that branch. The maximizer would never go there.

Alpha-beta evaluates the same moves as minimax but **prunes** (skips) branches that cannot affect the final decision. In the best case (perfectly ordered moves), it searches only the square root of the nodes that minimax would.

**Expectiminimax** extends this to games with chance elements (dice rolls, card draws). Chance nodes compute the weighted average over all random outcomes.

## How It Works

### Minimax

```
minimax(state, depth):
    if depth == 0 or state is terminal:
        return evaluate(state)
    if maximizer's turn:
        return max over children of minimax(child, depth - 1)
    else:
        return min over children of minimax(child, depth - 1)
```

**In plain English:** look ahead `depth` moves, assume both players play optimally, and bubble up the best achievable score.

### Alpha-Beta

Add two bounds that track what each player can already guarantee:

- **alpha:** the best score the maximizer can guarantee so far (starts at negative infinity).
- **beta:** the best score the minimizer can guarantee so far (starts at positive infinity).

```
alpha_beta(state, depth, alpha, beta):
    if depth == 0 or state is terminal:
        return evaluate(state)
    if maximizer's turn:
        for each child:
            value = alpha_beta(child, depth-1, alpha, beta)
            alpha = max(alpha, value)
            if alpha >= beta: break   # Beta cutoff -- minimizer would avoid this
        return alpha
    else:
        for each child:
            value = alpha_beta(child, depth-1, alpha, beta)
            beta = min(beta, value)
            if alpha >= beta: break   # Alpha cutoff -- maximizer would avoid this
        return beta
```

**In plain English:** "I already have a deal worth 5. This branch just showed the opponent can force 3 here. No need to look further -- I would never pick this branch."

### Expectiminimax

At chance nodes, compute:

```
expected_value = sum over outcomes of: probability * minimax(outcome_state, depth - 1)
```

**In plain English:** if there is a dice roll, weight each outcome by its probability and take the average.

## In Rust

The `machin-search` crate models adversarial games through the `GameState` trait:

```rust
use machin_search::adversarial::{
    GameState, AdversarialResult, minimax, alpha_beta, expectiminimax,
    StochasticGameState,
};

// A number game: two players add +1 or -1. Maximizer wants a high value.
#[derive(Clone, Debug)]
struct NumberGame {
    value: i32,
    max_turn: bool,
    turns_left: usize,
}

impl GameState for NumberGame {
    type Move = i32;

    fn legal_moves(&self) -> Vec<i32> {
        if self.is_terminal() { vec![] } else { vec![1, -1] }
    }

    fn apply_move(&self, m: &i32) -> Self {
        NumberGame {
            value: self.value + m,
            max_turn: !self.max_turn,
            turns_left: self.turns_left - 1,
        }
    }

    fn is_terminal(&self) -> bool {
        self.turns_left == 0 || self.value <= 0 || self.value >= 10
    }

    fn is_maximizer_turn(&self) -> bool {
        self.max_turn
    }

    fn evaluate(&self) -> f64 {
        self.value as f64
    }
}

fn main() {
    let state = NumberGame { value: 5, max_turn: true, turns_left: 6 };

    // --- Full minimax (exhaustive) ---
    let mm: AdversarialResult<i32> = minimax(&state, 6);
    println!("Minimax: best_move={:?}, value={}, nodes={}",
             mm.best_move, mm.value, mm.nodes_evaluated);

    // --- Alpha-Beta (same result, fewer nodes) ---
    let ab: AdversarialResult<i32> = alpha_beta(&state, 6);
    println!("Alpha-Beta: best_move={:?}, value={}, nodes={}",
             ab.best_move, ab.value, ab.nodes_evaluated);
    // ab.value == mm.value (always)
    // ab.nodes_evaluated <= mm.nodes_evaluated (usually much less)
}
```

### AdversarialResult fields

| Field | Type | Description |
|-------|------|-------------|
| `best_move` | `Option<M>` | The recommended move (None if terminal or no legal moves) |
| `value` | `f64` | The minimax value of the position |
| `nodes_evaluated` | `usize` | Total nodes visited during search |

### GameState trait

| Method | Signature | Purpose |
|--------|-----------|---------|
| `legal_moves` | `&self -> Vec<Move>` | All valid moves from this position |
| `apply_move` | `&self, &Move -> Self` | State after applying a move (immutable) |
| `is_terminal` | `&self -> bool` | Game over? |
| `is_maximizer_turn` | `&self -> bool` | Whose turn is it? |
| `evaluate` | `&self -> f64` | Static evaluation. Positive = good for maximizer. |

### Expectiminimax

For games with chance nodes (dice, card draws), implement `StochasticGameState`:

```rust
impl StochasticGameState for MyGameState {
    fn is_chance_node(&self) -> bool {
        self.next_is_dice_roll
    }

    fn chance_outcomes(&self) -> Vec<(f64, Self)> {
        // Return (probability, resulting_state) pairs
        vec![
            (1.0 / 6.0, self.with_roll(1)),
            (1.0 / 6.0, self.with_roll(2)),
            // ...
        ]
    }
}

let result = expectiminimax(&state, 8);
```

### Also available: Negamax

`negamax(&state, depth)` is a simplified alpha-beta implementation that uses score negation instead of separate max/min branches. Produces the same results, just cleaner code internally.

## When To Use This

| Situation | Algorithm | Why |
|-----------|-----------|-----|
| Small game tree (tic-tac-toe) | `minimax` | Solves the game completely |
| Medium game tree (checkers, Connect 4) | `alpha_beta` | Prunes enough to reach useful depth |
| Deterministic game, need perfect play | `alpha_beta` | Exact minimax value with pruning |
| Game with dice/cards (Backgammon) | `expectiminimax` | Handles chance nodes correctly |
| Huge game tree (Go, complex strategy) | MCTS instead | Alpha-beta cannot reach useful depth |
| Real-time (< 1ms) | `alpha_beta` at shallow depth | Deterministic, predictable timing |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `state` | `&S: GameState` | Current game position (passed by reference, not consumed) |
| `depth` | `usize` | How many moves ahead to search. Deeper = stronger but exponentially slower. |
| `alpha`, `beta` (alpha_beta only) | `f64` | Initial bounds. Use `f64::NEG_INFINITY` and `f64::INFINITY` at the root. The `alpha_beta` function handles this for you. |

### Choosing depth

The relationship between depth and nodes is exponential: `nodes ~ b^d` where `b` is the branching factor.

| Game | Branching factor | Depth 4 | Depth 6 | Depth 8 |
|------|-----------------|---------|---------|---------|
| Tic-tac-toe | ~4 | 256 | 4K | 65K |
| Checkers | ~8 | 4K | 260K | 17M |
| Chess | ~35 | 1.5M | 1.8B | too many |

Alpha-beta pruning roughly squares these: depth 8 with pruning costs about what depth 4 costs without it.

## Pitfalls

1. **Evaluate is not evaluate(terminal).** The `evaluate()` method is called at *leaf nodes* of the search -- both terminal states and states at the depth limit. For terminal states, return the true outcome. For non-terminal leaves, return your best static estimate. A bad evaluation function will make even deep search play poorly.

2. **Move ordering matters for alpha-beta.** Alpha-beta prunes best when good moves are tried first. If you examine the worst move first, no pruning happens and alpha-beta degrades to plain minimax. Sort `legal_moves()` by a quick heuristic (e.g., captures first in chess) for massive speedup.

3. **Depth is everything.** Going from depth 4 to depth 6 can transform a weak player into a strong one. If your search is slow, profile `successors` and `evaluate` -- those are the hot paths.

4. **Expectiminimax cannot prune as aggressively.** Chance nodes prevent the clean bounds that alpha-beta relies on. Expect expectiminimax to be significantly slower than alpha-beta at the same depth.

5. **Horizon effect.** At the depth limit, the AI cannot see what happens next. A move that looks good at depth 6 might be terrible at depth 7 (e.g., delaying an inevitable loss). Quiescence search (searching deeper in "unstable" positions) is the standard fix, but is not yet built into machin-search.

## Going Further

- **Iterative deepening:** Search at depth 1, then 2, then 3, etc., using time as the budget instead of depth. Each iteration reuses move ordering from the previous one.
- **Transposition table:** Cache evaluated positions (by hash) to avoid re-searching the same position reached via different move orders.
- **MCTS:** For games where alpha-beta cannot reach useful depth (Go, complex strategy games). See [MCTS](./mcts.md).
- **Negamax:** The `negamax(&state, depth)` function in machin-search implements alpha-beta using score negation, which is cleaner for two-player zero-sum games.
- Read: Knuth and Moore, "An Analysis of Alpha-Beta Pruning" (1975).
