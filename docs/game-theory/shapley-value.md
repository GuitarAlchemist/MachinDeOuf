# Shapley Value

## The Problem

Three departments at a company share a cloud computing cluster. The IT team alone would need a $100K server. Marketing alone would need $80K. Sales alone would need $90K. But sharing a single $200K cluster serves all three (instead of $270K separately). The company saves $70K -- but how should the cost be split?

If you split it equally ($66.7K each), Marketing complains -- their standalone cost was only $80K, so they are barely saving anything. If you split by standalone cost, IT pays the most but might argue they bring the biggest workloads.

The **Shapley value** is a principled answer: each department pays its average *marginal contribution* across all possible orderings of departments joining the coalition. It is the unique cost allocation that satisfies fairness axioms (symmetry, efficiency, linearity, and null player).

The same math is used for:
- **ML feature importance:** how much does each feature contribute to a model's prediction? (SHAP values are Shapley values.)
- **Voting power:** how much influence does each party have in a coalition government?
- **Network reliability:** which node's failure would hurt the network most?

## The Intuition

Imagine the three departments arrive one at a time in a random order to set up the shared cluster. The first to arrive pays the full cost of what it needs. Each subsequent arrival pays only the *additional* cost their presence adds.

If IT arrives first, they pay $100K. If Marketing arrives second, the cluster needs to grow from $100K to $150K, so Marketing pays $50K. If Sales arrives last, it grows from $150K to $200K, so Sales pays $50K.

But the order matters. In a different order, the costs change. The Shapley value is the **average of each player's marginal contribution across all possible arrival orders**. With 3 players, there are 3! = 6 orderings.

This averaging is what makes it fair:
- **Symmetry:** players who contribute equally pay equally.
- **Efficiency:** the total Shapley values add up to the grand coalition's value (no money left over or missing).
- **Null player:** a player who adds nothing to any coalition pays nothing.

## How It Works

For a game with `n` players and a characteristic function `v(S)` (the value of coalition S):

```
phi_i = sum over S not containing i:
    |S|! * (n - |S| - 1)! / n! * [v(S union {i}) - v(S)]
```

| Symbol | Meaning |
|--------|---------|
| `phi_i` | Shapley value of player `i` |
| `S` | A coalition (subset of players) not containing `i` |
| `v(S)` | Value that coalition `S` can achieve alone |
| `v(S union {i}) - v(S)` | Player `i`'s **marginal contribution** to coalition `S` |
| `\|S\|! * (n - \|S\| - 1)! / n!` | Weight = probability of `S` being "already there" in a random ordering |

**In plain English:** for every possible group that could have formed before player `i` arrives, calculate how much `i` adds to that group. Average these contributions across all possible groups, weighted by how likely each group is in a random arrival order.

### Coalition representation

Coalitions are stored as **bitmasks**: player `i` is in the coalition if bit `i` is set. With a `u64` bitmask, this supports up to **63 players**. The empty coalition is `0`, and the grand coalition (all players) is `(1 << n) - 1`.

Example with 3 players:
- `0b000 = 0` -- empty coalition
- `0b001 = 1` -- player 0 alone
- `0b011 = 3` -- players 0 and 1
- `0b111 = 7` -- grand coalition (all three)

## In Rust

The `machin-game` crate provides cooperative game theory using bitmask-based coalitions:

```rust
use machin_game::cooperative::{CooperativeGame, weighted_voting_game};

fn main() {
    // --- Cost allocation example ---
    // Three departments sharing infrastructure.
    let mut game = CooperativeGame::new(3);

    // Set values using player index slices (converted to bitmasks internally).
    game.set_value_for(&[0], 100.0);         // IT alone
    game.set_value_for(&[1], 80.0);          // Marketing alone
    game.set_value_for(&[2], 90.0);          // Sales alone
    game.set_value_for(&[0, 1], 150.0);      // IT + Marketing
    game.set_value_for(&[0, 2], 160.0);      // IT + Sales
    game.set_value_for(&[1, 2], 140.0);      // Marketing + Sales
    game.set_value_for(&[0, 1, 2], 200.0);   // Grand coalition

    // Or equivalently, using raw bitmasks:
    // game.set_value(0b001, 100.0);  // Player 0
    // game.set_value(0b011, 150.0);  // Players 0 and 1
    // game.set_value(0b111, 200.0);  // All players

    // --- Shapley Value ---
    let shapley: Vec<f64> = game.shapley_value();
    println!("Shapley values: IT={:.1}, Marketing={:.1}, Sales={:.1}",
             shapley[0], shapley[1], shapley[2]);
    // Values sum to 200.0 (grand coalition value).

    // --- Core membership ---
    // Check if a proposed allocation is stable (no coalition wants to break away).
    let proposal = vec![75.0, 55.0, 70.0]; // sums to 200
    println!("In core? {}", game.is_in_core(&proposal));

    // The Shapley value itself may or may not be in the core.
    println!("Shapley in core? {}", game.is_in_core(&shapley));

    // --- Superadditivity check ---
    println!("Superadditive? {}", game.is_superadditive());

    // --- Banzhaf Power Index ---
    // Measures voting power: how often is each player a swing voter?
    let banzhaf: Vec<f64> = game.banzhaf_index();
    println!("Banzhaf index: {:?}", banzhaf);

    // --- Weighted Voting Game ---
    // Model a parliament: parties have seats, need majority (> 50%) to pass laws.
    let voting_game = weighted_voting_game(
        &[45.0, 30.0, 15.0, 10.0],  // Party seat counts
        51.0,                         // Majority quota
    );
    let power = voting_game.shapley_value();
    println!("Voting power: {:?}", power);
    // Often surprising: a party with 15% of seats may have
    // much more or less than 15% of the power.

    let banzhaf_power = voting_game.banzhaf_index();
    println!("Banzhaf power: {:?}", banzhaf_power);
}
```

### API summary

| Method | Signature | What it does |
|--------|-----------|--------------|
| `CooperativeGame::new(n)` | `usize -> Self` | Create a game with `n` players (max 63) |
| `set_value(coalition, value)` | `u64, f64` | Set `v(S)` using a bitmask |
| `set_value_for(&[usize], value)` | `&[usize], f64` | Set `v(S)` using player indices |
| `value(coalition)` | `u64 -> f64` | Get `v(S)` (returns 0.0 for unset coalitions) |
| `grand_coalition()` | `-> u64` | Bitmask for all players |
| `shapley_value()` | `-> Vec<f64>` | Compute Shapley value for each player |
| `is_in_core(&allocation)` | `&[f64] -> bool` | Check if allocation is in the core |
| `is_superadditive()` | `-> bool` | Check if merging coalitions always helps |
| `banzhaf_index()` | `-> Vec<f64>` | Compute normalized Banzhaf power index |
| `weighted_voting_game(&weights, quota)` | `-> CooperativeGame` | Create a voting game |

## When To Use This

| Situation | Method | Why |
|-----------|--------|-----|
| Fair cost/profit allocation | `shapley_value()` | Unique allocation satisfying fairness axioms |
| ML feature importance (SHAP) | `shapley_value()` | Industry standard for explainability |
| Check if an allocation is stable | `is_in_core()` | No coalition wants to deviate |
| Voting power analysis | `weighted_voting_game()` + `shapley_value()` | Reveals true power vs. raw seat count |
| Power index comparison | `banzhaf_index()` | Alternative power measure, simpler weights |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `num_players` | `usize` | Number of players (max 63 due to u64 bitmask) |
| `coalition` | `u64` | Bitmask representing a set of players |
| `allocation` | `&[f64]` | Proposed payoff vector (one entry per player) |
| `weights` (voting game) | `&[f64]` | Voting weight of each player |
| `quota` (voting game) | `f64` | Threshold to win (e.g., 51 for simple majority) |

## Pitfalls

1. **Exponential complexity.** Computing the exact Shapley value requires iterating over all `2^n` coalitions for each of `n` players. With 20 players, that is 20 million coalitions. With 30, it is 30 billion. For large games, use sampling-based approximations (not yet in machin-game).

2. **63-player limit.** The bitmask representation uses `u64`, so the maximum number of players is 63. This is fine for most game-theoretic applications but too small for SHAP values on high-dimensional ML models.

3. **Unset coalitions default to 0.** If you forget to set `v(S)` for some coalition, it defaults to 0.0. This is correct for many games (where small coalitions cannot achieve anything) but can silently produce wrong results if you meant to set a nonzero value.

4. **Shapley value may not be in the core.** The Shapley value is always efficient (sums to `v(N)`) but may not satisfy coalition rationality. Some coalitions might prefer to break away and do better alone. Use `is_in_core()` to check.

5. **Voting game power is counterintuitive.** In a parliament with seats [50, 49, 1], the third party with 1 seat has zero Shapley power (it is never a swing voter if the quota is 51). But with seats [49, 49, 2] and quota 51, the tiny party has equal power to the large ones. Always compute, never guess.

## Going Further

- **SHAP values for ML:** The connection between Shapley values and ML feature importance. Each feature is a "player," and the "coalition value" is the model's prediction using that subset of features. Libraries like SHAP use this framework.
- **Sampling-based Shapley:** For large `n`, approximate the Shapley value by sampling random orderings instead of enumerating all coalitions.
- **Nucleolus:** The unique allocation that minimizes the worst-case unhappiness of any coalition. Not yet implemented but builds on the same `CooperativeGame` structure.
- **Nash equilibria:** Non-cooperative game theory where players act independently. See [Nash Equilibria](./nash-equilibria.md).
- Read: Shapley, "A Value for n-Person Games" (1953) -- the original paper, remarkably readable.
