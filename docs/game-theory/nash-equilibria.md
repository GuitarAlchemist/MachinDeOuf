# Nash Equilibria

## The Problem

Two competing ride-sharing companies are setting prices for the same city. If both set high prices, they split the market comfortably. If one undercuts the other, it captures most customers. If both undercut, they are in a price war where nobody profits.

Each company chooses a strategy. The outcome depends on *both* strategies together. Neither company can change its strategy to do better, *given what the other is doing*. That lock-in point is a **Nash equilibrium** -- the stable outcome where no player has an incentive to deviate unilaterally.

Nash equilibria appear everywhere: pricing strategy, network routing (where each packet picks the "best" path), evolutionary biology (stable population strategies), and arms races.

## The Intuition

Imagine two players sitting across a table. Each secretly writes down a strategy, then both reveal simultaneously. The payoffs depend on the combination.

A Nash equilibrium is a pair of strategies where both players look at the result and say: "Given what the other player did, I could not have done better." Neither player has regrets.

Key insight: a Nash equilibrium is not necessarily the *best* outcome for anyone. In the Prisoner's Dilemma, both players defecting is a Nash equilibrium -- but both would be better off if they cooperated. The equilibrium is *stable*, not *optimal*.

**Pure strategy:** each player picks one definite action. "Always defect."

**Mixed strategy:** each player randomizes. "Play Heads with probability 0.5, Tails with probability 0.5." Nash proved that every finite game has at least one equilibrium (possibly mixed).

## How It Works

### Bimatrix Game

A two-player game is defined by two payoff matrices:

- `payoff_a[i][j]` = Player A's payoff when A plays strategy `i` and B plays strategy `j`.
- `payoff_b[i][j]` = Player B's payoff in the same scenario.

### Best Response

Player A's **best response** to B's strategy is the strategy (or mix) that maximizes A's expected payoff:

```
BR_A(strategy_b) = argmax_i sum_j (strategy_b[j] * payoff_a[i][j])
```

**In plain English:** if I know what you are going to do (or your probability distribution), I pick whatever gives me the highest expected payoff.

### Nash Equilibrium

A strategy profile (strategy_a, strategy_b) is a Nash equilibrium if:

```
strategy_a is a best response to strategy_b
AND
strategy_b is a best response to strategy_a
```

**In plain English:** nobody can improve by switching, given what the other is doing.

### Finding equilibria

For 2x2 games, mixed equilibria can be found analytically using **indifference conditions**: Player B mixes so that Player A is indifferent between their strategies, and vice versa.

For Player B's mixing probability `q`:

```
payoff_a[0,0] * q + payoff_a[0,1] * (1-q) = payoff_a[1,0] * q + payoff_a[1,1] * (1-q)
```

**In plain English:** B randomizes so that A gets the same expected payoff no matter what A does. If A were not indifferent, A would have a strict preference, and B's mix would not be stable.

## In Rust

The `ix-game` crate provides bimatrix game analysis using `ndarray`:

```rust
use ix_game::nash::{
    BimatrixGame, StrategyProfile, fictitious_play, dominant_strategy_equilibrium,
};
use ndarray::{array, Array1};

fn main() {
    // --- Prisoner's Dilemma ---
    //          Cooperate    Defect
    // Coop     (3,3)        (0,5)
    // Defect   (5,0)        (1,1)
    let pd = BimatrixGame::new(
        array![[3.0, 0.0], [5.0, 1.0]],  // Player A's payoffs
        array![[3.0, 5.0], [0.0, 1.0]],  // Player B's payoffs
    );

    // Find dominant strategy equilibrium (both Defect).
    if let Some(dom) = dominant_strategy_equilibrium(&pd) {
        println!("Dominant strategy: A={:?}, B={:?}", dom.player_a, dom.player_b);
        // A=[0, 1], B=[0, 1]  -- both play Defect (index 1)
    }

    // --- Battle of the Sexes ---
    //          Opera     Football
    // Opera    (3,2)     (0,0)
    // Football (0,0)     (2,3)
    let bos = BimatrixGame::new(
        array![[3.0, 0.0], [0.0, 2.0]],
        array![[2.0, 0.0], [0.0, 3.0]],
    );

    // Support enumeration finds ALL Nash equilibria (pure + mixed).
    let equilibria = bos.support_enumeration();
    for (i, eq) in equilibria.iter().enumerate() {
        let pay_a = eq.expected_payoff_a(&bos);
        let pay_b = eq.expected_payoff_b(&bos);
        println!("NE {}: A={:?}, B={:?}, payoffs=({:.2}, {:.2})",
                 i, eq.player_a, eq.player_b, pay_a, pay_b);
    }
    // Finds: (Opera,Opera), (Football,Football), and a mixed equilibrium.

    // --- Best Response ---
    // If B plays 60% Opera, 40% Football:
    let b_strategy = Array1::from_vec(vec![0.6, 0.4]);
    let a_best = bos.best_response_a(&b_strategy);
    println!("A's best response to {:?}: {:?}", b_strategy, a_best);

    // --- Fictitious Play (iterative learning) ---
    let learned = fictitious_play(&pd, 1000);
    println!("Fictitious play converged to: A={:?}, B={:?}",
             learned.player_a, learned.player_b);

    // --- Zero-Sum Game (Matching Pennies) ---
    let mp = BimatrixGame::zero_sum(array![[1.0, -1.0], [-1.0, 1.0]]);
    let eq = mp.support_enumeration();
    // Finds the unique mixed NE: both play 50/50.

    // --- Check if a profile is a Nash Equilibrium ---
    let profile = StrategyProfile {
        player_a: Array1::from_vec(vec![0.0, 1.0]),  // Defect
        player_b: Array1::from_vec(vec![0.0, 1.0]),  // Defect
    };
    println!("Is NE? {}", pd.is_nash_equilibrium(&profile, 1e-8));
}
```

### API summary

| Function/Method | What it does |
|----------------|--------------|
| `BimatrixGame::new(payoff_a, payoff_b)` | Create a general two-player game |
| `BimatrixGame::zero_sum(payoff_a)` | Create a zero-sum game (B's payoff = -A's) |
| `game.best_response_a(&strategy_b)` | A's optimal (possibly mixed) response to B |
| `game.best_response_b(&strategy_a)` | B's optimal response to A |
| `game.support_enumeration()` | Find all Nash equilibria (exact, for small games) |
| `game.is_nash_equilibrium(&profile, tol)` | Check if a strategy profile is a NE |
| `dominant_strategy_equilibrium(&game)` | Find strictly dominant strategy NE (if one exists) |
| `fictitious_play(&game, iterations)` | Learn approximate NE through iterated best response |

See the full working example: [examples/game-theory/nash_equilibrium.rs](../../examples/game-theory/nash_equilibrium.rs)

## When To Use This

| Situation | Method | Why |
|-----------|--------|-----|
| 2x2 game, need all equilibria | `support_enumeration()` | Exact, fast for tiny games |
| Small game (up to ~5x5) | `support_enumeration()` | Enumerates all support pairs |
| Large game, approximate NE | `fictitious_play()` | Scalable, converges for many games |
| Check a known strategy | `is_nash_equilibrium()` | Quick verification |
| Zero-sum game | `BimatrixGame::zero_sum()` + any solver | Simpler structure, guaranteed unique value |
| Dominant strategy exists | `dominant_strategy_equilibrium()` | O(n*m) check, no enumeration needed |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `payoff_a` | `Array2<f64>` | Row player's payoff matrix (rows = A's strategies, cols = B's) |
| `payoff_b` | `Array2<f64>` | Column player's payoff matrix (same shape) |
| `tolerance` | `f64` | Numerical tolerance for equilibrium checks (1e-8 is typical) |
| `iterations` (fictitious play) | `usize` | Number of best-response rounds. More = closer to equilibrium. |

## Pitfalls

1. **Support enumeration is exponential.** It enumerates all subsets of strategies, so it is O(2^n * 2^m) for an n-by-m game. Fine for 5x5 games, impractical for 20x20. Use fictitious play for larger games.

2. **Mixed strategy NE solving is limited to 2x2.** The current `solve_support` implementation handles pure NE for any size but mixed NE only for 2x2 games. Larger mixed equilibria would need linear programming (not yet implemented).

3. **Fictitious play may not converge.** It is guaranteed to converge for zero-sum games and games with a unique NE. For games like Shapley's game (3x3 with cyclic best responses), it can cycle forever. Check convergence by comparing successive iterations.

4. **Multiple equilibria are common.** Battle of the Sexes has 3 equilibria; coordination games can have many more. The algorithm finds them all (via support enumeration), but your application must decide *which* one to use. This is the **equilibrium selection problem** -- game theory does not solve it for you.

5. **Nash equilibrium assumes rationality.** Real humans (and many AI agents) do not play Nash strategies. If you are modeling bounded-rational agents, consider evolutionary dynamics or level-k reasoning instead.

## Going Further

- **Evolutionary dynamics:** Model how a *population* converges to equilibrium over time. See [Evolutionary Dynamics](./evolutionary-dynamics.md).
- **Cooperative games (Shapley value):** When players can form coalitions and share payoffs. See [Shapley Value](./shapley-value.md).
- **Mechanism design (auctions):** Design the *rules of the game* so that the Nash equilibrium produces a desired outcome. See [Auction Mechanisms](./auction-mechanisms.md).
- **Correlated equilibrium:** A generalization of Nash where a mediator suggests strategies. More efficient to compute (linear programming).
- Read: Nisan et al., *Algorithmic Game Theory* (2007) -- the standard reference.
