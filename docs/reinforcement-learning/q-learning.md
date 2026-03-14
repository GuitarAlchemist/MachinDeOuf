# Q-Learning

## The Problem

You are building an AI agent for a grid-world game. The agent starts in a corner and needs to find the goal in the opposite corner, avoiding obstacles. Unlike bandits (where you just pick between arms), here every action changes the *state* of the world -- moving left puts you in a different cell, where a different set of actions is available and a different reward awaits.

Real-world scenarios:
- **Robot navigation:** A warehouse robot learns to navigate from the loading dock to a shelf without colliding with other robots.
- **Game AI:** An NPC learns to play tic-tac-toe, chess openings, or simple video games.
- **Resource management:** A thermostat learns when to heat, cool, or idle to minimize energy cost while maintaining comfort.
- **Network routing:** A router learns which path to forward packets through to minimize latency.

## The Intuition

Imagine you move to a new city and want to find the fastest route to work. On day one, you pick randomly. Some routes are terrible, some are okay. Each day, you write down how long each route took. Over time, you stop trying the routes that always take 45 minutes and start favoring the one that takes 20 minutes.

But here is the subtlety: each route is not just *one* decision -- it is a *sequence* of turns. The value of turning left at the first intersection depends on what options are available after that turn. Q-learning captures this: it assigns a value not just to "turn left" in general, but to "turn left *when you are at intersection #5*." That state-action pair is what makes it more powerful than bandits.

## How It Works

Q-learning maintains a **Q-table**: a matrix where rows are states, columns are actions, and each cell Q(s, a) estimates the total future reward of taking action a in state s.

### The Q-Table Update Rule

After taking action a in state s, receiving reward r, and landing in state s':

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

**In plain English:** Compare what you *expected* (Q(s, a)) with what you *actually got* (r plus the best you can do from the next state). If reality was better than expected, nudge the estimate up. If worse, nudge it down.

Breaking down the terms:
- **alpha** (learning rate, 0 < alpha <= 1): How much to trust new information vs. old estimates. alpha = 0.1 means "update slowly," alpha = 1.0 means "completely overwrite."
- **gamma** (discount factor, 0 <= gamma < 1): How much to value future rewards. gamma = 0.99 means "future rewards are almost as good as immediate ones." gamma = 0 means "only care about the next step."
- **max_a' Q(s', a')**: The best value achievable from the next state -- this is the "look ahead" that makes Q-learning powerful.

### The Learning Loop

```
Initialize Q(s, a) = 0 for all states and actions
Repeat for each episode:
    s = starting state
    While s is not terminal:
        a = choose action (epsilon-greedy on Q)
        Take action a, observe reward r and next state s'
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        s = s'
```

**In plain English:** Play the game repeatedly. Each time you take a step, look at what happened and update your notes. Over many games, the Q-table converges to the true values, and the greedy policy (always pick argmax Q) becomes optimal.

## In Rust

> **Note:** Q-learning in `ix-rl` is currently a stub/TODO. The `Environment` and `Agent` traits are defined but the tabular Q-learning implementation is not yet built. Below is what the API will look like based on the trait definitions, along with a manual implementation you can use today.

### The Trait Definitions (available now)

```rust
use ix_rl::traits::{Environment, Agent};

// The Environment trait defines the world the agent interacts with
// trait Environment {
//     type State: Clone;
//     type Action: Clone;
//     fn reset(&mut self) -> Self::State;
//     fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool);
//     fn actions(&self) -> Vec<Self::Action>;
// }

// The Agent trait defines the learner
// trait Agent<E: Environment> {
//     fn select_action(&self, state: &E::State) -> E::Action;
//     fn update(&mut self, state: &E::State, action: &E::Action,
//               reward: f64, next_state: &E::State, done: bool);
// }
```

### Manual Q-Learning (using standard Rust)

```rust
use std::collections::HashMap;

struct QLearner {
    q_table: HashMap<(usize, usize), f64>,  // (state, action) -> value
    alpha: f64,    // learning rate
    gamma: f64,    // discount factor
    epsilon: f64,  // exploration rate
    n_actions: usize,
}

impl QLearner {
    fn new(n_actions: usize, alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            alpha, gamma, epsilon, n_actions,
        }
    }

    fn q(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| self.q(state, a)
                .partial_cmp(&self.q(state, b)).unwrap())
            .unwrap()
    }

    fn update(&mut self, s: usize, a: usize, r: f64, s_next: usize, done: bool) {
        let max_next = if done {
            0.0
        } else {
            (0..self.n_actions)
                .map(|a| self.q(s_next, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let old = self.q(s, a);
        let new_val = old + self.alpha * (r + self.gamma * max_next - old);
        self.q_table.insert((s, a), new_val);
    }
}
```

### Related: Q* Search in ix-search

While tabular Q-learning is not yet implemented, the `ix-search` crate offers **Q\* search** -- a pathfinding algorithm that uses learned Q-values as heuristics for A\* search. This is a different concept (search, not learning) but shares the idea of using Q-values to guide decisions:

```rust
use ix_search::qstar::{qstar_search, TabularQ};

// Q* uses a QFunction trait to estimate cost-to-go
let mut q = TabularQ::new(10.0);  // default heuristic value
let result = qstar_search(start_state, &q);
```

## When To Use This

| Approach | Best When | Handles Sequential Decisions | State Space |
|---|---|---|---|
| **Multi-armed bandits** | Single decision, no state changes | No | N/A |
| **Tabular Q-learning** | Small, discrete state/action spaces | Yes | < ~10,000 states |
| **Deep Q-learning (DQN)** | Large or continuous state spaces | Yes | Unlimited (needs neural net) |
| **Q\* search** | Known goal, need optimal path | Yes (search, not learning) | Graph-structured |

**Use Q-learning when:**
- The state space is small enough to enumerate (grid worlds, board games, simple control).
- You can simulate the environment cheaply (many episodes needed).
- You want an off-policy algorithm (learns the optimal policy even while exploring).

**Do not use Q-learning when:**
- The state space is continuous or very large (use function approximation / DQN instead).
- You need real-time decisions with no training phase.
- Actions have continuous values (use policy gradient methods instead).

## Key Parameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `alpha` (learning rate) | 0.01 -- 0.5 | Higher = faster learning, but more noise. Lower = stable but slow |
| `gamma` (discount factor) | 0.9 -- 0.99 | Higher = values long-term rewards. Lower = myopic (greedy) |
| `epsilon` (exploration rate) | 0.05 -- 0.3 | Higher = more exploration. Often decayed over time |
| `episodes` | 100 -- 100,000 | More episodes = better convergence, but more compute |

## Pitfalls

1. **Q-table explodes with state size.** A 100x100 grid with 4 actions needs 40,000 entries. A chess board has ~10^43 states -- tabular Q-learning cannot handle it. Use function approximation for large problems.

2. **Slow convergence.** Q-learning needs to visit every (state, action) pair many times. If some states are rarely reached, their Q-values stay at zero. Consider optimistic initialization (start Q-values high to encourage exploration).

3. **Non-stationary environments.** If the environment changes over time, old Q-values become stale. Use a higher learning rate or periodically reset the Q-table.

4. **Overestimation bias.** The `max` in the update rule causes a systematic upward bias. **Double Q-learning** fixes this by maintaining two Q-tables and using one to select and the other to evaluate.

5. **Off-policy vs. on-policy.** Q-learning is off-policy (learns the greedy policy while following an exploratory one). **SARSA** is the on-policy variant: it updates using the action actually taken, not the best action. SARSA is safer in environments where exploration is dangerous.

## Going Further

- **Bandits as a special case:** Q-learning with 1 state and no discounting reduces to the bandit update rule. See [multi-armed-bandits.md](./multi-armed-bandits.md).
- **Exploration strategies:** The epsilon-greedy exploration in Q-learning is the same as in bandits. See [exploration-vs-exploitation.md](./exploration-vs-exploitation.md) for alternatives.
- **Q\* search:** If you have a learned Q-function and want to find optimal paths, see the `ix-search` crate's `qstar` module, which uses Q-values as A\* heuristics.
- **Deep Q-Networks (DQN):** Replace the Q-table with a neural network from `ix-nn`. Not yet integrated, but the architecture is: state -> `ix-nn::Layer` -> Q-values for each action.
- **Policy gradient methods:** Instead of learning a value function, directly learn a policy. A fundamentally different approach, not yet in ix.
