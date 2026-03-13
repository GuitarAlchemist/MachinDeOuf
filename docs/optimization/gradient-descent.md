# Gradient Descent: SGD, Momentum, and Adam

> Walk downhill, step by step, until you reach the bottom of the valley. The only question is *how* you walk.

**Prerequisites:** [Calculus Intuition](../foundations/calculus-intuition.md), [Vectors & Matrices](../foundations/vectors-and-matrices.md)

---

## The Problem

You are building a house price predictor. Your model takes square footage, number of bedrooms, and distance to downtown, then outputs an estimated price. Internally it has a weight for each feature (plus a bias). When you start, those weights are random -- the model predicts nonsense. You need a systematic way to adjust them so the predictions get closer to the actual prices.

You have 10,000 labeled house sales. For any set of weights, you can compute a single number -- the mean squared error -- that tells you how wrong the model is overall. Your job: find the weights that make that number as small as possible. This is an optimization problem, and gradient descent is the workhorse that solves it.

---

## The Intuition

### SGD: One Careful Step at a Time

Imagine you are lost in fog on a hilly landscape and you want to reach the lowest valley. You cannot see farther than your feet. At every step, you feel the slope of the ground beneath you and take a step in the steepest downhill direction. That is gradient descent.

The **gradient** is a vector that points uphill. You move in the opposite direction. The **learning rate** controls how big your steps are. Too big and you overshoot the valley, bouncing back and forth. Too small and you inch along, taking forever.

**Stochastic** gradient descent (SGD) means you estimate the slope using a random subset (a "batch") of your data instead of all 10,000 houses. This is noisier but much faster, and the noise actually helps you escape shallow local dips that are not the true bottom.

### Momentum: A Rolling Ball

Plain SGD is jittery -- each step might zig-zag because the gradient changes direction. Momentum fixes this by giving the optimizer "inertia," like a heavy ball rolling downhill. The ball accumulates speed in directions it has been moving consistently and dampens oscillations in directions where the gradient keeps flipping.

If the gradient keeps pointing the same way, momentum accelerates you toward the minimum. If the gradient zig-zags (common in elongated valleys), momentum smooths out the path.

### Adam: Momentum with Per-Parameter Speed Control

Adam combines momentum with an additional idea: it tracks not just the average gradient direction (like momentum) but also how *large* the gradients have been for each parameter individually. Parameters that consistently receive large gradients get their learning rate turned down; parameters with small, rare gradients get theirs turned up.

Think of it as a team of hikers where each person adjusts their stride length independently based on the terrain they personally experienced. One hiker on flat ground takes bigger steps; another on a steep rocky slope takes smaller, more cautious ones.

---

## How It Works

### SGD Update Rule

```
params_new = params - lr * gradient
```

**In plain English:** Take the current parameters. Compute the gradient (which direction increases the error). Move a small step in the opposite direction. The learning rate `lr` controls the step size.

### Momentum Update Rule

```
v(t) = mu * v(t-1) - lr * gradient
params_new = params + v(t)
```

**In plain English:** The velocity `v` is a running average of past gradients. Each step, you blend the old velocity (scaled by momentum factor `mu`, typically 0.9) with the new gradient. You move in the direction of the accumulated velocity instead of the raw gradient. This is why a ball rolling downhill speeds up -- it remembers which direction it has been going.

- `mu = 0`: No memory, identical to SGD.
- `mu = 0.9`: Heavy momentum, smooth trajectory.
- `mu = 0.99`: Very heavy ball, slow to change direction.

### Adam Update Rules

```
m(t) = beta1 * m(t-1) + (1 - beta1) * gradient         # first moment (mean)
v(t) = beta2 * v(t-1) + (1 - beta2) * gradient^2        # second moment (variance)
m_hat = m(t) / (1 - beta1^t)                             # bias correction
v_hat = v(t) / (1 - beta2^t)                             # bias correction
params_new = params - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**In plain English:**

- `m(t)` is the exponential moving average of the gradient -- it tracks the direction like momentum does.
- `v(t)` is the exponential moving average of the squared gradient -- it tracks how big the gradients have been.
- The bias correction (`m_hat`, `v_hat`) fixes a startup problem: since `m` and `v` are initialized at zero, they are biased toward zero early on. Dividing by `(1 - beta^t)` compensates for this.
- The actual update divides the momentum term by the square root of the variance. If a parameter has been receiving large gradients, `sqrt(v_hat)` is large, and the effective step shrinks. If gradients have been small, the step grows. This per-parameter adaptive learning rate is what makes Adam so robust.

---

## In Rust

### Setting Up the Objective Function

Every optimization in MachinDeOuf starts with an `ObjectiveFunction`. This trait has three methods:

- `evaluate(&self, x) -> f64` -- compute the error for a given set of parameters
- `gradient(&self, x) -> Array1<f64>` -- compute the gradient (defaults to numerical differentiation if you don't override it)
- `dim(&self) -> usize` -- how many parameters

For quick experiments, use `ClosureObjective` to wrap a closure:

```rust
use machin_optimize::traits::{ClosureObjective, ObjectiveFunction, OptimizeResult};
use machin_optimize::convergence::ConvergenceCriteria;
use machin_optimize::gradient::{SGD, Momentum, Adam, minimize};
use ndarray::{array, Array1};

// Mean squared error for a simple linear model: price = w0 * sqft + w1 * beds + w2
// Given training data, the loss surface is a bowl -- gradient descent will find the bottom.
let objective = ClosureObjective {
    f: |w: &Array1<f64>| {
        // Simulated loss: (w0 - 0.15)^2 + (w1 - 50.0)^2 + (w2 - 100_000.0)^2
        // True optimum is [0.15, 50.0, 100_000.0]
        (w[0] - 0.15).powi(2)
            + (w[1] - 50.0).powi(2)
            + (w[2] - 100_000.0).powi(2)
    },
    dimensions: 3,
};
```

### Minimizing with SGD

```rust
let mut sgd = SGD::new(0.01); // learning rate = 0.01
let criteria = ConvergenceCriteria {
    max_iterations: 5000,
    tolerance: 1e-8, // stop when gradient norm falls below this
};

let result = minimize(&objective, &mut sgd, array![0.0, 0.0, 0.0], &criteria);

println!("SGD found: {:?}", result.best_params);
println!("Loss: {:.6}", result.best_value);
println!("Converged: {} in {} iterations", result.converged, result.iterations);
```

### Adding Momentum

```rust
let mut momentum = Momentum::new(0.01, 0.9); // lr=0.01, momentum=0.9
let criteria = ConvergenceCriteria {
    max_iterations: 5000,
    tolerance: 1e-8,
};

let result = minimize(&objective, &mut momentum, array![0.0, 0.0, 0.0], &criteria);
println!("Momentum found: {:?}", result.best_params);
// Expect faster convergence than plain SGD on elongated loss surfaces
```

### Using Adam

```rust
let mut adam = Adam::new(0.001)       // lower lr is typical for Adam
    .with_betas(0.9, 0.999);          // defaults -- usually no need to change

let criteria = ConvergenceCriteria {
    max_iterations: 10000,
    tolerance: 1e-10,
};

let result = minimize(&objective, &mut adam, array![0.0, 0.0, 0.0], &criteria);
println!("Adam found: {:?}", result.best_params);
println!("Loss: {:.10}", result.best_value);
```

### Comparing All Three

```rust
let initial = array![0.0, 0.0, 0.0];
let criteria = ConvergenceCriteria { max_iterations: 5000, tolerance: 1e-8 };

let optimizers: Vec<(&str, Box<dyn FnMut() -> OptimizeResult>)> = vec![
    ("SGD",      Box::new(|| minimize(&objective, &mut SGD::new(0.01), initial.clone(), &criteria))),
    ("Momentum", Box::new(|| minimize(&objective, &mut Momentum::new(0.01, 0.9), initial.clone(), &criteria))),
    ("Adam",     Box::new(|| minimize(&objective, &mut Adam::new(0.001), initial.clone(), &criteria))),
];

for (name, mut run) in optimizers {
    let r = run();
    println!("{:10} | iters: {:5} | loss: {:.6e} | converged: {}",
             name, r.iterations, r.best_value, r.converged);
}
```

### Understanding the Return Value

`minimize` returns an `OptimizeResult`:

| Field         | Type          | Meaning                                      |
|---------------|---------------|----------------------------------------------|
| `best_params` | `Array1<f64>` | The parameter vector with the lowest loss     |
| `best_value`  | `f64`         | The loss at `best_params`                     |
| `iterations`  | `usize`       | How many steps the optimizer took              |
| `converged`   | `bool`        | `true` if gradient norm dropped below `tolerance` |

### The Optimizer Trait

All gradient-based optimizers implement the same trait:

```rust
pub trait Optimizer {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &str;
}
```

This means you can write generic code that works with any optimizer, swap SGD for Adam in one line, or build your own optimizer and plug it in.

---

## When To Use This

| Situation | Recommended Optimizer |
|-----------|-----------------------|
| First attempt on a new problem | **Adam** -- works well out of the box with minimal tuning |
| Large-scale deep learning (millions of parameters) | **SGD + Momentum** -- often generalizes better than Adam at convergence |
| You need the simplest possible baseline | **SGD** -- easiest to understand and debug |
| Loss is noisy (small batches, reinforcement learning) | **Adam** -- adaptive learning rate handles noise well |
| Your loss surface is smooth and convex (linear regression) | **SGD** or **Momentum** -- they converge reliably, Adam may overshoot |
| You suspect the optimizer is too aggressive | Lower the learning rate, or switch from SGD to Adam which auto-adapts |

---

## Key Parameters

### Learning Rate (`lr`)

The single most important hyperparameter. Typical starting points:

- SGD: `0.01` to `0.1`
- Momentum: `0.01` to `0.1`
- Adam: `0.001` to `0.0001`

If the loss oscillates wildly, cut the learning rate by 10x. If the loss decreases painfully slowly, increase it.

### Momentum Factor (`momentum` in `Momentum::new`)

- `0.9` is the standard starting point. Almost nobody changes this.
- Higher values (0.95, 0.99) give more inertia, useful for very noisy gradients.
- Lower values (0.5) respond faster to gradient changes but lose the smoothing benefit.

### Adam Betas (`beta1`, `beta2`)

- `beta1 = 0.9` controls the first moment (gradient direction memory). Higher means longer memory.
- `beta2 = 0.999` controls the second moment (gradient magnitude memory). Higher means more stable per-parameter rates.
- The defaults are almost always fine. Change `beta2` only if you see training instability late in optimization.

### Convergence Criteria

- `max_iterations`: Safety net. Set generously (5,000 -- 100,000 depending on problem complexity).
- `tolerance`: Gradient norm threshold. When the gradient is this small, we are effectively at a flat point. `1e-6` to `1e-8` for most problems.

---

## Pitfalls

**Learning rate too high.** The loss jumps around or diverges. The fix is always the same: reduce `lr` by a factor of 10. This is the most common issue beginners hit.

**Learning rate too low.** The loss decreases but takes forever. You might hit `max_iterations` before converging. If `converged` is `false` and `best_value` is still improving, increase `max_iterations` or increase `lr`.

**Saddle points and local minima.** In high dimensions, SGD can get stuck at saddle points (flat in some directions). Momentum and Adam both help escape these because their accumulated velocity carries them through flat regions.

**Adam's generalization gap.** Research has shown that Adam sometimes converges to sharper minima than SGD with momentum. For deep learning, this can mean slightly worse test accuracy. The solution is often to start with Adam (for fast early progress) and switch to SGD with momentum for fine-tuning.

**Numerical gradient is slow.** The default `gradient()` implementation uses numerical differentiation (finite differences). This evaluates the objective function `2 * dim` times per step. For high-dimensional problems, implement the gradient analytically by overriding the `gradient` method on `ObjectiveFunction`.

**Parameters with very different scales.** If one parameter is typically ~0.01 and another is ~100,000, SGD and Momentum will struggle because the same learning rate is applied everywhere. Adam handles this naturally (per-parameter rates). Alternatively, normalize your features first.

---

## Going Further

- **See it in action:** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) shows optimization on the Rosenbrock function, a classic test problem with a curved valley that challenges gradient methods.
- **Gradient-free alternatives:** When you do not have gradients (discrete problems, black-box functions), try [Simulated Annealing](simulated-annealing.md) or [Particle Swarm](particle-swarm.md).
- **Where gradients come from:** [Calculus Intuition](../foundations/calculus-intuition.md) explains derivatives and numerical differentiation.
- **Neural network training:** [Backpropagation](../neural-networks/backpropagation.md) shows how gradients flow through layers of a network -- the chain rule in action.
- **Learning rate schedules:** A fixed learning rate is often suboptimal. Advanced strategies reduce the learning rate as training progresses (cosine annealing, warmup). These can be implemented by modifying `lr` between calls to `minimize`.
