# Backpropagation

> How neural networks learn — the chain rule applied backwards through layers to compute gradients efficiently.

## The Problem

You've built an MLP with thousands of weights. After a forward pass, you know the prediction is wrong — the loss is high. But which weights caused the error? And in which direction should you adjust each one?

Computing the gradient of the loss with respect to every weight individually (by numerical perturbation) would require millions of forward passes — impossibly slow. Backpropagation does it in a single backward pass.

## The Intuition

Think of an assembly line making a product. The final inspector finds a defect. To fix it, you need to trace backward through the line: "Was it the last station? The one before? The one before that?"

Each station (layer) receives blame proportional to how much it contributed to the error. The key insight: **you don't need to trace each station independently** — you can pass the "blame signal" backward through the chain, and each station calculates its own share.

This "blame signal" is the **gradient**. It flows backward from the loss, through each layer, accumulating according to the chain rule:

```
Loss → Output Layer → Hidden Layer 2 → Hidden Layer 1 → Input
  ←gradient flows backward←
```

### The Chain Rule: The Mathematical Core

If f(g(x)) is a composition of functions, then:

`df/dx = df/dg × dg/dx`

In plain English: the rate of change through a chain of operations is the product of each step's rate of change.

For three layers: `dLoss/dW₁ = dLoss/dOutput × dOutput/dHidden × dHidden/dW₁`

Each layer only needs to know two things:
1. What gradient is coming from above (dLoss/d(this layer's output))
2. Its own local gradient (how its output changes with its input and weights)

## How It Works

### Forward Pass (Saving Intermediate Values)

During the forward pass, each layer saves its input — this is needed for the backward pass.

For a dense layer: `output = W × input + b`

Save `input` in a cache.

### Backward Pass

Given `grad_output` (gradient of loss w.r.t. this layer's output), compute:

1. **Gradient w.r.t. weights**: `dL/dW = grad_output^T × cached_input`

   In plain English: how much should each weight change? Multiply the error signal by the input that weight saw.

2. **Gradient w.r.t. bias**: `dL/db = sum(grad_output)` along the batch dimension

   In plain English: the bias gradient is just the sum of the error signals.

3. **Gradient w.r.t. input** (to pass backward): `dL/d(input) = grad_output × W`

   In plain English: how much error to pass to the previous layer. Each input contributed proportionally to the weights it was multiplied by.

4. **Update weights**: `W -= learning_rate × dL/dW`

### Complete Example: 2-Layer Network

```
Forward:
  h = W₁ × x + b₁       (hidden layer, save x)
  h_relu = relu(h)        (activation, save h)
  y = W₂ × h_relu + b₂   (output layer, save h_relu)
  loss = MSE(y, target)

Backward:
  dL/dy = 2(y - target)/n           (MSE gradient)
  dL/dW₂ = dL/dy^T × h_relu        (output weights)
  dL/db₂ = sum(dL/dy)               (output bias)
  dL/dh_relu = dL/dy × W₂           (pass backward)
  dL/dh = dL/dh_relu ⊙ relu'(h)    (through activation: ⊙ is element-wise)
  dL/dW₁ = dL/dh^T × x             (hidden weights)
  dL/db₁ = sum(dL/dh)               (hidden bias)
```

The ⊙ symbol means element-wise multiplication. `relu'(h)` is 1 where h > 0, 0 elsewhere.

## In Rust

MachinDeOuf's `Dense` layer handles forward and backward passes:

```rust
use ndarray::array;
use machin_nn::layer::{Dense, Layer};
use machin_nn::loss::{mse_loss, mse_gradient, binary_cross_entropy, binary_cross_entropy_gradient};

// Network: 2 inputs → 4 hidden (ReLU) → 1 output (sigmoid)
let mut hidden = Dense::new(2, 4);
let mut output = Dense::new(4, 1);

let x = array![[0.5, -0.3]];       // Input (batch of 1)
let target = array![[1.0]];         // Target

let learning_rate = 0.01;

// Training loop
for epoch in 0..1000 {
    // Forward pass
    let h = hidden.forward(&x);
    let h_relu = h.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let pred = output.forward(&h_relu);

    // Loss
    let loss = mse_loss(&pred, &target);
    if epoch % 100 == 0 {
        println!("Epoch {}: loss = {:.6}", epoch, loss);
    }

    // Backward pass
    let grad = mse_gradient(&pred, &target);
    let grad_hidden = output.backward(&grad, learning_rate);

    // Apply ReLU derivative before passing gradient back
    let relu_grad = h.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    let grad_through_relu = &grad_hidden * &relu_grad;
    hidden.backward(&grad_through_relu, learning_rate);
}
```

### What Dense::backward Does Internally

The `backward` method on `Dense` handles steps 1-4 from above:

```rust
// Simplified view of what Dense::backward does:
fn backward(&mut self, grad_output: &Array2<f64>, lr: f64) -> Array2<f64> {
    let input = self.input_cache.as_ref().unwrap();  // Saved during forward

    // Gradient w.r.t. weights: dL/dW = input^T × grad_output
    let grad_weights = input.t().dot(grad_output);

    // Gradient w.r.t. bias: sum along batch dimension
    let grad_bias = grad_output.sum_axis(ndarray::Axis(0));

    // Gradient to pass backward: dL/d(input) = grad_output × W^T
    let grad_input = grad_output.dot(&self.weights.t());

    // Update
    self.weights = &self.weights - &(&grad_weights * lr);
    self.bias = &self.bias - &(&grad_bias * lr);

    grad_input
}
```

## When To Use This

Backpropagation is used whenever you train a neural network. There's no alternative that's as efficient:

| Method | Cost per Gradient | Practical? |
|--------|------------------|------------|
| **Backpropagation** | 1 forward + 1 backward pass | Yes — standard |
| **Numerical gradient** | 2 forward passes *per weight* | Testing only (way too slow for training) |
| **Finite differences** | Same as numerical | Same |

Use numerical gradients to *verify* your backprop implementation is correct (gradient checking), not for actual training.

## Key Parameters

| Parameter | What It Controls | Guidance |
|-----------|-----------------|----------|
| Learning rate | How big each weight update is | Too high → loss oscillates. Too low → training is slow. Start with 0.01 |
| Batch size | How many samples per gradient computation | Larger = more stable gradients but slower per step. 32-256 is common. |

## Pitfalls

- **Vanishing gradients**: In deep networks with sigmoid/tanh, gradients get multiplied through many layers. Each multiplication by a number < 1 shrinks the signal. Result: early layers learn almost nothing. Fix: use ReLU activation, He initialization, or residual connections.
- **Exploding gradients**: The opposite — gradients grow exponentially through layers. Result: weights become NaN. Fix: gradient clipping, careful initialization, lower learning rate.
- **Dead ReLU**: If a ReLU neuron always outputs 0, its gradient is always 0 and it never recovers. Fix: use Leaky ReLU or lower learning rate.
- **Not caching inputs**: The backward pass needs the forward pass's intermediate values. `Dense` caches automatically — don't run backward without a preceding forward.
- **Gradient checking**: When implementing custom layers, always verify with numerical gradients. If analytical and numerical gradients disagree by more than 1e-5, there's a bug.

## Going Further

- **Next**: [Loss Functions](loss-functions.md) — what the gradient is trying to minimize
- **Uses**: [Calculus Intuition](../foundations/calculus-intuition.md) — the chain rule explained
- **Uses**: [Gradient Descent](../optimization/gradient-descent.md) — the optimizer that applies the gradients
- **Before**: [Perceptron to MLP](perceptron-to-mlp.md) — the architecture backprop trains
