# From Perceptron to MLP

> A single neuron can only draw a straight line. Stack them into layers and they can learn anything.

## The Problem

You're building a handwritten digit recognizer. Each image is 28x28 pixels = 784 numbers. Given these 784 pixel intensities, classify the image as 0-9.

Linear models can't do this — the relationship between raw pixels and digits is deeply nonlinear. A "7" and a "1" can share many pixel positions, but differ in subtle patterns that no straight line can separate. You need a model that builds up complex patterns from simple ones.

## The Intuition

### The Perceptron: A Single Neuron

A perceptron takes inputs, multiplies each by a weight, adds them up, and passes the result through an activation function.

```
inputs: [x1, x2, x3]
weights: [w1, w2, w3]
bias: b

output = activation(w1*x1 + w2*x2 + w3*x3 + b)
```

Think of it as a vote. Each input is a voter, each weight is how much their opinion matters, and the activation function makes the final yes/no decision.

A single perceptron can only learn **linear** boundaries — it draws a straight line (or hyperplane) to separate classes. It handles AND and OR, but famously fails on XOR.

### Multi-Layer Perceptron (MLP): Stacking Neurons

Stack perceptrons into layers and something magical happens — the network can learn any continuous function (given enough neurons).

```
Input Layer     Hidden Layer     Output Layer
[784 inputs] → [128 neurons] → [10 outputs]
   (pixels)    (learned features)  (digits 0-9)
```

**Hidden layers learn features automatically.** The first hidden layer might learn to detect edges. The second might combine edges into curves and corners. The output layer combines these into digit recognitions.

Each connection has a weight. Training adjusts these weights so the network's predictions match the correct answers.

### Activation Functions: Adding Nonlinearity

Without activation functions, stacking layers would just be one big linear function (useless — equivalent to a single layer). Activations add nonlinearity, which is what allows deep networks to model complex patterns.

- **ReLU**: `max(0, x)` — most common. Simple, fast, works well in practice. Outputs 0 for negative inputs, passes positive inputs through unchanged.
- **Sigmoid**: `1 / (1 + e^(-x))` — squashes output to (0, 1). Used for probability outputs. Can cause vanishing gradients in deep networks.
- **Tanh**: `(e^x - e^(-x)) / (e^x + e^(-x))` — like sigmoid but outputs (-1, 1). Zero-centered, which helps training.
- **Softmax**: Converts a vector of scores into probabilities that sum to 1. Used in the output layer for multi-class classification.

## How It Works

### Forward Pass

Data flows from input to output, layer by layer:

1. **Input**: Raw features (e.g., 784 pixel values)
2. **Hidden layer computation**: `h = activation(W₁ × x + b₁)`
   - W₁ is a weight matrix (128×784 for 784→128)
   - b₁ is a bias vector (128 elements)
   - activation is applied element-wise
3. **Output layer**: `ŷ = softmax(W₂ × h + b₂)`
   - W₂ is 10×128, b₂ is 10 elements
   - Softmax turns raw scores into probabilities

In plain English: multiply inputs by weights, add bias, apply nonlinearity. Repeat for each layer. The final output is the prediction.

### Training: Adjusting Weights

1. **Forward pass**: Compute prediction
2. **Loss**: Measure how wrong the prediction is (e.g., cross-entropy)
3. **Backward pass**: Compute gradients of loss w.r.t. every weight (backpropagation)
4. **Update**: Nudge each weight in the direction that reduces loss

This is gradient descent applied to neural networks. See [Backpropagation](backpropagation.md) for the details.

### Weight Initialization

How you set the initial weights matters enormously:

- **Xavier initialization**: std = √(2 / (fan_in + fan_out)). Good for sigmoid/tanh networks.
- **He initialization**: std = √(2 / fan_in). Designed for ReLU networks.
- **Zeros**: Never do this — all neurons learn the same thing (symmetry problem).

## In Rust

MachinDeOuf provides the building blocks in `machin-nn`:

```rust
use ndarray::array;
use machin_nn::layer::{Dense, Layer};
use machin_nn::loss::{mse_loss, mse_gradient};
use machin_math::activation::{relu_array, sigmoid_array, softmax};

// Build a simple 2-layer network: 3 inputs → 4 hidden → 2 outputs
let mut hidden = Dense::new(3, 4);   // Xavier-initialized
let mut output = Dense::new(4, 2);

// Forward pass
let input = array![[1.0, 0.5, -0.3]];  // batch of 1 sample, 3 features
let h = hidden.forward(&input);         // shape: (1, 4)
let h_activated = h.mapv(|x| if x > 0.0 { x } else { 0.0 }); // ReLU
let prediction = output.forward(&h_activated);  // shape: (1, 2)

println!("Prediction: {:?}", prediction);

// Compute loss
let target = array![[1.0, 0.0]];  // target output
let loss = mse_loss(&prediction, &target);
println!("Loss: {}", loss);

// Backward pass (learning rate = 0.01)
let grad = mse_gradient(&prediction, &target);
let grad_hidden = output.backward(&grad, 0.01);
hidden.backward(&grad_hidden, 0.01);
```

### Weight Initialization Options

```rust
use machin_nn::initializers;

let xavier_weights = initializers::xavier(784, 128);  // For sigmoid/tanh
let he_weights = initializers::he(784, 128);           // For ReLU
let zero_weights = initializers::zeros(784, 128);      // Don't use this
```

### Activation Functions

```rust
use ndarray::array;
use machin_math::activation;

let x = array![1.0, -0.5, 0.0, 2.0];

let relu = activation::relu_array(&x);       // [1.0, 0.0, 0.0, 2.0]
let sigmoid = activation::sigmoid_array(&x);  // [0.73, 0.38, 0.50, 0.88]
let tanh = activation::tanh_array(&x);        // [0.76, -0.46, 0.0, 0.96]

let scores = array![2.0, 1.0, 0.1];
let probs = activation::softmax(&scores);     // [0.66, 0.24, 0.10] — sums to 1.0
```

## When To Use This

| Model | Best For | Limitations |
|-------|----------|-------------|
| **Single perceptron** | Linearly separable problems | Can't handle nonlinear patterns |
| **MLP (1-2 hidden layers)** | Tabular data, moderate complexity | Needs more data than linear models |
| **Deep MLP (3+ layers)** | Complex patterns, large datasets | Slow to train, needs careful tuning |
| **Linear/Logistic Regression** | Simple relationships, small data | Can't capture nonlinearity |

MLPs are a good middle ground — more powerful than linear models, simpler than deep learning architectures (CNNs, Transformers).

## Key Parameters

| Parameter | What It Controls | Guidance |
|-----------|-----------------|----------|
| Hidden layer size | Model capacity | Start with 64-256. More neurons = more capacity but slower and more prone to overfitting |
| Number of hidden layers | Depth of feature hierarchy | 1-2 layers for most tabular data. Rarely need more than 3. |
| Activation function | Nonlinearity type | ReLU for hidden layers (default). Sigmoid for binary output. Softmax for multi-class output. |
| Learning rate | Step size during training | Start with 0.01. If loss oscillates, decrease. If stuck, increase. |
| Initialization | Starting weights | He for ReLU, Xavier for sigmoid/tanh |

## Pitfalls

- **Don't use zero initialization.** All neurons start identical and learn the same thing forever (symmetry breaking problem). Always use Xavier or He.
- **ReLU neurons can "die."** If a neuron's output is always negative, ReLU locks it to 0 and it never recovers. Use Leaky ReLU (`leaky_relu(x, 0.01)`) if this happens.
- **More layers ≠ better.** For tabular data, 1-2 hidden layers usually suffice. Deeper networks need techniques like batch normalization and residual connections.
- **Standardize inputs.** Neural networks train much faster when inputs are centered around 0 with similar scales.
- **Watch the loss curve.** If training loss decreases but validation loss increases, you're overfitting. Reduce model size or add regularization.

## Going Further

- **Next**: [Backpropagation](backpropagation.md) — how the network learns by computing gradients
- **Next**: [Loss Functions](loss-functions.md) — choosing the right objective
- **Foundation**: [Calculus Intuition](../foundations/calculus-intuition.md) — derivatives power backpropagation
- **Foundation**: [Vectors & Matrices](../foundations/vectors-and-matrices.md) — matrix multiplication is the core operation
