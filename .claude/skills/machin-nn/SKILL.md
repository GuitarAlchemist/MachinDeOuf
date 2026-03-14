---
name: machin-nn
description: Neural network operations — dense layers, loss functions, positional encodings
---

# Neural Networks

Forward pass through neural network components.

## When to Use
When the user needs to compute a dense layer forward pass, evaluate loss functions (MSE, BCE), or generate positional encodings for transformer models.

## Capabilities
- **Dense layer forward** — Fully-connected layer with Xavier initialization
- **MSE loss** — Mean squared error between predicted and target
- **BCE loss** — Binary cross-entropy for classification
- **Sinusoidal encoding** — Vaswani-style positional encoding for transformers

## Programmatic Usage
```rust
use machin_nn::layer::{Layer, Dense};
use machin_nn::loss::{mse_loss, binary_cross_entropy};
use machin_nn::positional::sinusoidal_encoding;
use machin_nn::attention::scaled_dot_product_attention;
```

## MCP Tool
Tool name: `machin_nn_forward`
Operations: `dense_forward`, `mse_loss`, `bce_loss`, `sinusoidal_encoding`
