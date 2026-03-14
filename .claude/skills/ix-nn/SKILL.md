---
name: ix-nn
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
use ix_nn::layer::{Layer, Dense};
use ix_nn::loss::{mse_loss, binary_cross_entropy};
use ix_nn::positional::sinusoidal_encoding;
use ix_nn::attention::scaled_dot_product_attention;
```

## MCP Tool
Tool name: `ix_nn_forward`
Operations: `dense_forward`, `mse_loss`, `bce_loss`, `sinusoidal_encoding`
