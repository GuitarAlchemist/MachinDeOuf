# Matrix Multiply on the GPU

## The Problem

You are running batch inference on a neural network. The core operation --- forward-passing a batch of 256 inputs through a layer with 1,024 neurons --- is a matrix multiplication: `Output = Input * Weights`. On the CPU, a naive implementation of 256 x 1,024 x 768 = ~201 million multiply-adds takes hundreds of milliseconds. You need it done in single-digit milliseconds because you have dozens of layers and real-time latency requirements.

Other places matrix multiply dominates:

- **Transformer attention.** Q * K^T and attention-weights * V are both matrix multiplications.
- **Embedding lookup + projection.** Multiplying a one-hot (or sparse) input by an embedding matrix.
- **PCA / SVD computations.** Covariance matrices, projections, and reconstructions all reduce to matmul.
- **Batch similarity search.** Computing `Queries * Corpus^T` to get all pairwise dot products (see [similarity search](./similarity-search.md)).

---

## The Intuition

Matrix multiplication is the "hello world" of GPU computing because it maps perfectly to the GPU's strengths:

- Every element of the output matrix is **independent** --- it only depends on one row of A and one column of B.
- Each element requires the same amount of work (a dot product of length K).
- There are M * N output elements to compute, giving the GPU millions of independent tasks.

The GPU shader assigns one thread to each output element `C[row][col]`. That thread walks along the shared dimension K, multiplying `A[row][k] * B[k][col]` and accumulating. With a 16x16 thread grid per workgroup, 256 threads compute a 16x16 tile of the output simultaneously.

---

## How It Works

### The math

```
C = A * B

where:
  A is M x K  (M rows, K columns)
  B is K x N  (K rows, N columns)
  C is M x N  (M rows, N columns)

C[i][j] = sum(A[i][k] * B[k][j] for k in 0..K)
```

**In plain English:** each element of the output is the dot product of a row from A and a column from B.

### Row-major flat layout

Both CPU and GPU functions expect matrices as **flat `Vec<f32>` in row-major order**:

```
Matrix:     Flat array:
| 1 2 3 |   [1, 2, 3, 4, 5, 6]
| 4 5 6 |

Element [i][j] is at index: i * num_columns + j
```

### The WGSL shader

The shader receives three buffers (A, B, C) and a uniform buffer with the dimensions (M, N, K):

```wgsl
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= M || col >= N) { return; }

    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}
```

**In plain English:** each GPU thread computes exactly one element of the output matrix. The dispatcher launches `ceil(M/16) * ceil(N/16)` workgroups, each with 256 threads arranged in a 16x16 grid.

### Dispatch geometry

```
workgroups_x = ceil(M / 16)
workgroups_y = ceil(N / 16)
total_threads = workgroups_x * 16 * workgroups_y * 16
```

Threads whose `(row, col)` falls outside the matrix bounds early-return without writing.

---

## In Rust

### Basic matrix multiply

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::{matmul_gpu, matmul_cpu};

let ctx = GpuContext::new().unwrap();

// A is 2x3, B is 3x2 -> C is 2x2
//
//  A = | 1 2 3 |    B = | 7  8 |
//      | 4 5 6 |        | 9 10 |
//                        |11 12 |
let a = vec![1.0_f32, 2.0, 3.0,
             4.0,     5.0, 6.0];
let b = vec![ 7.0_f32,  8.0,
              9.0,     10.0,
             11.0,     12.0];

let (m, k, n) = (2, 3, 2);

// GPU path
let c_gpu = matmul_gpu(&ctx, &a, &b, m, k, n);
// c_gpu = [58.0, 64.0, 139.0, 154.0]
//
// C = | 1*7+2*9+3*11  1*8+2*10+3*12 | = | 58   64 |
//     | 4*7+5*9+6*11  4*8+5*10+6*12 |   |139  154 |

// CPU fallback (same API, same result)
let c_cpu = matmul_cpu(&a, &b, m, k, n);

// Verify they match
for (g, c) in c_gpu.iter().zip(c_cpu.iter()) {
    assert!((g - c).abs() < 1e-3);
}
```

### Batch neural network inference

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::matmul_gpu;

let ctx = GpuContext::new().unwrap();

let batch_size = 256;
let input_dim = 768;
let output_dim = 1024;

// Random input batch (256 x 768) and weight matrix (768 x 1024)
let inputs: Vec<f32> = (0..batch_size * input_dim)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();
let weights: Vec<f32> = (0..input_dim * output_dim)
    .map(|i| (i as f32 * 0.0007).cos() * 0.01)
    .collect();

// Forward pass: Output = Input * Weights
let output = matmul_gpu(&ctx, &inputs, &weights, batch_size, input_dim, output_dim);

assert_eq!(output.len(), batch_size * output_dim);  // 256 * 1024 = 262,144
println!("Batch inference complete: {} outputs", output.len());
```

### Using CPU fallback when no GPU is available

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::matmul::{matmul_gpu, matmul_cpu};

let a = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity
let b = vec![5.0_f32, 6.0, 7.0, 8.0]; // 2x2 matrix

let result = match GpuContext::new() {
    Ok(ctx) => matmul_gpu(&ctx, &a, &b, 2, 2, 2),
    Err(_)  => matmul_cpu(&a, &b, 2, 2, 2),
};

println!("Result: {:?}", result);  // [5.0, 6.0, 7.0, 8.0]
```

---

## When To Use This

| Matrix size (M x K x N) | GPU benefit | Notes |
|:--|:-:|:--|
| 10 x 10 x 10 | None | Transfer overhead dominates |
| 64 x 64 x 64 | Marginal | CPU SIMD is competitive |
| 256 x 768 x 1024 | Significant (5--20x) | Typical NN layer size |
| 1024 x 1024 x 1024 | Large (20--100x) | Clearly GPU territory |
| 4096 x 4096 x 4096 | Massive (50--200x) | GPU is the only practical option |

### `matmul_gpu` vs `matmul_cpu`

| | `matmul_gpu` | `matmul_cpu` |
|--|:-:|:-:|
| Precision | f32 | f32 |
| Parallelism | Thousands of GPU threads | Single CPU thread (no SIMD) |
| Setup cost | ~1ms (first call: shader compile) | None |
| Transfer cost | Proportional to data size | None |
| Best for | Large matrices (>= 256 x 256) | Small matrices or no GPU |

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `m` | Number of rows in A / rows in C | Batch size in neural network inference |
| `k` | Shared dimension (columns of A, rows of B) | Input dimension / hidden size |
| `n` | Number of columns in B / columns in C | Output dimension |
| Workgroup size | 16 x 16 = 256 threads per group | Hardcoded in shader; optimal for most GPUs |
| Data layout | Row-major, flat `Vec<f32>` | Element `[i][j]` of an `M x N` matrix is at index `i * N + j` |

---

## Pitfalls

1. **Flat arrays, not 2D arrays.** Both functions take `&[f32]` with explicit `m`, `k`, `n` dimensions. Passing the wrong dimensions will produce garbage (or panic on length assertions). Double-check that `a.len() == m * k` and `b.len() == k * n`.

2. **Row-major layout is assumed.** If you have column-major data (e.g., from Fortran or some linear algebra libraries), you must transpose before passing to these functions. Alternatively, swap the arguments: `matmul(B^T, A^T)` computes `(A * B)^T` in column-major.

3. **The CPU fallback is naive O(M*K*N).** It does not use SIMD, cache blocking, or multithreading. For production CPU workloads on large matrices, consider linking against an optimized BLAS. The provided `matmul_cpu` is intended as a correctness reference and fallback, not a high-performance CPU implementation.

4. **GPU memory limits.** Three matrices must fit in VRAM simultaneously: A (M*K*4 bytes), B (K*N*4 bytes), C (M*N*4 bytes). A 10,000 x 10,000 matrix is 400 MB. Check your GPU's VRAM before attempting very large multiplications.

5. **No in-place accumulation.** Each call allocates new GPU buffers, dispatches, and reads back. For iterated multiplications (e.g., chaining neural network layers), the overhead of repeated uploads and readbacks is significant. A future optimization would keep intermediate results in GPU memory between operations.

---

## Going Further

- **Tiled matrix multiply with shared memory** divides the computation into tiles that fit in the GPU's fast workgroup-local memory, dramatically improving cache locality. The current shader uses a simple per-element approach; tiling is a natural next optimization.
- **Fused operations** (e.g., matmul + bias + ReLU) reduce the number of GPU dispatches and memory round-trips. Common in neural network inference engines.
- **Half-precision (f16)** support would double throughput on GPUs with tensor cores (NVIDIA Ampere and later). WGPU's f16 support is evolving.
- The `ix-nn` crate uses matrix operations for neural network forward and backward passes. Integrating `matmul_gpu` as the backend would accelerate training and inference.
- **[Similarity Search](./similarity-search.md)** builds on `matmul_gpu` to compute batch similarity matrices in a single GPU pass.
