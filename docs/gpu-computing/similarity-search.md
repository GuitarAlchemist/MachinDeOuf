# Similarity Search on the GPU

## The Problem

You are building a real-time recommendation engine. Every time a user interacts with an item, you need to find the 10 most similar items from a catalog of 100,000 embedding vectors, each with 512 dimensions. A brute-force CPU search computes 100,000 cosine similarities --- that is 51.2 million multiply-adds per query. At 10 queries per second, your CPU is pegged. You need to offload this embarrassingly parallel workload to the GPU, where thousands of dot products run simultaneously.

Other places this pattern shows up:

- **Semantic search.** Given a query embedding, find the closest documents in a vector database.
- **Duplicate detection.** Compute the full pairwise similarity matrix for a set of items to identify near-duplicates.
- **Retrieval-augmented generation (RAG).** Find the top-k relevant chunks before feeding them to an LLM.

---

## The Intuition

Cosine similarity measures the angle between two vectors, ignoring their length. Two vectors pointing in exactly the same direction have similarity 1.0. Perpendicular vectors have similarity 0.0. Opposite vectors have similarity -1.0.

On a GPU, you compute cosine similarity by:
1. Computing the dot product of the two vectors (multiply each pair of elements, sum them up).
2. Computing the magnitude (length) of each vector.
3. Dividing: `dot(a, b) / (|a| * |b|)`.

The GPU shader runs all three accumulations in parallel across 256 threads using a **workgroup reduction**: each thread sums a slice of the vector, then threads cooperatively merge their partial sums.

For batch search, the trick is even better: flatten all vectors into a single matrix, compute the entire query-corpus dot product matrix with a single GPU matrix multiply, then normalize by norms. This computes similarities for **all queries against all corpus vectors** in one GPU pass.

---

## How It Works

### Cosine similarity

```
cosine(a, b) = dot(a, b) / (||a|| * ||b||)

where:
  dot(a, b)  = sum(a_i * b_i)
  ||a||      = sqrt(sum(a_i^2))
```

**In plain English:** multiply the vectors element-wise, sum the products, and divide by both vectors' lengths. The result tells you how aligned the vectors are, regardless of scale.

### GPU reduction pattern

The WGSL shader uses a parallel reduction:

1. Each of 256 threads computes a partial sum over its slice of the vector (grid-stride loop).
2. Partial sums are written to shared (workgroup) memory.
3. Threads cooperatively halve the array: thread 0 adds thread 128's value, thread 1 adds thread 129's value, and so on. Repeat until only thread 0 has the total.
4. Thread 0 writes the final dot product, norm_a^2, and norm_b^2 to the output buffer.

### Batch top-k via matrix multiply

For `nq` queries against `nc` corpus vectors of dimension `d`:

```
dot_matrix = Q (nq x d) * C^T (d x nc) = (nq x nc) dot products
similarity[i][j] = dot_matrix[i][j] / (||query_i|| * ||corpus_j||)
```

**In plain English:** treat the query and corpus vectors as matrices, multiply them (GPU-accelerated), then normalize each cell by the product of the relevant norms. You get every query-corpus similarity in one shot.

---

## In Rust

> Full runnable example: [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs)

### Pairwise cosine similarity

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::similarity::{cosine_similarity_gpu, cosine_similarity_cpu};

let ctx = GpuContext::new().expect("Need GPU");

let query   = vec![0.1_f32, 0.5, 0.3, 0.8];
let product = vec![0.2_f32, 0.4, 0.3, 0.7];

// GPU path (f32)
let sim_gpu = cosine_similarity_gpu(&ctx, &query, &product);
println!("GPU similarity: {:.4}", sim_gpu);

// CPU fallback (also f32 for API compatibility)
let sim_cpu = cosine_similarity_cpu(&query, &product);
println!("CPU similarity: {:.4}", sim_cpu);
```

### Dot product

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::similarity::dot_product_gpu;

let ctx = GpuContext::new().unwrap();

let a = vec![1.0_f32, 2.0, 3.0];
let b = vec![4.0_f32, 5.0, 6.0];

let dot = dot_product_gpu(&ctx, &a, &b);
println!("Dot product: {}", dot);  // 1*4 + 2*5 + 3*6 = 32
```

### Full similarity matrix

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::batch::similarity_matrix;

let ctx = GpuContext::new().unwrap();

let vectors = vec![
    vec![1.0_f32, 0.0, 0.0],   // "electronics"
    vec![0.0_f32, 1.0, 0.0],   // "clothing"
    vec![0.7_f32, 0.7, 0.0],   // "tech fashion"
];

// N x N similarity matrix in one GPU pass
let matrix = similarity_matrix(Some(&ctx), &vectors);

// matrix[0][2] ~ 0.707 (electronics partially similar to tech fashion)
// matrix[0][1] ~ 0.0   (electronics orthogonal to clothing)
for (i, row) in matrix.iter().enumerate() {
    println!("Vector {}: {:?}", i, row);
}
```

### Top-k search (recommendation engine)

```rust
use ix_gpu::batch::top_k_similar;

let query  = vec![0.1_f32, 0.5, 0.3, 0.8];
let corpus = vec![
    vec![0.2, 0.4, 0.3, 0.7],   // Product A
    vec![0.9, 0.1, 0.0, 0.2],   // Product B
    vec![0.1, 0.6, 0.2, 0.9],   // Product C
    vec![0.5, 0.5, 0.5, 0.5],   // Product D
];

// Find top-2 most similar products (CPU path shown; pass Some(&ctx) for GPU)
let results = top_k_similar(None, &query, &corpus, 2);

for (index, similarity) in &results {
    println!("Product {}: similarity {:.4}", index, similarity);
}
// Results sorted by similarity, descending
```

### Batch top-k (multiple queries at once)

```rust
use ix_gpu::context::GpuContext;
use ix_gpu::batch::batch_top_k;

let ctx = GpuContext::new().unwrap();

let queries = vec![
    vec![1.0_f32, 0.0, 0.0],  // User A's taste
    vec![0.0_f32, 1.0, 0.0],  // User B's taste
];
let corpus = vec![
    vec![0.9, 0.1, 0.0],  // Item 0
    vec![0.1, 0.9, 0.0],  // Item 1
    vec![0.5, 0.5, 0.0],  // Item 2
];

// Each user gets their own top-2 recommendations
let results = batch_top_k(Some(&ctx), &queries, &corpus, 2);

for (qi, recs) in results.iter().enumerate() {
    println!("User {}: {:?}", qi, recs);
}
// User 0: [(0, 0.99..), (2, 0.70..)]  -- prefers Item 0
// User 1: [(1, 0.99..), (2, 0.70..)]  -- prefers Item 1
```

---

## When To Use This

| Scenario | Recommended Function | Why |
|----------|---------------------|-----|
| Compare two vectors | `cosine_similarity_gpu` / `_cpu` | Direct pairwise comparison |
| Find top-k matches for one query | `top_k_similar` | Computes all corpus similarities, returns sorted |
| Find top-k for many queries at once | `batch_top_k` | Single GPU matrix multiply for all queries |
| Pairwise similarity for clustering | `similarity_matrix` | Full N x N matrix in one pass |
| Just need the dot product | `dot_product_gpu` | Skips normalization |

### GPU vs. CPU decision

| Corpus size | Dimensions | GPU benefit |
|:-:|:-:|:-:|
| < 1,000 | < 100 | Negligible (use CPU) |
| 1,000 -- 10,000 | 100 -- 512 | 2--5x speedup |
| 10,000+ | 512+ | 10--100x speedup |
| 100,000+ | 768+ | Mandatory for real-time |

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `ctx: Option<&GpuContext>` | GPU vs CPU path | Pass `Some(&ctx)` for GPU, `None` for CPU fallback |
| `k` (in top-k functions) | Number of results to return | Keep small (10-100) for search; use full matrix for clustering |
| Vector dimension | Length of each embedding | Higher dimensions = more work per pair = more GPU benefit |
| `f32` precision | All GPU paths use 32-bit floats | Sufficient for similarity search; differences from f64 are < 0.001 |

---

## Pitfalls

1. **GPU uses f32, CPU math crate uses f64.** If you are comparing GPU similarity results against values computed by `ix-math` (which uses `f64`), expect small discrepancies on the order of 1e-4 to 1e-6. This is fine for ranking but be cautious with exact-match thresholds.

2. **Short vectors do not benefit from GPU.** For vectors shorter than ~64 dimensions, the parallel reduction has more overhead than sequential summation. The CPU fallback will often be faster.

3. **The top-k sort happens on the CPU.** `top_k_similar` and `batch_top_k` compute similarities on the GPU but sort results on the CPU. For very large corpus sizes, this final sort can be a bottleneck.

4. **Memory limits.** A similarity matrix for 100,000 vectors is 100,000^2 * 4 bytes = ~37 GB. For large datasets, use `batch_top_k` (which computes a queries x corpus matrix, not corpus x corpus) or tile the computation.

5. **First call is slow.** Shader compilation happens on the first invocation. Subsequent calls reuse the compiled pipeline and are much faster. Consider a warm-up call with dummy data during initialization.

---

## Going Further

- **Approximate nearest neighbor (ANN) indices** like HNSW or IVF-PQ trade a small amount of recall for orders-of-magnitude faster search. Combine with GPU-accelerated exact reranking of the top candidates.
- **Euclidean distance** is available as `euclidean_distance_cpu` for cases where you care about absolute distance rather than angular similarity. A GPU version could be added by adapting the cosine shader.
- **[Matrix Multiply on GPU](./matrix-multiply-gpu.md)** is the building block that makes `batch_top_k` and `similarity_matrix` fast. Read that doc to understand the tiled compute shader under the hood.
- **[Intro to GPU Compute](./intro-to-gpu-compute.md)** covers the WGPU fundamentals --- how `GpuContext` works, buffer management, and the dispatch model.
