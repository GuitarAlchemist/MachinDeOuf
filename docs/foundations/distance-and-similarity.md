# Distance & Similarity

> How to measure how "close" or "similar" two things are — the foundation of KNN, clustering, and recommendation systems.

## The Problem

You're building a music recommendation engine. A user likes Song A. You have 10,000 other songs in your catalog, each described by features: tempo, energy, danceability, loudness, acousticness. Which songs are most *similar* to Song A?

You need a way to measure the "distance" between any two songs in feature space. Different distance metrics give different answers — and choosing the right one matters more than most people realize.

## The Intuition

Think of each song as a point in space, where each feature is a dimension. Two songs that sound similar are "close together" in this space. The question is: how do you define "close"?

### Euclidean Distance: As the Crow Flies

The straight-line distance between two points. The Pythagorean theorem generalized to any number of dimensions.

```
Song A: [120 bpm, 0.8 energy, 0.7 dance]
Song B: [125 bpm, 0.75 energy, 0.65 dance]

Distance = √((120-125)² + (0.8-0.75)² + (0.7-0.65)²)
         = √(25 + 0.0025 + 0.0025)
         = √25.005 ≈ 5.0
```

**Problem**: BPM ranges from 60-200 but energy ranges from 0-1. The BPM dimension completely dominates. This is why you standardize features first.

### Manhattan Distance: City Blocks

The distance you'd walk in a city grid — you can only go horizontal or vertical, no diagonals.

```
Manhattan = |120-125| + |0.8-0.75| + |0.7-0.65|
          = 5 + 0.05 + 0.05 = 5.1
```

More robust to outliers than Euclidean because there's no squaring (outliers in one dimension don't dominate as much).

### Cosine Similarity: Direction, Not Magnitude

Measures the angle between two vectors, ignoring their length. Two vectors pointing the same direction have cosine similarity 1, perpendicular vectors have 0, opposite vectors have -1.

```
Song A: [120, 0.8, 0.7]
Song B: [240, 1.6, 1.4]  ← twice as large, but same direction!

Cosine similarity = 1.0 (identical direction)
Euclidean distance = large (very different magnitudes)
```

**When to use cosine**: When the *direction* (relative proportions) matters more than the *magnitude* (absolute values). This is common in text analysis (a long document has more words but the same topic) and recommendation systems.

### Chebyshev Distance: The Worst Dimension

The maximum difference across any single dimension:

```
Chebyshev = max(|120-125|, |0.8-0.75|, |0.7-0.65|) = 5
```

Useful when you care about the worst-case deviation in any one feature.

## How It Works

### Euclidean Distance

`d(a, b) = √(Σᵢ (aᵢ - bᵢ)²)`

In plain English: square the differences, add them up, take the square root. For comparing distances (not computing actual values), you can skip the square root — `euclidean_squared` is faster.

### Manhattan Distance

`d(a, b) = Σᵢ |aᵢ - bᵢ|`

In plain English: take absolute differences and add them. Also called L1 distance or taxicab distance.

### Minkowski Distance (Generalization)

`d(a, b) = (Σᵢ |aᵢ - bᵢ|^p)^(1/p)`

In plain English: Minkowski with p=1 is Manhattan, p=2 is Euclidean, p=∞ is Chebyshev. The parameter p controls how much you penalize large differences in a single dimension.

### Cosine Similarity

`cos(a, b) = (a · b) / (||a|| × ||b||)`

In plain English: the dot product of a and b, divided by the product of their lengths. This normalizes away the magnitude, leaving only the directional relationship.

**Cosine distance** = 1 - cosine similarity (so 0 means identical, 2 means opposite).

## In Rust

All distance functions live in `ix_math::distance`:

```rust
use ndarray::array;
use ix_math::distance;

let song_a = array![120.0, 0.8, 0.7, -5.0, 0.3];
let song_b = array![125.0, 0.75, 0.65, -6.0, 0.25];

// Euclidean (straight-line)
let d = distance::euclidean(&song_a, &song_b).unwrap();

// Euclidean squared (faster for comparisons — skip the sqrt)
let d2 = distance::euclidean_squared(&song_a, &song_b).unwrap();

// Manhattan (city blocks)
let m = distance::manhattan(&song_a, &song_b).unwrap();

// Cosine similarity (direction match, -1 to 1)
let cos = distance::cosine_similarity(&song_a, &song_b).unwrap();

// Cosine distance (0 = identical, 2 = opposite)
let cos_d = distance::cosine_distance(&song_a, &song_b).unwrap();

// Minkowski (generalized, p=3)
let mink = distance::minkowski(&song_a, &song_b, 3.0).unwrap();

// Chebyshev (maximum single-dimension difference)
let cheb = distance::chebyshev(&song_a, &song_b).unwrap();
```

### Finding Nearest Neighbors

Here's a practical example — finding the 3 most similar songs:

```rust
use ndarray::{array, Array1};
use ix_math::distance;

let query = array![120.0, 0.8, 0.7];
let catalog = vec![
    array![125.0, 0.75, 0.65],  // Song 1
    array![90.0, 0.3, 0.9],     // Song 2
    array![118.0, 0.82, 0.72],  // Song 3
    array![140.0, 0.9, 0.5],    // Song 4
];

// Compute distances and sort
let mut distances: Vec<(usize, f64)> = catalog.iter()
    .enumerate()
    .map(|(i, song)| (i, distance::euclidean(&query, song).unwrap()))
    .collect();

distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

// Top 3 nearest
for (idx, dist) in distances.iter().take(3) {
    println!("Song {}: distance = {:.4}", idx, dist);
}
```

> See [`examples/unsupervised/kmeans_clustering.rs`](../../examples/unsupervised/kmeans_clustering.rs) for clustering with distance metrics.

## When To Use This

| Distance Metric | Best For | Avoid When |
|----------------|----------|------------|
| **Euclidean** | General purpose; features on same scale | Features have very different scales (standardize first!) |
| **Euclidean squared** | Comparing distances (no need for actual value) | You need the actual distance value |
| **Manhattan** | High-dimensional data; sparse data; robust to outliers | Features are correlated |
| **Cosine** | Text similarity; when magnitude is irrelevant | Magnitude matters (e.g., actual prices) |
| **Minkowski (p)** | When you want to tune sensitivity | Unsure which p to use (start with p=2, i.e. Euclidean) |
| **Chebyshev** | When worst-case deviation in any dimension matters | Most ML tasks (too sensitive to single dimensions) |

## Key Parameters

| Parameter | What It Controls |
|-----------|-----------------|
| p (Minkowski) | How much to penalize large single-dimension differences. p=1 is Manhattan (tolerant), p=2 is Euclidean (balanced), p→∞ is Chebyshev (intolerant) |

## Pitfalls

- **Always standardize first.** If one feature ranges 0-1000 and another 0-1, Euclidean distance is dominated by the first feature. Use `linalg::standardize()` to normalize to zero mean and unit variance.
- **High dimensions are weird.** In high dimensions (100+ features), all points tend to be roughly equidistant — distance metrics become less meaningful. This is the "curse of dimensionality." Consider PCA or feature selection to reduce dimensions first.
- **Cosine similarity doesn't see magnitude.** Vectors [1, 2, 3] and [100, 200, 300] have cosine similarity 1.0. If magnitude matters (e.g., purchase amounts), use Euclidean or Manhattan instead.
- **Missing values break everything.** If some features are missing, distances are undefined. Either impute missing values or use a metric that handles them.

## Going Further

- **Uses this**: [KNN](../supervised-learning/knn.md) — classifies points by their nearest neighbors
- **Uses this**: [K-Means](../unsupervised-learning/kmeans.md) — clusters points by distance to centroids
- **Uses this**: [DBSCAN](../unsupervised-learning/dbscan.md) — density-based clustering using epsilon neighborhoods
- **Uses this**: [GPU Similarity Search](../gpu-computing/similarity-search.md) — GPU-accelerated cosine similarity at scale
- **Back to**: [INDEX](../INDEX.md)
