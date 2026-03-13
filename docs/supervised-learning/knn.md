# k-Nearest Neighbors (KNN)

## The Problem

You run a streaming music service and want to recommend new songs to users. Each song can be described by measurable features: tempo (BPM), energy level, danceability, acousticness, and valence (positivity). When a user finishes a song they liked, you want to find the most similar songs in your catalog and recommend those.

The intuition is dead simple: similar things are near each other. If a user loves an upbeat 120-BPM pop song with high danceability, they'll probably enjoy other upbeat, danceable songs at similar tempos. You don't need to learn a complex model -- you just need to find the nearest neighbors in feature space.

KNN formalizes this intuition. It stores the entire training dataset, and when asked to classify a new point, it finds the k closest training points, looks at their labels, and goes with the majority vote. No training phase, no learned parameters, no assumptions about the shape of the data. Just "tell me what the neighbors say."

## The Intuition

Imagine moving to a new neighborhood and wondering whether you'll like the local restaurants. You ask your five nearest neighbors (literally, the five people who live closest to you) what they think. If four out of five say "the Thai place is great," you'd probably try it. That's KNN with k=5.

The key insight is that proximity in feature space implies similarity. Two songs with similar tempo, energy, and danceability are "close" to each other, and users who like one tend to like the other.

The choice of k matters. With k=1, you're asking only your immediate neighbor -- their opinion might be an outlier. With k=100, you're asking the whole neighborhood, which smooths out individual quirks but might include people who live so far away that their taste is irrelevant. k=3 to k=10 is typically the sweet spot.

KNN is a "lazy learner" -- it does no work during training (just stores the data) and does all the work during prediction (computing distances to every training point). This makes training instant but prediction slow on large datasets.

## How It Works

### Step 1: Store the training data

```
fit(X_train, y_train) -> store X_train and y_train
```

In plain English, this means: memorize all the songs and their labels. There is no model to learn -- the training data *is* the model.

### Step 2: Compute distances to all training points

For a new query point x_q, compute the Euclidean distance to every training point:

$$
d(x_q, x_i) = \sqrt{\sum_{j=1}^{p} (x_{q,j} - x_{i,j})^2}
$$

In plain English, this means: measure how different the new song is from every song in the catalog, feature by feature, then combine those differences into a single number. The smaller the number, the more similar the songs.

### Step 3: Find the k nearest neighbors

Sort all training points by distance and take the k closest.

In plain English, this means: out of your entire catalog, find the k songs that are most similar to the query song.

### Step 4: Vote

$$
\hat{y} = \text{mode}\{y_{i_1}, y_{i_2}, \ldots, y_{i_k}\}
$$

In plain English, this means: look at the labels of the k nearest songs. If 4 out of 5 are in the "liked" category, predict "liked." The majority wins.

### Step 5: Estimate probabilities

$$
P(\text{class} = c) = \frac{\text{count of class } c \text{ among k neighbors}}{k}
$$

In plain English, this means: if 3 of 5 neighbors are "liked," the probability estimate is 60%. This gives you a confidence level, not just a hard prediction.

## In Rust

```rust
use ndarray::array;
use machin_supervised::knn::KNN;
use machin_supervised::traits::Classifier;
use machin_supervised::metrics::{accuracy, precision, recall};

fn main() {
    // Features: [tempo_bpm, energy, danceability, acousticness, valence]
    // Labels: 0 = user dislikes, 1 = user likes
    let x_train = array![
        [120.0, 0.8, 0.9, 0.1, 0.7],  // liked (upbeat pop)
        [125.0, 0.9, 0.85, 0.05, 0.8], // liked (dance)
        [115.0, 0.7, 0.8, 0.15, 0.6],  // liked (pop)
        [60.0,  0.2, 0.1, 0.9, 0.3],   // disliked (slow acoustic)
        [70.0,  0.3, 0.2, 0.85, 0.2],  // disliked (ballad)
        [55.0,  0.1, 0.15, 0.95, 0.25],// disliked (ambient)
    ];
    let y_train = array![1, 1, 1, 0, 0, 0];

    // Build KNN classifier with k=3
    let mut knn = KNN::new(3);
    knn.fit(&x_train, &y_train);

    // Classify new songs
    let new_songs = array![
        [118.0, 0.75, 0.82, 0.12, 0.65],  // similar to upbeat cluster
        [65.0,  0.25, 0.18, 0.88, 0.28],   // similar to acoustic cluster
    ];

    let predictions = knn.predict(&new_songs);
    println!("Predictions: {}", predictions);
    // Expected: [1, 0] (liked, disliked)

    // Get probability estimates
    let proba = knn.predict_proba(&new_songs);
    println!("Like probabilities: {}", proba.column(1));

    // Evaluate on training set
    let train_pred = knn.predict(&x_train);
    println!("Training accuracy: {:.2}%", accuracy(&y_train, &train_pred) * 100.0);
}
```

## When To Use This

| Situation | KNN | Alternative | Why |
|---|---|---|---|
| Small dataset, few features | Yes | -- | Simple, no training, often accurate |
| Need a quick baseline | Yes | -- | Zero hyperparameter tuning (just pick k) |
| Complex non-linear decision boundaries | Yes | -- | KNN can model any shape of boundary |
| Large dataset (100K+ rows) | No | Logistic regression, decision tree | Brute-force distance to every point is O(n) per query |
| High-dimensional data (100+ features) | No | Random forest, SVM | Curse of dimensionality -- distances become meaningless |
| Need interpretable model | Partial | Decision tree | "These were your 5 nearest neighbors" is somewhat interpretable |
| Online/streaming data | No | Logistic regression | KNN must store all data; adding points is cheap but prediction slows |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `k` | (required) | Number of neighbors to consider. Must be a positive integer. |

### Choosing k

| k | Behavior |
|---|---|
| 1 | Nearest-neighbor. Very sensitive to noise. Decision boundary is jagged. |
| 3-7 | Good starting range. Balances smoothness with local sensitivity. |
| sqrt(n) | A common heuristic for larger datasets. |
| n | Predicts the majority class for every point (useless). |

**Rule of thumb:** Use an odd k for binary classification to avoid ties.

## Pitfalls

**Curse of dimensionality.** In high dimensions, all points become roughly equidistant. Euclidean distance stops being meaningful above ~20 features. Reduce dimensionality first (PCA, feature selection) or use a different algorithm.

**Feature scaling is critical.** KNN uses raw Euclidean distances. If tempo ranges from 50-200 and valence ranges from 0-1, tempo will completely dominate the distance calculation. Always normalize features to the same scale before using KNN.

**Slow prediction.** The MachinDeOuf implementation uses brute-force distance computation -- it calculates the distance to every training point for every query. This is O(n * p) per prediction. For large datasets, approximate methods (KD-trees, ball trees) are needed.

**Sensitive to irrelevant features.** If you include features that don't relate to the task (e.g., track number in the album), they add noise to the distance calculation and degrade performance. Only include features that matter.

**Memory intensive.** KNN stores the entire training dataset. Unlike parametric models that compress the data into a small set of weights, KNN's "model" is the data itself.

## Going Further

- **Distance metrics:** The MachinDeOuf `machin_math::distance` module provides `euclidean`, `manhattan`, `cosine_distance`, `chebyshev`, and `minkowski` functions. Different metrics suit different data types.
- **Weighted voting:** Instead of equal votes, weight each neighbor's vote by the inverse of its distance. Closer neighbors have more influence.
- **Approximate nearest neighbors:** For large-scale similarity search, the `machin-gpu` crate offers batch vector search on the GPU, which can dramatically speed up the neighbor-finding step.
- **Dimensionality reduction:** Use `machin-unsupervised` for PCA or other dimensionality reduction to combat the curse of dimensionality before applying KNN.
