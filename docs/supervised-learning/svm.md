# Support Vector Machine (Linear SVM)

## The Problem

You're building a system that classifies images of handwritten digits as either "1" or "7" based on extracted features: the stroke angle, the number of endpoints, the height-to-width ratio, the presence of a horizontal bar, and the ink density. These two digits look similar -- a hastily written "1" can resemble a "7" without its crossbar, and vice versa.

You need a classifier that doesn't just find *any* boundary between the two classes, but finds the *best* one -- the boundary that sits as far as possible from both classes. If future handwriting samples are slightly different from the training data (and they will be), a boundary with a wide margin of safety will still classify correctly, while a boundary that barely threads between the two clusters will misclassify the slightest variation.

Support Vector Machines find this maximum-margin boundary. They ask: "What is the widest street I can draw between the two classes?" The data points closest to the street -- the ones that define its edges -- are called *support vectors*. Everything else is irrelevant to the boundary.

## The Intuition

Imagine you have red and blue marbles on a table and you need to place a ruler between them. There are many possible positions for the ruler that separate the two colors. But some positions are better than others: the best position maximizes the gap between the ruler and the nearest marble on each side.

Now imagine you glue the ruler in place and someone bumps the table, shifting all the marbles slightly. If the ruler was placed with a narrow gap, the marbles spill over. If the gap was wide, the classification survives the bump. SVM finds the ruler position with the widest gap.

The "support vectors" are the marbles closest to the ruler -- they're the only ones that matter for determining its position. You could remove every other marble and the boundary would be the same. This makes SVM elegant: the decision is driven by the hardest cases, not the easy ones.

What about marbles that are mixed together with no clean separation? The C parameter controls the trade-off: a high C says "classify everything correctly, even if the margin is narrow," while a low C says "allow some misclassification to get a wider margin." This soft margin lets SVM handle real-world data where classes overlap.

## How It Works

### Step 1: Define the decision boundary

The SVM finds a hyperplane defined by weights **w** and bias b:

$$
f(x) = \mathbf{w} \cdot \mathbf{x} + b
$$

Class prediction is based on the sign: if f(x) >= 0, predict class 1; otherwise predict class 0.

In plain English, this means: the model learns a weight for each feature, computes a weighted sum plus a bias, and checks which side of zero the result falls on. Positive side = class 1, negative side = class 0.

### Step 2: Map labels to {-1, +1}

Internally, class 0 becomes -1 and class 1 becomes +1. This makes the math cleaner.

In plain English, this means: instead of 0 and 1, the algorithm thinks in terms of "negative side" and "positive side" of the boundary.

### Step 3: Define the hinge loss

$$
L_i = \max(0, 1 - y_i \cdot f(x_i))
$$

In plain English, this means: if a point is on the correct side of the margin (the "street" is wide enough), the loss is zero -- everything is fine. If the point is inside the margin or on the wrong side, the loss grows linearly with how far it has strayed. This loss only "activates" for points that are problematic.

### Step 4: Minimize the SVM objective

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w} \cdot x_i + b))
$$

In plain English, this means: find weights that balance two goals. The first term (||w||^2) wants a wide margin -- small weights mean a wide street. The second term (the hinge loss sum) wants correct classification. C controls which goal wins: large C prioritizes correctness, small C prioritizes a wide margin.

### Step 5: Optimize via subgradient descent

For each training sample, compute whether it violates the margin. If it does, the gradient pushes the boundary to include it. If it doesn't, only the regularization gradient applies.

$$
\nabla_w = \begin{cases}
\mathbf{w} - C \cdot y_i \cdot x_i & \text{if } y_i(\mathbf{w} \cdot x_i + b) < 1 \\
\mathbf{w} & \text{otherwise}
\end{cases}
$$

In plain English, this means: the algorithm scans through the data repeatedly, adjusting the weights. For points that are safely classified with margin, it just shrinks the weights slightly (regularization). For points that are misclassified or too close to the boundary, it also shifts the boundary toward them. The learning rate decays over time for stability.

### Step 6: Predict probabilities (Platt scaling)

Since SVMs don't naturally produce probabilities, the implementation applies a sigmoid to the raw decision function output:

$$
P(\text{class} = 1 \mid x) = \frac{1}{1 + e^{-f(x)}}
$$

In plain English, this means: the further a point is from the decision boundary, the more confident the prediction. Points right on the boundary get ~50% probability.

## In Rust

```rust
use ndarray::array;
use machin_supervised::svm::LinearSVM;
use machin_supervised::traits::Classifier;
use machin_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [stroke_angle, endpoints, height_width_ratio, has_crossbar, ink_density]
    // Labels: 0 = digit "1", 1 = digit "7"
    let x = array![
        [85.0,  2.0, 3.5, 0.0, 0.15],   // "1"
        [88.0,  2.0, 3.8, 0.0, 0.12],   // "1"
        [82.0,  2.0, 3.2, 0.0, 0.18],   // "1"
        [90.0,  2.0, 4.0, 0.0, 0.10],   // "1"
        [45.0,  3.0, 1.8, 1.0, 0.35],   // "7"
        [50.0,  3.0, 2.0, 1.0, 0.30],   // "7"
        [40.0,  3.0, 1.5, 1.0, 0.40],   // "7"
        [48.0,  3.0, 1.9, 1.0, 0.33],   // "7"
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    // Build SVM with C=1.0 regularization
    let mut svm = LinearSVM::new(1.0)
        .with_learning_rate(0.01)
        .with_max_iterations(500);
    svm.fit(&x, &y);

    // Predict
    let predictions = svm.predict(&x);
    println!("Predictions: {}", predictions);
    println!("Accuracy: {:.2}%", accuracy(&y, &predictions) * 100.0);

    // Probability estimates (via Platt scaling)
    let proba = svm.predict_proba(&x);
    for i in 0..proba.nrows() {
        println!(
            "Sample {}: P('1')={:.3}, P('7')={:.3}",
            i, proba[[i, 0]], proba[[i, 1]]
        );
    }

    // Test on a new sample
    let new_digit = array![[86.0, 2.0, 3.6, 0.0, 0.14]];
    let pred = svm.predict(&new_digit);
    println!("New digit classified as: {}", if pred[0] == 0 { "'1'" } else { "'7'" });

    // Evaluation
    println!("Precision ('7'): {:.4}", precision(&y, &predictions, 1));
    println!("Recall ('7'):    {:.4}", recall(&y, &predictions, 1));
    println!("F1 ('7'):        {:.4}", f1_score(&y, &predictions, 1));
}
```

## When To Use This

| Situation | Linear SVM | Alternative | Why |
|---|---|---|---|
| Binary classification, clean margins | Yes | -- | Maximum margin gives excellent generalization |
| High-dimensional data (many features, few samples) | Yes | -- | SVM excels when features >> samples |
| Need robust generalization | Yes | -- | Margin maximization prevents overfitting |
| Non-linear decision boundary | No | Decision tree, neural net, kernel SVM | Linear SVM can only draw straight hyperplanes |
| Multi-class classification | Requires workaround | Decision tree, random forest | Build one SVM per class pair (one-vs-one or one-vs-rest) |
| Need calibrated probabilities | No | Logistic regression | Platt scaling is approximate; logistic regression is natively calibrated |
| Very large dataset | Depends | Logistic regression | Subgradient descent is iterative; logistic regression may converge faster |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `c` | (required) | Regularization parameter. Higher = less regularization, tighter fit to training data. |
| `learning_rate` | `0.001` | Initial step size for subgradient descent. Decays automatically over iterations. |
| `max_iterations` | `1000` | Number of passes through the data. |

Configure via the builder pattern:
```rust
LinearSVM::new(1.0)               // C = 1.0
    .with_learning_rate(0.01)
    .with_max_iterations(500)
```

### The C parameter

| C | Behavior |
|---|---|
| 0.001 | Very wide margin, allows many misclassifications (high bias, low variance) |
| 0.01 - 0.1 | Moderately regularized, good starting range |
| 1.0 | Balanced (the most common default) |
| 10 - 100 | Narrow margin, tries hard to classify every training point correctly (low bias, high variance) |
| 1000+ | Effectively hard margin, will overfit if data is noisy |

## Pitfalls

**Feature scaling is essential.** SVM computes dot products and norms. If one feature ranges from 0 to 1000 and another from 0 to 1, the first feature dominates the margin calculation. Always standardize features (zero mean, unit variance) before training.

**Binary only.** The MachinDeOuf `LinearSVM` supports two classes (0 and 1). For multi-class problems, train multiple SVMs in a one-vs-rest configuration and pick the class with the highest decision function value.

**Linear only.** This implementation finds a linear hyperplane. If the classes are not linearly separable (e.g., one class surrounds the other in a ring), the SVM will perform poorly. Kernel SVMs (RBF, polynomial) can handle non-linear boundaries but are not implemented in MachinDeOuf.

**Convergence.** Subgradient descent is not as smooth as standard gradient descent. The loss may oscillate. Increasing `max_iterations` and using a smaller `learning_rate` helps, but convergence is slow. Monitor the training loss if possible.

**Probability estimates are approximate.** The `predict_proba` method applies a sigmoid to the raw decision function (Platt scaling without calibration fitting). The probabilities are directionally correct but not well-calibrated. Do not use them for tasks that require precise probability estimates.

**Sensitive to outliers when C is high.** A single misplaced data point near the margin can shift the boundary substantially when C is large. Use a moderate C value or clean outliers from the training data.

## Going Further

- **Kernel trick:** Map features into a higher-dimensional space where non-linear patterns become linearly separable. Common kernels include RBF (Gaussian) and polynomial. This is the logical next step beyond `LinearSVM`.
- **One-vs-rest multi-class:** Train one `LinearSVM` per class (class c vs. all others) and predict the class whose SVM has the highest decision function output.
- **Feature importance:** The weight vector **w** directly tells you which features matter most. Large absolute weights correspond to important features.
- **Comparison with logistic regression:** Both find a linear boundary. Logistic regression minimizes log loss; SVM minimizes hinge loss. SVM focuses on the boundary (support vectors), while logistic regression uses all points. See [logistic-regression.md](./logistic-regression.md).
- **Evaluation:** See [evaluation-metrics.md](./evaluation-metrics.md) for choosing between accuracy, precision, recall, and F1 for binary classification tasks.
