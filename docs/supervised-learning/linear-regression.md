# Linear Regression

## The Problem

You work at a real estate company and need to price houses before they hit the market. Every house has measurable attributes -- square footage, number of bedrooms, distance to the nearest school, lot size -- and a sale price that past buyers actually paid. Your job is to look at these numbers and predict what a new house will sell for.

A human appraiser can do this, but they're slow and subjective. They might anchor on the last house they saw or let the nice kitchen sway them. You want a method that considers every feature simultaneously, weighs them objectively based on historical data, and produces a single dollar estimate in milliseconds.

Linear regression is that method. It finds the straight-line (or flat-plane, in multiple dimensions) relationship between your input features and the price, then uses that relationship to predict prices for houses it has never seen.

## The Intuition

Imagine you plot house prices on the y-axis and square footage on the x-axis. The dots form a rough upward cloud. Linear regression draws the single straight line through that cloud that minimizes the total distance between each dot and the line. "Distance" here means vertical distance -- how far off your prediction would have been for each historical house.

When you have more than one feature, the "line" becomes a flat sheet (a hyperplane) stretched through a higher-dimensional space. You can't visualize it anymore, but the math is the same: find the flat surface that sits as close as possible to every data point.

Think of it like balancing a rigid ruler on a scatter of nails poking up from a board. The ruler settles into the position where the total spring tension (the squared errors) is minimized. That resting position *is* the regression line.

## How It Works

### Step 1: Augment the feature matrix

We add a column of ones to the feature matrix **X** so that the bias term (the y-intercept) falls out naturally as another weight.

```
X_aug = [X | 1]
```

In plain English, this means: we tack a "dummy feature" of value 1 onto every data point so the model can learn a baseline price even when all real features are zero.

### Step 2: Solve the normal equation

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

In plain English, this means: we find the weight vector **w** that minimizes the sum of squared prediction errors in one shot, with no iterating. The formula looks at how features correlate with each other (X^T X) and how they correlate with the target (X^T y), then solves for the perfect balance.

### Step 3: Separate weights and bias

The last element of **w** becomes the bias (intercept). Everything before it is the per-feature weight vector.

In plain English, this means: we now know "each extra square foot adds $215 to the price" (a weight) and "the baseline price of an empty lot is $45,000" (the bias).

### Step 4: Predict

$$
\hat{y} = \mathbf{X} \mathbf{w} + b
$$

In plain English, this means: for a new house, multiply each feature by its weight, add them all up, add the bias, and that's your predicted price.

## In Rust

```rust
use ndarray::array;
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::traits::Regressor;
use ix_supervised::metrics::{mse, rmse, r_squared};

fn main() {
    // Features: [sqft, bedrooms, distance_to_school_miles]
    let x = array![
        [1400.0, 3.0, 0.5],
        [1600.0, 3.0, 1.2],
        [1700.0, 4.0, 0.8],
        [1875.0, 4.0, 0.3],
        [1100.0, 2.0, 2.0],
        [2200.0, 5.0, 0.6],
    ];

    // Sale prices in thousands of dollars
    let y = array![245.0, 312.0, 279.0, 308.0, 199.0, 395.0];

    // Train the model
    let mut model = LinearRegression::new();
    model.fit(&x, &y);

    // Inspect learned parameters
    let weights = model.weights.as_ref().unwrap();
    println!("Weights: {}", weights);   // per-feature coefficients
    println!("Bias: {:.2}", model.bias); // intercept

    // Predict on new houses
    let new_houses = array![
        [1500.0, 3.0, 1.0],
        [2000.0, 4.0, 0.4],
    ];
    let predictions = model.predict(&new_houses);
    println!("Predicted prices: {}", predictions);

    // Evaluate on training data
    let y_pred = model.predict(&x);
    println!("MSE:  {:.4}", mse(&y, &y_pred));
    println!("RMSE: {:.4}", rmse(&y, &y_pred));
    println!("R^2:  {:.4}", r_squared(&y, &y_pred));
}
```

## When To Use This

| Situation | Linear Regression | Alternative | Why |
|---|---|---|---|
| Continuous target, linear relationship | Yes | -- | This is its sweet spot |
| Need interpretable coefficients | Yes | -- | Each weight has a clear meaning |
| Non-linear patterns in data | No | Decision tree, neural net | Linear regression will underfit curves |
| Lots of features, many irrelevant | Careful | Lasso/Ridge (not yet in ix) | OLS can overfit with many collinear features |
| Classification task | No | Logistic regression, SVM | Linear regression predicts continuous values, not classes |
| Very large dataset (millions of rows) | Slow | SGD-based regression | Normal equation inverts an NxN matrix |

## Key Parameters

| Parameter | Type | Description |
|---|---|---|
| `weights` | `Option<Array1<f64>>` | Learned feature coefficients. `None` before `fit()` is called. |
| `bias` | `f64` | Learned intercept (y-value when all features are zero). Starts at `0.0`. |

`LinearRegression::new()` takes no configuration -- the normal equation has no hyperparameters to tune. This is one of its strengths: there is nothing to get wrong.

## Pitfalls

**Singular matrix.** If two features are perfectly correlated (e.g., square footage in both square feet and square meters), X^T X is singular and cannot be inverted. The `fit()` call will panic. Remove duplicate or perfectly collinear features before training.

**Feature scaling.** While linear regression produces correct results regardless of feature scale, the *weights* will be on wildly different scales if features are (e.g., square footage in thousands vs. bedrooms in single digits). This makes interpretation harder but does not affect predictions.

**Outliers dominate.** Because the model minimizes *squared* error, a single mansion priced at $10M in a neighborhood of $300K homes will yank the regression line toward itself. Consider removing extreme outliers or using robust regression techniques.

**Extrapolation is dangerous.** The model knows nothing outside the range of training data. Predicting the price of a 50,000 sqft warehouse when your largest training example was 3,000 sqft will produce nonsense.

**Overfitting with many features.** If you have nearly as many features as data points, the model can fit noise. The normal equation will still produce a solution, but it won't generalize. Collect more data or reduce features.

## Going Further

- **Polynomial features:** Create columns like `sqft^2` or `sqft * bedrooms` and feed them in as extra features. Linear regression on polynomial features can model curves.
- **Regularization:** Ridge and Lasso regression add a penalty term to prevent large weights. ix's `ix-optimize` crate provides SGD and Adam optimizers that could be used to implement these.
- **Gradient descent alternative:** For datasets too large for the normal equation, use `ix_optimize::sgd` to iteratively minimize MSE instead of inverting a matrix.
- **Evaluation:** See [evaluation-metrics.md](./evaluation-metrics.md) for a deep dive into MSE, RMSE, and R-squared.
