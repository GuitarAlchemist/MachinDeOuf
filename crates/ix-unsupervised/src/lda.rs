//! Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.
//!
//! LDA finds a linear projection that maximizes between-class variance while
//! minimizing within-class variance. Unlike PCA (which is unsupervised and
//! maximizes total variance), LDA uses class labels to find directions that
//! best separate the classes.
//!
//! For a c-class problem, LDA reduces the feature space to at most `c - 1`
//! dimensions. This is useful both as a preprocessing step for classifiers
//! and as a visualization technique for labeled data.
//!
//! # Algorithm
//!
//! 1. Compute the class means and overall mean.
//! 2. Build the within-class scatter matrix `S_W = sum_c sum_i (x_i - mu_c)(x_i - mu_c)^T`.
//! 3. Build the between-class scatter matrix `S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T`.
//! 4. Solve the generalized eigenvalue problem `S_B v = lambda S_W v`, which
//!    we convert to a standard eigenvalue problem `S_W^{-1} S_B v = lambda v`.
//! 5. Take the eigenvectors with the largest eigenvalues as the projection axes.
//!
//! # Example
//!
//! ```
//! use ix_unsupervised::lda::LinearDiscriminantAnalysis;
//! use ndarray::array;
//!
//! // Two classes, clearly separated along x-axis
//! let x = array![
//!     [0.0, 0.0],
//!     [0.1, 0.1],
//!     [0.0, 0.2],
//!     [5.0, 0.0],
//!     [5.1, 0.1],
//!     [5.0, 0.2],
//! ];
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let mut lda = LinearDiscriminantAnalysis::new(1);
//! lda.fit(&x, &y).unwrap();
//! let projected = lda.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ndarray::{Array1, Array2};
use std::collections::BTreeMap;

use ix_math::error::MathError;
use ix_math::linalg::inverse;

/// Linear Discriminant Analysis model.
///
/// After `fit`, the model stores up to `n_components` projection vectors
/// (columns of the transformation matrix) along with the class means.
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Number of discriminant components to keep. Upper bound is `min(n_classes - 1, n_features)`.
    pub n_components: usize,
    /// Transformation matrix with shape `(n_features, n_components)`, filled by `fit`.
    pub components: Option<Array2<f64>>,
    /// Eigenvalues associated with each retained component.
    pub explained: Option<Array1<f64>>,
    /// Overall mean used for centering inputs.
    pub mean: Option<Array1<f64>>,
}

impl LinearDiscriminantAnalysis {
    /// Create a new LDA model requesting `n_components` discriminant axes.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            explained: None,
            mean: None,
        }
    }

    /// Fit the LDA model on features `x` (n_samples x n_features) and labels `y`.
    ///
    /// Labels are arbitrary `usize` values; internally they are grouped into
    /// classes via a sorted map, and the number of classes is inferred.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) -> Result<(), MathError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(MathError::EmptyInput);
        }
        if y.len() != n_samples {
            return Err(MathError::DimensionMismatch {
                expected: n_samples,
                got: y.len(),
            });
        }

        // Group row indices by class label.
        let mut by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &lbl) in y.iter().enumerate() {
            by_class.entry(lbl).or_default().push(i);
        }
        let n_classes = by_class.len();
        if n_classes < 2 {
            return Err(MathError::InvalidParameter(
                "LDA needs at least 2 classes".into(),
            ));
        }

        let max_components = (n_classes - 1).min(n_features);
        if self.n_components > max_components {
            return Err(MathError::InvalidParameter(format!(
                "n_components={} exceeds max {} for {} classes and {} features",
                self.n_components, max_components, n_classes, n_features
            )));
        }

        // Overall mean and per-class means.
        let overall_mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let mut class_means: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
        for (&lbl, indices) in &by_class {
            let mut mu = Array1::<f64>::zeros(n_features);
            for &i in indices {
                mu = mu + x.row(i).to_owned();
            }
            mu /= indices.len() as f64;
            class_means.insert(lbl, mu);
        }

        // Within-class scatter S_W = sum_c sum_i (x_i - mu_c)(x_i - mu_c)^T
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        for (&lbl, indices) in &by_class {
            let mu_c = class_means.get(&lbl).unwrap();
            for &i in indices {
                let diff = &x.row(i).to_owned() - mu_c;
                for a in 0..n_features {
                    for b in 0..n_features {
                        sw[[a, b]] += diff[a] * diff[b];
                    }
                }
            }
        }

        // Between-class scatter S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T
        let mut sb = Array2::<f64>::zeros((n_features, n_features));
        for (&lbl, indices) in &by_class {
            let mu_c = class_means.get(&lbl).unwrap();
            let diff = mu_c - &overall_mean;
            let n_c = indices.len() as f64;
            for a in 0..n_features {
                for b in 0..n_features {
                    sb[[a, b]] += n_c * diff[a] * diff[b];
                }
            }
        }

        // Regularize S_W slightly to keep it invertible even when classes
        // share degenerate directions (very small numerical safeguard).
        for i in 0..n_features {
            sw[[i, i]] += 1e-10;
        }

        // Solve standard eigenvalue problem for M = S_W^{-1} S_B.
        let sw_inv = inverse(&sw)?;
        let m = sw_inv.dot(&sb);

        // Extract top eigenvectors via power iteration + deflation.
        let mut components = Array2::<f64>::zeros((n_features, self.n_components));
        let mut explained = Array1::<f64>::zeros(self.n_components);
        let mut current = m.clone();
        for k in 0..self.n_components {
            let (lambda, v) = power_iteration(&current, 500, 1e-10);
            for i in 0..n_features {
                components[[i, k]] = v[i];
            }
            explained[k] = lambda.max(0.0);
            // Deflate: M' = M - lambda v v^T (works for diagonalizable M).
            for a in 0..n_features {
                for b in 0..n_features {
                    current[[a, b]] -= lambda * v[a] * v[b];
                }
            }
        }

        self.components = Some(components);
        self.explained = Some(explained);
        self.mean = Some(overall_mean);
        Ok(())
    }

    /// Project new samples into the LDA subspace. Requires `fit` first.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| MathError::InvalidParameter("model not fitted".into()))?;
        let mean = self.mean.as_ref().unwrap();

        if x.ncols() != mean.len() {
            return Err(MathError::DimensionMismatch {
                expected: mean.len(),
                got: x.ncols(),
            });
        }

        let centered = x - &mean.view().insert_axis(ndarray::Axis(0));
        Ok(centered.dot(components))
    }

    /// Convenience: fit then transform in one call.
    pub fn fit_transform(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<Array2<f64>, MathError> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

/// Power iteration on a (possibly non-symmetric) matrix.
/// Returns (dominant eigenvalue, normalized eigenvector).
fn power_iteration(m: &Array2<f64>, max_iter: usize, tol: f64) -> (f64, Array1<f64>) {
    let n = m.nrows();
    let mut v = Array1::<f64>::from_elem(n, 1.0 / (n as f64).sqrt());
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let v_new = m.dot(&v);
        let new_lambda = v.dot(&v_new);
        let norm = v_new.dot(&v_new).sqrt();
        if norm < 1e-15 {
            return (0.0, v);
        }
        v = v_new / norm;
        if (new_lambda - lambda).abs() < tol {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;
    }
    (lambda, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_two_classes_1d_projection() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.2],
            [-0.1, 0.1],
            [5.0, 0.0],
            [5.1, 0.2],
            [4.9, 0.1],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        lda.fit(&x, &y).unwrap();
        let projected = lda.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
        // Class 0 and class 1 should be well separated in the projection
        let c0_mean: f64 = projected.slice(ndarray::s![0..3, 0]).mean().unwrap();
        let c1_mean: f64 = projected.slice(ndarray::s![3..6, 0]).mean().unwrap();
        assert!(
            (c0_mean - c1_mean).abs() > 1.0,
            "classes should be separated: {} vs {}",
            c0_mean,
            c1_mean
        );
    }

    #[test]
    fn test_lda_three_classes_2d_projection() {
        // Three clusters in 3D, LDA should find a 2D separating subspace
        let x = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [5.0, 0.0, 0.0],
            [5.1, 0.0, 0.0],
            [5.0, 0.1, 0.0],
            [0.0, 5.0, 0.0],
            [0.1, 5.0, 0.0],
            [0.0, 5.1, 0.0],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut lda = LinearDiscriminantAnalysis::new(2);
        lda.fit(&x, &y).unwrap();
        let projected = lda.fit_transform(&x, &y).unwrap();
        assert_eq!(projected.ncols(), 2);
    }

    #[test]
    fn test_lda_rejects_single_class() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0usize, 0];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_rejects_too_many_components() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![0usize, 0, 1, 1];
        // 2 classes allow max 1 component
        let mut lda = LinearDiscriminantAnalysis::new(2);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_transform_dimension_check() {
        let x = array![[0.0, 0.0], [5.0, 0.0]];
        let y = array![0usize, 1];
        let mut lda = LinearDiscriminantAnalysis::new(1);
        lda.fit(&x, &y).unwrap();
        let bad = array![[0.0, 0.0, 0.0]];
        assert!(lda.transform(&bad).is_err());
    }
}
