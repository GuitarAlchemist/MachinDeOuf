//! Distance and similarity metrics.

use ndarray::Array1;

use crate::error::MathError;

fn check_same_len(a: &Array1<f64>, b: &Array1<f64>) -> Result<(), MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

/// Euclidean (L2) distance.
pub fn euclidean(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt())
}

/// Squared Euclidean distance (avoids sqrt for comparisons).
pub fn euclidean_squared(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>())
}

/// Manhattan (L1) distance.
pub fn manhattan(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>())
}

/// Minkowski distance of order p.
pub fn minkowski(a: &Array1<f64>, b: &Array1<f64>, p: f64) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    if p < 1.0 {
        return Err(MathError::InvalidParameter("p must be >= 1".into()));
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum();
    Ok(sum.powf(1.0 / p))
}

/// Cosine similarity (not distance). Returns value in [-1, 1].
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    let dot: f64 = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

/// Cosine distance = 1 - cosine_similarity.
pub fn cosine_distance(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    cosine_similarity(a, b).map(|s| 1.0 - s)
}

/// Chebyshev (L∞) distance.
pub fn chebyshev(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
    check_same_len(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert!((euclidean(&a, &b).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert!((manhattan(&a, &b).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).unwrap().abs() < 1e-10); // Orthogonal

        let c = array![1.0, 1.0];
        let d = array![1.0, 1.0];
        assert!((cosine_similarity(&c, &d).unwrap() - 1.0).abs() < 1e-10); // Same direction
    }

    #[test]
    fn test_minkowski_equals_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let mink = minkowski(&a, &b, 2.0).unwrap();
        let euc = euclidean(&a, &b).unwrap();
        assert!((mink - euc).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 0.0, 3.0];
        assert!((chebyshev(&a, &b).unwrap() - 3.0).abs() < 1e-10);
    }
}
