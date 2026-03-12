//! Seeded random number generation and sampling utilities.

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};

/// Create a seeded RNG for reproducibility.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Random matrix from uniform distribution [low, high).
pub fn uniform_matrix(rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    Array2::random((rows, cols), Uniform::new(low, high).unwrap())
}

/// Random matrix from normal distribution N(mean, std^2).
pub fn normal_matrix(rows: usize, cols: usize, mean: f64, std: f64) -> Array2<f64> {
    Array2::random((rows, cols), Normal::new(mean, std).unwrap())
}

/// Random vector from uniform distribution.
pub fn uniform_vector(len: usize, low: f64, high: f64) -> Array1<f64> {
    Array1::random(len, Uniform::new(low, high).unwrap())
}

/// Random vector from normal distribution.
pub fn normal_vector(len: usize, mean: f64, std: f64) -> Array1<f64> {
    Array1::random(len, Normal::new(mean, std).unwrap())
}

/// Shuffle indices (Fisher-Yates).
pub fn shuffle_indices(n: usize, rng: &mut StdRng) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices
}

/// Sample k indices without replacement from [0, n).
pub fn sample_indices(n: usize, k: usize, rng: &mut StdRng) -> Vec<usize> {
    use rand::seq::index::sample;
    sample(rng, n, k.min(n)).into_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_matrix_shape() {
        let m = uniform_matrix(3, 4, 0.0, 1.0);
        assert_eq!(m.dim(), (3, 4));
    }

    #[test]
    fn test_normal_vector_len() {
        let v = normal_vector(10, 0.0, 1.0);
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_shuffle_indices() {
        let mut rng = seeded_rng(42);
        let indices = shuffle_indices(10, &mut rng);
        assert_eq!(indices.len(), 10);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_sample_indices() {
        let mut rng = seeded_rng(42);
        let indices = sample_indices(100, 5, &mut rng);
        assert_eq!(indices.len(), 5);
        // All unique
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }
}
