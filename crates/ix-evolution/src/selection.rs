//! Selection operators for evolutionary algorithms.

use rand::Rng;

use crate::traits::Individual;

/// Tournament selection: pick `k` random individuals, return the best.
pub fn tournament<I: Individual>(population: &[I], k: usize, rng: &mut impl Rng) -> I {
    assert!(!population.is_empty());
    let n = population.len();
    let mut best_idx = rng.random_range(0..n);
    for _ in 1..k {
        let idx = rng.random_range(0..n);
        if population[idx].fitness() < population[best_idx].fitness() {
            best_idx = idx;
        }
    }
    population[best_idx].clone()
}

/// Roulette wheel selection (fitness-proportional, for minimization).
/// Converts to maximization internally by using max_fitness - fitness.
pub fn roulette<I: Individual>(population: &[I], rng: &mut impl Rng) -> I {
    assert!(!population.is_empty());
    let max_fitness = population
        .iter()
        .map(|i| i.fitness())
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = population
        .iter()
        .map(|i| max_fitness - i.fitness() + 1e-10) // Shift so lower fitness = higher weight
        .collect();
    let total: f64 = weights.iter().sum();

    let mut r = rng.random::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return population[i].clone();
        }
    }
    population.last().unwrap().clone()
}

/// Rank-based selection: probability proportional to rank (best = highest rank).
pub fn rank<I: Individual>(population: &[I], rng: &mut impl Rng) -> I {
    assert!(!population.is_empty());
    let n = population.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        population[a]
            .fitness()
            .partial_cmp(&population[b].fitness())
            .unwrap()
    });

    // Rank weights: best gets n, worst gets 1
    let total: f64 = (n * (n + 1) / 2) as f64;
    let mut r = rng.random::<f64>() * total;
    for (rank, &idx) in indices.iter().enumerate() {
        let weight = (n - rank) as f64;
        r -= weight;
        if r <= 0.0 {
            return population[idx].clone();
        }
    }
    population[indices[0]].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::RealIndividual;
    use ndarray::array;
    use rand::SeedableRng;

    #[test]
    fn test_tournament_selects_best() {
        let pop = vec![
            RealIndividual::new(array![1.0]).with_fitness(10.0),
            RealIndividual::new(array![2.0]).with_fitness(5.0),
            RealIndividual::new(array![3.0]).with_fitness(1.0),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // With k=10 (many samples from 3 items), should almost always pick the best
        let selected = tournament(&pop, 10, &mut rng);
        assert!((selected.fitness() - 1.0).abs() < 1e-10);
    }
}
